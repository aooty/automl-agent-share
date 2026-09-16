"""The only module that talks to the Anthropic API.

Responsibilities kept here so nodes stay simple: prompt loading, structured-output
enforcement, one corrective retry, timeouts, transport retries, and archiving every
prompt/response pair under ``artifacts/<thread_id>/llm/``.

Credentials are read from the environment only (``ANTHROPIC_API_KEY``, or
``AWS_REGION`` when ``AUTOML_USE_BEDROCK`` is set) and are never written to a log,
an artifact, or an exception message.

Being the only egress point, this is also where the raw-data guard sits: see
:func:`render_prompt` and :mod:`automl_agent.privacy`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

from ..config import (
    API_KEY_ENV,
    AWS_REGION_ENV,
    DEFAULT_LLM_MAX_RETRIES,
    PROMPTS_DIR,
    RunConfig,
    bedrock_signing_available,
    use_bedrock,
)
from ..privacy import assert_clean


class LLMUnavailable(RuntimeError):
    """Raised when a call cannot be completed. Callers fall back deterministically."""


class TextCompletion(NamedTuple):
    """A free-form completion, plus whether the model was cut off before finishing.

    ``truncated`` exists because the fact was being thrown away. :meth:`LLMClient._archive`
    has always written the API's ``stop_reason`` into the exchange JSON and no code path read
    it, so ``test-1`` spent exactly its output allowance, wrote a ``report.md`` that ends in
    the middle of a word, and reported success. A cut-off response is not an error the
    transport can raise — it is a well-formed reply that happens to be incomplete — so the
    only place it can be noticed is here, at the return.

    Called ``truncated`` and not ``stop_reason`` on purpose:
    :func:`automl_agent.nodes.report.stop_reason` already means *why the loop ended*, and one
    name for both would make "why the run stopped" and "why the sentence stopped" the same
    field in a reader's head.
    """

    text: str
    truncated: bool


# Name of the single tool used when structured output has to be enforced via tool use.
STRUCTURED_TOOL_NAME = "submit_result"

# A model id beginning with this goes to a locally served model instead of the API. A prefix on
# the id and not a separate flag, because "which model" and "which transport" are one decision
# here: there is no configuration in which a caller wants ``gemma3`` sent to Anthropic.
OLLAMA_PREFIX = "ollama:"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def needs_anthropic(config: RunConfig) -> bool:
    """Whether any node in this run will reach the Anthropic API.

    The two halves resolve separately — the proposer follows ``proposer_model`` when it is set
    and ``llm_model`` otherwise — so a run can be fully local, fully remote, or split. Only the
    fully local case needs no credentials, and ``main.check_credentials`` refuses before the
    graph starts on the strength of this. Without it ``--model ollama:...`` was refused by a
    guard written for a route it does not use.
    """
    proposer = config.proposer_model or config.llm_model
    return not (
        config.llm_model.startswith(OLLAMA_PREFIX) and proposer.startswith(OLLAMA_PREFIX)
    )


def ollama_host() -> str:
    """Where the local server is. ``OLLAMA_HOST`` is Ollama's own variable, so a machine that
    already runs it elsewhere needs nothing set here."""
    host = (os.environ.get(OLLAMA_HOST_ENV) or "").strip() or DEFAULT_OLLAMA_HOST
    return host if "://" in host else f"http://{host}"


class OllamaTextBlock(NamedTuple):
    """One text block, shaped like the SDK's so :func:`extract_text` needs no branch."""

    text: str
    type: str = "text"


class OllamaUsage(NamedTuple):
    """Ollama's token counts under the names the archive already writes.

    No cache fields on purpose: ``_archive`` reads them with ``getattr(..., None)``, so their
    absence records ``null`` — which is the honest answer for a route that has no prompt cache
    to hit rather than a route that missed one.
    """

    input_tokens: int | None
    output_tokens: int | None


class OllamaCompletion(NamedTuple):
    """Enough of a ``Message`` for :func:`extract_text`, :func:`extract_structured` and the
    archive. Deliberately not more — a fuller imitation would invite code that treats the two
    routes as interchangeable in ways they are not."""

    content: list[OllamaTextBlock]
    usage: OllamaUsage
    stop_reason: str

# How structured output is enforced, cached per transport for the life of the process.
# Every node builds its own LLMClient, so holding this per instance would make each
# node re-pay a rejected `output_config` round-trip before downgrading again.
_STRUCTURED_MODE: dict[str, str] = {}


def _route_key() -> str:
    return "bedrock" if use_bedrock() else "anthropic"


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

# Where a prompt's cacheable prefix ends. Written in the ``.md`` file rather than decided here
# because *which* blocks are stable is a property of the prompt, and this repository keeps
# prompts out of the code. See ``docs/rationale.md``.
CACHE_MARKER = "<!-- cache -->"
# The API allows four per request; a prompt asking for more is a mistake worth naming rather
# than a 400 from inside the transport.
MAX_CACHE_BREAKPOINTS = 4
# 1h and not the 5-minute default: the gap between two calls to the same node is a whole
# iteration — a fit that may take minutes — and a 5-minute entry is measured from the *start*
# of the request that wrote it, so it is usually cold by the next call. The 1h write costs 2x
# instead of 1.25x and needs three reads to pay for itself; a run does nine.
CACHE_TTL = "1h"


def prompt_content(prompt: str) -> str | list[dict[str, Any]]:
    """One user message's content, split into cache blocks at every :data:`CACHE_MARKER`.

    Returns the string unchanged when the prompt carries no marker, so a prompt that has not
    been ordered for caching (``report.md`` — one call per run, nothing to reuse) keeps the
    shape it always had.

    The marker is left in the text that is sent. It costs a handful of tokens and it keeps
    :meth:`LLMClient._archive` honest: what the archive shows is still byte-for-byte what the
    model was sent, and a reader of the archive can see where the breakpoints were.
    """
    if CACHE_MARKER not in prompt:
        return prompt
    head, *rest = prompt.split(CACHE_MARKER)
    # Each marker ends the block before it, so the marker text belongs to that block.
    chunks = [head + CACHE_MARKER, *rest[:-1]]
    chunks = [chunk if index == 0 else chunk + CACHE_MARKER for index, chunk in enumerate(chunks)]
    if len(chunks) > MAX_CACHE_BREAKPOINTS:
        raise ValueError(
            f"{len(chunks)} cache breakpoints requested, the API allows {MAX_CACHE_BREAKPOINTS}"
        )
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": chunk, "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}
        for chunk in chunks
    ]
    tail = rest[-1]
    if tail:
        blocks.append({"type": "text", "text": tail})
    return blocks


def render_prompt(name: str, variables: dict[str, Any], prompts_dir: Path = PROMPTS_DIR) -> str:
    """Load ``<name>.md`` and substitute ``{{var}}`` placeholders.

    Deliberately not ``str.format``: prompts contain JSON braces that would break it.
    Non-string values are serialised as indented JSON so the model sees readable data.

    Every prompt in the system is built here — including the ones only archived under
    ``--dry-run``/``--no-llm`` — which makes this the single chokepoint for
    :func:`automl_agent.privacy.assert_clean`. A registered dataset path raises
    :class:`~automl_agent.privacy.RawDataLeak`; anything path-shaped is redacted. What
    the archive shows is therefore byte-for-byte what the model was sent.
    """
    template = (prompts_dir / f"{name}.md").read_text(encoding="utf-8")

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        value = variables[key]
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    rendered = _PLACEHOLDER.sub(substitute, template)
    missing = sorted(set(_PLACEHOLDER.findall(rendered)))
    if missing:
        raise KeyError(f"prompt {name!r} has unfilled placeholders: {missing}")
    return assert_clean(rendered, label=f"prompt {name!r}")


def archive_prompt_only(config: RunConfig, label: str, prompt: str) -> Path | None:
    """Archive a rendered prompt without calling the API (the ``--dry-run`` path).

    Rendering for real under ``--dry-run`` is deliberate: it proves the templates
    have no unfilled placeholders, and it makes the reasoning trail — e.g. that
    iteration 2's planning prompt really does carry iteration 1's ``failure_type``
    and ``direction`` — auditable without spending a token.
    """
    directory = config.llm_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_next_index(directory):03d}_{label}.dryrun.md"
        path.write_text(prompt, encoding="utf-8")
    except OSError:
        return None
    return path


def archive_label(prompt_name: str, *, iteration: int | None = None, attempt: int = 1) -> str:
    """Name one archived exchange so it maps back to the iteration that made it.

    The real-route archive used to be ``planning_attempt1`` for every iteration —
    ``attempt`` is the corrective-retry counter, not the loop counter — so files 001,
    004 and 007 of a three-iteration run all carried the same name and could only be
    tied to an iteration by counting. The ``--dry-run`` writer already said
    ``planning_iter2``; this makes both routes say it.
    """
    parts = [prompt_name]
    if iteration:
        parts.append(f"iter{iteration}")
    if attempt > 1:
        # Only a real retry earns a suffix, so the common case stays readable.
        parts.append(f"attempt{attempt}")
    return "_".join(parts)


def _next_index(directory: Path) -> int:
    """Next sequence number, counting *every* archived file.

    Both writers share the counter: a ``--force`` re-run that switches between
    ``--dry-run`` and a real route writes into the same directory, and counting only
    one suffix would restart the numbering and overwrite the other route's trail.
    """
    return len([path for path in directory.glob("*") if path.is_file()]) + 1


class LLMClient:
    """Thin, logged wrapper around ``messages.create``."""

    def __init__(self, config: RunConfig, *, proposer: bool = False) -> None:
        """``proposer=True`` for the two nodes that propose rather than judge.

        A keyword and not a model string, so the two callers say *which node they are* and this
        module owns the mapping. The alternative — every node reading ``config`` and picking a
        field — puts the same three-line decision in four places, and a fifth node added later
        gets it wrong silently.
        """
        self.config = config
        self._client: Any | None = None
        self._model = (config.proposer_model if proposer else "") or config.llm_model

    @property
    def _structured_mode(self) -> str:
        """Starts on the stricter ``output_config`` json_schema, downgrades to forced
        tool use once a transport rejects it."""
        return _STRUCTURED_MODE.get(_route_key(), "output_config")

    @_structured_mode.setter
    def _structured_mode(self, mode: str) -> None:
        _STRUCTURED_MODE[_route_key()] = mode

    # -- transport --------------------------------------------------------- #

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable(f"the anthropic SDK is not installed: {exc}") from exc

        if use_bedrock():
            region = os.environ.get(AWS_REGION_ENV)
            if not region:
                raise LLMUnavailable(f"{AWS_REGION_ENV} is not set, required for the Bedrock route")
            if not bedrock_signing_available():
                # The SDK imports botocore only when signing, so without this the
                # failure would escape as ModuleNotFoundError and abort the graph
                # instead of falling back deterministically.
                raise LLMUnavailable(
                    'botocore is not installed; run `pip install "anthropic[bedrock]"` '
                    "for the Bedrock route"
                )
            self._client = anthropic.AnthropicBedrockMantle(
                aws_region=region,
                timeout=self.config.llm_timeout_sec,
                max_retries=DEFAULT_LLM_MAX_RETRIES,
            )
            # Bedrock names the vendor in the model id, so the default ``claude-opus-5`` has to
            # become ``anthropic.claude-opus-5`` — which is the form this route wants, and the
            # reason the prefixing exists at all.
            #
            # The test is for the vendor *anywhere* in the id rather than at the front, and that
            # is about an id the caller passes explicitly. Bedrock also has cross-region
            # inference profiles, which put a scope in front of the vendor
            # (``global.anthropic.claude-opus-5``, ``us.anthropic.claude-...``). Anchored at the
            # front, ``--model global.anthropic.claude-opus-5`` picked up a second prefix and
            # became ``anthropic.global.anthropic.claude-opus-5``; the call 404s, and because
            # ``planning`` treats an unavailable LLM as a fallback rather than an error, the run
            # would go on to produce a rule-based result with one printed line about the outage.
            # A scoped id may not be right for every endpoint, but mangling it is right for none.
            if "anthropic." not in self._model:
                self._model = f"anthropic.{self._model}"
        else:
            if not os.environ.get(API_KEY_ENV):
                raise LLMUnavailable(
                    f"{API_KEY_ENV} is not set. Export it, set {AWS_REGION_ENV} with "
                    "AUTOML_USE_BEDROCK=1, or run with --dry-run."
                )
            self._client = anthropic.Anthropic(
                timeout=self.config.llm_timeout_sec,
                max_retries=DEFAULT_LLM_MAX_RETRIES,
            )
        return self._client

    def _request_kwargs(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
            "thinking": {"type": "adaptive"},
        }
        if system:
            kwargs["system"] = system
        if schema is None:
            return kwargs
        if self._structured_mode == "tool":
            kwargs["tools"] = [
                {
                    "name": STRUCTURED_TOOL_NAME,
                    "description": "Submit the result. Its input is the answer.",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        else:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return kwargs

    def _create(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> Any:
        if self._model.startswith(OLLAMA_PREFIX):
            return self._create_ollama(
                messages, system=system, max_tokens=max_tokens, schema=schema
            )

        import anthropic

        client = self._ensure_client()
        try:
            return client.messages.create(
                **self._request_kwargs(messages, system, max_tokens, schema)
            )
        except anthropic.BadRequestError as exc:
            # Not every transport accepts `output_config`: the Bedrock route rejects it
            # with "output_config.format: Extra inputs are not permitted". Downgrade to
            # forced tool use — which the spec also sanctions and every route supports —
            # and remember the choice for the rest of the run.
            if schema is None or self._structured_mode != "output_config":
                raise LLMUnavailable(f"API rejected the request: {describe_api_error(exc)}") from exc
            self._structured_mode = "tool"
            try:
                return client.messages.create(
                    **self._request_kwargs(messages, system, max_tokens, schema)
                )
            except anthropic.APIStatusError as retry_exc:
                raise LLMUnavailable(describe_api_error(retry_exc)) from retry_exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable(f"rate limited after retries: {describe_api_error(exc)}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(describe_api_error(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"could not reach the API: {exc}") from exc

    def _create_ollama(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None,
        max_tokens: int,
        schema: dict[str, Any] | None,
    ) -> Any:
        """The same call against a locally served model.

        ``urllib`` and not a client library: this is one POST with a JSON body, and the reason
        the Anthropic SDK is worth a dependency — retries, signing, streaming, typed errors —
        does not apply to a request to localhost.

        Structured output is Ollama's ``format`` field, which takes the JSON schema directly.
        That is the whole reason this route is viable: the enforcement the nodes rely on
        (``validate_plan``, the registry, the clamps) sits *behind* the schema, and a route with
        no schema at all would push every malformed answer onto those guards and record it as a
        fallback. It is a different mechanism from forced tool use, not a weaker one — but
        whether a given local model *obeys* it is unmeasured, which is what ``plan_source`` in
        each attempt exists to report.

        ``temperature`` 0 and the run's own ``seed``, because everything else in this repository
        pins its seed and a proposer that does not would make two legs of one comparison differ
        for a reason the comparison is not about.
        """
        import urllib.error
        import urllib.request

        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Cache breakpoints are an Anthropic billing feature; here they are just blocks
                # to join back. Dropping the marker text would make the archive disagree with
                # what was sent, so it stays in — see :func:`prompt_content`.
                content = "".join(str(block.get("text", "")) for block in content)
            chat.append({"role": str(message.get("role") or "user"), "content": str(content)})

        payload: dict[str, Any] = {
            "model": self._model[len(OLLAMA_PREFIX) :],
            "messages": chat,
            "stream": False,
            # Off, and this is the difference between this route working and not. A thinking
            # model puts its reasoning in ``message.thinking`` and the answer in
            # ``message.content``, and both come out of one output allowance. On the real
            # planning prompt ``gemma4:12b`` spent all 8,000 tokens thinking, returned
            # ``done_reason: length`` with **empty content**, and the node recorded a fallback —
            # so the first measurement of that model scored 0% and was measuring this, not the
            # model. With thinking off the same call answers in 6 output tokens.
            #
            # Safe on models that do not think: Ollama accepts the flag and ignores it (checked
            # on qwen2.5-coder:14b and medgemma:4b). And the two nodes on this route have their
            # answers read by a code gate rather than by a person, so the reasoning was never
            # consumed by anything — unlike the Critic's ``evidence``, which is why the Critic
            # stays on the other route.
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0, "seed": self.config.seed},
        }
        if schema is not None:
            payload["format"] = schema

        request = urllib.request.Request(
            f"{ollama_host()}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.llm_timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The body carries Ollama's own message ("model 'x' not found"), which is the one
            # thing that makes a 404 here actionable.
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise LLMUnavailable(f"ollama returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailable(f"could not reach ollama at {ollama_host()}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"ollama sent a body that is not JSON: {exc}") from exc

        message = body.get("message") or {}
        return OllamaCompletion(
            content=[OllamaTextBlock(text=str(message.get("content") or ""))],
            usage=OllamaUsage(
                input_tokens=body.get("prompt_eval_count"),
                output_tokens=body.get("eval_count"),
            ),
            # Ollama says ``length`` where the Anthropic route says ``max_tokens``; every caller
            # reads the latter, so the translation belongs here rather than in each of them.
            stop_reason="max_tokens" if body.get("done_reason") == "length" else "end_turn",
        )

    # -- public API -------------------------------------------------------- #

    def complete_text(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        iteration: int | None = None,
    ) -> TextCompletion:
        """Free-form completion. Used only where prose is the product (the report).

        Returns the text *and* whether it was cut off — see :class:`TextCompletion` for why
        that second field is not something the caller can be trusted to remember to ask for.
        """
        prompt = render_prompt(prompt_name, variables)
        label = archive_label(prompt_name, iteration=iteration)
        try:
            response = self._create(
                [{"role": "user", "content": prompt_content(prompt)}],
                system=system,
                max_tokens=max_tokens or self.config.llm_max_tokens,
                schema=None,
            )
        except LLMUnavailable as exc:
            self._archive_failure(label, prompt, system, str(exc))
            raise
        text = extract_text(response)
        self._archive(label, prompt, system, text, response)
        return TextCompletion(text, getattr(response, "stop_reason", None) == "max_tokens")

    def complete_json(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: dict[str, Any],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        """Schema-enforced completion with exactly one corrective retry.

        Raises :class:`LLMUnavailable` when the second attempt still fails to parse;
        the calling node is responsible for the deterministic fallback.
        """
        prompt = render_prompt(prompt_name, variables)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_content(prompt)}]
        last_error = ""

        for attempt in (1, 2):
            label = archive_label(prompt_name, iteration=iteration, attempt=attempt)
            try:
                response = self._create(
                    messages,
                    system=system,
                    max_tokens=max_tokens or self.config.llm_max_tokens,
                    schema=schema,
                )
            except LLMUnavailable as exc:
                self._archive_failure(label, prompt, system, str(exc))
                raise
            text = extract_text(response)
            payload = extract_structured(response)
            self._archive(
                label,
                prompt,
                system,
                text or (json.dumps(payload, indent=2, ensure_ascii=False) if payload else ""),
                response,
            )
            if payload is not None:
                return payload
            last_error = "the response contained no JSON object"

            if attempt == 1:
                # One corrective turn, then give up and let the node fall back.
                messages = [
                    *messages,
                    {"role": "assistant", "content": text or "(empty)"},
                    {
                        "role": "user",
                        "content": (
                            f"That response could not be parsed ({last_error}). Reply with a single "
                            "JSON object matching the schema exactly. No prose, no code fences."
                        ),
                    },
                ]

        raise LLMUnavailable(f"structured output failed twice for {prompt_name}: {last_error}")

    # -- archiving --------------------------------------------------------- #

    def _archive_failure(self, label: str, prompt: str, system: str | None, reason: str) -> None:
        """Record a call that never produced a response.

        Without this a fallback leaves no trace on disk, so a transient 5xx is
        indistinguishable after the fact from a malformed request.
        """
        payload = {
            "label": label,
            "model": self._model,
            "structured_mode": self._structured_mode,
            "route": "bedrock" if use_bedrock() else "anthropic",
            "system": system,
            "prompt": prompt,
            "error": reason,
        }
        self._write_artifact(f"{label}.error", payload)

    def _write_artifact(self, stem: str, payload: dict[str, Any]) -> None:
        """Write one numbered JSON artifact. Losing it must never fail the run."""
        directory = self.config.llm_dir
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{_next_index(directory):03d}_{stem}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _archive(
        self,
        label: str,
        prompt: str,
        system: str | None,
        text: str,
        response: Any,
    ) -> None:
        """Persist the exchange so the reasoning trail is auditable after the run."""
        usage = getattr(response, "usage", None)
        payload = {
            "label": label,
            "model": self._model,
            "structured_mode": self._structured_mode,
            "route": "bedrock" if use_bedrock() else "anthropic",
            "system": system,
            "prompt": prompt,
            "response": text,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                # Whether the cache breakpoints in the prompt actually held. Recorded because
                # a miss is invisible everywhere else: the run works, the numbers are right,
                # and only the bill moves — so without these two fields "we turned caching on"
                # is a claim no artifact can check. ``None`` on a route that reports neither.
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            },
            "stop_reason": getattr(response, "stop_reason", None),
        }
        self._write_artifact(label, payload)


def describe_api_error(exc: Any) -> str:
    """Summarise an ``APIStatusError`` without leaking credentials.

    The status code alone is not diagnosable: a 500 needs its ``request_id`` to be
    traceable, and a 400 needs the server's message to say *what* was rejected.
    The error body carries neither keys nor headers, so it is safe to log.
    """
    parts = [f"status {getattr(exc, 'status_code', '?')}"]

    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if message:
        parts.append(str(message))
    elif isinstance(body, dict) and body.get("message"):
        parts.append(str(body["message"]))

    request_id = getattr(exc, "request_id", None) or (
        body.get("request_id") if isinstance(body, dict) else None
    )
    if request_id:
        parts.append(f"request_id={request_id}")
    return ", ".join(parts)


def extract_text(response: Any) -> str:
    """Concatenate the text blocks, skipping thinking blocks."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def extract_structured(response: Any) -> dict[str, Any] | None:
    """Read the structured answer from whichever enforcement mode produced it.

    Forced tool use puts it in a ``tool_use`` block's ``input``; ``output_config``
    json_schema puts it in the text. ``None`` means neither yielded an object.
    """
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload

    text = extract_text(response)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
