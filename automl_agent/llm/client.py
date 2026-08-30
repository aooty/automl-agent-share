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

# How structured output is enforced, cached per transport for the life of the process.
# Every node builds its own LLMClient, so holding this per instance would make each
# node re-pay a rejected `output_config` round-trip before downgrading again.
_STRUCTURED_MODE: dict[str, str] = {}


def _route_key() -> str:
    return "bedrock" if use_bedrock() else "anthropic"


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


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

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._model = config.llm_model

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
            if not self._model.startswith("anthropic."):
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
                [{"role": "user", "content": prompt}],
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
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
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
