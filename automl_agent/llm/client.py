"""The only module that talks to an LLM (Anthropic API, Bedrock, or a local Ollama server).

Roles:

* Routing — pick the transport for a model id.
* Ollama answer shape — make a local answer look like SDK.
* Prompt building — fill prompt files, check raw data, split cache blocks.
* Archive — save every prompt and response to disk.
* Calls — send, force JSON, retry once, raise LLMUnavailable.
* Response reading — pull text or JSON out of a response.
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
    """Raised when a call cannot be finished. The caller then uses its fixed fallback."""


class TextCompletion(NamedTuple):
    """A free-text answer, plus whether it was cut off"""

    text: str
    truncated: bool


# Tool name used when JSON is forced via tool use.
STRUCTURED_TOOL_NAME = "submit_result"

# Model ids with this prefix go to local Ollama.
OLLAMA_PREFIX = "ollama:"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


# --- Role: routing ----------------------------------------------------------------


def needs_anthropic(config: RunConfig) -> bool:
    """Tell whether any node calls a remote API; False only when all models are Ollama.

    ``main.check_credentials`` uses it to refuse a run early
    """
    proposer = config.proposer_model or config.llm_model
    return not (
        config.llm_model.startswith(OLLAMA_PREFIX) and proposer.startswith(OLLAMA_PREFIX)
    )


def ollama_host() -> str:
    """Return the local Ollama base URL from ``OLLAMA_HOST``, adding ``http://`` if missing."""
    host = (os.environ.get(OLLAMA_HOST_ENV) or "").strip() or DEFAULT_OLLAMA_HOST
    return host if "://" in host else f"http://{host}"


# --- Role: Ollama answer shape ----------------------------------------------------


class OllamaTextBlock(NamedTuple):
    """One text block of an Ollama answer, shaped like the SDK's so :func:`extract_text` reads both."""

    text: str
    type: str = "text"


class OllamaUsage(NamedTuple):
    """Ollama's token counts; no cache fields on purpose"""

    input_tokens: int | None
    output_tokens: int | None


class OllamaCompletion(NamedTuple):
    """An Ollama answer shaped like an SDK ``Message``, only the fields we read."""

    content: list[OllamaTextBlock]
    usage: OllamaUsage
    stop_reason: str

# JSON forcing mode per transport; shared by all clients on purpose.
_STRUCTURED_MODE: dict[str, str] = {}


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

# End of a cacheable prefix; set in prompt files
CACHE_MARKER = "<!-- cache -->"
# API limit on cache breakpoints per request.
MAX_CACHE_BREAKPOINTS = 4
# One hour, not five minutes
CACHE_TTL = "1h"


# --- Role: prompt building --------------------------------------------------------


def prompt_content(prompt: str) -> str | list[dict[str, Any]]:
    """Split a prompt into cache blocks at each :data:`CACHE_MARKER`; no marker, no split.

    Raises ValueError above :data:`MAX_CACHE_BREAKPOINTS`. Markers stay in the text.
    """
    if CACHE_MARKER not in prompt:
        return prompt
    head, *rest = prompt.split(CACHE_MARKER)
    # Each marker belongs to the block it ends.
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
    """Fill ``{{var}}`` in ``<name>.md`` (not ``str.format``: JSON braces), then run the privacy check.

    Raises KeyError for an unfilled placeholder, RawDataLeak for raw data.
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


# --- Role: archive ----------------------------------------------------------------


def archive_prompt_only(config: RunConfig, label: str, prompt: str) -> Path | None:
    """Save a rendered prompt as ``<NNN>_<label>.dryrun.md`` without calling the API.

    Returns the path, or ``None`` if the write failed.
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
    """Name an archived exchange, e.g. ``planning_iter2`` or ``planning_iter2_attempt2``.

    ``attempt`` is the correction retry, added only above 1
    """
    parts = [prompt_name]
    if iteration:
        parts.append(f"iter{iteration}")
    if attempt > 1:
        parts.append(f"attempt{attempt}")
    return "_".join(parts)


def _next_index(directory: Path) -> int:
    """_next_index | Archive: next file number; counts every file, so both writers share it."""
    return len([path for path in directory.glob("*") if path.is_file()]) + 1


# --- Role: calls ------------------------------------------------------------------


class LLMClient:
    """A thin ``messages.create`` wrapper for one node that archives every exchange.

    Every failure becomes :class:`LLMUnavailable`.
    """

    def __init__(self, config: RunConfig, *, proposer: bool = False) -> None:
        """Make a client; ``proposer=True`` uses ``proposer_model`` if set"""
        self.config = config
        self._client: Any | None = None
        self._model = (config.proposer_model if proposer else "") or config.llm_model

    @property
    def _route(self) -> str:
        """_route | Calls: the transport this client really uses (ollama, bedrock, anthropic)."""
        # Per client: one run can use two transports.
        if self._model.startswith(OLLAMA_PREFIX):
            return "ollama"
        return "bedrock" if use_bedrock() else "anthropic"

    @property
    def _structured_mode(self) -> str:
        """_structured_mode | Calls: how this transport forces JSON (output_config, tool, format)."""
        # ``tool`` once a transport rejects ``output_config``.
        if self._route == "ollama":
            return "format"
        return _STRUCTURED_MODE.get(self._route, "output_config")

    @_structured_mode.setter
    def _structured_mode(self, mode: str) -> None:
        _STRUCTURED_MODE[self._route] = mode

    # -- transport ---------------------------------------------------- #

    def _ensure_client(self) -> Any:
        """_ensure_client | Calls: build the SDK client once, checking credentials first."""
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
                # Fail early so the node falls back, not crashes.
                raise LLMUnavailable(
                    'botocore is not installed; run `pip install "anthropic[bedrock]"` '
                    "for the Bedrock route"
                )
            self._client = anthropic.AnthropicBedrockMantle(
                aws_region=region,
                timeout=self.config.llm_timeout_sec,
                max_retries=DEFAULT_LLM_MAX_RETRIES,
            )
            # Add vendor prefix unless present anywhere
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
        """_request_kwargs | Calls: build the ``messages.create`` arguments for the current mode."""
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
        """_create | Calls: send one request, falling back to tool use if output_config is refused."""
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
            # Bedrock rejects `output_config`; switch to tool use for good.
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
        """_create_ollama | Calls: send the same request to the local Ollama server with urllib."""
        import urllib.error
        import urllib.request

        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Ollama has no cache blocks; join them back.
                content = "".join(str(block.get("text", "")) for block in content)
            chat.append({"role": str(message.get("role") or "user"), "content": str(content)})

        payload: dict[str, Any] = {
            "model": self._model[len(OLLAMA_PREFIX) :],
            "messages": chat,
            "stream": False,
            # Thinking can eat the whole budget
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
            # Keep Ollama's message; it explains a 404.
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
            # Map Ollama's ``length`` to Anthropic's ``max_tokens``.
            stop_reason="max_tokens" if body.get("done_reason") == "length" else "end_turn",
        )

    # -- public API ---------------------------------------------------- #

    def complete_text(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        iteration: int | None = None,
    ) -> TextCompletion:
        """Get a free-text answer (only for the report), archived; raises LLMUnavailable."""
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
        """Get a JSON object for ``schema``, with one correction retry; each try is archived.

        Raises LLMUnavailable on failure or no JSON twice; the caller falls back.
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
                # One correction turn, then give up.
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

    # -- archive -------------------------------------------------------- #

    def _archive_failure(self, label: str, prompt: str, system: str | None, reason: str) -> None:
        """_archive_failure | Archive: record a call that got no answer, so fallbacks leave a trace."""
        payload = {
            "label": label,
            "model": self._model,
            "structured_mode": self._structured_mode,
            "route": self._route,
            "system": system,
            "prompt": prompt,
            "error": reason,
        }
        self._write_artifact(f"{label}.error", payload)

    def _write_artifact(self, stem: str, payload: dict[str, Any]) -> None:
        """_write_artifact | Archive: write one numbered JSON file; a write error is ignored."""
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
        """_archive | Archive: save one prompt/response pair so it can be checked after the run."""
        usage = getattr(response, "usage", None)
        payload = {
            "label": label,
            "model": self._model,
            "structured_mode": self._structured_mode,
            "route": self._route,
            "system": system,
            "prompt": prompt,
            "response": text,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                # Cache hit or miss; seen nowhere else.
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            },
            "stop_reason": getattr(response, "stop_reason", None),
        }
        self._write_artifact(label, payload)


# --- Role: response reading -------------------------------------------------------


def describe_api_error(exc: Any) -> str:
    """Summarise an ``APIStatusError`` as ``status 400, <message>, request_id=<id>``.

    Safe to log: the error body holds no keys or headers.
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
    """Join a response's text blocks, skipping thinking blocks, and strip it."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def extract_structured(response: Any) -> dict[str, Any] | None:
    """Read the JSON answer from a ``tool_use`` block or the text; ``None`` if none."""
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
