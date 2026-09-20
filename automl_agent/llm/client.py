"""Anthropic API와 이야기하는 단 하나의 모듈.

노드를 단순하게 두기 위해 여기 모아 둔 책임들 — 프롬프트 로딩, 구조화 출력 강제, 정정 재시도 한
번, 타임아웃, 전송 재시도, 그리고 모든 프롬프트와 응답 짝을 ``artifacts/<thread_id>/llm/``에
남기는 것.

자격 증명은 환경에서만 읽고(``ANTHROPIC_API_KEY``, 또는 ``AUTOML_USE_BEDROCK``이 설정됐으면
``AWS_REGION``) 로그·아티팩트·예외 메시지에 절대 쓰지 않는다.

유일한 외부 출구이므로 원본 데이터 가드도 여기 있다 — :func:`render_prompt`와
:mod:`automl_agent.privacy`.
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
    """호출을 끝낼 수 없을 때 올린다. 호출자는 결정적으로 폴백한다."""


class TextCompletion(NamedTuple):
    """자유 서술 완성과, 모델이 끝내기 전에 잘렸는지.

    잘린 응답은 전송 계층이 올릴 수 있는 오류가 아니라 우연히 불완전한, 형식이 올바른 답이다.
    그래서 그것을 알아챌 수 있는 자리는 반환하는 여기뿐이다.
    """

    text: str
    truncated: bool


# 구조화 출력을 tool use로 강제해야 할 때 쓰는 단 하나의 tool 이름.
STRUCTURED_TOOL_NAME = "submit_result"

# 이것으로 시작하는 모델 id는 API가 아니라 로컬에서 서비스되는 모델로 간다. 별도 플래그가 아니라
# id의 접두사인 이유는 여기서 "어느 모델"과 "어느 전송"이 한 결정이기 때문이다 — ``gemma3``를
# Anthropic으로 보내고 싶은 구성은 없다.
OLLAMA_PREFIX = "ollama:"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def needs_anthropic(config: RunConfig) -> bool:
    """이 실행의 노드 중 하나라도 Anthropic API에 닿는지.

    두 반쪽이 따로 풀린다 — proposer는 ``proposer_model``이 설정돼 있으면 그것을, 아니면
    ``llm_model``을 따른다. 자격 증명이 필요 없는 것은 전부 로컬인 경우뿐이고,
    ``main.check_credentials``가 이것을 근거로 그래프 시작 전에 거절한다.
    """
    proposer = config.proposer_model or config.llm_model
    return not (
        config.llm_model.startswith(OLLAMA_PREFIX) and proposer.startswith(OLLAMA_PREFIX)
    )


def ollama_host() -> str:
    """로컬 서버가 어디 있는지. ``OLLAMA_HOST``는 Ollama 자신의 변수이므로, 이미 다른 자리에서
    그것을 돌리는 기계는 여기서 따로 설정할 것이 없다."""
    host = (os.environ.get(OLLAMA_HOST_ENV) or "").strip() or DEFAULT_OLLAMA_HOST
    return host if "://" in host else f"http://{host}"


class OllamaTextBlock(NamedTuple):
    """텍스트 블록 하나. SDK의 것과 같은 모양이라 :func:`extract_text`에 분기가 필요 없다."""

    text: str
    type: str = "text"


class OllamaUsage(NamedTuple):
    """Ollama의 토큰 수를, 아카이브가 이미 쓰는 이름으로.

    캐시 필드가 없는 것은 일부러다. ``_archive``가 ``getattr(..., None)``으로 읽으므로 없으면
    ``null``이 기록되고, 그것이 캐시를 놓친 경로가 아니라 맞힐 프롬프트 캐시가 아예 없는 경로에
    대한 정직한 답이다.
    """

    input_tokens: int | None
    output_tokens: int | None


class OllamaCompletion(NamedTuple):
    """:func:`extract_text`, :func:`extract_structured`, 아카이브에 필요한 만큼의 ``Message``.
    그 이상은 일부러 아니다 — 더 완전한 모방은 두 경로를, 실제로는 아닌 방식으로 서로 바꿔 쓸 수
    있는 것처럼 다루는 코드를 불러들인다."""

    content: list[OllamaTextBlock]
    usage: OllamaUsage
    stop_reason: str

# 구조화 출력을 어떻게 강제하는지. 프로세스 수명 동안 전송별로 캐시한다. 모든 노드가 자기
# LLMClient를 만들므로 이것을 인스턴스마다 들고 있으면, 노드마다 거절당하는 `output_config`
# 왕복을 다시 치르고서야 내려앉는다.
_STRUCTURED_MODE: dict[str, str] = {}


def _route_key() -> str:
    return "bedrock" if use_bedrock() else "anthropic"


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

# 프롬프트의 캐시 가능한 접두사가 끝나는 자리. 여기서 정하지 않고 ``.md`` 파일에 적는 이유는
# *어느* 블록이 안정적인지가 프롬프트의 성질이기 때문이다.
CACHE_MARKER = "<!-- cache -->"
# API가 요청당 넷을 허용한다. 그보다 많이 요구하는 프롬프트는 전송 계층 안쪽의 400이 아니라
# 이름을 붙여 줄 값이 있는 실수다.
MAX_CACHE_BREAKPOINTS = 4
# 기본 5분이 아니라 1시간. 같은 노드의 두 호출 사이 간격이 반복 하나 — 수 분이 걸릴 수 있는
# 적합 — 이고, 5분 항목은 그것을 쓴 요청의 *시작*부터 재므로 다음 호출 때는 보통 식어 있다.
# 1시간 쓰기는 1.25배가 아니라 2배이고 읽기 세 번이면 본전인데, 한 실행이 아홉 번 읽는다.
CACHE_TTL = "1h"


def prompt_content(prompt: str) -> str | list[dict[str, Any]]:
    """user 메시지 하나의 content를, :data:`CACHE_MARKER`마다 캐시 블록으로 자른다.

    마커가 없는 프롬프트는 문자열을 그대로 돌려준다. 그래서 캐시를 위해 순서를 잡지 않은
    프롬프트(``report.md`` — 실행당 한 번 호출, 재사용할 것이 없다)는 늘 갖던 모양을 유지한다.

    마커는 전송되는 텍스트에 남긴다. 몇 토큰이고, :meth:`LLMClient._archive`가 정직한 상태를
    유지한다 — 아카이브가 보여 주는 것이 여전히 모델에 보낸 것과 바이트 단위로 같다.
    """
    if CACHE_MARKER not in prompt:
        return prompt
    head, *rest = prompt.split(CACHE_MARKER)
    # 각 마커는 자기 앞 블록을 끝내므로, 마커 텍스트는 그 블록에 속한다.
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
    """``<name>.md``를 읽고 ``{{var}}`` 자리표시자를 치환한다.

    ``str.format``이 아닌 것은 일부러다 — 프롬프트에 그것을 깨뜨리는 JSON 중괄호가 들어 있다.
    문자열이 아닌 값은 들여쓴 JSON으로 직렬화해서 모델이 읽을 수 있는 데이터를 보게 한다.

    이 시스템의 모든 프롬프트가 여기서 만들어진다 — ``--dry-run``과 ``--no-llm``에서 아카이브만
    되는 것까지 — 그래서 여기가 :func:`automl_agent.privacy.assert_clean`의 단일 관문이다. 등록된
    데이터셋 경로는 :class:`~automl_agent.privacy.RawDataLeak`을 올리고, 경로 모양인 것은 가려진다.
    따라서 아카이브가 보여 주는 것은 모델에 보낸 것과 바이트 단위로 같다.
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
    """API를 부르지 않고 렌더된 프롬프트만 아카이브한다(``--dry-run`` 경로).

    ``--dry-run``에서도 실제로 렌더하는 것은 일부러다. 템플릿에 채워지지 않은 자리표시자가 없음을
    보이고, 추론 흔적 — 예컨대 반복 2의 planning 프롬프트가 정말 반복 1의 ``failure_type``과
    ``direction``을 담고 있다는 것 — 을 토큰 한 개 쓰지 않고 감사할 수 있게 한다.
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
    """아카이브된 교환 하나에 이름을 붙여, 그것을 만든 반복으로 되짚을 수 있게 한다.

    ``attempt``는 정정 재시도 계수기이고 루프 계수기가 아니다.
    """
    parts = [prompt_name]
    if iteration:
        parts.append(f"iter{iteration}")
    if attempt > 1:
        # 접미사는 실제 재시도에만 붙인다. 그래야 흔한 경우가 읽기 쉽다.
        parts.append(f"attempt{attempt}")
    return "_".join(parts)


def _next_index(directory: Path) -> int:
    """다음 순번. 아카이브된 파일을 *전부* 센다.

    두 writer가 계수기를 공유한다. ``--dry-run``과 실제 경로를 번갈아 쓰는 ``--force`` 재실행이
    같은 디렉터리에 쓰므로, 한쪽 접미사만 세면 번호가 처음으로 돌아가 다른 경로의 흔적을 덮어쓴다.
    """
    return len([path for path in directory.glob("*") if path.is_file()]) + 1


class LLMClient:
    """``messages.create``를 감싼, 얇고 기록을 남기는 래퍼."""

    def __init__(self, config: RunConfig, *, proposer: bool = False) -> None:
        """판단하지 않고 제안하는 두 노드는 ``proposer=True``.

        모델 문자열이 아니라 키워드인 이유는, 두 호출자가 *자기가 어느 노드인지*만 말하고 대응은
        이 모듈이 갖게 하려는 것이다. 다른 방법 — 노드마다 ``config``를 읽고 필드를 고르는 것 — 은
        같은 세 줄짜리 결정을 네 곳에 두고, 나중에 더해진 다섯 번째 노드가 조용히 틀린다.
        """
        self.config = config
        self._client: Any | None = None
        self._model = (config.proposer_model if proposer else "") or config.llm_model

    @property
    def _structured_mode(self) -> str:
        """더 엄격한 ``output_config`` json_schema로 시작하고, 어느 전송이 그것을 거절하면
        강제 tool use로 내려앉는다."""
        return _STRUCTURED_MODE.get(_route_key(), "output_config")

    @_structured_mode.setter
    def _structured_mode(self, mode: str) -> None:
        _STRUCTURED_MODE[_route_key()] = mode

    # -- 전송 --------------------------------------------------------- #

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
                # SDK는 서명할 때만 botocore를 import하므로, 이것이 없으면 실패가
                # ModuleNotFoundError로 빠져나가 결정적인 폴백 대신 그래프를 중단시킨다.
                raise LLMUnavailable(
                    'botocore is not installed; run `pip install "anthropic[bedrock]"` '
                    "for the Bedrock route"
                )
            self._client = anthropic.AnthropicBedrockMantle(
                aws_region=region,
                timeout=self.config.llm_timeout_sec,
                max_retries=DEFAULT_LLM_MAX_RETRIES,
            )
            # Bedrock은 모델 id에 벤더를 적으므로 기본값 ``claude-opus-5``가
            # ``anthropic.claude-opus-5``가 되어야 한다. 검사가 앞자리가 아니라 id의 *어디든*을
            # 보는 이유 — cross-region inference profile은 벤더 앞에 스코프를 붙이므로
            # (``global.anthropic.claude-...``) 앞자리 앵커로 보면 접두사가 두 번 붙는다.
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
            # 모든 전송이 `output_config`를 받지는 않는다 — Bedrock 경로는
            # "output_config.format: Extra inputs are not permitted"로 거절한다. 스펙도 허용하고
            # 모든 경로가 지원하는 강제 tool use로 내려앉고, 그 선택을 남은 실행 동안 기억한다.
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
        """같은 호출을 로컬에서 서비스되는 모델에 대고 한다.

        클라이언트 라이브러리가 아니라 ``urllib``인 이유 — 이것은 JSON 본문 하나를 실은 POST 한
        번이고, Anthropic SDK가 의존성 값을 하는 이유(재시도, 서명, 스트리밍, 타입 있는 오류)는
        localhost로 가는 요청에 해당하지 않는다.

        구조화 출력은 Ollama의 ``format`` 필드이고 JSON 스키마를 그대로 받는다. 이 경로가
        성립하는 이유가 그것이다 — 노드가 의지하는 강제(``validate_plan``, registry, clamp)는
        스키마 *뒤*에 있고, 스키마가 아예 없는 경로는 형식이 깨진 답을 전부 그 가드들에 떠넘기고
        폴백으로 기록한다. 강제 tool use와 다른 기제이고 더 약한 기제는 아니다 — 다만 주어진 로컬
        모델이 그것을 *지키는지*는 측정되지 않았고, 그것을 보고하기 위해 시도마다
        ``plan_source``가 있다.

        ``temperature``는 0, ``seed``는 실행 자신의 것이다. 이 저장소의 다른 모든 것이 시드를
        고정하는데 proposer가 그러지 않으면, 한 비교의 두 다리가 그 비교가 다루지 않는 이유로
        달라진다.
        """
        import urllib.error
        import urllib.request

        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # 캐시 breakpoint는 Anthropic의 청구 기능이고, 여기서는 다시 이어 붙일
                # 블록일 뿐이다. 마커 텍스트를 떼면 아카이브가 보낸 것과 어긋나므로 남긴다
                # (:func:`prompt_content`).
                content = "".join(str(block.get("text", "")) for block in content)
            chat.append({"role": str(message.get("role") or "user"), "content": str(content)})

        payload: dict[str, Any] = {
            "model": self._model[len(OLLAMA_PREFIX) :],
            "messages": chat,
            "stream": False,
            # 끈다. 이것이 이 경로가 되는 것과 안 되는 것의 차이다 — thinking 모델은 추론을
            # ``message.thinking``에, 답을 ``message.content``에 넣고 둘이 하나의 출력 허용량에서
            # 나온다. 생각하지 않는 모델에서도 안전하다(Ollama가 플래그를 받고 무시한다).
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
            # 본문에 Ollama 자신의 메시지("model 'x' not found")가 있고, 여기서 404를 조치
            # 가능하게 만드는 것은 그것뿐이다.
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
            # Anthropic 경로가 ``max_tokens``라고 하는 자리에서 Ollama는 ``length``라고 한다.
            # 모든 호출자가 후자를 읽으므로, 번역은 그들 각각이 아니라 여기 있어야 한다.
            stop_reason="max_tokens" if body.get("done_reason") == "length" else "end_turn",
        )

    # -- 공개 API -------------------------------------------------------- #

    def complete_text(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        iteration: int | None = None,
    ) -> TextCompletion:
        """자유 서술 완성. 산문 자체가 산출물인 곳(보고서)에서만 쓴다.

        텍스트*와* 잘렸는지를 함께 돌려준다 — 두 번째 필드를 호출자가 물어 볼 것이라고 믿을 수
        없는 이유는 :class:`TextCompletion`에.
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
        """스키마를 강제하는 완성. 정정 재시도는 정확히 한 번.

        두 번째 시도도 파싱에 실패하면 :class:`LLMUnavailable`을 올린다. 결정적인 폴백은
        호출하는 노드의 책임이다.
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
                # 정정 한 턴, 그다음에는 포기하고 노드가 폴백하게 둔다.
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

    # -- 보관 --------------------------------------------------------- #

    def _archive_failure(self, label: str, prompt: str, system: str | None, reason: str) -> None:
        """응답을 한 번도 내지 못한 호출을 기록한다.

        이것이 없으면 폴백이 디스크에 흔적을 남기지 않고, 그러면 일시적인 5xx와 형식이 깨진 요청을
        사후에 구분할 수 없다.
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
        """번호가 붙은 JSON 아티팩트 하나를 쓴다. 그것을 잃는 것이 실행을 실패시켜서는 안 된다."""
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
        """교환을 남겨서 실행 뒤에도 추론 흔적을 감사할 수 있게 한다."""
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
                # 프롬프트의 캐시 breakpoint가 실제로 들었는지. 기록하는 이유는 놓친 것이
                # 다른 어디에서도 보이지 않기 때문이다 — 실행은 되고 수도 맞고 청구서만 움직인다.
                # 이 두 필드가 없으면 "캐싱을 켰다"는 어느 아티팩트도 확인할 수 없는 주장이다.
                # 둘 다 보고하지 않는 경로에서는 ``None``.
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            },
            "stop_reason": getattr(response, "stop_reason", None),
        }
        self._write_artifact(label, payload)


def describe_api_error(exc: Any) -> str:
    """자격 증명을 흘리지 않고 ``APIStatusError``를 요약한다.

    상태 코드만으로는 진단이 안 된다. 500은 추적 가능하려면 ``request_id``가 필요하고, 400은
    *무엇이* 거절됐는지 말해 주는 서버 메시지가 필요하다. 오류 본문에는 키도 헤더도 없으므로
    로그에 남겨도 안전하다.
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
    """텍스트 블록들을 이어 붙인다. thinking 블록은 건너뛴다."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def extract_structured(response: Any) -> dict[str, Any] | None:
    """어느 강제 방식이 만들었든 구조화된 답을 읽는다.

    강제 tool use는 그것을 ``tool_use`` 블록의 ``input``에 넣고, ``output_config`` json_schema는
    텍스트에 넣는다. ``None``은 둘 다 객체를 내지 않았다는 뜻이다.
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
