"""원본 데이터 경계: 데이터 행에 닿은 것은 프롬프트에 닿지 않는다.

방어선 둘, 중요한 순서대로.

1. **구조로.** 원본 행은 subprocess로 도는 고정 스크립트(``scripts/profile.py``,
   ``scripts/train.py``) 안에서만 읽힌다. 경계의 비공개 절반(파일 경로와 목표 열)은 실행 노드만
   읽는 자기 state 채널 ``data_ref``에 산다.
2. **backstop으로.** :func:`assert_clean`이 API 호출 지점 하나에서 모든 렌더된 프롬프트에 돈다.
   등록된 비공개 재료(데이터셋 경로)는 전송되는 대신 *실행을 중단시키고*, 그저 파일시스템 경로처럼
   보이는 것은 가려진다.

왜 둘인지는 ``docs/rationale.md``. 여기서는 pandas를 import하지도 데이터 파일을 열지도 않는다:
이 모듈은 체이고 reader가 아니다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .scoring.intervals import (
    PAIRED_FIELDS,
    PAIRED_KEY,
    PAIRED_REASONS,
    PAIRED_STATUSES,
    RESAMPLE_UNITS,
)
from .scoring.metrics import METRICS, canonical


class RawDataLeak(RuntimeError):
    """등록된 비공개 재료가 나가는 프롬프트에서 발견됐다.

    추론 노드가 일부러 잡지 *않는다*: 유출은 이 코드의 결함이므로 실행이 시끄럽게 멈춘다.
    """


# --------------------------------------------------------------------------- #
# 프롬프트에 결코 나타나면 안 되는 문자열의 registry
# --------------------------------------------------------------------------- #

# 일부러 프로세스 전역: 가드는 API와 말하는 그 한 지점에서 닿을 수 있어야 하고, 그 지점은 실행의
# 데이터 출처를 아무것도 모른다.
_PRIVATE: set[str] = set()

# 이보다 짧으면 "비공개" 문자열이 프롬프트 절반과 우연히 일치한다.
_MIN_PRIVATE_LEN = 4


def register_private(*values: Any) -> None:
    """데이터셋 경로(또는 비슷한 것)를 프롬프트 금지어로 등록한다.

    값 하나가 세 형태를 낸다 — 준 그대로, resolve된 절대 경로, 파일 이름만. 어느 것이든 출처
    파일을 특정하기에 충분하기 때문이다.
    """
    for value in values:
        if not value:
            continue
        text = str(value)
        candidates = {text, text.replace("\\", "/")}
        path = Path(text)
        candidates.add(path.name)
        try:
            resolved = str(path.resolve())
            candidates.update({resolved, resolved.replace("\\", "/")})
        except OSError:  # pragma: no cover - resolve() on a hostile path
            pass
        _PRIVATE.update(item for item in candidates if len(item) >= _MIN_PRIVATE_LEN)


def clear_private() -> None:
    """등록된 문자열을 전부 잊는다. 테스트와 오래 사는 프로세스를 위해."""
    _PRIVATE.clear()


def private_strings() -> frozenset[str]:
    return frozenset(_PRIVATE)


# --------------------------------------------------------------------------- #
# 텍스트 씻어내기
# --------------------------------------------------------------------------- #

_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w\-.\\/]+")
_DATA_FILE = re.compile(
    r"[\w\-.]*[\w\-]\.(?:csv|tsv|parquet|feather|xlsx?|jsonl?|pkl|npy|npz)\b",
    re.IGNORECASE,
)
# 인용된 리터럴만 가린다. 숫자는 일부러 건드리지 않는다 — Critic이 읽는 OOM·타이밍 증거가 거기
# 있다 (``docs/rationale.md``).
_QUOTED = re.compile(r"(['\"])(?:(?!\1).){1,200}\1")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt):\s")

REDACTED_PATH = "<path>"
REDACTED_VALUE = "'<redacted>'"


def redact_paths(text: str) -> str:
    """파일시스템 경로와 데이터 파일 이름을 placeholder로 바꾼다."""
    return _DATA_FILE.sub(REDACTED_PATH, _ABS_PATH.sub(REDACTED_PATH, text))


def scrub_message(text: str, limit: int = 300) -> str:
    """학습 로그 꼬리를 씻어낸 진단 한 줄로 줄인다.

    예외 타입과 메시지를 남기고 — Critic에게 필요한 신호가 그것이다 — traceback과 경로, 인용된
    리터럴을 버린다. 셀 값이 숨는 곳이 마지막 것이다.
    """
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    chosen = next((line for line in reversed(lines) if _EXCEPTION_LINE.match(line)), lines[-1])
    return redact_paths(_QUOTED.sub(REDACTED_VALUE, chosen))[:limit]


def assert_clean(text: str, label: str = "prompt") -> str:
    """보낼 수 있는 ``text``. 등록된 재료에는 raise, 경로처럼 보이는 것은 가린다.

    비대칭은 의도다. 왜 한쪽은 중단이고 한쪽은 가림인지는 ``docs/rationale.md``.
    """
    for secret in _PRIVATE:
        if secret in text:
            raise RawDataLeak(
                f"{label}: 프롬프트에 비공개 데이터 참조가 포함되어 전송을 중단했습니다 "
                f"(길이 {len(secret)}자 문자열이 일치). "
                "dataset_card/result를 privacy.public_* 를 통과시키지 않은 노드가 있습니다."
            )
    return redact_paths(text)


# --------------------------------------------------------------------------- #
# 데이터셋 카드: 공개 요약 vs 비공개 데이터 참조
# --------------------------------------------------------------------------- #

# 카드 파일이 실어도 되는 단 하나의 비공개 키. 카드가 state에 들어가기 전에 떼어낸다.
DATA_KEY = "data"

# 손으로 쓴 카드가 예시 행을 몰래 넣는 데 쓸 수 있는 키들. 카드는 정의상 요약이므로 떨어뜨린다.
SAMPLE_KEYS = frozenset(
    {"sample", "samples", "sample_rows", "head", "rows", "examples", "preview", "raw", "raw_rows"}
)


# 카드가 실어도 되는 최상위 키 전부 — denylist가 아니라 allowlist다. 실패 양식은 아무도 생각하지
# 못한 키이고, 한 번 물렸다 (``docs/rationale.md``).
CARD_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "task",
    "target_column",
    "n_rows",
    "n_features",
    "n_features_dropped_non_numeric",
    "encoding",
    "n_informative",
    "n_classes",
    "class_balance",
    "imbalance_ratio",
    # 위 셋의 회귀 짝: 버킷과 비율만 담는다. 목표 열의 단위로 적힌 min·max·분위수는 예측 대상
    # 열의 셀 값이다.
    "target",
    "missing",
    "features",
    # 자유 서술, 그리고 사람이 손으로 쓰는 유일한 카드 필드 — 무엇으로 묶여 있는지는
    # :mod:`automl_agent.dataset.caveats`.
    "caveats",
    "preprocessing",
    "constraints",
    "difficulty",
    "feature_notes",
    "profile",
    "baseline",
    "simulate",
    DATA_KEY,
    "target_missing",
)

_SCALARS = (str, int, float, bool)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, _SCALARS) or value is None


def _is_number(value: Any) -> bool:
    """실수/정수이고 ``bool``이 아닐 때만 True. 값은 바꾸지 않는다 — ``414``는 ``414.0``이 아니다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CardSchemaError(ValueError):
    """카드가 실어도 되지 않는 것을 싣고 있다."""


def _check_value(value: Any, path: str) -> None:
    """규칙 하나를 재귀로: 컨테이너는 통과, 잎은 스칼라여야 한다.

    키별 타입 표가 아닌 이유는 ``docs/rationale.md``. 이 규칙이 정확히 행을 불가능하게 만드는
    규칙이다 — 레코드는 살 구조가 필요하다.
    """
    if _is_scalar(value):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            if name.lower() in SAMPLE_KEYS:
                raise CardSchemaError(
                    f"오류: 카드의 '{path}.{name}' 는 원본 행이 실릴 수 있는 키입니다. "
                    "카드는 열 전체에 대한 집계만 담습니다."
                )
            _check_value(item, f"{path}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_value(item, f"{path}[{index}]")
        return
    raise CardSchemaError(
        f"오류: 카드의 '{path}' 값 타입({type(value).__name__})은 허용되지 않습니다. "
        "스칼라·배열·객체만 쓸 수 있습니다."
    )


def validate_card(card: Any) -> dict[str, Any]:
    """fail-closed 스키마 검사. 카드를 그대로 돌려주거나 ``CardSchemaError``.

    *첫* 번째 선이다: 모르는 키는 전달되는 대신 실행을 멈춘다. :func:`public_card`가 뒤에 두 번째
    선으로 그대로 남아 있는 이유는 ``docs/rationale.md``.
    """
    if not isinstance(card, dict):
        raise CardSchemaError("오류: 데이터셋 카드는 JSON 객체여야 합니다.")
    unknown = [str(key) for key in card if key not in CARD_KEYS]
    if unknown:
        raise CardSchemaError(
            "오류: 카드에 허용되지 않은 키가 있습니다: "
            + ", ".join(repr(key) for key in unknown)
            + "\n  - 허용되는 최상위 키: "
            + ", ".join(CARD_KEYS)
            + "\n  - 카드는 집계 요약입니다. 원본 행이 실릴 자리를 두지 않기 위해 "
            "모르는 키는 통과시키지 않고 중단합니다."
        )
    for key, value in card.items():
        _check_value(value, str(key))
    return card


def public_card(card: dict[str, Any]) -> dict[str, Any]:
    """추론 노드가 봐도 되는 모양의 카드: 경로 없음, 예시 행 없음."""
    return {
        key: value
        for key, value in (card or {}).items()
        if key != DATA_KEY and key.lower() not in SAMPLE_KEYS
    }


def data_ref(
    card: dict[str, Any],
    path: Any = None,
    target_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, Any]:
    """비공개 ``data_ref`` 채널을 만든다. 명시된 인자가 카드를 이긴다.

    빈 dict는 "실제 데이터 없음": 실행기가 카드에 적힌 모양으로 행을 합성하고, 그래서 루프가
    카드만으로도 돈다.

    ``group_column``이 공개 카드가 아니라 여기로 오는 이유는 ``path``와 같다 — 어느 행이 떼어지는지를
    정하므로 어떤 프롬프트도 볼 수 없다 (:mod:`automl_agent.scoring.splits`).
    """
    declared = dict((card or {}).get(DATA_KEY) or {})
    resolved_path = path or declared.get("path")
    if not resolved_path:
        return {}
    target = target_column or declared.get("target_column") or (card or {}).get("target_column") or "target"
    reference = {"path": str(resolved_path), "target_column": str(target)}
    grouped = group_column or declared.get("group_column")
    if grouped:
        reference["group_column"] = str(grouped)
    return reference


# --------------------------------------------------------------------------- #
# 학습 결과: denylist가 아니라 allowlist
# --------------------------------------------------------------------------- #

# 오케스트레이터가 학습 실행에서 state에 남기는 것 전부. ``log_tail``과 ``artifacts``는 설계상
# 없다 — 전체 로그는 디스크에 남고 프롬프트가 렌더되는 채널에 들어가지 않는다.
PUBLIC_RESULT_FIELDS: tuple[str, ...] = (
    "status",
    # 지표가 어느 행에 대한 것인지: 루프 중에는 "val", 루프 뒤 한 번의 측정은 "test".
    # ``model_path``와 ``schema_path``는 일부러 이 목록에 없다 — 적합된 estimator는 데이터와
    # 등가이고, 스키마는 범주 수준과 클래스 라벨을 글자 그대로 나열한다.
    "split",
    "error_type",
    "train_time_sec",
    "wall_time_sec",
    "returncode",
    "dry_run",
    "dropped_hyperparams",
)

# 아래 넷은 키만이 아니라 *값*까지 걸러야 해서 따로 나른다. 필터는 값이 아니라 모양에 대한
# 가드다 — 앞으로 생길 키가 객체로 도착하는 것을 막는다 (``docs/rationale.md``).
APPLIED_HYPERPARAMS_KEY = "applied_hyperparams"
# {"impute": "median", "scale": true} — 실행기가 실제로 세운 파이프라인.
APPLIED_PREPROCESSING_KEY = "applied_preprocessing"
# {"held_out_rows": 414, "fit_rows": 2346, ...} — estimator의 early stopping이 떼어 둔 행 *개수*,
# 그리고 결정 cut의 요청과 실행기의 거절 이유(``cut_requested``/``cut_declined``). 떼어 둔 것이 없고
# cut을 묻지도 않은 흔한 경우에는 없다.
INTERNAL_VALIDATION_KEY = "internal_validation"
# ``["missing_indicator(gcs, paco2)", "impute(median (constant: gcs, paco2))", "scale(auto)"]`` —
# ``pipeline`` 명세가 실제로 낸 단계들, 순서대로 한 줄씩. mapping이 아니라 list여서 아래에
# ``_public_params`` 대신 자기 필터가 붙는다.
APPLIED_PIPELINE_KEY = "applied_pipeline"
MAX_PIPELINE_LINE = 500


def _public_params(params: Any) -> dict[str, Any]:
    """렌더해도 되는 하이퍼파라미터 값: 스칼라, 또는 스칼라의 평평한 컨테이너.

    LLM의 제안을 sklearn 파라미터 이름으로 좁힌 것이므로 행이 도착할 데이터 경로가 없다. 필터는
    반대 경우를 위해 있다 — 프롬프트에 서식될 이유가 없는 객체나 중첩 구조.
    """
    clean: dict[str, Any] = {}
    for key, value in (params if isinstance(params, dict) else {}).items():
        if _is_scalar(value):
            clean[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(_is_scalar(item) for item in value):
            clean[str(key)] = list(value)
        elif isinstance(value, dict) and all(
            _is_scalar(item) for pair in value.items() for item in pair
        ):
            # 한 단계만: 값은 가중치이고, 여기서 중첩 구조는 아예 다른 것이다.
            clean[str(key)] = {str(name): item for name, item in value.items()}
    return clean


def _public_paired(block: Any) -> dict[str, Any]:
    """짝지은 비교 블록, 키 하나씩, 또는 ``{}``.

    모든 문자열 필드를 ``str``이 아니라 자기 어휘에 대고 검사한다 — ``status``·``unit``·``reason``은
    :mod:`automl_agent.scoring.intervals`가 쓰는 낱말에, ``metric``은 지표 registry에. 왜 그래야
    하는지는 ``docs/rationale.md``.
    """
    raw = block if isinstance(block, dict) else {}
    clean: dict[str, Any] = {}
    if raw.get("status") in PAIRED_STATUSES:
        clean["status"] = raw["status"]
    if raw.get("unit") in RESAMPLE_UNITS:
        clean["unit"] = raw["unit"]
    if canonical(str(raw.get("metric") or "")) in METRICS:
        # 정규화된 이름이 아니라 호출자가 쓴 별칭: 이것은 passthrough이고, 장부는 이 필드를 목표가
        # 말하는 것과 같은 지표 이름 옆에 찍는다.
        clean["metric"] = raw["metric"]
    if str(raw.get("reason") or "") in PAIRED_REASONS:
        clean["reason"] = raw["reason"]
    for key in (*PAIRED_FIELDS, "baseline_iteration", "resamples"):
        if _is_number(raw.get(key)):
            clean[key] = raw[key]
    if isinstance(raw.get("threads_changed"), bool):
        # 여기 유일한 boolean이라 자기 줄이 필요하다: 위 루프는 ``bool``을 거절한다 — 델타가 와야 할
        # 자리의 ``True``는 점수로 렌더된다 (:mod:`automl_agent.threads`).
        clean["threads_changed"] = raw["threads_changed"]
    return clean


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    """날 ``result.json`` 짐을 allowlist를 통과시켜 공유 가능한 모양으로.

    지표는 숫자일 때만 남는다 — 앞으로의 model wrapper에서 나온 엉뚱한 문자열이 얹혀 올 자리가
    지표 칸이다.
    """
    raw = dict(result or {})
    metrics = raw.get("metrics")
    clean: dict[str, Any] = {
        "metrics": {
            key: value
            for key, value in (metrics if isinstance(metrics, dict) else {}).items()
            if _is_number(value)
        }
    }
    for key in (APPLIED_HYPERPARAMS_KEY, APPLIED_PREPROCESSING_KEY, INTERNAL_VALIDATION_KEY):
        if key in raw:
            clean[key] = _public_params(raw[key])
    if APPLIED_PIPELINE_KEY in raw:
        lines = raw[APPLIED_PIPELINE_KEY]
        clean[APPLIED_PIPELINE_KEY] = [
            line[:MAX_PIPELINE_LINE]
            for line in (lines if isinstance(lines, list) else [])
            if isinstance(line, str)
        ]
    if PAIRED_KEY in raw:
        clean[PAIRED_KEY] = _public_paired(raw[PAIRED_KEY])
    for field in PUBLIC_RESULT_FIELDS:
        if field in raw:
            clean[field] = raw[field]
    summary = scrub_message(str(raw.get("log_tail") or ""))
    if summary and clean.get("status") != "ok":
        # 살아남는 단 하나의 텍스트 필드: 씻어낸 예외 줄.
        clean["error_summary"] = summary
    return clean
