"""The raw data boundary: nothing that touched data rows may reach a prompt.

Roles:

* Private registry — remember paths no prompt may contain.
* Text scrubbing — hide paths and quoted values, stop leaks.
* Dataset card — check a card, split public and private parts.
* Training result — allowlist a raw result before it enters state.
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
    as_number,
)
from .scoring.metrics import METRICS, canonical


class RawDataLeak(RuntimeError):
    """Raised when a registered private string is found in an outgoing prompt.

    Reasoning nodes do not catch it on purpose
    """


# --- Role: private registry -----------------------------------------------------------

# Process-wide so the API call site can reach it.
_PRIVATE: set[str] = set()

# Shorter strings would match prompts by chance.
_MIN_PRIVATE_LEN = 4


def register_private(*values: Any) -> None:
    """Register paths as strings no prompt may contain: as given, resolved, and file name.

    Forms shorter than ``_MIN_PRIVATE_LEN`` are skipped.
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
    """Forget every registered string. For tests and long-lived processes."""
    _PRIVATE.clear()


def private_strings() -> frozenset[str]:
    """Return a copy of the registered private strings."""
    return frozenset(_PRIVATE)


# --- Role: text scrubbing -------------------------------------------------------------

_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w\-.\\/]+")
_DATA_FILE = re.compile(
    r"[\w\-.]*[\w\-]\.(?:csv|tsv|parquet|feather|xlsx?|jsonl?|pkl|npy|npz)\b",
    re.IGNORECASE,
)
# Numbers are kept on purpose
_QUOTED = re.compile(r"(['\"])(?:(?!\1).){1,200}\1")
# A line that starts like ``SomeError: ...``.
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt):\s")

REDACTED_PATH = "<path>"
REDACTED_VALUE = "'<redacted>'"


def redact_paths(text: str) -> str:
    """Replace file system paths and data file names with ``<path>``."""
    return _DATA_FILE.sub(REDACTED_PATH, _ABS_PATH.sub(REDACTED_PATH, text))


def scrub_message(text: str, limit: int = 300) -> str:
    """Reduce a log tail to its last exception line, with paths and quotes hidden.

    Quoted literals are where cell values hide; ``""`` for empty text.
    """
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    chosen = next((line for line in reversed(lines) if _EXCEPTION_LINE.match(line)), lines[-1])
    return redact_paths(_QUOTED.sub(REDACTED_VALUE, chosen))[:limit]


def assert_clean(text: str, label: str = "prompt") -> str:
    """Return ``text`` with path-like strings hidden; raise ``RawDataLeak`` on registered ones.

    The two cases differ on purpose
    """
    for secret in _PRIVATE:
        if secret in text:
            raise RawDataLeak(
                f"{label}: 프롬프트에 비공개 데이터 참조가 포함되어 전송을 중단했습니다 "
                f"(길이 {len(secret)}자 문자열이 일치). "
                "dataset_card/result를 privacy.public_* 를 통과시키지 않은 노드가 있습니다."
            )
    return redact_paths(text)


# --- Role: dataset card ---------------------------------------------------------------

# The only private key; removed before the card enters state.
DATA_KEY = "data"

# Keys that could slip example rows into a card.
SAMPLE_KEYS = frozenset(
    {"sample", "samples", "sample_rows", "head", "rows", "examples", "preview", "raw", "raw_rows"}
)


# An allowlist, not a denylist
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
    # Buckets and ratios only
    "target",
    "missing",
    "features",
    # Hand-written free text; limits in ``dataset/caveats.py``.
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
    """_is_scalar | Dataset card: True for a str, int, float, bool, or ``None`` leaf."""
    return isinstance(value, _SCALARS) or value is None


def _is_number(value: Any) -> bool:
    """_is_number | Dataset card: True for a non-bool number, as ``as_number`` decides."""
    return as_number(value) is not None


class CardSchemaError(ValueError):
    """Raised when a card carries something it may not carry."""


def _check_value(value: Any, path: str) -> None:
    """_check_value | Dataset card: every leaf must be a scalar and no key a sample key.

    One rule, not a per-key type table
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
    """Check a card's schema and return it unchanged; raise ``CardSchemaError`` if bad.

    First line only; :func:`public_card` still runs later
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
    """Return the card as reasoning nodes may see it: no ``data`` block, no sample keys."""
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
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Build the private ``data_ref``; arguments win over the card's ``data`` block.

    Returns ``{}`` without a path, meaning no real data
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
    # Table and query pick rows, so they stay private.
    for key, value in (("table", table), ("query", query)):
        resolved = value or declared.get(key)
        if resolved:
            reference[key] = str(resolved)
    return reference


# --- Role: training result ------------------------------------------------------------

# No ``log_tail`` or ``artifacts`` on purpose
PUBLIC_RESULT_FIELDS: tuple[str, ...] = (
    "status",
    # "val" in the loop, "test" for holdout.
    "split",
    "error_type",
    "train_time_sec",
    "wall_time_sec",
    "returncode",
    "dry_run",
    "dropped_hyperparams",
)

# These keys get their values filtered too
APPLIED_HYPERPARAMS_KEY = "applied_hyperparams"
APPLIED_PREPROCESSING_KEY = "applied_preprocessing"
# Row counts held back and cut requests; often absent.
INTERNAL_VALIDATION_KEY = "internal_validation"
# A list of step lines, so it has its own filter.
APPLIED_PIPELINE_KEY = "applied_pipeline"
MAX_PIPELINE_LINE = 500  # longest pipeline line kept, in characters


def _public_params(params: Any) -> dict[str, Any]:
    """_public_params | Training result: keep scalars and flat containers of scalars.

    Guards shape, not data
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
            # One level only; values are weights.
            clean[str(key)] = {str(name): item for name, item in value.items()}
    return clean


def _public_paired(block: Any) -> dict[str, Any]:
    """_public_paired | Training result: rebuild the paired block key by key, or ``{}``.

    Each string field is checked against its own word list
    """
    raw = block if isinstance(block, dict) else {}
    clean: dict[str, Any] = {}
    if raw.get("status") in PAIRED_STATUSES:
        clean["status"] = raw["status"]
    if raw.get("unit") in RESAMPLE_UNITS:
        clean["unit"] = raw["unit"]
    if canonical(str(raw.get("metric") or "")) in METRICS:
        # Keep the caller's alias, not the main name.
        clean["metric"] = raw["metric"]
    if str(raw.get("reason") or "") in PAIRED_REASONS:
        clean["reason"] = raw["reason"]
    for key in (*PAIRED_FIELDS, "baseline_iteration", "resamples"):
        if _is_number(raw.get(key)):
            clean[key] = raw[key]
    if isinstance(raw.get("threads_changed"), bool):
        # The loop above rejects bools
        clean["threads_changed"] = raw["threads_changed"]
    return clean


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Pass a raw ``result.json`` through the allowlist into a new, shareable dict.

    Only numeric metrics are kept; a failed result also gets ``error_summary``.
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
        # The only free-text field that survives.
        clean["error_summary"] = summary
    return clean
