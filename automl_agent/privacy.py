"""The raw-data boundary: nothing that touched a data row reaches a prompt.

Two mechanisms, in order of importance.

1. **By construction.** Raw rows are read only inside the fixed scripts that run as
   subprocesses (``scripts/profile.py``, ``scripts/train.py``). What they hand back
   to the orchestrator is an aggregate card and a metrics dict — never cells. The
   private half of the boundary (the file path and the target column) lives in its
   own state channel, ``data_ref``, which only the execution nodes read. The
   reasoning nodes cannot leak what they never receive.
2. **As a backstop.** :func:`assert_clean` runs on every rendered prompt at the
   single API call site. Registered private material (the dataset path) *aborts the
   run* instead of being sent; anything that merely looks like a filesystem path is
   redacted.

The backstop exists because "no reasoning node reads that channel" is an invariant
a future edit can break silently, and a leak is not a bug you want to learn about
from someone else's logs.

Nothing here imports pandas or opens a data file: this module is the sieve, not a
reader.
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
    """Registered private material was found in an outgoing prompt.

    Deliberately *not* caught by the reasoning nodes (they catch ``LLMUnavailable``,
    ``KeyError`` and ``OSError``): a leak is a defect in this code, so the run stops
    loudly rather than degrading to a fallback that hides it.
    """


# --------------------------------------------------------------------------- #
# Registry of strings that must never appear in a prompt
# --------------------------------------------------------------------------- #

# Process-global on purpose: the guard has to be reachable from the one place that
# talks to the API, which knows nothing about the run's data source.
_PRIVATE: set[str] = set()

# Below this length a "private" string would match half the prompt by accident.
_MIN_PRIVATE_LEN = 4


def register_private(*values: Any) -> None:
    """Register a dataset path (or similar) as forbidden in prompts.

    Each value contributes three forms — as given, resolved absolute, and the bare
    file name — because any of them is enough to identify the source file.
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
    return frozenset(_PRIVATE)


# --------------------------------------------------------------------------- #
# Text scrubbing
# --------------------------------------------------------------------------- #

_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w\-.\\/]+")
_DATA_FILE = re.compile(
    r"[\w\-.]*[\w\-]\.(?:csv|tsv|parquet|feather|xlsx?|jsonl?|pkl|npy|npz)\b",
    re.IGNORECASE,
)
# Data values surface in exception text almost exclusively as quoted literals:
# "could not convert string to float: 'Male'". Numbers are left alone on purpose —
# masking them would destroy the OOM and timing evidence the Critic reasons about,
# and bare numeric cell values in exception messages are rare.
_QUOTED = re.compile(r"(['\"])(?:(?!\1).){1,200}\1")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt):\s")

REDACTED_PATH = "<path>"
REDACTED_VALUE = "'<redacted>'"


def redact_paths(text: str) -> str:
    """Replace filesystem paths and data-file names with a placeholder."""
    return _DATA_FILE.sub(REDACTED_PATH, _ABS_PATH.sub(REDACTED_PATH, text))


def scrub_message(text: str, limit: int = 300) -> str:
    """Reduce a training log tail to one scrubbed diagnostic line.

    Keeps the exception type and message — which is the signal the Critic needs, e.g.
    ``ValueError: Must have at least 1 validation dataset for early stopping`` — and
    drops the traceback, paths and quoted literals, which is where cell values hide.
    """
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    chosen = next((line for line in reversed(lines) if _EXCEPTION_LINE.match(line)), lines[-1])
    return redact_paths(_QUOTED.sub(REDACTED_VALUE, chosen))[:limit]


def assert_clean(text: str, label: str = "prompt") -> str:
    """Return ``text`` fit to send: raises on registered material, redacts paths.

    The asymmetry is deliberate. A registered dataset path in a prompt means our own
    plumbing broke, so it raises. A string that merely *looks* like a path (often
    LLM-authored prose echoing a filename) is redacted, because aborting a run over
    a false positive would be worse than the non-leak it prevents.
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
# Dataset card: public summary vs private data reference
# --------------------------------------------------------------------------- #

# The one private key a card file may carry. Split off before the card enters state.
DATA_KEY = "data"

# Keys a hand-written card might use to smuggle example rows in. Dropped, because a
# card is a summary by definition and the LLM has no use for individual records.
SAMPLE_KEYS = frozenset(
    {"sample", "samples", "sample_rows", "head", "rows", "examples", "preview", "raw", "raw_rows"}
)


# Every top-level key a card may carry — an allowlist, because the failure mode is a key
# nobody thought of. ``{"metadata": {"sample_rows": [...]}}`` went straight through the
# denylist below (which only ever looked at top-level names) and reached the prompt.
CARD_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "task",
    "target_column",
    "n_rows",
    "n_features",
    "n_features_dropped_non_numeric",
    # Counts plus the names of the columns that were dropped — both already disclosed by
    # ``features`` above, which carries every column's name and dtype.
    "encoding",
    "n_informative",
    "n_classes",
    "class_balance",
    "imbalance_ratio",
    # The regression counterpart of the three above: a continuous target has no classes to
    # count, and what a reader needs instead is its scale and shape. Buckets and rates only,
    # by the same rule as ``features`` — no min, no max, no quantile in the target's units,
    # because those are cell values of the column being predicted.
    "target",
    "missing",
    "features",
    # Free text, and the only card field a human writes by hand — see
    # :mod:`automl_agent.dataset.caveats` for why that is in policy and what it is bounded by.
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


class CardSchemaError(ValueError):
    """A card carrying something a card is not allowed to carry."""


def _check_value(value: Any, path: str) -> None:
    """One rule, applied recursively: containers pass through, leaves must be scalars.

    Deliberately not a per-key type table. A card that grows a field should not require
    this function to be edited, and a rule short enough to state in a sentence — "every
    leaf is a number, a string or a boolean" — is one that stays enforced. It is also
    exactly the rule that makes a row impossible: a record needs a structure to live in.
    """
    if value is None or isinstance(value, _SCALARS):
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
    """Fail-closed schema check. Returns the card unchanged, or raises ``CardSchemaError``.

    This is the *first* line: an unknown key stops the run rather than being forwarded,
    which is the opposite of :func:`public_card`'s stance. ``public_card`` stays in place
    behind it as the second line — it drops the private ``data`` block on every path,
    including the ones that skip validation (a resumed run, say).
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
    """The card as the reasoning nodes may see it: no path, no example rows."""
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
    """Build the private ``data_ref`` channel. Explicit arguments win over the card.

    An empty dict means "no real data": the executor then synthesises rows from the
    card's declared shape, which is how the loop stays runnable from a card alone.

    ``group_column`` travels here rather than in the public card for the same reason
    ``path`` does — it decides which rows are held out, so no prompt may see it and no
    plan may change it (:mod:`automl_agent.scoring.splits`). A card profiled with a group column
    already names it in this private block, so a ``--dataset-card`` run inherits it and
    keeps splitting the way the card's baseline was measured.
    """
    declared = dict((card or {}).get(DATA_KEY) or {})
    resolved_path = path or declared.get("path")
    if not resolved_path:
        return {}
    target = (
        target_column
        or declared.get("target_column")
        or (card or {}).get("target_column")
        or "target"
    )
    reference = {"path": str(resolved_path), "target_column": str(target)}
    grouped = group_column or declared.get("group_column")
    if grouped:
        reference["group_column"] = str(grouped)
    return reference


# --------------------------------------------------------------------------- #
# Training result: allowlist, not denylist
# --------------------------------------------------------------------------- #

# Everything the orchestrator keeps in state from a training run. `log_tail` and
# `artifacts` are absent by design: the full log stays on disk under
# artifacts/<thread_id>/train/iter_NN/ for humans, and never enters a state channel
# that a prompt is rendered from.
PUBLIC_RESULT_FIELDS: tuple[str, ...] = (
    "status",
    # Which rows the metrics are about: "val" during the loop, "test" for the single
    # post-loop measurement. ``model_path`` is deliberately absent from this list — a
    # fitted estimator is data-equivalent, so its location stays on the private side — and
    # so is ``schema_path``, which is the other half of that same artifact and lists the
    # category levels and class labels verbatim.
    "split",
    "error_type",
    "train_time_sec",
    "wall_time_sec",
    "returncode",
    "dry_run",
    "dropped_hyperparams",
)

# Carried through separately because their *values* need filtering, not just their keys.
APPLIED_HYPERPARAMS_KEY = "applied_hyperparams"
# {"impute": "median", "scale": true} — the pipeline the executor built, so a write-up can
# state the preprocessing that ran instead of the preprocessing the card suggested.
APPLIED_PREPROCESSING_KEY = "applied_preprocessing"
# {"held_out_rows": 414, "fit_rows": 2346, ...} — the rows the estimator's own early stopping
# kept back. Row *counts*, like the split sizes already in every prompt, not row contents — plus
# ``cut_requested``/``cut_declined``, which are the decision cut's request and the executor's
# reason for refusing it, both out of a closed set in ``scripts/train.py``. The filter is here for
# the same reason it is on the other two, to keep a future key from arriving as an object. Absent
# when nothing was held back and no cut was asked for, which is the common case.
INTERNAL_VALIDATION_KEY = "internal_validation"
# ``["missing_indicator(gcs, paco2)", "impute(median (constant: gcs, paco2))", "scale(auto)"]`` —
# the steps a ``pipeline`` spec really produced, in order, one rendered line each. A list rather
# than a mapping, so it gets its own filter below instead of ``_public_params``.
#
# Every part of every line comes from a closed set: the step name is one of
# :data:`automl_agent.dataset.pipeline.STEPS` and the columns are the fitted schema's, which the
# dataset card already publishes by name. So this is not a new class of thing crossing — but the
# filter is here for the reason the hyperparameter one is, to keep a future step from arriving as
# an object or as a line long enough to be a data dump.
APPLIED_PIPELINE_KEY = "applied_pipeline"
MAX_PIPELINE_LINE = 500


def _public_params(params: Any) -> dict[str, Any]:
    """Hyperparameter values fit to render: scalars, or flat containers of scalars.

    These come from the LLM's own proposal narrowed to sklearn parameter names, so there
    is no data path for a row to arrive by. The filter is here for the opposite case — an
    object or nested structure that has no business being formatted into a prompt.
    ``hidden_layer_sizes`` is why flat sequences are allowed and ``class_weight``'s
    ``{class code: weight}`` map is why flat mappings are: dropping either would hide a
    parameter that really was applied, which is the defect this plumbing exists to fix.
    """
    clean: dict[str, Any] = {}
    for key, value in (params if isinstance(params, dict) else {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(_is_scalar(item) for item in value):
            clean[str(key)] = list(value)
        elif isinstance(value, dict) and all(
            _is_scalar(item) for pair in value.items() for item in pair
        ):
            # One level only: the values are weights, and a nested structure here would be
            # something else entirely.
            clean[str(key)] = {str(name): item for name, item in value.items()}
    return clean


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _public_paired(block: Any) -> dict[str, Any]:
    """The paired-comparison block, key by key, or ``{}``.

    Filtered like ``metrics`` rather than forwarded, and for the same reason: this block is
    written by the training subprocess, so it is on the far side of the boundary, and the
    rule that makes it safe is that every field is a number or one of a fixed set of words.
    So every string field is checked against its own vocabulary rather than against ``str``:
    ``status``, ``unit`` and ``reason`` against the words
    :mod:`automl_agent.scoring.intervals` writes, ``metric`` against the metric registry. A
    string field that admits any string is a field a future edit can turn into a free-text
    detail, and a detail about a failed comparison is where a column name or a cell value
    would arrive.
    """
    raw = block if isinstance(block, dict) else {}
    clean: dict[str, Any] = {}
    if raw.get("status") in PAIRED_STATUSES:
        clean["status"] = raw["status"]
    if raw.get("unit") in RESAMPLE_UNITS:
        clean["unit"] = raw["unit"]
    if canonical(str(raw.get("metric") or "")) in METRICS:
        # The alias the caller wrote, not the canonical name: this is a passthrough, and the
        # ledger prints this field beside the same metric name the goal states.
        clean["metric"] = raw["metric"]
    if str(raw.get("reason") or "") in PAIRED_REASONS:
        clean["reason"] = raw["reason"]
    for key in (*PAIRED_FIELDS, "baseline_iteration", "resamples"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[key] = value
    if isinstance(raw.get("threads_changed"), bool):
        # The one boolean here, so it needs its own line: the loop above rejects ``bool``
        # because ``True`` where a delta belongs would render as a score. What this field says
        # is that the two prediction vectors were produced in different thread states, which is
        # what stops a reader from reading the delta as the plan's doing
        # (:mod:`automl_agent.threads`). The blocks it was derived from stay private — not
        # because they are data, but because they are free-form strings out of the
        # environment, and this block's rule is that every string in it is checked against a
        # closed set. A boolean has no free-text room to grow into.
        clean["threads_changed"] = raw["threads_changed"]
    return clean


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Allowlist a raw ``result.json`` payload into its shareable form.

    Metrics are kept only if numeric — a metric slot is where a stray string from a
    future model wrapper would otherwise ride along.
    """
    raw = dict(result or {})
    metrics = raw.get("metrics")
    clean: dict[str, Any] = {
        "metrics": {
            key: value
            for key, value in (metrics if isinstance(metrics, dict) else {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
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
        # The one text field that survives: the exception line, scrubbed.
        clean["error_summary"] = summary
    return clean
