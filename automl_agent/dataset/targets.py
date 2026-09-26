"""Target column handling shared by profile.py and train.py.

Roles:

* Task detection — classification or regression, read from the column.
* Target encoding — class codes or floats, applying the missing-label policy.
* Class labels — what each class code means, for labelling outputs.
"""

from __future__ import annotations

from typing import Any

from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION

POLICY_REJECT = "reject"
POLICY_DROP = "drop"
TARGET_MISSING_POLICIES: tuple[str, ...] = (POLICY_REJECT, POLICY_DROP)
DEFAULT_TARGET_MISSING_POLICY = POLICY_REJECT

# Above this, an integer target is regression
CLASSIFICATION_MAX_DISTINCT = 20


class TargetMissingError(ValueError):
    """Raised when the target has missing labels and the policy is ``reject``."""


class TargetUnusableError(ValueError):
    """Raised when the target has fewer than two distinct labels."""


# --- Role: task detection ---------------------------------------------------------


def detect_task(series: Any) -> str:
    """Decide classification or regression from the labelled rows; rule order matters.

    Raises TargetUnusableError with fewer than two distinct labels."""
    import pandas as pd

    present = series.dropna()
    distinct = int(present.nunique())
    if distinct < 2:
        raise TargetUnusableError(
            f"target column {str(series.name)!r} has {distinct} distinct value(s) in "
            f"{int(len(present))} labelled rows, so there is nothing to predict. Check "
            "that the right column was named, and that the rows were not filtered down "
            "to a single outcome."
        )
    if pd.api.types.is_bool_dtype(present) or not pd.api.types.is_numeric_dtype(present):
        return TASK_CLASSIFICATION
    if distinct == 2:
        return TASK_CLASSIFICATION
    if pd.api.types.is_float_dtype(present) and not bool((present % 1 == 0).all()):
        return TASK_REGRESSION
    return TASK_CLASSIFICATION if distinct <= CLASSIFICATION_MAX_DISTINCT else TASK_REGRESSION


# --- Role: target encoding --------------------------------------------------------


def encode_target(
    series: Any, policy: str = DEFAULT_TARGET_MISSING_POLICY, task: str | None = None
) -> tuple[Any, Any, int]:
    """Return ``(values, keep, n_missing)``; ``keep`` masks the kept rows.

    Codes are never ``-1``; regression stays unscaled. Raises TargetMissingError unless ``drop``."""
    n_rows = int(len(series))
    n_missing = int(series.isna().sum())
    if n_missing and policy != POLICY_DROP:
        raise TargetMissingError(
            f"target column {str(series.name)!r} has {n_missing} missing values out of "
            f"{n_rows} rows. Pass --on-missing-target drop to train on the remaining "
            f"{n_rows - n_missing} rows, or fix the labels."
        )
    keep = series.notna()
    present = series[keep] if n_missing else series
    if (task or detect_task(series)) == TASK_REGRESSION:
        return present.astype("float64"), keep, n_missing
    codes = present.astype("category").cat.codes
    return codes, keep, n_missing


# --- Role: class labels -----------------------------------------------------------


def target_classes(series: Any, task: str | None = None) -> list[Any] | None:
    """List the label for each class code, or ``None`` for regression.

    One-way JSON conversion: label outputs with it, never re-encode inputs."""
    import pandas as pd

    present = series.dropna()
    if (task or detect_task(series)) == TASK_REGRESSION:
        return None
    # Must match encode_target's codes (pinned by a test).
    categories = pd.Series(present).astype("category").cat.categories
    return [_json_scalar(value) for value in categories.tolist()]


def _json_scalar(value: Any) -> Any:
    """_json_scalar | Class labels: make one label JSON-safe."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)
