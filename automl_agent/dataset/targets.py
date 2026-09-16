"""Target-column encoding, what task it implies, and what to do about missing labels.

**Shared by the two fixed scripts because they must agree** — a row the profiler counted and the
trainer dropped makes the card describe a dataset that was never trained on, and the baseline stops
being comparable to what it is compared against.

**A bare ``astype("category").cat.codes`` maps NaN to ``-1``**, which pandas is right to do and nothing
downstream knew: one missing label turned a binary target into ``n_classes = 3``, with a phantom
stratum in the split and every per-class metric averaged over it. So the two cases are named:

``reject``  the default — stop and say how many are missing. A missing label is usually a
            data-preparation bug, and training on a guess about it is the expensive kind of wrong.
``drop``    drop them and record how many, applied identically by both scripts.

**The task comes from this column too, and is not the caller's to choose**: a continuous target
category-coded into class labels trains a classifier on thousands of one-row "classes", and every
score it reports is about an encoding accident. :func:`detect_task` decides, both scripts call it, and
the answer is on the card so the goal, the metric and the registry check against one claim.

pandas is imported inside the functions, so the process that renders prompts never loads it.
"""

from __future__ import annotations

from typing import Any

from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION

POLICY_REJECT = "reject"
POLICY_DROP = "drop"
TARGET_MISSING_POLICIES: tuple[str, ...] = (POLICY_REJECT, POLICY_DROP)
DEFAULT_TARGET_MISSING_POLICY = POLICY_REJECT

# How many distinct values an integer-valued target may hold and still be read as classes.
# Above it, a column of whole numbers is a count or an age, not a label set: 40 classes on
# a 3000-row file leaves ~75 rows each, which no stratified split survives and no macro
# average means anything over. Below it, an ordinal grade (1..5) stays classification,
# which is the reading a human would give it too.
CLASSIFICATION_MAX_DISTINCT = 20


class TargetMissingError(ValueError):
    """The target column has missing labels and the policy is ``reject``."""


class TargetUnusableError(ValueError):
    """The target column cannot be a target at all — one value, or no values."""


def detect_task(series: Any) -> str:
    """Whether ``series`` is a classification target or a regression one.

    The rules, in order, over the rows that *have* a label:

    * a non-numeric column (strings, categories) is classification — there is nothing
      else it could be;
    * booleans are classification, before the numeric rules see them as 0/1 ints;
    * two distinct values are classification whatever the dtype, so a target stored as
      ``0.0``/``1.0`` floats is not read as a regression on the unit interval;
    * a float column holding a non-integral value is regression — ``2.5`` is not a class
      code, and this is the check that catches a continuous target with few distinct
      values (a 3-valued rounded score stays classification, which is honest: nothing in
      the column says otherwise);
    * otherwise it is the count that decides, at :data:`CLASSIFICATION_MAX_DISTINCT`.

    Raises :class:`TargetUnusableError` for a column with fewer than two distinct labels:
    a constant target has no signal to learn and every metric on it is degenerate, so it
    is a refusal rather than a task.
    """
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


def encode_target(
    series: Any, policy: str = DEFAULT_TARGET_MISSING_POLICY, task: str | None = None
) -> tuple[Any, Any, int]:
    """Encode a target column, without inventing a class for NaN or for a real number.

    Returns ``(values, keep, n_missing)``:

    ``values``     for classification, integer class codes for the rows that have a label
                   — never ``-1``. For regression, those rows' numbers as ``float64``,
                   untouched: no scaling, no binning, so a score in the target's units
                   (``mae``) is in the units the file uses.
    ``keep``       boolean mask over the original rows, for filtering the feature frame
                   to exactly the rows ``values`` covers.
    ``n_missing``  how many labels were missing (0 whenever the policy is ``reject``,
                   since a non-zero count raises).

    ``task`` is detected from the column when not given. Callers that have already
    detected it (both scripts do, to put it on the card) pass it in rather than paying for
    a second ``nunique`` over the same column.
    """
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


def target_classes(series: Any, task: str | None = None) -> list[Any] | None:
    """What each class code from :func:`encode_target` means. ``None`` for a regression target.

    Index-aligned with those codes by construction (both go through ``astype("category")``, whose
    categories are the sorted distinct labels) and bound to them by
    ``test_the_class_labels_line_up_with_the_codes``. It exists because a prediction of ``1`` is not
    an answer: the caller asked about a column of ``died``/``survived``, and only this list turns the
    code back into their word.

    Values normalised to JSON scalars, since this is written to disk and read by another process.
    One-way for an exotic label type (a Timestamp becomes its string form) — hence for *labelling
    output*, never for re-encoding input.
    """
    import pandas as pd

    present = series.dropna()
    if (task or detect_task(series)) == TASK_REGRESSION:
        return None
    categories = pd.Series(present).astype("category").cat.categories
    return [_json_scalar(value) for value in categories.tolist()]


def _json_scalar(value: Any) -> Any:
    """A label as something ``json.dumps`` accepts, preferring not to change what it says."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)
