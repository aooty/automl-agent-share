"""Score intervals: how much of a score is model, how much is rows.

Roles:

* Interval settings — fixed level, resample count, minimum sizes.
* Single-score interval — bootstrap interval for one score.
* Paired difference — bootstrap interval for two trials, same rows.
* Reading back — read published intervals, describe them in Korean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .metrics import MINIMIZE

# --- Role: interval settings ------------------------------------------------------

# Not configurable, so runs stay comparable.
CI_LEVEL = 0.95

# About 10 per tail; ``resamples=0`` turns intervals off.
DEFAULT_RESAMPLES = 400

# Counted in rows, or groups on a grouped split.
MIN_UNITS = 20

# Below this defined share, no interval
MIN_VALID_SHARE = 0.8

# Closed set checked by ``privacy.public_result``.
UNIT_ROW = "row"
UNIT_GROUP = "group"
RESAMPLE_UNITS: tuple[str, ...] = (UNIT_ROW, UNIT_GROUP)


# --- Role: single-score interval --------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval, and what was resampled to get it."""

    low: float
    high: float
    resamples: int
    # Kept for the logs, not published as a metric.
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        """bounds | Role: ``(low, high)``."""
        return self.low, self.high

    @property
    def width(self) -> float:
        """width | Role: ``high - low``, rounded to 6 decimals."""
        return round(self.high - self.low, 6)

    def flatten(self, metric: str) -> dict[str, float]:
        """flatten | Role: two floats, since ``public_result`` keeps only numbers."""
        return {f"{metric}_ci_low": self.low, f"{metric}_ci_high": self.high}


def _sample_units(y_true: Any, groups: Any, resamples: int) -> tuple[Any, list[Any] | None, int, str] | None:
    """_sample_units | Role: prepare what to resample, or ``None`` if too little."""
    import numpy as np

    # Order matters: these checks run before ``_units`` raises.
    if resamples <= 0:
        return None
    y_arr = np.asarray(y_true)
    n_rows = int(len(y_arr))
    if n_rows == 0:
        return None
    rows_by_unit, unit = _units(groups, n_rows)
    n_units = n_rows if rows_by_unit is None else len(rows_by_unit)
    if n_units < MIN_UNITS:
        return None
    return y_arr, rows_by_unit, n_units, unit


def _resample(
    measure: Callable[[Any], float | None],
    rows_by_unit: list[Any] | None,
    n_units: int,
    seed: int,
    resamples: int,
) -> list[float] | None:
    """_resample | Role: run ``measure`` per resample; ``None`` if too few valid."""
    import numpy as np

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n_units, n_units)
        index = picks if rows_by_unit is None else np.concatenate([rows_by_unit[p] for p in picks])
        # Undefined resample is skipped, not a failure.
        try:
            value = measure(index)
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))
    if len(values) < MIN_VALID_SHARE * resamples:
        return None
    return values


def _percentiles(values: list[float]) -> tuple[float, float]:
    """_percentiles | Role: the two tail percentiles for :data:`CI_LEVEL`."""
    import numpy as np

    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = (float(x) for x in np.percentile(values, [tail, 100.0 - tail]))
    return low, high


def bootstrap_interval(
    score: Callable[[Any, Any, Any], float | None],
    y_true: Any,
    pred: Any,
    proba: Any = None,
    *,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> Interval | None:
    """bootstrap_interval | Role: percentile :class:`Interval` for a score, or ``None``.

    Raises ``ValueError`` when ``groups`` length differs from the rows.
    """
    import numpy as np

    prepared = _sample_units(y_true, groups, resamples)
    if prepared is None:
        return None
    y_arr, rows_by_unit, n_units, unit = prepared

    pred_arr = np.asarray(pred)
    proba_arr = None if proba is None else np.asarray(proba)

    def measure(index: Any) -> float | None:
        return score(
            y_arr[index],
            pred_arr[index],
            None if proba_arr is None else proba_arr[index],
        )

    values = _resample(measure, rows_by_unit, n_units, seed, resamples)
    if values is None:
        return None
    low, high = _percentiles(values)
    return Interval(low=round(low, 6), high=round(high, 6), resamples=len(values), unit=unit)


def _units(groups: Any, n_rows: int) -> tuple[list[Any] | None, str]:
    """_units | Role: row indices per group (``None`` for rows), and the unit."""
    import numpy as np

    if groups is None:
        return None, UNIT_ROW
    group_arr = np.asarray(groups)
    if len(group_arr) != n_rows:
        raise ValueError(
            f"group 배열의 길이가 행 수와 다릅니다 — groups={len(group_arr)}, rows={n_rows}"
        )
    _, inverse = np.unique(group_arr, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return [np.flatnonzero(inverse == code) for code in range(int(inverse.max()) + 1)], UNIT_GROUP


# --- Role: paired difference ------------------------------------------------------

# Own block, not in ``metrics``, so not read as score.
PAIRED_KEY = "paired"

# Four floats: ``public_result`` keeps numbers, drops structure.
PAIRED_FIELDS: tuple[str, ...] = ("delta_vs_best", "delta_ci_low", "delta_ci_high", "p_better")

PAIRED_MEASURED = "measured"
PAIRED_SKIPPED = "skipped"
# Closed set, same reason as :data:`RESAMPLE_UNITS`.
PAIRED_STATUSES: tuple[str, ...] = (PAIRED_MEASURED, PAIRED_SKIPPED)

# Published so silence is not read as "found nothing".
PAIRED_REASONS: dict[str, str] = {
    "no_baseline": "짝지을 직전 최고가 없습니다 — 첫 측정입니다",
    "baseline_missing": "직전 최고의 행별 예측 파일이 없습니다",
    "split_changed": "이 시도의 val 행이 직전 최고의 val 행과 다릅니다 — 짝지을 수 없습니다",
    "degenerate": "차이를 재표집할 수 없었습니다 — 행이 너무 적거나 지표가 축퇴했습니다",
}


@dataclass(frozen=True)
class PairedDelta:
    """The resampled *difference* between two trials scored on the same rows."""

    # Always candidate minus baseline; direction affects only p_better.
    delta: float
    low: float
    high: float
    # Share where candidate was better; ties count as not.
    p_better: float
    resamples: int
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        """bounds | Role: ``(low, high)``."""
        return self.low, self.high

    @property
    def resolved(self) -> bool:
        """resolved | Role: True when the interval leaves out 0; says nothing about cause."""
        return not (self.low <= 0.0 <= self.high)

    def flatten(self) -> dict[str, float]:
        """flatten | Role: the four published numbers, keyed by :data:`PAIRED_FIELDS`."""
        return dict(
            zip(PAIRED_FIELDS, (self.delta, self.low, self.high, self.p_better), strict=True)
        )


def paired_delta(
    score: Callable[[Any, Any, Any], float | None],
    y_true: Any,
    candidate: tuple[Any, Any],
    baseline: tuple[Any, Any],
    *,
    direction: str,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> PairedDelta | None:
    """paired_delta | Role: resample the difference of two trials on the same draws.

    ``None`` like :func:`bootstrap_interval`; ``ValueError`` on length mismatch.
    """
    import numpy as np

    prepared = _sample_units(y_true, groups, resamples)
    if prepared is None:
        return None
    y_arr, rows_by_unit, n_units, unit = prepared
    n_rows = int(len(y_arr))

    sides = [_side(part, n_rows) for part in (candidate, baseline)]
    (a_pred, a_proba), (b_pred, b_proba) = sides

    def difference(index: Any) -> float | None:
        y_slice = y_arr[index]
        a = score(y_slice, a_pred[index], None if a_proba is None else a_proba[index])
        b = score(y_slice, b_pred[index], None if b_proba is None else b_proba[index])
        if a is None or b is None:
            return None
        return float(a) - float(b)

    try:
        observed = difference(np.arange(n_rows))
    except (ValueError, IndexError, ZeroDivisionError):
        observed = None
    if observed is None or not np.isfinite(observed):
        return None

    values = _resample(difference, rows_by_unit, n_units, seed, resamples)
    if values is None:
        return None
    deltas = np.asarray(values)
    low, high = _percentiles(values)
    better = deltas < 0.0 if direction == MINIMIZE else deltas > 0.0
    return PairedDelta(
        delta=round(float(observed), 6),
        low=round(low, 6),
        high=round(high, 6),
        p_better=round(float(better.mean()), 4),
        resamples=len(values),
        unit=unit,
    )


def _side(part: tuple[Any, Any], n_rows: int) -> tuple[Any, Any]:
    """_side | Role: one trial's ``(pred, proba)`` as arrays; raise on bad length."""
    import numpy as np

    pred, proba = part
    pred_arr = np.asarray(pred)
    if len(pred_arr) != n_rows:
        raise ValueError(
            f"짝지을 예측의 길이가 행 수와 다릅니다 — pred={len(pred_arr)}, rows={n_rows}"
        )
    if proba is None:
        return pred_arr, None
    proba_arr = np.asarray(proba)
    if len(proba_arr) != n_rows:
        raise ValueError(
            f"짝지을 확률의 길이가 행 수와 다릅니다 — proba={len(proba_arr)}, rows={n_rows}"
        )
    return pred_arr, proba_arr


# --- Role: reading back -----------------------------------------------------------


def interval_of(metrics: Mapping[str, Any] | None, metric: str) -> tuple[float, float] | None:
    """interval_of | Role: read back :meth:`Interval.flatten`; ``None`` if no valid interval."""
    if not metrics:
        return None
    low = as_number(metrics.get(f"{metric}_ci_low"))
    high = as_number(metrics.get(f"{metric}_ci_high"))
    if low is None or high is None or high < low:
        return None
    return low, high


def contains(bounds: tuple[float, float] | None, value: Any) -> bool:
    """contains | Role: is ``value`` inside ``bounds``; not a test of equality."""
    number = as_number(value)
    if bounds is None or number is None:
        return False
    return bounds[0] <= number <= bounds[1]


def describe_interval(metric: str, score: Any, bounds: tuple[float, float] | None) -> str:
    """describe_interval | Role: text like ``f1=0.7412 (95% CI 0.7108~0.7702, 폭 0.0594)``.

    Returns "" when there is nothing to say.
    """
    value = as_number(score)
    if bounds is None:
        return f"{metric}={value:.4f}" if value is not None else ""
    low, high = bounds
    head = f"{metric}={value:.4f} " if value is not None else f"{metric} "
    return f"{head}({int(CI_LEVEL * 100)}% CI {low:.4f}~{high:.4f}, 폭 {high - low:.4f})"


def paired_of(
    result: Mapping[str, Any] | None, *, baseline_iteration: Any
) -> dict[str, Any] | None:
    """paired_of | Role: the paired block, only if against ``baseline_iteration``.

    ``None`` when skipped, against another iteration, or incomplete.
    """
    block = (result or {}).get(PAIRED_KEY)
    if not isinstance(block, Mapping) or block.get("status") != PAIRED_MEASURED:
        return None
    if as_iteration(block.get("baseline_iteration")) != as_iteration(baseline_iteration):
        return None
    values = {field: as_number(block.get(field)) for field in PAIRED_FIELDS}
    if any(value is None for value in values.values()):
        return None
    return {**dict(block), **values}


def describe_paired(block: Mapping[str, Any] | None) -> str:
    """describe_paired | Role: Korean ledger piece; says only if Δ differs from 0."""
    if not isinstance(block, Mapping):
        return ""
    if block.get("status") == PAIRED_SKIPPED:
        reason = PAIRED_REASONS.get(str(block.get("reason") or ""))
        return f"[짝지은 검정 없음: {reason}]" if reason else ""
    delta, low, high, p_better = (as_number(block.get(field)) for field in PAIRED_FIELDS)
    if delta is None or low is None or high is None or p_better is None:
        return ""
    verdict = "0과 구분됨" if not low <= 0.0 <= high else "이 행들로는 0과 구분되지 않음"
    against = block.get("baseline_iteration")
    head = f"짝지은 Δ(iteration {against} 대비)" if against is not None else "짝지은 Δ"
    text = (
        f"[{head} {delta:+.4f}, {int(CI_LEVEL * 100)}% CI {low:+.4f}~{high:+.4f}, "
        f"P(개선) {p_better:.3f} — {verdict}]"
    )
    if block.get("threads_changed") is True:
        # A note, not the verdict: threads alone move scores.
        text += " [baseline과 스레드 상태가 다릅니다 — 이 Δ에는 환경 차이가 섞여 있습니다]"
    return text


def as_iteration(value: Any) -> int | None:
    """as_iteration | Role: iteration as ``int``, or ``None``; ``bool`` does not count."""
    number = as_number(value)
    return None if number is None else int(number)


def resolution_note(
    metrics: Mapping[str, Any] | None,
    metric: str,
    *,
    others: Mapping[str, Any] | None = None,
) -> str:
    """resolution_note | Role: Korean prompt paragraph on what the slice can tell apart.

    Names each of ``others`` inside the interval; gives no advice.
    """
    bounds = interval_of(metrics, metric)
    if bounds is None:
        return (
            f"{metric}의 신뢰구간이 이 시도에는 없습니다 — 슬라이스가 너무 작거나 지표가 "
            "축퇴했습니다. 차이의 크기를 노이즈와 견줄 근거가 없으니 작은 변화를 "
            "개선이나 악화로 읽지 마십시오."
        )
    line = describe_interval(metric, (metrics or {}).get(metric), bounds)
    collisions = [
        f"{label}({as_number(value):.4f})"
        for label, value in (others or {}).items()
        if contains(bounds, value)
    ]
    if not collisions:
        return (
            f"{line}. 비교 대상 중 이 구간 안에 들어오는 값은 없으므로, 위 점수와의 차이는 "
            "이 슬라이스가 실제로 구분해 낸 차이입니다."
        )
    return (
        f"{line}. 다음 값들이 이 구간 안에 있습니다: {', '.join(collisions)} — 이 행들로는 "
        "위 점수와 구분되지 않는 값들이므로, 그 차이를 원인이 있는 변화로 진단하지 "
        "마십시오. 구간 밖의 차이만 증거로 쓸 수 있습니다."
    )


def as_number(value: Any) -> float | None:
    """as_number | Role: int or float as ``float``, else ``None``; ``bool`` does not count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
