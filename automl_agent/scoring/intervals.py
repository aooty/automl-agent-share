"""How much of a score is the model, and how much is the rows it was measured on.

Every number here is a point estimate over one slice, and every comparison is between two of them.
So each scored split publishes a 95% percentile bootstrap interval on the goal metric, and the
consumers that compare say whether their difference fits inside it.

**Two halves, for two different decisions.**

*Marginal* (:func:`bootstrap_interval`) — the width of one score. **Overlapping intervals mean
"these rows cannot tell these models apart", not "the models are the same".** And it is sampling
noise, **not** the selection bias of taking the best of five: that error points one way, and the
held-back test split is what measures it (:mod:`automl_agent.nodes.holdout`).

*Paired* (:func:`paired_delta`) — resamples the difference itself, because most of a marginal width
is row noise common to both attempts and cancels. Strictly narrower question: "did *this change* move
the score on these rows". **It does not fix confounding** — two attempts that each moved two levers
have a well-resolved difference and still no attribution.

**Grouped splits resample whole groups.** A row-level bootstrap over five visits per patient reports
an interval for ``5 * n_patients`` samples when the data has ``n_patients``, too narrow by a factor
that grows with cluster size. ``unit`` says which was done.

**The paired half is a steering instrument, tuned for sensitivity; the test split is the acceptance
test.** Collapsing the two would make every steering decision as expensive as an acceptance. The
price of that sensitivity is multiplicity, and **nothing here corrects for it** — the ledger reports
each comparison on its own terms.

Rationale: ``docs/rationale.md``.

Train-split scores deliberately get no interval. An in-sample score's uncertainty is not
what its width would describe, and the Critic reads the train number for the gap against
validation, which is about capacity rather than about noise.

Nothing here imports numpy or sklearn at module level: the orchestrator process imports
this module to *read* a published interval, and only the fixed scripts measure one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .metrics import MINIMIZE

# Not configurable. Two runs whose intervals were computed at different levels would print
# numbers that look comparable and are not, and there is no question here that a 90% or a
# 99% interval answers better.
CI_LEVEL = 0.95

# Enough that each 2.5% tail is estimated from ~10 resamples rather than from 1, and cheap
# enough to leave switched on unconditionally: one metric over 400 resamples of a 10,000-row
# split costs well under a second against a fit that costs tens of them. A caller that has
# to have it cheaper passes ``resamples=0`` and gets no interval rather than a bad one.
DEFAULT_RESAMPLES = 400

# Below this there is nothing to resample. An interval from 15 rows would be wide, correct
# and useless; a *grouped* interval from 4 patients is drawn from four distinct values, so
# its percentiles are those values. Counted in resampling units — rows, or groups when the
# split was grouped — because that is what the sample size actually is.
MIN_UNITS = 20

# A metric undefined in more than a fifth of the resamples has no interval worth
# publishing: a single-class resample of a 3%-positive split makes ``roc_auc`` undefined,
# and quietly keeping the resamples that happened to work would report an interval
# conditioned on being lucky.
MIN_VALID_SHARE = 0.8

# What can be resampled. Named rather than written inline at the two return sites because
# :func:`automl_agent.privacy.public_result` checks this field against the closed set instead of
# forwarding any string, and a set the writer and the sieve state separately can drift.
UNIT_ROW = "row"
UNIT_GROUP = "group"
RESAMPLE_UNITS: tuple[str, ...] = (UNIT_ROW, UNIT_GROUP)


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval and what was resampled to get it."""

    low: float
    high: float
    resamples: int
    # One of :data:`RESAMPLE_UNITS`. Not published as a metric — it is already implied by the protocol's
    # ``grouped_by`` — but logged, because an interval measured the wrong way is narrow
    # rather than absent and nothing else in the record would say so.
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    @property
    def width(self) -> float:
        return round(self.high - self.low, 6)

    def flatten(self, metric: str) -> dict[str, float]:
        """``{"<metric>_ci_low": ..., "<metric>_ci_high": ...}``.

        Two floats rather than one pair because :func:`automl_agent.privacy.public_result`
        keeps only numeric metric values — a tuple would be dropped on the way to the state
        channel the report and the Critic read, which is the one place the interval has to
        arrive.
        """
        return {f"{metric}_ci_low": self.low, f"{metric}_ci_high": self.high}


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
    """Resample the scored split ``resamples`` times and return the percentile interval.

    ``score`` is a callable over ``(y_true, pred, proba)``, so this module knows nothing about which
    metric or which library — the two fixed scripts pass their own scorer and stay the only importers
    of sklearn. A degenerate resample may return ``None`` or raise ``ValueError``; both are "undefined
    here", not a failed run.

    **``None``, never an exception**, when there is no interval worth publishing (too few units,
    ``resamples`` at 0, a metric undefined too often). Callers read that as "not measured", so a
    degenerate split costs a disclosure rather than an attempt.

    **Deterministic in ``seed``** — which is what lets two iterations' intervals be compared at all.
    """
    import numpy as np

    if resamples <= 0:
        return None
    y_arr = np.asarray(y_true)
    n_rows = int(len(y_arr))
    if n_rows == 0:
        return None

    # Raises on a misaligned group array rather than zipping to the shorter of the two:
    # resampling the wrong groups yields a narrow interval, which reads as a confident
    # measurement rather than as a bug — the same argument ``split_three_way`` makes.
    rows_by_unit, unit = _units(groups, n_rows)
    n_units = n_rows if rows_by_unit is None else len(rows_by_unit)
    if n_units < MIN_UNITS:
        return None

    pred_arr = np.asarray(pred)
    proba_arr = None if proba is None else np.asarray(proba)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n_units, n_units)
        index = picks if rows_by_unit is None else np.concatenate([rows_by_unit[p] for p in picks])
        try:
            value = score(
                y_arr[index],
                pred_arr[index],
                None if proba_arr is None else proba_arr[index],
            )
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))

    if len(values) < MIN_VALID_SHARE * resamples:
        return None
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = (float(x) for x in np.percentile(values, [tail, 100.0 - tail]))
    return Interval(low=round(low, 6), high=round(high, 6), resamples=len(values), unit=unit)


def _units(groups: Any, n_rows: int) -> tuple[list[Any] | None, str]:
    """``(row indices per resampling unit, unit name)``.

    ``None`` for the row-level path, where the unit *is* the row and building a list of
    1-element index arrays would only cost memory. Otherwise one index array per distinct
    group, which is what makes a resample draw whole patients.
    """
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


# --------------------------------------------------------------------------- #
# The paired half: resampling a difference instead of two scores
# --------------------------------------------------------------------------- #

# Where a paired comparison lands in ``result.json``. A block of its own rather than four
# more keys in ``metrics``, because these are properties of a *comparison* and not of the
# model on these rows: ``p_better`` sitting beside ``roc_auc`` in the metrics dict would ride
# into the report's metric table and read as a score. The precedent for a diagnostic living
# in ``metrics`` — ``balanced_accuracy_cut_headroom`` — is a single-model measurement, which this is not.
PAIRED_KEY = "paired"

# The four numbers, named once so the writer and every reader cannot disagree. Same argument
# as :meth:`Interval.flatten`: four floats rather than a pair plus a scalar, because
# :func:`automl_agent.privacy.public_result` keeps numbers and drops structures.
PAIRED_FIELDS: tuple[str, ...] = ("delta_vs_best", "delta_ci_low", "delta_ci_high", "p_better")

PAIRED_MEASURED = "measured"
PAIRED_SKIPPED = "skipped"
# Same argument as :data:`RESAMPLE_UNITS`: the sieve has to check ``status`` against the words
# this module writes, not against ``str``.
PAIRED_STATUSES: tuple[str, ...] = (PAIRED_MEASURED, PAIRED_SKIPPED)

# Why a comparison was not made. Published rather than left absent, because "no paired
# verdict" and "the paired verdict found nothing" are different facts and a reader with only
# the first would take silence for the second.
PAIRED_REASONS: dict[str, str] = {
    "no_baseline": "짝지을 직전 최고가 없습니다 — 첫 측정입니다",
    "baseline_missing": "직전 최고의 행별 예측 파일이 없습니다",
    "split_changed": "이 시도의 val 행이 직전 최고의 val 행과 다릅니다 — 짝지을 수 없습니다",
    "degenerate": "차이를 재표집할 수 없었습니다 — 행이 너무 적거나 지표가 축퇴했습니다",
}


@dataclass(frozen=True)
class PairedDelta:
    """A resampled *difference* between two attempts scored on the same rows."""

    # Always candidate minus baseline, whichever way the metric goes. A signed difference
    # that flipped with the metric's direction would make the ledger's "직전 최고 대비
    # -0.0500" and this number disagree in sign on ``rmse`` — the direction is applied to
    # ``p_better`` alone, where it belongs.
    delta: float
    low: float
    high: float
    # Share of resamples in which the candidate was *better*, so it reads the same way on a
    # minimize metric. Ties count as not better: two attempts with bit-identical predictions
    # get 0.0, which is the honest reading of "no resample showed an improvement".
    p_better: float
    resamples: int
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    @property
    def resolved(self) -> bool:
        """Whether the interval of the difference excludes zero.

        Narrow on purpose, exactly like :func:`contains`: excluding zero means these rows
        did separate the two attempts. It says nothing about *what* separated them — a
        transition that moved two levers has a well-resolved difference and no attribution.
        """
        return not (self.low <= 0.0 <= self.high)

    def flatten(self) -> dict[str, float]:
        """The four published numbers, keyed by :data:`PAIRED_FIELDS`."""
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
    """Resample the *difference* between two predictions of the same rows.

    **Each resample scores both on the identical draw** — the whole point, and why this cannot be
    assembled out of two :func:`bootstrap_interval` calls.

    **``direction`` is required, never defaulted**: it decides which sign of ``delta`` counts toward
    ``p_better``, and a wrong default reports an ``rmse`` regression as a 0.99 probability of
    improvement.

    ``None`` on the same three conditions :func:`bootstrap_interval` uses. **A *misaligned* baseline
    does raise** — predictions of a different row count are not this split's, and pairing them gives a
    narrow interval around a meaningless number.
    """
    import numpy as np

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

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n_units, n_units)
        index = picks if rows_by_unit is None else np.concatenate([rows_by_unit[p] for p in picks])
        try:
            value = difference(index)
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))

    if len(values) < MIN_VALID_SHARE * resamples:
        return None
    deltas = np.asarray(values)
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = (float(x) for x in np.percentile(deltas, [tail, 100.0 - tail]))
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
    """One attempt's ``(pred, proba)`` as arrays, checked against the row count.

    Raises rather than truncating, for the reason :func:`_units` does: a length mismatch
    means these are not the rows the caller thinks they are, and the cost of guessing is a
    confident number about the wrong comparison.
    """
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


# --------------------------------------------------------------------------- #
# Reading one back
# --------------------------------------------------------------------------- #


def interval_of(metrics: Mapping[str, Any] | None, metric: str) -> tuple[float, float] | None:
    """The published bounds for ``metric``, or ``None`` if this result has none.

    The reader half of :meth:`Interval.flatten`. Every consumer goes through it rather than
    spelling the two key names out, because a result written before intervals existed —
    or one whose split was too small for one — has to read as "not measured" everywhere.
    """
    if not metrics:
        return None
    low = as_number(metrics.get(f"{metric}_ci_low"))
    high = as_number(metrics.get(f"{metric}_ci_high"))
    if low is None or high is None or high < low:
        return None
    return low, high


def contains(bounds: tuple[float, float] | None, value: Any) -> bool:
    """Whether ``value`` falls inside the interval. ``False`` when either is missing.

    The one comparison the consumers make, and its meaning is narrow on purpose: a value
    inside the interval is one *these rows cannot distinguish* from the measured score. It
    is not a hypothesis test, and it does not say the two numbers are equal.
    """
    number = as_number(value)
    if bounds is None or number is None:
        return False
    return bounds[0] <= number <= bounds[1]


def describe_interval(metric: str, score: Any, bounds: tuple[float, float] | None) -> str:
    """One Korean fragment — ``f1=0.7412 (95% CI 0.7108~0.7702, 폭 0.0594)``.

    Empty string when there is nothing to say, so a caller can append it unconditionally.
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
    """This result's paired comparison — but only if it is the pair the caller is reporting.

    ``baseline_iteration`` is not a filter, it is the invariant. The published fields are named
    ``delta_vs_best`` and ``p_better``, so both must be about the *same* baseline as the sentence they
    render into — subtracting against iteration 1 while printing a P computed against iteration 4 gives
    a line where every number is real and the claim is not. Only here are both the block and the
    caller's baseline in hand, so the check is required rather than optional, and a mismatch reads
    "not measured" instead of being quietly rendered.

    ``None`` also for a skipped comparison, and for a block missing any of :data:`PAIRED_FIELDS` —
    what a result written before this existed looks like.
    """
    block = (result or {}).get(PAIRED_KEY)
    if not isinstance(block, Mapping) or block.get("status") != PAIRED_MEASURED:
        return None
    if _iteration(block.get("baseline_iteration")) != _iteration(baseline_iteration):
        return None
    values = {field: as_number(block.get(field)) for field in PAIRED_FIELDS}
    if any(value is None for value in values.values()):
        return None
    return {**dict(block), **values}


def describe_paired(block: Mapping[str, Any] | None) -> str:
    """One Korean fragment for a ledger row, or ``""`` when there is nothing to say.

    Deliberately states the verdict and stops. "0과 구분되지 않음" is the whole claim an
    interval of a difference supports — not that the change did nothing, and not which of
    the levers in that transition is responsible.
    """
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
        # Appended rather than folded into the verdict, because the interval is still the
        # interval: those two prediction vectors really do differ by that much. What the row
        # can no longer say is that the plan is why: the same config across thread counts spans
        # about as much as one of these intervals is wide (:mod:`automl_agent.threads`), so the
        # environment is not a rounding error next to what is being claimed.
        text += " [baseline과 스레드 상태가 다릅니다 — 이 Δ에는 환경 차이가 섞여 있습니다]"
    return text


def _iteration(value: Any) -> int | None:
    """An iteration number as ``int``, or ``None``. ``bool`` is not an iteration number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def resolution_note(
    metrics: Mapping[str, Any] | None,
    metric: str,
    *,
    others: Mapping[str, Any] | None = None,
) -> str:
    """What this attempt's slice can and cannot resolve, as a paragraph for a prompt.

    ``others`` maps a label to a number this attempt will be compared against — the bar, each prior
    score — and every one landing inside the interval is named. That answers the Critic's actual
    failure mode: handed two point estimates and asked why the second is lower, it diagnosed a cause
    for a difference the rows never established.

    **Deliberately does not say what to do about it.** An overlap can mean the knob did nothing or
    that the slice is too small to see what it did; only the rest of the evidence separates those, and
    a prescription here would preempt the diagnosis.

    **A missing interval gets its own sentence, never silence** — the section exists either way, and an
    empty one reads as "no uncertainty".
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
    """``float`` if this is a real number, else ``None``. ``bool`` is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
