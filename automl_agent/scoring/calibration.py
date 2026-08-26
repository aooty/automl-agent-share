"""How much a predicted probability is worth as a probability.

Everything else this project measures is about *ranking* or about *labels*: ``roc_auc`` asks
whether the positive rows scored above the negative ones, ``f1`` asks how the hard 0/1 call
came out. Neither asks the question an operator actually asks of the output CSV — "the model
says 0.7 for this row; does 70% of anything happen?" A model can rank perfectly and be
systematically overconfident, and every metric in the registry will be happy.

That gap is not hypothetical here. ``scripts/train.py`` predicts labels at sklearn's fixed 0.5
cut and :mod:`automl_agent.capabilities` declares choosing a threshold a non-capability, so the
probabilities are the part of the output that a caller *can* act on with their own cut. What
this module does is measure whether they are worth acting on. What it deliberately does not do
is change them: no ``CalibratedClassifierCV``, no shifted cut. Recalibration is a model change
fitted on labelled rows, and the two places labelled rows exist here are the holdout — which is
scored once and gates nothing — and a caller's own backtest file, which this process is not
entitled to fit anything on. Measure, report, leave the model alone.

Two numbers, and they are not interchangeable:

``brier``
    Mean squared error of the positive-class probability, ``mean((p - y)**2)``. A proper
    scoring rule: it is minimised only by the true probabilities, so it prices calibration and
    discrimination together. Unbiased at any sample size, which is why it is reported on a
    40-row batch and the other one is not.

``calibration_error``
    Expected calibration error — bin the rows by predicted probability, and take the
    count-weighted mean distance between each bin's mean prediction and its observed rate.
    This is the one that answers the operator's question directly, and it is *biased upward on
    few rows*: with ten bins and forty rows, a bin holds four rows and its observed rate can
    only be 0, 0.25, 0.5, 0.75 or 1. So it is only reported above
    :data:`MIN_CALIBRATION_ROWS`, and its absence is disclosed rather than silent.

Both are diagnostics, on the same terms as ``specificity`` and ``cut_headroom``: they are not
in :mod:`automl_agent.scoring.metrics`' registry, so no goal can be set against them and no attempt can
be selected for them. A run that optimised ``brier`` would be a different run than the one the
operator asked for.

Dependency-free at import time, for the reason :mod:`automl_agent.scoring.ranking` is: the orchestrator
process imports this to *describe* a number, while the two fixed scripts import it to measure
one, and only the measuring functions touch numpy.
"""

from __future__ import annotations

import math
from typing import Any

# The two keys this module contributes to a metrics dict. Named here so the script that writes
# them and the node that reads them back cannot disagree by a typo.
BRIER_KEY = "brier"
CALIBRATION_KEY = "calibration_error"

# Equal-width bins over [0, 1]. Ten is the convention and it is also the most a 50-row split can
# fill without most bins holding one row.
N_BINS = 10

# Below this, ``calibration_error`` is not reported. See the module docstring: the estimator is
# biased upward when a bin holds a handful of rows, and a number that reads as "4% miscalibrated"
# when it is measuring bin granularity is worse than no number. 50 rows is five per bin at the
# mean, which is the point where the observed rate stops being a coarse fraction.
MIN_CALIBRATION_ROWS = 50


def brier_score(y_true: Any, proba: Any) -> float | None:
    """``mean((p - y)**2)`` over the positive-class probability. ``None`` when unavailable.

    ``None`` rather than an exception for a regression target, a model without
    ``predict_proba``, or an empty split — every caller treats a missing value as "not
    measured", so a degenerate split costs a disclosure rather than a run.
    """
    import numpy as np

    if proba is None:
        return None
    try:
        p = np.asarray(proba, dtype="float64").ravel()
        y = np.asarray(y_true, dtype="float64").ravel()
    except (TypeError, ValueError):
        return None
    if len(p) == 0 or len(p) != len(y):
        return None
    value = float(np.mean((p - y) ** 2))
    return round(value, 6) if math.isfinite(value) else None


def reliability(y_true: Any, proba: Any, bins: int = N_BINS) -> list[dict[str, Any]]:
    """One entry per non-empty probability bin: ``{low, high, rows, predicted, observed}``.

    Empty bins are dropped rather than reported as zeros — a bin no row landed in is not a bin
    where the model was wrong. Aggregates only: a row count, a mean prediction and an observed
    rate, which is the same shape of fact a dataset card publishes about a column.
    """
    import numpy as np

    if proba is None:
        return []
    try:
        p = np.asarray(proba, dtype="float64").ravel()
        y = np.asarray(y_true, dtype="float64").ravel()
    except (TypeError, ValueError):
        return []
    if len(p) == 0 or len(p) != len(y):
        return []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # ``right=True`` with the first edge folded in, so 0.0 lands in the first bin and 1.0 in the
    # last rather than in bins of their own.
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, int(bins) - 1)
    table: list[dict[str, Any]] = []
    for position in range(int(bins)):
        hit = index == position
        rows = int(hit.sum())
        if not rows:
            continue
        table.append(
            {
                "low": round(float(edges[position]), 4),
                "high": round(float(edges[position + 1]), 4),
                "rows": rows,
                "predicted": round(float(p[hit].mean()), 4),
                "observed": round(float(y[hit].mean()), 4),
            }
        )
    return table


def calibration_error(y_true: Any, proba: Any, bins: int = N_BINS) -> float | None:
    """Count-weighted mean distance between predicted and observed rate. ``None`` under
    :data:`MIN_CALIBRATION_ROWS` rows, because below that it measures the binning."""
    table = reliability(y_true, proba, bins)
    total = sum(int(item["rows"]) for item in table)
    if total < MIN_CALIBRATION_ROWS:
        return None
    weighted = sum(
        int(item["rows"]) * abs(float(item["predicted"]) - float(item["observed"]))
        for item in table
    )
    value = weighted / total
    return round(value, 6) if math.isfinite(value) else None


def measure(y_true: Any, proba: Any, bins: int = N_BINS) -> dict[str, float]:
    """The two diagnostic keys, omitting whichever could not be measured.

    Omitted rather than ``None``: these ride in a metrics dict whose consumers assume every
    value is a number (``privacy.public_result`` keeps numeric values and drops the rest), so a
    key that is sometimes null would read as a metric that sometimes failed.
    """
    found: dict[str, float] = {}
    score = brier_score(y_true, proba)
    if score is not None:
        found[BRIER_KEY] = score
    error = calibration_error(y_true, proba, bins)
    if error is not None:
        found[CALIBRATION_KEY] = error
    return found


def describe(metrics: Any, rows: int | None = None) -> str:
    """One Korean line, or "" when there is nothing measured to say.

    Deliberately not a verdict. There is no bar at which a Brier score is "good" — it depends
    on the base rate, so 0.09 is poor on a 2%-positive target and excellent on a balanced one —
    and inventing one would be the harness making the operator's decision for it. What the line
    does say is what the number *is about*, and that a probability is not the label.
    """
    values = dict(metrics or {})
    score = values.get(BRIER_KEY)
    error = values.get(CALIBRATION_KEY)
    if not isinstance(score, (int, float)) and not isinstance(error, (int, float)):
        return ""
    parts: list[str] = []
    if isinstance(score, (int, float)):
        parts.append(f"brier={float(score):.4f}")
    if isinstance(error, (int, float)):
        parts.append(f"확률 오차={float(error):.4f}")
    line = "확률 품질(진단, 목표로 삼을 수 없음): " + ", ".join(parts)
    if isinstance(error, (int, float)):
        line += (
            f" — 예측 확률과 실제 발생률의 차이가 평균 {float(error) * 100:.1f}%p입니다"
            f" ({N_BINS}개 구간, 개수 가중)"
        )
    elif rows is not None and rows < MIN_CALIBRATION_ROWS:
        line += (
            f" — 행이 {rows}개뿐이라 구간별 확률 오차는 측정하지 않았습니다 "
            f"({MIN_CALIBRATION_ROWS}행 이상 필요)"
        )
    return line


def describe_table(table: list[dict[str, Any]]) -> list[str]:
    """The reliability table as lines, for a console that stays on the operator's machine.

    Per-bin counts, so this is not printed anywhere a prompt can reach — the two callers are
    ``scripts/predict.py``'s stdout and a local report file.
    """
    if not table:
        return []
    lines = [f"확률 구간별 실제 발생률 ({N_BINS}구간, 빈 구간 제외):"]
    for item in table:
        lines.append(
            f"  {float(item['low']):.1f}~{float(item['high']):.1f}: "
            f"{int(item['rows'])}행, 예측 평균 {float(item['predicted']):.3f}, "
            f"실제 {float(item['observed']):.3f}"
        )
    return lines
