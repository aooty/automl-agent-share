"""Probability quality: can predicted probabilities be read as probabilities.

Roles:

* Measures — Brier score, reliability table, calibration error.
* Metrics entries — put measured values into a metrics dict.
* Descriptions — Korean lines for reports and console.
"""

from __future__ import annotations

import math
from typing import Any

from .intervals import as_number

# Shared names so writer and reader never drift apart.
BRIER_KEY = "brier"
CALIBRATION_KEY = "calibration_error"

# Equal-width bins on [0, 1]
N_BINS = 10

# Fewer rows: no calibration_error (about five per bin).
MIN_CALIBRATION_ROWS = 50


# --- Role: measures ---------------------------------------------------------------


def brier_score(y_true: Any, proba: Any) -> float | None:
    """brier_score | Role: ``mean((p - y)**2)``, or ``None`` if it cannot be computed."""
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
    """reliability | Role: one ``{low, high, rows, predicted, observed}`` per non-empty bin.

    Empty bins are left out; bad inputs give ``[]``.
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
    # Inner edges only: 0.0 and 1.0 get no own bin.
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
    """calibration_error | Role: row-weighted mean gap per bin.

    ``None`` below :data:`MIN_CALIBRATION_ROWS` rows.
    """
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


# --- Role: metrics entries --------------------------------------------------------


def measure(y_true: Any, proba: Any, bins: int = N_BINS) -> dict[str, float]:
    """measure | Role: the two diagnostic keys; unmeasured ones are left out, not ``None``."""
    found: dict[str, float] = {}
    score = brier_score(y_true, proba)
    if score is not None:
        found[BRIER_KEY] = score
    error = calibration_error(y_true, proba, bins)
    if error is not None:
        found[CALIBRATION_KEY] = error
    return found


# --- Role: descriptions -----------------------------------------------------------


def describe(metrics: Any, rows: int | None = None) -> str:
    """describe | Role: one Korean line of numbers, no verdict; "" if nothing measured."""
    values = dict(metrics or {})
    score = as_number(values.get(BRIER_KEY))
    error = as_number(values.get(CALIBRATION_KEY))
    if score is None and error is None:
        return ""
    parts: list[str] = []
    if score is not None:
        parts.append(f"brier={score:.4f}")
    if error is not None:
        parts.append(f"확률 오차={error:.4f}")
    line = "확률 품질(진단, 목표로 삼을 수 없음): " + ", ".join(parts)
    if error is not None:
        line += (
            f" — 예측 확률과 실제 발생률의 차이가 평균 {error * 100:.1f}%p입니다"
            f" ({N_BINS}개 구간, 개수 가중)"
        )
    elif rows is not None and rows < MIN_CALIBRATION_ROWS:
        line += (
            f" — 행이 {rows}개뿐이라 구간별 확률 오차는 측정하지 않았습니다 "
            f"({MIN_CALIBRATION_ROWS}행 이상 필요)"
        )
    return line


def describe_table(table: list[dict[str, Any]]) -> list[str]:
    """describe_table | Role: Korean console lines; per-bin counts, never in a prompt."""
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
