"""Ranking ceiling: ``max_cut balanced_accuracy = (1 + KS) / 2``, an exact identity.

Roles:

* KS measure — KS statistic of a binary ranking.
* Ceiling math — KS to best score, bar back to KS/margin.
* Card ceiling — read a metric's ceiling from a dataset card.
"""

from __future__ import annotations

import math
from typing import Any

from .intervals import as_number

# Not merged with critic.SYMMETRIC_METRICS on purpose
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})


# --- Role: KS measure -------------------------------------------------------------


def ks_statistic(y_true: Any, proba: Any) -> float | None:
    """ks_statistic | Role: ``max(TPR - FPR)``, or ``None`` if unusable or NaN."""
    if proba is None:
        return None
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, proba)
    except (ValueError, ImportError, IndexError):
        return None
    if len(fpr) == 0:
        return None
    value = float((tpr - fpr).max())
    return value if math.isfinite(value) else None


# --- Role: ceiling math -----------------------------------------------------------


def best_cut_ceiling(ks: float | None) -> float | None:
    """best_cut_ceiling | Role: ``(1 + KS) / 2``, best balanced_accuracy of any cut.

    ``None`` when ``ks`` is missing or outside [0, 1].
    """
    value = as_number(ks)
    if value is None or not 0.0 <= value <= 1.0:
        return None
    return round((1.0 + value) / 2.0, 4)


def required_ks(threshold: float | None) -> float | None:
    """required_ks | Role: ``2 * threshold - 1``, the KS a bar needs.

    ``None`` when outside [0, 1].
    """
    number = as_number(threshold)
    if number is None:
        return None
    value = 2.0 * number - 1.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 4)


def passable_margin(baseline: float, ceiling: float) -> float | None:
    """passable_margin | Role: largest ``auto`` margin whose bar stays under ``ceiling``.

    Inverts ``goal._target``; ``None`` when there is no room.
    """
    headroom = 1.0 - float(baseline)
    if headroom <= 0:
        return None
    margin = (float(ceiling) - float(baseline)) / headroom
    if margin <= 0:
        return None
    # Round down: rounding up crosses the ceiling again.
    return max(0.0, float(int(margin * 1000)) / 1000)


# --- Role: card ceiling -----------------------------------------------------------


def card_ceiling(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """card_ceiling | Role: ``(ks, ceiling)`` from the card, else ``(None, None)``."""
    if metric not in SYMMETRIC_METRICS:
        return None, None
    baseline_block = (card or {}).get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None
    value = as_number(baseline_block.get("ks"))
    if value is None:
        return None, None
    return value, best_cut_ceiling(value)
