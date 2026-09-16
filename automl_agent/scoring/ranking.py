"""Ranking ceiling: ``max_cut balanced_accuracy = (1 + KS) / 2``, an identity.

A bar above it is out of reach of any cut, so the *ranking* must improve — advisory, never a
refusal. Import-light: only the measuring function touches sklearn, lazily.

Rationale: ``docs/rationale.md``.
"""

from __future__ import annotations

import math
from typing import Any

# Equally-weighted error rates only, so ``f1`` and ``accuracy`` are out. Not merged with
# ``critic.SYMMETRIC_METRICS``: same name, different claim. Rationale: ``docs/rationale.md``.
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})


def ks_statistic(y_true: Any, proba: Any) -> float | None:
    """``max(TPR - FPR)`` of a binary ranking, or ``None`` when the split cannot support one.

    ``None`` covers a single-class holdout (no ROC curve) and a non-finite maximum —
    ``roc_curve`` warns rather than raising there, and a NaN would make the card invalid JSON.
    """
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


def best_cut_ceiling(ks: float | None) -> float | None:
    """``(1 + KS) / 2`` — the best ``balanced_accuracy`` any cut of this ranking allows."""
    if ks is None or not isinstance(ks, (int, float)) or isinstance(ks, bool):
        return None
    value = float(ks)
    if not 0.0 <= value <= 1.0:
        return None
    return round((1.0 + value) / 2.0, 4)


def required_ks(threshold: float | None) -> float | None:
    """``2 * threshold - 1`` — the identity read the other way, putting the bar on the lever's axis.

    ``None`` above 1.0: such a bar is already disclosed as over
    :data:`automl_agent.scoring.goal.CEILING`. Rationale: ``docs/rationale.md``.
    """
    if threshold is None or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    value = 2.0 * float(threshold) - 1.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 4)


def passable_margin(baseline: float, ceiling: float) -> float | None:
    """The largest ``auto`` margin whose bar still fits under ``ceiling``.

    Inverts ``bar = baseline + (1 - baseline) * margin`` (:func:`automl_agent.scoring.goal._target`),
    so the disclosure names the margin instead of telling the caller to guess downward.
    """
    headroom = 1.0 - float(baseline)
    if headroom <= 0:
        return None
    margin = (float(ceiling) - float(baseline)) / headroom
    if margin <= 0:
        return None
    # Floored, not rounded: a rounded-up margin would put the bar back over the ceiling.
    return max(0.0, float(int(margin * 1000)) / 1000)


def card_ceiling(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """``(ks, ceiling)`` for ``metric`` from a card's baseline block, or ``(None, None)``.

    Only :data:`SYMMETRIC_METRICS`: elsewhere there is no identity, and inventing a bound is worse
    than staying quiet.
    """
    if metric not in SYMMETRIC_METRICS:
        return None, None
    baseline_block = (card or {}).get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None
    raw = baseline_block.get("ks")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, None
    return float(raw), best_cut_ceiling(float(raw))
