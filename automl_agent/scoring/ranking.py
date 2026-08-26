"""What the ranking alone decides, and what the operating point can still buy.

Two runs were spent finding out the hard way. ``m-llm7`` and ``m-llm8`` were given
``balanced_accuracy`` bars of 0.7842 and 0.8038 on the same MIMIC sample; the first
cleared its bar by 0.0005 and the second never got past 0.7881 in four attempts. The
reason is not search quality. On a binary target,

    balanced_accuracy = (recall + specificity) / 2
                      = (TPR + (1 - FPR)) / 2
                      = (1 + (TPR - FPR)) / 2

so maximising it over the decision threshold maximises ``TPR - FPR``, whose maximum *is*
the definition of the Kolmogorov-Smirnov statistic. Hence

    max_threshold balanced_accuracy = (1 + KS) / 2

an identity, not an estimate. Two rankings of that sample were measured. The card's
baseline logreg has KS 0.5168, so no cut of *it* scores above 0.7584; ``m-llm8``'s best
hist_gbdt improved the ranking to KS 0.5802, whose ceiling is 0.7901 — and the oracle
threshold on the holdout confirmed exactly that number. **The 0.8038 bar was above both**,
so no amount of ``class_weight`` search could have found it; clearing it needed KS 0.6076.

That makes the number worth two things, which is why it lives here rather than inside
either caller:

* :mod:`automl_agent.scoring.goal` compares an ``auto`` bar against it before the loop starts,
  so an unreachable bar is disclosed instead of discovered four iterations later.
* :mod:`automl_agent.scripts.train` reports each attempt's distance from it, which is
  the number that answers the threshold-tuning question every real LLM run asks.

**It is a property of one model's ranking, not of the data.** The pair of numbers above is
the proof: a better family raised the ceiling from 0.7584 to 0.7901, and roc_auc from the
baseline's 0.8294 to 0.8738. So a card's ceiling bounds the *baseline*, not the run, and
the honest reading of "the bar exceeds the ceiling" is *"the ranking itself has to
improve"* — features or model family — never "this bar is impossible". That also makes the
card's number the conservative one, which is the right direction for a warning issued
before the loop starts. Everything here is advisory for the same reason; a caller that
cannot get a KS carries on without one.

Dependency-free at import time, for the same reason :mod:`automl_agent.scoring.metrics` is: the
orchestrator process imports this to read a card, while the two fixed scripts import it
to measure one. Only the measuring function touches sklearn, and it does so lazily.
"""

from __future__ import annotations

import math
from typing import Any

# Metrics whose best-threshold value is an exact function of the ranking, per the
# identity above. It holds because ``balanced_accuracy`` weights the two error rates
# equally — ``f1`` and ``accuracy`` do not, and have no such closed form. The same
# symmetry is why ``class_weight`` is the lever that centres it, which is what
# ``nodes/critic.py`` uses the set for.
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})


def ks_statistic(y_true: Any, proba: Any) -> float | None:
    """The Kolmogorov-Smirnov statistic ``max(TPR - FPR)`` of a binary ranking.

    ``None`` rather than an exception when the split cannot support one — a single-class
    holdout has no ROC curve. Every caller treats a missing KS as "no ceiling known",
    so a degenerate split costs a disclosure, not a run.

    The finiteness check is not defensive padding. On an all-negative holdout ``roc_curve``
    does not raise: it warns and returns a TPR column of NaN, so the maximum is NaN, and a
    NaN written into the card would make it invalid JSON.
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
    """The KS a ranking needs before *any* cut of it reaches ``threshold``.

    The same identity as :func:`best_cut_ceiling`, read the other way: ``2 * bar - 1``. Worth
    stating separately because it puts the bar on the axis the lever is on. "The bar exceeds
    this ranking's ceiling" names the obstacle but not its size, and mv-llm-4 through 6 show
    what that costs — all three spent iteration 1 on the operating point and only found out
    from the ledger, one iteration later, that 96% of the shortfall was in the ranking. On the
    MIMIC sample the 0.822 bar demands KS 0.6440 against the baseline's 0.4945, so the ranking
    has to gain 0.1495 — next to which a family swap's measured 0.0022 to 0.0077 of `roc_auc`
    is visibly not the lever, and that comparison cannot be made without this number.

    ``None`` above 1.0 rather than a KS no ranking can have: a bar that far out is already
    disclosed as over :data:`automl_agent.scoring.goal.CEILING`, and two impossibility notes on one
    line say less than one.
    """
    if threshold is None or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    value = 2.0 * float(threshold) - 1.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 4)


def passable_margin(baseline: float, ceiling: float) -> float | None:
    """The largest ``auto`` margin whose bar still fits under ``ceiling``.

    Inverts :func:`automl_agent.scoring.goal._target`'s bounded-maximise form,
    ``bar = baseline + (1 - baseline) * margin``, so the disclosure can name the number
    the caller would have to pass rather than telling them to guess downward. On the
    MIMIC sample: baseline 0.6077 and the card's ceiling 0.7584 give 0.384, against the
    0.5 the failing run used. Measured against the best ranking anyone reached there
    (ceiling 0.7901) it is 0.464 — still under 0.5, which is why that run failed either way.
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

    Only :data:`SYMMETRIC_METRICS` get a ceiling: for anything else this module has no
    identity to offer, and inventing a bound would be worse than staying quiet.
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
