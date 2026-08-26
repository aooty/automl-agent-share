"""Evaluate node: deterministic metric bookkeeping. No LLM.

Owns the two counters ``route`` depends on — ``best`` and ``stall_count`` — so the
stopping decision is derived from numbers computed here, never from a judgement.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig
from ..scoring.metrics import direction_of
from ..state import AutoMLState, effective_hyperparams, goal_met, is_better, metric_value


def evaluate(state: AutoMLState, *, config: RunConfig) -> dict:
    """Score the latest result, update ``best`` and ``stall_count``."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = str(goal.get("direction") or direction_of(metric))
    result = dict(state.get("result") or {})
    iteration = int(state.get("iteration", 0) or 0)

    score = metric_value(result, metric)
    previous_best = dict(state.get("best") or {})
    improved = is_better(score, previous_best.get("score"), direction)

    best = previous_best
    if improved:
        best = {
            "iteration": iteration,
            "model": str(state.get("model") or ""),
            # The applied params, so "최고 성능 구성" is a configuration that can be
            # re-run and reproduce this score. The pipeline the executor built belongs to
            # that configuration too: the same estimator fitted on median-imputed columns
            # and on raw NaNs are two different models.
            "hyperparams": effective_hyperparams(state),
            "preprocessing": dict(result.get("applied_preprocessing") or {}),
            "metric": metric,
            "score": score,
            "metrics": dict(result.get("metrics") or {}),
            "train_time_sec": result.get("train_time_sec"),
            "plan_strategy": (state.get("plan") or {}).get("strategy", ""),
        }

    # A failed or non-improving attempt counts as a stall; any improvement resets it.
    stall_count = 0 if improved else int(state.get("stall_count", 0) or 0) + 1

    evaluation: dict[str, Any] = {
        "iteration": iteration,
        "metric": metric,
        "score": score,
        "goal_met": goal_met(result, goal),
        "improved": improved,
        "status": result.get("status", "error"),
        "error_type": result.get("error_type"),
    }
    return {"best": best, "stall_count": stall_count, "evaluation": evaluation}
