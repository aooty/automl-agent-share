"""Evaluate node: score the last attempt with plain code. No LLM.

Roles:

* Scoring: compare the last score with the best so far.
* Loop counters: own ``best`` and ``stall_count`` for ``route``.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig
from ..scoring.metrics import direction_of
from ..state import AutoMLState, effective_hyperparams, goal_met, is_better, metric_value, state_int

# --- Role: scoring and loop counters --------------------------------------------------


def evaluate(state: AutoMLState, *, config: RunConfig) -> dict:
    """Score the last result and update ``best`` and ``stall_count``."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = str(goal.get("direction") or direction_of(metric))
    result = dict(state.get("result") or {})
    iteration = state_int(state, "iteration")

    score = metric_value(result, metric)
    previous_best = dict(state.get("best") or {})
    improved = is_better(score, previous_best.get("score"), direction)

    best = previous_best
    if improved:
        best = {
            "iteration": iteration,
            "model": str(state.get("model") or ""),
            # Applied values, so the best config can be re-run.
            "hyperparams": effective_hyperparams(state),
            "preprocessing": dict(result.get("applied_preprocessing") or {}),
            "metric": metric,
            "score": score,
            "metrics": dict(result.get("metrics") or {}),
            "train_time_sec": result.get("train_time_sec"),
            "plan_strategy": (state.get("plan") or {}).get("strategy", ""),
        }

    # Failed or non-improving counts as a stall.
    stall_count = 0 if improved else state_int(state, "stall_count") + 1

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
