"""Shared state of the AutoML graph that every node reads and writes.

Roles:

* State schema — channels of the state and one attempt.
* State reading — read counters and scores without the LLM.
* Attempt records — build one iteration's ``history`` entry.
* Time budget — add up spent time, split what is left.
* Goal check — tell whether a result reached the goal.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .config import HOLDOUT_RESERVE_FRACTION, MIN_FIT_TIMEOUT_SEC
from .scoring.intervals import as_number

FAILURE_TYPES: tuple[str, ...] = (
    "underfitting",
    "overfitting",
    "data_issue",
    "hyperparam",
    "oom",
    "too_slow",
    "wrong_model_family",
    "unknown",
)


# --- Role: state schema ---------------------------------------------------------------


class Attempt(TypedDict):
    """Record of one full pass through the loop; appended to ``history``."""

    iteration: int
    plan: dict
    model: str
    # Applied values, not the proposal (``effective_hyperparams``).
    hyperparams: dict
    result: dict
    critic: dict | None
    # "llm" | "fallback" | "rules": who chose model and hyperparams.
    selection_source: str


class AutoMLState(TypedDict):
    """The state shared by every node in the graph.

    ``data_ref`` is private: only run nodes read it, never a prompt.
    """

    dataset_card: dict
    # Private path and target; {} builds data from the card.
    data_ref: dict
    goal: dict
    plan: dict
    model: str
    hyperparams: dict
    selection_source: str
    result: dict
    critic: dict
    history: Annotated[list[Attempt], operator.add]  # only channel that grows
    iteration: int
    max_iterations: int
    stall_count: int
    best: dict
    report: str
    # Evaluate's numbers, so the CLI prints them on resume.
    evaluation: dict
    # Kept out of ``result`` so no decision reads it.
    holdout: dict
    # Written by ``graph._bind``; adds up across resumes.
    budget: dict


# --- Role: state reading --------------------------------------------------------------


def state_int(state: AutoMLState, key: str, default: int = 0) -> int:
    """Read one counter channel as an ``int``; missing, ``None`` or ``0`` give ``default``.

    Folding ``0`` into ``default`` is on purpose
    """
    # Non-literal key makes TypedDict ``get`` typed ``object``.
    value: Any = state.get(key, default)
    return int(value or default)


def metric_value(result: dict, metric: str) -> float | None:
    """Read ``metric`` from a nested or flat result; ``None`` for errors or missing.

    Nested is ``{"metrics": {...}}`` from ``scripts/train.py``; flat is the spec's shape.
    """
    if not isinstance(result, dict):
        return None
    if result.get("status") == "error":
        return None
    metrics = result.get("metrics")
    return as_number(metrics.get(metric) if isinstance(metrics, dict) else result.get(metric))


def is_better(candidate: float | None, incumbent: float | None, direction: str) -> bool:
    """Return True if ``candidate`` beats ``incumbent`` in ``direction``.

    A ``None`` candidate never wins; a ``None`` incumbent always loses.
    """
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if direction == "minimize":
        return candidate < incumbent
    return candidate > incumbent


# --- Role: attempt records ------------------------------------------------------------


def effective_hyperparams(state: AutoMLState) -> dict:
    """Return the applied hyperparameters of an ok result, else the proposal (a new dict).

    A failed attempt keeps the proposal so the critic can use it.
    """
    result = state.get("result") or {}
    applied = result.get("applied_hyperparams") if isinstance(result, dict) else None
    if result.get("status") == "ok" and isinstance(applied, dict):
        return dict(applied)
    return dict(state.get("hyperparams") or {})


def drop_pinned_seed(hyperparams: Any, seed: int | None) -> dict:
    """Return a new dict without a ``random_state`` equal to the run seed.

    Callers compare two dicts and need this first
    """
    values = dict(hyperparams or {})
    pinned = values.get("random_state")
    if seed is not None and not isinstance(pinned, bool) and pinned == seed:
        values.pop("random_state")
    return values


def build_attempt(state: AutoMLState, critic: dict | None = None) -> Attempt:
    """Build this iteration's :class:`Attempt` for ``history``.

    Only ``critic`` and ``report`` append it, once per iteration, already judged.
    """
    return Attempt(
        iteration=state_int(state, "iteration"),
        plan=dict(state.get("plan") or {}),
        model=str(state.get("model") or ""),
        hyperparams=effective_hyperparams(state),
        result=dict(state.get("result") or {}),
        critic=dict(critic) if critic else None,
        selection_source=str(state.get("selection_source") or ""),
    )


# --- Role: time budget ----------------------------------------------------------------
# All fail open: no positive ``total_sec`` means no budget.


def accrue_budget(previous: dict | None, seconds: float, total_sec: float) -> dict:
    """Add ``seconds`` (negative counts as 0) to spent time; returns a new dict.

    The only writer of the ``budget`` channel.
    """
    spent = float((previous or {}).get("spent_sec", 0.0) or 0.0) + max(0.0, float(seconds))
    return {"spent_sec": round(spent, 3), "total_sec": float(total_sec)}


def budget_total_sec(state: AutoMLState) -> float | None:
    """Return the run's total budget in seconds, or ``None`` when there is none to enforce."""
    total = as_number((state.get("budget") or {}).get("total_sec"))
    return total if total is not None and total > 0 else None


def budget_spent_sec(state: AutoMLState) -> float:
    """Return the seconds the run has spent so far (0 when not recorded)."""
    return max(0.0, as_number((state.get("budget") or {}).get("spent_sec")) or 0.0)


def loop_time_remaining_sec(state: AutoMLState) -> float | None:
    """Return loop seconds left after the holdout reserve (may be negative), or ``None``.

    The holdout reserve is kept out on purpose
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    return total * (1.0 - HOLDOUT_RESERVE_FRACTION) - budget_spent_sec(state)


def loop_budget_exhausted(state: AutoMLState) -> bool:
    """Return True if another iteration would use time the run does not have."""
    remaining = loop_time_remaining_sec(state)
    return remaining is not None and remaining <= 0.0


def fit_share_sec(state: AutoMLState) -> float | None:
    """Return one fit's share of loop time, split over iterations left, or ``None``.

    May be 0 or less; the caller decides, never clamp
    """
    remaining = loop_time_remaining_sec(state)
    if remaining is None:
        return None
    left = max(1, state_int(state, "max_iterations") - state_int(state, "iteration") + 1)
    share = remaining / left
    return share if share <= 0 else max(MIN_FIT_TIMEOUT_SEC, share)


def holdout_share_sec(state: AutoMLState) -> float | None:
    """Return holdout seconds, never less than the reserve

    ``None`` when there is no budget.
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    reserve = total * HOLDOUT_RESERVE_FRACTION
    return max(reserve, total - budget_spent_sec(state))


def describe_budget(budget: dict) -> str:
    """Format a ``budget`` like ``2,913초 / 3,600초 (81%)`` for console and report."""
    spent = float(budget.get("spent_sec", 0.0) or 0.0)
    total = as_number(budget.get("total_sec"))
    if total is not None and total > 0:
        return f"{spent:,.0f}초 / {total:,.0f}초 ({spent / total:.0%})"
    return f"{spent:,.0f}초 (예산 없음)"


# --- Role: goal check -----------------------------------------------------------------


def goal_met(result: dict, goal: dict) -> bool:
    """Return True if ``result`` reached the goal threshold.

    An error result or no threshold is never met
    """
    metric = str(goal.get("metric", "f1"))
    score = metric_value(result, metric)
    if score is None:
        return False
    raw = goal.get("threshold")
    if raw is None:
        return False
    threshold = float(raw)
    if str(goal.get("direction", "maximize")) == "minimize":
        return score <= threshold
    return score >= threshold
