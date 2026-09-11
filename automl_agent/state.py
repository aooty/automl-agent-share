"""Shared state (blackboard) for the AutoML graph.

Every node is a pure function ``state -> dict`` that returns a *partial* update.
``history`` is the only accumulating channel: it uses an ``operator.add`` reducer
so each attempt is appended instead of overwriting the previous ones.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from .config import HOLDOUT_RESERVE_FRACTION, MIN_FIT_TIMEOUT_SEC

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

FailureType = Literal[
    "underfitting",
    "overfitting",
    "data_issue",
    "hyperparam",
    "oom",
    "too_slow",
    "wrong_model_family",
    "unknown",
]

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


class Attempt(TypedDict):
    """One full trip around the loop, appended to ``history``."""

    iteration: int
    plan: dict
    model: str
    # The params the executor applied (see ``effective_hyperparams``), not the proposal.
    hyperparams: dict
    result: dict  # {"f1": 0.72, "train_time_sec": 812} or {"error": "oom"}
    critic: dict | None  # {"failure_type": ..., "direction": ...}


class AutoMLState(TypedDict):
    """The blackboard shared by every node in the graph.

    Two channels carry data-adjacent material, and the split between them is a
    security boundary, not a tidiness one:

    ``data_ref``      the file path and target column. Read *only* by the execution
                      nodes (``profiling``, ``training``), which pass it to a
                      subprocess. Never rendered into a prompt.
    ``dataset_card``  aggregate summary produced by ``profiling``. This is the only
                      thing the reasoning nodes learn about the data.

    Adding a read of ``data_ref`` to a reasoning node would defeat the isolation, so
    ``tests/test_privacy.py`` asserts the path never appears in an archived prompt.
    """

    dataset_card: dict
    # Private: {"path": ..., "target_column": ...}, or {} to synthesise from the card.
    data_ref: dict
    goal: dict  # {"metric": ..., "threshold": ..., "direction": "maximize"}
    plan: dict
    model: str
    hyperparams: dict
    result: dict
    critic: dict
    history: Annotated[list[Attempt], operator.add]  # accumulated via reducer
    iteration: int
    max_iterations: int
    stall_count: int  # consecutive non-improving iterations
    best: dict  # snapshot of the best result so far
    report: str
    # Extension beyond the spec's schema: the evaluate node's derived numbers
    # (score / improved / goal_met), kept in state so the CLI can print the
    # per-iteration summary line and resumed runs can reprint it.
    evaluation: dict
    # The one score no decision in the run was made against: the best saved model on the
    # test slice, measured once after the loop stopped (``nodes/holdout.py``). Deliberately
    # *not* merged into ``result``, because ``route`` and ``goal_met`` read that channel
    # and a held-back number that steered the loop would not be held back.
    holdout: dict
    # ``{"spent_sec": 812.4, "total_sec": 3600.0}`` — the run's time budget, accounted for
    # rather than assumed. Written by the graph's node wrapper (``graph._bind``) so every node
    # is counted, including the ones that spend their time inside an LLM call. Cumulative and
    # therefore resume-safe: ``--time-budget-sec`` bounds the seconds the run *works*, not the
    # wall clock since it started, so an interrupted run resumes with what it already spent.
    budget: dict


# --------------------------------------------------------------------------- #
# Deterministic helpers (no LLM involved — used by evaluate/route)
# --------------------------------------------------------------------------- #


def metric_value(result: dict, metric: str) -> float | None:
    """Pull ``metric`` out of a training result, tolerating both shapes.

    ``scripts/train.py`` returns ``{"metrics": {"f1": ...}, ...}`` while the spec's
    ``Attempt.result`` example is flat (``{"f1": ...}``). Accept either, and return
    ``None`` for an errored/absent metric so callers can treat it as "no score".
    """
    if not isinstance(result, dict):
        return None
    if result.get("status") == "error":
        return None
    metrics = result.get("metrics")
    raw = metrics.get(metric) if isinstance(metrics, dict) else result.get(metric)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def is_better(candidate: float | None, incumbent: float | None, direction: str) -> bool:
    """True when ``candidate`` beats ``incumbent`` under ``direction``."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if direction == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def effective_hyperparams(state: AutoMLState) -> dict:
    """What the record should quote: the applied params, else the proposal.

    ``state["hyperparams"]`` is a *proposal*. ``scripts/train.py`` narrows it to the
    parameters the chosen estimator actually accepts, so a report quoting the proposal
    can list a parameter that was silently dropped and present it as the configuration
    that produced the score.

    A failed attempt is the one case where the proposal is the better record: nothing was
    applied, and the proposal is exactly what the Critic needs to see to diagnose the
    failure (the ``batch_size`` behind an OOM, say).
    """
    result = state.get("result") or {}
    applied = result.get("applied_hyperparams") if isinstance(result, dict) else None
    if result.get("status") == "ok" and isinstance(applied, dict):
        return dict(applied)
    return dict(state.get("hyperparams") or {})


def build_attempt(state: AutoMLState, critic: dict | None = None) -> Attempt:
    """Snapshot the current iteration for the ``history`` channel.

    Appending happens in ``critic`` (mid-loop) and ``report`` (final iteration) —
    exactly one of which runs per iteration — so every attempt lands in history
    once, already carrying its verdict. ``evaluate`` cannot do it: with an
    ``operator.add`` reducer an earlier entry can never be patched afterwards.
    """
    return Attempt(
        iteration=int(state.get("iteration", 0) or 0),
        plan=dict(state.get("plan") or {}),
        model=str(state.get("model") or ""),
        hyperparams=effective_hyperparams(state),
        result=dict(state.get("result") or {}),
        critic=dict(critic) if critic else None,
    )


# --------------------------------------------------------------------------- #
# The time budget, as arithmetic over the ``budget`` channel
# --------------------------------------------------------------------------- #
#
# ``--time-budget-sec`` used to be handed to every training subprocess as *its own* timeout and
# read by nothing else, so at the defaults (5 iterations, 3600s) the worst case was 21,600
# seconds of fits and the flag bounded no run. These helpers are what make it a run budget: one
# accrues, one decides the loop is over, two divide what is left, and the rest read the channel.
#
# Every one of them treats a missing or non-positive ``total_sec`` as "no budget" and answers
# ``None``/``False``. A node invoked directly (the unit tests, a hand-edited checkpoint) then
# behaves exactly as it did before this channel existed, and a budget that cannot be read never
# becomes a budget of zero — being cut off by an absent number would be the worse failure.


def accrue_budget(previous: dict | None, seconds: float, total_sec: float) -> dict:
    """Add ``seconds`` to what the run has spent. The only writer of the channel."""
    spent = float((previous or {}).get("spent_sec", 0.0) or 0.0) + max(0.0, float(seconds))
    return {"spent_sec": round(spent, 3), "total_sec": float(total_sec)}


def budget_total_sec(state: AutoMLState) -> float | None:
    """The run's whole budget, or ``None`` when there is none to enforce."""
    raw = (state.get("budget") or {}).get("total_sec")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    return float(raw)


def budget_spent_sec(state: AutoMLState) -> float:
    raw = (state.get("budget") or {}).get("spent_sec")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return max(0.0, float(raw))


def loop_time_remaining_sec(state: AutoMLState) -> float | None:
    """What the *loop* may still spend — the budget less holdout's reserved share.

    The loop does not get the whole budget, because the number this run reports is the one
    ``holdout`` measures after the loop stops. A budget the loop can spend down to zero is a
    budget that deletes the run's own answer, and ``holdout`` failing on a timeout is recorded
    as ``skipped`` — the report would then be written from selected-on validation scores with
    nothing to correct them.
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    return total * (1.0 - HOLDOUT_RESERVE_FRACTION) - budget_spent_sec(state)


def loop_budget_exhausted(state: AutoMLState) -> bool:
    """Whether another iteration would spend time the run does not have."""
    remaining = loop_time_remaining_sec(state)
    return remaining is not None and remaining <= 0.0


def fit_share_sec(state: AutoMLState) -> float | None:
    """One fit's slice: what the loop has left, divided by the iterations that may still run.

    Divided rather than handed over whole. Either choice bounds the run, but giving the whole
    remainder to the next fit lets iteration 1 spend the run — and a loop that cannot reach
    iteration 2 is not the thing this repository is measuring. The current iteration counts
    itself, so at iteration 1 of 5 a fit gets a fifth and at iteration 5 it gets the rest.

    Can come back non-positive: ``route`` checks the budget between iterations, and the
    planning and model-selection calls that follow its decision also cost time. The caller
    decides what to do with that (``nodes/training.py`` declines to start the fit) — clamping
    it to something positive here would spend budget the run does not have.
    """
    remaining = loop_time_remaining_sec(state)
    if remaining is None:
        return None
    iteration = int(state.get("iteration", 0) or 0)
    max_iterations = int(state.get("max_iterations", 0) or 0)
    left = max(1, max_iterations - iteration + 1)
    share = remaining / left
    return share if share <= 0 else max(MIN_FIT_TIMEOUT_SEC, share)


def holdout_share_sec(state: AutoMLState) -> float | None:
    """What is left for the final scoring pass, and never less than the reserve.

    Never less, because a fit already in flight can overrun the loop's share — its own timeout
    is a slice of what remained when it started, and the LLM calls around it are not bounded
    at all. The reserve is what the loop was kept away from, so holdout gets it even when the
    accounting says the run is already over.
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    reserve = total * HOLDOUT_RESERVE_FRACTION
    return max(reserve, total - budget_spent_sec(state))


def describe_budget(budget: dict) -> str:
    """``2,913초 / 3,600초 (81%)`` for the console and the report."""
    spent = float(budget.get("spent_sec", 0.0) or 0.0)
    total = budget.get("total_sec")
    if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
        return f"{spent:,.0f}초 / {float(total):,.0f}초 ({spent / float(total):.0%})"
    return f"{spent:,.0f}초 (예산 없음)"


def goal_met(result: dict, goal: dict) -> bool:
    """Whether ``result`` reaches the goal threshold. Errors never satisfy a goal.

    A goal with no threshold is never met. That state means the bar could not be derived
    (a metric in the target's units with no measured baseline — see
    :func:`automl_agent.scoring.goal.derive_threshold`), and ``profiling`` stops such a run before
    the loop starts; this branch only keeps a hand-edited checkpoint from crashing here
    instead of being reported as unmet.
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
