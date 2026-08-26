"""Shared state (blackboard) for the AutoML graph.

Every node is a pure function ``state -> dict`` that returns a *partial* update.
``history`` is the only accumulating channel: it uses an ``operator.add`` reducer
so each attempt is appended instead of overwriting the previous ones.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

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

GoalDirection = Literal["maximize", "minimize"]


class Goal(TypedDict):
    """Target metric the loop is optimizing toward."""

    metric: str
    threshold: float
    direction: str  # "maximize" | "minimize"


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
