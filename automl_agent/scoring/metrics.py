"""Metric registry: one declared list of every metric a run can target.

Roles:

* Task labels — two task names, card labels mapped onto them.
* Metric specs — each metric's task, direction, and default bar.
* Metric substitution — swap a metric that fits the other task.
* Name lookup — aliases, spec, task, direction for a name.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Role: task labels ------------------------------------------------------------

# Set by the profiler on the card; read from there.
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASKS: tuple[str, ...] = (TASK_CLASSIFICATION, TASK_REGRESSION)

MAXIMIZE = "maximize"
MINIMIZE = "minimize"

# Card's finer task labels mapped onto the two above.
CARD_TASKS: dict[str, str] = {
    "binary_classification": TASK_CLASSIFICATION,
    "multiclass_classification": TASK_CLASSIFICATION,
    "regression": TASK_REGRESSION,
}


# --- Role: metric specs -----------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """What every consumer needs to know about one metric."""

    needs_proba: bool
    binary_only: bool
    # Best value is 1, so "remaining room" makes sense.
    bounded: bool
    # Default bar; ``None`` for unit-bound metrics
    fallback: float | None
    task: str = TASK_CLASSIFICATION
    # Property of the metric, never of the caller.
    direction: str = MAXIMIZE


METRICS: dict[str, MetricSpec] = {
    "f1": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "accuracy": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    # Lower default: harder number on unbalanced targets.
    "balanced_accuracy": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.80),
    "precision": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "recall": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "roc_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.90),
    # Chance level is the positive rate, not 0.5.
    "pr_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.70),
    # -- regression -------------------------------------------------------- #
    # Only regression metric with a unit-free bar.
    "r2": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=True,
        fallback=0.80,
        task=TASK_REGRESSION,
    ),
    # Target's units: no default bar; ``auto`` uses the baseline.
    "mae": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=False,
        fallback=None,
        task=TASK_REGRESSION,
        direction=MINIMIZE,
    ),
    "rmse": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=False,
        fallback=None,
        task=TASK_REGRESSION,
        direction=MINIMIZE,
    ),
}

# Same-sign sklearn names only
ALIASES: dict[str, str] = {
    "average_precision": "pr_auc",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
}

GOAL_METRICS: tuple[str, ...] = tuple(METRICS)

# --- Role: metric substitution ----------------------------------------------------

# Why ``r2``, not ``rmse``:
DEFAULT_METRICS: dict[str, str] = {
    TASK_CLASSIFICATION: "f1",
    TASK_REGRESSION: "r2",
}


def substitute_metric(task: str, metric: str) -> str | None:
    """substitute_metric | Role: default for ``task`` if ``metric`` fits the other task.

    Else ``None``. The caller must tell the user about the change.
    """
    wanted = task_of(metric)
    if wanted is None or task not in DEFAULT_METRICS or wanted == task:
        return None
    return DEFAULT_METRICS[task]


# --- Role: name lookup ------------------------------------------------------------


def canonical(name: str) -> str:
    """canonical | Role: alias to registry key; unknown names pass through."""
    return ALIASES.get(name, name)


def spec(name: str) -> MetricSpec | None:
    """spec | Role: the spec for ``name`` (aliases allowed), or ``None``."""
    return METRICS.get(canonical(name))


def metrics_for(task: str) -> tuple[str, ...]:
    """metrics_for | Role: registry keys for ``task``, in order; the scorer loops these."""
    return tuple(name for name, item in METRICS.items() if item.task == task)


def card_task(card: dict[str, object]) -> str | None:
    """card_task | Role: the card's task, or ``None`` (no claim, so no check)."""
    label = card.get("task")
    return CARD_TASKS.get(str(label)) if label else None


def task_of(name: str) -> str | None:
    """task_of | Role: the task ``name`` scores, or ``None`` if unknown."""
    found = spec(name)
    return None if found is None else found.task


def direction_of(name: str, default: str = MAXIMIZE) -> str:
    """direction_of | Role: which way is better for ``name``; unknown gets ``default``."""
    found = spec(name)
    return default if found is None else found.direction
