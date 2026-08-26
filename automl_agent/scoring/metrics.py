"""The metric registry: one declaration of every metric a run can target.

Three places used to keep their own list — the goal thresholds in :mod:`automl_agent.scoring.goal`,
the scores :mod:`automl_agent.scripts.profile` measures, and the ones
:mod:`automl_agent.scripts.train` emits. When those lists disagreed the failure was
silent rather than loud: ``--metric balanced_accuracy`` derived a threshold and the
profiler measured a baseline for it, but the trainer never produced the key, so
``goal_met`` read a missing score and stayed False for every iteration. This module is
the single list; the other three read from it.

Deliberately dependency-free — names and properties only, no sklearn, no imports from
the rest of the package. The orchestrator process must never import pandas or sklearn,
and the fixed scripts are run as *files* by the nodes rather than imported, so anything
both sides share has to be importable from either direction with nothing else attached.

Each metric declares the ``task`` it belongs to, because a run's task is a property of
the *target column* rather than of the caller: a continuous target cannot be scored with
``f1``, and asking for one is a setup error worth refusing before the loop starts rather
than a run that reports "목표 미달" five times. ``direction`` is declared here for the same
reason — minimizing ``f1`` or maximizing ``mae`` is not a preference, it is a mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

# What the target column is. The profiler decides which one a dataset is (from the target's
# dtype and cardinality) and records it in the card; every consumer reads it from there.
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASKS: tuple[str, ...] = (TASK_CLASSIFICATION, TASK_REGRESSION)

MAXIMIZE = "maximize"
MINIMIZE = "minimize"

# The card's own ``task`` labels, which are deliberately finer than the two above: a plan
# prompt reads "binary_classification" and knows there is one positive class, which
# "classification" alone does not say. This maps the card's vocabulary onto the metric's,
# so the two can be compared without either having to give up its own resolution.
CARD_TASKS: dict[str, str] = {
    "binary_classification": TASK_CLASSIFICATION,
    "multiclass_classification": TASK_CLASSIFICATION,
    "regression": TASK_REGRESSION,
}


@dataclass(frozen=True)
class MetricSpec:
    """What every consumer needs to know about one metric.

    ``needs_proba`` and ``binary_only`` are the two legitimate reasons a metric can be
    missing from a result even though the run asked for it. Declaring them here is what
    lets the trainer and the profiler skip the same metrics for the same reasons, instead
    of each deciding on its own and drifting apart.
    """

    needs_proba: bool
    binary_only: bool
    # Has a best value of 1 to measure remaining headroom against, so "close a fraction of
    # what is left" means something. True for the classification metrics on [0, 1] and for
    # ``r2``, which is unbounded *below* but tops out at 1 — the end the margin counts from.
    bounded: bool
    # The bar ``fixed`` mode uses when the caller names no number, and ``auto``'s last
    # resort when the card carries no baseline to derive from. ``None`` for a metric in the
    # target's own units: there is no portable default error for a column whose scale is
    # unknown, and inventing one (``mae <= 0.85``?) would be a bar about the units rather
    # than about the model. Those metrics require ``--threshold`` or a measured baseline.
    fallback: float | None
    task: str = TASK_CLASSIFICATION
    # Which way is better. A property of the metric, never of the caller.
    direction: str = MAXIMIZE


METRICS: dict[str, MetricSpec] = {
    "f1": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "accuracy": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    # Lower default than accuracy: on an imbalanced target it is the harder number, and a
    # 0.85 bar there asks for far more than 0.85 accuracy does.
    "balanced_accuracy": MetricSpec(
        needs_proba=False, binary_only=False, bounded=True, fallback=0.80
    ),
    "precision": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "recall": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "roc_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.90),
    # Area under precision-recall. Chance level is the positive rate rather than 0.5, so a
    # 0.90-style default would be unreachable on a rare-positive target.
    "pr_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.70),
    # -- regression ---------------------------------------------------------- #
    # The one regression metric with a portable bar: 1 is a perfect fit and 0 is what
    # predicting the mean scores, on every dataset and in every unit. So ``auto``'s
    # headroom margin and ``fixed``'s default both mean the same thing here that they mean
    # for a classification metric.
    "r2": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=True,
        fallback=0.80,
        task=TASK_REGRESSION,
    ),
    # In the target's units, so no default bar exists — see ``fallback`` above. ``auto``
    # mode derives one from the measured baseline, which is in those same units.
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

# sklearn's names for the same numbers, accepted wherever a metric name is read and emitted
# alongside the canonical key in results so a card or report written under either name stays
# readable. Only names for the *same* quantity with the *same* sign belong here:
# ``neg_mean_absolute_error`` is deliberately absent, because a sign-flipped alias would
# make ``direction`` a lie for half the readers of one number.
ALIASES: dict[str, str] = {
    "average_precision": "pr_auc",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
}

# What ``--metric`` accepts, and the set a config is validated against.
GOAL_METRICS: tuple[str, ...] = tuple(METRICS)

# What to score a task with when the caller named a metric belonging to the *other* task.
# Used by :func:`substitute_metric`; see there for why the answer is a substitution rather
# than a refusal.
#
# ``r2`` rather than ``rmse`` for regression, deliberately: it is the one regression metric
# with a portable bar (``fallback=0.80``), so the substituted run has a goal it can be judged
# against. Substituting ``rmse`` would trade one setup error for another — the bar would come
# back ``None`` and the run would be refused for a *different* reason, which is a worse
# message about a decision nobody made.
DEFAULT_METRICS: dict[str, str] = {
    TASK_CLASSIFICATION: "f1",
    TASK_REGRESSION: "r2",
}


def substitute_metric(task: str, metric: str) -> str | None:
    """The metric to score ``task`` with instead of ``metric``, or ``None`` to keep it.

    ``None`` covers every case where there is nothing to fix and every case where this
    module cannot tell: the metric already belongs to the task, the task label is one this
    build does not know, or the name is not a registry metric (``RunConfig`` refuses those).
    A substitution is only ever proposed when both facts are known and they disagree.

    The disagreement is reachable in two ways, and both used to cost a whole run.
    ``--metric rmse`` against a classification target passes ``RunConfig`` — it validates the
    name against the registry, and it is built before profiling, so at that point nobody
    knows what the target column is. And a card declaring ``regression`` over a column that
    reads as classification puts the same mismatch inside a run whose flags were all
    consistent. Either way the trainer computes the *other* task's metrics, so the goal
    metric is simply absent from every result: ``goal_met`` reads a missing score, every
    iteration is recorded as "목표 미달", and the loop spends its whole budget finding out.

    Substituting is not the only defensible answer — refusing at profiling time is the other
    one, and it is what this function replaced. It loses to substitution on the case that
    actually happens: a mistyped ``--metric`` on a long run is a typo, and a typo should cost
    a line of output rather than the run. What makes it safe is that the substitution is
    never quiet — the caller who acts on it is expected to say so (see
    :mod:`automl_agent.nodes.profiling`), because a run judged by a metric nobody asked for
    is only honest if the swap is on screen.
    """
    wanted = task_of(metric)
    if wanted is None or task not in DEFAULT_METRICS or wanted == task:
        return None
    return DEFAULT_METRICS[task]


def canonical(name: str) -> str:
    """Resolve an alias to the registry key. Unknown names pass through unchanged.

    Passing through rather than raising keeps the derivation path tolerant: a
    hand-written card naming a metric this harness cannot compute still gets a
    fallback threshold, and the rejection happens once, in ``RunConfig``.
    """
    return ALIASES.get(name, name)


def spec(name: str) -> MetricSpec | None:
    """The spec for ``name`` (alias-aware), or None if this harness has no such metric."""
    return METRICS.get(canonical(name))


def metrics_for(task: str) -> tuple[str, ...]:
    """The registry keys belonging to ``task``, in registry order.

    What the scorers iterate. Scoring a regression attempt with the classification list
    would not merely produce nothing — ``f1_score`` on continuous values raises, and the
    result would read as a metric this prediction "cannot support" rather than as a metric
    that never applied.
    """
    return tuple(name for name, item in METRICS.items() if item.task == task)


def card_task(card: dict[str, object]) -> str | None:
    """Which of :data:`TASKS` a dataset card is about, or ``None`` if it does not say.

    ``None`` covers both a card written before the field existed and one whose label this
    build does not know, and callers treat both the same way: no claim, so no check. A
    guessed task would be worse than an absent one — it would refuse a metric on the basis
    of a label nobody wrote.
    """
    label = card.get("task")
    return CARD_TASKS.get(str(label)) if label else None


def task_of(name: str) -> str | None:
    """Which task ``name`` scores (alias-aware), or ``None`` for an unknown metric."""
    found = spec(name)
    return None if found is None else found.task


def direction_of(name: str, default: str = MAXIMIZE) -> str:
    """Which way is better for ``name``. Unknown metrics keep ``default``.

    Read rather than asked. ``--direction minimize --metric f1`` used to be an accepted
    combination that inverted the whole run's notion of progress: ``is_better`` kept the
    *worst* attempt as ``best`` and ``goal_met`` fired on any score below the bar.
    """
    found = spec(name)
    return default if found is None else found.direction
