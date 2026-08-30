"""Model selection: the judgement is the LLM's, the commitment is code's.

The LLM picks from a registry of models this system can actually run; this module
then validates the identifier and clamps the hyperparameters. An unrunnable choice
is repaired here rather than discovered at training time.

There are two registries, one per task, because a model name is only runnable *against a
target*. ``logreg`` on a continuous column is not a bad choice, it is not a choice — the
executor refuses it outright (``scripts/train.py::build_estimator``). Repairing it here is
the same job this module already does for a misspelled id, one layer earlier: the menu the
LLM is shown, the schema its answer is validated against, and the fallback are all built
from the task's own registry, so the cross-task name should never be proposed at all. The
executor's refusal stays behind that as the backstop.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from typing import Any

from ..config import RunConfig
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION, card_task
from ..state import AutoMLState

# --------------------------------------------------------------------------- #
# Registry: the only models the executor knows how to build
# --------------------------------------------------------------------------- #

MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "logreg",
        "family": "linear",
        "cost": 1,
        "params": ["C", "max_iter", "class_weight"],
        "notes": "fast, low-capacity baseline; features are scaled automatically",
    },
    {
        "id": "decision_tree",
        "family": "tree",
        "cost": 1,
        "params": ["max_depth", "min_samples_leaf", "class_weight"],
        "notes": "interpretable, overfits easily",
    },
    {
        "id": "knn",
        "family": "instance",
        "cost": 2,
        "params": ["n_neighbors", "weights"],
        "notes": "no training cost, slow at prediction, sensitive to dimensionality",
    },
    {
        "id": "hist_gbdt",
        "family": "gbdt",
        "cost": 3,
        "params": [
            "max_iter",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "max_leaf_nodes",
            "l2_regularization",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of max_iter",
    },
    {
        "id": "random_forest",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf", "class_weight"],
        "notes": "robust, memory-hungry with many trees",
    },
    {
        "id": "extra_trees",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf", "class_weight"],
        "notes": "more randomised than random_forest, often better with noisy labels",
    },
    {
        "id": "gradient_boosting",
        "family": "gbdt",
        "cost": 5,
        "params": ["n_estimators", "learning_rate", "max_depth", "subsample"],
        "notes": "sequential and slow; prefer hist_gbdt unless small data",
    },
    {
        "id": "xgboost",
        "family": "gbdt",
        "cost": 4,
        "params": ["n_estimators", "learning_rate", "max_depth", "reg_lambda", "subsample"],
        "notes": "requires the xgboost package",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": ["hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size", "max_iter"],
        "notes": "highest capacity available here, and the most likely to hit the memory budget",
    },
    {
        "id": "svc",
        "family": "kernel",
        "cost": 5,
        "params": ["C", "kernel", "gamma", "class_weight"],
        "notes": "quadratic in rows; unusable above roughly 20k rows",
    },
)

# The regression side of the same list, deliberately using the *same ids* wherever the
# family exists on both — ``hist_gbdt`` builds a HistGradientBoostingRegressor here and a
# HistGradientBoostingClassifier there. One vocabulary means a plan reads the same way
# against either card and ``capabilities`` has one list to publish. Only where sklearn's
# estimator genuinely differs does the name: ``logreg``→``ridge``, ``svc``→``svr``.
#
# ``class_weight`` is absent from every entry, because there are no classes to weight. It is
# not merely useless here — it is the one lever the whole imbalance story is built on, so
# leaving it in the published params would invite a prescription the executor drops.
#
# ``ridge`` precedes ``linreg`` deliberately, even though they cost the same: every "cheapest
# in this family" lookup sorts by cost and ties break on this order, so the family named
# ``linear`` resolves to the regularised member. It is also the profiler's own baseline
# model, and it is the one of the two that a prescription can actually act on — a
# "strengthen the regularisation" verdict landing on ``linreg`` would have every knob it
# names dropped.
REGRESSION_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "ridge",
        "family": "linear",
        "cost": 1,
        "params": ["alpha", "max_iter"],
        "notes": "l2-regularised linear fit, and the profiler's own baseline model; "
        "features are scaled automatically. `alpha` is the regularisation strength, so it "
        "runs the *opposite* way from logreg's `C`",
    },
    {
        "id": "linreg",
        "family": "linear",
        "cost": 1,
        # Empty on purpose: OLS has a closed-form solution and nothing to tune. Listing a
        # knob it ignores would let the loop spend an iteration on a byte-identical run —
        # ``planning._signature`` reads this list precisely to stop that.
        "params": [],
        "notes": "unregularised least squares; the reference point, not a tuning target",
    },
    {
        "id": "elasticnet",
        "family": "linear",
        "cost": 2,
        "params": ["alpha", "l1_ratio", "max_iter"],
        "notes": "l1 plus l2; drives coefficients to zero, so it is the nearest thing here "
        "to feature selection, which the executor will not do separately",
    },
    {
        "id": "decision_tree",
        "family": "tree",
        "cost": 1,
        "params": ["max_depth", "min_samples_leaf"],
        "notes": "interpretable, overfits easily; predicts a constant per leaf",
    },
    {
        "id": "knn",
        "family": "instance",
        "cost": 2,
        "params": ["n_neighbors", "weights"],
        "notes": "no training cost, slow at prediction, sensitive to dimensionality",
    },
    {
        "id": "hist_gbdt",
        "family": "gbdt",
        "cost": 3,
        "params": [
            "max_iter",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "max_leaf_nodes",
            "l2_regularization",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of max_iter",
    },
    {
        "id": "random_forest",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf"],
        "notes": "robust, memory-hungry with many trees",
    },
    {
        "id": "extra_trees",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf"],
        "notes": "more randomised than random_forest, often better with a noisy target",
    },
    {
        "id": "gradient_boosting",
        "family": "gbdt",
        "cost": 5,
        "params": ["n_estimators", "learning_rate", "max_depth", "subsample"],
        "notes": "sequential and slow; prefer hist_gbdt unless small data",
    },
    {
        "id": "xgboost",
        "family": "gbdt",
        "cost": 4,
        "params": ["n_estimators", "learning_rate", "max_depth", "reg_lambda", "subsample"],
        "notes": "requires the xgboost package",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": ["hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size", "max_iter"],
        "notes": "highest capacity available here, and the most likely to hit the memory budget",
    },
    {
        "id": "svr",
        "family": "kernel",
        "cost": 5,
        "params": ["C", "kernel", "gamma", "epsilon"],
        "notes": "quadratic in rows; unusable above roughly 20k rows. `epsilon` is a "
        "tolerance in the target's own units, so its useful size depends on the target's scale",
    },
)

# One registry per task, and every lookup goes through :func:`registry` rather than
# touching a module-level name, so a new task is one entry here instead of a grep.
REGISTRIES: dict[str, tuple[dict[str, Any], ...]] = {
    TASK_CLASSIFICATION: MODEL_REGISTRY,
    TASK_REGRESSION: REGRESSION_REGISTRY,
}

# Both tasks have a ``hist_gbdt``, so the default needs no branch — which is one of the
# reasons the ids were kept the same.
DEFAULT_MODEL = "hist_gbdt"

# Clamps applied to whatever the LLM proposes: proposals are advice, not commands.
LIMITS: dict[str, tuple[float, float]] = {
    "max_iter": (1, 3000),
    "n_estimators": (1, 2000),
    "learning_rate": (1e-4, 1.0),
    "learning_rate_init": (1e-5, 1.0),
    "max_depth": (1, 64),
    "max_leaf_nodes": (2, 1024),
    "min_samples_leaf": (1, 1000),
    "l2_regularization": (0.0, 100.0),
    "reg_lambda": (0.0, 100.0),
    # Two estimators' worth of range: MLP's weight decay lives at the low end, and ridge's
    # and elasticnet's regularisation strength wants the high one. The old ceiling of 10 was
    # MLP's alone, and it silently clamped a proposed `alpha: 100` on ridge down to a tenth
    # of what was asked for — a stronger-regularisation attempt that was never actually run.
    "alpha": (1e-8, 1000.0),
    # elasticnet's l1/l2 mix. sklearn refuses anything outside [0, 1] outright, so an
    # unclamped proposal costs the whole attempt.
    "l1_ratio": (0.0, 1.0),
    # SVR's tolerance band. Only the sign is guarded: the useful magnitude is in the
    # target's units, which nothing here knows, and a ceiling in those units would be the
    # same guess this repo refuses to make about a default `mae` bar.
    "epsilon": (0.0, 1e9),
    "C": (1e-4, 1e4),
    "batch_size": (1, 8192),
    "n_neighbors": (1, 200),
    "subsample": (0.05, 1.0),
    "train_subsample": (0.01, 1.0),
    "gamma": (1e-6, 100.0),
}

ALLOWED_STRINGS = {"class_weight", "weights", "kernel", "precision", "solver", "penalty"}

# ``class_weight`` is the one key that may also arrive as a mapping, because it is the
# only imbalance lever the executor has and ``'balanced'`` is not its optimum: on the
# MIMIC sample ``'balanced'`` pins the ratio at the class frequency (8.08) where the best
# measured ratio was 10 (balanced_accuracy 0.7849 -> 0.7895). Until this branch existed a
# proposed map was dropped here *silently* — it never reached the executor, so it could
# not even show up in ``dropped_hyperparams``.
WEIGHT_MAP_KEYS = {"class_weight"}
# Weights are per class code, so a plausible map is small. The bounds only exist to stop
# a runaway value (1e9 makes every metric degenerate) from reaching the estimator.
WEIGHT_RANGE = (1e-3, 1000.0)
MAX_WEIGHT_ENTRIES = 32


def registry(task: str | None = None) -> tuple[dict[str, Any], ...]:
    """The registry for ``task``, defaulting to classification.

    ``None`` — a card that declares no task, or one whose label this build does not know —
    reads as classification rather than raising, on the same terms as
    :func:`automl_agent.scoring.metrics.card_task`: this is the menu, not a check, and a card too
    old to say what it is should still get a runnable one.
    """
    return REGISTRIES.get(task or TASK_CLASSIFICATION, MODEL_REGISTRY)


def selection_schema(task: str | None = None) -> dict[str, Any]:
    """The response schema for this task, so the enum cannot offer the other task's models.

    Built per call rather than once at import: the enum *is* the strongest guard here — a
    name outside it is rejected by the API layer before it costs an attempt — and a single
    module-level schema could only ever encode one task's menu.
    """
    return {
        "type": "object",
        "properties": {
            "model": {"type": "string", "enum": [entry["id"] for entry in registry(task)]},
            "hyperparams": {"type": "object", "additionalProperties": True},
            "rationale": {"type": "string"},
        },
        "required": ["model", "hyperparams", "rationale"],
        "additionalProperties": False,
    }


def available_models(task: str | None = None) -> list[dict[str, Any]]:
    """Registry entries for ``task`` whose backing package is importable right now."""
    entries = []
    for entry in registry(task):
        if entry["id"] == "xgboost" and importlib.util.find_spec("xgboost") is None:
            continue
        entries.append(entry)
    return entries


def available_ids(task: str | None = None) -> set[str]:
    return {entry["id"] for entry in available_models(task)}


def task_of_state(state: AutoMLState) -> str:
    """Which registry this run's card selects. The one place that reads it for both nodes."""
    return card_task(dict(state.get("dataset_card") or {})) or TASK_CLASSIFICATION


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


def model_selection(state: AutoMLState, *, config: RunConfig) -> dict:
    """Commit to one runnable ``(model, hyperparams)`` pair."""
    plan = dict(state.get("plan") or {})
    task = task_of_state(state)
    variables = {
        "plan": plan,
        "dataset_card": state.get("dataset_card") or {},
        "available_models": available_models(task),
        "history": _history_digest(state),
        "critic": state.get("critic") or "(no critic verdict yet — this is the first attempt)",
    }

    iteration = int(state.get("iteration", 0) or 0) or None
    choice: dict[str, Any] | None = None
    if not config.use_llm:
        archive_prompt_only(
            config,
            f"model_selection_iter{iteration or 0}",
            render_prompt("model_selection", variables),
        )
    else:
        try:
            choice = LLMClient(config).complete_json(
                "model_selection", variables, selection_schema(task), iteration=iteration
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [model_selection] LLM 호출 실패({exc}) — 계획의 후보 모델로 폴백합니다")

    if not choice:
        choice = fallback_selection(plan, task)

    model = normalise_model(choice.get("model"), plan, task)
    hyperparams = sanitise_hyperparams(
        {**(plan.get("hyperparams") or {}), **(choice.get("hyperparams") or {})}
    )
    return {"model": model, "hyperparams": hyperparams}


def fallback_selection(plan: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    """Deterministic stand-in: take the plan's first runnable candidate."""
    candidates = plan.get("candidate_models") or []
    if isinstance(candidates, str):
        candidates = [candidates]
    ids = available_ids(task)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip().lower() in ids:
            return {
                "model": candidate.strip().lower(),
                "hyperparams": dict(plan.get("hyperparams") or {}),
                "rationale": "deterministic fallback: first runnable candidate from the plan",
            }
    return {
        "model": DEFAULT_MODEL,
        "hyperparams": dict(plan.get("hyperparams") or {}),
        "rationale": "deterministic fallback: no runnable candidate in the plan, using the default",
    }


def normalise_model(proposed: Any, plan: dict[str, Any], task: str | None = None) -> str:
    """Repair an unrunnable identifier instead of letting training fail on it.

    "Unrunnable" now includes *runnable against the other task*: ``logreg`` is a real id
    that this run cannot use, and the repair is the same one a typo gets — the plan's family
    if it has one, the default otherwise. Which family the name belonged to does not survive
    the swap, and it should not: ``linear`` means ridge here and logreg there.
    """
    ids = available_ids(task)
    if isinstance(proposed, str) and proposed.strip().lower() in ids:
        return proposed.strip().lower()

    # Try the plan's family, then fall back to the default.
    family = str(plan.get("model_family") or "").strip().lower()
    if family:
        for entry in sorted(available_models(task), key=lambda item: int(item["cost"])):
            if entry["family"] == family:
                return str(entry["id"])
    return DEFAULT_MODEL


def _weight_map(value: Mapping[Any, Any]) -> dict[int, float] | None:
    """A ``{class code: weight}`` mapping, or ``None`` when the proposal is not one.

    All or nothing: half a weight map is a different request from the one that was made,
    and sklearn wants exactly one weight per class anyway. Codes may arrive as strings
    because the plan comes back as JSON, and they leave as strings again when the config
    is written — ``scripts/train.py::weight_map`` is the other half of this repair.
    """
    if not value or len(value) > MAX_WEIGHT_ENTRIES:
        return None
    low, high = WEIGHT_RANGE
    clean: dict[int, float] = {}
    for raw_code, raw_weight in value.items():
        if isinstance(raw_code, bool) or isinstance(raw_weight, bool):
            return None
        if not isinstance(raw_weight, (int, float)):
            return None
        try:
            code = int(str(raw_code).strip())
        except (TypeError, ValueError):
            return None
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0 or code < 0 or code in clean:
            return None
        clean[code] = min(max(weight, low), high)
    return clean


def sanitise_hyperparams(raw: Any) -> dict[str, Any]:
    """Keep values the executor can use, clamped to sane ranges; drop the rest."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, (int, float)):
            low, high = LIMITS.get(key, (-1e12, 1e12))
            clamped = min(max(float(value), low), high)
            cleaned[key] = int(clamped) if isinstance(value, int) and float(clamped).is_integer() else clamped
        elif isinstance(value, str):
            if key in ALLOWED_STRINGS:
                cleaned[key] = value
        elif isinstance(value, dict) and key in WEIGHT_MAP_KEYS:
            mapping = _weight_map(value)
            if mapping is not None:
                cleaned[key] = mapping
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            cleaned[key] = value  # e.g. hidden_layer_sizes
        elif value is None and key in ALLOWED_STRINGS:
            cleaned[key] = None
    return cleaned


def _history_digest(state: AutoMLState) -> list[dict[str, Any]]:
    """Compact history for prompts: enough to avoid repeats, small enough to read."""
    return [digest_attempt(attempt) for attempt in state.get("history") or []]


def digest_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """One attempt as the prompts see it.

    ``report`` digests its own final attempt through this too. It used to keep a second
    copy of this shape, and the copies drifted: the last iteration's
    ``dropped_hyperparams`` reached the write-up only when a Critic happened to quote it.
    """
    result = attempt.get("result") or {}
    metrics = result.get("metrics") or {}
    plan = attempt.get("plan") or {}
    return {
        "iteration": attempt.get("iteration"),
        "model": attempt.get("model"),
        # Already the applied set (``build_attempt`` uses ``effective_hyperparams``); the
        # two keys below say what was asked for and did not happen, which is what stops a
        # reader from crediting the score to it.
        "hyperparams": attempt.get("hyperparams"),
        "dropped_hyperparams": result.get("dropped_hyperparams") or [],
        # The pipeline the executor built, not the block the plan asked for — the two differ
        # whenever the requested strategy was downgraded. Absent for runs recorded before
        # the executor reported it, which is why the report prompt has a fallback.
        "preprocessing": result.get("applied_preprocessing") or {},
        # Beside ``preprocessing`` for the same reason: it is what the executor did rather than
        # what was asked for. This attempt's ``hyperparams`` may say ``validation_fraction:
        # 0.15`` and nothing else said what that cost, so a Planner comparing two attempts on
        # score alone read a 15% smaller training set as a fair tie. Empty when the estimator
        # held no rows back, and absent for runs recorded before the executor reported it.
        "internal_validation": result.get("internal_validation") or {},
        "unsupported_claims": plan.get("unsupported_claims") or [],
        "status": result.get("status"),
        "error_type": result.get("error_type"),
        # ``result["paired"]`` is deliberately not here. This digest feeds the Planner and the
        # report, and the paired verdict is a steering instrument the Critic's ledger renders
        # against a baseline it tracks itself (``nodes/critic.py``). Copied into a digest it
        # would arrive without that baseline, and in the report it would sit next to the
        # held-back score as though the two answered the same question — see
        # ``nodes/holdout.py`` for why they do not.
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "train_time_sec": result.get("train_time_sec"),
        "critic": attempt.get("critic"),
    }
