"""Model selection: the LLM makes the choice, code makes it final.

Roles:

* Registry — models the executor can build, one list per task.
* Selection node — pick one runnable model and hyperparameters.
* Hyperparameter cleaning — keep usable values, clamped to sane ranges.
* History digest — the short attempt summary prompts see.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from typing import Any

from ..config import RunConfig
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.intervals import as_number
from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION, card_task
from ..state import AutoMLState, state_int

# --- Role: registry -------------------------------------------------------------------

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
            "class_weight",
            "early_stopping",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of "
        "max_iter. `early_stopping` defaults to 'auto', which is *on* above 10k rows, so "
        "saying nothing about it does not mean fitting on every train row",
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
        "params": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_lambda",
            "subsample",
            "scale_pos_weight",
            "early_stopping_rounds",
        ],
        "notes": "requires the xgboost package. `early_stopping_rounds` needs nothing else "
        "from the plan — the executor holds its own stopping slice back and supplies the eval "
        "set; `eval_set` and `callbacks` are the two keys it refuses. `scale_pos_weight` is "
        "**binary only** — xgboost ignores it on a multiclass objective, so above two classes "
        "the executor drops it and the attempt records that in `dropped_hyperparams`",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": [
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "early_stopping",
        ],
        "notes": "highest capacity available here, and the most likely to hit the memory "
        "budget. Its `early_stopping` is a boolean only — the `'auto'` spelling belongs to "
        "hist_gbdt, and the executor drops it here rather than letting `fit` raise on it",
    },
    {
        "id": "svc",
        "family": "kernel",
        "cost": 5,
        "params": ["C", "kernel", "gamma", "class_weight"],
        "notes": "quadratic in rows; unusable above roughly 20k rows",
    },
)

# Same ids; no class_weight; ridge first
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
        # Empty on purpose: OLS has nothing to tune.
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
            "early_stopping",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of "
        "max_iter. `early_stopping` defaults to 'auto', which is *on* above 10k rows, so "
        "saying nothing about it does not mean fitting on every train row",
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
        "params": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_lambda",
            "subsample",
            "early_stopping_rounds",
        ],
        "notes": "requires the xgboost package. `early_stopping_rounds` needs nothing else "
        "from the plan — the executor holds its own stopping slice back and supplies the eval "
        "set; `eval_set` and `callbacks` are the two keys it refuses",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": [
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "early_stopping",
        ],
        "notes": "highest capacity available here, and the most likely to hit the memory "
        "budget. Its `early_stopping` is a boolean only — the `'auto'` spelling belongs to "
        "hist_gbdt, and the executor drops it here rather than letting `fit` raise on it",
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

# Read only through :func:`registry`.
REGISTRIES: dict[str, tuple[dict[str, Any], ...]] = {
    TASK_CLASSIFICATION: MODEL_REGISTRY,
    TASK_REGRESSION: REGRESSION_REGISTRY,
}

# Both tasks have ``hist_gbdt``, so the default needs no branch.
DEFAULT_MODEL = "hist_gbdt"

# Clamp ranges; an LLM proposal is advice, not an order.
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
    # Shared by MLP, ridge, elasticnet
    "alpha": (1e-8, 1000.0),
    # sklearn rejects values outside [0, 1].
    "l1_ratio": (0.0, 1.0),
    # Only the sign is guarded
    "epsilon": (0.0, 1e9),
    "C": (1e-4, 1e4),
    # Floor above 0: 0 drops the positive class.
    "scale_pos_weight": (1e-3, 1000.0),
    # 0 means no early stopping.
    "early_stopping_rounds": (0, 1000),
    # Capped at half of train
    "validation_fraction": (0.01, 0.5),
    "batch_size": (1, 8192),
    "n_neighbors": (1, 200),
    "subsample": (0.05, 1.0),
    "train_subsample": (0.01, 1.0),
    "gamma": (1e-6, 100.0),
}

# Keys that may carry any string value.
ALLOWED_STRINGS = {"class_weight", "weights", "kernel", "precision", "solver", "penalty"}

# Keys with a fixed set of strings
ALLOWED_STRING_VALUES: dict[str, frozenset[str]] = {"early_stopping": frozenset({"auto"})}

# Keys that may also come as a map
WEIGHT_MAP_KEYS = {"class_weight"}
# Stops runaway weights like 1e9 that break metrics.
WEIGHT_RANGE = (1e-3, 1000.0)
MAX_WEIGHT_ENTRIES = 32


def registry(task: str | None = None) -> tuple[dict[str, Any], ...]:
    """Return the registry for ``task``; unknown or ``None`` means classification, never raises."""
    return REGISTRIES.get(task or TASK_CLASSIFICATION, MODEL_REGISTRY)


def selection_schema(task: str | None = None) -> dict[str, Any]:
    """Build the response schema per task, so the enum offers only this task's models."""
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
    """List runnable registry entries for ``task``; ``xgboost`` only if installed."""
    entries = []
    for entry in registry(task):
        if entry["id"] == "xgboost" and importlib.util.find_spec("xgboost") is None:
            continue
        entries.append(entry)
    return entries


def available_ids(task: str | None = None) -> set[str]:
    """Return the ids of :func:`available_models` for ``task``."""
    return {entry["id"] for entry in available_models(task)}


def task_of_state(state: AutoMLState) -> str:
    """Return the card's task; classification when unset. Planning reads it here too."""
    return card_task(dict(state.get("dataset_card") or {})) or TASK_CLASSIFICATION


# --- Role: selection node -------------------------------------------------------------


def model_selection(state: AutoMLState, *, config: RunConfig) -> dict:
    """Settle on one runnable model and hyperparams.

    ``selection_source`` is ``llm``, ``fallback``, or ``rules``.
    """
    plan = dict(state.get("plan") or {})
    task = task_of_state(state)
    variables = {
        "plan": plan,
        "dataset_card": state.get("dataset_card") or {},
        "available_models": available_models(task),
        "history": _history_digest(state),
        "critic": state.get("critic") or "(no critic verdict yet — this is the first attempt)",
    }

    iteration = state_int(state, "iteration") or None
    choice: dict[str, Any] | None = None
    if not config.use_llm:
        archive_prompt_only(
            config,
            f"model_selection_iter{iteration or 0}",
            render_prompt("model_selection", variables),
        )
    else:
        try:
            choice = LLMClient(config, proposer=True).complete_json(
                "model_selection", variables, selection_schema(task), iteration=iteration
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [model_selection] LLM 호출 실패({exc}) — 계획의 후보 모델로 폴백합니다")

    proposed = bool(choice)
    if not choice:
        choice = fallback_selection(plan, task)

    model = normalise_model(choice.get("model"), plan, task)
    hyperparams = sanitise_hyperparams(
        {**(plan.get("hyperparams") or {}), **(choice.get("hyperparams") or {})}
    )
    # Same values as ``plan["source"]`` (``nodes/planning.py``).
    source = "llm" if proposed else ("fallback" if config.use_llm else "rules")
    return {"model": model, "hyperparams": hyperparams, "selection_source": source}


def fallback_selection(plan: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    """Pick the plan's first runnable candidate, else :data:`DEFAULT_MODEL`, with no LLM."""
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
    """Fix a model id that cannot run (other-task ids too), instead of failing in training.

    Uses the cheapest model in the plan's family, else the default.
    """
    ids = available_ids(task)
    if isinstance(proposed, str) and proposed.strip().lower() in ids:
        return proposed.strip().lower()

    family = str(plan.get("model_family") or "").strip().lower()
    if family:
        for entry in sorted(available_models(task), key=lambda item: int(item["cost"])):
            if entry["family"] == family:
                return str(entry["id"])
    return DEFAULT_MODEL


# --- Role: hyperparameter cleaning ----------------------------------------------------


def _weight_map(value: Mapping[Any, Any]) -> dict[int, float] | None:
    """_weight_map | Hyperparameter cleaning: read a ``{class code: weight}`` map, all or nothing."""
    if not value or len(value) > MAX_WEIGHT_ENTRIES:
        return None
    low, high = WEIGHT_RANGE
    clean: dict[int, float] = {}
    for raw_code, raw_weight in value.items():
        # ``True`` would pass ``int()`` as class 1.
        if isinstance(raw_code, bool):
            return None
        weight = as_number(raw_weight)
        if weight is None:
            return None
        try:
            code = int(str(raw_code).strip())
        except (TypeError, ValueError):
            return None
        if not math.isfinite(weight) or weight <= 0 or code < 0 or code in clean:
            return None
        clean[code] = min(max(weight, low), high)
    return clean


def _string_allowed(key: str, value: str) -> bool:
    """_string_allowed | Hyperparameter cleaning: check whether this key may carry this string."""
    if key in ALLOWED_STRING_VALUES:
        return value in ALLOWED_STRING_VALUES[key]
    return key in ALLOWED_STRINGS


def _clamped_number(key: str, value: int | float) -> int | float:
    """_clamped_number | Hyperparameter cleaning: clamp into :data:`LIMITS`, keeping ints as ints."""
    low, high = LIMITS.get(key, (-1e12, 1e12))
    clamped = min(max(float(value), low), high)
    if isinstance(value, int) and float(clamped).is_integer():
        return int(clamped)
    return clamped


def sanitise_hyperparams(raw: Any) -> dict[str, Any]:
    """Keep only usable values, clamped to sane ranges; ``{}`` if ``raw`` is not a dict."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, (int, float)):
            cleaned[key] = _clamped_number(key, value)
        elif isinstance(value, str) and _string_allowed(key, value):
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


# --- Role: history digest -------------------------------------------------------------


def _history_digest(state: AutoMLState) -> list[dict[str, Any]]:
    """_history_digest | History digest: short history for prompts, one digest per attempt."""
    return [digest_attempt(attempt) for attempt in state.get("history") or []]


def digest_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise one attempt as prompts see it; ``report`` uses it too"""
    result = attempt.get("result") or {}
    metrics = result.get("metrics") or {}
    plan = attempt.get("plan") or {}
    return {
        "iteration": attempt.get("iteration"),
        "model": attempt.get("model"),
        # Already the applied set, not the proposed one.
        "hyperparams": attempt.get("hyperparams"),
        "dropped_hyperparams": result.get("dropped_hyperparams") or [],
        # What the executor built, not what was planned.
        "preprocessing": result.get("applied_preprocessing") or {},
        # Ordered steps of a spec; empty for flags
        "applied_pipeline": result.get("applied_pipeline") or [],
        # Rows the estimator held back; empty if none.
        "internal_validation": result.get("internal_validation") or {},
        "unsupported_claims": plan.get("unsupported_claims") or [],
        # Can change per iteration
        "plan_source": plan.get("source") or "",
        "selection_source": attempt.get("selection_source") or "",
        "status": result.get("status"),
        "error_type": result.get("error_type"),
        # ``paired`` left out on purpose
        "metrics": {k: v for k, v in metrics.items() if as_number(v) is not None},
        "train_time_sec": result.get("train_time_sec"),
        "critic": attempt.get("critic"),
    }
