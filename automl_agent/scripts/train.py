"""Fixed training script, run by ``nodes/training.py`` as a subprocess.

Roles:

* Data — load or make the data, pick the goal metric.
* Model registry — map model and hyperparameter names to estimators.
* Preprocessing — impute, scale, and missing-column steps around the model.
* Fitting — fit the estimator, holding rows back for early stopping.
* Scoring — every metric the predictions support, with bootstrap intervals.
* Decision rule — choose and save the binary probability cut.
* Artifacts — save and load model, schema, predictions, decision rule.
* Training run — one full attempt: split, fit, score, save.
* Result plumbing — turn outcomes and errors into ``result.json``.
* Command line — parse arguments, run train or score-only mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Run by path, so add repo root; hence the E402s.
if __package__ in (None, ""):  # pragma: no cover - only when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.config import (  # noqa: E402
    DECISION_FILENAME,
    MODEL_FILENAME,
    PREDICTIONS_FILENAME,
    SCHEMA_FILENAME,
    file_size_text,
)
from automl_agent.dataset.features import (  # noqa: E402
    append_missing_count,
    append_missing_indicator,
    build_schema,
    describe_drift,
    describe_encoding,
    encode_features,
    encode_with_schema,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    APPENDING_STEPS as PIPELINE_APPENDING_STEPS,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    STEP_IMPUTE as PIPELINE_STEP_IMPUTE,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    STEP_SCALE as PIPELINE_STEP_SCALE,
)
from automl_agent.dataset.pipeline import build_steps  # noqa: E402
from automl_agent.dataset.source import load_frame  # noqa: E402
from automl_agent.dataset.targets import (  # noqa: E402
    DEFAULT_TARGET_MISSING_POLICY,
    detect_task,
    encode_target,
    target_classes,
)
from automl_agent.scoring.calibration import (  # noqa: E402
    measure as calibration_measure,
)
from automl_agent.scoring.intervals import (  # noqa: E402
    DEFAULT_RESAMPLES,
    PAIRED_KEY,
    PAIRED_MEASURED,
    PAIRED_SKIPPED,
    as_number,
    bootstrap_interval,
    describe_interval,
    describe_paired,
    paired_delta,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES as METRIC_ALIASES,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    DEFAULT_METRICS,
    METRICS,
    MINIMIZE,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    canonical,
    direction_of,
    substitute_metric,
)
from automl_agent.scoring.ranking import (  # noqa: E402
    best_cut_ceiling,
    ks_statistic,
)
from automl_agent.scoring.splits import (  # noqa: E402
    describe_protocol,
    protocol,
    split_three_way,
    val_fingerprint,
)
from automl_agent.threads import (  # noqa: E402
    describe_thread_state,
    thread_state,
    thread_state_changed,
)

LOG_TAIL_CHARS = 4000

# Bigger saved models get a warning line, not a refusal.
LARGE_MODEL_BYTES = 100 * 1024**2

# Also scored on train, to tell underfit from overfit.
TRAIN_METRICS: dict[str, tuple[str, ...]] = {
    TASK_CLASSIFICATION: ("f1", "accuracy"),
    TASK_REGRESSION: ("r2", "mae"),
}

# Never goal metrics; a test ties these to ``capabilities._LEVER_AXES``.
CUT_DIAGNOSTICS: tuple[str, str] = ("balanced_accuracy_at_best_cut", "balanced_accuracy_cut_headroom")


class LogBuffer:
    """Collect progress lines; the tail goes into result.json.

    ``echo=False`` (predict.py) keeps lines quiet to place them later.
    """

    def __init__(self, echo: bool = True) -> None:
        self.lines: list[str] = []
        self._echo = echo

    def write(self, message: str) -> None:
        self.lines.append(message)
        if self._echo:
            print(message, flush=True)

    def tail(self, limit: int = LOG_TAIL_CHARS) -> str:
        return "\n".join(self.lines)[-limit:]


# --- Role: data ------------------------------------------------------------------------


def goal_metric(cfg: dict[str, Any], task: str, log: LogBuffer | None = None) -> str:
    """Pick the goal metric: the configured one, or the task default.

    A metric that does not fit the column's task is swapped and logged
    """
    configured = canonical(str(cfg.get("metric") or "")) or DEFAULT_METRICS[task]
    swap = substitute_metric(task, configured)
    if swap is None:
        return configured if configured in METRICS else DEFAULT_METRICS[task]
    if log is not None:
        log.write(
            f"metric {configured} is not defined for a {task} target: steering by {swap} "
            f"instead, and recording {swap} in the schema so a labelled batch is scored on "
            "the same number"
        )
    return swap


def load_data(
    cfg: dict[str, Any], log: LogBuffer, schema: dict[str, Any] | None = None
) -> tuple[Any, Any, int, Any, str, dict[str, Any] | None]:
    """Load ``data.path`` (encoded to ``schema`` if given), or make synthetic data.

    Returns ``(X, y, n_classes, groups, task, schema)``; ValueError on bad columns.
    """
    import numpy as np

    data = dict(cfg.get("data") or {})
    path = data.get("path")
    seed = int(cfg.get("seed", 42))
    group_column = data.get("group_column")
    declared_task = str(cfg.get("task") or TASK_CLASSIFICATION)

    if path:
        target = str(data.get("target_column") or "target")
        frame = load_frame(path, table=data.get("table"), query=data.get("query"))
        if target not in frame.columns:
            raise ValueError(f"target_column {target!r} not found in {path}")
        y_series = frame[target]
        group_series = None
        drop = [target]
        if group_column:
            group_column = str(group_column)
            if group_column not in frame.columns:
                raise ValueError(f"group_column {group_column!r} not found in {path}")
            if group_column == target:
                raise ValueError(f"group_column {group_column!r} is the target column")
            group_series = frame[group_column]
            drop.append(group_column)
        # Same encoding as the profiler, so scores compare.
        raw_features = frame.drop(columns=drop)
        if schema is None:
            features, encoding = encode_features(raw_features)
            log.write(describe_encoding(encoding))
            fitted_schema: dict[str, Any] | None = build_schema(raw_features)
        else:
            features, drift = encode_with_schema(raw_features, schema)
            for line in describe_drift(drift):
                log.write(line)
            fitted_schema = dict(schema)
        if features.shape[1] == 0:
            raise ValueError("no usable feature columns remain after dropping the target")
        # Card's missing-label policy, so rows match the baseline.
        policy = str((cfg.get("target_missing") or {}).get("policy") or DEFAULT_TARGET_MISSING_POLICY)
        task = detect_task(y_series)
        if task != declared_task:
            # Column wins; a mismatch means the file changed.
            log.write(
                f"target column reads as {task}, but the config declares {declared_task}; "
                "following the column"
            )
        codes, keep, n_missing = encode_target(y_series, policy, task)
        if n_missing:
            features = features[keep]
            if group_series is not None:
                group_series = group_series[keep]
            log.write(f"dropped {n_missing} rows with a missing target (policy={policy})")
        log.write(f"loaded {path}: {features.shape[0]} rows x {features.shape[1]} encoded features")
        groups = None
        if group_series is not None:
            # Text, so 1 and 1.0 match; NaN ids share a group.
            groups = group_series.astype("string").fillna("<missing>").to_numpy()
            log.write(f"grouped by {group_column}: {len(set(groups.tolist()))} distinct groups")
        y_codes = codes.to_numpy()
        x_arr = np.asarray(features.to_numpy(), dtype="float64")
        goal = goal_metric(cfg, task, log)
        if fitted_schema is not None:
            # What ``predict`` needs that the estimator does not hold.
            fitted_schema.update(
                {
                    "task": task,
                    "target_column": target,
                    "group_column": group_column,
                    "classes": target_classes(y_series, task),
                    "n_features": int(x_arr.shape[1]),
                    "seed": seed,
                    # Later labelled batches are scored on this.
                    "metric": goal,
                }
            )
        return (
            x_arr,
            y_codes,
            0 if task == TASK_REGRESSION else int(len(set(y_codes.tolist()))),
            groups,
            task,
            fitted_schema,
        )

    syn = dict(data.get("synthetic") or {})
    n_samples = int(syn.get("n_samples", 5000))
    n_features = int(syn.get("n_features", 20))
    n_informative = int(syn.get("n_informative", max(2, n_features // 2)))
    n_informative = min(n_informative, n_features)

    if declared_task == TASK_REGRESSION:
        from sklearn.datasets import make_regression

        x_arr, y_arr = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            # Like ``flip_y``: keeps synthetic r2 below 1.0.
            noise=float(syn.get("noise", 10.0)),
            random_state=seed,
        )
        log.write(
            f"synthesised regression dataset from card: {n_samples} rows x {n_features} "
            f"features, noise={syn.get('noise', 10.0)}"
        )
        return x_arr, y_arr, 0, None, TASK_REGRESSION, None

    from sklearn.datasets import make_classification

    n_classes = int(syn.get("n_classes", 2))
    weights = syn.get("class_weights")
    x_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, min(n_features - n_informative, int(syn.get("n_redundant", 2)))),
        n_classes=n_classes,
        n_clusters_per_class=int(syn.get("n_clusters_per_class", 2)),
        weights=list(weights) if weights else None,
        class_sep=float(syn.get("class_sep", 0.8)),
        flip_y=float(syn.get("flip_y", 0.03)),
        random_state=seed,
    )
    log.write(
        f"synthesised dataset from card: {n_samples} rows x {n_features} features, "
        f"{n_classes} classes, class_sep={syn.get('class_sep', 0.8)}, flip_y={syn.get('flip_y', 0.03)}"
    )
    return x_arr, y_arr, n_classes, None, TASK_CLASSIFICATION, None


def guard_memory(x_arr: Any, cfg: dict[str, Any], log: LogBuffer) -> None:
    """Raise MemoryError when the estimated working set is over ``memory_limit_mb``.

    ``batch_size`` lowers the estimate, so "reduce batch size" advice works.
    """
    limit_mb = cfg.get("memory_limit_mb")
    if not limit_mb:
        return
    rows, cols = int(x_arr.shape[0]), int(x_arr.shape[1])
    hyperparams = dict(cfg.get("hyperparams") or {})
    batch = hyperparams.get("batch_size")
    working_rows = min(rows, int(batch)) if isinstance(batch, int) and batch > 0 else rows
    # Tree ensembles hold several copies of the working set.
    copies = 4 if cfg.get("model") in {"random_forest", "extra_trees", "gradient_boosting"} else 2
    if str(hyperparams.get("precision", "")).lower() in {"fp16", "float16", "half"}:
        copies = max(1, copies // 2)
    estimate_mb = working_rows * cols * 8 * copies / (1024 * 1024)
    log.write(f"memory estimate {estimate_mb:.1f} MB vs budget {float(limit_mb):.1f} MB")
    if estimate_mb > float(limit_mb):
        raise MemoryError(
            f"out of memory: estimated {estimate_mb:.1f} MB exceeds budget {float(limit_mb):.1f} MB"
        )


# --- Role: model registry --------------------------------------------------------------

# Hyperparameter names an LLM may write, to sklearn names.
ALIASES: dict[str, dict[str, str]] = {
    "hist_gbdt": {"n_estimators": "max_iter", "reg_lambda": "l2_regularization"},
    "mlp": {"learning_rate": "learning_rate_init", "hidden_size": "hidden_layer_sizes"},
    "logreg": {"reg_strength": "C", "epochs": "max_iter"},
    "xgboost": {"lr": "learning_rate", "l2": "reg_lambda"},
    # alpha is a penalty; logreg's C is its inverse.
    "ridge": {"reg_strength": "alpha", "l2": "alpha", "lambda": "alpha"},
    "elasticnet": {"reg_strength": "alpha", "l1_ratio": "l1_ratio"},
}

# Models whose fit needs scaled features.
SCALE_SENSITIVE = {"logreg", "mlp", "svc", "knn", "ridge", "linreg", "elasticnet", "svr"}

# Only these may skip imputation
NATIVE_NAN = {"hist_gbdt", "xgboost"}

# Accepted by set_params but break fit here
UNSUPPORTED_BY_HARNESS: dict[str, tuple[str, ...]] = {
    "xgboost": ("eval_set", "callbacks"),
}

# Only these take a string ``early_stopping``
STRING_EARLY_STOPPING = frozenset({"hist_gbdt"})

# Dropped when there are more than two classes.
BINARY_ONLY_PARAMS = frozenset({"scale_pos_weight"})


def weight_map(value: dict[Any, Any], labels: Sequence[Any] | None) -> dict[int, float] | None:
    """Turn a proposed ``class_weight`` into ``{class code: weight}``, or ``None``.

    Needs one positive finite weight per class code; else ``None``
    """
    if not value:
        return None
    clean: dict[int, float] = {}
    for raw_code, raw_weight in value.items():
        if isinstance(raw_code, bool) or isinstance(raw_weight, bool):
            return None
        try:
            code = int(str(raw_code).strip())
            weight = float(raw_weight)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(weight) or weight <= 0 or code in clean:
            return None
        clean[code] = weight
    if labels is not None and set(clean) != {int(label) for label in labels}:
        return None
    return clean


# Per task, so wrong-task names fail
MODEL_ALIASES: dict[str, dict[str, str]] = {
    TASK_CLASSIFICATION: {
        "logistic_regression": "logreg",
        "logisticregression": "logreg",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "hist_gradient_boosting": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
        "lightgbm": "hist_gbdt",  # not installed; the closest tested model
        "xgb": "xgboost",
        "neural_net": "mlp",
        "mlp_classifier": "mlp",
    },
    TASK_REGRESSION: {
        "linear_regression": "linreg",
        "linearregression": "linreg",
        "ols": "linreg",
        "ridge_regression": "ridge",
        "ridgeregression": "ridge",
        "elastic_net": "elasticnet",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "hist_gradient_boosting": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
        "lightgbm": "hist_gbdt",
        "xgb": "xgboost",
        "neural_net": "mlp",
        "mlp_regressor": "mlp",
        "mlpregressor": "mlp",
        "svm": "svr",
    },
}


def _classifier(key: str, seed: int) -> Any:
    """_classifier | Model registry: the classifier for this key, or ``None``."""
    if key == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed)
    if key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=seed)
    if key == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=seed)
    if key == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier()
    if key == "svc":
        from sklearn.svm import SVC

        return SVC(random_state=seed)
    if key == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(max_iter=400, random_state=seed)
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - only without the extra installed
            raise ValueError(f"model 'xgboost' is unavailable: {exc}") from exc

        return XGBClassifier(
            n_estimators=300, tree_method="hist", eval_metric="logloss", random_state=seed
        )
    return None


def _regressor(key: str, seed: int) -> Any:
    """_regressor | Model registry: the regressor for this key, or ``None``."""
    if key == "linreg":
        from sklearn.linear_model import LinearRegression

        # No ``random_state``: a closed-form fit has nothing to seed.
        return LinearRegression()
    if key == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed)
    if key == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(random_state=seed)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed)
    if key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed)
    if key == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(random_state=seed)
    if key == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor()
    if key == "svr":
        from sklearn.svm import SVR

        return SVR()
    if key == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(max_iter=400, random_state=seed)
    if key == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - only without the extra installed
            raise ValueError(f"model 'xgboost' is unavailable: {exc}") from exc

        return XGBRegressor(n_estimators=300, tree_method="hist", random_state=seed)
    return None


def build_estimator(
    name: str,
    hyperparams: dict[str, Any],
    seed: int,
    log: LogBuffer,
    preprocessing: dict[str, Any] | None = None,
    labels: Sequence[Any] | None = None,
    task: str = TASK_CLASSIFICATION,
    declared: list[tuple[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Build the model with only the params it accepts; return ``(pipeline, applied, dropped)``.

    Bad params are dropped and logged; an unknown model raises ValueError.
    """
    key = resolve_model_key(name, task)
    model = _regressor(key, seed) if task == TASK_REGRESSION else _classifier(key, seed)
    if model is None:
        raise ValueError(f"unsupported model {name!r} for a {task} target")

    mapped: dict[str, Any] = {}
    dropped: list[str] = []
    accepted = model.get_params(deep=False)
    blocked = UNSUPPORTED_BY_HARNESS.get(key, ())
    for raw_key, value in hyperparams.items():
        param = ALIASES.get(key, {}).get(raw_key, raw_key)
        if param in blocked:
            dropped.append(raw_key)
            continue
        if param == "class_weight" and isinstance(value, dict):
            repaired = weight_map(value, labels)
            if repaired is None:
                log.write(f"dropped class_weight={value!r}: not one positive weight per class")
                dropped.append(raw_key)
                continue
            value = repaired
        if param == "early_stopping" and isinstance(value, str) and key not in STRING_EARLY_STOPPING:
            log.write(
                f"dropped early_stopping={value!r}: {key} takes this as a boolean only, and "
                "the string would raise inside fit"
            )
            dropped.append(raw_key)
            continue
        if param in BINARY_ONLY_PARAMS and labels is not None and len(set(labels)) > 2:
            log.write(
                f"dropped {raw_key}={value!r}: {len(set(labels))} classes, and this lever only "
                "acts on a binary objective"
            )
            dropped.append(raw_key)
            continue
        if param in accepted:
            if param == "hidden_layer_sizes" and isinstance(value, list):
                value = tuple(value)
            mapped[param] = value
        else:
            dropped.append(raw_key)
    if mapped:
        try:
            model.set_params(**mapped)
        except (ValueError, TypeError) as exc:
            log.write(f"rejected hyperparams {mapped}: {exc}; falling back to defaults")
            mapped = {}
    if dropped:
        log.write(f"dropped hyperparams not applicable to {key}: {sorted(dropped)}")
    log.write(f"estimator={key} applied_hyperparams={mapped}")
    built = (
        _wrap_preprocessing(key, model, preprocessing or {}, log)
        if declared is None
        else _wrap_declared(key, model, declared, log)
    )
    return built, mapped, sorted(dropped)


def resolve_model_key(name: str, task: str) -> str:
    """Return the registry key for a model name, after aliases."""
    key = name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(task, {}).get(key, key)


# --- Role: preprocessing ---------------------------------------------------------------

# Checked again here to guard hand-written configs.
IMPUTE_STRATEGIES = ("median", "mean", "most_frequent")
DEFAULT_IMPUTE = "median"
# Drops the imputer; honoured only for NATIVE_NAN models.
IMPUTE_NONE = "none"
# Imputer is a per-column ColumnTransformer from a spec.
PER_COLUMN_IMPUTE = "per_column"

# Run before the imputer; transformers live in features.py
MISSING_INDICATOR = "missing_indicator"
MISSING_COUNT = "missing_count"
# Other names for the two steps above, like ALIASES.
PREPROCESSING_ALIASES: dict[str, str] = {
    "add_missing_indicators": MISSING_INDICATOR,
    "add_missing_indicator": MISSING_INDICATOR,
    "missing_indicators": MISSING_INDICATOR,
    "add_indicator": MISSING_INDICATOR,
    "missing_counts": MISSING_COUNT,
    "n_missing": MISSING_COUNT,
}


def _wrap_preprocessing(
    key: str, model: Any, preprocessing: dict[str, Any], log: LogBuffer
) -> Any:
    """_wrap_preprocessing | Preprocessing: build the ``preprocessing`` pipeline"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    preprocessing = {
        PREPROCESSING_ALIASES.get(str(name), str(name)): value
        for name, value in preprocessing.items()
    }
    requested = preprocessing.get("impute")
    strategy = DEFAULT_IMPUTE if requested is None else str(requested)
    if strategy == IMPUTE_NONE and key not in NATIVE_NAN:
        log.write(
            f"impute={IMPUTE_NONE} needs a family that splits on NaN "
            f"({', '.join(sorted(NATIVE_NAN))}); {key} does not, using {DEFAULT_IMPUTE}"
        )
        strategy = DEFAULT_IMPUTE
    elif strategy not in (*IMPUTE_STRATEGIES, IMPUTE_NONE):
        log.write(f"unknown impute strategy {requested!r}; using {DEFAULT_IMPUTE}")
        strategy = DEFAULT_IMPUTE

    # An explicit ``scale`` decides; otherwise the estimator's family does.
    scale = preprocessing.get("scale")
    should_scale = bool(scale) if isinstance(scale, bool) else key in SCALE_SENSITIVE

    steps: list[tuple[str, Any]] = []
    # Before the imputer; indicator adds no NaN, so counts stay.
    indicate = bool(preprocessing.get(MISSING_INDICATOR))
    count = bool(preprocessing.get(MISSING_COUNT))
    if indicate:
        steps.append((MISSING_INDICATOR, FunctionTransformer(append_missing_indicator)))
    if count:
        steps.append((MISSING_COUNT, FunctionTransformer(append_missing_count)))
    if strategy != IMPUTE_NONE:
        steps.append(("impute", SimpleImputer(strategy=strategy)))
    if should_scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    log.write(
        f"preprocessing applied: impute={strategy} scale={should_scale} "
        f"{MISSING_INDICATOR}={indicate} {MISSING_COUNT}={count}"
    )
    return Pipeline(steps)


def declared_steps(
    cfg: dict[str, Any],
    schema: dict[str, Any] | None,
    width: int,
    task: str,
    log: LogBuffer,
) -> tuple[list[tuple[str, Any]] | None, list[str]]:
    """Resolve the ``pipeline`` spec into ``(steps, applied)`` sklearn steps.

    ``(None, [])`` means no spec: use the flags
    """
    spec = cfg.get("pipeline")
    if not spec:
        return None, []
    columns = [str(name) for name in (schema or {}).get("columns") or []]
    if not columns:
        columns = [f"x{position}" for position in range(int(width))]
        log.write(
            f"pipeline: this run encoded no columns (synthetic data), so steps can only "
            f"address all {len(columns)} of them or none by name"
        )
    key = resolve_model_key(str(cfg.get("model") or "hist_gbdt"), task)
    steps, _names, applied = build_steps(spec, columns, log, native_nan=key in NATIVE_NAN)
    return steps, applied


def base_step_name(name: str) -> str:
    """Strip a trailing repeat number: ``missing_count_2`` -> ``missing_count``."""
    return re.sub(r"_\d+$", "", name)


def describe_pipeline(estimator: Any, declared: list[str]) -> list[str]:
    """List the built pipeline's steps; unrequested ones read ``<name>(auto)``.

    Read from the estimator itself, not the config
    """
    steps = [name for name, _step in getattr(estimator, "steps", ()) if name != "model"]
    remaining = list(declared)
    echo: list[str] = []
    for name in steps:
        base = base_step_name(name)
        if remaining and remaining[0].split("(", 1)[0] == base:
            echo.append(remaining.pop(0))
        else:
            echo.append(f"{base}(auto)")
    return echo


def _wrap_declared(
    key: str, model: Any, declared: list[tuple[str, Any]], log: LogBuffer
) -> Any:
    """_wrap_declared | Preprocessing: spec pipeline plus needed imputer/scaler"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = list(declared)
    names = {base_step_name(name) for name, _step in steps}
    if PIPELINE_STEP_IMPUTE not in names and key not in NATIVE_NAN:
        # After appending steps, so NaN is marked first.
        at = 1 + max(
            (
                position
                for position, (name, _step) in enumerate(steps)
                if base_step_name(name) in PIPELINE_APPENDING_STEPS
            ),
            default=-1,
        )
        steps.insert(at, (PIPELINE_STEP_IMPUTE, SimpleImputer(strategy=DEFAULT_IMPUTE)))
        log.write(
            f"pipeline: no impute step declared and {key} cannot be fitted on a NaN; "
            f"inserted {DEFAULT_IMPUTE} imputation at position {at}"
        )
    if PIPELINE_STEP_SCALE not in names and key in SCALE_SENSITIVE:
        steps.append((PIPELINE_STEP_SCALE, StandardScaler()))
        log.write(f"pipeline: no scale step declared and {key} is scale-sensitive; appended one")
    steps.append(("model", model))
    log.write("pipeline steps: " + " -> ".join(name for name, _step in steps))
    return Pipeline(steps)


# --- Role: fitting ---------------------------------------------------------------------

# Train share held back; matches HistGradientBoosting's default.
EARLY_STOPPING_FRACTION = 0.1
# Fewer held-back rows: stop signal is noise, skip it.
MIN_EARLY_STOPPING_ROWS = 50


def held_back_indices(
    x: Any, y: Any, seed: int, stratify: bool, groups: Any, fraction: float
) -> tuple[Any, Any]:
    """Split *train* (not val) into ``(fit_idx, held_idx)``, keeping groups whole.

    Round counts and cuts are picked on held rows
    """
    import numpy as np

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        return next(iter(splitter.split(x, y, groups=groups)))

    from sklearn.model_selection import train_test_split

    return train_test_split(
        np.arange(len(y)),
        test_size=fraction,
        random_state=seed,
        stratify=y if stratify else None,
    )


def _early_stopping_split(
    x: Any, y: Any, seed: int, stratify: bool, groups: Any
) -> tuple[Any, Any, Any, Any]:
    """_early_stopping_split | Fitting: split train into ``(x_fit, x_stop, y_fit, y_stop)``."""
    fit_idx, stop_idx = held_back_indices(
        x, y, seed, stratify, groups, EARLY_STOPPING_FRACTION
    )
    return x[fit_idx], x[stop_idx], y[fit_idx], y[stop_idx]


def fit_estimator(
    pipeline: Any,
    x: Any,
    y: Any,
    seed: int,
    log: LogBuffer,
    *,
    stratify: bool = True,
    groups: Any = None,
) -> dict[str, Any]:
    """Fit the pipeline, making the eval set itself for ``early_stopping_rounds``.

    Returns held-back rows like :func:`describe_internal_validation`, or ``{}``.
    """
    final = pipeline.steps[-1][1]
    rounds = getattr(final, "early_stopping_rounds", None)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        if isinstance(rounds, int) and not isinstance(rounds, bool):
            # Log it: applied_hyperparams still shows this number.
            log.write(
                f"early_stopping_rounds={rounds} not applied: a round count of {rounds} is not "
                "a stop, so the fit sees every train row"
            )
            # xgboost raises on any non-zero count without eval set.
            final.set_params(early_stopping_rounds=None)
        pipeline.fit(x, y)
        return {}

    x_fit, x_stop, y_fit, y_stop = _early_stopping_split(x, y, seed, stratify, groups)
    if len(y_stop) < MIN_EARLY_STOPPING_ROWS:
        # Too few rows is noise; full n_estimators is honest.
        log.write(
            f"early_stopping_rounds={rounds} not applied: the stopping slice would hold "
            f"{len(y_stop)} rows, under the {MIN_EARLY_STOPPING_ROWS} needed"
        )
        final.set_params(early_stopping_rounds=None)
        pipeline.fit(x, y)
        return {}

    head = pipeline.steps[:-1]
    if head:
        from sklearn.pipeline import Pipeline

        pre = Pipeline(list(head))
        x_fit_t = pre.fit_transform(x_fit, y_fit)
        x_stop_t = pre.transform(x_stop)
    else:
        x_fit_t, x_stop_t = x_fit, x_stop

    final.fit(x_fit_t, y_fit, eval_set=[(x_stop_t, y_stop)], verbose=False)
    best = getattr(final, "best_iteration", None)
    reached = f", best_iteration={best}" if isinstance(best, int) else ""
    log.write(
        f"early_stopping_rounds={rounds}: fitted on {len(y_fit)} rows, stopped against "
        f"{len(y_stop)} rows held out of train{reached}"
    )
    held_back: dict[str, Any] = {
        "held_out_rows": len(y_stop),
        "fit_rows": len(y_fit),
        "validation_fraction": EARLY_STOPPING_FRACTION,
    }
    if isinstance(best, int) and not isinstance(best, bool):
        # best_iteration counts from 0; this counts rounds.
        held_back["stopped_at_iter"] = best + 1
    cap = getattr(final, "n_estimators", None)
    if isinstance(cap, int) and not isinstance(cap, bool):
        held_back["max_iter"] = cap
    return held_back


def describe_preprocessing(estimator: Any) -> dict[str, Any]:
    """Describe the preprocessing the built pipeline really has, or ``{}``.

    Read from the object; the config may differ
    """
    steps = getattr(estimator, "named_steps", None)
    if not steps:
        return {}
    imputer = steps.get("impute")
    return {
        # ``per_column``: a ColumnTransformer has no single strategy.
        "impute": IMPUTE_NONE
        if imputer is None
        else str(getattr(imputer, "strategy", "") or PER_COLUMN_IMPUTE),
        "scale": "scale" in steps,
        # Always reported, like ``scale``, so "off" reads as false instead of missing.
        MISSING_INDICATOR: MISSING_INDICATOR in steps,
        MISSING_COUNT: MISSING_COUNT in steps,
    }


def describe_internal_validation(estimator: Any, n_train: int) -> dict[str, Any]:
    """Describe train rows the estimator's early stopping held back, or ``{}``.

    Counts use ``train_test_split``, as sklearn does
    """
    from sklearn.model_selection import train_test_split

    fraction = getattr(estimator, "validation_fraction", None)
    if fraction is None or not _held_rows_back(estimator):
        return {}
    held_out = len(train_test_split(list(range(n_train)), test_size=fraction)[1])
    described: dict[str, Any] = {
        "held_out_rows": held_out,
        "fit_rows": n_train - held_out,
        "validation_fraction": fraction,
    }
    # stopped_at_iter == max_iter means the cap was hit.
    stopped = getattr(estimator, "n_iter_", None)
    if stopped is None:
        # ``gradient_boosting`` names the same count ``n_estimators_``.
        stopped = getattr(estimator, "n_estimators_", None)
    cap = getattr(estimator, "max_iter", None)
    if cap is None:
        cap = getattr(estimator, "n_estimators", None)
    if isinstance(stopped, int) and not isinstance(stopped, bool):
        described["stopped_at_iter"] = stopped
    if isinstance(cap, int) and not isinstance(cap, bool):
        described["max_iter"] = cap
    return described


def _held_rows_back(estimator: Any) -> bool:
    """_held_rows_back | Fitting: whether the estimator held back train rows"""
    for attribute in ("validation_score_", "validation_scores_"):
        if hasattr(estimator, attribute):
            scores = getattr(estimator, attribute)
            return scores is not None and len(scores) > 0
    return getattr(estimator, "n_iter_no_change", None) is not None


# --- Role: scoring ---------------------------------------------------------------------


def scorers(
    y_true: Any, pred: Any, proba: Any, average: str, task: str = TASK_CLASSIFICATION
) -> dict[str, Any]:
    """Return one lazy thunk per registry metric.

    ``scripts/profile.py`` shares this, so bar and scores match
    """
    if task == TASK_REGRESSION:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "r2": lambda: r2_score(y_true, pred),
            "mae": lambda: mean_absolute_error(y_true, pred),
            # Works on all sklearn versions, old and new.
            "rmse": lambda: float(mean_squared_error(y_true, pred)) ** 0.5,
        }

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "f1": lambda: f1_score(y_true, pred, average=average, zero_division=0),
        "accuracy": lambda: accuracy_score(y_true, pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, pred),
        "precision": lambda: precision_score(y_true, pred, average=average, zero_division=0),
        "recall": lambda: recall_score(y_true, pred, average=average, zero_division=0),
        "roc_auc": lambda: roc_auc_score(y_true, proba),
        "pr_auc": lambda: average_precision_score(y_true, proba),
    }


def score_split(
    names: list[str],
    y_true: Any,
    pred: Any,
    proba: Any,
    average: str,
    log: LogBuffer,
    task: str = TASK_CLASSIFICATION,
) -> dict[str, float]:
    """Score ``names`` on one split, skipping what these predictions cannot support.

    Skips are logged, except metrics of another task
    """
    thunks = scorers(y_true, pred, proba, average, task)
    scores: dict[str, float] = {}
    for name in names:
        spec = METRICS.get(name)
        if spec is None:  # pragma: no cover - RunConfig validation prevents this
            continue
        if spec.task != task:
            # Not logged: undefined here, not missing.
            continue
        if spec.binary_only and average != "binary":
            log.write(f"{name} skipped: defined for binary targets only (average={average})")
            continue
        if spec.needs_proba and proba is None:
            continue
        try:
            value = float(thunks[name]())
        except (ValueError, AttributeError) as exc:
            log.write(f"{name} skipped: {exc}")
            continue
        if not math.isfinite(value):
            # NaN is not a score; leave it out
            log.write(f"{name} skipped: not defined on these rows (returned {value})")
            continue
        scores[name] = value
    return scores


def specificity(y_true: Any, pred: Any) -> float:
    """Return the true negative rate; ``0.0`` with no negatives.

    A diagnostic only, never a goal
    """
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, pred, pos_label=0, zero_division=0))


def bootstrap_resamples(cfg: dict[str, Any]) -> int:
    """Return bootstrap resamples from config; ``0`` turns intervals off.

    No CLI flag on purpose; bad values use the default
    """
    raw = as_number(dict(cfg.get("bootstrap") or {}).get("resamples", DEFAULT_RESAMPLES))
    return DEFAULT_RESAMPLES if raw is None else max(0, int(raw))


def _proba(model: Any, x_arr: Any, n_classes: int, log: LogBuffer) -> Any:
    """_proba | Scoring: positive-class probabilities, or ``None`` when not available."""
    if n_classes != 2 or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(x_arr)[:, 1]
    except (ValueError, AttributeError, IndexError) as exc:
        log.write(f"predict_proba unavailable: {exc}")
        return None


# --- Role: decision rule ---------------------------------------------------------------
#
# Cut picked on held-back train rows, saved beside model

# Pick the cut on held-back rows.
DECISION_TUNED = "tuned"

# Train share held back for the cut
CUT_FRACTION = 0.2
# Fewer held-back rows make the pick noise.
MIN_CUT_ROWS = 200

# Quantiles plus even grid plus 0.5
CUT_QUANTILES = 99
CUT_GRID = 49


def label_at_cut(proba: Any, cut: float) -> Any:
    """Turn probabilities into 0/1 labels; ``proba >= cut`` is 1.

    Ties at 0.5 differ from sklearn ``predict``
    """
    import numpy as np

    return (np.asarray(proba) >= float(cut)).astype(int)


def requested_cut(cfg: dict[str, Any], log: LogBuffer) -> float | str | None:
    """Read ``decision.threshold``: a cut in (0, 1), ``"tuned"``, or ``None``.

    Bad values are logged and ignored
    """
    raw = (cfg.get("decision") or {}).get("threshold")
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() == DECISION_TUNED:
            return DECISION_TUNED
        log.write(
            f"decision.threshold={raw!r} ignored: expected a number in (0, 1) "
            f'or "{DECISION_TUNED}"'
        )
        return None
    cut = as_number(raw)
    if cut is None or not 0.0 < cut < 1.0:
        log.write(
            f"decision.threshold={raw!r} ignored: outside (0, 1), which labels every row "
            "the same way"
        )
        return None
    return round(cut, 6)


def tune_threshold(
    y_held: Any, proba: Any, metric: str, average: str, task: str, log: LogBuffer
) -> float | None:
    """Pick the best cut for the goal metric on held-back rows, or ``None``.

    Ties go to the middle of the plateau
    """
    if proba is None:
        log.write("threshold tuning skipped: this model gives no positive-class probabilities")
        return None
    import numpy as np

    values = np.asarray(proba)
    grid = np.concatenate(
        [
            np.quantile(values, np.linspace(0.01, 0.99, CUT_QUANTILES)),
            np.linspace(0.02, 0.98, CUT_GRID),
            [0.5],
        ]
    )
    candidates = sorted({round(float(c), 6) for c in grid if 0.0 < float(c) < 1.0})
    if not candidates:
        log.write("threshold tuning skipped: the held-back probabilities leave no cut inside (0, 1)")
        return None
    thunk_key = canonical(metric)
    scored: list[tuple[float, float]] = []
    for cut in candidates:
        try:
            score = float(
                scorers(y_held, label_at_cut(proba, cut), proba, average, task)[thunk_key]()
            )
        except (ValueError, AttributeError, KeyError) as exc:
            log.write(f"threshold tuning skipped: {metric} not scorable on the held-back rows ({exc})")
            return None
        if math.isfinite(score):
            scored.append((score, cut))
    if not scored:
        log.write(f"threshold tuning skipped: {metric} is not defined on the held-back rows")
        return None
    # Keeps this right if a minimised metric appears.
    sign = -1.0 if direction_of(thunk_key) == MINIMIZE else 1.0
    best_value = max(sign * score for score, _cut in scored)
    plateau = [cut for score, cut in scored if sign * score == best_value]
    if len(plateau) == len(scored):
        log.write(
            f"threshold tuning skipped: {metric} is the same at every one of {len(scored)} "
            "candidate cuts — it is computed from the probabilities, not from the labels, so no "
            "cut changes it"
        )
        return None
    cut = plateau[len(plateau) // 2]
    spread = (
        f", indistinguishable over {len(plateau)} cuts from {plateau[0]} to {plateau[-1]}"
        if len(plateau) > 1
        else ""
    )
    log.write(
        f"decision threshold tuned on the held-back rows: {cut} "
        f"({metric} {sign * best_value:.6f} over {len(scored)} candidate cuts{spread}) — "
        "these rows were in neither the fit nor the reported split"
    )
    return cut


def save_decision(rule: dict[str, Any], path: Path, log: LogBuffer) -> str | None:
    """Save the applied decision rule beside the model; path or ``None``.

    Never fatal; failures are logged
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rule, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.write(f"decision rule not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"decision rule saved to {path.name}: {json.dumps(rule, ensure_ascii=False)}")
    return str(path)


def load_decision(path: Path, log: LogBuffer) -> float | None:
    """Load the saved cut, or ``None`` for the default 0.5 rule.

    Bad files are logged, not raised
    """
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.write(f"{path.name} unusable ({exc}); scoring at the default rule")
        return None
    # ``as_number`` checks and narrows to float.
    raw = loaded.get("threshold") if isinstance(loaded, dict) else None
    cut = as_number(raw)
    if cut is None:
        log.write(f"{path.name} holds no numeric threshold; scoring at the default rule")
        return None
    if not 0.0 < cut < 1.0:
        log.write(f"{path.name} holds threshold={cut}, outside (0, 1); scoring at the default rule")
        return None
    log.write(f"applying the decision rule saved with this model: threshold={cut}")
    return cut


# --- Role: scoring (a full split) ------------------------------------------------------


def evaluate_split(
    model: Any,
    x_eval: Any,
    y_eval: Any,
    n_classes: int,
    average: str,
    log: LogBuffer,
    *,
    task: str = TASK_CLASSIFICATION,
    interval_metric: str | None = None,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
    pred: Any = None,
    proba: Any = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Score one split on every supported metric, plus diagnostics.

    Shared by val and test scoring, so both use the same code.
    """
    if pred is None:
        pred = model.predict(x_eval)
        proba = _proba(model, x_eval, n_classes, log)
    if threshold is not None:
        if proba is None:
            # Not applied, so no ``applied_threshold``.
            log.write("the saved decision rule needs probabilities this model cannot give; default rule")
            threshold = None
        else:
            # Relabel from ``proba``; same result if already relabelled.
            pred = label_at_cut(proba, threshold)
    metrics = score_split(list(METRICS), y_eval, pred, proba, average, log, task)
    for alias, canonical_name in METRIC_ALIASES.items():
        if canonical_name in metrics:
            metrics[alias] = metrics[canonical_name]

    # The other half of balanced_accuracy
    if n_classes == 2:
        metrics["specificity"] = round(specificity(y_eval, pred), 6)

    # What a better cut could gain; diagnostic only.
    ceiling = best_cut_ceiling(ks_statistic(y_eval, proba))
    if ceiling is not None:
        best_cut, headroom = CUT_DIAGNOSTICS
        metrics[best_cut] = ceiling
        if "balanced_accuracy" in metrics:
            metrics[headroom] = round(ceiling - metrics["balanced_accuracy"], 6)

    # Absent means the default rule.
    if threshold is not None:
        metrics["applied_threshold"] = threshold

    # Calibration: diagnostic only, never a goal.
    if proba is not None:
        metrics.update(calibration_measure(y_eval, proba))

    # Bootstrap interval for the goal metric only.
    target = canonical(interval_metric) if interval_metric else None
    if target and target in metrics:

        def one(y_slice: Any, pred_slice: Any, proba_slice: Any) -> float:
            return float(scorers(y_slice, pred_slice, proba_slice, average, task)[str(target)]())

        interval = bootstrap_interval(
            one, y_eval, pred, proba, groups=groups, seed=seed, resamples=resamples
        )
        if interval is None:
            log.write(f"{target}: no confidence interval (split too small, or metric degenerate)")
        else:
            metrics.update(interval.flatten(target))
            log.write(
                describe_interval(target, metrics[target], interval.bounds)
                + f" — {interval.unit} 단위 재표집 {interval.resamples}회"
            )
    return metrics


# --- Role: artifacts -------------------------------------------------------------------


def save_model(model: Any, path: Path, log: LogBuffer) -> str | None:
    """Save the fitted pipeline with joblib; path or ``None``.

    Never fatal; the path stays private
    """
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except (OSError, ImportError, TypeError, ValueError) as exc:
        log.write(f"model not saved: {type(exc).__name__}: {exc}")
        return None
    size = path.stat().st_size
    line = f"model saved to {path.name} ({file_size_text(size)})"
    if size > LARGE_MODEL_BYTES:
        # Size comes from hyperparameters
        line += (
            f" — over {file_size_text(LARGE_MODEL_BYTES)}, driven by this iteration's "
            "hyperparameters (tree count and depth, mostly). The run keeps it; the disk is "
            "what bounds a long loop, not memory"
        )
    log.write(line)
    return str(path)




def save_schema(schema: dict[str, Any] | None, path: Path, log: LogBuffer) -> str | None:
    """Save the fitted feature encoding beside the model; path or ``None``.

    Never fatal; ``predict`` needs it; path stays private
    """
    if schema is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.write(f"feature schema not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"feature schema saved to {path.name} ({len(schema.get('columns') or ())} columns)")
    return str(path)


def load_schema(path: Path, log: LogBuffer) -> dict[str, Any] | None:
    """Load the saved feature schema; ``None`` if missing.

    A broken file raises ValueError
    """
    if not path.exists():
        log.write(f"no feature schema beside the model ({path.name}); deriving the encoding from the file")
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"feature schema {path} is not a JSON object")
    log.write(f"encoding from {path.name}: {len(loaded.get('columns') or ())} columns")
    return loaded


def save_predictions(
    path: Path, pred: Any, proba: Any, fingerprint: str, log: LogBuffer
) -> str | None:
    """Save val predictions as ``.npz`` for later paired comparison; path or ``None``.

    Never fatal; labels are left out on purpose
    """
    try:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "pred": np.asarray(pred),
            "fingerprint": np.asarray(str(fingerprint)),
            # Thread counts can change results too.
            "threads": np.asarray(json.dumps(thread_state(), ensure_ascii=False, sort_keys=True)),
        }
        if proba is not None:
            arrays["proba"] = np.asarray(proba)
        np.savez_compressed(path, **arrays)
    except (OSError, ImportError, TypeError, ValueError) as exc:
        log.write(f"val predictions not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"val predictions saved to {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    return str(path)


def paired_against_baseline(
    cfg: dict[str, Any],
    y_val: Any,
    pred: Any,
    proba: Any,
    fingerprint: str,
    *,
    metric: str,
    average: str,
    task: str,
    log: LogBuffer,
    groups: Any = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap the goal-metric difference against the best attempt so far.

    Always returns a block: measured, or skipped with a reason
    """
    declared = dict(cfg.get("paired_baseline") or {})
    iteration = declared.get("iteration")
    raw_path = declared.get("path")
    block: dict[str, Any] = {"metric": metric, "baseline_iteration": iteration}

    def skipped(reason: str, detail: str = "") -> dict[str, Any]:
        log.write(f"paired comparison skipped ({reason}){f': {detail}' if detail else ''}")
        return {**block, "status": PAIRED_SKIPPED, "reason": reason}

    if not raw_path or iteration is None:
        return skipped("no_baseline")

    import numpy as np

    path = Path(str(raw_path))
    try:
        # Loading a pickle would run code.
        with np.load(path, allow_pickle=False) as loaded:
            stored = str(loaded["fingerprint"])
            base_pred = np.asarray(loaded["pred"])
            base_proba = np.asarray(loaded["proba"]) if "proba" in loaded.files else None
            # Older files lack ``threads``; ``None`` means not recorded.
            base_threads = (
                json.loads(str(loaded["threads"])) if "threads" in loaded.files else None
            )
    except (OSError, ValueError, KeyError) as exc:
        return skipped("baseline_missing", f"{path.name}: {type(exc).__name__}: {exc}")

    if stored != fingerprint:
        # Same rows needed; checked, not assumed from seed.
        return skipped("split_changed", f"{stored[:8]} != {fingerprint[:8]}")

    # A thread change is reported, not fatal.
    current_threads = thread_state()
    changed = thread_state_changed(base_threads, current_threads)
    if changed is not None:
        block["threads_changed"] = changed
    if changed:
        log.write(
            f"baseline iteration {iteration} was fitted in a different thread state: "
            f"[{describe_thread_state(base_threads)}] -> [{describe_thread_state(current_threads)}]"
            " — part of the delta below is the environment, not the plan"
        )

    spec = METRICS.get(metric)
    if spec is not None and spec.needs_proba and (proba is None or base_proba is None):
        return skipped("degenerate", f"{metric} needs probabilities and one side has none")

    def one(y_slice: Any, pred_slice: Any, proba_slice: Any) -> float:
        return float(scorers(y_slice, pred_slice, proba_slice, average, task)[metric]())

    delta = paired_delta(
        one,
        y_val,
        (pred, proba),
        (base_pred, base_proba),
        direction=direction_of(metric),
        groups=groups,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
    )
    if delta is None:
        return skipped("degenerate", f"{metric}: no interval over the difference")
    measured = {
        **block,
        "status": PAIRED_MEASURED,
        "resamples": delta.resamples,
        "unit": delta.unit,
        **delta.flatten(),
    }
    log.write(f"{metric} vs iteration {iteration}: {describe_paired(measured)}")
    return measured


# --- Role: training run ----------------------------------------------------------------


@dataclass
class TrainingRun:
    """Everything one fit produced; returned by :func:`run_training`."""

    metrics: dict[str, float]
    applied: dict[str, Any]
    dropped: list[str]
    preprocessing: dict[str, Any]
    model_path: str | None
    paired: dict[str, Any]
    schema_path: str | None
    internal_validation: dict[str, Any]
    applied_pipeline: list[str] = field(default_factory=list)


@dataclass
class HeldBackCut:
    """A decision-cut request's outcome, and the train rows held back for it."""

    request: float | str | None
    asked: float | str | None
    declined: str | None
    rows: dict[str, Any]
    x_fit: Any
    y_fit: Any
    groups_fit: Any
    x_cut: Any
    y_cut: Any


def hold_back_cut(
    cfg: dict[str, Any],
    x_train: Any,
    y_train: Any,
    groups_train: Any,
    seed: int,
    *,
    task: str,
    n_classes: int,
    log: LogBuffer,
) -> HeldBackCut:
    """Hold back train rows for picking the cut, before the fit.

    Refusals are checked first, so no rows are lost
    """
    request = requested_cut(cfg, log)
    # Refusals below clear ``request``; keep the ask.
    asked = request
    regression = task == TASK_REGRESSION
    goal = goal_metric(cfg, task)
    declined: str | None = None
    if request is not None and (METRICS.get(goal) or METRICS["f1"]).needs_proba:
        # Probability-only metrics score the same at every cut.
        log.write(
            f"decision.threshold={request!r} ignored: the goal metric {goal} is computed from "
            "the probabilities, not from the labels, so no cut changes it"
        )
        request, declined = None, f"the goal metric {goal} is cut-invariant"
    if request is not None and (regression or n_classes != 2):
        # Regression has no cut; multiclass has no single cut.
        log.write(
            f"decision.threshold={request!r} ignored: "
            + (
                "the target is continuous, so there is no decision rule to move"
                if regression
                else f"the target has {n_classes} classes and a single cut is a binary rule"
            )
        )
        request = None
        declined = (
            "the target is continuous" if regression else f"the target has {n_classes} classes"
        )
    # Outcome when no slice is cut.
    kept = HeldBackCut(
        request=request,
        asked=asked,
        declined=declined,
        rows={},
        x_fit=x_train,
        y_fit=y_train,
        groups_fit=groups_train,
        x_cut=None,
        y_cut=None,
    )
    if request != DECISION_TUNED:
        return kept
    fit_idx, cut_idx = held_back_indices(
        x_train, y_train, seed, not regression, groups_train, CUT_FRACTION
    )
    if len(cut_idx) < MIN_CUT_ROWS:
        # Too few rows: turn tuning off.
        log.write(
            f"decision.threshold={DECISION_TUNED!r} not applied: the slice held back to "
            f"choose the cut would hold {len(cut_idx)} rows, under the {MIN_CUT_ROWS} needed"
        )
        kept.request = None
        kept.declined = (
            f"the slice to choose the cut on would hold {len(cut_idx)} rows, "
            f"under the {MIN_CUT_ROWS} needed"
        )
        return kept
    log.write(
        f"decision.threshold={DECISION_TUNED!r}: {len(cut_idx)} of train's rows are held "
        f"out of the fit to choose the cut on, leaving {len(fit_idx)} to fit — the price "
        "of the lever, and reported in internal_validation"
    )
    return HeldBackCut(
        request=request,
        asked=asked,
        declined=None,
        rows={"cut_held_out_rows": len(cut_idx), "cut_fraction": CUT_FRACTION},
        x_fit=x_train[fit_idx],
        y_fit=y_train[fit_idx],
        groups_fit=None if groups_train is None else groups_train[fit_idx],
        x_cut=x_train[cut_idx],
        y_cut=y_train[cut_idx],
    )


def held_back_by_estimator(model: Any, n_train: int, log: LogBuffer) -> dict[str, Any]:
    """Log and return train rows the estimator's early stopping held back.

    Read after the fit from the object
    """
    held = describe_internal_validation(
        (getattr(model, "named_steps", None) or {}).get("model"), n_train
    )
    if held:
        stopped = (
            f", iteration {held['stopped_at_iter']}/{held['max_iter']}에서 멈춤"
            if {"stopped_at_iter", "max_iter"} <= held.keys()
            else ""
        )
        log.write(
            f"early_stopping이 위 {n_train}행 중 {held['held_out_rows']}행을 자체 검증으로 "
            f"떼어 갔습니다 — 실제 학습은 {held['fit_rows']}행{stopped}"
        )
    return held


def record_train_val_gap(
    metrics: dict[str, float], target_metric: str, task: str, log: LogBuffer
) -> None:
    """Write ``metrics["train_val_gap"]`` in place when both scores exist.

    Positive always means val is worse
    """
    gap_metric = target_metric if f"train_{target_metric}" in metrics else TRAIN_METRICS[task][0]
    if gap_metric not in metrics or f"train_{gap_metric}" not in metrics:
        return
    # Direction-aware, so positive always means overfitting.
    train_value, val_value = metrics[f"train_{gap_metric}"], metrics[gap_metric]
    worse_by = (
        val_value - train_value
        if direction_of(gap_metric) == MINIMIZE
        else train_value - val_value
    )
    metrics["train_val_gap"] = round(worse_by, 6)
    log.write(f"train_val_gap measured on {gap_metric} ({direction_of(gap_metric)})")


def run_training(
    cfg: dict[str, Any],
    log: LogBuffer,
    model_out: Path | None = None,
    predictions_out: Path | None = None,
    schema_out: Path | None = None,
    decision_out: Path | None = None,
) -> TrainingRun:
    """Run one attempt: load, split, fit, score val, save artifacts.

    Never reads the test split. Raises MemoryError or ValueError.
    """
    seed = int(cfg.get("seed", 42))
    x_arr, y_arr, n_classes, groups, task, schema = load_data(cfg, log)
    guard_memory(x_arr, cfg, log)
    regression = task == TASK_REGRESSION

    group_column = (cfg.get("data") or {}).get("group_column")
    splits = split_three_way(x_arr, y_arr, seed, groups=groups, stratify=not regression)
    x_train, y_train = splits.x_train, splits.y_train
    x_val, y_val = splits.x_val, splits.y_val
    log.write(
        describe_protocol(protocol(seed, group_column, stratified=not regression))
        + f" {splits.sizes}"
    )

    # Must stay the same length as x_train.
    groups_train = splits.groups_train

    subsample = as_number((cfg.get("hyperparams") or {}).get("train_subsample"))
    if subsample is not None and 0 < subsample < 1:
        keep = max(50, int(len(x_train) * subsample))
        x_train, y_train = x_train[:keep], y_train[:keep]
        if groups_train is not None:
            groups_train = groups_train[:keep]
        log.write(f"train_subsample={subsample} -> {keep} rows")

    # Cut rows leave train before the fit.
    cut = hold_back_cut(cfg, x_train, y_train, groups_train, seed, task=task, n_classes=n_classes, log=log)
    request, cut_requested, cut_declined = cut.request, cut.asked, cut.declined
    x_cut, y_cut, cut_rows = cut.x_cut, cut.y_cut, cut.rows
    x_train, y_train, groups_train = cut.x_fit, cut.y_fit, cut.groups_fit

    # Needs the schema for column names; ``None`` uses flags.
    declared, applied_pipeline = declared_steps(cfg, schema, x_train.shape[1], task, log)

    model, applied, dropped = build_estimator(
        str(cfg.get("model") or "hist_gbdt"),
        dict(cfg.get("hyperparams") or {}),
        seed,
        log,
        dict(cfg.get("preprocessing") or {}),
        # Classes ``class_weight`` must cover; none for regression.
        labels=None if regression else sorted(set(y_train.tolist())),
        task=task,
        declared=declared,
    )
    log.write(f"fitting on {len(x_train)} rows, validating on {len(x_val)} rows")
    internal_validation = fit_estimator(
        model,
        x_train,
        y_train,
        seed,
        log,
        stratify=not regression,
        groups=groups_train,
    ) or held_back_by_estimator(model, len(x_train), log)
    model_path = save_model(model, model_out, log) if model_out is not None else None
    # After the fit, so it matches this estimator.
    schema_path = save_schema(schema, schema_out, log) if schema_out is not None else None

    # No log: ``load_data`` already logged any swap.
    target_metric = goal_metric(cfg, task)
    average = "binary" if n_classes == 2 else "macro"
    pred_train = model.predict(x_train)
    # Predict once; scored and saved from the same arrays.
    pred_val = model.predict(x_val)
    proba_val = _proba(model, x_val, n_classes, log)

    # Set the cut before scoring; float or ``None`` after.
    threshold: float | None
    if request == DECISION_TUNED:
        threshold = tune_threshold(
            y_cut, _proba(model, x_cut, n_classes, log), target_metric, average, task, log
        )
    else:
        threshold = as_number(request)
    proba_train = None
    if threshold is not None:
        proba_train = _proba(model, x_train, n_classes, log)
    if threshold is not None and (proba_train is None or proba_val is None):
        log.write(f"decision.threshold={threshold} not applied: this model gives no probabilities")
        threshold, cut_declined = None, "this model gives no probabilities"
    if request is not None and threshold is None and cut_declined is None:
        # The sweep found nothing; reason is logged.
        cut_declined = "the sweep found no cut worth applying"
    if threshold is not None:
        # Relabel both splits so one rule applies everywhere.
        pred_train = label_at_cut(proba_train, threshold)
        pred_val = label_at_cut(proba_val, threshold)
        if decision_out is not None:
            save_decision(
                {
                    "threshold": threshold,
                    # Tuned on held-back rows, or given by config.
                    "chosen_on": "held_back" if request == DECISION_TUNED else "config",
                    "metric": target_metric,
                    **cut_rows,
                },
                decision_out,
                log,
            )
    if cut_requested is not None:
        # Absent means no cut was asked for.
        cut_rows["cut_requested"] = cut_requested
        if cut_declined is not None:
            cut_rows["cut_declined"] = cut_declined
    if cut_rows:
        # Reported even without a cut: fewer rows were fitted.
        internal_validation = {**internal_validation, **cut_rows}
    applied_pipeline = (
        describe_pipeline(model, applied_pipeline) if declared is not None else []
    )
    metrics = evaluate_split(
        model,
        x_val,
        y_val,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=target_metric,
        # Val groups, since the interval is about val.
        groups=splits.groups_val,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
        pred=pred_val,
        proba=proba_val,
        threshold=threshold,
    )

    fingerprint = val_fingerprint(x_val, y_val)
    if predictions_out is not None:
        save_predictions(predictions_out, pred_val, proba_val, fingerprint, log)
    paired = paired_against_baseline(
        cfg,
        y_val,
        pred_val,
        proba_val,
        fingerprint,
        metric=target_metric,
        average=average,
        task=task,
        log=log,
        groups=splits.groups_val,
        seed=seed,
    )

    # Train scores, plus the goal metric for the gap.
    train_names = list(TRAIN_METRICS[task])
    if target_metric in METRICS and target_metric not in train_names:
        train_names.append(target_metric)
    train_proba = proba_train
    if train_proba is None and any(METRICS[name].needs_proba for name in train_names):
        train_proba = _proba(model, x_train, n_classes, log)
    for name, value in score_split(
        train_names, y_train, pred_train, train_proba, average, log, task
    ).items():
        metrics[f"train_{name}"] = value

    record_train_val_gap(metrics, target_metric, task, log)

    log.write("metrics: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return TrainingRun(
        metrics=metrics,
        applied=applied,
        dropped=dropped,
        preprocessing=describe_preprocessing(model),
        model_path=model_path,
        paired=paired,
        schema_path=schema_path,
        internal_validation=internal_validation,
        applied_pipeline=applied_pipeline,
    )


def score_saved_model(
    cfg: dict[str, Any], log: LogBuffer, model_path: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score the test split with a saved model; returns ``(metrics, preprocessing)``.

    Nothing is fitted; saved files beat the config
    """
    import joblib

    seed = int(cfg.get("seed", 42))
    schema = load_schema(model_path.parent / SCHEMA_FILENAME, log)
    threshold = load_decision(model_path.parent / DECISION_FILENAME, log)
    x_arr, y_arr, n_classes, groups, task, _schema = load_data(cfg, log, schema)
    splits = split_three_way(
        x_arr, y_arr, seed, groups=groups, stratify=task != TASK_REGRESSION
    )
    model = joblib.load(model_path)
    average = "binary" if n_classes == 2 else "macro"
    log.write(
        f"scoring the held-back test split: {len(splits.y_test)} rows, "
        f"model={model_path.name}"
    )
    metrics = evaluate_split(
        model,
        splits.x_test,
        splits.y_test,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=canonical(str(cfg.get("metric") or "f1")),
        groups=splits.groups_test,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
        threshold=threshold,
    )
    log.write("test metrics: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return metrics, describe_preprocessing(model)


# --- Role: result plumbing -------------------------------------------------------------


def classify_exception(exc: BaseException) -> str:
    """Map an exception to a coarse ``error_type`` for the Critic.

    One of ``oom``, ``data_issue``, ``unsupported_model``, ``exception``.
    """
    if isinstance(exc, MemoryError):
        return "oom"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(token in text for token in ("out of memory", "cuda", "cannot allocate", "alloc failed")):
        return "oom"
    if isinstance(exc, (FileNotFoundError, KeyError)) or "not found" in text:
        return "data_issue"
    if isinstance(exc, ValueError) and "unsupported model" in text:
        return "unsupported_model"
    if isinstance(exc, ValueError):
        return "data_issue"
    return "exception"


# Lists dropped non-finite values; private, not public.
NONFINITE_KEY = "nonfinite_dropped"


def drop_nonfinite(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """Drop NaN and inf; returns ``(clean value, dropped paths)``.

    Dict keys go; list items become ``None``
    """
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        dropped: list[str] = []
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            clean, gone = drop_nonfinite(item, here)
            dropped += gone
            if gone and clean is None and not isinstance(item, (dict, list)):
                continue
            kept[str(key)] = clean
        return kept, dropped
    if isinstance(value, list):
        kept_list: list[Any] = []
        dropped = []
        for index, item in enumerate(value):
            clean, gone = drop_nonfinite(item, f"{path}[{index}]")
            dropped += gone
            # Keep list positions, e.g. ``[low, high]``.
            kept_list.append(clean)
        return kept_list, dropped
    # Test only; return the raw value unchanged.
    number = as_number(value)
    if number is None or math.isfinite(number):
        return value, []
    return None, [path or "<root>"]


def write_result(
    out_path: Path,
    *,
    status: str,
    metrics: dict[str, Any],
    train_time_sec: float,
    error_type: str | None,
    log_tail: str,
    applied_hyperparams: dict[str, Any] | None = None,
    dropped_hyperparams: list[str] | None = None,
    applied_preprocessing: dict[str, Any] | None = None,
    applied_pipeline: list[str] | None = None,
    model_path: str | None = None,
    schema_path: str | None = None,
    paired: dict[str, Any] | None = None,
    internal_validation: dict[str, Any] | None = None,
    split: str = "val",
) -> None:
    """Write ``result.json`` for one attempt, success or failure.

    Always writes valid JSON, with a smaller fallback if needed.
    """
    payload = {
        "metrics": metrics,
        # Always present, so val and test never mix.
        "split": split,
        # Always present, even on failure.
        "applied_hyperparams": dict(applied_hyperparams or {}),
        "dropped_hyperparams": list(dropped_hyperparams or []),
        # What was really built, not what was asked.
        "applied_preprocessing": dict(applied_preprocessing or {}),
        "train_time_sec": round(float(train_time_sec), 3),
        "status": status,
        "error_type": error_type,
        "log_tail": log_tail,
        # Private on purpose
        "threads": thread_state(),
    }
    if applied_pipeline:
        # Left out on the flag path
        payload["applied_pipeline"] = list(applied_pipeline)
    if paired:
        # Kept even when skipped; not part of ``metrics``.
        payload[PAIRED_KEY] = paired
    if internal_validation:
        # Absent means no rows held back.
        payload["internal_validation"] = internal_validation
    if model_path:
        # Local only: a fitted model can hold data.
        payload["model_path"] = model_path
    if schema_path:
        # Local only: the schema holds cell values.
        payload["schema_path"] = schema_path
    payload, nonfinite = drop_nonfinite(payload)
    if nonfinite:
        payload[NONFINITE_KEY] = nonfinite
        # Also noted in the log tail, read first.
        payload["log_tail"] = (
            str(payload.get("log_tail") or "")
            + f"\nnon-finite values dropped from result.json: {', '.join(nonfinite)}"
        ).strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Checks the cleanup; failure here means a bug.
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        # Missing result.json would read as a crash.
        text = json.dumps(
            {
                "metrics": {},
                "split": split,
                "applied_hyperparams": {},
                "dropped_hyperparams": list(dropped_hyperparams or []),
                "applied_preprocessing": {},
                "train_time_sec": round(float(train_time_sec), 3),
                "status": status,
                "error_type": error_type,
                "log_tail": f"{log_tail}\nresult.json could not be serialised strictly: {exc}",
                "threads": thread_state(),
                NONFINITE_KEY: nonfinite or ["<unknown>"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    out_path.write_text(text, encoding="utf-8")


# --- Role: command line ----------------------------------------------------------------


def apply_simulation(cfg: dict[str, Any], log: LogBuffer) -> None:
    """Inject a planned failure from ``cfg["simulate"]`` for tests and drills.

    May sleep, raise MemoryError (``oom``), or exit with code 3 (``crash``).
    """
    sim = dict(cfg.get("simulate") or {})
    sleep_sec = float(sim.get("sleep_sec", 0) or 0)
    if sleep_sec > 0:
        log.write(f"simulate.sleep_sec={sleep_sec}: sleeping to trigger the orchestrator timeout")
        time.sleep(sleep_sec)
    failure = str(sim.get("fail") or "").lower()
    if failure == "crash":
        # Like a segfault: tests the orchestrator survives it.
        log.write("simulate.fail=crash: exiting hard without a result file")
        sys.stderr.write("simulated hard crash in training subprocess\n")
        sys.stderr.flush()
        os._exit(3)
    if failure == "oom":
        raise MemoryError("simulated out of memory during training")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--config``, ``--out`` and ``--score-model``."""
    parser = argparse.ArgumentParser(description="Fixed AutoML training script.")
    parser.add_argument("--config", required=True, help="path to the training config JSON")
    parser.add_argument("--out", required=True, help="path to write result.json")
    parser.add_argument(
        "--score-model",
        default=None,
        help=(
            "score the held-back test split with this saved model instead of fitting. "
            "Run once, after the loop, by nodes/holdout.py"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one attempt (or ``--score-model`` test scoring), write ``result.json``.

    Always returns ``0``; failures go inside ``result.json``.
    """
    args = parse_args(argv)
    out_path = Path(args.out)
    log = LogBuffer()
    started = time.perf_counter()

    try:
        cfg: dict[str, Any] = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type="config_error",
            log_tail=f"failed to read config {args.config}: {exc}",
        )
        return 0

    if args.score_model:
        return _score_only(cfg, Path(args.score_model), out_path, log, started)

    try:
        apply_simulation(cfg, log)
        run = run_training(
            cfg,
            log,
            model_out=out_path.parent / MODEL_FILENAME,
            predictions_out=out_path.parent / PREDICTIONS_FILENAME,
            schema_out=out_path.parent / SCHEMA_FILENAME,
            decision_out=out_path.parent / DECISION_FILENAME,
        )
    except BaseException as exc:  # noqa: BLE001 - every failure must become a result
        error_type = classify_exception(exc)
        log.write(f"training failed ({error_type}): {type(exc).__name__}: {exc}")
        log.write(traceback.format_exc(limit=6))
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type=error_type,
            log_tail=log.tail(),
        )
        return 0

    write_result(
        out_path,
        status="ok",
        metrics=run.metrics,
        train_time_sec=time.perf_counter() - started,
        error_type=None,
        log_tail=log.tail(),
        applied_hyperparams=run.applied,
        dropped_hyperparams=run.dropped,
        applied_preprocessing=run.preprocessing,
        applied_pipeline=run.applied_pipeline,
        model_path=run.model_path,
        schema_path=run.schema_path,
        paired=run.paired,
        internal_validation=run.internal_validation,
    )
    return 0


def _score_only(
    cfg: dict[str, Any], model_path: Path, out_path: Path, log: LogBuffer, started: float
) -> int:
    """_score_only | Command line: score test once, no fit or simulation"""
    try:
        metrics, preprocessing = score_saved_model(cfg, log, model_path)
    except BaseException as exc:  # noqa: BLE001 - same contract as the training path
        error_type = classify_exception(exc)
        log.write(f"test scoring failed ({error_type}): {type(exc).__name__}: {exc}")
        log.write(traceback.format_exc(limit=6))
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type=error_type,
            log_tail=log.tail(),
            split="test",
        )
        return 0

    write_result(
        out_path,
        status="ok",
        metrics=metrics,
        train_time_sec=time.perf_counter() - started,
        error_type=None,
        log_tail=log.tail(),
        applied_preprocessing=preprocessing,
        split="test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
