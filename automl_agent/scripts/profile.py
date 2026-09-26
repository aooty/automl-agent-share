"""Fixed profiling script: read raw data, write an aggregates-only dataset card.

Roles:

* Column profiling — buckets and rates per column, never values.
* Reference baseline — fixed linear model and ``chance``, with intervals.
* Card assembly — load data, apply target policy, build the card.
* Console summary — short aggregate-only summary on stdout.
* Command line — parse arguments, write the card, return exit code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

# Run by path, so add the repo root for imports.
if __package__ in (None, ""):  # pragma: no cover - only when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.dataset.caveats import (  # noqa: E402
    CAVEATS_KEY,
    card_caveats,
    grouping_caveats,
    merge_caveats,
    sentinel_caveats,
)
from automl_agent.dataset.features import (  # noqa: E402
    describe_encoding,
    encode_features,
    is_encodable_column,
    is_numeric_column,
)
from automl_agent.dataset.sentinels import (  # noqa: E402
    describe_sentinels,
    detect_sentinels,
)
from automl_agent.dataset.source import as_source, load_frame, source_kind  # noqa: E402
from automl_agent.dataset.targets import (  # noqa: E402
    DEFAULT_TARGET_MISSING_POLICY,
    TARGET_MISSING_POLICIES,
    detect_task,
    encode_target,
)
from automl_agent.scoring.intervals import (  # noqa: E402
    CI_LEVEL,
    DEFAULT_RESAMPLES,
    bootstrap_interval,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES,
    METRICS,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from automl_agent.scoring.ranking import (  # noqa: E402
    best_cut_ceiling,
    ks_statistic,
)
from automl_agent.scoring.splits import (  # noqa: E402
    describe_protocol,
    protocol,
    split_three_way,
)

# Same scorers as train.py
from automl_agent.scripts.train import scorers  # noqa: E402
from automl_agent.threads import thread_state  # noqa: E402

# (upper bound, label). Coarse, so one record cannot change a bucket.
DISTINCT_EDGES: tuple[tuple[int, str], ...] = ((1, "constant"), (2, "binary"), (10, "low"), (100, "medium"))
SKEW_EDGES: tuple[tuple[float, str], ...] = ((0.5, "low"), (2.0, "moderate"))
MAGNITUDE_EDGES: tuple[tuple[float, str], ...] = (
    (1.0, "sub_unit"),
    (10.0, "unit"),
    (100.0, "tens"),
    (1000.0, "hundreds"),
)
CORRELATION_EDGES: tuple[tuple[float, str], ...] = ((0.05, "none"), (0.15, "weak"), (0.30, "moderate"))

# Written into the card's ``constraints``.
DEFAULT_MEMORY_LIMIT_MB = 2048
DEFAULT_MAX_TRAIN_TIME_SEC = 600

# Fixed per task, so scores compare across runs
BASELINE_MODELS: dict[str, str] = {
    TASK_CLASSIFICATION: "logreg (median impute + standard scale)",
    TASK_REGRESSION: "ridge (median impute + standard scale)",
}
BASELINE_ESTIMATORS: dict[str, str] = {
    TASK_CLASSIFICATION: "LogisticRegression",
    TASK_REGRESSION: "Ridge",
}
BASELINE_NOTE = (
    "{estimator} fitted on the training split and scored on the validation split "
    "of automl_agent.scoring.splits' protocol, seed {seed} — the same split scripts/train.py "
    "uses, so the numbers are directly comparable. The test slice is not read here."
)
# Capped: profiling runs before any time budget applies.
BASELINE_MAX_ROWS = 50_000


# --- Role: column profiling -------------------------------------------------------


def _bucket(value: float, edges: tuple[tuple[float, str], ...], last: str) -> str:
    """_bucket | Column profiling: first label whose edge fits, else ``last``."""
    for edge, label in edges:
        if value <= edge:
            return label
    return last


def profile_column(series: Any, target: Any) -> dict[str, Any]:
    """Describe one column with buckets and rates only, never its values.

    ``target`` is the numeric target for ``target_corr``, or ``None``.
    """
    import numpy as np
    import pandas as pd

    n_rows = int(len(series))
    n_missing = int(series.isna().sum())
    present = series.dropna()
    distinct = int(present.nunique())
    numeric = bool(pd.api.types.is_numeric_dtype(series)) and not bool(
        pd.api.types.is_bool_dtype(series)
    )

    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "missing_rate": round(n_missing / n_rows, 4) if n_rows else 0.0,
        "distinct": _bucket(distinct, DISTINCT_EDGES, "high"),
        # features.py alone knows which category columns get one-hot.
        "usable_by_executor": (
            is_numeric_column(series) or is_encodable_column(series, distinct)
        ),
    }

    if distinct <= 1:
        profile["kind"] = "constant"
    elif distinct == 2:
        profile["kind"] = "binary"
    elif not numeric:
        # Labels are cell values; only their count leaves.
        profile["kind"] = "categorical"
    elif bool(pd.api.types.is_integer_dtype(series)) and distinct <= 20:
        profile["kind"] = "discrete"
    else:
        profile["kind"] = "continuous"

    # Only flagged; the aggregates below still count these codes.
    suspects = detect_sentinels(series)
    if suspects:
        profile["sentinel_suspects"] = suspects

    if not numeric or len(present) < 8:
        return profile

    values = present.to_numpy(dtype="float64", copy=False)
    magnitude = float(np.median(np.abs(values)))
    profile["magnitude"] = _bucket(magnitude, MAGNITUDE_EDGES, "thousands+")
    profile["skew"] = _bucket(abs(float(pd.Series(values).skew() or 0.0)), SKEW_EDGES, "high")

    q1, q3 = (float(x) for x in np.percentile(values, [25, 75]))
    spread = q3 - q1
    if spread > 0:
        outliers = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
        profile["outlier_rate"] = round(outliers / len(values), 4)

    if target is not None:
        mask = present.index
        try:
            correlation = float(pd.Series(values, index=mask).corr(target.loc[mask]))
        except (ValueError, TypeError):  # pragma: no cover - degenerate column
            correlation = float("nan")
        if correlation == correlation:  # not NaN
            profile["target_corr"] = _bucket(abs(correlation), CORRELATION_EDGES, "strong")
    return profile


# --- Role: reference baseline -----------------------------------------------------


def _mirror_aliases(scores: dict[str, float]) -> dict[str, float]:
    """_mirror_aliases | Reference baseline: copy each score under its alias names too."""
    for alias, target in ALIASES.items():
        if target in scores:
            scores[alias] = scores[target]
    return scores


def _score_all(
    y_true: Any, pred: Any, proba: Any, average: str, task: str = TASK_CLASSIFICATION
) -> dict[str, float]:
    """_score_all | Reference baseline: score every registry metric that applies here."""
    # Registry-driven, not a hand-kept list.
    thunks = scorers(y_true, pred, proba, average, task)
    scores: dict[str, float] = {}
    for name, spec in METRICS.items():
        if spec.task != task:
            continue
        scorer = thunks.get(name)
        if scorer is None:  # pragma: no cover - registry entry with no scorer here
            continue
        if spec.binary_only and average != "binary":
            continue
        if spec.needs_proba and proba is None:
            continue
        # One-class holdout has no AUC; skip only that metric.
        with contextlib.suppress(ValueError):
            scores[name] = round(float(scorer()), 4)
    return _mirror_aliases(scores)


def _baseline_intervals(
    y_true: Any,
    pred: Any,
    proba: Any,
    average: str,
    scores: dict[str, float],
    *,
    task: str = TASK_CLASSIFICATION,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any] | None:
    """_baseline_intervals | Reference baseline: bootstrap interval per score, or ``None``."""
    # All metrics: ``--metric`` is picked later.
    bounds: dict[str, Any] = {}
    unit, drawn = "row", 0
    for name in METRICS:
        if name not in scores:
            continue

        def one(y_slice: Any, pred_slice: Any, proba_slice: Any, metric: str = name) -> float:
            return float(scorers(y_slice, pred_slice, proba_slice, average, task)[metric]())

        interval = bootstrap_interval(
            one, y_true, pred, proba, groups=groups, seed=seed, resamples=resamples
        )
        if interval is None:
            continue
        # Four decimals; resamples are not more precise than that.
        bounds[name] = {"low": round(interval.low, 4), "high": round(interval.high, 4)}
        unit, drawn = interval.unit, interval.resamples
    if not bounds:
        return None
    # Alias names too, so any spelling finds it.
    for alias, target in ALIASES.items():
        if target in bounds:
            bounds[alias] = bounds[target]
    return {"level": CI_LEVEL, "unit": unit, "resamples": drawn, "scores": bounds}


def reference_scores(
    features: Any,
    codes: Any,
    n_classes: int,
    *,
    task: str = TASK_CLASSIFICATION,
    seed: int = 42,
    groups: Any = None,
    group_column: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any] | None:
    """Fit the fixed baseline and a no-feature ``chance`` model; score both on validation.

    Returns ``None`` (and prints why) instead of raising when it cannot fit.
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    regression = task == TASK_REGRESSION
    if features.shape[1] == 0 or (not regression and n_classes < 2):
        return None

    x_arr = np.asarray(features.to_numpy(), dtype="float64")
    y_arr = np.asarray(codes, dtype="float64" if regression else None)
    group_arr = np.asarray(groups) if groups is not None else None
    if len(y_arr) > BASELINE_MAX_ROWS:
        x_arr, y_arr = x_arr[:BASELINE_MAX_ROWS], y_arr[:BASELINE_MAX_ROWS]
        if group_arr is not None:
            group_arr = group_arr[:BASELINE_MAX_ROWS]

    try:
        # Same split as train.py; test rows stay unread here.
        splits = split_three_way(x_arr, y_arr, seed, groups=group_arr, stratify=not regression)
        x_train, y_train = splits.x_train, splits.y_train
        x_val, y_val = splits.x_val, splits.y_val
        if regression:
            from sklearn.dummy import DummyRegressor
            from sklearn.linear_model import Ridge

            estimator: Any = Ridge(random_state=seed)
            dummy: Any = DummyRegressor(strategy="mean")
        else:
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(max_iter=1000, random_state=seed)
            dummy = DummyClassifier(strategy="prior")
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        model.fit(x_train, y_train)
        dummy.fit(x_train, y_train)

        average = "binary" if not regression and n_classes == 2 else "macro"
        proba = (
            model.predict_proba(x_val)[:, 1] if not regression and n_classes == 2 else None
        )
        pred = model.predict(x_val)
        scores = _score_all(y_val, pred, proba, average, task)
        chance = _score_all(y_val, dummy.predict(x_val), None, average, task)
        # No ranking: AUC is 0.5, average precision is positive share.
        if not regression and n_classes == 2:
            chance["roc_auc"] = 0.5
            chance["pr_auc"] = round(float(np.mean(np.asarray(y_val) == 1)), 4)
            _mirror_aliases(chance)
    except (ValueError, MemoryError, ImportError) as exc:
        print(f"reference baseline skipped: {type(exc).__name__}: {exc}", flush=True)
        return None

    reference: dict[str, Any] = {
        "model": BASELINE_MODELS[task],
        "note": BASELINE_NOTE.format(estimator=BASELINE_ESTIMATORS[task], seed=seed),
        "n_rows_used": int(len(y_arr)),
        # Lets a run with another split protocol reject this card.
        "protocol": protocol(seed, group_column, stratified=not regression),
        # Thread settings can shift scores
        "threads": thread_state(),
        "scores": scores,
        "chance": chance,
    }
    # Best any cut of this ranking can reach; none without probabilities.
    ks = ks_statistic(y_val, proba)
    ceiling = best_cut_ceiling(ks)
    if ks is not None and ceiling is not None:
        reference["ks"] = round(ks, 4)
        reference["balanced_accuracy_at_best_cut"] = ceiling
    # Resample whole groups when the split was grouped.
    intervals = _baseline_intervals(
        y_val,
        pred,
        proba,
        average,
        scores,
        task=task,
        groups=splits.groups_val,
        seed=seed,
        resamples=resamples,
    )
    if intervals is not None:
        reference["ci"] = intervals
    return reference


# --- Role: card assembly ----------------------------------------------------------


def _card_task(task: str, n_classes: int | None) -> str:
    """_card_task | Card assembly: task label that also says binary or multiclass."""
    # CARD_TASKS maps these back to registry tasks.
    if task == TASK_REGRESSION:
        return "regression"
    return "binary_classification" if n_classes == 2 else "multiclass_classification"


def target_profile(series: Any) -> dict[str, Any]:
    """Profile a continuous target like a feature, minus feature-only keys.

    ``magnitude`` gives ``mae``/``rmse`` a scale
    """
    profile = profile_column(series, None)
    return {
        key: value for key, value in profile.items() if key not in ("name", "usable_by_executor")
    }


def build_card(
    data_path: Path | str,
    target_column: str,
    *,
    name: str | None = None,
    memory_limit_mb: float = DEFAULT_MEMORY_LIMIT_MB,
    max_train_time_sec: float = DEFAULT_MAX_TRAIN_TIME_SEC,
    baseline: bool = True,
    seed: int = 42,
    on_missing_target: str = DEFAULT_TARGET_MISSING_POLICY,
    caveats: tuple[str, ...] | list[str] = (),
    group_column: str | None = None,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Read the data and return its card; the only function here that sees rows.

    Raises ``ValueError`` for a bad target or group column, ``TargetMissingError`` under ``reject``.
    """
    frame = load_frame(data_path, table=table, query=query)
    if target_column not in frame.columns:
        # Column names are not raw data; listing them helps.
        raise ValueError(
            f"target_column {target_column!r} not found; available columns: "
            f"{sorted(str(c) for c in frame.columns)}"
        )
    if group_column:
        if group_column not in frame.columns:
            raise ValueError(
                f"group_column {group_column!r} not found; available columns: "
                f"{sorted(str(c) for c in frame.columns)}"
            )
        if group_column == target_column:
            raise ValueError(f"group_column {group_column!r} is the target column")

    target_raw = frame[target_column]
    # Group id leaves the features too: a readable id leaks labels.
    group_raw = frame[group_column] if group_column else None
    features = frame.drop(columns=[target_column] + ([group_column] if group_column else []))
    # Task comes from the column, never from the caller.
    task = detect_task(target_raw)
    regression = task == TASK_REGRESSION
    codes, keep, n_missing_target = encode_target(target_raw, on_missing_target, task)
    if n_missing_target:
        # Filter first, so every number below matches the trainer's rows.
        features = features[keep]
        if group_raw is not None:
            group_raw = group_raw[keep]
        print(
            f"dropped {n_missing_target} rows with a missing target "
            f"({target_column!r}); {len(codes)} rows remain",
            flush=True,
        )
    # Only shares leave, sorted so order hides which label.
    balance = (
        []
        if regression
        else sorted((codes.value_counts(normalize=True)).round(4).tolist(), reverse=True)
    )
    n_classes = None if regression else int(codes.nunique())

    # 3+ class codes have no order, so no ``target_corr``.
    numeric_target = codes.astype("float64") if regression or n_classes == 2 else None
    profiles = [profile_column(features[column], numeric_target) for column in features.columns]
    usable = [item for item in profiles if item["usable_by_executor"]]

    # Warn before the baseline scores it applies to.
    suspects = {
        item["name"]: item["sentinel_suspects"]
        for item in profiles
        if item.get("sentinel_suspects")
    }
    warning = describe_sentinels(suspects)
    if warning:
        print(warning, flush=True)

    # Same encoding as train.py, so baseline sees the same matrix.
    executor_features, encoding = encode_features(features)
    print(describe_encoding(encoding), flush=True)
    # Same group handling as train.py, so splits match.
    group_values = (
        group_raw.astype("string").fillna("<missing>").to_numpy() if group_raw is not None else None
    )
    if group_values is not None:
        print(
            f"grouped by {group_column}: {len(set(group_values.tolist()))} distinct groups"
            f" over {len(group_values)} rows — no group is split across train/val/test",
            flush=True,
        )
    reference = (
        reference_scores(
            executor_features,
            codes,
            n_classes or 0,
            task=task,
            seed=seed,
            groups=group_values,
            group_column=group_column,
            resamples=bootstrap_resamples,
        )
        if baseline
        else None
    )

    card: dict[str, Any] = {
        "name": name or f"{target_column}-prediction",
        # Source kind only: this reaches prompts, so no URL.
        "description": (
            f"{source_kind(data_path)} 데이터에서 자동 생성된 카드입니다. "
            "원본 행은 이 카드에 포함되지 않습니다 — 모든 수치는 열 전체에 대한 집계입니다."
        ),
        "task": _card_task(task, n_classes),
        "target_column": target_column,
        # Rows after ``drop``; the executor drops the same rows.
        "n_rows": int(len(codes)),
        "n_features": len(usable),
        # Old name kept; counts columns the executor cannot use.
        "n_features_dropped_non_numeric": len(profiles) - len(usable),
        # What the estimator really gets.
        "encoding": encoding,
        # ``None`` for regression; key kept on purpose.
        "n_classes": n_classes,
        "class_balance": balance or None,
        "imbalance_ratio": round(balance[0] / balance[-1], 2) if balance and balance[-1] else None,
        "missing": {
            "overall_rate": round(float(features.isna().to_numpy().mean()), 4),
            "columns_with_missing": sum(1 for item in profiles if item["missing_rate"] > 0),
            "worst_rate": max((item["missing_rate"] for item in profiles), default=0.0),
        },
        "features": profiles,
        # Trainer reuses this policy; readers see rows dropped.
        "target_missing": {"policy": on_missing_target, "n_dropped": int(n_missing_target)},
        "preprocessing": {"impute": "median", "scale": True},
        "constraints": {
            "memory_limit_mb": memory_limit_mb,
            "max_train_time_sec": max_train_time_sec,
        },
        "profile": {
            "generated_by": "automl_agent.scripts.profile",
            "policy": "aggregates only — no cell values, no min/max, no class labels",
        },
        # Private block; LLM plans cannot change it.
        "data": {"path": str(data_path), "target_column": target_column},
    }
    # Database path alone does not fix rows; resume rereads these.
    if table:
        card["data"]["table"] = str(table)
    if query:
        card["data"]["query"] = str(query)
    if regression:
        # Target scale, so mae/rmse can be read.
        card["target"] = target_profile(codes)
    if group_column:
        card["data"]["group_column"] = group_column
    # Machine findings first, then ``--caveat``; none means no key.
    notes = merge_caveats(
        sentinel_caveats(suspects),
        grouping_caveats(
            group_column, len(set(group_values.tolist())) if group_values is not None else None
        ),
        list(caveats),
    )
    if notes:
        card[CAVEATS_KEY] = notes
    if reference is not None:
        # Goal threshold is set from this baseline.
        card["baseline"] = reference
    return card


# --- Role: console summary --------------------------------------------------------


def summarise(card: dict[str, Any]) -> str:
    """Build the Korean console summary of a card; aggregates only."""
    missing = card.get("missing") or {}
    worst = sorted(
        (item for item in card.get("features") or [] if item.get("missing_rate")),
        key=lambda item: -float(item["missing_rate"]),
    )[:5]
    target = card.get("target") or {}
    if card.get("task") == "regression":
        # Target size and shape, to read mae/rmse below.
        target_line = (
            f"  target={card['target_column']}  kind={target.get('kind')}"
            f"  magnitude={target.get('magnitude')}  skew={target.get('skew')}"
            f"  outlier_rate={target.get('outlier_rate')}"
        )
    else:
        target_line = (
            f"  target={card['target_column']}  n_classes={card['n_classes']}"
            f"  balance={card['class_balance']}  imbalance_ratio={card['imbalance_ratio']}"
        )
    lines = [
        f"dataset card: {card['name']} ({card['task']})",
        f"  rows={card['n_rows']}  features={card['n_features']}"
        f"  dropped_non_numeric={card['n_features_dropped_non_numeric']}",
        target_line,
        f"  missing: overall={missing.get('overall_rate')}"
        f" columns={missing.get('columns_with_missing')} worst={missing.get('worst_rate')}",
    ]
    declared = (card.get("baseline") or {}).get("protocol")
    if isinstance(declared, dict):
        # Before baseline: its scores are validation-only numbers.
        lines.append("  " + describe_protocol(declared))
    encoding = card.get("encoding")
    if isinstance(encoding, dict):
        # Baseline below was measured on this encoded matrix.
        lines.append("  " + describe_encoding(encoding))
    dropped = int((card.get("target_missing") or {}).get("n_dropped") or 0)
    if dropped:
        lines.append(f"  target 결측 {dropped}행 제외 (--on-missing-target drop) — 위 수치는 남은 행 기준")
    if worst:
        lines.append(
            "  결측 상위: "
            + ", ".join(f"{item['name']}={item['missing_rate']}" for item in worst)
        )
    suspects = [item for item in card.get("features") or [] if item.get("sentinel_suspects")]
    if suspects:
        # Repeated on purpose; sentinel codes skew the missing line.
        lines.append(
            f"  결측 코드 의심 {len(suspects)}열: "
            + ", ".join(
                f"{item['name']}={item['sentinel_suspects'][0]['value']}" for item in suspects[:5]
            )
            + " (변환 안 함 — 위 경고 참고)"
        )
    notes = card_caveats(card)
    if notes:
        # Full text, so the operator sees what prompts carry.
        lines.append(f"  데이터 주의사항 {len(notes)}건 — 모든 추론 프롬프트에 그대로 실립니다:")
        lines.extend(f"    - {note}" for note in notes)
    reference = card.get("baseline")
    if isinstance(reference, dict):
        scores = reference.get("scores") or {}
        chance = reference.get("chance") or {}
        # Main names only, no aliases.
        published = {key: value for key, value in scores.items() if key not in ALIASES}
        lines.append(f"  기준선({reference.get('model')}, {reference.get('n_rows_used')}행):")
        lines.append(
            "    "
            + "  ".join(
                f"{key}={value}(chance {chance.get(key, '-')})" for key, value in published.items()
            )
        )
        ci = reference.get("ci")
        if isinstance(ci, dict) and isinstance(ci.get("scores"), dict):
            # Width shows whether a gap is real or noise.
            unit = "그룹" if ci.get("unit") == "group" else "행"
            lines.append(
                f"    {int(float(ci.get('level') or CI_LEVEL) * 100)}% CI "
                f"({unit} 단위 부트스트랩 {ci.get('resamples')}회): "
                + "  ".join(
                    f"{key}={bound.get('low')}~{bound.get('high')}"
                    for key, bound in ci["scores"].items()
                    if key in published
                )
            )
        ceiling = reference.get("balanced_accuracy_at_best_cut")
        if ceiling is not None:
            lines.append(
                f"    랭킹 KS={reference.get('ks')} → 어떤 컷으로도 "
                f"balanced_accuracy는 {ceiling}이 상한입니다 (이 기준선의 랭킹 기준). "
                "더 높은 바를 원하면 랭킹 자체를 올려야 합니다 — 특성이나 모델 family"
            )
    else:
        lines.append(
            "  기준선: 없음 — 목표 임계값은 지표별 기본값을 씁니다 "
            "(mae·rmse처럼 정답 열의 단위로 나오는 지표는 기본값이 없어 --threshold가 필요합니다)"
        )
    return "\n".join(lines)


# --- Role: command line -----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments; ``None`` means ``sys.argv``."""
    parser = argparse.ArgumentParser(description="Build a dataset card from raw data.")
    parser.add_argument(
        "--data",
        required=True,
        help="path to the CSV, path to a sqlite file (.db/.sqlite/.sqlite3), or a "
        "SQLAlchemy connection URL. A database source needs --table or --query",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="database source only: read every row of this table (shorthand for "
        '--query \'SELECT * FROM "<name>"\')',
    )
    parser.add_argument(
        "--query",
        default=None,
        help="database source only: the SELECT whose rows are the dataset",
    )
    parser.add_argument("--target", required=True, help="name of the target column")
    parser.add_argument("--out", required=True, help="where to write the dataset card JSON")
    parser.add_argument(
        "--name",
        default=None,
        help="dataset name for the card (defaults to '<target>-prediction'; the file "
        "name is deliberately not used, since it would put the data source in the prompt)",
    )
    parser.add_argument("--memory-limit-mb", type=float, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--max-train-time-sec", type=float, default=DEFAULT_MAX_TRAIN_TIME_SEC)
    parser.add_argument("--seed", type=int, default=42, help="seed for the baseline split")
    parser.add_argument(
        "--on-missing-target",
        choices=list(TARGET_MISSING_POLICIES),
        default=DEFAULT_TARGET_MISSING_POLICY,
        help="what to do about rows whose target is missing: 'reject' (default) fails and "
        "reports the count, 'drop' removes them and records how many",
    )
    parser.add_argument(
        "--caveat",
        action="append",
        default=[],
        metavar="TEXT",
        dest="caveats",
        help="a fact about this data the card's aggregates cannot show, carried into every "
        "reasoning prompt. Repeatable. This is the only channel for what a human learned by "
        "looking at the raw file — e.g. that a flag with no missing values changes meaning "
        "along row order, so a model leaning on it learns the charting regime, not acuity.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        metavar="COLUMN",
        help="a column whose rows must not be divided across train/val/test — a patient id "
        "when the table has one row per visit, say. Without it a random split puts the same "
        "patient on both sides and every score in the run is inflated by an amount nothing "
        "in the run can detect. The column is dropped from the features, and the choice is "
        "recorded in the card's private data block so the LLM cannot alter it.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the reference baseline fit. The card is then thresholdless, so the "
        "goal falls back to the per-metric default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the card, write it to ``--out``, print the summary.

    Returns 0 on success, 1 on failure (reason on stderr).
    """
    args = parse_args(argv)
    try:
        card = build_card(
            as_source(args.data),
            args.target,
            name=args.name,
            memory_limit_mb=args.memory_limit_mb,
            max_train_time_sec=args.max_train_time_sec,
            baseline=not args.no_baseline,
            seed=args.seed,
            on_missing_target=args.on_missing_target,
            caveats=list(args.caveats or []),
            group_column=args.group_column,
            table=args.table,
            query=args.query,
        )
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        sys.stderr.write(f"profiling failed: {type(exc).__name__}: {exc}\n")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summarise(card), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
