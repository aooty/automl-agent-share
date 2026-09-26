"""Fixed prediction script: apply a fitted model to new rows and write labelled rows.

Roles:

* Loading and checks — read schema, drop non-features, check width.
* Labels and probabilities — codes back to labels, probability columns.
* Batch scoring — with ``--label-column``, score on the run's metric.
* Prediction run — encode, predict, apply saved rule, write CSV.
* Console summary — Korean summary, including what to check.
* Command line — parse arguments, write report, return exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Run by path, so add the repo root for imports.
if __package__ in (None, ""):  # pragma: no cover - only when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.config import DECISION_FILENAME  # noqa: E402
from automl_agent.dataset.features import (  # noqa: E402
    FeatureSchemaMismatch,
    describe_drift,
    encode_with_schema,
)
from automl_agent.dataset.source import as_source, load_frame  # noqa: E402
from automl_agent.scoring import calibration  # noqa: E402
from automl_agent.scoring.intervals import (  # noqa: E402
    DEFAULT_RESAMPLES,
    as_number,
    interval_of,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES as METRIC_ALIASES,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    TASK_REGRESSION,
    canonical,
)

# Same scoring code as the holdout
from automl_agent.scripts.train import (  # noqa: E402
    LogBuffer,
    evaluate_split,
    label_at_cut,
    load_decision,
)

# Callers' code keys on these output names.
PREDICTION_COLUMN = "prediction"
PROBA_PREFIX = "proba_"

# Own key, so not mixed up with holdout scores.
BATCH_SCORE_KEY = "batch_metrics"


# --- Role: loading and checks -----------------------------------------------------


def load_schema(path: Path) -> dict[str, Any]:
    """load_schema | Loading: read the saved schema; never re-derive it.

    Raises ``FileNotFoundError`` if missing, ``ValueError`` if not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"feature schema not found at {path}. A model saved without one cannot be applied "
            "to new rows: nothing records which column of its input was which. Re-run training "
            "on a real data file to write one — a --dry-run or synthetic run never fits an "
            "encoding, so it has no schema to save."
        )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"feature schema {path} is not a JSON object")
    return loaded


def prepare_features(
    frame: Any, schema: dict[str, Any], label_column: str | None = None
) -> tuple[Any, list[str]]:
    """prepare_features | Loading: drop schema target/group and label columns.

    Returns ``(features, dropped_names)``.
    """
    dropped: list[str] = []
    for key in ("target_column", "group_column"):
        name = schema.get(key)
        if name and str(name) in frame.columns:
            dropped.append(str(name))
    if label_column and label_column in frame.columns and label_column not in dropped:
        dropped.append(str(label_column))
    return (frame.drop(columns=dropped) if dropped else frame), dropped


def check_width(model: Any, n_columns: int) -> None:
    """check_width | Loading: raise ``FeatureSchemaMismatch`` on a width mismatch.

    A model without ``n_features_in_`` is let through.
    """
    expected = getattr(model, "n_features_in_", None)
    if expected is None or int(expected) == int(n_columns):
        return
    raise FeatureSchemaMismatch(
        f"the model expects {int(expected)} encoded feature(s) but this schema produces "
        f"{int(n_columns)}. The model and the schema are not the pair that was saved together "
        "— check that both came from the same iteration directory."
    )


# --- Role: labels and probabilities -----------------------------------------------


def label_predictions(raw: Any, schema: dict[str, Any]) -> list[Any]:
    """label_predictions | Labels: class codes back to labels; unknown codes stay codes."""
    if schema.get("task") == TASK_REGRESSION:
        return [float(value) for value in raw.tolist()]
    classes = list(schema.get("classes") or [])
    if not classes:
        return list(raw.tolist())
    return [
        classes[int(code)] if 0 <= int(code) < len(classes) else int(code)
        for code in raw.tolist()
    ]


def probability_matrix(model: Any, matrix: Any, schema: dict[str, Any]) -> Any:
    """probability_matrix | Probabilities: call ``predict_proba`` once, or ``None``.

    Once, so the CSV and batch score share numbers.
    """
    if schema.get("task") == TASK_REGRESSION or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(matrix)
    except (AttributeError, ValueError, NotImplementedError):
        return None


def probability_columns(proba: Any, model: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """probability_columns | Probabilities: one ``proba_<label>`` column per class.

    Follows ``model.classes_``; empty when ``proba`` is ``None``.
    """
    if proba is None:
        return {}
    classes = list(schema.get("classes") or [])
    codes = list(getattr(model, "classes_", range(proba.shape[1])))
    columns: dict[str, Any] = {}
    for position, code in enumerate(codes):
        index = int(code)
        label = classes[index] if 0 <= index < len(classes) else index
        columns[f"{PROBA_PREFIX}{label}"] = proba[:, position]
    return columns


def positive_class_proba(proba: Any, model: Any) -> Any:
    """positive_class_proba | Probabilities: column for class code ``1``, or ``None``.

    Found via ``model.classes_``, not assumed ``[:, 1]``.
    """
    if proba is None or getattr(proba, "ndim", 0) != 2 or proba.shape[1] < 2:
        return None
    codes = [int(code) for code in getattr(model, "classes_", range(proba.shape[1]))]
    if 1 not in codes:
        return None
    return proba[:, codes.index(1)]


# --- Role: batch scoring ----------------------------------------------------------


def encode_batch_labels(series: Any, schema: dict[str, Any]) -> tuple[Any, Any, dict[str, int]]:
    """encode_batch_labels | Batch scoring: code true labels by schema ``classes``.

    Returns ``(codes, keep, counts)``; ``ValueError`` if no class list.
    """
    import numpy as np
    import pandas as pd

    task = schema.get("task")
    if task == TASK_REGRESSION:
        values = pd.to_numeric(series, errors="coerce")
        keep = values.notna().to_numpy()
        counts = {"missing": int((~keep).sum()), "unknown": 0}
        return values.to_numpy(dtype="float64")[keep], keep, counts

    classes = list(schema.get("classes") or [])
    if not classes:
        raise ValueError(
            "this schema records no class list, so the batch's labels cannot be coded the way "
            "the model's outputs were. Scoring would compare two different numberings."
        )
    # As text, so 1 and "1" match.
    lookup = {str(label): index for index, label in enumerate(classes)}
    raw = series.astype("string")
    mapped = raw.map(lookup)
    blank = raw.isna().to_numpy()
    keep = mapped.notna().to_numpy()
    counts = {
        "missing": int(blank.sum()),
        "unknown": int((~keep & ~blank).sum()),
    }
    return np.asarray(mapped[keep].to_numpy(), dtype="int64"), keep, counts


def score_batch(
    model: Any,
    matrix: Any,
    labels: Any,
    proba: Any,
    pred: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """score_batch | Batch scoring: score labelled rows on the run's goal metric.

    Same ``evaluate_split`` as the holdout; caller fills ``label_column``.
    """
    codes, keep, counts = encode_batch_labels(labels, schema)
    scored = int(len(codes))
    result: dict[str, Any] = {
        "label_column": None,  # filled by the caller, who knows the name
        "scored_rows": scored,
        "excluded_rows": counts,
        "metric": None,
        BATCH_SCORE_KEY: {},
        "reliability": [],
        "notes": [],
    }
    if not scored:
        result["notes"].append(
            "채점할 수 있는 행이 없습니다 — 라벨이 모두 비었거나 학습 때 없던 값입니다"
        )
        return result

    task = str(schema.get("task") or "classification")
    n_classes = 0 if task == TASK_REGRESSION else len(list(schema.get("classes") or []))
    average = "binary" if n_classes == 2 else "macro"
    metric = schema.get("metric")
    metric = canonical(str(metric)) if metric else None
    # Collect skip reasons for the summary, not stdout.
    log = LogBuffer(echo=False)

    kept_pred = pred[keep]
    kept_proba = None if proba is None else proba[keep]
    metrics = evaluate_split(
        model,
        matrix[keep],
        codes,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=metric,
        # Batch grouping unknown; row-level interval.
        groups=None,
        seed=int(schema.get("seed") or 42),
        resamples=DEFAULT_RESAMPLES,
        pred=kept_pred,
        proba=kept_proba,
    )
    result["metric"] = metric
    result[BATCH_SCORE_KEY] = metrics
    result["resamples"] = DEFAULT_RESAMPLES
    # Drop interval line; ``describe_score`` already prints it.
    result["notes"] = [line for line in log.lines if not line.startswith(f"{metric}=")]
    if kept_proba is not None and scored >= calibration.MIN_CALIBRATION_ROWS:
        # Same row minimum as ``calibration_error``.
        result["reliability"] = calibration.reliability(codes, kept_proba)
    return result


# --- Role: prediction run ---------------------------------------------------------


def run_prediction(
    model_path: Path,
    schema_path: Path,
    data_path: Path | str,
    out_path: Path,
    id_column: str | None = None,
    label_column: str | None = None,
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Predict every row, write the CSV, and return a summary dict.

    ``label_column`` adds a score but never changes predictions.
    """
    import joblib
    import pandas as pd

    schema = load_schema(schema_path)
    model = joblib.load(model_path)
    frame = load_frame(data_path, table=table, query=query)
    if id_column and id_column not in frame.columns:
        raise ValueError(f"id_column {id_column!r} not found in {data_path}")
    if label_column and label_column not in frame.columns:
        raise ValueError(
            f"label_column {label_column!r} not found in {data_path}. Without it there is "
            "nothing to score against; drop the flag to predict without scoring."
        )

    features, dropped = prepare_features(frame, schema, label_column)
    encoded, drift = encode_with_schema(features, schema)
    check_width(model, int(encoded.shape[1]))

    import numpy as np

    matrix = np.asarray(encoded.to_numpy(), dtype="float64")
    raw = model.predict(matrix)
    proba = probability_matrix(model, matrix, schema)
    positive = positive_class_proba(proba, model)

    # Saved rule, never chosen here; none means 0.5.
    decision_log = LogBuffer(echo=False)
    threshold = load_decision(model_path.parent / DECISION_FILENAME, decision_log)
    if threshold is not None and positive is None:
        decision_log.write(
            "저장된 결정 규칙이 있지만 이 모델에서는 양성 클래스 확률을 얻을 수 없어 기본 규칙으로 "
            "라벨을 만들었습니다 — 이 실행이 기록한 점수와 다른 규칙입니다"
        )
        threshold = None
    if threshold is not None:
        raw = label_at_cut(positive, threshold)

    predictions = label_predictions(raw, schema)
    # Id, prediction, probabilities; other input columns never copied.
    out: dict[str, Any] = {}
    if id_column:
        out[id_column] = frame[id_column]
    out[PREDICTION_COLUMN] = predictions
    out.update(probability_columns(proba, model, schema))
    result = pd.DataFrame(out, index=frame.index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "model": str(model_path),
        "schema": str(schema_path),
        "data": str(data_path),
        "out": str(out_path),
        "task": schema.get("task"),
        "n_features": int(encoded.shape[1]),
        "columns": [str(name) for name in result.columns],
        # Shows the label column was dropped, not used.
        "dropped_non_features": dropped,
        "drift": drift,
        # Unused rule file changes labels, so warn.
        "warnings": describe_drift(drift) + (decision_log.lines if threshold is None else []),
    }
    if threshold is not None:
        summary["threshold"] = threshold
    if label_column:
        scored = score_batch(
            model,
            matrix,
            frame[label_column],
            positive,
            raw,
            schema,
        )
        scored["label_column"] = str(label_column)
        summary["score"] = scored
    return summary


# --- Role: console summary --------------------------------------------------------


def describe_score(scored: dict[str, Any]) -> list[str]:
    """describe_score | Console summary: Korean lines, always with the protocol note."""
    metrics = dict(scored.get(BATCH_SCORE_KEY) or {})
    rows = int(scored.get("scored_rows") or 0)
    label = scored.get("label_column")
    # No Korean particle after the name; it depends on it.
    lines = [f"이 배치를 채점했습니다 — 라벨 컬럼 {label!r}, {rows}행"]

    excluded = dict(scored.get("excluded_rows") or {})
    missing, unknown = int(excluded.get("missing") or 0), int(excluded.get("unknown") or 0)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"라벨이 빈 행 {missing}개")
        if unknown:
            # Unseen class, not a dirty cell; say so.
            parts.append(f"학습 때 없던 라벨 값을 가진 행 {unknown}개")
        lines.append(f"  채점에서 제외: {', '.join(parts)}")

    if not rows:
        lines += [f"  - {note}" for note in scored.get("notes") or []]
        return lines

    metric = scored.get("metric")
    if metric and metric in metrics:
        headline = f"  {metric}={float(metrics[metric]):.4f}"
        bounds = interval_of(metrics, metric)
        if bounds is not None:
            headline += (
                f" (95% 구간 {bounds[0]:.4f}~{bounds[1]:.4f}, "
                f"row 단위 재표집 {int(scored.get('resamples') or 0)}회)"
            )
        lines.append(headline + " ← 이 실행이 목표로 삼았던 지표")
    else:
        # Say why, not skip: two cases.
        lines.append(
            "  이 배치에는 목표 지표가 없습니다"
            # Name in parentheses to avoid a Korean particle.
            + (f" — 스키마가 적은 지표({metric})는 이 행들에서 계산할 수 없었습니다" if metric else
               " — 스키마에 목표 지표가 적혀 있지 않습니다 (지표를 대체해 기록하기 전에 만들어진 "
               "스키마입니다). 이 모델을 다시 학습하면 기록됩니다")
            + ". 아래 지표는 모두 같은 채점에서 나온 값이지만, 어느 것이 이 실행이 목표로 "
            "삼았던 숫자인지는 여기서 알 수 없고 신뢰구간도 없습니다"
        )
    # Skip goal, probability lines, bounds, and aliases: shown elsewhere.
    hidden = {metric, calibration.BRIER_KEY, calibration.CALIBRATION_KEY, *METRIC_ALIASES}
    others = ", ".join(
        f"{name}={number:.4f}"
        for name, value in sorted(metrics.items())
        if name not in hidden
        and not name.endswith(("_ci_low", "_ci_high"))
        and (number := as_number(value)) is not None
    )
    if others:
        lines.append(f"  그 밖의 지표: {others}")

    probability = calibration.describe(metrics, rows)
    if probability:
        lines.append(f"  {probability}")
    lines += [f"  {line}" for line in calibration.describe_table(scored.get("reliability") or [])]

    # "Not the run's protocol" note, every time.
    lines.append(
        "  이 점수는 이 실행의 채점 프로토콜이 아닙니다 — result.json의 홀드아웃은 학습 전에 "
        "떼어 둔 행을 한 번만 채점한 값이지만, 이 파일이 어떤 행으로 이루어졌는지는 여기서 알 "
        "수 없습니다. 낮게 나오는 것이 정상일 수도 있고, 위의 '확인할 점'이 그 이유일 수도 "
        "있습니다. 이 행들이 한 대상에서 여러 번 나온 것이라면 위 신뢰구간은 실제보다 좁습니다."
    )
    for note in scored.get("notes") or []:
        lines.append(f"  - {note}")
    return lines


def summarise(summary: dict[str, Any]) -> str:
    """Render a run summary as Korean text; the warnings block matters most."""
    lines = [
        f"{summary['rows']}행을 예측해 {summary['out']}에 저장했습니다 "
        f"(인코딩된 피처 {summary['n_features']}개, task={summary['task']})"
    ]
    if summary.get("dropped_non_features"):
        names = ", ".join(summary["dropped_non_features"])
        lines.append(f"피처가 아닌 컬럼은 제외했습니다: {names}")
    if summary.get("threshold") is not None:
        # Always printed: labels differ from the 0.5 rule.
        lines.append(
            f"라벨은 이 모델과 함께 저장된 결정 규칙으로 만들었습니다 — 양성 확률 "
            f"{summary['threshold']} 이상 (sklearn 기본값 0.5가 아닙니다)"
        )
    if summary.get("score"):
        lines += describe_score(dict(summary["score"]))
    warnings = list(summary.get("warnings") or [])
    if warnings:
        # "Things to check": not every line is a problem.
        lines.append(f"확인할 점 {len(warnings)}건 — 예측은 나왔지만 아래를 읽으십시오:")
        lines += [f"  - {line}" for line in warnings]
    else:
        lines.append("학습 때의 인코딩과 어긋난 곳은 없었습니다")
    return "\n".join(lines)


# --- Role: command line -----------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line flags; ``None`` means ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        description="Fixed prediction script: apply a saved model to new rows."
    )
    parser.add_argument("--model", required=True, help="path to the saved model.joblib")
    parser.add_argument(
        "--schema",
        required=True,
        help="path to the feature_schema.json saved beside that model. Required rather than "
        "derived: re-deriving the encoding from the new file is what produces a misaligned "
        "matrix that scores without complaining",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="path to the CSV, path to a sqlite file (.db/.sqlite/.sqlite3), or a "
        "SQLAlchemy connection URL. A database source needs --table or --query",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="database source only: predict on every row of this table",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="database source only: the SELECT whose rows are the batch",
    )
    parser.add_argument("--out", required=True, help="path to write the predictions CSV")
    parser.add_argument(
        "--report",
        default=None,
        help="path to write the run summary as JSON (rows, drift, warnings). Optional; the "
        "same summary is printed either way",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="a column copied through to the output so the predictions can be joined back to "
        "the input by key instead of by row position",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="a column of true labels in --data. Given one, the batch is also scored on the "
        "metric the run was steered by, using the same scoring code as the run's holdout. The "
        "score is not the run's protocol — how these rows were assembled is unknown here — and "
        "the output printed with it says so",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run a prediction from the command line.

    Returns 0 on success, 1 on failure (reason on stderr).
    """
    args = parse_args(argv)
    try:
        summary = run_prediction(
            Path(args.model),
            Path(args.schema),
            as_source(args.data),
            Path(args.out),
            id_column=args.id_column,
            label_column=args.label_column,
            table=args.table,
            query=args.query,
        )
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        # FeatureSchemaMismatch is a ValueError; message, not traceback.
        sys.stderr.write(f"prediction failed: {type(exc).__name__}: {exc}\n")
        return 1

    if args.report:
        report_path = Path(args.report)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except OSError as exc:
            # Predictions already saved; still print summary.
            sys.stderr.write(
                f"report not written: {type(exc).__name__}: {exc}\n"
                f"  the predictions themselves are in {args.out}\n"
            )
    print(summarise(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
