"""Fixed prediction script: a fitted model plus new rows in, labelled rows out.

The third and last file that opens a data file, and the only one that opens a file the run
never trained on. Like the other two it runs as a subprocess, so the orchestrator process
still never holds a data row — and unlike the other two it produces *per-row* output, which
is why nothing it writes goes anywhere near a state channel.

Why this exists
---------------
``model.joblib`` on its own is not a usable model. The matrix it was fitted on was produced
by :func:`automl_agent.dataset.features.encode_features`, which reads the level set, the column order
and whether a column got its own missing column off the file in front of it. Handed a second
file, that function answers differently: a level that happens to be absent removes a column,
a new level adds none, and the column order follows the new file's. When the widths differ,
sklearn raises. When they *match* — which is the ordinary case for a monthly export of the
same table — nothing raises and every prediction is computed from columns that mean something
else. That failure is invisible in the output and in every metric over it.

So the encoding is saved at fit time (``feature_schema.json``, written by
``scripts/train.py``) and replayed here by :func:`automl_agent.dataset.features.encode_with_schema`.
This script's job is to refuse everything that cannot be replayed and to disclose everything
that was replayed with a caveat:

* a source column the fit needs and this file does not have — refused
  (:class:`automl_agent.dataset.features.FeatureSchemaMismatch`);
* a schema whose version this build does not know — refused;
* an assembled width the estimator disagrees with — refused, even though the encoder and the
  schema already agree, because two artifacts that were meant to be saved together are the
  one thing a directory cannot prove;
* a new category, a gap where training had none, a column that has become text, a column the
  file added — carried out as a warning per finding, because each of those has a defined
  encoding and pretending otherwise would be the silence this file exists to remove.

Scoring a batch that already has its labels
-------------------------------------------
``--label-column`` names a column of true labels in the input file, and turns the run into a
backtest: the batch is predicted exactly as it would be without the flag, and then scored.
Three things about that score are deliberate.

It is **not the run's protocol.** The holdout in ``result.json`` is rows held back before any
model was fitted, scored once, gating nothing — a number whose meaning comes from how it was
produced. How *this* file was assembled is unknown to this script: it may be a later month, a
different site, or the training rows themselves. So the score is reported with that said out
loud, because "0.71 on the holdout, 0.62 here" is a fact about two different things until
someone says which two.

It is scored on **the metric the run was steered by** (``schema["metric"]``, recorded at fit
time) rather than on whatever the person scoring the batch picks, so the two numbers are the
same measurement over different rows.

The labels are coded through **the schema's** ``classes`` list, never re-derived from this
file. Re-deriving is the same defect this whole script exists to prevent, one level up: a
batch where one class happens to be absent would code the remaining labels to different
integers, and the model's ``1`` would be scored against the file's other class. Rows whose
label is missing or is not in the schema's list are excluded from the score and counted.

Nothing about the score changes the model. It is measured after the fact on rows this process
is not entitled to fit anything on — see :mod:`automl_agent.scoring.calibration` for the same argument
about the probabilities.

Contract
--------
Input  : ``--model <model.joblib> --schema <feature_schema.json> --data <csv> --out <csv>``,
         optionally ``--label-column <name>`` to score the batch
Output : the predictions CSV at ``--out``, one row per input row in input order; a JSON
         summary at ``--report`` when asked for; a Korean summary on stdout; exit 0 on
         success, 1 on failure (stderr stays local and is never prompted).

Every output of this script is per-row data. The CSV is the point of the run and belongs
wherever the caller keeps their data; the report holds category names, so it defaults to
``artifacts/``, which ``.gitignore`` blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Same reason as the other two scripts: run by path, so the repo root is not on sys.path and
# a relative import is impossible. Adding it is what lets this file replay the *same* encoder
# the fit used instead of a copy of it.
if __package__ in (None, ""):  # pragma: no cover - only when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.dataset.features import (  # noqa: E402 - needs the path fix above
    FeatureSchemaMismatch,
    describe_drift,
    encode_with_schema,
)
from automl_agent.scoring import calibration  # noqa: E402 - needs the path fix above
from automl_agent.scoring.intervals import DEFAULT_RESAMPLES  # noqa: E402 - needs the path fix above
from automl_agent.scoring.metrics import (  # noqa: E402 - needs the path fix above
    ALIASES as METRIC_ALIASES,
)
from automl_agent.scoring.metrics import (  # noqa: E402 - needs the path fix above
    TASK_REGRESSION,
    canonical,
)

# Imported rather than copied, for the same reason the encoder is: a second implementation of
# "score this split" is a second answer, and the whole point of a batch score is that it is
# comparable to the holdout in ``result.json``. train.py's own imports are all pure-python
# package modules — sklearn and pandas are imported inside its functions — so this costs no
# import-time dependency that this script does not already have.
from automl_agent.scripts.train import (  # noqa: E402 - needs the path fix above
    LogBuffer,
    evaluate_split,
)

# The column the predicted label lands in, and the prefix each class probability gets. Named
# here because the caller's downstream code keys on them.
PREDICTION_COLUMN = "prediction"
PROBA_PREFIX = "proba_"

# The label a batch score is reported under, so a report that carries both this and the run's
# holdout cannot have them confused by a reader or by a later script.
BATCH_SCORE_KEY = "batch_metrics"


def load_schema(path: Path) -> dict[str, Any]:
    """Read the encoding saved beside the model.

    A missing file is a refusal with the reason spelled out, not a fallback to re-deriving the
    layout. Re-deriving is precisely the operation that produces a silently misaligned matrix,
    and the two runs that legitimately have no schema — a synthetic-data run, and a run from
    before schemas were written — are both runs whose model cannot be applied to new rows at
    all. Saying so is the useful answer.
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
    """Drop the columns that were never features, and say which were dropped.

    The target and the group column are dropped by *name from the schema*, not by guessing:
    a new file may well carry the true label (a backtest) or the patient id, and both were
    excluded from the matrix at fit time. Without this they would be reported as columns the
    training file did not have, which is true and useless.

    ``label_column`` is dropped for the same reason and named in the same list. It is usually
    the schema's target column under the same name, in which case it is already handled — but a
    backtest export that calls it ``outcome_actual`` would otherwise reach the encoder as an
    unexpected extra column, i.e. reported as drift when in fact it is the thing being scored
    against.
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
    """Refuse a matrix the estimator was not fitted on, before it silently accepts one.

    The encoder already asserted that it produced the schema's columns, so this only fires
    when the schema and the model are not the pair they were saved as — a file copied out of
    one iteration's directory next to another's. sklearn would catch a width mismatch itself,
    with a message about arrays; this one names the cause.

    ``n_features_in_`` is absent on an estimator that was never fitted and on a few that do not
    record it. Absent means "cannot check", not "mismatch": refusing there would reject working
    pairs to protect against a case that has not happened.
    """
    expected = getattr(model, "n_features_in_", None)
    if expected is None or int(expected) == int(n_columns):
        return
    raise FeatureSchemaMismatch(
        f"the model expects {int(expected)} encoded feature(s) but this schema produces "
        f"{int(n_columns)}. The model and the schema are not the pair that was saved together "
        "— check that both came from the same iteration directory."
    )


def label_predictions(raw: Any, schema: dict[str, Any]) -> list[Any]:
    """Class codes back into the labels the file used. Regression values pass through.

    ``encode_target`` category-coded the target column, so a fitted classifier predicts ``1``
    where the caller asked about ``died``. The schema's ``classes`` list is index-aligned with
    those codes, so this is a lookup rather than an inference. A code outside the list is left
    as the code: it means the schema and the model disagree about the label set, and inventing
    a name for it would be worse than showing the number.
    """
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
    """``predict_proba`` once, or ``None``. Computed here so it is computed once.

    Both the output columns and the batch score need these numbers, and calling
    ``predict_proba`` twice would let a non-deterministic estimator put two different sets of
    probabilities under one report — the CSV saying 0.70 and the Brier score priced on 0.68.
    """
    if schema.get("task") == TASK_REGRESSION or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(matrix)
    except (AttributeError, ValueError, NotImplementedError):
        return None


def probability_columns(proba: Any, model: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Per-class probabilities, keyed by the label rather than by the code.

    Empty for a regression target, and empty for a classifier without ``predict_proba`` —
    absent columns rather than a fabricated 0/1 confidence, which is what a hard label
    reported as a probability would be.

    Keyed off ``model.classes_``, not off the schema's list, because a class with no rows in
    the training split has no column in the output and the two lists would then be offset by
    one for every class after it.
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
    """The column of ``proba`` belonging to class code ``1``, or ``None``.

    ``train.py``'s ``_proba`` takes ``[:, 1]`` because its ``classes_`` is the sorted codes
    ``[0, 1]`` and column 1 is therefore code 1. Here the position is looked up instead of
    assumed: it is the same column whenever both classes were present at fit time, and when one
    was not, ``[:, 1]`` would be some other class's probability scored as if it were the
    positive one. A binary metric computed off the wrong column is the failure mode of this
    entire file, so it is not worth saving three lines over.
    """
    if proba is None or getattr(proba, "ndim", 0) != 2 or proba.shape[1] < 2:
        return None
    codes = [int(code) for code in getattr(model, "classes_", range(proba.shape[1]))]
    if 1 not in codes:
        return None
    return proba[:, codes.index(1)]


def encode_batch_labels(series: Any, schema: dict[str, Any]) -> tuple[Any, Any, dict[str, int]]:
    """True labels as the codes the model predicts, plus a mask of which rows are usable.

    Returns ``(codes, keep, counts)`` where ``keep`` is a boolean mask over the input rows.

    Coded through ``schema["classes"]`` by position — the same list ``label_predictions`` reads
    the other direction — and never through ``encode_target``. Re-deriving categories from this
    file is what would silently renumber the labels: a batch missing one class would code its
    remaining labels ``0..n-2``, so the model's ``1`` would be scored against a different class
    than the fit meant by it, and every metric over that is a number about nothing.

    Rows the schema has no code for are dropped and counted rather than mapped to something:
    ``missing`` for a blank label, ``unknown`` for a value the fit never saw. A new class in a
    backtest file is real news, and the honest form of it is "these rows could not be scored",
    not a score computed as if they were negatives.
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
    # As text on both sides, because a CSV round-trip turns the integer label 1 into the string
    # "1" and a JSON schema stores whatever the fit's column held. Matching on the rendered
    # value is what makes 1 and "1" the same class instead of one known and one unknown.
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
    """Score the labelled rows of this batch, on the metric the run was steered by.

    Everything here is measured through :func:`automl_agent.scripts.train.evaluate_split`, the
    same function that produced the holdout number in ``result.json``. What differs is the rows
    and — said in the summary, not hidden — that how these rows were assembled is unknown.
    """
    codes, keep, counts = encode_batch_labels(labels, schema)
    scored = int(len(codes))
    result: dict[str, Any] = {
        "label_column": None,  # filled by the caller, which knows the name
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
    # Collect without printing: the skip reasons belong in the summary block beside the score
    # they explain, not scattered above it.
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
        # No groups: this file's clustering is not knowable here. The schema's group column may
        # well be present, but whether *these* rows are one export of many per subject is a
        # fact about how the batch was assembled, which is the thing this script does not know.
        # A row-resampled interval on clustered rows is too narrow, so the interval is reported
        # with that caveat rather than dressed up as a group interval it is not.
        groups=None,
        seed=int(schema.get("seed") or 42),
        resamples=DEFAULT_RESAMPLES,
        pred=kept_pred,
        proba=kept_proba,
    )
    result["metric"] = metric
    result[BATCH_SCORE_KEY] = metrics
    result["resamples"] = DEFAULT_RESAMPLES
    # The interval line is dropped: ``describe_score`` prints the same bounds beside the metric
    # itself, and the same number twice trains the reader to skip the block that also carries
    # the "roc_auc skipped: only one class present" kind of line.
    result["notes"] = [line for line in log.lines if not line.startswith(f"{metric}=")]
    if kept_proba is not None and scored >= calibration.MIN_CALIBRATION_ROWS:
        # Gated on the same floor as ``calibration_error``, and for the same reason: below it a
        # bin holds a handful of rows, so a table of "1행, 예측 0.79, 실제 0.000" reads as a
        # model that is wrong where it is only unmeasured. Withholding the summary number while
        # printing the bins it was withheld over would be having it both ways.
        result["reliability"] = calibration.reliability(codes, kept_proba)
    return result


def run_prediction(
    model_path: Path,
    schema_path: Path,
    data_path: Path,
    out_path: Path,
    id_column: str | None = None,
    label_column: str | None = None,
) -> dict[str, Any]:
    """Predict every row of ``data_path`` and write the CSV. Returns the summary.

    ``label_column`` adds a score over the rows whose label the schema can code. It does not
    change a single prediction: the batch is encoded, checked and predicted identically either
    way, and the labels are read only after that.
    """
    import joblib
    import pandas as pd

    schema = load_schema(schema_path)
    model = joblib.load(model_path)
    frame = pd.read_csv(data_path)
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
    predictions = label_predictions(raw, schema)
    # The id column first, so the output can be joined back without positional trust, then the
    # prediction, then the probabilities. The input's other columns are deliberately not copied:
    # this file is a second copy of the caller's data if it is, and they already have the first.
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
        # Named, so a backtest can tell "the label column was excluded" from "the label column
        # was fed to the model as a feature".
        "dropped_non_features": dropped,
        "drift": drift,
        "warnings": describe_drift(drift),
    }
    if label_column:
        scored = score_batch(
            model,
            matrix,
            frame[label_column],
            positive_class_proba(proba, model),
            raw,
            schema,
        )
        scored["label_column"] = str(label_column)
        summary["score"] = scored
    return summary


def describe_score(scored: dict[str, Any]) -> list[str]:
    """The batch score as Korean lines, with what it is not.

    The caveat is not a footnote here, it is the second line. A batch score's meaning comes
    entirely from how the batch was assembled, and this script cannot see that: the same
    function that scored the holdout produced this number, over rows whose provenance is the
    caller's knowledge and not the harness's. A number printed without that said is a number
    that will be compared to the holdout as though the comparison were valid.
    """
    metrics = dict(scored.get(BATCH_SCORE_KEY) or {})
    rows = int(scored.get("scored_rows") or 0)
    label = scored.get("label_column")
    # Phrased so the column name is not followed by a Korean particle: the right particle
    # depends on the last syllable of a name this script does not choose.
    lines = [f"이 배치를 채점했습니다 — 라벨 컬럼 {label!r}, {rows}행"]

    excluded = dict(scored.get("excluded_rows") or {})
    missing, unknown = int(excluded.get("missing") or 0), int(excluded.get("unknown") or 0)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"라벨이 빈 행 {missing}개")
        if unknown:
            # Worth its own words: an unknown label is not a dirty cell, it is a class the fit
            # never saw, and the model has no code to predict it with.
            parts.append(f"학습 때 없던 라벨 값을 가진 행 {unknown}개")
        lines.append(f"  채점에서 제외: {', '.join(parts)}")

    if not rows:
        lines += [f"  - {note}" for note in scored.get("notes") or []]
        return lines

    metric = scored.get("metric")
    if metric and metric in metrics:
        headline = f"  {metric}={float(metrics[metric]):.4f}"
        low, high = metrics.get(f"{metric}_ci_low"), metrics.get(f"{metric}_ci_high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            headline += (
                f" (95% 구간 {float(low):.4f}~{float(high):.4f}, "
                f"row 단위 재표집 {int(scored.get('resamples') or 0)}회)"
            )
        lines.append(headline + " ← 이 실행이 목표로 삼았던 지표")
    else:
        # Said, not skipped. Without this line the block below reads as an ordinary score
        # sheet where one of the numbers happens to be the goal — and the reader has no way
        # to tell which, because the schema could not name one. The metric also carries the
        # only confidence interval, so its absence quietly removes the interval too.
        #
        # Two different absences, and the branch below keeps them apart. A *named* metric that
        # this batch cannot compute is ordinary and current: ``roc_auc`` over rows that all
        # carry the same label is undefined, and ``score_split`` drops it rather than
        # recording NaN. An *unnamed* one is a schema written before ``train.goal_metric``
        # existed, when a goal metric belonging to the other task was recorded as ``null``
        # instead of being substituted — a file on disk, not a path this build can produce.
        lines.append(
            "  이 배치에는 목표 지표가 없습니다"
            # Parenthesised rather than inflected: the particle after a metric name depends on
            # its last syllable ("f1을" but "rmse를"), and the name comes from the schema.
            + (f" — 스키마가 적은 지표({metric})는 이 행들에서 계산할 수 없었습니다" if metric else
               " — 스키마에 목표 지표가 적혀 있지 않습니다 (지표를 대체해 기록하기 전에 만들어진 "
               "스키마입니다). 이 모델을 다시 학습하면 기록됩니다")
            + ". 아래 지표는 모두 같은 채점에서 나온 값이지만, 어느 것이 이 실행이 목표로 "
            "삼았던 숫자인지는 여기서 알 수 없고 신뢰구간도 없습니다"
        )
    # Everything else the same scoring call produced, minus three groups that would be noise
    # here: the goal metric (its own line above), the two probability diagnostics (their own
    # line below), the interval bounds (already printed with the metric they bound), and the
    # registry's aliases, which are the same number under a second name.
    hidden = {metric, calibration.BRIER_KEY, calibration.CALIBRATION_KEY, *METRIC_ALIASES}
    others = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in sorted(metrics.items())
        if name not in hidden
        and isinstance(value, (int, float))
        and not name.endswith(("_ci_low", "_ci_high"))
    )
    if others:
        lines.append(f"  그 밖의 지표: {others}")

    probability = calibration.describe(metrics, rows)
    if probability:
        lines.append(f"  {probability}")
    lines += [f"  {line}" for line in calibration.describe_table(scored.get("reliability") or [])]

    # The label the module docstring argues for, printed every time the score is.
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
    """The run as Korean lines for a human. The warnings are the part worth reading."""
    lines = [
        f"{summary['rows']}행을 예측해 {summary['out']}에 저장했습니다 "
        f"(인코딩된 피처 {summary['n_features']}개, task={summary['task']})"
    ]
    if summary.get("dropped_non_features"):
        names = ", ".join(summary["dropped_non_features"])
        lines.append(f"피처가 아닌 컬럼은 제외했습니다: {names}")
    if summary.get("score"):
        lines += describe_score(dict(summary["score"]))
    warnings = list(summary.get("warnings") or [])
    if warnings:
        # "확인할 점" rather than "경고": a column the fit already ignored is reported every
        # time and is not an anomaly, so calling all of these warnings would train the reader
        # to skip the block that also carries the unseen-category lines.
        lines.append(f"확인할 점 {len(warnings)}건 — 예측은 나왔지만 아래를 읽으십시오:")
        lines += [f"  - {line}" for line in warnings]
    else:
        lines.append("학습 때의 인코딩과 어긋난 곳은 없었습니다")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--data", required=True, help="path to the CSV to predict on")
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
    args = parse_args(argv)
    try:
        summary = run_prediction(
            Path(args.model),
            Path(args.schema),
            Path(args.data),
            Path(args.out),
            id_column=args.id_column,
            label_column=args.label_column,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        # FeatureSchemaMismatch is a ValueError, so the refusals this script exists for come
        # out as a message rather than a traceback.
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
            # The predictions are already on disk by now. Unguarded, a bad --report path
            # ended the command in a traceback, and what the operator concluded from that
            # was that the prediction had failed — so the message names the file that *did*
            # get written, and the summary below is printed either way.
            sys.stderr.write(
                f"report not written: {type(exc).__name__}: {exc}\n"
                f"  the predictions themselves are in {args.out}\n"
            )
    print(summarise(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
