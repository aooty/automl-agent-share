"""The fixed executor's can and cannot list, shared by the prompts and the plan check.

Roles:

* Capability records — one thing the executor does not do.
* Executor numbers — sklearn and split numbers the entries quote.
* Capability list — what the executor does and does not do.
* Negation words — words that mark a phrase as denied.
* Prompt rendering — turn the list and row counts into prompts.
* Claim detection — find plan prose needing a missing capability.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION
from .scoring.splits import TEST_FRACTION, TRAIN_SHARE, VAL_SHARE, row_counts

# --- Role: capability records ---------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """One thing the executor does not do, and the closest thing it does instead."""

    name: str
    summary: str
    instead: str
    # Lowercase substrings; empty means state it, never detect it.
    markers: tuple[str, ...] = field(default_factory=tuple)
    # Tasks to state it for; detection ignores this
    tasks: tuple[str, ...] = (TASK_CLASSIFICATION, TASK_REGRESSION)


# --- Role: executor numbers -----------------------------------------------------------

# sklearn auto early stopping: strictly above this, on train rows.
EARLY_STOPPING_AUTO_MIN_ROWS = 10_000

# sklearn defaults: models that early-stop on a train slice.
SELF_VALIDATING_MODELS = ("hist_gbdt", "mlp")
DEFAULT_VALIDATION_FRACTION = 0.1

# Copy of ``scripts.train``'s value (avoids sklearn); a test ties them.
CUT_FRACTION = 0.2


# --- Role: capability list ------------------------------------------------------------

# Copied from ``scripts/train.py`` and ``dataset/pipeline.py``, not intent.

_ONE_ESTIMATOR = "Fit exactly one estimator from the model list per attempt."
_HYPERPARAMS = (
    "Apply proposed `hyperparams` through `set_params`, under sklearn's own names "
    "(a small alias map covers `n_estimators`→`max_iter` for hist_gbdt and similar). "
    "Keys the estimator does not accept are dropped and recorded in "
    "`dropped_hyperparams`."
)
_CLASS_WEIGHT = (
    "Rebalance classes through the estimator's own `class_weight` — `'balanced'`, or an "
    'explicit weight per class such as `{"0": 1, "1": 10}`, whose keys are class codes '
    "0..n_classes-1 in the order of the card's `class_balance` — or xgboost's "
    "`scale_pos_weight`, where that estimator supports it. `'balanced'` is one point on this "
    "lever, not the best one: it pins the ratio at the class frequency. A map that does not "
    "cover every class exactly once is dropped into `dropped_hyperparams`. "
    "It is not the only way to move the operating point — `tune_threshold` moves the cut "
    "directly — and the two are alternatives on the same axis rather than a pair to stack. "
    "Reweighting changes the fit, so it moves the ranking as well as the cut and its effect on "
    "the cut is indirect; the threshold moves the cut and nothing else. Doing both in one "
    "attempt leaves two owners for whatever the score does."
)
_SUBSAMPLE = "Cut the training set with `train_subsample` (a float strictly between 0 and 1)."
# No slash between words here: ``redact_paths`` would mangle it.

_PREPROCESSING = (
    "Impute and scale as the plan's `preprocessing` block asks, falling back to the dataset "
    "card's: `impute` as one of `median`, `mean` or `most_frequent`, applied to every column "
    "with one strategy, and `scale` (standardisation) for scale-sensitive families only. "
    "`impute` as `none` removes the imputer so the model splits on NaN itself — available "
    "for `hist_gbdt` and `xgboost` only, and any other family is put back on `median`. "
    "Each attempt reports the pipeline that was really built in `applied_preprocessing`, so "
    "a downgrade is visible in the history rather than only in the training log."
)
_MISSINGNESS = (
    "Turn missingness into columns, both booleans and both applied before imputation: "
    "`missing_indicator` appends one 0 or 1 column per input column, and `missing_count` "
    "appends a single column holding how many of that row's fields were not measured. "
    "`missing_indicator` covers every column or none — there is no naming a subset of the "
    "high-missing ones. "
    "`missing_count` is the one a NaN-splitting family cannot already express — its splits "
    "are per column, and this aggregates across them, so on clinical rows it stands in for "
    "how much workup a patient received. "
    "One thing about `missing_indicator` is settled. Handed to a family that splits on NaN "
    "natively under `impute: none` it is *redundant*, not merely weak — appending the "
    "indicator columns returned predictions identical bit for bit, because the indicator's "
    "only split is the NaN branch the tree already had. That reproduced under both of the "
    "splits below, so do not spend an iteration on it alongside `impute: none`. "
    "What it is worth when the plan *does* impute is not settled, and the reason is worth "
    "reading before spending an iteration on it. Under a stratified random split of one "
    "clinical sample the gains were large and mutually consistent: imputing cost roc_auc, "
    "the indicator recovered part of that in the same tree family, and it recovered the most "
    "on `logreg`, which cannot be handed a NaN at all — which fits a linear model having no "
    "way to express *this value was invented* while a tree can by splitting at the imputed "
    "value. Refitting the same configurations on the same file under a contiguous split, so "
    "that training and validation rows fall on either side of a change in what the file "
    "records, all three collapsed to intervals spanning 0. Absolute roc_auc barely moved, so "
    "what failed to carry over is the missingness lever specifically and not "
    "the models. The likeliest reading is that under the random split those columns were "
    "partly identifying which recording regime a row came from, which is not a subject's "
    "state; the contiguous split does not cleanly separate that from the weaker reading, "
    "since the column with the most missingness is observed in most of its training rows and "
    "almost none of its validation rows, and an indicator on it is nearly constant there. "
    "Either way the instruction is the same: no size for this lever is established, so do "
    "not cite one as what an iteration here will buy. `missing_count` bought nothing under "
    "the random split, on its own or on top of the indicator in either family, "
    "and was not refitted under the contiguous one. Being representable is not the same as "
    "being worth a column. What generalises is the check rather than the sizes: read the "
    "card's caveats for *why* values are missing, and where missingness tracks when or where "
    "a row was recorded rather than the subject's state, these columns let the model learn "
    "the recording regime — a random split scores that as a gain instead of showing it."
)
# Dropped sizes live in ``docs/contracts.md`` and ``docs/FINDINGS-mimic.md``.
_LEVER_AXES = (
    "Price a lever on the axis it moves, because two of them move independently and "
    "`balanced_accuracy` is their sum. The *ranking* axis is what `roc_auc`, `pr_auc` and "
    "`balanced_accuracy_at_best_cut` measure — how well the model orders rows, which no "
    "decision rule can improve. The *operating point* axis is where that ranking is cut, which "
    "is what `tune_threshold` moves directly and what `class_weight` and `scale_pos_weight` "
    "move sideways, and what `balanced_accuracy_cut_headroom` measures the remaining size of. "
    "A `balanced_accuracy` "
    "difference on its own does not say which of the two moved, so it cannot rank levers. "
    "Measured on one clinical sample, most of the spread between the best and the worst of a "
    "run's attempts was the cut landing somewhere else rather than the ranking, and the worst "
    "attempt was carrying `balanced_accuracy_cut_headroom` it had never collected. On the "
    "ranking axis in the same measurements, retuning hyperparameters inside one family and "
    "swapping tree family under an unchanged pipeline overlapped: the smaller family swap is "
    "not distinguishable from retuning at the resolution those measurements have, and there "
    "is no single size for 'swap the family' to budget against either — it depended on which "
    "family. The "
    "imputation lever is deliberately given no size at all: on this file it reads as a gain "
    "under a random split and as nothing under a contiguous one, so there is nothing stable "
    "to rank it by, and the missing-data entry above is where "
    "that is set out. No measurement licenses calling a lever the largest available. One "
    "verdict read a size that way, prescribed the imputation change together with a family it "
    "had not tried, and the attempt lost `balanced_accuracy` with two owners for "
    "the loss. What survives is not a lever ranking: the two axes have to be read apart, the "
    "order the `balanced_accuracy` column suggests is not the order the ranking axis has, and "
    "`balanced_accuracy_cut_headroom` against the distance still to go is the number that says which axis a "
    "shortfall is on. The operating-point share is now collectable rather than only measurable "
    "— it is what "
    "`tune_threshold` reaches for — which changes what the decomposition is *for*: it no longer "
    "says that share is out of reach, it says how much of the shortfall the "
    "cheap lever can take and how much needs the ranking."
)
_CALIBRATION_DIAGNOSTICS = (
    "Report whether the predicted probabilities are worth reading as probabilities, without "
    "changing them: `brier` is the mean squared error of the positive-class probability, and "
    "`calibration_error` is the count-weighted distance between predicted and observed rate "
    "over ten bins — so 0.04 means the probabilities are off by four percentage points on "
    "average. Both are diagnostics no goal may target, and `calibration_error` is omitted "
    "under 50 rows because that estimator is biased upward on few rows; its absence means "
    "not measured, not fine. There is no recalibration lever here — nothing refits or rescales "
    "the probabilities, so do not propose `CalibratedClassifierCV` or a Platt scaling of them. "
    "Moving the *cut* over unchanged probabilities is a different thing and is available "
    "(`tune_threshold`); it does not fix calibration, and neither `brier` nor "
    "`calibration_error` will move when it is used. What the two numbers are for is reading a "
    "shortfall: a model can rank well and still be systematically overconfident, and no metric "
    "in the registry would show it."
)
_PIPELINE_SPEC = (
    "Take an ordered `pipeline` — a list of steps, each naming the columns it applies to — "
    "instead of the four `preprocessing` flags. Give one or the other: a spec makes the "
    "executor ignore `preprocessing` entirely, because two descriptions of one pipeline leave "
    "nothing able to say which of them ran. The steps that exist are `impute`, `interactions`, "
    "`scale`, `missing_indicator` and `missing_count`, and a step this list does not name is "
    "dropped with a reason. "
    "A step's `columns` are the card's column names; absent means every column. `impute` also "
    'takes `groups` — `{"columns": [...], "strategy": "constant", "indicator": false}` — so the '
    "columns whose imputation differs are named and the rest take the step's own `strategy`. "
    "`interactions` is degree-2 pairwise products appended to their inputs; there are no "
    "squares, and over 50 input columns it is declined rather than expanded. Whatever ran comes "
    "back as `applied_pipeline`, one line per step in order, with `(auto)` marking a step the "
    "executor added because the family needed it and the spec did not ask — an imputer for a "
    "family that cannot take a NaN, scaling for a scale-sensitive one. "
    "What this buys, measured on one clinical sample against the same rows (paired Δ, 95% CI): "
    "naming the high-missing columns for constant imputation moved `roc_auc` on both `logreg` "
    "and `hist_gbdt`, both intervals clear of zero; `interactions` moved `logreg` and left "
    "`hist_gbdt` with an interval spanning zero, which fits "
    "— a tree can already express an interaction by splitting twice. Together on `logreg` they "
    "gave *more than their sum*, because the interaction step consumes the indicator "
    "columns the imputation step appended. That composition is the reason the order is yours. "
    "The sharpest thing the flags could not express, and the one worth reading before writing a "
    "spec: on `hist_gbdt` with `strategy: none` — NaN left for the tree to split on — plus "
    "`constant` on the high-missing columns *and a `missing_indicator` naming only those "
    "columns*, `roc_auc` cleared the best thing the flags could reach at all, interval clear of "
    "zero. Without that subset indicator the same spec's interval spanned "
    "zero. The mechanism is why: an indicator over *every* column is bit-for-bit redundant under "
    "`strategy: none`, since the tree already has the NaN branch — but the columns just "
    "filled with a constant no longer have one, so an indicator on exactly those restores "
    "what the fill destroyed and nothing else. "
    "Every gain above is on the *ranking* axis. Read them next to `_LEVER_AXES`: all of the "
    "`balanced_accuracy` intervals spanned zero, because a better ranking does not show at a "
    "fixed cut. What it does is raise `balanced_accuracy_at_best_cut` — the same column-wise "
    "imputation moved it on `hist_gbdt` — which is what `tune_threshold` then collects. "
    "One sample, one split, and the sizes are deliberately not quoted: read these as the "
    "reason the step exists, not as what it will buy here."
)
_MEMORY_HINTS = (
    "Read `batch_size` and `precision: fp16` as *memory-budget hints only* — they "
    "change the pre-flight memory estimate, not how the model is fitted."
)


def _scoring_protocol(strata: str, extra: str = "") -> str:
    """_scoring_protocol | Capability list: split protocol text both scoring entries share."""
    return (
        f"Score on a fixed protocol: {strata} three-way split at the run's seed — train "
        f"{TRAIN_SHARE:.0%}, validation {VAL_SHARE:.0%}, test {TEST_FRACTION:.0%} — where every "
        "score you are shown, and the choice of the run's best attempt, comes from the "
        f"validation {VAL_SHARE:.0%}.{extra} The test {TEST_FRACTION:.0%} is carved off first, is "
        "scored exactly once after the loop ends, and is never visible to a plan or a diagnosis. "
        "Reported on the validation split: "
    )


_CLASSIFICATION_SCORING = _scoring_protocol("a stratified") + (
    "every "
    "metric in the registry, plus `train_f1`/`train_accuracy`/`train_<goal metric>` "
    "and `train_val_gap` measured on the goal metric. On a binary target it also reports "
    "`specificity`, which no goal may target but which names the direction the imbalance "
    "lever has to move: `balanced_accuracy` is the mean of `recall` and `specificity`."
)
_CUT_DIAGNOSTICS = (
    "Report what the decision threshold is worth on this ranking, whether or not the plan "
    "moved it: `balanced_accuracy_at_best_cut` is the best `balanced_accuracy` any cut of this "
    "model's ranking allows, and `balanced_accuracy_cut_headroom` is the distance from the score actually "
    "achieved. Both are diagnostics no goal may target, and both are measured on the "
    "validation rows — so they are the *residual* even when `tune_threshold` moved the cut, "
    "because that cut was chosen elsewhere. Read `balanced_accuracy_cut_headroom` before reaching for the "
    "operating point at all: when it is small the cut is near-optimal already and the "
    "remaining gap is in the ranking, which means the model family or the features, not "
    "`class_weight` and not the threshold."
)
_THRESHOLD_TUNING = (
    "Choose the decision threshold, when the plan sets `tune_threshold: true` on a binary "
    f"target. The executor then holds {CUT_FRACTION:.0%} of the *training* rows out of the fit, sweeps the "
    "goal metric over ~150 candidate cuts on that held-back slice, applies the winning cut to "
    "every score this attempt reports, and saves it beside the model so the final test scoring "
    "and `predict` label rows by the same rule. The cut appears as `applied_threshold` and the "
    "rows it cost appear as `cut_held_out_rows`; its absence means the default rule, which is "
    "still `predict()` at 0.5. "
    "The shape of the lever is fixed: it is an argmax of the goal metric and nothing else, so "
    "there is no asking for the cut that holds precision above some level, no per-class cut on "
    "a multiclass target, and no cut on a model without `predict_proba`. Under 200 held-back "
    "rows it is declined and said so, and so is a goal metric the cut cannot move — "
    "`roc_auc` and `pr_auc` are computed from the probabilities alone, so asking there "
    "spends nothing rather than spending the rows: the refusal comes before the carve. "
    "Whether it is worth an iteration is a question about the target's balance, and the "
    "measurements say so plainly. On imbalanced synthetic targets it moved "
    "validation `balanced_accuracy` and `f1` up, and it "
    "moved the same direction on the held-back test rows every time. On a 50:50 target every "
    "one of those measurements sat within noise of zero and several were negative — at a "
    f"balanced prior the default 0.5 is already near the argmax, and the {CUT_FRACTION:.0%} of rows is then "
    "paid for nothing. So read the card's `class_balance` first, and "
    "`balanced_accuracy_cut_headroom` second. "
    f"Two side effects to plan around rather than discover. The fit sees {CUT_FRACTION:.0%} fewer training "
    "rows, which is a real cost on a small split and which can drop the fit under the "
    "`early_stopping='auto'` row boundary described in the row-budget block — the "
    "`internal_validation` field reports both counts, so a comparison against an attempt that "
    "did not tune is visibly not a comparison of one lever. And `f1` chosen on the goal metric "
    "is not `balanced_accuracy` chosen on it: the two argmaxes are different cuts, so the "
    "metric the run is judged on is the one that moves, and the others can fall."
)
_REGRESSION_SCORING = _scoring_protocol(
    "an unstratified", " There are no strata because the target is continuous."
) + (
    "`r2`, `mae` and `rmse`, plus `train_r2`, `train_mae` and "
    "`train_<goal metric>`, and `train_val_gap` measured on the goal metric. That gap is "
    "always *how much worse validation is than training*, so a positive value means "
    "overfitting whichever way the goal metric runs — do not read it as a subtraction in "
    "one fixed order."
)

# Order carries meaning
CAN: tuple[str, ...] = (
    _ONE_ESTIMATOR,
    _HYPERPARAMS,
    _CLASS_WEIGHT,
    _SUBSAMPLE,
    _PREPROCESSING,
    _MISSINGNESS,
    _PIPELINE_SPEC,
    _MEMORY_HINTS,
    _CLASSIFICATION_SCORING,
    _CUT_DIAGNOSTICS,
    _THRESHOLD_TUNING,
    _CALIBRATION_DIAGNOSTICS,
    _LEVER_AXES,
)

REGRESSION_CAN: tuple[str, ...] = (
    _ONE_ESTIMATOR,
    _HYPERPARAMS,
    _SUBSAMPLE,
    _PREPROCESSING,
    _MISSINGNESS,
    _PIPELINE_SPEC,
    _MEMORY_HINTS,
    _REGRESSION_SCORING,
)

CAN_BY_TASK: dict[str, tuple[str, ...]] = {
    TASK_CLASSIFICATION: CAN,
    TASK_REGRESSION: REGRESSION_CAN,
}


# Moving to CAN: move markers too, never narrow
CANNOT: tuple[Capability, ...] = (
    Capability(
        name="cross_validation",
        summary="Run cross-validation or produce out-of-fold predictions.",
        instead=(
            # No "stratified": the scoring entry already says so.
            f"there is one {VAL_SHARE:.0%} validation split, fixed by the seed and shared with the "
            "card's baseline so the two numbers are comparable"
        ),
        markers=(
            "cross-validation",
            "cross validation",
            "cross-validated",
            "out-of-fold",
            "out of fold",
            "oof probabilit",
            "oof predict",
            "k-fold",
            "kfold",
            # No bare "-fold": "a ten-fold speedup" would match.
            " folds",
            "per-fold",
            "each fold",
            "-fold cv",
            "-fold stratified",
            "stratified cv",
            "cv_folds",
            "nested cv",
            "repeated cv",
        ),
    ),
    Capability(
        name="feature_engineering",
        summary=(
            "Derive, encode or drop features. The two missingness columns are the whole "
            "exception, and the CAN list above states them."
        ),
        instead=(
            "every numeric column of the card goes in as it is, and the only columns the "
            "executor will add are `missing_indicator` and `missing_count`. Nothing is "
            "combined except by the `interactions` step, which is pairwise products and "
            "nothing else: no ratio, no difference, no square, no re-encoding, and no "
            "dropping — a column you want out has to leave the card, not the plan"
        ),
        markers=(
            # No indicator or interaction markers
            "feature engineering",
            "engineered feature",
            "derived feature",
            "feature selection",
            "drop the feature",
            "drop features",
            "one-hot",
        ),
    ),
    Capability(
        name="resampling",
        summary="Resample the training set (SMOTE, over-sampling, under-sampling).",
        instead="every training row goes in exactly once; `train_subsample` only shrinks",
        markers=(
            "smote",
            "adasyn",
            "oversampl",
            "over-sampl",
            "undersampl",
            "under-sampl",
            "resampl",
        ),
    ),
    Capability(
        name="calibration",
        summary="Calibrate predicted probabilities.",
        instead=(
            "read `brier` and `calibration_error`, which are reported on every binary attempt: "
            "the miscalibration is measured for you, it just is not corrected. Ranking metrics "
            "(`roc_auc`, `pr_auc`) are calibration-free anyway, and the threshold that "
            "calibration would inform is not tunable here either"
        ),
        markers=(
            "calibratedclassifiercv",
            "probability calibration",
            "calibrate the probabilit",
            "isotonic regression",
            "platt scaling",
            "sigmoid calibration",
        ),
        # A continuous target has no probabilities.
        tasks=(TASK_CLASSIFICATION,),
    ),
    Capability(
        name="model_averaging",
        summary="Combine several models: stacking, blending, voting, seed averaging.",
        instead=(
            "one estimator per attempt. A tree *ensemble* like hist_gbdt or random_forest "
            "is a single estimator and is fine — combining separate attempts is not"
        ),
        markers=(
            "stacking",
            "stacked ensemble",
            "blending",
            "blend the",
            "voting classifier",
            "votingclassifier",
            "soft voting",
            "hard voting",
            "average the probabilit",
            "averaging the probabilit",
            "seed averaging",
            "average across seeds",
            "combine the two models",
        ),
    ),
    Capability(
        name="per_column_preprocessing",
        summary="Assemble your own column transformer, or scale and encode per column.",
        instead=(
            "imputation *is* per column — a `pipeline` step takes `groups`, and the CAN list "
            "above states the shape. What is still one decision for the whole matrix is "
            "scaling, and re-encoding is not available at all. So name the columns whose "
            "imputation differs; do not describe a transformer to build"
        ),
        markers=(
            # Name the missing object; per-column imputation is supported.
            "columntransformer",
            "column-specific scal",
            "per-column scal",
            "per column scal",
            "re-encode",
        ),
    ),
    Capability(
        name="xgboost_early_stopping",
        summary="Name your own `eval_set`, or pass `callbacks`, to xgboost.",
        instead=(
            "`early_stopping_rounds` works and needs nothing else from you — the executor "
            f"holds {DEFAULT_VALIDATION_FRACTION:.0%} of train back and supplies the eval set "
            "itself, the way hist_gbdt's `early_stopping=True` + `validation_fraction` does, and "
            f"reports both counts in `internal_validation`. Those {DEFAULT_VALIDATION_FRACTION:.0%} "
            "are rows the fit does not see, so on a small "
            "training split the loss can outweigh what the stop buys. What the executor "
            "cannot forward is an eval set naming rows it did not split, or a callback list"
        ),
        markers=("eval_set", "callbacks"),
    ),
    Capability(
        name="protocol_changes",
        summary="Change the split, the seed, the goal metric, or add a separate test set.",
        instead="those are the run's configuration, fixed before the loop starts",
    ),
)


CAPABILITIES_BY_NAME: dict[str, Capability] = {item.name: item for item in CANNOT}

# --- Role: negation words -------------------------------------------------------------

# Cues deny only what follows
NEGATION_CUES: tuple[str, ...] = (
    "without",
    # "not " and "n't " already cover "cannot" and "can't".
    "n't ",
    "not ",
    "no ",
    "never",
    "blocked",
    "unavailable",
    "unsupported",
    "instead of",
    "in place of",
    "rather than",
    "avoid",
    "skip",
    "forgo",
    "unable",
    "impossible",
)

# Deny the subject before them; verb-bound
DISCLAIMER_PREDICATES: tuple[str, ...] = (
    "is forbidden",
    "are forbidden",
    "is not available",
    "are not available",
    "is unavailable",
    "are unavailable",
    "is blocked",
    "are blocked",
    "is not supported",
    "are not supported",
    "is unsupported",
    "are unsupported",
    "is not permitted",
    "are not permitted",
    "is not allowed",
    "are not allowed",
    "is not possible",
    "are not possible",
    "is impossible",
    "are impossible",
    "is not an option",
    "are not an option",
    "cannot be used",
    "does not exist",
    "do not exist",
    # --- not "cannot get it" but "not worth using" -------------------------- #
    "is not worth",
    "are not worth",
    "would not be worth",
    "is negligible",
    "are negligible",
    "would be negligible",
    "buys nothing",
    "buys almost nothing",
    "buys next to nothing",
    "buys little",
    "is worthless",
    "are worthless",
    "is already near-optimal",
    "is already near optimal",
    "is already optimal",
    "has no headroom",
    "have no headroom",
    "gains nothing",
    "would gain nothing",
    "is not the bottleneck",
    "are not the bottleneck",
)

# How far a cue reaches on each side.
NEGATION_WINDOW = 60

# Cues stop at clause breaks; brackets are their own clause.
_CLAUSE_BREAK = re.compile(r"[.;:!?,()\[\]\n]")


# --- Role: prompt rendering -----------------------------------------------------------


def describe(task: str | None = None) -> str:
    """Build the can and cannot markdown for the planning, critic, and report prompts.

    A ``None`` task is read as classification.
    """
    lines = ["### The executor will do", ""]
    lines += [f"- {item}" for item in CAN_BY_TASK.get(task or TASK_CLASSIFICATION, CAN)]
    lines += ["", "### The executor will not do", ""]
    for item in CANNOT:
        if (task or TASK_CLASSIFICATION) not in item.tasks:
            continue
        lines.append(f"- **{item.summary}** Instead: {item.instead}.")
    lines += [
        "",
        "Do not build a plan on anything in the second list. A plan that assumes it "
        "spends an attempt and then reports a configuration that never ran.",
    ]
    return "\n".join(lines)


def describe_row_budget(n_rows: Any, *, grouped: bool = False) -> str:
    """Build the planning block on how many rows the fit really sees; never empty.

    Forecasts what ``scripts.train`` measures; a test ties them
    """
    try:
        counts = row_counts(int(n_rows))
    except (TypeError, ValueError):
        return (
            "(unknown — the card does not say how many rows the file has. In particular there "
            "is no way to tell which side of the "
            f"{EARLY_STOPPING_AUTO_MIN_ROWS:,}-training-row `early_stopping='auto'` boundary "
            "this dataset falls on, so name `early_stopping` explicitly if the plan depends on "
            "it either way.)"
        )
    train = counts["train"]
    held_out = math.ceil(DEFAULT_VALIDATION_FRACTION * train)
    auto_on = train > EARLY_STOPPING_AUTO_MIN_ROWS
    models = " and ".join(f"`{name}`" for name in SELF_VALIDATING_MODELS)

    lines = [
        f"- **train {train:,} rows** / val {counts['val']:,} / test {counts['test']:,}, out of "
        f"the card's {int(n_rows):,}. The card gives the fractions and the total but not this "
        "product, and capacity and early-stopping choices are made against the product.",
        f"- {models} early-stop on a slice of the **training** rows, not on the validation set "
        "above — so that slice is subtracted from the number in the first bullet, and an "
        "attempt that pays it is being compared on score against attempts that did not.",
        f"- `tune_threshold: true` subtracts a slice of the same rows for the same reason, and a "
        f"larger one: {CUT_FRACTION:.0%} of the first bullet, leaving "
        f"{train - math.ceil(CUT_FRACTION * train):,} to fit. That is the second thing on this "
        "list that quietly changes what the fit sees, and the two stack — the cut slice comes "
        "off first, and early stopping then takes its share of what is left.",
    ]
    if auto_on:
        lines.append(
            f"- Their default `early_stopping='auto'` is **on** here ({train:,} > "
            f"{EARLY_STOPPING_AUTO_MIN_ROWS:,}). Saying nothing about early stopping therefore "
            f"does not mean running without it: at the default `validation_fraction` of "
            f"{DEFAULT_VALIDATION_FRACTION:g} the fit sees {train - held_out:,} rows, not "
            f"{train:,}. Setting `early_stopping: false` is what turns it off; the executor "
            "reports what really happened in `internal_validation`."
        )
    else:
        lines.append(
            f"- Their default `early_stopping='auto'` is **off** here ({train:,} is not above "
            f"{EARLY_STOPPING_AUTO_MIN_ROWS:,}), so the fit sees all {train:,} rows unless the "
            f"plan asks for `early_stopping: true`. Asking costs `validation_fraction` of them "
            f"— {held_out:,} rows at the default {DEFAULT_VALIDATION_FRACTION:g}, leaving "
            f"{train - held_out:,}. On a training split this size that loss can outweigh what "
            "the stop buys; the executor reports both counts in `internal_validation`."
        )
    if grouped:
        lines.append(
            "- These three counts are **approximate**: the split keeps every row of a group "
            "together, and a group cannot be divided to make a share come out even."
        )
    return "\n".join(lines)


# --- Role: claim detection ------------------------------------------------------------


def _is_negated(haystack: str, start: int, end: int) -> bool:
    """_is_negated | Claim detection: True if the prose denies the phrase at ``start``:``end``.

    A cue before or a denying predicate after, within its own clause.
    """
    before = _CLAUSE_BREAK.split(haystack[max(0, start - NEGATION_WINDOW) : start])[-1]
    if any(cue in before for cue in NEGATION_CUES):
        return True
    after = _CLAUSE_BREAK.split(haystack[end : end + NEGATION_WINDOW])[0]
    return any(predicate in after for predicate in DISCLAIMER_PREDICATES)


def unsupported_claims(*texts: object) -> list[str]:
    """Return sorted names of missing capabilities that plan or verdict prose depends on.

    Never use on the report; one undenied match is enough
    """
    # Newlines stop a cue reaching into the next field.
    haystack = "\n".join(str(text).lower() for text in texts if text)
    if not haystack:
        return []
    found = []
    for item in CANNOT:
        for marker in item.markers:
            if any(
                not _is_negated(haystack, match.start(), match.end())
                for match in re.finditer(re.escape(marker), haystack)
            ):
                found.append(item.name)
                break
    return sorted(dict.fromkeys(found))


def explain_claims(names: list[str]) -> str:
    """Summarize capability names in one line for warnings and the fallback report."""
    parts = []
    for name in names:
        item = CAPABILITIES_BY_NAME.get(name)
        parts.append(f"{name}({item.summary.rstrip('.')})" if item else name)
    return ", ".join(parts)
