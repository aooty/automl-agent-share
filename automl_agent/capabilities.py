"""What the fixed executor can and cannot do, as one list the prompts and the plan
check both read.

The reasoning/execution split only holds if the reasoning side knows the executor's
shape, and it did not. Every real LLM run planned "add missing-indicator columns for
the high-missing features" and "pick the probability cut-off that maximises balanced
accuracy"; ``scripts/train.py`` does neither. When such a claim arrives as a
hyperparameter key it lands in ``dropped_hyperparams`` and the record stays honest, but
in prose it survives into the report's "최고 성능 구성" section as though it had run.

One run paid for it in score rather than only in bookkeeping: iteration 2 removed
``class_weight='balanced'`` on the strength of a threshold sweep that never happened,
and balanced_accuracy fell from 0.7863 to 0.6519 — a whole iteration spent on a plan
built around a capability that does not exist.

Claims are *flagged*, never used to reject a plan. Rejecting costs the same iteration
the flawed plan would have, and the markers below are substring matches: a false
positive that silently discarded a sound plan would be worse than the disclosure. Keep
markers narrow and multi-word for the same reason — ``threshold`` alone would fire on
"reach the goal threshold", which is the one threshold the run legitimately talks about.

The false positive turned out to be the common case, and injecting this list into the
prompt is what caused it: a Planner that read the list writes prose *about* the list.
All three flags of one three-iteration run were disclaimers — "shifts the operating
point without threshold tuning", "no eval_set / early_stopping_rounds / callbacks are
requested (blocked by the executor)", "the executor cannot add missing indicators" —
and the report duly explained the shortfall as "계획이 의도한 레버 중 실행되지 않은 것",
which was the exact inverse of what happened. Hence :func:`_is_negated`: a marker
governed by a negation cue in front of it is compliance, not a dependency.

Since that fix, **no flag has fired in 23 attempts across 10 real runs** — every
``unsupported_claims`` in every ``history.json`` is empty. Read carefully, that is
evidence for the injection rather than against the detector: before the list went into
the prompt, every run planned missing indicators and a probability cut-off, and after it,
none did. So the markers stay, unproven in the field by design; what makes them
trustworthy is that the tests pin them against the prose those early runs actually
produced, on both sides — the nine phrasings that must fire and the disclaimers that
must not.

One of those two claims has since changed sides, and how it changed is the point of this
file. Missing-indicator columns moved from ``CANNOT`` to ``CAN`` because a run finally
produced the evidence: five attempts across three model families held ``roc_auc`` inside
0.859~0.871 while ``cut_headroom`` fell to 0.003, which says the ceiling is the information
in the columns and not the search over them. The planner had asked for those columns in
every early run and been refused every time. So a capability list is not only a fence — a
limit the reasoning side keeps pushing against is a request for a feature, and the way to
answer it is a measurement, not a firmer no.

The sequel is inside ``_MISSINGNESS``. The sizes that first measurement produced did not
survive a second, harder split of the same file, so what this list now carries for that
capability is the mechanism and the check rather than the numbers. A measurement is how a
no becomes a yes; it is not a promise that the yes is worth an iteration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION


@dataclass(frozen=True)
class Capability:
    """One thing the executor will not do, and the nearest thing it will."""

    name: str
    summary: str
    instead: str
    # Lowercase substrings that indicate a plan is counting on this. Empty means the
    # limit is worth stating in the prompt but too fuzzy to detect without false alarms.
    markers: tuple[str, ...] = field(default_factory=tuple)
    # Which tasks the limit is worth *stating* for. A limit about a decision threshold is
    # not a limit on a continuous target, it is a category error, and printing it would
    # invite the planner to write about something the run has no notion of. Detection is
    # deliberately not filtered this way — see :func:`unsupported_claims`.
    tasks: tuple[str, ...] = (TASK_CLASSIFICATION, TASK_REGRESSION)


# What ``scripts/train.py`` actually does, in the order it does it. Written from the
# script, not from intent — ``run_training`` and ``build_estimator`` are the source.
#
# Named individually and composed below rather than written out twice, because the two
# tasks share most of this list and a second copy is a second thing to keep true.
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
    "`scale_pos_weight`, where that estimator supports it. This is the only imbalance "
    "lever available, and `'balanced'` is one point on it, not the best one: it pins the "
    "ratio at the class frequency. A map that does not cover every class exactly once is "
    "dropped into `dropped_hyperparams`."
)
_SUBSAMPLE = "Cut the training set with `train_subsample` (a float strictly between 0 and 1)."
# Slashes between words are avoided throughout this module: ``redact_paths`` runs over
# the rendered prompt and reads ``median/mean`` as an absolute path, so the list the
# planner reads would arrive as ``median<path>``.
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
    "natively under `impute: none` it is *redundant*, not merely weak — appending 14 "
    "indicator columns returned predictions identical bit for bit, because the indicator's "
    "only split is the NaN branch the tree already had. That reproduced under both of the "
    "splits below, so do not spend an iteration on it alongside `impute: none`. "
    "What it is worth when the plan *does* impute is not settled, and the reason is worth "
    "reading before spending an iteration on it. Under a stratified random split of one "
    "clinical sample the sizes were large and mutually consistent: imputing cost 0.0097 "
    "roc_auc, the indicator recovered 0.0065 of that in the same tree family, and on "
    "`logreg`, which cannot be handed a NaN at all, it gained 0.0151 — the largest of the "
    "three, which fits a linear model having no way to express *this value was invented* "
    "while a tree can by splitting at the imputed value. Refitting the same configurations "
    "on the same file under a contiguous split, so that training and validation rows fall "
    "on either side of a change in what the file records, all three collapsed to +0.0011, "
    "-0.0022 and +0.0017, every interval spanning 0. Absolute roc_auc barely moved (0.8709 "
    "to 0.8671), so what failed to carry over is the missingness lever specifically and not "
    "the models. The likeliest reading is that under the random split those columns were "
    "partly identifying which recording regime a row came from, which is not a subject's "
    "state; the contiguous split does not cleanly separate that from the weaker reading, "
    "since the column with the most missingness is observed in 83% of its training rows and "
    "8% of its validation rows and an indicator on it is nearly constant there. Either way "
    "the instruction is the same: on that sample the three sizes are not established, so do "
    "not cite them as what an iteration here will buy. `missing_count` bought nothing under "
    "the random split — 0.0003 on its own, 0.0000 on top of the indicator in either family — "
    "and was not refitted under the contiguous one. Being representable is not the same as "
    "being worth a column. What generalises is the check rather than the sizes: read the "
    "card's caveats for *why* values are missing, and where missingness tracks when or where "
    "a row was recorded rather than the subject's state, these columns let the model learn "
    "the recording regime — a random split scores that as a gain instead of showing it."
)
# Why the sizes are split by axis rather than listed as one ranking: mv-llm-2's own numbers
# invert between the two. Ordered by `balanced_accuracy`, the family swap is the biggest
# lever on the board (0.0526 between iterations 3 and 4) and preprocessing does not appear
# at all. Ordered by `roc_auc` on the same attempts, preprocessing is 0.0098 and the two
# family swaps are 0.0022 and 0.0077 — the reverse order. A single table would have to pick
# one of those, and whichever it picked would misprice the other axis.
#
# And the ranking axis gets sizes without an order, because the order was below the
# resolution of the measurement that produced it. The paired half-width for a roc_auc
# difference here is 0.0029 read off the indicator interval and 0.0059 read off the logreg
# one, while the gaps being ranked are 0.0021 (pipeline vs the larger family swap) and 0.0010
# (retuning vs the smaller one). The pipeline-vs-retuning gap, 0.0066, was the one comparison
# that cleared both readings and it was stated as one — until the imputation lever turned out
# to have no stable size to compare with (see ``_MISSINGNESS``). So the pipeline lever now
# carries no figure and no rank, and the entry says outright that no size here makes a lever
# "the largest available": mv-llm-6 iteration 3 used exactly that phrase off the old text,
# prescribed ``impute: median`` together with an untried family, and iteration 4 lost 0.0572.
# The decomposition needs no resolution at all — 0.0526 = 0.0066 + 0.0460 is subtraction.
_LEVER_AXES = (
    "Price a lever on the axis it moves, because two of them move independently and "
    "`balanced_accuracy` is their sum. The *ranking* axis is what `roc_auc`, `pr_auc` and "
    "`balanced_accuracy_at_best_cut` measure — how well the model orders rows, which no "
    "decision rule can improve. The *operating point* axis is where the default rule "
    "happens to cut that ranking, which is what `class_weight` and `scale_pos_weight` move "
    "and what `cut_headroom` measures the remaining size of. A `balanced_accuracy` "
    "difference on its own does not say which of the two moved, so it cannot rank levers. "
    "Measured on one clinical sample, the best and the worst of five attempts sat 0.0526 of "
    "`balanced_accuracy` apart (0.7334 to 0.7860), and that gap splits exactly in two: "
    "0.0066 of it was their `balanced_accuracy_at_best_cut` (0.7823 to 0.7889) and the other "
    "0.0460 was the cut landing somewhere else. 87% of the apparent swing was the operating "
    "point, and the worst attempt carried `cut_headroom` 0.0489 with recall 0.5197 against "
    "specificity 0.9471. On the ranking axis in the same measurements, retuning "
    "hyperparameters inside one family spanned 0.0032 of roc_auc, and swapping tree family "
    "under an unchanged pipeline moved 0.0077 for one family and 0.0022 for another. Read "
    "those as sizes, not as a ranking: the paired resolution of a roc_auc difference in these "
    "measurements is somewhere between 0.003 and 0.006, so the smaller family swap is not "
    "distinguishable from retuning, and there is no single size for 'swap the family' to "
    "budget against either — it spanned 0.0022 to 0.0077 depending on which family. The "
    "imputation lever is deliberately given no size at all: on this file it reads 0.0097 of "
    "roc_auc under a random split and +0.0011 with the interval spanning 0 under a contiguous "
    "one, so there is nothing stable to rank it by, and the missing-data entry above is where "
    "that is set out. No number here licenses calling a lever the largest available. One "
    "verdict read a size that way, prescribed the imputation change together with a family it "
    "had not tried, and the attempt lost 0.0572 of `balanced_accuracy` with two owners for "
    "the loss. What survives is not a lever ranking: the two axes have to be read apart, the "
    "order the `balanced_accuracy` column suggests is not the order the ranking axis has, and "
    "`cut_headroom` against the distance still to go is the number that says which axis a "
    "shortfall is on."
)
_CALIBRATION_DIAGNOSTICS = (
    "Report whether the predicted probabilities are worth reading as probabilities, without "
    "changing them: `brier` is the mean squared error of the positive-class probability, and "
    "`calibration_error` is the count-weighted distance between predicted and observed rate "
    "over ten bins — so 0.04 means the probabilities are off by four percentage points on "
    "average. Both are diagnostics no goal may target, and `calibration_error` is omitted "
    "under 50 rows because that estimator is biased upward on few rows; its absence means "
    "not measured, not fine. There is no recalibration lever here — nothing refits the "
    "probabilities, so do not propose `CalibratedClassifierCV` or a shifted cut. What the two "
    "numbers are for is reading a shortfall: a model can rank well and still be "
    "systematically overconfident, and no metric in the registry would show it."
)
_MEMORY_HINTS = (
    "Read `batch_size` and `precision: fp16` as *memory-budget hints only* — they "
    "change the pre-flight memory estimate, not how the model is fitted."
)
_CLASSIFICATION_SCORING = (
    "Score on a fixed protocol: a stratified three-way split at the run's seed — train "
    "60%, validation 20%, test 20% — where every score you are shown, and the choice of "
    "the run's best attempt, comes from the validation 20%. The test 20% is carved off "
    "first, is scored exactly once after the loop ends, and is never visible to a plan or "
    "a diagnosis. Reported on the validation split: every "
    "metric in the registry, plus `train_f1`/`train_accuracy`/`train_<goal metric>` "
    "and `train_val_gap` measured on the goal metric. On a binary target it also reports "
    "`specificity`, which no goal may target but which names the direction the imbalance "
    "lever has to move: `balanced_accuracy` is the mean of `recall` and `specificity`."
)
_CUT_DIAGNOSTICS = (
    "Report what choosing a decision threshold would have been worth, without choosing "
    "one: `balanced_accuracy_at_best_cut` is the best `balanced_accuracy` any cut of this "
    "model's ranking allows, and `cut_headroom` is the distance from the score actually "
    "achieved. Both are diagnostics no goal may target. Read `cut_headroom` before "
    "reaching for the operating point — when it is small the cut is near-optimal already "
    "and the remaining gap is in the ranking, which means the model family or the "
    "features, not `class_weight`."
)
# The same protocol, minus the two things a continuous target cannot have: strata, and a
# decision rule. The `train_val_gap` sentence is the one genuinely new claim — a raw
# difference would flip sign between `r2` and `mae`, so the executor normalises it.
_REGRESSION_SCORING = (
    "Score on a fixed protocol: an unstratified three-way split at the run's seed — train "
    "60%, validation 20%, test 20% — where every score you are shown, and the choice of "
    "the run's best attempt, comes from the validation 20%. There are no strata because "
    "the target is continuous. The test 20% is carved off first, is scored exactly once "
    "after the loop ends, and is never visible to a plan or a diagnosis. Reported on the "
    "validation split: `r2`, `mae` and `rmse`, plus `train_r2`, `train_mae` and "
    "`train_<goal metric>`, and `train_val_gap` measured on the goal metric. That gap is "
    "always *how much worse validation is than training*, so a positive value means "
    "overfitting whichever way the goal metric runs — do not read it as a subtraction in "
    "one fixed order."
)

CAN: tuple[str, ...] = (
    _ONE_ESTIMATOR,
    _HYPERPARAMS,
    _CLASS_WEIGHT,
    _SUBSAMPLE,
    _PREPROCESSING,
    _MISSINGNESS,
    _MEMORY_HINTS,
    _CLASSIFICATION_SCORING,
    _CUT_DIAGNOSTICS,
    # Next to the cut diagnostics because it is the other half of the same subject: what the
    # probabilities are worth, given that the harness will not pick a threshold over them.
    _CALIBRATION_DIAGNOSTICS,
    # After the two scoring entries: it is about how to read what they report, and the
    # operating-point half of it has no meaning until `cut_headroom` has been defined.
    _LEVER_AXES,
)

REGRESSION_CAN: tuple[str, ...] = (
    _ONE_ESTIMATOR,
    _HYPERPARAMS,
    _SUBSAMPLE,
    _PREPROCESSING,
    _MISSINGNESS,
    _MEMORY_HINTS,
    _REGRESSION_SCORING,
)

CAN_BY_TASK: dict[str, tuple[str, ...]] = {
    TASK_CLASSIFICATION: CAN,
    TASK_REGRESSION: REGRESSION_CAN,
}


CANNOT: tuple[Capability, ...] = (
    Capability(
        name="threshold_tuning",
        summary="Choose or search a decision threshold / probability cut-off.",
        instead=(
            "the executor always calls `predict()`, so the operating point is whatever "
            "the estimator's default rule gives. Move it with `class_weight='balanced'` "
            "or `scale_pos_weight`, and read `cut_headroom` first — it reports what the "
            "sweep you are about to propose would have been worth"
        ),
        markers=(
            "threshold sweep",
            "threshold search",
            "threshold tuning",
            "threshold_search",
            "tune the threshold",
            "tuning the threshold",
            "tuned threshold",
            "decision threshold",
            "probability threshold",
            "classification threshold",
            # "cut-off" alone was too broad: "recall 0.229 at the default 0.5 cut-off"
            # *diagnoses* the fixed operating point, which is exactly what the executor
            # does and what `class_weight` moves. Only choosing one is unavailable, so
            # the marker has to carry the choosing.
            "probability cut-off",
            "probability cutoff",
            "optimal cut-off",
            "optimal cutoff",
            "tuned cut-off",
            "tuned cutoff",
            "cut-off choice",
            "cutoff choice",
            "cut-off search",
            "cutoff search",
            "cut-off sweep",
            "cutoff sweep",
            "cut-off tuning",
            "cutoff tuning",
            "pick the cut-off",
            "pick the cutoff",
            "choose the cut-off",
            "choose the cutoff",
            "choose a cut-off",
            "choose a cutoff",
            "select the cut-off",
            "select the cutoff",
            "tune the cut-off",
            "tune the cutoff",
            "tuning the cut-off",
            "tuning the cutoff",
            "adjust the cut-off",
            "adjust the cutoff",
            "optimise the cut-off",
            "optimize the cut-off",
        ),
        # A continuous prediction has no decision rule to move, so on that side this is not
        # a limit to disclose. The markers still run — see :func:`unsupported_claims`.
        tasks=(TASK_CLASSIFICATION,),
    ),
    Capability(
        name="cross_validation",
        summary="Run cross-validation or produce out-of-fold predictions.",
        instead=(
            # "stratified" is left out rather than branched: whether the split has strata is
            # already stated once, by the scoring entry of the task's own ``CAN`` list, and
            # what matters here is that there is exactly one of them.
            "there is one 20% validation split, fixed by the seed and shared with the "
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
            # Counted folds, as they are actually written: "average the optimal cutoff
            # over 5 folds", "the per-fold optima". Bare "-fold" is left out — it also
            # matches "a ten-fold speedup".
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
            "combined: no ratio or difference between two columns, no interaction, no "
            "polynomial, no re-encoding, and no dropping — a column you want out has to "
            "leave the card, not the plan"
        ),
        markers=(
            # The missing-indicator markers were removed when the executor gained the
            # columns. Leaving them in would have flagged a plan for asking for a capability
            # it had just been told it has, which is the false positive this module treats as
            # worse than the miss.
            "feature engineering",
            "engineered feature",
            "derived feature",
            "interaction feature",
            "interaction term",
            "polynomial feature",
            "feature selection",
            "drop the feature",
            "drop features",
            "one-hot",
        ),
    ),
    Capability(
        name="resampling",
        summary="Resample the training set (SMOTE, over-sampling, under-sampling).",
        # The class_weight half of this sentence moved out: it is stated by the ``CAN`` entry
        # that owns it, and on a regression run it would name a lever that does not exist.
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
        # There are no predicted probabilities on a continuous target to calibrate.
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
        summary="Treat columns differently in preprocessing.",
        instead="one imputer and one scaler for the whole matrix",
        markers=(
            "columntransformer",
            "per-column impute",
            "per column impute",
            "different imputer for",
            "column-specific",
        ),
    ),
    Capability(
        name="xgboost_early_stopping",
        summary="Use `eval_set`, `early_stopping_rounds` or `callbacks` for xgboost.",
        instead=(
            "a Pipeline cannot forward an eval set, so these are blocked before `fit`. "
            "hist_gbdt's own `early_stopping=True` + `validation_fraction` does work"
        ),
        markers=("eval_set", "early_stopping_rounds", "callbacks"),
    ),
    Capability(
        name="protocol_changes",
        summary="Change the split, the seed, the goal metric, or add a separate test set.",
        instead="those are the run's configuration, fixed before the loop starts",
    ),
)


CAPABILITIES_BY_NAME: dict[str, Capability] = {item.name: item for item in CANNOT}

# Words that disclaim whatever follows them. Only what *follows* is disclaimed, which
# is the whole reason this is a directional check: "instead of a threshold sweep" gives
# the sweep up, "a threshold sweep instead of class_weight" asks for it.
NEGATION_CUES: tuple[str, ...] = (
    "without",
    "cannot",
    "can not",
    "can't",
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

# Predicates that disclaim the phrase *in front of* them, where the phrase is the
# subject: "indicator columns are forbidden". Kept separate from the cues above and
# deliberately short — a general forward scan would swallow "run a threshold sweep
# because the default cut is not optimal", which is a request.
#
# Two families, and the second one exists because of a measurement. Every predicate below
# the divider says the capability is not *worth* using rather than not available, which is
# prose the system now actively invites: ``scripts/train.py`` reports ``cut_headroom``, and
# the capability list tells the planner to read it before reaching for the operating point.
# A planner that complies writes "cut_headroom is 0.0024, so threshold tuning is not worth
# an iteration" — five such sentences were flagged as *dependencies* before this family
# existed, which is m-llm6's disclaimer defect arriving by a new road.
#
# All of them are verb-anchored on purpose, so the phrase has to be the subject. A bare
# "negligible" would silence "a threshold sweep at negligible cost", which is a request.
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
    # --- not worth using, as opposed to not available ------------------------ #
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

# How far a cue can reach, in either direction. Short on purpose: a cue further away
# than this is usually about something else in the same sentence.
NEGATION_WINDOW = 60

# A cue does not reach across a clause, and a parenthesis is its own clause — which is
# where the acknowledgement usually sits: "(the executor cannot add missing indicators)".
_CLAUSE_BREAK = re.compile(r"[.;:!?,()\[\]\n]")


def describe(task: str | None = None) -> str:
    """The markdown block injected into the planning, critic and report prompts.

    Rendered from :data:`CAN`/:data:`CANNOT` rather than written out in each template,
    so a template can never drift from what the executor does — the drift this whole
    module exists to close.

    ``task`` selects which executor is being described. ``None`` — a card that does not say
    — reads as classification, which is what every card written before the field existed is.
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


def _is_negated(haystack: str, start: int, end: int) -> bool:
    """True when the prose disclaims the phrase at ``start``:``end`` rather than asking for it.

    Two patterns, both taken from real plans, both limited to the phrase's own clause:

    * a cue in front of it — "without threshold tuning", "cannot add missing
      indicators", "no eval_set ... are requested". A general cue may not be read
      forwards: "a threshold sweep instead of class_weight" *is* a request for the
      sweep, and only "instead of" happens to sit after it.
    * a disclaiming predicate behind it, where the phrase is the subject —
      "indicator columns are forbidden".
    """
    before = _CLAUSE_BREAK.split(haystack[max(0, start - NEGATION_WINDOW) : start])[-1]
    if any(cue in before for cue in NEGATION_CUES):
        return True
    after = _CLAUSE_BREAK.split(haystack[end : end + NEGATION_WINDOW])[0]
    return any(predicate in after for predicate in DISCLAIMER_PREDICATES)


def unsupported_claims(*texts: object) -> list[str]:
    """Names of unavailable capabilities the given prose appears to count on.

    Scans plan/verdict prose, not the report: the report *should* be free to explain
    that a capability is missing, and flagging it there would invert the meaning.

    A mention the prose disclaims does not count — see :func:`_is_negated`. Prose that
    both disclaims a capability somewhere and relies on it elsewhere is still flagged:
    every occurrence is checked, and one un-negated one is enough.

    Not filtered by task, unlike :func:`describe`. A limit that is not worth *stating* for a
    task is still worth *catching* there: prose asking for a threshold sweep on a continuous
    target is more wrong, not less, and silencing the marker would make the one case that
    most needs saying out loud the one case that says nothing.
    """
    # Newline, not space: the fields are separate sentences, and a cue at the end of one
    # must not reach into the next.
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
    """One-line Korean summary for the console warning and the fallback report."""
    parts = []
    for name in names:
        item = CAPABILITIES_BY_NAME.get(name)
        parts.append(f"{name}({item.summary.rstrip('.')})" if item else name)
    return ", ".join(parts)
