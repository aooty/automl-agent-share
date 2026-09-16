"""What the fixed executor can and cannot do, as one list the prompts and the plan
check both read.

**Why it exists.** The reasoning/execution split only holds if the reasoning side knows the
executor's shape, and it did not. Every real LLM run planned "add missing-indicator columns for
the high-missing features" and "pick the probability cut-off that maximises balanced accuracy";
``scripts/train.py`` did neither at the time and does both now — the arc this file records. As a
hyperparameter key such a claim lands in ``dropped_hyperparams`` and the record stays honest; in
*prose* it survives into the report's "최고 성능 구성" as though it had run. One run paid in score
and not only in bookkeeping: iteration 2 removed ``class_weight='balanced'`` on the strength of a
threshold sweep that never happened, and balanced_accuracy fell 0.7863 → 0.6519.

**Flagged, never rejected.** Rejecting costs the same iteration the flawed plan would, and the
markers are substring matches — a false positive that silently discarded a sound plan would be
worse than the disclosure. Markers stay narrow and multi-word for the same reason: ``threshold``
alone fires on "reach the goal threshold", the one threshold a run legitimately talks about.

**Negation.** The false positive turned out to be the common case, and injecting this list into
the prompt caused it: a Planner that reads the list writes prose *about* the list. All three flags
of one three-iteration run were disclaimers ("shifts the operating point without threshold
tuning", "…blocked by the executor", "the executor cannot add missing indicators"), and the report
duly explained the shortfall as "계획이 의도한 레버 중 실행되지 않은 것" — the exact inverse of what
happened. Hence :func:`_is_negated`: a marker under a negation cue is compliance, not a dependency.

**Unproven in the field, by design.** Since that fix **no flag has fired in 23 attempts across 10
real runs** — every ``unsupported_claims`` is empty. That is evidence for the injection rather than
against the detector: before the list reached the prompt every run planned both of those things,
and after it none did. What makes the markers trustworthy is that the tests pin them against the
prose those early runs produced, on both sides — nine phrasings that must fire, and the
disclaimers that must not.

**How a CANNOT becomes a CAN — twice, the same way.** A limit the reasoning side keeps pushing
against is a feature request, and the answer is a measurement rather than a firmer no.

* *Missing indicators.* Five attempts across three families held ``roc_auc`` inside 0.859~0.871
  while ``balanced_accuracy_cut_headroom`` fell to 0.003 — the ceiling is the information in the
  columns, not the search over them. Sequel in ``_MISSINGNESS``: those sizes did not survive a
  second, harder split of the same file, so the entry now carries the mechanism and the check
  rather than the numbers.
  A measurement is how a no becomes a yes; it is not a promise the yes is worth an iteration.
* *Decision cut.* Here **the refusal was carrying the number that overturned it.**
  ``balanced_accuracy_cut_headroom`` was added to talk the planner *out* of the sweep, and it
  measured that 87% of the spread between the best and worst of five attempts was the cut
  position (``_LEVER_AXES``) — more than every lever the executor did offer.
  ``_THRESHOLD_TUNING`` is that number collected,
  and the *shape* of the yes came from a second measurement: choosing the cut on the rows the
  model was fitted on fails on a family that memorises them, so the yes holds rows back and costs
  the fit 20% of train. That entry therefore carries three things — mechanism, sizes with the
  balance they depend on, and price.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION
from .scoring.splits import row_counts


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
# Why the sizes are split by axis rather than listed as one ranking: on one run's own numbers the
# order *inverts* between the two axes. Read by `balanced_accuracy` the family swap is the biggest
# lever on the board and preprocessing does not appear at all; read by `roc_auc` on the same
# attempts preprocessing leads and the family swaps trail it. A single table would have to pick one
# of those, and whichever it picked would misprice the other axis. (`FINDINGS-mimic.md`.)
#
# And the ranking axis gets sizes without an order, because the gaps being ranked came out smaller
# than the paired half-width that produced them — the ordering was below the resolution of the
# measurement. One comparison did clear both readings and was stated as one, until the imputation
# lever turned out to have no stable size to compare with (see ``_MISSINGNESS``). So the pipeline
# lever now carries no figure and no rank, and the entry says outright that no size here makes a
# lever "the largest available": a planner quoted exactly that phrase off the old text, prescribed
# imputation together with an untried family, and the next attempt lost ground. The
# *decomposition* below needs no resolution at all — it is a subtraction, not a comparison.
_LEVER_AXES = (
    "Price a lever on the axis it moves, because two of them move independently and "
    "`balanced_accuracy` is their sum. The *ranking* axis is what `roc_auc`, `pr_auc` and "
    "`balanced_accuracy_at_best_cut` measure — how well the model orders rows, which no "
    "decision rule can improve. The *operating point* axis is where that ranking is cut, which "
    "is what `tune_threshold` moves directly and what `class_weight` and `scale_pos_weight` "
    "move sideways, and what `balanced_accuracy_cut_headroom` measures the remaining size of. "
    "A `balanced_accuracy` "
    "difference on its own does not say which of the two moved, so it cannot rank levers. "
    "Measured on one clinical sample, the best and the worst of five attempts sat 0.0526 of "
    "`balanced_accuracy` apart (0.7334 to 0.7860), and that gap splits exactly in two: "
    "0.0066 of it was their `balanced_accuracy_at_best_cut` (0.7823 to 0.7889) and the other "
    "0.0460 was the cut landing somewhere else. 87% of the apparent swing was the operating "
    "point, and the worst attempt carried `balanced_accuracy_cut_headroom` 0.0489 with recall 0.5197 against "
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
    "`balanced_accuracy_cut_headroom` against the distance still to go is the number that says which axis a "
    "shortfall is on. That 0.0460 is now collectable rather than only measurable — it is what "
    "`tune_threshold` reaches for — which changes what the decomposition is *for*: it no longer "
    "says the operating-point share is out of reach, it says how much of the shortfall the "
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
    "naming the high-missing columns for constant imputation moved `roc_auc` +0.0083 on "
    "`logreg` and +0.0073 on `hist_gbdt`, both intervals clear of zero; `interactions` moved "
    "+0.0134 on `logreg` and +0.0019 with the interval spanning zero on `hist_gbdt`, which fits "
    "— a tree can already express an interaction by splitting twice. Together on `logreg` they "
    "gave +0.0301, *more than their sum*, because the interaction step consumes the indicator "
    "columns the imputation step appended. That composition is the reason the order is yours. "
    "The sharpest thing the flags could not express, and the one worth reading before writing a "
    "spec: on `hist_gbdt` with `strategy: none` — NaN left for the tree to split on — plus "
    "`constant` on the three high-missing columns *and a `missing_indicator` naming only those "
    "three*, `roc_auc` was +0.0050 [+0.0018, +0.0082] over the best thing the flags could reach "
    "at all. Without that subset indicator the same spec was +0.0022 with the interval spanning "
    "zero. The mechanism is why: an indicator over *every* column is bit-for-bit redundant under "
    "`strategy: none`, since the tree already has the NaN branch — but the three columns just "
    "filled with a constant no longer have one, so an indicator on exactly those three restores "
    "what the fill destroyed and nothing else. "
    "Every gain above is on the *ranking* axis. Read them next to `_LEVER_AXES`: all of the "
    "`balanced_accuracy` intervals spanned zero, because a better ranking does not show at a "
    "fixed cut. What it does is raise `balanced_accuracy_at_best_cut` — the same column-wise "
    "imputation moved it +0.0076 on `hist_gbdt` — which is what `tune_threshold` then collects. "
    "One sample, one split. Do not quote these as what a step will buy here; quote them as the "
    "reason the step exists."
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
    "target. The executor then holds 20% of the *training* rows out of the fit, sweeps the "
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
    "measurements say so plainly. On imbalanced synthetic targets (9:1 and 5.7:1) it moved "
    "validation `balanced_accuracy` by +0.036 to +0.095 and `f1` by +0.02 to +0.15, and it "
    "moved the same direction on the held-back test rows every time. On a 50:50 target every "
    "one of those measurements sat within 0.016 of zero and several were negative — at a "
    "balanced prior the default 0.5 is already near the argmax, and the 20% of rows is then "
    "paid for nothing. So read the card's `class_balance` first, and "
    "`balanced_accuracy_cut_headroom` second. "
    "Two side effects to plan around rather than discover. The fit sees 20% fewer training "
    "rows, which is a real cost on a small split and which can drop the fit under the "
    "`early_stopping='auto'` row boundary described in the row-budget block — the "
    "`internal_validation` field reports both counts, so a comparison against an attempt that "
    "did not tune is visibly not a comparison of one lever. And `f1` chosen on the goal metric "
    "is not `balanced_accuracy` chosen on it: the two argmaxes are different cuts, so the "
    "metric the run is judged on is the one that moves, and the others can fall."
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
    # After both of those, because it is the ordered form of what they describe: the two entries
    # above name the transforms, and this one names how to sequence and scope them. Reading it
    # first would be reading a syntax before the things it sequences.
    _PIPELINE_SPEC,
    _MEMORY_HINTS,
    _CLASSIFICATION_SCORING,
    _CUT_DIAGNOSTICS,
    # After the diagnostics, not before: the first thing to do about the operating point is read
    # `balanced_accuracy_cut_headroom`, and an entry offering the lever ahead of the number that
    # prices it would
    # invite an iteration spent on a cut that is already near-optimal.
    _THRESHOLD_TUNING,
    # Next to the cut diagnostics because it is the other half of the same subject: what the
    # probabilities are worth, given that the harness will not pick a threshold over them.
    _CALIBRATION_DIAGNOSTICS,
    # After the two scoring entries: it is about how to read what they report, and the
    # operating-point half of it has no meaning until `balanced_accuracy_cut_headroom` has been defined.
    _LEVER_AXES,
)

REGRESSION_CAN: tuple[str, ...] = (
    _ONE_ESTIMATOR,
    _HYPERPARAMS,
    _SUBSAMPLE,
    _PREPROCESSING,
    _MISSINGNESS,
    # Available on a continuous target too: none of its steps is about a decision rule. The
    # measurements quoted inside it are on a binary sample, which the entry says.
    _PIPELINE_SPEC,
    _MEMORY_HINTS,
    _REGRESSION_SCORING,
)

CAN_BY_TASK: dict[str, tuple[str, ...]] = {
    TASK_CLASSIFICATION: CAN,
    TASK_REGRESSION: REGRESSION_CAN,
}


CANNOT: tuple[Capability, ...] = (
    # ``threshold_tuning`` used to be the first entry here, with the longest marker list in the
    # file. It moved to ``CAN`` as ``_THRESHOLD_TUNING`` when the executor learned to choose a
    # cut, which is the second capability to cross that way and the second time the measurement
    # came before the yes — see the module docstring. Its markers went with it rather than
    # staying behind on a narrower entry: what is left unavailable on that axis (a cut under a
    # precision constraint, a per-class cut, a cut on a model with no probabilities) is a *shape*
    # rather than a capability, and shapes are stated in the CAN text, where the planner reads
    # them alongside the lever they constrain. Substring markers cannot tell "tune the threshold"
    # asking for the argmax from "tune the threshold to hold precision at 0.9" asking for the
    # constraint, so a narrowed entry would have flagged the supported request and the
    # unsupported one identically — which is the false positive this file's own docstring argues
    # is worse than the miss.
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
            "combined except by the `interactions` step, which is pairwise products and "
            "nothing else: no ratio, no difference, no square, no re-encoding, and no "
            "dropping — a column you want out has to leave the card, not the plan"
        ),
        markers=(
            # The missing-indicator markers were removed when the executor gained the
            # columns. Leaving them in would have flagged a plan for asking for a capability
            # it had just been told it has, which is the false positive this module treats as
            # worse than the miss. The three interaction markers left for the same reason when
            # the ``pipeline`` spec arrived — including ``polynomial feature``, which is a
            # near miss rather than a match: the step is `interaction_only`, so it omits the
            # squares that word implies, and the CAN entry says so. A marker cannot make that
            # distinction and would flag the supported request to catch the unsupported one.
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
        summary="Assemble your own column transformer, or scale and encode per column.",
        instead=(
            "imputation *is* per column — a `pipeline` step takes `groups`, and the CAN list "
            "above states the shape. What is still one decision for the whole matrix is "
            "scaling, and re-encoding is not available at all. So name the columns whose "
            "imputation differs; do not describe a transformer to build"
        ),
        markers=(
            # ``per-column impute`` and ``different imputer for`` were removed when the
            # ``pipeline`` spec arrived: those are now requests the executor grants, and flagging
            # one would tell the planner the CAN list lied — the same reason the
            # missing-indicator markers left ``feature_engineering``. What is left names the
            # *object* rather than the outcome, which is the part that has no implementation.
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
            "holds 10% of train back and supplies the eval set itself, the way hist_gbdt's "
            "`early_stopping=True` + `validation_fraction` does, and reports both counts in "
            "`internal_validation`. Those 10% are rows the fit does not see, so on a small "
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
# prose the system now actively invites: ``scripts/train.py`` reports ``balanced_accuracy_cut_headroom``, and
# the capability list tells the planner to read it before reaching for the operating point.
# A planner that complies writes "balanced_accuracy_cut_headroom is small, so threshold tuning is not worth an
# iteration" — sentences of exactly that shape were flagged as *dependencies* before this family
# existed, which is the disclaimer defect above arriving by a new road.
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


# sklearn resolves ``early_stopping='auto'`` to ``n_samples > 10_000`` — strictly greater, and
# counted on the rows handed to ``fit``, which is the train split rather than the file. Named
# here because it is the one executor default whose *value* depends on the dataset: one config
# file means early stopping on one card and no early stopping on another.
EARLY_STOPPING_AUTO_MIN_ROWS = 10_000

# The estimators that carve their own validation slice out of the training rows, and the share
# they take when told to stop early and not told how much. Both are sklearn defaults, read off
# this module rather than written into a prompt so a version bump lands in one place.
SELF_VALIDATING_MODELS = ("hist_gbdt", "mlp")
DEFAULT_VALIDATION_FRACTION = 0.1

# The share of train ``tune_threshold`` holds out to choose the cut on. Restated here rather
# than imported for the reason ``nodes/training.py`` restates its alias map: this module is read
# by the orchestrator process, ``scripts.train`` reaches sklearn, and a prompt block is not worth
# that import. A test pins the two together — a forecast that disagrees with the executor by a
# row is worse than no forecast (see :func:`describe_row_budget`).
CUT_FRACTION = 0.2


def describe_row_budget(n_rows: Any, *, grouped: bool = False) -> str:
    """The planning prompt's block for how many rows the fit will actually see.

    Why the card is not enough. It publishes ``n_rows`` and it publishes
    ``baseline.protocol.train_fraction``, and it never publishes their product — but every
    choice a plan makes about capacity and early stopping is made against the product. The
    concrete cost: :data:`EARLY_STOPPING_AUTO_MIN_ROWS` is counted on training rows, so a file
    between 10,001 and 16,667 rows falls on *opposite* sides of that boundary depending on which
    of the two numbers is read. None of ``bench/``'s five datasets land in that window, which is
    exactly why reading the wrong one would have gone unnoticed there.

    The third bullet is the one worth the prompt space. ``early_stopping='auto'`` means a plan
    that says nothing about early stopping is not a plan that runs without it: above the
    boundary the fit silently holds back a tenth of the training rows. Two arms of the
    ``bench/`` comparison set ``validation_fraction: 0.1`` and inherited it respectively, and on
    the three large datasets those were the same treatment — which was written up as a
    difference until the fitted objects were read.

    On repeating the executor's arithmetic. :func:`automl_agent.scripts.train.describe_internal_validation`
    deliberately does *not* compute the held-out count, because it can ask the fitted object.
    This function has no fitted object — it runs before the attempt — so it forecasts, and a
    forecast that disagrees with the measurement by a row would be worse than none. The test
    suite pins the two against each other for that reason.

    Never empty, and never silently partial: a card without a usable ``n_rows`` gets a block
    saying which decision it can no longer inform.
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
