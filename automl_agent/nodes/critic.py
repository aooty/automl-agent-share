"""Result Critic: diagnose the failure as a structured verdict, never free text.

The schema is enforced at the API layer. If it still fails to parse after one
corrective retry, or the API is unreachable, we fall back to a deterministic
heuristic diagnosis rather than dropping the iteration — the loop must keep its
reasoning trail intact.

This node also appends the finished attempt to ``history``, with its verdict
attached. See ``state.build_attempt`` for why the append lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..capabilities import describe as describe_capabilities
from ..capabilities import explain_claims, unsupported_claims
from ..config import RunConfig
from ..dataset.caveats import describe_caveats
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.goal import describe as describe_goal
from ..scoring.goal import goal_threshold
from ..scoring.intervals import (
    PAIRED_KEY,
    PAIRED_SKIPPED,
    as_number,
    describe_paired,
    paired_of,
    resolution_note,
)
from ..scoring.metrics import METRICS, MINIMIZE, TASK_CLASSIFICATION, TASK_REGRESSION, direction_of, spec
from ..state import (
    FAILURE_TYPES,
    AutoMLState,
    build_attempt,
    drop_pinned_seed,
    effective_hyperparams,
    goal_met,
    is_better,
    metric_value,
)
from .model_selection import WEIGHT_RANGE, _history_digest, task_of_state

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "failure_type": {"type": "string", "enum": list(FAILURE_TYPES)},
        "evidence": {"type": "string"},
        "direction": {"type": "string"},
        "concrete_changes": {"type": "object", "additionalProperties": True},
    },
    "required": ["failure_type", "evidence", "direction", "concrete_changes"],
    "additionalProperties": False,
}

# A drop this big on the goal metric is a real regression, not seed noise; the ranking is
# allowed to slip by the tolerance and still count as held.
GOAL_METRIC_DROP = 0.05
RANKING_TOLERANCE = 0.005

# Overfitting gap, in two forms: absolute for a bounded metric, a fraction of the training score for
# one in the target's own units. An absolute gap there is an arbitrary number of dollars or days.
OVERFIT_GAP = 0.15
OVERFIT_GAP_RATIO = 0.25

# How far short of the bar the *training* score sits before the diagnosis is capacity rather than
# noise. Split for the same reason as the pair above.
UNDERFIT_MARGIN = 0.05
UNDERFIT_MARGIN_RATIO = 0.05

# The one diagnosis the canned directions could not express: a `balanced_accuracy` drop with
# `roc_auc` *unchanged or higher*. The ranking held and only the decision rule moved, so more
# capacity — what the heuristic used to prescribe — cannot move a threshold-dependent metric back.
# Measured instances are in ``FINDINGS-mimic.md``.
OPERATING_POINT_DIRECTION = (
    "랭킹 품질(roc_auc)은 유지됐으므로 용량이 아니라 운영점 문제다: 불균형 레버를 되살린다 — "
    'class_weight=\'balanced\' 또는 클래스별 가중치 맵({"0": 1, "1": 10}), xgboost면 '
    "scale_pos_weight. 용량을 더 키우는 것은 이 격차를 되돌리지 못한다."
)

# Metrics whose optimum sits where recall and specificity meet, so "close the gap" is right advice.
# **Deliberately not imported from ``ranking.SYMMETRIC_METRICS``**: same name, and there it marks an
# identity while here it marks an approximation to it (see :data:`CUT_HEADROOM_FLOOR`).
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})

# Below this the two sides are close enough that the remaining shortfall is not about where the
# operating point sits. Set above the skews observed on runs the branch *should* fire for, so it
# separates them rather than sitting inside their range (``FINDINGS-mimic.md``).
OPERATING_POINT_SKEW = 0.08

# **``balanced_accuracy_cut_headroom`` outranks the skew branch**, because it is the exact form of
# the premise that branch argues from and the two diverge on an asymmetric ROC curve. Below this
# floor the approximation does not get to prescribe a move. Rationale: ``docs/rationale.md``.
CUT_HEADROOM_FLOOR = 0.005

# What is left to say once the cut is optimal and even the best cut misses the bar. Not a
# guess: ``balanced_accuracy_at_best_cut`` below the threshold is a proof that no decision
# rule over *this* ranking reaches the goal, which is the same argument ``goal.describe``
# makes before the loop starts, applied to one attempt instead of the baseline.
RANKING_LIMIT_DIRECTION = (
    "운영점은 이미 최적이므로(balanced_accuracy_cut_headroom) 남은 격차는 컷이 아니라 랭킹에 있다: 이 랭킹의 "
    "어떤 임계값도 목표에 닿지 않으므로 모델 family를 바꾸거나 특성을 늘린다. 가중치나 "
    "임계값을 더 만지는 것은 이 격차를 줄이지 못한다."
)

# A search step, not an estimate — there is no closed form from a recall/specificity gap to the weight
# that closes it. Replaced by ``_interpolated_weight`` once two observations straddle the crossing,
# and **applies from the second rung on**: the first comes from the card (``_first_rung``).
WEIGHT_STEP = 1.5

# Bounded by what the sanitiser will actually pass through: a weight outside
# ``WEIGHT_RANGE`` is dropped before it reaches the executor, so prescribing one would
# spend the next iteration on a map that never gets applied.
MIN_WEIGHT, MAX_WEIGHT = WEIGHT_RANGE

# No weighting at all. ``_weight_value`` returns exactly this for an absent ``class_weight``
# and for a map that puts the same number on both codes, and both are the same thing to the
# estimator, so both take the first rung.
UNWEIGHTED = 1.0

# Training-level error types map straight through; no inference needed.
ERROR_TYPE_MAP: dict[str, str] = {
    "oom": "oom",
    "too_slow": "too_slow",
    "data_issue": "data_issue",
    "config_error": "data_issue",
    "unsupported_model": "wrong_model_family",
    "crash": "unknown",
    "no_result": "unknown",
    "exception": "unknown",
    # The environment failed, not the model: the iteration could not write its own config
    # (see nodes/training.py::_unwritable). Listed rather than left to the ``unknown``
    # default so nobody later files it under ``data_issue`` next to ``config_error`` — a
    # full disk is not something a column fix reaches.
    "write_failed": "unknown",
}


def critic(state: AutoMLState, *, config: RunConfig) -> dict:
    """Produce ``{failure_type, evidence, direction, concrete_changes}`` and log the attempt."""
    task = task_of_state(state)
    variables = {
        # What this node is being asked, which is not always "explain the shortfall": under
        # ``--search-past-goal`` the attempt may have cleared the bar. Rendered rather than
        # written into the template because the template's opening sentence used to assert the
        # miss — see describe_verdict_frame.
        "frame": describe_verdict_frame(state, config),
        "goal": state.get("goal") or {},
        # The bar in prose. It is the same dict above, but "this bar sits above the
        # baseline ranking's ceiling" is an obligation and ``"exceeds_ranking_ceiling":
        # true`` does not read as one — see nodes/planning.py for the run that proves it.
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "plan": state.get("plan") or {},
        "model": state.get("model") or "",
        "hyperparams": state.get("hyperparams") or {},
        "result": state.get("result") or {},
        "history": _history_digest(state),
        "best": state.get("best") or "(no successful attempt yet)",
        # The join between each verdict and the attempt it produced. Computed, because it is
        # arithmetic over ``history`` and instruction 4 asking the model to do it was not enough —
        # a run repeated one diagnosis and ended where it started (``FINDINGS-mimic.md``).
        "ledger": _ledger(state, config),
        "failure_types": ", ".join(FAILURE_TYPES),
        # A caveat can be the reason the score is where it is, and it rules out a
        # prescription that depends on what it invalidates — automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # How much of the difference it is about to explain the rows actually establish.
        # Handed over as a sentence, not as two more metric keys: the keys are already in
        # ``result``, and four iterations of a real run were spent diagnosing movement
        # inside the band because nothing said the band was there — automl_agent.scoring.intervals.
        "resolution": _resolution(state, config),
        "executor_capabilities": describe_capabilities(task),
    }

    iteration = int(state.get("iteration", 0) or 0) or None
    verdict: dict[str, Any] | None = None
    if not config.use_llm:
        archive_prompt_only(config, f"critic_iter{iteration or 0}", render_prompt("critic", variables))
    else:
        try:
            verdict = LLMClient(config).complete_json(
                "critic", variables, CRITIC_SCHEMA, iteration=iteration
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [critic] LLM 진단 실패({exc}) — 규칙 기반 진단으로 폴백합니다")

    verdict = validate_verdict(verdict) or heuristic_verdict(state, config)
    if verdict.get("unsupported_claims"):
        # A prescription the executor cannot carry out is how iteration 2 of a real run
        # was lost: the Planner dropped class_weight expecting a threshold sweep to
        # compensate, and the sweep does not exist.
        print(
            f"  [critic] 진단이 실행기에 없는 기능을 처방한 것으로 보입니다 — "
            f"{explain_claims(list(verdict['unsupported_claims']))}"
        )
    return {"critic": verdict, "history": [build_attempt(state, verdict)]}


def cleared_the_bar(state: AutoMLState, config: RunConfig) -> bool:
    """Whether the attempt being judged is already at or past the goal.

    Only reachable under :attr:`automl_agent.config.RunConfig.search_past_goal` — without it
    ``route`` sends a passing attempt straight to the report and this node never sees one. That is
    why the prose here no longer asserts a miss: a run clearing the bar at iteration 1 and then
    searching on would otherwise be told four times that it missed a bar it passed.

    Read off the same ``goal_met`` the router uses, on the same ``result`` channel, so the two cannot
    disagree about which side of the bar an attempt is on.
    """
    if not config.search_past_goal:
        return False
    return goal_met(dict(state.get("result") or {}), dict(state.get("goal") or {}))


def describe_verdict_frame(state: AutoMLState, config: RunConfig) -> str:
    """The prompt's opening: what the Critic is being asked about *this* attempt.

    **The first sentence cannot assert a miss unconditionally.** Everything after it is read in that
    light, and an LLM told a passing attempt failed will find a failure to report.
    Rationale: ``docs/rationale.md``.
    """
    if not cleared_the_bar(state, config):
        return (
            "The most recent training attempt did not reach the goal. Diagnose *why*, citing "
            "the numbers, and name one concrete change for the next attempt."
        )
    return (
        "The most recent training attempt **already cleared the bar**. The run is continuing "
        "because it was started with `--search-past-goal`, which spends the remaining iteration "
        "budget instead of stopping at the first pass.\n\n"
        "So there is no shortfall to explain, and you must not invent one. Say what is still "
        "worth trying, citing the numbers: where the remaining headroom is, and whether "
        "anything about this attempt is fragile — a wide train/validation gap or an interval "
        "that reaches back below the bar is worth naming even on a passing attempt, and a "
        "single passing score is exactly what hides it.\n\n"
        "`best` is chosen by validation score, so a next attempt that scores worse cannot cost "
        "the run its result. That is what makes this budget cheap to spend: prescribe the change "
        "that would teach the most, not the safest one."
    )


def validate_verdict(verdict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Coerce the model's answer into the schema; ``None`` when unusable.

    A ``failure_type`` outside the taxonomy degrades to ``unknown`` rather than
    poisoning the Planner's prompt with an invented category.
    """
    if not isinstance(verdict, dict):
        return None
    failure_type = str(verdict.get("failure_type") or "").strip().lower()
    if failure_type not in FAILURE_TYPES:
        failure_type = "unknown"
    changes = verdict.get("concrete_changes")
    evidence = str(verdict.get("evidence") or "").strip() or "(근거 없음)"
    direction = str(verdict.get("direction") or "").strip() or "(방향 제시 없음)"
    return {
        "failure_type": failure_type,
        "evidence": evidence,
        "direction": direction,
        "concrete_changes": changes if isinstance(changes, dict) else {},
        # ``direction`` is copied into the next planning prompt verbatim, so an
        # undeliverable prescription propagates unless it is marked here.
        "unsupported_claims": unsupported_claims(direction, evidence),
        "source": "llm",
    }


def heuristic_verdict(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Deterministic diagnosis from the numbers alone. Used by --dry-run and as fallback."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    threshold = goal_threshold(goal, config.fallback_threshold)
    result = dict(state.get("result") or {})
    metrics = dict(result.get("metrics") or {})
    history = list(state.get("history") or [])

    task = task_of_state(state)

    # 1. An explicit execution error already names the failure.
    if result.get("status") == "error":
        error_type = str(result.get("error_type") or "")
        failure_type = ERROR_TYPE_MAP.get(error_type, "unknown")
        direction = _direction_for(failure_type, task)
        return {
            "failure_type": failure_type,
            "evidence": f"학습이 status=error, error_type={error_type or 'unknown'}로 종료됨.",
            "direction": direction,
            "concrete_changes": _changes_for(failure_type, state, task),
            "unsupported_claims": unsupported_claims(direction),
            "source": "heuristic",
        }

    # 2. Otherwise read the train/validation numbers.
    measured = as_number(metric_value(result, metric))
    if measured is None:
        # **A missing score is not a perfect score.** ``0.0`` is the safe reading on a maximizing
        # metric and the *worst* one on an error metric, where it would say "no error at all" and send
        # the attempt down the overfitting branch. The bar is the neutral stand-in there.
        measured = threshold if direction_of(metric) == MINIMIZE else 0.0
    score = float(measured)
    train_score = metrics.get(f"train_{metric}")
    gap = metrics.get("train_val_gap")
    if gap is None and isinstance(train_score, (int, float)):
        # Normalised the way the executor normalises it — how much *worse* validation is —
        # so the sign means overfitting whichever way the metric runs.
        gap = (
            score - float(train_score)
            if direction_of(metric) == MINIMIZE
            else float(train_score) - score
        )

    # Whether this attempt is on the far side of the bar — only possible under
    # ``--search-past-goal``, and it changes what several of the branches below may claim.
    # Overfitting and the operating-point branches are deliberately *not* gated on it: a wide
    # train/validation gap is a real finding about a passing attempt, and it is the finding a
    # single passing score hides.
    cleared = cleared_the_bar(state, config)

    direction_override: str | None = None
    changes_override: dict[str, Any] | None = None
    if isinstance(gap, (int, float)) and _overfits(metric, float(gap), train_score):
        failure_type = "overfitting"
        evidence = (
            f"train_{metric}={train_score}, {metric}={score:.4f}, "
            f"train_val_gap={float(gap):.4f} — 검증이 학습보다 그만큼 나쁨."
        )
    elif (collapse := _operating_point_collapse(metric, score, metrics, history)) is not None:
        # Checked before underfitting, which is what this shape used to be mistaken for,
        # and after overfitting, whose evidence (a train/val gap) is independent of it.
        failure_type = "data_issue"
        evidence = collapse
        direction_override = OPERATING_POINT_DIRECTION
    elif (skew := _operating_point_skew(metric, metrics, state)) is not None:
        # Also before underfitting, and for a sharper reason: a skewed operating point depresses the
        # *train* score too, so "both low and close together" reads as missing capacity when the fix
        # is one weight.
        failure_type = "hyperparam"
        evidence, direction_override, changes_override = skew
    elif (
        not cleared
        and isinstance(train_score, (int, float))
        and _underfits(metric, float(train_score), threshold)
    ):
        # Gated on the miss, not just worded for it. An attempt whose validation score cleared
        # the bar is not capacity-starved whatever its training score says, and prescribing
        # *more* capacity there is the one direction that also widens a gap.
        failure_type = "underfitting"
        evidence = (
            f"train_{metric}={float(train_score):.4f}, {metric}={score:.4f} 모두 목표 {threshold}에 "
            f"닿지 못하고 격차도 작음 — 용량 부족."
        )
    elif _family_plateaued(history, state):
        failure_type = "wrong_model_family"
        evidence = (
            f"같은 계열 모델로 {len(history) + 1}회 시도했으나 {metric}={score:.4f}에서 더 오르지 "
            f"않음 — 목표 {threshold}는 이미 넘었고 남은 여유는 이 계열 안에 없어 보인다."
            if cleared
            else f"같은 계열 모델로 {len(history) + 1}회 시도했으나 {metric}={score:.4f}로 목표 "
            f"{threshold}에 정체됨."
        )
    elif (limited := _ranking_limited(metric, metrics, threshold)) is not None:
        # After the plateau check, whose evidence spans attempts and is therefore stronger,
        # and after the skew branch, which this one's gate has already silenced whenever
        # both would fire.
        failure_type = "wrong_model_family"
        evidence = limited
        direction_override = RANKING_LIMIT_DIRECTION
    elif cleared:
        # The branch --search-past-goal actually lands on most of the time. It has to be its
        # own case rather than the miss wording with a different number: "목표에 미달" about a
        # score above the bar is a false sentence, and it rides into the next planning prompt
        # as this verdict's ``evidence``.
        failure_type = "hyperparam"
        evidence = (
            f"{metric}={score:.4f}로 목표 {threshold}를 이미 넘었고 과적합 징후도 뚜렷하지 않다 — "
            f"고칠 실패가 없으므로 남은 예산은 같은 계열 안에서 여유를 더 찾는 데 쓴다. best는 "
            f"검증 최고로 고르므로 더 나쁜 다음 시도가 이 결과를 깎지 않는다."
        )
    else:
        failure_type = "hyperparam"
        evidence = f"{metric}={score:.4f}로 목표 {threshold}에 미달하나 과적합/과소적합 징후는 뚜렷하지 않음."

    direction = direction_override or _direction_for(failure_type, task)
    return {
        "failure_type": failure_type,
        "evidence": evidence,
        "direction": direction,
        "concrete_changes": changes_override or _changes_for(failure_type, state, task),
        "unsupported_claims": unsupported_claims(direction),
        "source": "heuristic",
    }


def _overfits(metric: str, gap: float, train_score: Any) -> bool:
    """Whether the train/validation gap is wide enough to call overfitting.

    ``gap`` arrives normalised to "how much worse validation is than training", so only scale is left.
    A bounded metric carries its own; ``mae``/``rmse`` are judged as a fraction of the training score,
    and a missing or zero score **declines rather than guessing at units**.
    Rationale: ``docs/rationale.md``.
    """
    found = spec(metric)
    if found is None or found.bounded:
        return gap > OVERFIT_GAP
    scale = as_number(train_score)
    if scale is None or scale <= 0.0:
        return False
    return gap > scale * OVERFIT_GAP_RATIO


def _underfits(metric: str, train_score: float, threshold: float) -> bool:
    """Whether even the training score misses the bar by enough to blame capacity.

    Direction-aware in the comparison and scale-aware in the slack: for an error metric
    "short of the bar" means sitting *above* it, and the slack has to be a fraction of the
    bar rather than a fixed 0.05 in units this module does not know.
    """
    if direction_of(metric) == MINIMIZE:
        return train_score > threshold * (1.0 + UNDERFIT_MARGIN_RATIO)
    return train_score < threshold - UNDERFIT_MARGIN


def _resolution(state: AutoMLState, config: RunConfig) -> str:
    """The ``resolution`` section of the prompt: this attempt's interval and what it swallows.

    The comparison set is everything the Critic reasons across: the bar it missed, and every prior
    attempt's score on the same metric. **A prior attempt with no score contributes nothing, not a
    zero** — a zero collides with every interval and reads as "indistinguishable from a crash".
    """
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    others: dict[str, Any] = {}
    threshold = goal.get("threshold")
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        others["목표"] = float(threshold)
    for attempt in state.get("history") or []:
        value = metric_value(dict(attempt.get("result") or {}), metric)
        if value is not None:
            others[f"iteration {attempt.get('iteration')}"] = value
    return resolution_note(dict(state.get("result") or {}).get("metrics"), metric, others=others)


def _ledger(state: AutoMLState, config: RunConfig) -> str:
    """The ``ledger`` section: what each prescription so far was actually worth.

    One row per *prescription*, not per attempt — iteration N's verdict produced iteration N+1, so
    the last row's verdict is the one being written now and has no score yet. The attempt under
    judgement is not in ``history`` (``critic`` appends it), so it is added here.

    ``paired_of`` is passed this ledger's own running baseline, never trusted to hold the right one:
    a row whose Δ was computed against a different iteration than the sentence claims is a line where
    every number is real and the claim is not.

    Steering only. No row here sees the test split, and none may become a gate on which iteration
    wins (:mod:`automl_agent.nodes.holdout`). Rationale: ``docs/rationale.md``.
    """
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = direction_of(metric)
    attempts: list[Mapping[str, Any]] = [
        *(state.get("history") or []),
        {
            "iteration": state.get("iteration"),
            "model": state.get("model"),
            # Through the same function the history rows went through, so "the hyperparameters
            # did not move" is a comparison of like with like. The proposal and the applied
            # block differ exactly when a key was dropped, and reading one row each way would
            # report a dropped key as a lever that moved.
            "hyperparams": effective_hyperparams(state),
            "result": state.get("result") or {},
            "critic": None,
        },
    ]

    lines: list[str] = []
    # Attempts per family, counting the ones that errored. An `oom` inside a family is an
    # iteration that family cost, and `wrong_model_family` is partly a claim about how many
    # of those have been spent — see ``_family_plateaued``.
    families: dict[str, list[float | None]] = {}
    # ``failure_type`` -> did any attempt it produced set a new best? A verdict that has
    # been issued twice and paid nothing either time is the exact shape of a stuck loop.
    paid: dict[str, bool] = {}
    best: float | None = None
    # Tracked, not derived afterwards: it is the key the paired block has to agree with for its Δ to
    # be about the same comparison as this row's subtraction.
    best_iteration: int | None = None
    # The last pipeline actually *built*, carried across attempts that errored — those have no
    # ``applied_preprocessing`` and treating the absence as a change reports the whole pipeline torn
    # out and put back. ``previous_built`` rides along because ``attempts[index - 1]`` is not the row
    # being compared against when something in between errored.
    previous_pipeline: dict[str, Any] = {}
    previous_built: Mapping[str, Any] | None = None
    # Rows whose transition moved the family *and* the pipeline. Collected rather than
    # decided per row, because what has to be said about them is one rule, not five copies
    # of it — see the footer line and :func:`_pipeline_change`.
    two_levers: list[str] = []
    # The same accounting from the other side: how many transitions moved the pipeline at all, and
    # which of those moved *nothing else*. Only the second kind is evidence about the preprocessing
    # lever, and single-lever transitions are rare in practice — most runs move the family alongside.
    pipeline_moves = 0
    pipeline_alone: list[str] = []
    for index, attempt in enumerate(attempts):
        score = metric_value(dict(attempt.get("result") or {}), metric)
        family = str(attempt.get("model") or "?")
        prior = dict(attempts[index - 1].get("critic") or {}) if index else {}
        prescription = str(prior.get("failure_type") or "") if prior else ""
        families.setdefault(family, []).append(score)
        parts = [f"iteration {attempt.get('iteration')}", f"{family:<13}"]
        if score is None:
            result = dict(attempt.get("result") or {})
            parts.append(f"점수 없음({result.get('error_type') or result.get('status') or 'no score'})")
        else:
            parts.append(f"{metric}={score:.4f}")
            if best is None:
                parts.append("첫 측정")
            elif is_better(score, best, direction):
                parts.append(f"직전 최고 대비 {score - best:+.4f} — 최고 갱신")
            else:
                parts.append(f"직전 최고 대비 {score - best:+.4f} — 갱신 못 함")
            paired_note = _paired_note(attempt, best_iteration)
            if paired_note:
                parts.append(paired_note)
        pipeline = _pipeline_of(attempt)
        if pipeline:
            changed = _pipeline_change(previous_pipeline, pipeline)
            if changed and previous_pipeline and previous_built is not None:
                parts.append(f"[파이프라인도 바뀜: {changed}]")
                pipeline_moves += 1
                if family != str(previous_built.get("model") or "?"):
                    two_levers.append(f"iteration {attempt.get('iteration')}")
                elif _other_levers_held(previous_built, attempt, config.seed):
                    pipeline_alone.append(f"iteration {attempt.get('iteration')}")
            previous_pipeline, previous_built = pipeline, attempt
        if prescription:
            parts.append(f"({prescription} 처방의 결과)")
            # The Planner is allowed to overrule the verdict, and has: a prescribed family swap can
            # come back as a different family that then beats everything. Crediting that score to
            # the prescription would be a lie in the direction that matters most, since it is the
            # row a reader takes as proof the diagnosis worked.
            asked = dict(prior.get("concrete_changes") or {}).get("model")
            if isinstance(asked, str) and asked and asked != family:
                parts.append(f"— 다만 처방은 {asked}였고 계획이 {family}로 바꿨다")
            # The same override on the axis the family note does not cover: a prescribed
            # preprocessing step can come back switched off, with nothing anywhere saying so — and
            # the next verdict would read that row as evidence about a column that never existed.
            dropped = _dropped_preprocessing(prior.get("concrete_changes"), pipeline)
            if dropped:
                parts.append(f"— 다만 처방의 전처리가 이 시도에 없다: {dropped}")
            gained = score is not None and is_better(score, best, direction)
            paid[prescription] = paid.get(prescription, False) or gained
        if is_better(score, best, direction):
            best, best_iteration = score, _iteration_of(attempt)
        lines.append("  " + "  ".join(parts))

    spent = []
    for name, scores in families.items():
        top: float | None = None
        for value in scores:
            if is_better(value, top, direction):
                top = value
        best_of = f"최고 {top:.4f}" if top is not None else "점수 없음"
        spent.append(f"{name} {len(scores)}회({best_of})")
    lines.append("")
    lines.append(f"  써 본 계열: {', '.join(spent) if spent else '없음'}")
    wasted = [name for name, gained in paid.items() if not gained]
    if wasted:
        lines.append(
            "  한 번 이상 처방했고 최고 점수를 갱신하지 못한 진단: "
            + ", ".join(sorted(wasted))
            + " — 같은 진단을 다시 내리려면 지난번과 무엇이 다른지 evidence에 적으십시오."
        )
    if two_levers:
        # **A marker, not a ban** — ``logreg`` cannot take ``impute: none``, so trying it *requires*
        # moving the pipeline in the same transition. What must not happen is that row's subtraction
        # being read as one lever's work. Rationale: ``docs/rationale.md``.
        lines.append(
            "  한 행에 레버가 둘인 전이: "
            + ", ".join(two_levers)
            + " — 계열과 전처리가 같은 전이에서 움직였으므로 그 행의 Δ는 어느 한쪽의 공로로 "
            "읽을 수 없습니다. 둘을 함께 움직이는 것이 맞을 때도 있으니(logreg는 대치를 "
            "강제합니다) 금지가 아니라 표시이고, 필요한 것은 그 뺄셈을 원인으로 읽지 않는 "
            "것입니다."
        )
    # **Silent when the pipeline never moved.** "You have not tried preprocessing yet" would be true
    # and would also be an invitation, and a prompt that lists an untried lever gets it tried. Only
    # moves that already happened are worth a line. Rationale: ``docs/rationale.md``.
    if pipeline_moves:
        if pipeline_alone:
            lines.append(
                "  전처리 레버 단독 전이: "
                + ", ".join(pipeline_alone)
                + " — 계열과 하이퍼파라미터가 그대로였으므로 그 행의 Δ는 전처리에 귀속됩니다. "
                "그 Δ가 0과 구분되는지는 같은 행의 짝지은 Δ가 말합니다."
            )
        else:
            lines.append(
                f"  전처리 레버 단독 전이: 없음 — 파이프라인이 바뀐 전이가 {pipeline_moves}건 "
                "있지만 모두 계열이나 하이퍼파라미터가 같이 움직였으므로, 이 히스토리에는 "
                "전처리 레버에 귀속되는 증거가 한 줄도 없습니다. 그 레버를 여전히 원한다면 "
                "그것만 바꾸는 전이를 처방하십시오. 원하지 않는다면 재본 적이 없다는 사실이 "
                "크기를 추정할 근거가 되지는 않습니다."
            )
    cut_note = _cut_lever_note(state, goal, metric, config)
    if cut_note:
        lines.append(cut_note)
    ceiling_note = _ranking_ceiling_note(attempts, goal, metric, config)
    if ceiling_note:
        lines.append(ceiling_note)
    remaining = max(0, int(config.max_iterations) - len(attempts))
    lines.append(f"  남은 iteration: {remaining}")
    return "\n".join(lines)


def _iteration_of(attempt: Mapping[str, Any]) -> int | None:
    value = attempt.get("iteration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _paired_note(attempt: Mapping[str, Any], baseline_iteration: int | None) -> str:
    """This row's paired verdict, or the reason it has none. ``""`` when there is nothing.

    Three outcomes, and each is said out loud rather than left to silence. A measured
    comparison against the ledger's own baseline renders as a verdict. A *skipped* one
    renders its reason, because "not compared" and "compared, found nothing" are different
    facts — except for ``no_baseline``, where the row already reads 첫 측정 and repeating it
    would be noise. A comparison that was measured against a *different* baseline renders as
    a mismatch: that is a bookkeeping defect, and a defect hidden by an empty string is one
    nobody finds.
    """
    result = dict(attempt.get("result") or {})
    measured = paired_of(result, baseline_iteration=baseline_iteration)
    if measured is not None:
        return describe_paired(measured)
    block = result.get(PAIRED_KEY)
    if not isinstance(block, Mapping):
        return ""
    if block.get("status") == PAIRED_SKIPPED:
        return "" if str(block.get("reason")) == "no_baseline" else describe_paired(block)
    against = block.get("baseline_iteration")
    return (
        f"[짝지은 검정 제외: 이 행의 기준은 iteration {baseline_iteration}인데 "
        f"짝은 iteration {against}에 대해 계산됐습니다]"
    )


def _pipeline_of(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """The pipeline the executor really built for this attempt, or ``{}``.

    ``applied_preprocessing``, not the plan's ``preprocessing`` block: the two differ
    exactly when a request was downgraded, and the downgrade is the thing worth reporting.
    """
    applied = dict(attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


# The preprocessing settings a verdict can prescribe, in the two shapes verdicts write them:
# nested under ``preprocessing``, and flat beside the hyperparameters. Both appeared in real
# runs, and reading only one of them would report a dropped request as honoured.
_PRESCRIBABLE_PREPROCESSING = ("impute", "scale", "missing_indicator", "missing_count")


def _prescribed_preprocessing(changes: Any) -> dict[str, Any]:
    """The preprocessing keys a ``concrete_changes`` dict asks for, nested or flat."""
    if not isinstance(changes, Mapping):
        return {}
    asked: dict[str, Any] = {}
    for key in _PRESCRIBABLE_PREPROCESSING:
        if key in changes:
            asked[key] = changes[key]
    nested = changes.get("preprocessing")
    if isinstance(nested, Mapping):
        for key in _PRESCRIBABLE_PREPROCESSING:
            if key in nested:
                asked[key] = nested[key]
    return asked


def _setting_matches(asked: Any, got: Any) -> bool:
    """``"none"`` against ``none``, ``True`` against ``true``: prose, not identity."""
    if isinstance(asked, bool) or isinstance(got, bool):
        return bool(asked) is bool(got)
    return str(asked).strip().lower() == str(got).strip().lower()


def _dropped_preprocessing(changes: Any, pipeline: Mapping[str, Any]) -> str:
    """Preprocessing a verdict prescribed that the attempt it produced did not carry.

    Against the pipeline that was really built, so the note covers both ways it can go
    missing — the Planner overruling the verdict, and the executor downgrading a request the
    family cannot take. Which of the two happened is not claimed here; what the next verdict
    needs is that the setting was not in the attempt it is about to read as the prescription's
    result. Reasoning about a column that was never added is reasoning about a run that did not
    happen. The family side of the same override is reported alongside it.
    """
    asked = _prescribed_preprocessing(changes)
    if not asked or not pipeline:
        return ""
    missing = [
        f"{key}: {_setting(value)} 처방 → {_setting(pipeline.get(key))}"
        for key, value in asked.items()
        if not _setting_matches(value, pipeline.get(key))
    ]
    return ", ".join(missing)


def _other_levers_held(
    previous: Mapping[str, Any], current: Mapping[str, Any], seed: int | None = None
) -> bool:
    """Whether the hyperparameters are the same across two attempts, so only the pipeline moved.

    The caller compares the family; this is the other half, and not a formality — retuning *and*
    changing the pipeline gives a row two owners on one axis, exactly as a family swap does.

    **A missing ``hyperparams`` key reads "cannot confirm held", not "held"**, so the only rows
    refused are ones whose record does not say. Rationale: ``docs/rationale.md``.
    """
    if "hyperparams" not in previous or "hyperparams" not in current:
        return False
    return drop_pinned_seed(previous.get("hyperparams"), seed) == drop_pinned_seed(
        current.get("hyperparams"), seed
    )


def _pipeline_change(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """``impute: none → median`` for every key whose applied value moved.

    A row that moved family *and* pipeline spent two levers and the ledger's subtraction cannot
    separate them. Naming the change here is not the decomposition — it stops the row reading as one
    clean lever. Rationale: ``docs/rationale.md``.
    """
    moved = sorted(key for key in {*previous, *current} if previous.get(key) != current.get(key))
    return ", ".join(
        f"{key}: {_setting(previous.get(key))} → {_setting(current.get(key))}" for key in moved
    )


def _setting(value: Any) -> str:
    """A preprocessing value as the plan writes it: JSON spelling, and ``(없음)`` for absent."""
    if value is None:
        return "(없음)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cut_lever_note(
    state: AutoMLState, goal: Mapping[str, Any], metric: str, config: RunConfig
) -> str | None:
    """``balanced_accuracy_cut_headroom`` put next to the distance still to go, as a share of it.

    Both numbers were already in the prompt, just not next to each other; the *ratio* is what decides
    whether the cut is worth an iteration, and it is a division, so it is computed rather than asked
    for.

    ``balanced_accuracy`` only, on the same set as :func:`_ranking_limited` and the skew branch.
    ``balanced_accuracy_cut_headroom`` is a ``balanced_accuracy`` measurement whatever the goal
    metric is (``scripts/train.py`` computes it against that one ceiling), while ``shortfall`` below
    is in the goal metric's own units — so on any other goal this ratio divides one metric's headroom
    by another metric's distance and reports the quotient as a share. This guard is the one the other
    two consumers of the number already had; its absence here was the omission, not a rule of its
    own. Found by an ``f1`` run in which two plans reached for the cut citing the headroom: the line
    itself never rendered, but only because that run's bar was already met.

    The set is maximizing, so it subsumes the minimizing check this replaces — on a minimizing
    metric the shortfall runs the other way and ``balanced_accuracy_cut_headroom`` does not exist at all.
    Rationale: ``docs/rationale.md``.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    metrics = dict(dict(state.get("result") or {}).get("metrics") or {})
    headroom = as_number(metrics.get("balanced_accuracy_cut_headroom"))
    score = as_number(metric_value(dict(state.get("result") or {}), metric))
    if headroom is None or score is None:
        return None
    shortfall = goal_threshold(dict(goal), config.fallback_threshold) - score
    if shortfall <= 0:
        return None
    share = headroom / shortfall
    head = (
        f"  운영점 레버의 크기: balanced_accuracy_cut_headroom {headroom:.4f} 대 목표까지 남은 거리 "
        f"{shortfall:.4f} — 임계값과 클래스 가중치로 살 수 있는 최대치는 남은 거리의 "
        f"{share:.0%}"
    )
    # A headroom that covers the whole distance means the ranking is already good enough and
    # only the cut is in the way, which is the opposite prescription — so it must not be
    # reported as a negative remainder.
    if share >= 1:
        return head + "이므로, 이 격차는 랭킹이 아니라 운영점에 있습니다."
    return head + f"이고, 나머지 {1 - share:.0%}는 랭킹에 있습니다."


def _ranking_ceiling_note(
    attempts: Sequence[Mapping[str, Any]],
    goal: Mapping[str, Any],
    metric: str,
    config: RunConfig,
) -> str | None:
    """How far the bar is, in units of how much swapping families has actually moved.

    **The span is a range over N maxima, not a paired comparison, and the line says so.** Under a null
    where the families rank identically a handful of maxima spread on their own by about as much, so
    the span is never evidence that the families differ.

    Restricted to :data:`SYMMETRIC_METRICS` (the ceiling identity exists only there) and to a ceiling
    that still misses the bar — above it :func:`_cut_lever_note` owns the case.
    Rationale: ``docs/rationale.md``.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    # Best ceiling per family, so a family tried three times contributes one value: the span
    # is meant to be across families, and counting repeats inside one would let
    # hyperparameter noise widen the number that stands for the family lever.
    ceilings: dict[str, float] = {}
    for attempt in attempts:
        metrics = dict(dict(attempt.get("result") or {}).get("metrics") or {})
        ceiling = as_number(metrics.get("balanced_accuracy_at_best_cut"))
        if ceiling is None:
            continue
        family = str(attempt.get("model") or "?")
        ceilings[family] = max(ceilings.get(family, ceiling), ceiling)
    if len(ceilings) < 2:
        return None
    low, high = min(ceilings.values()), max(ceilings.values())
    span = high - low
    threshold = goal_threshold(dict(goal), config.fallback_threshold)
    shortfall = threshold - high
    if shortfall <= 0:
        return None
    line = (
        f"  랭킹 상한의 산포: 계열 {len(ceilings)}개의 balanced_accuracy_at_best_cut이 "
        f"{low:.4f}~{high:.4f}(폭 {span:.4f})이고, 컷을 최적으로 골라도 바까지 남는 거리는 "
        f"{shortfall:.4f}입니다"
    )
    if span > 0:
        # No decimals on the multiple: one significant figure is all the span supports, and
        # "5.0배" would read as a measured ratio.
        line += f" — 부족분이 그 폭의 {round(shortfall / span)}배입니다"
    line += (
        f". 폭은 계열 {len(ceilings)}개의 최댓값과 최솟값의 차이이지 짝지은 비교가 아니므로, "
        "계열 사이에 차이가 있다는 근거로 쓰지 마십시오."
    )
    # KS is affine in the ceiling (``ranking.best_cut_ceiling`` is ``(1 + KS) / 2``), so this
    # inverts to the same statement in ranking-quality units and creates no new comparison.
    line += (
        f" 같은 말을 KS로 하면 바의 요구치가 {2 * threshold - 1:.4f}이고 "
        f"지금 최고는 {2 * high - 1:.4f}입니다."
    )
    return line


def _operating_point_collapse(
    metric: str,
    score: float,
    metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> str | None:
    """Evidence that only the decision rule moved, or ``None``.

    Restricted to threshold-dependent metrics: ``roc_auc`` and ``pr_auc`` *are* the
    ranking, so a drop in one of those is the opposite of this finding. ``roc_auc`` is the
    witness because the executor emits it for every binary attempt that can produce
    probabilities, alongside whatever the goal metric is.
    """
    spec = METRICS.get(metric)
    ranking = metrics.get("roc_auc")
    if spec is None or spec.needs_proba or not isinstance(ranking, (int, float)):
        return None
    for attempt in history:
        prior = dict(attempt.get("result") or {})
        was = metric_value(prior, metric)
        ranking_was = metric_value(prior, "roc_auc")
        if was is None or ranking_was is None:
            continue
        if was - score > GOAL_METRIC_DROP and float(ranking) >= ranking_was - RANKING_TOLERANCE:
            return (
                f"{metric}={score:.4f}로 iteration {attempt.get('iteration')}의 {was:.4f}보다 "
                f"{was - score:.4f} 낮은데 roc_auc는 {float(ranking):.4f} vs {ranking_was:.4f}로 "
                f"유지됨 — 순위는 그대로이고 판정 규칙만 움직였다."
            )
    return None


def _operating_point_skew(
    metric: str, metrics: Mapping[str, Any], state: AutoMLState
) -> tuple[str, str, dict[str, Any]] | None:
    """``(evidence, direction, concrete_changes)`` when the two halves are lopsided.

    Restricted to :data:`SYMMETRIC_METRICS`; ``specificity`` is binary-only, so its presence doubles
    as the binary guard that lets the prescription name codes 0 and 1.
    Rationale: ``docs/rationale.md``.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    # The measurement outranks the approximation. Advisory both ways: a multiclass attempt, and
    # every result written before this number carried this name, hold no
    # ``balanced_accuracy_cut_headroom`` — those keep the old behaviour rather than losing the
    # branch.
    headroom = as_number(metrics.get("balanced_accuracy_cut_headroom"))
    if headroom is not None and headroom < CUT_HEADROOM_FLOOR:
        return None
    skew = _skew(metrics)
    if skew is None or abs(skew) < OPERATING_POINT_SKEW:
        return None
    recall, specificity = float(metrics["recall"]), float(metrics["specificity"])

    weight = _positive_weight(state)
    # One step by default; the interpolation replaces it as soon as there is a bracket, and
    # is skipped if it rounds to where the weight already is — a no-op prescription would
    # spend an iteration re-measuring the same point. From an unweighted attempt the card
    # names the rung instead of the step — see ``_first_rung``.
    from_nothing = skew < 0 and weight == UNWEIGHTED
    proposed = _clamp_weight(
        _first_rung(state)
        if from_nothing
        else weight * (WEIGHT_STEP if skew < 0 else 1 / WEIGHT_STEP)
    )
    how = ""
    if from_nothing and proposed != _clamp_weight(WEIGHT_STEP):
        # Where the number came from, because the next attempt is planned off this sentence and
        # "1 → 5.07" with no source reads as a guess. Only when the card is what chose it: on a
        # nearly balanced card ``_first_rung`` returns the step, and saying otherwise would put
        # a false attribution into the prompt.
        how = (
            f"가중치가 없던 시도이므로 한 스텝을 밟는 대신 카드가 적은 클래스 불균형 비율 "
            f"{_frequency_ratio(state):g}에서 시작한다. "
        )
    bracket = _interpolated_weight([*_weight_history(state), (weight, skew)])
    if bracket is not None and _clamp_weight(bracket[0]) != _clamp_weight(weight):
        crossing, below, above = bracket
        proposed = _clamp_weight(crossing)
        how = (
            f"가중치 {below[0]:g}에서 {below[1]:+.4f}, {above[0]:g}에서 {above[1]:+.4f}로 "
            f"부호가 뒤집혔으므로 한 스텝 더 밟는 대신 그 사이를 보간한다. "
        )
    low, high = ("recall", "specificity") if skew < 0 else ("specificity", "recall")
    evidence = (
        f"{metric}={(recall + specificity) / 2:.4f}는 recall={recall:.4f}와 "
        f"specificity={specificity:.4f}의 평균이고 둘의 차이가 {abs(skew):.4f}다 — "
        f"{low}가 낮아 운영점이 한쪽으로 기울어 있다."
    )
    direction = (
        f"{metric}는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. "
        f"{low}({min(recall, specificity):.4f})가 {high}({max(recall, specificity):.4f})보다 낮으니 "
        f"양성 클래스 가중치를 {'올린다' if proposed > weight else '내린다'}: "
        f"{weight:g} → {proposed:g}. "
        f"{how}"
        # 슬래시 없이 씁니다 — ``redact_paths``가 "train/validation"을 경로로 읽습니다.
        "train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다."
    )
    return evidence, direction, {"class_weight": {"0": 1, "1": proposed}}


def _ranking_limited(
    metric: str, metrics: Mapping[str, Any], threshold: float
) -> str | None:
    """Evidence that the ranking, not the decision rule, is what falls short.

    **Both facts are required, and that is what keeps it a proof rather than a heuristic:** the cut is
    already within :data:`CUT_HEADROOM_FLOOR` of this ranking's best, *and* that best is still under
    the bar. Either alone leaves the operating point worth fixing first — the skew branch's case.
    Rationale: ``docs/rationale.md``.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    headroom = as_number(metrics.get("balanced_accuracy_cut_headroom"))
    ceiling = as_number(metrics.get("balanced_accuracy_at_best_cut"))
    if headroom is None or ceiling is None:
        return None
    if headroom >= CUT_HEADROOM_FLOOR or ceiling >= threshold:
        return None
    return (
        f"balanced_accuracy_cut_headroom={headroom:.4f}로 운영점은 이미 최적인데 "
        f"balanced_accuracy_at_best_cut={ceiling:.4f}가 목표 {threshold}보다 낮다 — "
        f"이 랭킹은 어떤 임계값으로도 목표에 닿지 못한다."
    )


def _skew(metrics: Mapping[str, Any]) -> float | None:
    """``recall - specificity``: negative means the positive class needs more weight."""
    recall = metrics.get("recall")
    specificity = metrics.get("specificity")
    if not isinstance(recall, (int, float)) or not isinstance(specificity, (int, float)):
        return None
    return float(recall) - float(specificity)


def _weight_history(state: AutoMLState) -> list[tuple[float, float]]:
    """``(positive weight, skew)`` for every prior attempt of the *same* model.

    **Same model only.** A point from another family says nothing about where this family's operating
    point sits, and interpolating across the two places the crossing somewhere neither model has
    been. Rationale: ``docs/rationale.md``.
    """
    model = str(state.get("model") or "")
    points: list[tuple[float, float]] = []
    for attempt in state.get("history") or []:
        if str(attempt.get("model") or "") != model:
            continue
        skew = _skew(dict((attempt.get("result") or {}).get("metrics") or {}))
        if skew is None:
            continue
        # ``history`` carries the applied set (see ``state.effective_hyperparams``), which
        # is what these numbers came from.
        points.append((_weight_value(dict(attempt.get("hyperparams") or {}), state), skew))
    return points


def _interpolated_weight(
    points: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
    """Where the skew crosses zero, when two observed weights straddle it.

    Skew rises with the positive weight, so a sign change brackets the optimum and a secant through
    the two tightest points beats another blind step.

    ``None`` without a bracket, **or when the two points are not ordered as monotonicity requires** —
    capacity changes between attempts can invert them, and a secant through inverted points points
    the wrong way. Rationale: ``docs/rationale.md``.
    """
    below = max((point for point in points if point[1] < 0), key=lambda p: p[1], default=None)
    above = min((point for point in points if point[1] > 0), key=lambda p: p[1], default=None)
    if below is None or above is None or below[0] >= above[0]:
        return None
    span = above[1] - below[1]
    crossing = below[0] + (above[0] - below[0]) * (-below[1]) / span
    return crossing, below, above


def _clamp_weight(value: float) -> float:
    return round(min(max(value, MIN_WEIGHT), MAX_WEIGHT), 3)


def _positive_weight(state: AutoMLState) -> float:
    """The weight the last attempt actually put on the positive class.

    ``applied_hyperparams`` first: a proposal the executor dropped is not what produced
    these numbers.
    """
    applied = dict((state.get("result") or {}).get("applied_hyperparams") or {})
    return _weight_value(applied or dict(state.get("hyperparams") or {}), state)


def _weight_value(params: Mapping[str, Any], state: AutoMLState) -> float:
    """Read one hyperparameter set's weight on the positive class.

    A map's keys are class codes and survive JSON as strings, and on a binary target the
    positive class is the higher code. ``'balanced'`` is the class frequency ratio, which
    the card knows; no weighting at all is 1.
    """
    current = params.get("class_weight")
    if isinstance(current, dict) and current:
        try:
            weights = {int(str(code).strip()): float(weight) for code, weight in current.items()}
        except (TypeError, ValueError):
            return 1.0
        return weights[max(weights)]
    if current == "balanced":
        return _frequency_ratio(state)
    return 1.0


def _first_rung(state: AutoMLState) -> float:
    """The positive-class weight to start from: the card's imbalance ratio, floored at one step.

    ``max``, not the ratio itself — a near-balanced card's ratio sits *below* one step, so using it
    directly would lower the weight this branch exists to raise. First rung only; the step and then
    the interpolation own the search after it. Rationale: ``docs/rationale.md``.
    """
    return max(WEIGHT_STEP, _frequency_ratio(state))


def _frequency_ratio(state: AutoMLState) -> float:
    """What ``'balanced'`` amounts to: majority frequency over minority frequency."""
    card = dict(state.get("dataset_card") or {})
    balance = card.get("class_balance")
    if isinstance(balance, (list, tuple)) and len(balance) == 2:
        major, minor = float(max(balance)), float(min(balance))
        if minor > 0:
            return round(major / minor, 4)
    ratio = card.get("imbalance_ratio")
    return float(ratio) if isinstance(ratio, (int, float)) and ratio > 0 else 1.0


def _family_plateaued(history: Sequence[Mapping[str, Any]], state: AutoMLState) -> bool:
    """Two or more prior attempts with the same model and no real gain."""
    model = str(state.get("model") or "")
    same_model = [item for item in history if str(item.get("model") or "") == model]
    return len(same_model) >= 2


def _direction_for(failure_type: str, task: str = TASK_CLASSIFICATION) -> str:
    if failure_type == "data_issue" and task == TASK_REGRESSION:
        # The classification wording below leads with 클래스 가중치, which the regression
        # executor has no equivalent of: prescribing it would put the Planner's next attempt
        # on a key that lands in ``dropped_hyperparams``. What is left that the executor
        # really does is imputation and a family less moved by a heavy tail.
        return (
            "데이터 문제를 먼저 처리한다: 결측치 대치 전략을 점검하고, 정답 열의 꼬리와 "
            "이상치에 덜 흔들리는 트리 계열로 옮긴다. 회귀에는 클래스 가중치에 해당하는 레버가 없다."
        )
    return {
        # batch_size/precision만 줄여서는 학습 자체가 가벼워지지 않는다 — 실행기에서 그 둘은
        # 메모리 추정치에만 반영된다. 실제로 footprint를 줄이는 것은 앞의 세 가지다.
        # 슬래시로 두 항목을 묶지 않습니다 — ``redact_paths``가 "A/B"를 경로로 읽어
        # "A<path>"로 바꿔 버려서, 프롬프트에 실제로 실리는 문장이 망가집니다.
        "oom": "메모리 사용량을 줄인다: 더 작은 모델, 반복 수와 추정기 수 축소, 학습 데이터 서브샘플링. "
        "batch_size 축소와 precision=fp16은 메모리 추정치를 낮춰 가드를 통과시키는 용도.",
        "too_slow": "시간 예산 안에 들어오도록 반복 수와 데이터 규모를 줄이고 더 빠른 계열로 교체한다.",
        "underfitting": "모델 용량과 학습량을 늘린다: 반복 수 증가, 트리 깊이·리프 수 확대, 정규화 완화.",
        "overfitting": "정규화를 강화하고 용량을 줄인다: l2 증가, 깊이 축소, learning_rate 하향.",
        "hyperparam": "계열은 유지하고 learning_rate와 깊이 조합을 다르게 탐색한다.",
        "wrong_model_family": "모델 계열 자체를 바꾼다.",
        "data_issue": "데이터 문제를 먼저 처리한다: 클래스 가중치 조정, 결측치와 이상치 처리.",
        "unknown": "원인이 불명확하므로 로그를 남기면서 가장 단순하고 안전한 구성으로 되돌린다.",
    }.get(failure_type, "다음 시도에서 구성을 변경한다.")


def _changes_for(
    failure_type: str, state: AutoMLState, task: str = TASK_CLASSIFICATION
) -> dict[str, Any]:
    """The concrete knobs the Planner should turn next."""
    hyperparams = dict(state.get("hyperparams") or {})
    if failure_type == "oom":
        return {
            "model": "smaller",
            "batch_size": 16,
            "precision": "fp16",
            "train_subsample": 0.5,
            "n_estimators": max(20, int(hyperparams.get("n_estimators", 200)) // 4),
        }
    if failure_type == "too_slow":
        return {"model": "smaller", "max_iter": 60, "train_subsample": 0.4}
    if failure_type == "underfitting":
        return {
            "max_iter": min(1200, max(200, int(hyperparams.get("max_iter", 150)) * 3)),
            "max_leaf_nodes": 63,
            "learning_rate": 0.08,
        }
    if failure_type == "overfitting":
        return {"l2_regularization": 1.0, "max_depth": 4, "learning_rate": 0.05}
    if failure_type == "hyperparam":
        # Walk a small grid so a repeated `hyperparam` verdict never proposes the
        # same numbers twice — otherwise the loop stalls on identical retries.
        step = len(state.get("history") or []) % 3
        return {
            "learning_rate": [0.05, 0.02, 0.12][step],
            "max_depth": [8, 5, 12][step],
            "max_iter": [400, 800, 250][step],
        }
    if failure_type == "wrong_model_family":
        return {"model_family": "different"}
    if failure_type == "data_issue":
        # ``class_weight`` is the whole prescription on a classification target and does not
        # exist on a continuous one, so the regression side names the family instead. Not a
        # hyperparameter, deliberately: ``fallback_plan`` reads ``model_family`` out of the
        # changes rather than merging it, so nothing here reaches the executor as a key it
        # would have to drop.
        return {"model_family": "different"} if task == TASK_REGRESSION else {"class_weight": "balanced"}
    return {"model": "safe_default"}
