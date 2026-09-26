"""Result Critic: diagnoses each finished attempt as a structured verdict, not free text.

Roles:

* Critic node — get a verdict, add the attempt to history.
* Verdict checking — force the LLM answer into the schema.
* Rule-based diagnosis — verdict from numbers, for dry-run and fallback.
* Prompt sections — the ``resolution`` and ``ledger`` text for the LLM.
* Operating point — spot threshold problems, not ranking problems.
* Class weight search — pick the next positive-class weight.
* Fallback prescriptions — fixed direction and changes per failure type.
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
    as_iteration,
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
    state_int,
)
from .model_selection import WEIGHT_RANGE, _history_digest, task_of_state

# JSON schema the LLM verdict must follow.
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

# Goal drop that is real, not seed noise.
GOAL_METRIC_DROP = 0.05
# roc_auc slip that still counts as held.
RANKING_TOLERANCE = 0.005

# Overfit gap: absolute, or share of train
OVERFIT_GAP = 0.15
OVERFIT_GAP_RATIO = 0.25

# Train miss that blames capacity; same two forms.
UNDERFIT_MARGIN = 0.05
UNDERFIT_MARGIN_RATIO = 0.05

# Goal metric fell, roc_auc held: threshold moved
OPERATING_POINT_DIRECTION = (
    "랭킹 품질(roc_auc)은 유지됐으므로 용량이 아니라 운영점 문제다: 불균형 레버를 되살린다 — "
    'class_weight=\'balanced\' 또는 클래스별 가중치 맵({"0": 1, "1": 10}), xgboost면 '
    "scale_pos_weight. 용량을 더 키우는 것은 이 격차를 되돌리지 못한다."
)

# Not ``ranking.SYMMETRIC_METRICS``: only an approximation here
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})

# Smaller recall/specificity gap is not skew
OPERATING_POINT_SKEW = 0.08

# Below this headroom the skew branch stays off
CUT_HEADROOM_FLOOR = 0.005

# Best cut still misses: ranking is the limit
RANKING_LIMIT_DIRECTION = (
    "운영점은 이미 최적이므로(balanced_accuracy_cut_headroom) 남은 격차는 컷이 아니라 랭킹에 있다: 이 랭킹의 "
    "어떤 임계값도 목표에 닿지 않으므로 모델 family를 바꾸거나 특성을 늘린다. 가중치나 "
    "임계값을 더 만지는 것은 이 격차를 줄이지 못한다."
)

# Weight step factor; a guess, not an estimate.
WEIGHT_STEP = 1.5

# Same range the sanitiser lets through.
MIN_WEIGHT, MAX_WEIGHT = WEIGHT_RANGE

UNWEIGHTED = 1.0

ERROR_TYPE_MAP: dict[str, str] = {
    "oom": "oom",
    "too_slow": "too_slow",
    "data_issue": "data_issue",
    "config_error": "data_issue",
    "unsupported_model": "wrong_model_family",
    "crash": "unknown",
    "no_result": "unknown",
    "exception": "unknown",
    # Disk problem, not data; keep it out of data_issue.
    "write_failed": "unknown",
}


# --- Role: critic node ------------------------------------------------------------


def critic(state: AutoMLState, *, config: RunConfig) -> dict:
    """Diagnose the last attempt (LLM, else :func:`heuristic_verdict`) and add it to ``history``.

    Returns ``{"critic": verdict, "history": [attempt]}``.
    """
    task = task_of_state(state)
    variables = {
        # The attempt may have passed (``--search-past-goal``).
        "frame": describe_verdict_frame(state, config),
        "goal": state.get("goal") or {},
        # LLM obeys a sentence better than a flag.
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "plan": state.get("plan") or {},
        "model": state.get("model") or "",
        "hyperparams": state.get("hyperparams") or {},
        "result": state.get("result") or {},
        "history": _history_digest(state),
        "best": state.get("best") or "(no successful attempt yet)",
        # Arithmetic done in code, not by the model
        "ledger": _ledger(state, config),
        "failure_types": ", ".join(FAILURE_TYPES),
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # Which score changes the rows can show
        "resolution": _resolution(state, config),
        "executor_capabilities": describe_capabilities(task),
    }

    iteration = state_int(state, "iteration") or None
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
        # Prescribes something the executor cannot do
        print(
            f"  [critic] 진단이 실행기에 없는 기능을 처방한 것으로 보입니다 — "
            f"{explain_claims(list(verdict['unsupported_claims']))}"
        )
    return {"critic": verdict, "history": [build_attempt(state, verdict)]}


def cleared_the_bar(state: AutoMLState, config: RunConfig) -> bool:
    """Tell whether the judged attempt already met the goal (only under ``--search-past-goal``).

    Uses the router's own ``goal_met`` check, so both always agree.
    """
    if not config.search_past_goal:
        return False
    return goal_met(dict(state.get("result") or {}), dict(state.get("goal") or {}))


def describe_verdict_frame(state: AutoMLState, config: RunConfig) -> str:
    """Write the prompt's opening: "explain the miss", or "bar cleared, what next".

    Must never claim a miss on a pass
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


# --- Role: verdict checking -------------------------------------------------------


def validate_verdict(verdict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fit the LLM answer to the verdict schema; unknown ``failure_type`` becomes ``unknown``.

    Returns ``None`` if the answer is not a dict.
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
        # Flag before ``direction`` reaches the next plan.
        "unsupported_claims": unsupported_claims(direction, evidence),
        "source": "llm",
    }


# --- Role: rule-based diagnosis ---------------------------------------------------


def heuristic_verdict(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Diagnose the last attempt from its numbers with fixed rules (dry-run and fallback)."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    threshold = goal_threshold(goal, config.fallback_threshold)
    result = dict(state.get("result") or {})
    metrics = dict(result.get("metrics") or {})
    history = list(state.get("history") or [])

    task = task_of_state(state)

    # 1. A training error names the failure.
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

    # 2. Otherwise read the train and validation numbers.
    measured = as_number(metric_value(result, metric))
    if measured is None:
        # No score must not read as perfect.
        measured = threshold if direction_of(metric) == MINIMIZE else 0.0
    score = float(measured)
    train_score = as_number(metrics.get(f"train_{metric}"))
    gap = as_number(metrics.get("train_val_gap"))
    if gap is None and train_score is not None:
        # Executor's form: positive means validation is worse.
        gap = score - train_score if direction_of(metric) == MINIMIZE else train_score - score

    # Overfit and operating point branches ignore this on purpose.
    cleared = cleared_the_bar(state, config)

    direction_override: str | None = None
    changes_override: dict[str, Any] | None = None
    if gap is not None and _overfits(metric, gap, train_score):
        failure_type = "overfitting"
        evidence = (
            f"train_{metric}={train_score}, {metric}={score:.4f}, "
            f"train_val_gap={gap:.4f} — 검증이 학습보다 그만큼 나쁨."
        )
    elif (collapse := _operating_point_collapse(metric, score, metrics, history)) is not None:
        # Before underfitting, once mistaken for it.
        failure_type = "data_issue"
        evidence = collapse
        direction_override = OPERATING_POINT_DIRECTION
    elif (skew := _operating_point_skew(metric, metrics, state)) is not None:
        # Before underfitting: skew also lowers the train score.
        failure_type = "hyperparam"
        evidence, direction_override, changes_override = skew
    elif (
        not cleared
        and train_score is not None
        and _underfits(metric, train_score, threshold)
    ):
        # Not on a pass: capacity is clearly enough.
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
        # After plateau: cross-attempt evidence is stronger.
        failure_type = "wrong_model_family"
        evidence = limited
        direction_override = RANKING_LIMIT_DIRECTION
    elif cleared:
        # Own text: "missed the goal" would be false.
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
    """_overfits | Rule-based diagnosis: tell if the train/val gap is wide enough to call overfitting."""
    # Unbounded metrics use a share of train
    found = spec(metric)
    if found is None or found.bounded:
        return gap > OVERFIT_GAP
    scale = as_number(train_score)
    if scale is None or scale <= 0.0:
        return False
    return gap > scale * OVERFIT_GAP_RATIO


def _underfits(metric: str, train_score: float, threshold: float) -> bool:
    """_underfits | Rule-based diagnosis: tell if the train score misses the goal enough to blame capacity."""
    # Error metrics miss upward; margin is a share.
    if direction_of(metric) == MINIMIZE:
        return train_score > threshold * (1.0 + UNDERFIT_MARGIN_RATIO)
    return train_score < threshold - UNDERFIT_MARGIN


# --- Role: prompt sections --------------------------------------------------------


def _resolution(state: AutoMLState, config: RunConfig) -> str:
    """_resolution | Prompt sections: say which compared scores fall inside this attempt's interval."""
    # Attempts with no score are skipped, not zero.
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    others: dict[str, Any] = {}
    threshold = as_number(goal.get("threshold"))
    if threshold is not None:
        others["목표"] = threshold
    for attempt in state.get("history") or []:
        value = metric_value(dict(attempt.get("result") or {}), metric)
        if value is not None:
            others[f"iteration {attempt.get('iteration')}"] = value
    return resolution_note(dict(state.get("result") or {}).get("metrics"), metric, others=others)


def _ledger(state: AutoMLState, config: RunConfig) -> str:
    """_ledger | Prompt sections: show how much each past prescription actually gained."""
    # For steering only; never picks the winner
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = direction_of(metric)
    attempts: list[Mapping[str, Any]] = [
        *(state.get("history") or []),
        # The judged attempt is not in history yet.
        {
            "iteration": state.get("iteration"),
            "model": state.get("model"),
            # Applied set, like the history rows.
            "hyperparams": effective_hyperparams(state),
            "result": state.get("result") or {},
            "critic": None,
        },
    ]

    lines: list[str] = []
    # Errors count too: an oom still used an iteration.
    families: dict[str, list[float | None]] = {}
    # failure_type -> did it ever set a new best.
    paid: dict[str, bool] = {}
    best: float | None = None
    # Paired results must match this same iteration.
    best_iteration: int | None = None
    # Last built pipeline; failed attempts have none, skip them.
    previous_pipeline: dict[str, Any] = {}
    previous_built: Mapping[str, Any] | None = None
    # Rows where family and pipeline moved together.
    two_levers: list[str] = []
    # Only pipeline-alone rows are evidence about preprocessing.
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
            # Planner may override the prescribed model
            asked = dict(prior.get("concrete_changes") or {}).get("model")
            if isinstance(asked, str) and asked and asked != family:
                parts.append(f"— 다만 처방은 {asked}였고 계획이 {family}로 바꿨다")
            # Same check for prescribed preprocessing.
            dropped = _dropped_preprocessing(prior.get("concrete_changes"), pipeline)
            if dropped:
                parts.append(f"— 다만 처방의 전처리가 이 시도에 없다: {dropped}")
            gained = score is not None and is_better(score, best, direction)
            paid[prescription] = paid.get(prescription, False) or gained
        if is_better(score, best, direction):
            best, best_iteration = score, as_iteration(attempt.get("iteration"))
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
        # A warning, not a ban
        lines.append(
            "  한 행에 레버가 둘인 전이: "
            + ", ".join(two_levers)
            + " — 계열과 전처리가 같은 전이에서 움직였으므로 그 행의 Δ는 어느 한쪽의 공로로 "
            "읽을 수 없습니다. 둘을 함께 움직이는 것이 맞을 때도 있으니(logreg는 대치를 "
            "강제합니다) 금지가 아니라 표시이고, 필요한 것은 그 뺄셈을 원인으로 읽지 않는 "
            "것입니다."
        )
    # Silent if the pipeline never moved
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


def _paired_note(attempt: Mapping[str, Any], baseline_iteration: int | None) -> str:
    """_paired_note | Prompt sections: render this row's paired test result, or why there is none."""
    # Wrong baseline gets a visible mismatch note.
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
    """_pipeline_of | Prompt sections: the pipeline the executor actually built, or ``{}``."""
    applied = dict(attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


# Read both flat and under ``preprocessing``.
_PRESCRIBABLE_PREPROCESSING = ("impute", "scale", "missing_indicator", "missing_count")


def _prescribed_preprocessing(changes: Any) -> dict[str, Any]:
    """_prescribed_preprocessing | Prompt sections: preprocessing keys asked for in ``concrete_changes``."""
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
    """_setting_matches | Prompt sections: compare two settings by spelling (``True`` matches ``true``)."""
    if isinstance(asked, bool) or isinstance(got, bool):
        return bool(asked) is bool(got)
    return str(asked).strip().lower() == str(got).strip().lower()


def _dropped_preprocessing(changes: Any, pipeline: Mapping[str, Any]) -> str:
    """_dropped_preprocessing | Prompt sections: prescribed preprocessing missing from the built pipeline."""
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
    """_other_levers_held | Prompt sections: tell if two attempts have the same hyperparameters."""
    # Missing key means "cannot confirm"
    if "hyperparams" not in previous or "hyperparams" not in current:
        return False
    return drop_pinned_seed(previous.get("hyperparams"), seed) == drop_pinned_seed(
        current.get("hyperparams"), seed
    )


def _pipeline_change(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """_pipeline_change | Prompt sections: list changed pipeline keys as ``impute: none → median``."""
    moved = sorted(key for key in {*previous, *current} if previous.get(key) != current.get(key))
    return ", ".join(
        f"{key}: {_setting(previous.get(key))} → {_setting(current.get(key))}" for key in moved
    )


def _setting(value: Any) -> str:
    """_setting | Prompt sections: show a preprocessing value in JSON style, or ``(없음)``."""
    if value is None:
        return "(없음)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cut_lever_note(
    state: AutoMLState, goal: Mapping[str, Any], metric: str, config: RunConfig
) -> str | None:
    """_cut_lever_note | Prompt sections: threshold headroom as a share of the distance to the goal."""
    # Other metrics would mix units
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
    # Full cover: say so, no negative remainder.
    if share >= 1:
        return head + "이므로, 이 격차는 랭킹이 아니라 운영점에 있습니다."
    return head + f"이고, 나머지 {1 - share:.0%}는 랭킹에 있습니다."


def _ranking_ceiling_note(
    attempts: Sequence[Mapping[str, Any]],
    goal: Mapping[str, Any],
    metric: str,
    config: RunConfig,
) -> str | None:
    """_ranking_ceiling_note | Prompt sections: distance to the goal vs the spread across families."""
    # Spread is not a paired comparison
    if metric not in SYMMETRIC_METRICS:
        return None
    # Best per family, so spread is between families.
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
        # Whole number: the spread supports one digit.
        line += f" — 부족분이 그 폭의 {round(shortfall / span)}배입니다"
    line += (
        f". 폭은 계열 {len(ceilings)}개의 최댓값과 최솟값의 차이이지 짝지은 비교가 아니므로, "
        "계열 사이에 차이가 있다는 근거로 쓰지 마십시오."
    )
    # Same fact in KS units: ceiling is (1 + KS) / 2.
    line += (
        f" 같은 말을 KS로 하면 바의 요구치가 {2 * threshold - 1:.4f}이고 "
        f"지금 최고는 {2 * high - 1:.4f}입니다."
    )
    return line


# --- Role: operating point --------------------------------------------------------


def _operating_point_collapse(
    metric: str,
    score: float,
    metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> str | None:
    """_operating_point_collapse | Operating point: evidence the goal metric fell while roc_auc held."""
    # Threshold metrics only; roc_auc is the ranking witness.
    spec = METRICS.get(metric)
    ranking = as_number(metrics.get("roc_auc"))
    if spec is None or spec.needs_proba or ranking is None:
        return None
    for attempt in history:
        prior = dict(attempt.get("result") or {})
        was = metric_value(prior, metric)
        ranking_was = metric_value(prior, "roc_auc")
        if was is None or ranking_was is None:
            continue
        if was - score > GOAL_METRIC_DROP and ranking >= ranking_was - RANKING_TOLERANCE:
            return (
                f"{metric}={score:.4f}로 iteration {attempt.get('iteration')}의 {was:.4f}보다 "
                f"{was - score:.4f} 낮은데 roc_auc는 {ranking:.4f} vs {ranking_was:.4f}로 "
                f"유지됨 — 순위는 그대로이고 판정 규칙만 움직였다."
            )
    return None


def _operating_point_skew(
    metric: str, metrics: Mapping[str, Any], state: AutoMLState
) -> tuple[str, str, dict[str, Any]] | None:
    """_operating_point_skew | Operating point: a weight change when recall and specificity are far apart."""
    # ``specificity`` means binary, so codes 0 and 1 are safe.
    if metric not in SYMMETRIC_METRICS:
        return None
    # Measured headroom wins; if missing, keep old behavior.
    headroom = as_number(metrics.get("balanced_accuracy_cut_headroom"))
    if headroom is not None and headroom < CUT_HEADROOM_FLOOR:
        return None
    skew = _skew(metrics)
    if skew is None or abs(skew) < OPERATING_POINT_SKEW:
        return None
    recall, specificity = float(metrics["recall"]), float(metrics["specificity"])

    weight = _positive_weight(state)
    # One step; first from card; interpolate once bracketed.
    from_nothing = skew < 0 and weight == UNWEIGHTED
    proposed = _clamp_weight(
        _first_rung(state)
        if from_nothing
        else weight * (WEIGHT_STEP if skew < 0 else 1 / WEIGHT_STEP)
    )
    how = ""
    if from_nothing and proposed != _clamp_weight(WEIGHT_STEP):
        # Name the source so it reads as reasoned.
        how = (
            f"가중치가 없던 시도이므로 한 스텝을 밟는 대신 카드가 적은 클래스 불균형 비율 "
            f"{_frequency_ratio(state):g}에서 시작한다. "
        )
    bracket = _interpolated_weight([*_weight_history(state), (weight, skew)])
    # Skip if it rounds to the current weight.
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
        # No slash: ``redact_paths`` would mask it.
        "train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다."
    )
    return evidence, direction, {"class_weight": {"0": 1, "1": proposed}}


def _ranking_limited(
    metric: str, metrics: Mapping[str, Any], threshold: float
) -> str | None:
    """_ranking_limited | Operating point: evidence the ranking, not the threshold, causes the miss."""
    # Needs tiny headroom and best cut below goal
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
    """_skew | Operating point: ``recall - specificity``; negative means raise the positive weight."""
    recall = as_number(metrics.get("recall"))
    specificity = as_number(metrics.get("specificity"))
    if recall is None or specificity is None:
        return None
    return recall - specificity


# --- Role: class weight search ----------------------------------------------------


def _weight_history(state: AutoMLState) -> list[tuple[float, float]]:
    """_weight_history | Class weight search: ``(weight, skew)`` for earlier attempts of this model."""
    # Same model only
    model = str(state.get("model") or "")
    points: list[tuple[float, float]] = []
    for attempt in state.get("history") or []:
        if str(attempt.get("model") or "") != model:
            continue
        skew = _skew(dict((attempt.get("result") or {}).get("metrics") or {}))
        if skew is None:
            continue
        # History holds the applied set that made these numbers.
        points.append((_weight_value(dict(attempt.get("hyperparams") or {}), state), skew))
    return points


def _interpolated_weight(
    points: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
    """_interpolated_weight | Class weight search: the weight where skew crosses 0, or ``None``."""
    # None if no bracket or wrong order
    below = max((point for point in points if point[1] < 0), key=lambda p: p[1], default=None)
    above = min((point for point in points if point[1] > 0), key=lambda p: p[1], default=None)
    if below is None or above is None or below[0] >= above[0]:
        return None
    span = above[1] - below[1]
    crossing = below[0] + (above[0] - below[0]) * (-below[1]) / span
    return crossing, below, above


def _clamp_weight(value: float) -> float:
    """_clamp_weight | Class weight search: keep a weight inside the allowed range, rounded to 3 places."""
    return round(min(max(value, MIN_WEIGHT), MAX_WEIGHT), 3)


def _positive_weight(state: AutoMLState) -> float:
    """_positive_weight | Class weight search: the positive weight the last attempt actually used."""
    applied = dict((state.get("result") or {}).get("applied_hyperparams") or {})
    return _weight_value(applied or dict(state.get("hyperparams") or {}), state)


def _weight_value(params: Mapping[str, Any], state: AutoMLState) -> float:
    """_weight_value | Class weight search: read the positive-class weight from one hyperparameter set."""
    current = params.get("class_weight")
    if isinstance(current, dict) and current:
        # Positive class is the larger class code.
        try:
            weights = {int(str(code).strip()): float(weight) for code, weight in current.items()}
        except (TypeError, ValueError):
            return 1.0
        return weights[max(weights)]
    if current == "balanced":
        return _frequency_ratio(state)
    return 1.0


def _first_rung(state: AutoMLState) -> float:
    """_first_rung | Class weight search: first weight: the card's imbalance ratio, at least one step."""
    # max() so a balanced card never lowers it
    return max(WEIGHT_STEP, _frequency_ratio(state))


def _frequency_ratio(state: AutoMLState) -> float:
    """_frequency_ratio | Class weight search: majority over minority frequency (``'balanced'``)."""
    card = dict(state.get("dataset_card") or {})
    balance = card.get("class_balance")
    if isinstance(balance, (list, tuple)) and len(balance) == 2:
        major, minor = float(max(balance)), float(min(balance))
        if minor > 0:
            return round(major / minor, 4)
    ratio = as_number(card.get("imbalance_ratio"))
    return ratio if ratio is not None and ratio > 0 else 1.0


# --- Role: fallback prescriptions -------------------------------------------------


def _family_plateaued(history: Sequence[Mapping[str, Any]], state: AutoMLState) -> bool:
    """_family_plateaued | Rule-based diagnosis: tell if this model was already tried twice or more."""
    model = str(state.get("model") or "")
    same_model = [item for item in history if str(item.get("model") or "") == model]
    return len(same_model) >= 2


def _direction_for(failure_type: str, task: str = TASK_CLASSIFICATION) -> str:
    """_direction_for | Fallback prescriptions: the fixed direction text for a failure type."""
    if failure_type == "data_issue" and task == TASK_REGRESSION:
        # Regression has no class weights to suggest.
        return (
            "데이터 문제를 먼저 처리한다: 결측치 대치 전략을 점검하고, 정답 열의 꼬리와 "
            "이상치에 덜 흔들리는 트리 계열로 옮긴다. 회귀에는 클래스 가중치에 해당하는 레버가 없다."
        )
    return {
        # No slashes: ``redact_paths`` would mask them.
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
    """_changes_for | Fallback prescriptions: settings the Planner should change next for a failure type."""
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
        # Cycle a grid so retries are not identical.
        step = len(state.get("history") or []) % 3
        return {
            "learning_rate": [0.05, 0.02, 0.12][step],
            "max_depth": [8, 5, 12][step],
            "max_iter": [400, 800, 250][step],
        }
    if failure_type == "wrong_model_family":
        return {"model_family": "different"}
    if failure_type == "data_issue":
        # Regression has no ``class_weight``; change family instead.
        return {"model_family": "different"} if task == TASK_REGRESSION else {"class_weight": "balanced"}
    return {"model": "safe_default"}
