"""Result Critic: 실패를 자유 서술이 아니라 구조화된 판정으로 진단한다.

스키마는 API 계층에서 강제된다. 정정 재시도 한 번 뒤에도 파싱에 실패하거나 API에 닿지 못하면
반복을 버리는 대신 결정적인 규칙 기반 진단으로 폴백한다 — 루프의 추론 흔적은 끊기면 안 된다.

끝난 시도를 판정과 함께 ``history``에 붙이는 것도 이 노드다. 그 append가 왜 여기 있는지는
``state.build_attempt``에 있다.
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

# 목표 지표가 이만큼 떨어지면 시드 잡음이 아니라 실제 퇴행이다. 랭킹은 tolerance만큼
# 미끄러져도 유지된 것으로 센다.
GOAL_METRIC_DROP = 0.05
RANKING_TOLERANCE = 0.005

# 과적합 격차의 두 형태 — 유계 지표는 절대값, 타깃 자기 단위의 지표는 학습 점수에 대한 비율.
# 후자에서 절대 격차는 임의의 달러 수나 날짜 수다.
OVERFIT_GAP = 0.15
OVERFIT_GAP_RATIO = 0.25

# *학습* 점수가 바에서 이만큼 멀어야 진단이 잡음이 아니라 용량이 된다. 위 짝과 같은 이유로
# 갈라져 있다.
UNDERFIT_MARGIN = 0.05
UNDERFIT_MARGIN_RATIO = 0.05

# 정해진 방향 문구들로는 표현할 수 없던 진단: `balanced_accuracy`는 떨어졌는데 `roc_auc`는
# *유지되거나 올라간* 경우. 순위는 그대로이고 판정 규칙만 움직였으므로 용량은 임계값 의존
# 지표를 되돌리지 못한다.
OPERATING_POINT_DIRECTION = (
    "랭킹 품질(roc_auc)은 유지됐으므로 용량이 아니라 운영점 문제다: 불균형 레버를 되살린다 — "
    'class_weight=\'balanced\' 또는 클래스별 가중치 맵({"0": 1, "1": 10}), xgboost면 '
    "scale_pos_weight. 용량을 더 키우는 것은 이 격차를 되돌리지 못한다."
)

# 최적점이 recall과 specificity가 만나는 자리에 있는 지표들 — 그래서 "격차를 좁혀라"가 맞는 조언.
# **``ranking.SYMMETRIC_METRICS``에서 가져오지 않은 것은 일부러다**: 이름은 같지만 그쪽은 항등식,
# 여기는 그 항등식에 대한 근사다(:data:`CUT_HEADROOM_FLOOR`).
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})

# 이 아래면 두 쪽이 충분히 가까워서 남은 미달은 운영점의 문제가 아니다. 분기가 발동해야 하는
# 실행들의 skew *위*에 둔다 — 그 범위 안에 있으면 그것들을 가르지 못한다.
OPERATING_POINT_SKEW = 0.08

# **``balanced_accuracy_cut_headroom``이 skew 분기를 이긴다** — 그 분기가 논거로 삼는 전제의
# 정확한 형태이고, 둘은 비대칭 ROC 곡선에서 갈린다. 이 바닥 아래에서는 근사가 이동을 처방하지
# 못한다.
CUT_HEADROOM_FLOOR = 0.005

# 컷이 이미 최적인데 최고 컷으로도 바에 못 닿을 때 남는 말. 추측이 아니다 — 임계값 아래의
# ``balanced_accuracy_at_best_cut``은 *이* 랭킹 위의 어떤 판정 규칙도 목표에 닿지 못한다는
# 증명이고, 루프 시작 전 ``goal.describe``가 베이스라인에 대해 하는 것과 같은 논증을 시도
# 하나에 적용한 것이다.
RANKING_LIMIT_DIRECTION = (
    "운영점은 이미 최적이므로(balanced_accuracy_cut_headroom) 남은 격차는 컷이 아니라 랭킹에 있다: 이 랭킹의 "
    "어떤 임계값도 목표에 닿지 않으므로 모델 family를 바꾸거나 특성을 늘린다. 가중치나 "
    "임계값을 더 만지는 것은 이 격차를 줄이지 못한다."
)

# 추정이 아니라 탐색 스텝 — recall과 specificity의 격차에서 그것을 닫는 가중치로 가는 닫힌
# 형태는 없다. 관측 둘이 교차점을 감싸면 ``_interpolated_weight``가 대신하고, **두 번째 단부터**
# 적용된다: 첫 단은 카드에서 온다(``_first_rung``).
WEIGHT_STEP = 1.5

# sanitiser가 실제로 통과시키는 범위에 묶는다. ``WEIGHT_RANGE`` 밖의 가중치는 실행기에 닿기
# 전에 버려지므로, 그런 값을 처방하면 다음 반복을 적용되지 않는 맵에 쓴다.
MIN_WEIGHT, MAX_WEIGHT = WEIGHT_RANGE

# 가중치가 전혀 없는 상태. ``_weight_value``는 ``class_weight``가 없을 때와 두 코드에 같은 수를
# 얹은 맵에 대해 정확히 이 값을 돌려준다 — 추정기에게 둘은 같은 것이므로 둘 다 첫 단을 밟는다.
UNWEIGHTED = 1.0

# 학습 단계의 error type은 그대로 대응된다. 추론할 것이 없다.
ERROR_TYPE_MAP: dict[str, str] = {
    "oom": "oom",
    "too_slow": "too_slow",
    "data_issue": "data_issue",
    "config_error": "data_issue",
    "unsupported_model": "wrong_model_family",
    "crash": "unknown",
    "no_result": "unknown",
    "exception": "unknown",
    # 모델이 아니라 환경이 실패한 것 — 반복이 자기 config를 쓰지 못했다
    # (nodes/training.py::_unwritable). ``unknown`` 기본값에 맡기지 않고 적어 둔 이유는 나중에
    # 누가 ``config_error`` 옆에 두고 ``data_issue``로 분류하지 않게 하려는 것이다. 디스크가
    # 꽉 찬 것은 열 수정으로 닿는 문제가 아니다.
    "write_failed": "unknown",
}


def critic(state: AutoMLState, *, config: RunConfig) -> dict:
    """``{failure_type, evidence, direction, concrete_changes}``를 만들고 시도를 기록한다."""
    task = task_of_state(state)
    variables = {
        # 이 노드가 무엇을 묻고 있는지. 늘 "미달을 설명하라"는 아니다 —
        # ``--search-past-goal``에서는 시도가 바를 넘었을 수 있다. 템플릿에 적지 않고 렌더하는
        # 이유는 describe_verdict_frame에 있다.
        "frame": describe_verdict_frame(state, config),
        "goal": state.get("goal") or {},
        # 바를 산문으로. 위의 같은 dict이지만 "이 바는 베이스라인 랭킹의 상한보다 높다"는
        # 의무로 읽히고 ``"exceeds_ranking_ceiling": true``는 그렇지 않다 — 그것을 보인 실행은
        # nodes/planning.py에 있다.
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "plan": state.get("plan") or {},
        "model": state.get("model") or "",
        "hyperparams": state.get("hyperparams") or {},
        "result": state.get("result") or {},
        "history": _history_digest(state),
        "best": state.get("best") or "(no successful attempt yet)",
        # 각 판정과 그것이 낳은 시도를 잇는 것. ``history`` 위의 산수이고 지시 4로 모델에게
        # 시키는 것으로는 부족했으므로 여기서 계산한다.
        "ledger": _ledger(state, config),
        "failure_types": ", ".join(FAILURE_TYPES),
        # caveat이 점수가 그 자리에 있는 이유일 수 있고, 그것이 무효화하는 것에 의존하는
        # 처방을 배제한다 — automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # 설명하려는 차이 중 얼마를 행들이 실제로 세우는지. 지표 키 둘을 더 넘기지 않고
        # 문장으로 넘긴다 — 키는 이미 ``result``에 있는데, 띠가 있다고 말해 주는 것이 없어서
        # 실제 실행의 반복 넷이 띠 안의 움직임을 진단하는 데 쓰였다 —
        # automl_agent.scoring.intervals.
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
        # 실행기가 수행할 수 없는 처방으로 실제 실행의 iteration 2가 날아갔다 — Planner가
        # 임계값 sweep이 보상해 줄 것으로 보고 class_weight를 버렸는데, 그 sweep은 없다.
        print(
            f"  [critic] 진단이 실행기에 없는 기능을 처방한 것으로 보입니다 — "
            f"{explain_claims(list(verdict['unsupported_claims']))}"
        )
    return {"critic": verdict, "history": [build_attempt(state, verdict)]}


def cleared_the_bar(state: AutoMLState, config: RunConfig) -> bool:
    """판정 대상 시도가 이미 목표에 닿았거나 넘었는지.

    :attr:`automl_agent.config.RunConfig.search_past_goal`에서만 도달한다 — 그것이 없으면
    ``route``가 통과한 시도를 곧장 보고로 보내므로 이 노드는 그런 시도를 보지 못한다.

    라우터가 쓰는 것과 같은 ``goal_met``을 같은 ``result`` 채널에서 읽는다. 그래야 둘이 어느
    시도가 바의 어느 쪽에 있는지를 두고 어긋나지 못한다.
    """
    if not config.search_past_goal:
        return False
    return goal_met(dict(state.get("result") or {}), dict(state.get("goal") or {}))


def describe_verdict_frame(state: AutoMLState, config: RunConfig) -> str:
    """프롬프트의 첫 부분: Critic이 *이* 시도에 대해 무엇을 묻고 있는지.

    **첫 문장은 미달을 무조건 단정할 수 없다.** 뒤의 모든 것이 그 빛으로 읽힌다.
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
    """모델의 답을 스키마에 맞춘다. 쓸 수 없으면 ``None``.

    분류 체계 밖의 ``failure_type``은 발명된 범주로 Planner의 프롬프트를 오염시키는 대신
    ``unknown``으로 내려앉는다.
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
        # ``direction``은 다음 planning 프롬프트에 그대로 복사되므로, 수행 불가능한 처방은
        # 여기서 표시하지 않으면 그대로 번져 나간다.
        "unsupported_claims": unsupported_claims(direction, evidence),
        "source": "llm",
    }


def heuristic_verdict(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """수만 보고 내리는 결정적인 진단. --dry-run과 폴백에서 쓴다."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    threshold = goal_threshold(goal, config.fallback_threshold)
    result = dict(state.get("result") or {})
    metrics = dict(result.get("metrics") or {})
    history = list(state.get("history") or [])

    task = task_of_state(state)

    # 1. 명시적인 실행 오류는 이미 실패를 이름으로 대고 있다.
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

    # 2. 그 밖에는 train과 validation의 수를 읽는다.
    measured = as_number(metric_value(result, metric))
    if measured is None:
        # **점수가 없는 것은 만점이 아니다.** ``0.0``은 최대화 지표에서 안전한 읽기이지만
        # 오차 지표에서는 *최악*이다 — "오차가 전혀 없다"가 되어 시도를 과적합 분기로 보낸다.
        # 거기서는 바가 중립적인 대체값이다.
        measured = threshold if direction_of(metric) == MINIMIZE else 0.0
    score = float(measured)
    train_score = as_number(metrics.get(f"train_{metric}"))
    gap = as_number(metrics.get("train_val_gap"))
    if gap is None and train_score is not None:
        # 실행기가 정규화하는 방식 그대로 — validation이 얼마나 *더 나쁜지* — 로 맞춘다.
        # 그래야 지표가 어느 방향이든 부호가 과적합을 뜻한다.
        gap = score - train_score if direction_of(metric) == MINIMIZE else train_score - score

    # 이 시도가 바의 건너편에 있는지. ``--search-past-goal``에서만 가능하고, 아래 분기 몇 개가
    # 무엇을 주장할 수 있는지를 바꾼다. 과적합과 운영점 분기는 일부러 이것으로 막지 *않는다* —
    # 넓은 train과 validation의 격차는 통과한 시도에 대해서도 실재하는 발견이고, 단일 통과
    # 점수가 감추는 것이 바로 그것이다.
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
        # 과소적합보다 먼저 본다 — 이 모양이 과거에 그것으로 오인됐다. 과적합보다는
        # 나중인데, 그쪽 근거(train과 val의 격차)는 이것과 독립이다.
        failure_type = "data_issue"
        evidence = collapse
        direction_override = OPERATING_POINT_DIRECTION
    elif (skew := _operating_point_skew(metric, metrics, state)) is not None:
        # 이것도 과소적합보다 먼저이고, 이유가 더 날카롭다 — 기울어진 운영점은 *학습* 점수도
        # 함께 끌어내리므로, 고칠 것이 가중치 하나인데 "둘 다 낮고 서로 가깝다"가 용량 부족으로
        # 읽힌다.
        failure_type = "hyperparam"
        evidence, direction_override, changes_override = skew
    elif (
        not cleared
        and train_score is not None
        and _underfits(metric, train_score, threshold)
    ):
        # 미달에 맞춰 문구만 쓴 것이 아니라 미달로 막는다. 검증 점수가 바를 넘은 시도는 학습
        # 점수가 무엇이든 용량이 모자란 것이 아니고, 거기서 용량을 *더* 처방하는 것은 격차까지
        # 넓히는 유일한 방향이다.
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
        # 정체 검사보다 나중 — 그쪽 근거는 시도들에 걸쳐 있어서 더 강하다. skew 분기보다도
        # 나중인데, 둘이 함께 발동할 상황은 이쪽 게이트가 이미 그쪽을 침묵시킨 경우다.
        failure_type = "wrong_model_family"
        evidence = limited
        direction_override = RANKING_LIMIT_DIRECTION
    elif cleared:
        # --search-past-goal이 대부분의 경우 실제로 닿는 분기. 미달 문구에 수만 바꿔 넣는 것이
        # 아니라 자기 사건이어야 한다 — 바 위의 점수에 대한 "목표에 미달"은 거짓 문장이고, 이
        # 판정의 ``evidence``로 다음 planning 프롬프트에 그대로 실린다.
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
    """train과 validation의 격차가 과적합이라고 부를 만큼 넓은지.

    ``gap``은 "validation이 학습보다 얼마나 더 나쁜지"로 정규화되어 오므로 남은 것은 스케일뿐이다.
    유계 지표는 스케일을 자기가 갖고 있고, ``mae``와 ``rmse``는 학습 점수에 대한 비율로 판정하며,
    점수가 없거나 0이면 **단위를 추측하는 대신 판정을 포기한다**.
    """
    found = spec(metric)
    if found is None or found.bounded:
        return gap > OVERFIT_GAP
    scale = as_number(train_score)
    if scale is None or scale <= 0.0:
        return False
    return gap > scale * OVERFIT_GAP_RATIO


def _underfits(metric: str, train_score: float, threshold: float) -> bool:
    """학습 점수마저 바에서 충분히 멀어서 용량을 탓할 수 있는지.

    비교는 방향을, 여유는 스케일을 본다 — 오차 지표에서 "바에 못 미친다"는 바보다 *위*에 있는
    것이고, 여유는 이 모듈이 모르는 단위의 고정된 0.05가 아니라 바에 대한 비율이어야 한다.
    """
    if direction_of(metric) == MINIMIZE:
        return train_score > threshold * (1.0 + UNDERFIT_MARGIN_RATIO)
    return train_score < threshold - UNDERFIT_MARGIN


def _resolution(state: AutoMLState, config: RunConfig) -> str:
    """프롬프트의 ``resolution`` 절: 이 시도의 구간과 그것이 삼키는 것.

    비교 집합은 Critic이 추론에 쓰는 전부다 — 놓친 바, 그리고 같은 지표에서 이전 시도들이 받은
    점수. **점수가 없는 이전 시도는 0이 아니라 아무것도 기여하지 않는다** — 0은 모든 구간에
    걸리고 "크래시와 구분되지 않는다"로 읽힌다.
    """
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
    """``ledger`` 절: 지금까지의 각 처방이 실제로 얼마의 값을 했는지.

    한 행이 시도 하나가 아니라 *처방* 하나다 — iteration N의 판정이 iteration N+1을 낳았으므로
    마지막 행의 판정은 지금 쓰이고 있는 것이고 아직 점수가 없다. 판정 대상 시도는 ``history``에
    없으므로(``critic``이 붙인다) 여기서 더한다.

    ``paired_of``에는 이 ledger 자신의 현재 기준을 넘긴다. 맞는 것을 들고 있으리라고 믿지 않는다 —
    문장이 말하는 것과 다른 iteration에 대해 Δ가 계산된 행은 모든 수가 참인데 주장은 거짓인 줄이다.

    조종 전용. 여기 어느 행도 테스트 분할을 보지 않고, 어느 반복이 이기는지에 대한 게이트가
    되어서도 안 된다(:mod:`automl_agent.nodes.holdout`).
    """
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = direction_of(metric)
    attempts: list[Mapping[str, Any]] = [
        *(state.get("history") or []),
        {
            "iteration": state.get("iteration"),
            "model": state.get("model"),
            # history 행들이 지나온 것과 같은 함수를 통과시킨다. 그래야 "하이퍼파라미터가
            # 움직이지 않았다"가 같은 것끼리의 비교가 된다. 제안과 적용된 블록은 키가 버려졌을
            # 때 정확히 갈리므로, 행마다 다른 쪽을 읽으면 버려진 키가 움직인 레버로 보고된다.
            "hyperparams": effective_hyperparams(state),
            "result": state.get("result") or {},
            "critic": None,
        },
    ]

    lines: list[str] = []
    # 계열별 시도 수. 오류로 끝난 것도 센다. 어떤 계열 안의 `oom`은 그 계열이 쓴 반복이고,
    # `wrong_model_family`는 그것을 몇 번 썼는지에 대한 주장이기도 하다(``_family_plateaued``).
    families: dict[str, list[float | None]] = {}
    # ``failure_type`` -> 그것이 낳은 시도 중 최고를 갱신한 것이 있는지. 두 번 내려졌고 두 번
    # 다 아무것도 갚지 못한 판정은 멈춘 루프의 정확한 모양이다.
    paid: dict[str, bool] = {}
    best: float | None = None
    # 나중에 유도하지 않고 따라간다. 짝지은 블록의 Δ가 이 행의 뺄셈과 같은 비교에 대한 것이
    # 되려면 그 블록이 동의해야 하는 키다.
    best_iteration: int | None = None
    # 실제로 *세워진* 마지막 파이프라인. 오류로 끝난 시도들을 건너 이어진다 — 그런 시도에는
    # ``applied_preprocessing``이 없고, 없는 것을 변경으로 다루면 파이프라인 전체가 뜯겼다가
    # 다시 붙은 것으로 보고된다. ``previous_built``가 같이 가는 이유는 중간에 오류가 끼면
    # ``attempts[index - 1]``이 비교 대상 행이 아니기 때문이다.
    previous_pipeline: dict[str, Any] = {}
    previous_built: Mapping[str, Any] | None = None
    # 계열*과* 파이프라인을 같이 움직인 전이의 행들. 행마다 판단하지 않고 모으는 이유는
    # 그것들에 대해 할 말이 규칙 하나이고 그 사본 다섯 개가 아니기 때문이다(꼬리 줄과
    # :func:`_pipeline_change`).
    two_levers: list[str] = []
    # 같은 셈을 반대쪽에서 — 파이프라인이 움직인 전이가 몇 건인지, 그중 *다른 것은 아무것도*
    # 움직이지 않은 것이 어느 것인지. 전처리 레버에 대한 근거가 되는 것은 두 번째뿐이고, 단독
    # 전이는 실제로 드물다(대부분의 실행이 계열을 같이 움직인다).
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
            # Planner는 판정을 뒤집을 수 있고 실제로 뒤집었다 — 처방한 계열 교체가 다른
            # 계열로 돌아와서 전부를 이기는 일이 있다. 그 점수를 처방의 공로로 적는 것은 가장
            # 중요한 방향에서의 거짓이다. 진단이 맞았다는 증거로 읽히는 행이 바로 그 행이니까.
            asked = dict(prior.get("concrete_changes") or {}).get("model")
            if isinstance(asked, str) and asked and asked != family:
                parts.append(f"— 다만 처방은 {asked}였고 계획이 {family}로 바꿨다")
            # 계열 쪽 문구가 덮지 못하는 축에서의 같은 뒤집기 — 처방한 전처리 단계가 꺼진
            # 채로 돌아올 수 있고, 어디에서도 그것을 말해 주지 않는다. 다음 판정은 그 행을
            # 존재한 적 없는 열에 대한 근거로 읽는다.
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
        # **금지가 아니라 표시** — ``logreg``는 ``impute: none``을 받지 못하므로 그것을
        # 시도하는 것이 같은 전이에서 파이프라인 변경을 *요구*한다. 일어나서는 안 되는 것은 그
        # 행의 뺄셈이 한 레버의 공로로 읽히는 것뿐이다.
        lines.append(
            "  한 행에 레버가 둘인 전이: "
            + ", ".join(two_levers)
            + " — 계열과 전처리가 같은 전이에서 움직였으므로 그 행의 Δ는 어느 한쪽의 공로로 "
            "읽을 수 없습니다. 둘을 함께 움직이는 것이 맞을 때도 있으니(logreg는 대치를 "
            "강제합니다) 금지가 아니라 표시이고, 필요한 것은 그 뺄셈을 원인으로 읽지 않는 "
            "것입니다."
        )
    # **파이프라인이 한 번도 안 움직였으면 침묵한다.** "전처리를 아직 안 써 봤습니다"는
    # 참이면서 동시에 초대이고, 안 써 본 레버를 목록으로 보여 주는 프롬프트는 그것을 써 보게
    # 만든다. 줄을 쓸 값이 있는 것은 이미 일어난 전이뿐이다.
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
    """이 행의 짝지은 판정, 또는 그것이 없는 이유. 아무것도 없으면 ``""``.

    결과가 셋이고 각각을 침묵에 맡기지 않고 소리 내어 말한다. ledger 자신의 기준에 대해 측정된
    비교는 판정으로 렌더한다. *건너뛴* 비교는 이유를 렌더한다 — "비교하지 않았다"와 "비교했고
    아무것도 못 찾았다"는 다른 사실이다. 예외는 ``no_baseline``인데, 그 행은 이미 첫 측정으로
    읽히므로 반복하면 잡음이다. *다른* 기준에 대해 측정된 비교는 불일치로 렌더한다. 그것은 장부
    결함이고, 빈 문자열에 감춰진 결함은 아무도 못 찾는다.
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
    """실행기가 이 시도에 대해 실제로 세운 파이프라인, 없으면 ``{}``.

    계획의 ``preprocessing`` 블록이 아니라 ``applied_preprocessing``이다. 둘은 요청이 하향됐을 때
    정확히 갈리고, 보고할 값이 있는 것은 그 하향이다.
    """
    applied = dict(attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


# 판정이 처방할 수 있는 전처리 설정들. 판정이 쓰는 두 모양 — ``preprocessing`` 아래 중첩된
# 것과 하이퍼파라미터 옆에 평평하게 놓인 것 — 둘 다 실제 실행에서 나왔고, 한쪽만 읽으면 버려진
# 요청이 받아들여진 것으로 보고된다.
_PRESCRIBABLE_PREPROCESSING = ("impute", "scale", "missing_indicator", "missing_count")


def _prescribed_preprocessing(changes: Any) -> dict[str, Any]:
    """``concrete_changes`` dict가 요청하는 전처리 키들. 중첩이든 평평하든."""
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
    """``"none"``과 ``none``, ``True``와 ``true``를 맞춘다. 동일성이 아니라 표기 비교."""
    if isinstance(asked, bool) or isinstance(got, bool):
        return bool(asked) is bool(got)
    return str(asked).strip().lower() == str(got).strip().lower()


def _dropped_preprocessing(changes: Any, pipeline: Mapping[str, Any]) -> str:
    """판정이 처방했는데 그것이 낳은 시도가 들고 있지 않은 전처리.

    실제로 세워진 파이프라인에 대고 본다. 그래야 사라지는 두 경로 — Planner가 판정을 뒤집는 것,
    실행기가 계열이 받지 못하는 요청을 하향하는 것 — 를 다 덮는다. 둘 중 무엇이었는지는 여기서
    주장하지 않는다. 다음 판정에 필요한 것은 그 설정이, 처방의 결과로 읽으려는 시도 안에
    없었다는 사실이다. 추가된 적 없는 열에 대한 추론은 일어나지 않은 실행에 대한 추론이다. 같은
    뒤집기의 계열 쪽은 이 옆에서 함께 보고된다.
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
    """두 시도의 하이퍼파라미터가 같은지 — 즉 파이프라인만 움직였는지.

    계열은 호출자가 비교하고 이것이 나머지 반쪽인데, 형식이 아니다. 재튜닝*과* 파이프라인 변경이
    함께 일어난 전이는 계열 교체와 똑같이 한 축에 주인이 둘이다.

    **``hyperparams`` 키가 없으면 "유지됐다"가 아니라 "유지를 확인할 수 없다"로 읽는다.** 그래서
    거부되는 행은 기록이 말하지 않는 행뿐이다.
    """
    if "hyperparams" not in previous or "hyperparams" not in current:
        return False
    return drop_pinned_seed(previous.get("hyperparams"), seed) == drop_pinned_seed(
        current.get("hyperparams"), seed
    )


def _pipeline_change(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """적용값이 움직인 모든 키에 대해 ``impute: none → median``.

    계열*과* 파이프라인을 같이 움직인 행은 레버 둘을 썼고 ledger의 뺄셈은 둘을 가르지 못한다.
    여기서 변경을 이름으로 대는 것이 분해는 아니다 — 그 행이 깨끗한 레버 하나로 읽히는 것을
    막는다.
    """
    moved = sorted(key for key in {*previous, *current} if previous.get(key) != current.get(key))
    return ", ".join(
        f"{key}: {_setting(previous.get(key))} → {_setting(current.get(key))}" for key in moved
    )


def _setting(value: Any) -> str:
    """계획이 쓰는 대로의 전처리 값. JSON 표기이고, 없으면 ``(없음)``."""
    if value is None:
        return "(없음)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cut_lever_note(
    state: AutoMLState, goal: Mapping[str, Any], metric: str, config: RunConfig
) -> str | None:
    """``balanced_accuracy_cut_headroom``을 남은 거리 옆에 놓고, 그 거리에 대한 비율로 적는다.

    컷이 반복 하나를 쓸 값이 있는지를 정하는 것은 그 *비*이고, 나눗셈이므로 묻지 않고 계산한다.

    ``balanced_accuracy``에서만, :func:`_ranking_limited`와 skew 분기와 같은 집합에서 동작한다.
    ``balanced_accuracy_cut_headroom``은 목표 지표가 무엇이든 ``balanced_accuracy``의 측정값인데
    (``scripts/train.py``가 그 하나의 상한에 대고 계산한다) 아래의 ``shortfall``은 목표 지표 자기
    단위다 — 다른 목표에서는 이 비가 한 지표의 headroom을 다른 지표의 거리로 나누고 그 몫을
    비율이라고 보고한다. 이 가드는 이 수를 쓰는 다른 두 곳이 이미 갖고 있던 것이다.

    집합이 최대화 쪽이므로 이것이 대체하는 최소화 검사를 포함한다 — 최소화 지표에서는 미달이
    반대로 흐르고 ``balanced_accuracy_cut_headroom``은 아예 존재하지 않는다.
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
    # 남은 거리 전부를 덮는 headroom은 랭킹이 이미 충분하고 컷만 막고 있다는 뜻이고, 그것은
    # 반대 처방이다 — 음수 나머지로 보고되어서는 안 된다.
    if share >= 1:
        return head + "이므로, 이 격차는 랭킹이 아니라 운영점에 있습니다."
    return head + f"이고, 나머지 {1 - share:.0%}는 랭킹에 있습니다."


def _ranking_ceiling_note(
    attempts: Sequence[Mapping[str, Any]],
    goal: Mapping[str, Any],
    metric: str,
    config: RunConfig,
) -> str | None:
    """바가 얼마나 먼지를, 계열 교체가 실제로 움직인 폭의 단위로 적는다.

    **폭은 최댓값 N개의 범위이고 짝지은 비교가 아니며, 줄 자체가 그렇게 말한다.** 계열들이 똑같이
    순위를 매기는 귀무가설 아래에서도 최댓값 몇 개는 그만큼 저절로 퍼지므로, 폭은 계열 사이에
    차이가 있다는 근거가 되지 못한다.

    :data:`SYMMETRIC_METRICS`(상한 항등식이 거기에만 있다)로, 그리고 여전히 바에 못 미치는 상한으로
    제한한다 — 그 위는 :func:`_cut_lever_note`의 사건이다.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    # 계열당 최고 상한만. 세 번 시도한 계열도 값 하나를 낸다 — 폭은 계열 사이의 것이어야 하고,
    # 한 계열 안의 반복을 세면 하이퍼파라미터 잡음이 계열 레버를 대표하는 수를 넓힌다.
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
        # 배수에 소수점을 쓰지 않는다. 폭이 받쳐 주는 것은 유효숫자 한 자리이고, "5.0배"는
        # 측정된 비율로 읽힌다.
        line += f" — 부족분이 그 폭의 {round(shortfall / span)}배입니다"
    line += (
        f". 폭은 계열 {len(ceilings)}개의 최댓값과 최솟값의 차이이지 짝지은 비교가 아니므로, "
        "계열 사이에 차이가 있다는 근거로 쓰지 마십시오."
    )
    # KS는 상한의 아핀 함수이므로(``ranking.best_cut_ceiling``이 ``(1 + KS) / 2``) 이것은 같은
    # 말을 랭킹 품질 단위로 뒤집은 것이고 새 비교를 만들지 않는다.
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
    """판정 규칙만 움직였다는 근거, 없으면 ``None``.

    임계값 의존 지표로 제한한다 — ``roc_auc``와 ``pr_auc``는 랭킹 *자체*이므로 그중 하나가
    떨어지는 것은 이 발견의 반대다. 증인이 ``roc_auc``인 이유는, 실행기가 확률을 낼 수 있는 모든
    이진 시도에서 목표 지표와 나란히 그것을 내보내기 때문이다.
    """
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
    """두 쪽이 기울어 있을 때 ``(evidence, direction, concrete_changes)``.

    :data:`SYMMETRIC_METRICS`로 제한한다. ``specificity``는 이진 전용이므로 그것이 있다는 사실이,
    처방이 코드 0과 1을 이름으로 댈 수 있게 해 주는 이진 가드 역할까지 한다.
    """
    if metric not in SYMMETRIC_METRICS:
        return None
    # 측정값이 근사를 이긴다. 양방향으로 권고적이다 — 다중 분류 시도와, 이 수가 이 이름을 갖기
    # 전에 쓰인 모든 결과는 ``balanced_accuracy_cut_headroom``을 갖지 않는다. 그런 경우는 분기를
    # 잃는 대신 옛 동작을 유지한다.
    headroom = as_number(metrics.get("balanced_accuracy_cut_headroom"))
    if headroom is not None and headroom < CUT_HEADROOM_FLOOR:
        return None
    skew = _skew(metrics)
    if skew is None or abs(skew) < OPERATING_POINT_SKEW:
        return None
    recall, specificity = float(metrics["recall"]), float(metrics["specificity"])

    weight = _positive_weight(state)
    # 기본은 한 스텝. 구간이 잡히는 즉시 보간이 대신하고, 보간값이 지금 가중치 자리로
    # 반올림되면 건너뛴다 — 아무것도 안 하는 처방은 같은 점을 다시 재는 데 반복 하나를 쓴다.
    # 가중치가 없던 시도에서는 스텝 대신 카드가 단을 정한다(``_first_rung``).
    from_nothing = skew < 0 and weight == UNWEIGHTED
    proposed = _clamp_weight(
        _first_rung(state)
        if from_nothing
        else weight * (WEIGHT_STEP if skew < 0 else 1 / WEIGHT_STEP)
    )
    how = ""
    if from_nothing and proposed != _clamp_weight(WEIGHT_STEP):
        # 수가 어디서 왔는지. 다음 시도가 이 문장을 보고 계획되고, 출처 없는 "1 → 5.07"은
        # 추측으로 읽힌다. 카드가 고른 경우에만 적는다 — 균형에 가까운 카드에서는
        # ``_first_rung``이 스텝을 돌려주므로, 그때 이 말을 하면 프롬프트에 거짓 귀속이 실린다.
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
    """미달의 원인이 판정 규칙이 아니라 랭킹이라는 근거.

    **사실 둘이 함께 성립해야 하고, 그것이 이것을 휴리스틱이 아니라 증명으로 만든다** — 컷이 이미
    이 랭킹의 최고에서 :data:`CUT_HEADROOM_FLOOR` 안에 있고, *그리고* 그 최고가 여전히 바 아래다.
    어느 한쪽만으로는 운영점을 먼저 고칠 값이 남는다(그쪽은 skew 분기의 사건이다).
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
    """``recall - specificity``. 음수면 양성 클래스에 가중치가 더 필요하다는 뜻이다."""
    recall = as_number(metrics.get("recall"))
    specificity = as_number(metrics.get("specificity"))
    if recall is None or specificity is None:
        return None
    return recall - specificity


def _weight_history(state: AutoMLState) -> list[tuple[float, float]]:
    """*같은* 모델의 모든 이전 시도에 대한 ``(양성 가중치, skew)``.

    **같은 모델만.** 다른 계열의 점은 이 계열의 운영점이 어디 있는지에 대해 아무 말도 하지 않고,
    둘을 걸쳐 보간하면 교차점이 어느 모델도 가 본 적 없는 자리에 놓인다.
    """
    model = str(state.get("model") or "")
    points: list[tuple[float, float]] = []
    for attempt in state.get("history") or []:
        if str(attempt.get("model") or "") != model:
            continue
        skew = _skew(dict((attempt.get("result") or {}).get("metrics") or {}))
        if skew is None:
            continue
        # ``history``는 적용된 집합을 들고 있고(``state.effective_hyperparams``), 이 수들이
        # 나온 것이 그 집합이다.
        points.append((_weight_value(dict(attempt.get("hyperparams") or {}), state), skew))
    return points


def _interpolated_weight(
    points: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
    """관측된 가중치 둘이 0을 감쌀 때, skew가 0을 지나는 자리.

    skew는 양성 가중치와 함께 오르므로 부호 변화가 최적점을 감싸고, 가장 가까운 두 점을 지나는
    secant가 또 한 번의 눈먼 스텝보다 낫다.

    구간이 없으면 ``None``이고, **두 점이 단조성이 요구하는 순서로 놓여 있지 않을 때도** 그렇다 —
    시도 사이의 용량 변화가 순서를 뒤집을 수 있고, 뒤집힌 점들을 지나는 secant는 틀린 방향을
    가리킨다.
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
    """마지막 시도가 양성 클래스에 실제로 얹은 가중치.

    ``applied_hyperparams``를 먼저 본다. 실행기가 버린 제안은 이 수들을 만든 것이 아니다.
    """
    applied = dict((state.get("result") or {}).get("applied_hyperparams") or {})
    return _weight_value(applied or dict(state.get("hyperparams") or {}), state)


def _weight_value(params: Mapping[str, Any], state: AutoMLState) -> float:
    """하이퍼파라미터 집합 하나에서 양성 클래스의 가중치를 읽는다.

    맵의 키는 클래스 코드이고 JSON을 지나면 문자열이 되며, 이진 타깃에서 양성 클래스는 더 큰
    코드다. ``'balanced'``는 클래스 빈도비이고 그것은 카드가 안다. 가중치가 전혀 없으면 1이다.
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
    """양성 클래스 가중치의 출발점: 카드의 불균형비, 최소 한 스텝.

    비율 자체가 아니라 ``max``인 이유 — 균형에 가까운 카드의 비율은 한 스텝 *아래*에 있으므로
    그대로 쓰면 이 분기가 올리려고 있는 가중치를 낮춘다. 첫 단에만 쓴다. 그 뒤의 탐색은 스텝과
    보간이 맡는다.
    """
    return max(WEIGHT_STEP, _frequency_ratio(state))


def _frequency_ratio(state: AutoMLState) -> float:
    """``'balanced'``가 결국 무엇인지 — 다수 클래스 빈도를 소수 클래스 빈도로 나눈 값."""
    card = dict(state.get("dataset_card") or {})
    balance = card.get("class_balance")
    if isinstance(balance, (list, tuple)) and len(balance) == 2:
        major, minor = float(max(balance)), float(min(balance))
        if minor > 0:
            return round(major / minor, 4)
    ratio = as_number(card.get("imbalance_ratio"))
    return ratio if ratio is not None and ratio > 0 else 1.0


def _family_plateaued(history: Sequence[Mapping[str, Any]], state: AutoMLState) -> bool:
    """같은 모델로 이전에 두 번 이상 시도했고 실질적인 개선이 없는 상태."""
    model = str(state.get("model") or "")
    same_model = [item for item in history if str(item.get("model") or "") == model]
    return len(same_model) >= 2


def _direction_for(failure_type: str, task: str = TASK_CLASSIFICATION) -> str:
    if failure_type == "data_issue" and task == TASK_REGRESSION:
        # 아래 분류 문구는 클래스 가중치로 시작하는데, 회귀 실행기에는 그에 해당하는 것이
        # 없다 — 그것을 처방하면 Planner의 다음 시도가 ``dropped_hyperparams``로 떨어지는 키에
        # 얹힌다. 실행기가 실제로 하는 것 중 남는 것은 대치와, 두꺼운 꼬리에 덜 흔들리는 계열이다.
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
    """Planner가 다음에 돌려야 할 구체적인 손잡이들."""
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
        # 작은 grid를 걸어서 `hyperparam` 판정이 반복돼도 같은 수를 두 번 제안하지 않게
        # 한다. 그러지 않으면 루프가 똑같은 재시도에서 멈춘다.
        step = len(state.get("history") or []) % 3
        return {
            "learning_rate": [0.05, 0.02, 0.12][step],
            "max_depth": [8, 5, 12][step],
            "max_iter": [400, 800, 250][step],
        }
    if failure_type == "wrong_model_family":
        return {"model_family": "different"}
    if failure_type == "data_issue":
        # ``class_weight``는 분류 타깃에서 처방의 전부이고 연속 타깃에는 없으므로, 회귀 쪽은
        # 대신 계열을 이름으로 댄다. 하이퍼파라미터가 아닌 것은 일부러다 — ``fallback_plan``은
        # changes에서 ``model_family``를 병합하지 않고 꺼내 쓰므로, 여기서 실행기가 버려야 할
        # 키로 닿는 것이 없다.
        return {"model_family": "different"} if task == TASK_REGRESSION else {"class_weight": "balanced"}
    return {"model": "safe_default"}
