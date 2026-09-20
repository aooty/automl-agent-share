"""AutoML 그래프가 공유하는 state — 칠판.

모든 노드는 *부분* 갱신을 돌려주는 순수 함수 ``state -> dict``다. 누적되는 채널은
``history``뿐이다: ``operator.add`` reducer를 써서 각 시도가 앞의 것을 덮어쓰지 않고
뒤에 붙는다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .config import HOLDOUT_RESERVE_FRACTION, MIN_FIT_TIMEOUT_SEC
from .scoring.intervals import as_number

FAILURE_TYPES: tuple[str, ...] = (
    "underfitting",
    "overfitting",
    "data_issue",
    "hyperparam",
    "oom",
    "too_slow",
    "wrong_model_family",
    "unknown",
)


class Attempt(TypedDict):
    """루프를 한 바퀴 완주한 기록. ``history``에 붙는다."""

    iteration: int
    plan: dict
    model: str
    # 제안이 아니라 실행기가 실제로 적용한 값 (``effective_hyperparams`` 참고).
    hyperparams: dict
    result: dict  # {"f1": 0.72, "train_time_sec": 812} 또는 {"error": "oom"}
    critic: dict | None  # {"failure_type": ..., "direction": ...}
    # "llm" | "fallback" | "rules" — 이 ``(model, hyperparams)``를 누가 골랐는지. 계획은 자기
    # 것을 ``plan["source"]``에 담는다. 세 상태가 왜 둘이 아닌지는 ``nodes/planning.py``.
    selection_source: str


class AutoMLState(TypedDict):
    """그래프의 모든 노드가 공유하는 칠판.

    데이터에 가까운 재료를 나르는 채널이 둘 있고, 그 둘의 경계는 정돈이 아니라 보안 경계다:

    ``data_ref``      파일 경로와 목표 열. 실행 노드(``profiling``, ``training``)만 읽고,
                      그 노드들은 이것을 subprocess로 넘긴다. 프롬프트에 그려지지 않는다.
    ``dataset_card``  ``profiling``이 만든 집계 요약. 추론 노드가 데이터에 대해 아는 것은
                      이것뿐이다.

    추론 노드에 ``data_ref`` 읽기를 하나 추가하면 이 격리가 무너지므로,
    ``tests/test_privacy.py``가 보관된 프롬프트에 경로가 나타나지 않음을 확인한다.
    """

    dataset_card: dict
    # 비공개: {"path": ..., "target_column": ...}, 또는 카드에서 합성하려면 {}.
    data_ref: dict
    goal: dict  # {"metric": ..., "threshold": ..., "direction": "maximize"}
    plan: dict
    model: str
    hyperparams: dict
    selection_source: str
    result: dict
    critic: dict
    history: Annotated[list[Attempt], operator.add]  # reducer로 누적된다
    iteration: int
    max_iterations: int
    stall_count: int  # 개선 없는 반복의 연속 횟수
    best: dict  # 지금까지 최고 결과의 스냅샷
    report: str
    # 명세의 스키마를 넘어선 확장: evaluate 노드가 낸 파생 숫자(score / improved / goal_met).
    # CLI가 반복별 요약 한 줄을 찍고 재개된 실행이 그것을 다시 찍을 수 있도록 state에 둔다.
    evaluation: dict
    # 실행 중 어떤 결정도 근거로 삼지 않은 유일한 점수: 루프가 멈춘 뒤 test 조각에서 저장된
    # 최고 모델을 한 번 측정한 것 (``nodes/holdout.py``). 일부러 ``result``에 합치지 *않는다* —
    # ``stop_condition``과 ``goal_met``이 그 채널을 읽으므로, 루프를 조종한 숫자라면 그것은 떼어
    # 둔 숫자가 아니다.
    holdout: dict
    # ``{"spent_sec": 812.4, "total_sec": 3600.0}`` — 가정하지 않고 계산한 실행의 시간 예산.
    # 그래프의 노드 wrapper(``graph._bind``)가 쓰므로 LLM 호출 안에서 시간을 쓰는 노드까지
    # 모두 세어진다. 누적이라서 재개에 안전하다: ``--time-budget-sec``는 실행이 *일한* 초를
    # 제한하고 시작 이후의 wall clock을 제한하지 않으므로, 중단된 실행은 이미 쓴 만큼을 안고
    # 재개된다.
    budget: dict


# --------------------------------------------------------------------------- #
# 결정론적 보조 함수 — LLM이 끼지 않는다
# --------------------------------------------------------------------------- #


def state_int(state: AutoMLState, key: str, default: int = 0) -> int:
    """카운터 채널 하나를 ``int``로. 없거나 ``None``이거나 ``0``이면 ``default``.

    ``0``을 ``default``로 접는 것까지가 계약이다. 호출자마다 fallback을 따로 적으면 갈라진다 —
    ``max_iterations``의 fallback이 ``graph.route``에서는 ``DEFAULT_MAX_ITERATIONS``,
    ``report.stop_reason``에서는 ``0``이어서 일어나지 않은 멈춤을 보고한 일이 있었다.
    """
    # ``key``가 리터럴이 아니므로 TypedDict의 ``get``은 ``object``를 돌려준다. 값이 ``int``라는
    # 것은 채널 정의가 보장하고, ``int()``가 그 밖을 걸러낸다.
    value: Any = state.get(key, default)
    return int(value or default)


def metric_value(result: dict, metric: str) -> float | None:
    """학습 결과에서 ``metric``을 꺼낸다. 두 가지 형태를 모두 받는다.

    ``scripts/train.py``는 ``{"metrics": {"f1": ...}, ...}``를 돌려주는데 명세의
    ``Attempt.result`` 예시는 평평하다 (``{"f1": ...}``). 둘 다 받고, 오류이거나 없는 지표에는
    ``None``을 돌려줘서 호출자가 "점수 없음"으로 다룰 수 있게 한다.
    """
    if not isinstance(result, dict):
        return None
    if result.get("status") == "error":
        return None
    metrics = result.get("metrics")
    return as_number(metrics.get(metric) if isinstance(metrics, dict) else result.get(metric))


def is_better(candidate: float | None, incumbent: float | None, direction: str) -> bool:
    """``direction``에서 ``candidate``가 ``incumbent``를 이기면 True."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if direction == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def effective_hyperparams(state: AutoMLState) -> dict:
    """기록이 인용해야 하는 것: 적용된 값, 없으면 제안.

    ``state["hyperparams"]``는 *제안*이고, ``scripts/train.py``가 그것을 선택된 estimator가 받는
    것으로 좁힌다. 제안을 인용한 보고서는 조용히 버려진 파라미터를 그 점수를 낸 설정으로
    적을 수 있다.

    실패한 시도만은 제안이 더 나은 기록이다 — 적용된 것이 없고, Critic이 실패를 진단하려면
    제안이 필요하다 (OOM 뒤에 있던 ``batch_size`` 같은 것).
    """
    result = state.get("result") or {}
    applied = result.get("applied_hyperparams") if isinstance(result, dict) else None
    if result.get("status") == "ok" and isinstance(applied, dict):
        return dict(applied)
    return dict(state.get("hyperparams") or {})


def drop_pinned_seed(hyperparams: Any, seed: int | None) -> dict:
    """실행 자신의 seed를 되풀이하는 ``random_state``만 뺀 하이퍼파라미터.

    :mod:`automl_agent.scripts.train`의 모든 estimator는 ``random_state=seed``로 만들어지므로,
    같은 숫자를 적은 계획은 적합을 바꾸지 않고 기록만 바꾼다. 그리고 ``model_selection``이 그것을
    무시할 수 없을 만큼 자주 되풀이한다.

    호출자 둘(``critic._other_levers_held``, ``planning._signature``)은 하이퍼파라미터 dict 두 개를
    견주므로 먼저 이 되풀이를 없애야 한다 — 둘 다 남겨 뒀다가 물린 자리다 (``docs/rationale.md``).

    값이 seed와 같을 때만이다. ``--seed 42``에서의 ``random_state: 7``은 실제 레버다 —
    ``early_stopping``의 내부 분할을 움직인다 — 그래서 계속 세어진다.
    """
    values = dict(hyperparams or {})
    pinned = values.get("random_state")
    if seed is not None and not isinstance(pinned, bool) and pinned == seed:
        values.pop("random_state")
    return values


def build_attempt(state: AutoMLState, critic: dict | None = None) -> Attempt:
    """``history`` 채널에 넣을 현재 반복의 스냅샷.

    붙이는 일은 ``critic``(루프 중간)과 ``report``(마지막 반복)에서 일어나고 반복마다 정확히
    하나만 실행되므로, 모든 시도가 판정을 이미 안고 history에 한 번씩 들어간다. ``evaluate``는
    할 수 없다: ``operator.add`` reducer에서는 먼저 들어간 항목을 나중에 고칠 수 없다.
    """
    return Attempt(
        iteration=state_int(state, "iteration"),
        plan=dict(state.get("plan") or {}),
        model=str(state.get("model") or ""),
        hyperparams=effective_hyperparams(state),
        result=dict(state.get("result") or {}),
        critic=dict(critic) if critic else None,
        selection_source=str(state.get("selection_source") or ""),
    )


# --------------------------------------------------------------------------- #
# 시간 예산 — ``budget`` 채널 위의 산술
# --------------------------------------------------------------------------- #
#
# ``--time-budget-sec``는 원래 학습 subprocess마다 *그 자신의* timeout으로 넘겨졌고 다른
# 누구도 읽지 않았다. 그래서 기본값(반복 5회, 3600초)에서 최악의 경우는 적합에 쓰는
# 21,600초였고, 플래그는 어떤 실행도 제한하지 않았다. 이 보조 함수들이 그것을 실행 예산으로
# 만든다: 하나는 누적하고, 하나는 루프가 끝났음을 정하고, 둘은 남은 것을 나누고, 나머지는
# 채널을 읽는다.
#
# 모두 fail-open이다 — ``total_sec``이 없거나 0 이하이면 "예산 없음"으로 답한다
# (``docs/rationale.md``).


def accrue_budget(previous: dict | None, seconds: float, total_sec: float) -> dict:
    """실행이 쓴 시간에 ``seconds``를 더한다. 이 채널의 유일한 writer."""
    spent = float((previous or {}).get("spent_sec", 0.0) or 0.0) + max(0.0, float(seconds))
    return {"spent_sec": round(spent, 3), "total_sec": float(total_sec)}


def budget_total_sec(state: AutoMLState) -> float | None:
    """실행의 예산 전체, 강제할 예산이 없으면 ``None``."""
    total = as_number((state.get("budget") or {}).get("total_sec"))
    return total if total is not None and total > 0 else None


def budget_spent_sec(state: AutoMLState) -> float:
    return max(0.0, as_number((state.get("budget") or {}).get("spent_sec")) or 0.0)


def loop_time_remaining_sec(state: AutoMLState) -> float | None:
    """*루프*가 아직 쓸 수 있는 시간 — 예산에서 holdout이 떼어 둔 몫을 뺀 것.

    루프가 예산 전체를 받지 않는 이유: 이 실행이 보고하는 숫자는 루프가 멈춘 뒤 ``holdout``이
    측정하는 것이고, 0까지 쓸 수 있는 예산은 실행 자신의 답을 지우는 예산이다
    (``docs/rationale.md``).
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    return total * (1.0 - HOLDOUT_RESERVE_FRACTION) - budget_spent_sec(state)


def loop_budget_exhausted(state: AutoMLState) -> bool:
    """반복을 하나 더 하면 실행에 없는 시간을 쓰게 되는지."""
    remaining = loop_time_remaining_sec(state)
    return remaining is not None and remaining <= 0.0


def fit_share_sec(state: AutoMLState) -> float | None:
    """적합 하나의 몫: 루프에 남은 시간을 아직 돌 수 있는 반복 수로 나눈 것.

    통째로 넘기지 않고 나눈다 — 반복 1이 전부를 쓰면 반복 2에 닿지 못하고, 그런 루프는 이
    저장소가 측정하는 것이 아니다 (``docs/rationale.md``). 현재 반복은 자기를 센다: 5회 중 1회에서
    적합은 5분의 1을 받고, 5회에서는 남은 것을 전부 받는다.

    0 이하로 돌아올 수 있다 — ``stop_condition``은 반복 사이에 예산을 검사하고, 그 결정 뒤의
    planning·model_selection 호출도 시간을 쓴다. 판단은 호출자가 한다(``nodes/training.py``가
    적합 시작을 거절한다). 여기서 양수로 clamp하면 실행에 없는 예산을 쓰게 된다.
    """
    remaining = loop_time_remaining_sec(state)
    if remaining is None:
        return None
    left = max(1, state_int(state, "max_iterations") - state_int(state, "iteration") + 1)
    share = remaining / left
    return share if share <= 0 else max(MIN_FIT_TIMEOUT_SEC, share)


def holdout_share_sec(state: AutoMLState) -> float | None:
    """마지막 채점에 남은 시간. 떼어 둔 몫보다 적어지지는 않는다.

    적어지지 않는 이유: 이미 진행 중인 적합이 루프의 몫을 넘길 수 있다. 떼어 둔 몫은 루프가
    손대지 못하게 한 것이므로, 계산상 실행이 이미 끝났다고 나와도 holdout은 그것을 받는다
    (``docs/rationale.md``).
    """
    total = budget_total_sec(state)
    if total is None:
        return None
    reserve = total * HOLDOUT_RESERVE_FRACTION
    return max(reserve, total - budget_spent_sec(state))


def describe_budget(budget: dict) -> str:
    """콘솔과 보고서를 위한 ``2,913초 / 3,600초 (81%)``."""
    spent = float(budget.get("spent_sec", 0.0) or 0.0)
    total = as_number(budget.get("total_sec"))
    if total is not None and total > 0:
        return f"{spent:,.0f}초 / {total:,.0f}초 ({spent / total:.0%})"
    return f"{spent:,.0f}초 (예산 없음)"


def goal_met(result: dict, goal: dict) -> bool:
    """``result``가 목표 기준선에 닿았는지. 오류는 결코 목표를 만족시키지 않는다.

    기준선이 없는 목표는 결코 달성되지 않는다. ``profiling``이 그런 실행을 루프 시작 전에 멈추므로,
    이 분기는 손으로 고친 체크포인트가 여기서 터지는 것만 막는다 (``docs/rationale.md``).
    """
    metric = str(goal.get("metric", "f1"))
    score = metric_value(result, metric)
    if score is None:
        return False
    raw = goal.get("threshold")
    if raw is None:
        return False
    threshold = float(raw)
    if str(goal.get("direction", "maximize")) == "minimize":
        return score <= threshold
    return score >= threshold
