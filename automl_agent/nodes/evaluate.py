"""Evaluate 노드: 결정적인 지표 기록. LLM 없음.

``route``가 의지하는 두 계수기 — ``best``와 ``stall_count`` — 의 주인이다. 그래서 멈춤 결정은 여기서
계산된 수에서 유도되고, 판단에서는 절대 나오지 않는다.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig
from ..scoring.metrics import direction_of
from ..state import AutoMLState, effective_hyperparams, goal_met, is_better, metric_value, state_int


def evaluate(state: AutoMLState, *, config: RunConfig) -> dict:
    """마지막 결과를 채점하고 ``best``와 ``stall_count``를 갱신한다."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    direction = str(goal.get("direction") or direction_of(metric))
    result = dict(state.get("result") or {})
    iteration = state_int(state, "iteration")

    score = metric_value(result, metric)
    previous_best = dict(state.get("best") or {})
    improved = is_better(score, previous_best.get("score"), direction)

    best = previous_best
    if improved:
        best = {
            "iteration": iteration,
            "model": str(state.get("model") or ""),
            # 적용된 파라미터. 그래야 "최고 성능 구성"이 다시 돌려서 이 점수를 재현할 수 있는 구성이
            # 된다. executor가 세운 파이프라인도 그 구성에 속한다: 같은 추정기를 중앙값으로 대치된 열에
            # 적합한 것과 날 NaN에 적합한 것은 서로 다른 두 모델이다.
            "hyperparams": effective_hyperparams(state),
            "preprocessing": dict(result.get("applied_preprocessing") or {}),
            "metric": metric,
            "score": score,
            "metrics": dict(result.get("metrics") or {}),
            "train_time_sec": result.get("train_time_sec"),
            "plan_strategy": (state.get("plan") or {}).get("strategy", ""),
        }

    # 실패했거나 나아지지 않은 시도는 정체로 센다. 개선은 무엇이든 그것을 되돌린다.
    stall_count = 0 if improved else state_int(state, "stall_count") + 1

    evaluation: dict[str, Any] = {
        "iteration": iteration,
        "metric": metric,
        "score": score,
        "goal_met": goal_met(result, goal),
        "improved": improved,
        "status": result.get("status", "error"),
        "error_type": result.get("error_type"),
    }
    return {"best": best, "stall_count": stall_count, "evaluation": evaluation}
