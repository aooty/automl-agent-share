"""Planning 에이전트: 추론의 진입점이고, history를 소비하는 노드.

재시도에서 이 노드는 앞선 모든 시도와 다른 계획을 내야 한다. 그래서 프롬프트가 시도 요약 전체와
Critic의 판정을 함께 나른다 — 두 번째 시도를 첫 번째보다 똑똑하게 만드는 것은 공유된 state이고, 모델의
기억이 아니다.

반복 계수기의 주인도 이 노드다: planning에 들어오는 것이 곧 새 반복이다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..capabilities import describe as describe_capabilities
from ..capabilities import describe_row_budget, explain_claims, unsupported_claims
from ..config import RunConfig
from ..dataset.caveats import describe_caveats
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.goal import describe as describe_goal
from ..scoring.metrics import TASK_REGRESSION
from ..state import AutoMLState, drop_pinned_seed, state_int
from .model_selection import (
    DEFAULT_MODEL,
    _history_digest,
    available_ids,
    available_models,
    registry,
    sanitise_hyperparams,
    task_of_state,
)


def families(task: str | None = None) -> list[str]:
    """이 task의 registry가 제공하는 계열 이름, 계획 스키마의 enum용."""
    return sorted({str(entry["family"]) for entry in registry(task)})


def plan_schema(task: str | None = None) -> dict[str, Any]:
    """이 task의 계획 스키마. 그래서 어느 enum도 다른 task의 모델을 지목할 수 없다.

    모듈마다가 아니라 호출마다인 이유는 :func:`automl_agent.nodes.model_selection.selection_schema`와
    같다: task를 건너뛴 제안이 시도 하나를 쓰기 전에 막는 것이 그 enum이고, 모듈 수준 상수 하나는 한
    task의 메뉴만 담을 수 있다.
    """
    return {
        "type": "object",
        "properties": {
            "strategy": {"type": "string"},
            "model_family": {"type": "string", "enum": families(task)},
            "candidate_models": {
                "type": "array",
                "items": {"type": "string", "enum": [entry["id"] for entry in registry(task)]},
                "minItems": 1,
            },
            "hyperparams": {"type": "object", "additionalProperties": True},
            "preprocessing": {"type": "object", "additionalProperties": True},
            # 같은 주제의 순서 있는 형태. 계획은 둘 중 하나를 준다: spec이 오면 executor가
            # ``preprocessing``을 무시하는데, 한 파이프라인의 기술이 둘이면 어느 쪽이 돌았는지
            # 말할 수 있는 것이 없어지기 때문이다. 항목이 제약 없는 객체인 이유는
            # ``preprocessing``과 같다 — executor가 모든 키를 곧 세울 대상에 대고 검증하고 결과를
            # ``applied_pipeline``에 보고하며, 그 목록의 두 번째 사본이 여기 있으면 그것이 어긋난다.
            "pipeline": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            # 수가 아니라 불리언인 이유: planner는 이 모델의 확률을 본 적이 없고, 손으로 고른 컷은
            # 그 분포에 대한 짐작이다. executor는 손으로 쓴 config에서 명시적 실수를 받는다. 계획이
            # 받는 것은 지렛대이고 값이 아니다. 선택 사항이다: 없음은 기본 0.5 규칙이고, 이 키가 생기기
            # 전에 쓰인 모든 계획이 그 뜻이다.
            "tune_threshold": {"type": "boolean"},
            "changes_from_last": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "strategy",
            "model_family",
            "candidate_models",
            "hyperparams",
            "changes_from_last",
            "rationale",
        ],
        "additionalProperties": False,
    }


# 계열 자체가 틀렸다고 들었을 때 결정적인 planner가 걷는 순서.
FAMILY_ROTATION = ("gbdt", "bagging", "linear", "neural", "tree", "kernel", "instance")
RESOURCE_FAILURES = {"oom", "too_slow"}
# registry 자신의 params 위에 executor가 존중하는 설정. 그래서 이것만 다시 조율한 것도 새 계획으로
# 읽힌다. 메뉴가 아니라 여기 있는 것은 일부러다: registry는 프롬프트로 렌더되므로, 그것을 넓히면 LLM이
# 조율하라고 초대받는 범위가 넓어진다.
#
# 마지막 둘은 early-stopping 짝이다. 둘 다 광고되지 않고, 멈춤이 켜지면 둘 다 적합이 보는 것을 바꾼다 —
# ``validation_fraction``은 train의 얼마를 떼어 둘지 정하고(``capabilities``가 ``'auto'`` 경계를 알릴 때
# 그 수를 계획에 인용한다), ``n_iter_no_change``는 그것이 물 때를 정하는 인내다. 이 둘이 없으면 "덜
# 떼어 두자"가 유일한 변경인 계획이 다르려 하는 그 시도와 지문이 같아지고, novelty 가드가 청하지 않은
# 계열 교체로 보낸다.
EXECUTOR_PARAMS = frozenset(
    {
        "train_subsample",
        "precision",
        "batch_size",
        "class_weight",
        "validation_fraction",
        "n_iter_no_change",
    }
)


def planning(state: AutoMLState, *, config: RunConfig) -> dict:
    """``{"plan": ..., "iteration": n + 1}``."""
    iteration = state_int(state, "iteration") + 1
    critic_verdict = dict(state.get("critic") or {})
    task = task_of_state(state)

    variables = {
        "dataset_card": state.get("dataset_card") or {},
        # 위의 집계가 표현할 수 없는, 계획에 대한 제약 — automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        "goal": state.get("goal") or {},
        # 같은 dict를 산문으로. 그 키 둘이 의무인데 JSON에서는 어느 쪽도 의무로 읽히지 않기 때문이다.
        # ``"exceeds_ranking_ceiling": true``와 ``"passable_margin"``을 맨 키로 건네면, 실행은 그 순위
        # 위의 어떤 문턱값도 닿을 수 없는 바에 대고 예산 전부를 조율과 계열 교체에 쓸 수 있다. 바로
        # 그것을 설명하는 문장이 ``goal.describe``에 이미 있었고 콘솔과 보고서에만 갔다.
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "iteration": str(iteration),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "history": _history_digest(state) or "(no previous attempt — this is the first plan)",
        "best": state.get("best") or "(no successful attempt yet)",
        "critic": critic_verdict or "(no critic verdict yet — this is the first attempt)",
        "available_models": available_models(task),
        "executor_capabilities": describe_capabilities(task),
        # 카드의 행 수를 분할에 통과시킨 것, 그리고 그 곱이 무엇을 정하는지 —
        # automl_agent.capabilities.describe_row_budget. 이것을 받는 것은 planning 프롬프트뿐이다:
        # iteration 2부터는 executor가 실제 개수를 결과의 ``internal_validation``에 보고했고 Critic이
        # 그 결과를 읽는다. iteration 1에는 결과가 없는데, bench/의 모든 시도가 early-stopping 설정을
        # 고른 곳이 거기다.
        "row_budget": describe_row_budget(
            (state.get("dataset_card") or {}).get("n_rows"),
            grouped=bool(_grouped_by(state)),
        ),
    }

    plan: dict[str, Any] | None = None
    if not config.use_llm:
        archive_prompt_only(config, f"planning_iter{iteration}", render_prompt("planning", variables))
    else:
        try:
            plan = LLMClient(config, proposer=True).complete_json(
                "planning", variables, plan_schema(task), iteration=iteration
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [planning] LLM 계획 수립 실패({exc}) — 규칙 기반 계획으로 폴백합니다")

    proposed = validate_plan(plan, task)
    plan = proposed or fallback_plan(state, critic_verdict, iteration, task)
    # 둘 중 어느 쪽이 이 계획을 썼는지.
    plan["source"] = "llm" if proposed else ("fallback" if config.use_llm else "rules")
    plan = enforce_novelty(plan, state, task, config.seed)
    plan["iteration"] = iteration
    if plan.get("unsupported_claims"):
        # 표시만 하고 거절하지 않는다: 거절해도 같은 반복을 쓰고, 검출기는 부분 문자열 일치다.
        # 소리 내어 말하는 것이 보고서가 실행되지 않은 단계를 이긴 구성의 일부로 인용하지 않게 한다.
        print(
            f"  [planning] 계획이 실행기에 없는 기능을 전제한 것으로 보입니다 — "
            f"{explain_claims(list(plan['unsupported_claims']))}. 그 부분은 실행되지 않습니다"
        )
    return {"plan": plan, "iteration": iteration}


def validate_plan(plan: dict[str, Any] | None, task: str | None = None) -> dict[str, Any] | None:
    """돌릴 수 있는 후보만 남기고, 하나도 남지 않으면 계획 전체를 거절한다.

    "돌릴 수 있음"은 task에 상대적이므로, 후보 목록 전체가 다른 task에 속한 계획은 여기서 거절되고
    결정적인 planner가 이어받는다 — 그것이 옳은 결과다: 연속 정답에 대한 분류기 목록은 첫 선택이 나쁜
    계획이 아니라 틀린 메뉴에 대고 세운 계획이다.
    """
    if not isinstance(plan, dict):
        return None
    ids = available_ids(task)
    candidates = [
        candidate.strip().lower()
        for candidate in (plan.get("candidate_models") or [])
        if isinstance(candidate, str) and candidate.strip().lower() in ids
    ]
    if not candidates:
        return None
    strategy = str(plan.get("strategy") or "").strip() or "(전략 설명 없음)"
    changes = str(plan.get("changes_from_last") or "").strip()
    rationale = str(plan.get("rationale") or "").strip()
    return {
        "strategy": strategy,
        "model_family": str(plan.get("model_family") or "").strip().lower(),
        "candidate_models": list(dict.fromkeys(candidates)),
        "hyperparams": sanitise_hyperparams(plan.get("hyperparams")),
        "preprocessing": plan.get("preprocessing") if isinstance(plan.get("preprocessing"), dict) else {},
        # 산문이 아니라 *키*인 지렛대 둘. ``preprocessing``과 같은 이유로 실어 나른다: 이 함수는 고정된
        # 목록에서 계획을 다시 세우므로, 여기서 이름을 대지 않는 키는 어느 노드가 읽기도 전에 떨어진다.
        "pipeline": plan.get("pipeline") if isinstance(plan.get("pipeline"), list) else [],
        "tune_threshold": plan.get("tune_threshold") is True,
        "changes_from_last": changes,
        "rationale": rationale,
        # 없는 기능이 숨는 곳은 산문이다: 하이퍼파라미터 키였다면 이미 결과의
        # ``dropped_hyperparams``에 나타난다.
        "unsupported_claims": unsupported_claims(strategy, changes, rationale),
        "source": "llm",
    }


# --------------------------------------------------------------------------- #
# 결정적인 planner (--dry-run, 그리고 LLM을 못 쓸 때의 폴백)
# --------------------------------------------------------------------------- #


def fallback_plan(
    state: AutoMLState,
    verdict: dict[str, Any],
    iteration: int,
    task: str | None = None,
) -> dict[str, Any]:
    """Critic의 방향은 그대로 존중하는 규칙 기반 재계획.

    프롬프트의 규칙을 그대로 비춘다: 자원 실패는 언제나 발자국을 줄이고, 모든 계획은 직전 시도에 대해
    무언가를 바꾼다.
    """
    failure_type = str(verdict.get("failure_type") or "")
    changes = dict(verdict.get("concrete_changes") or {})
    previous_hyperparams = dict(state.get("hyperparams") or {})
    tried_models = [str(item.get("model") or "") for item in (state.get("history") or [])]
    # 진단이 계열이 아니라 *조율*에 대한 것일 때 머무를 계열. 아래 네 분기가 같은 조회를 원해서 끌어
    # 올렸다. 분기마다 되풀이되는 registry 스캔은 기본값이 어긋날 자리가 하나 더 생기는 일이다.
    same_family = _family_of(state.get("model"), task) or "gbdt"

    if not state.get("history"):
        family = "gbdt"
        hyperparams: dict[str, Any] = {"max_iter": 150, "learning_rate": 0.1}
        strategy = "baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다"
        change_note = "첫 시도이므로 비교 대상 없음"
    elif failure_type in RESOURCE_FAILURES:
        # 자원 실패에는 줄이기만 하고 절대 늘리지 않는다.
        family = "linear" if _cost_of(state.get("model"), task) >= 3 else "tree"
        hyperparams = {
            **{k: v for k, v in previous_hyperparams.items() if k not in {"n_estimators", "max_iter"}},
            "max_iter": 300 if family == "linear" else 80,
            "train_subsample": 0.5,
            "batch_size": 16,
            "precision": "fp16",
        }
        strategy = (
            f"자원 실패({failure_type}) 대응: 더 가벼운 {family} 계열로 축소하고 "
            "학습 데이터를 절반만 사용"
        )
        change_note = (
            f"이전 시도가 {failure_type}로 실패 — 모델 축소 + batch_size 16 + fp16 + 서브샘플 0.5"
        )
    elif failure_type == "underfitting":
        family = same_family
        hyperparams = {
            **previous_hyperparams,
            "max_iter": min(1200, max(300, int(previous_hyperparams.get("max_iter", 150)) * 3)),
            "learning_rate": 0.08,
            "max_leaf_nodes": 63,
        }
        strategy = "과소적합 대응: 반복 수를 3배로 늘리고 리프 수를 확대해 용량을 키운다"
        change_note = "과소적합 진단 — max_iter 증가, max_leaf_nodes 63으로 용량 확대"
    elif failure_type == "overfitting":
        family = same_family
        hyperparams = {
            **previous_hyperparams,
            "l2_regularization": 1.0,
            "max_depth": 4,
            "learning_rate": 0.05,
        }
        strategy = "과적합 대응: l2 정규화를 넣고 깊이와 learning_rate를 낮춘다"
        change_note = "과적합 진단 — l2_regularization 1.0, max_depth 4로 정규화 강화"
    elif failure_type == "wrong_model_family":
        family = _next_family(tried_models, task)
        hyperparams = {"max_iter": 300} if family == "linear" else {"n_estimators": 300}
        strategy = f"계열 교체: 기존 계열이 정체되었으므로 {family} 계열로 전환한다"
        change_note = f"모델 계열 정체 진단 — {family} 계열로 완전 교체"
    elif failure_type == "data_issue":
        family = same_family
        if task == TASK_REGRESSION:
            # 연속 정답에는 불균형 지렛대가 없으므로, 여기서 ``class_weight``를 처방하면 executor가
            # 떨어뜨리는 키를 계획에 넣는 것이 된다 — ``capabilities``가 막으려고 존재하는 바로 그
            # 결함이, LLM이 아니라 우리 자신의 폴백에서 오는 것이다. executor가 실제로 하는 것 중 남는
            # 것은 두꺼운 꼬리에 덜 흔들리는 계열이므로, 계획은 대신 그것을 말한다.
            family = "gbdt" if family in {"linear", "kernel", "instance", "neural"} else family
            hyperparams = dict(previous_hyperparams)
            strategy = "데이터 문제 대응: 정답 열의 꼬리와 이상치에 덜 흔들리는 트리 계열로 옮긴다"
            change_note = f"데이터 이슈 진단 — 회귀에는 class_weight가 없으므로 {family} 계열로 전환"
        else:
            hyperparams = {**previous_hyperparams, "class_weight": "balanced"}
            strategy = "데이터 문제 대응: 클래스 가중치를 균형화한다"
            change_note = "데이터 이슈 진단 — class_weight=balanced 적용"
    else:  # "hyperparam" and "unknown"
        family = same_family
        step = iteration % 3
        hyperparams = {
            **previous_hyperparams,
            "learning_rate": [0.03, 0.05, 0.15][step],
            "max_depth": [6, 8, 12][step],
            "max_iter": 400,
        }
        strategy = "하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다"
        change_note = f"{failure_type or 'unknown'} 진단 — learning_rate/max_depth 조합 변경"

    # Critic의 구체적인 수가 규칙의 기본값을 이긴다.
    hyperparams.update({k: v for k, v in changes.items() if k not in {"model", "model_family"}})

    rationale = (
        f"critic.failure_type={failure_type or 'none'} / direction="
        f"{verdict.get('direction', '(없음)')}"
    )

    return {
        "strategy": strategy,
        "model_family": family,
        "candidate_models": _candidates_for(family, task),
        "hyperparams": sanitise_hyperparams(hyperparams),
        "preprocessing": {"impute": "median"},
        "changes_from_last": change_note,
        "rationale": rationale,
        # 빈 목록을 박아 넣는 대신 같은 검출기를 돌린다: 규칙 기반 산문은 여기서 쓰지만, 그것이
        # 인용하는 Critic의 ``direction``은 여기서 쓴 것이 아니다.
        "unsupported_claims": unsupported_claims(strategy, change_note, rationale),
        "source": "heuristic",
    }


def enforce_novelty(
    plan: dict[str, Any], state: AutoMLState, task: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    """계획이 앞선 시도의 완전한 반복이 아님을 보장한다.

    반복된 시도는 정보 없이 반복 하나를 태우므로, 최상위 후보와 하이퍼파라미터가 앞선 시도와 맞으면
    시도되지 않은 모델로 돌린다.

    **지문은 executor에 닿는 모든 것을 봐야 한다.**
    """
    history = list(state.get("history") or [])
    if not history:
        return plan
    # executor가 각 모델에 적용한다고 알려진 것. 광고된 메뉴가 아니라 executor 자신의 보고에서 읽는다.
    # ``registry``가 펴내는 것은 골라낸 목록이고 — 그것을 넓히면 프롬프트가 LLM에게 조율하라고 초대하는
    # 범위도 넓어진다 — 거기서 벗어난 키는 그것이 유일한 변경일 때 반복으로 읽힌다.
    # ``EXECUTOR_PARAMS``가 미리 이름 댈 만한 것을 덮고, 나머지는 ``applied_hyperparams``가 덮는다.
    # 그것이 executor가 무엇을 받았다고 말하는 것이기 때문이다. 이 절반만 잡는 예가
    # ``min_child_weight``다. (``early_stopping``도 그랬지만 지금은 메뉴에 있다.)
    applied_keys: dict[str, set[str]] = {}
    for item in history:
        applied = (item.get("result") or {}).get("applied_hyperparams")
        if isinstance(applied, dict):
            applied_keys.setdefault(str(item.get("model") or ""), set()).update(applied)
    candidates = list(plan.get("candidate_models") or [DEFAULT_MODEL])
    top = str(candidates[0])
    signature = (
        top,
        _signature(
            plan.get("hyperparams") or {},
            top,
            task,
            preprocessing=plan.get("preprocessing"),
            extra_keys=applied_keys.get(top, frozenset()),
            seed=seed,
            tune_threshold=plan.get("tune_threshold"),
            spec=plan.get("pipeline"),
        ),
    )
    seen = set()
    for item in history:
        tried_model = str(item.get("model") or "")
        seen.add(
            (
                tried_model,
                _signature(
                    item.get("hyperparams") or {},
                    tried_model,
                    task,
                    preprocessing=_requested_preprocessing(item),
                    extra_keys=applied_keys.get(tried_model, frozenset()),
                    seed=seed,
                    # 옆의 파이프라인처럼 계획 자신의 키다: 청한 것에 대고 청한 것. 컷을 청했다가
                    # 거절당한(떼어 둔 행이 너무 적어) 시도도 청했다고 읽히고, 그것이 틀려야 하는 쪽이다
                    # — 반대쪽은 정당한 재시도를 계열 교체로 다시 쓴다.
                    tune_threshold=(item.get("plan") or {}).get("tune_threshold"),
                    spec=(item.get("plan") or {}).get("pipeline"),
                ),
            )
        )
    if signature not in seen:
        return plan

    tried = {str(item.get("model") or "") for item in history}
    for entry in sorted(available_models(task), key=lambda item: int(item["cost"])):
        if str(entry["id"]) not in tried:
            plan["candidate_models"] = [str(entry["id"]), *candidates]
            plan["changes_from_last"] = (
                f"{plan.get('changes_from_last', '')} / 동일 구성 반복을 피해 미시도 모델 "
                f"{entry['id']}로 교체"
            ).strip(" /")
            plan["model_family"] = str(entry["family"])
            break
    else:
        # 모든 모델을 시도했다: 대신 하이퍼파라미터를 흔든다.
        hyperparams = dict(plan.get("hyperparams") or {})
        hyperparams["learning_rate"] = round(float(hyperparams.get("learning_rate", 0.1)) * 0.5, 5)
        hyperparams["max_iter"] = int(hyperparams.get("max_iter", 200)) + 100 * len(history)
        plan["hyperparams"] = sanitise_hyperparams(hyperparams)
        plan["changes_from_last"] = (
            f"{plan.get('changes_from_last', '')} / 모든 모델을 시도했으므로 하이퍼파라미터를 변형"
        ).strip(" /")
    return plan


def _signature(
    hyperparams: dict[str, Any],
    model: str | None = None,
    task: str | None = None,
    *,
    preprocessing: Any = None,
    extra_keys: Iterable[str] = (),
    seed: int | None = None,
    tune_threshold: Any = None,
    spec: Any = None,
) -> str:
    """executor에 닿는 모든 것을, 그리고 그것이 무시할 것은 아무것도, 지문으로 뜬다.

    ``logreg``의 ``learning_rate``를 바꾸면 새 계획처럼 보이면서 바이트까지 같은 실행이 나온다. 그래서
    걸러지지 않은 서명은 루프가 아무 효과 없는 변형에 반복을 태우게 한다.

    params 목록은 task에 상대적이다 — ``linreg``는 *아무것도* 소비하지 않으므로, 빈 registry 항목이 그것에
    대한 모든 제안이 무효임을 루프에게 보이게 한다. ``extra_keys``는 반대 방향이고,
    ``applied_hyperparams``에서 읽으므로 발을 맞춰야 하는 두 번째 목록이 아니라 무엇이 돌았다는 사실로
    남는다.

    **파이프라인은 지문에 있고, 청한 것에 대고 청한 것으로 견준다** — 계획의 *적용된* 파이프라인은 아직
    없으므로 그것이 유일하게 대칭인 선택이다. 그래서 executor가 같은 파이프라인으로 줄이는 두 요청이 다르게
    읽히고, 이미 거절된 키를 빼는 계획이 새것으로 통과해 반복 하나를 쓴다. **그쪽이 틀려야 하는 방향이다**:
    반대쪽은 정당한 한 지렛대 계획을 조용히 계열 교체로 다시 쓴다.
    """
    effective = hyperparams
    entry = _entry_of(model, task) if model else None
    if entry is not None:
        allowed = set(entry["params"]) | EXECUTOR_PARAMS | set(extra_keys)
        effective = {key: value for key, value in hyperparams.items() if key in allowed}
    effective = drop_pinned_seed(effective, seed)
    parts = [f"{key}={_canonical(effective[key])}" for key in sorted(effective)]
    pipeline = dict(preprocessing) if isinstance(preprocessing, Mapping) else {}
    parts += [f"prep.{key}={_canonical(pipeline[key])}" for key in sorted(pipeline)]
    if isinstance(spec, (list, tuple)) and spec:
        # 선언된 파이프라인. 플래그 블록과 같은 방식으로, 같은 이유로 정규화한다: executor에 닿으므로,
        # 어떤 단계의 열 목록만 옮긴 재계획은 다른 시도이고 반복으로 읽혀서는 안 된다. 키를 정렬해
        # ``json.dumps``로 렌더하므로 지문이 LLM이 어쩌다 단계의 키를 내놓은 순서에 달리지 않는다 —
        # 오직 *단계*의 순서에만 달리고, 그것이 실제로 도는 부분이다.
        parts.append("spec=" + json.dumps(spec, sort_keys=True, default=str))
    if tune_threshold is True:
        # 청했을 때만 덧붙인다. 그래서 이 지렛대가 생기기 전에 계산된 모든 서명이 그대로이고, 재개된
        # 실행의 history가 여전히 자기와 맞는다. 애초에 여기 있어야 하는 이유는 파이프라인과 같다:
        # executor에 닿는 세 번째 축이고, 그것이 없으면 컷만 옮긴 재계획이 완전한 반복으로 읽혀 계열
        # 교체로 다시 쓰인다 — ``early_stopping``과 ``scale_pos_weight``에 대해 이미 닫은 바로 그
        # 결함이다.
        parts.append("cut=tuned")
    return ",".join(parts)


def _canonical(value: Any) -> str:
    """값의 표기를, 그것이 했거나 하지 않은 JSON 왕복과 무관하게 만든다.

    ``sanitise_hyperparams``는 가중치 맵에 정수 키를 주고, executor의 ``train_result.json``에서 되읽은
    같은 맵은 문자열 키를 갖는다. 그 두 표기를 견주다가 진짜 반복이 novelty 가드를 통과한 적이 있고,
    판정이 dict가 서브프로세스 경계의 어느 쪽에서 왔는지에 달리는 가드는 가드가 아니다.
    """
    if isinstance(value, Mapping):
        return json.dumps({str(key): value[key] for key in value}, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _requested_preprocessing(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """앞선 시도가 *청한* 파이프라인. 없으면 실제로 돈 것으로 떨어진다.

    비교의 옳은 쪽은 계획이고(:func:`_signature` 참고), 그것은
    :func:`automl_agent.state.build_attempt`가 만드는 모든 항목에 있다. 폴백은 계획을 보관하기 전의
    기록을 위한 것이고, 거기서는 적용된 파이프라인이 파일에 있는 유일한 것이다.
    """
    plan = attempt.get("plan")
    if isinstance(plan, Mapping) and isinstance(plan.get("preprocessing"), Mapping):
        return dict(plan["preprocessing"])
    applied = (attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


def _grouped_by(state: AutoMLState) -> str | None:
    """분할이 온전히 지킨 열, 또는 행 단위 경로에서는 ``None``.

    비공개 ``data`` 블록이 아니라 카드가 펴낸 protocol에서 읽는다. 이 노드가 돌 때쯤 그 블록은
    :func:`automl_agent.privacy.public_card`가 이미 벗겨 냈다. ``--no-baseline``으로 세운 카드에는
    protocol 블록이 없고, 그때 호출자가 이것으로 하는 단 하나 — 행 개수를 근사라고 부를지 정하는 것 —
    에 대해 ``None``이 정직한 답이다.
    """
    protocol = ((state.get("dataset_card") or {}).get("baseline") or {}).get("protocol")
    column = protocol.get("grouped_by") if isinstance(protocol, Mapping) else None
    return str(column) if column else None


def _entry_of(model: Any, task: str | None = None) -> dict[str, Any] | None:
    """``model``에 대한 이 task의 registry 항목, 그 메뉴에 그런 id가 없으면 ``None``.

    같은 항목에서 서로 다른 필드를 원했던 호출자 셋을 위한 조회 하나. 각자 ``registry(task)``를 직접
    걷고 있었는데, 그것은 "무엇이 일치인가"가 갈라질 자리가 셋이라는 뜻이다.
    """
    return next((entry for entry in registry(task) if entry["id"] == model), None)


def _family_of(model: Any, task: str | None = None) -> str | None:
    entry = _entry_of(model, task)
    return None if entry is None else str(entry["family"])


def _cost_of(model: Any, task: str | None = None) -> int:
    entry = _entry_of(model, task)
    return 3 if entry is None else int(entry["cost"])


def _next_family(tried_models: list[str], task: str | None = None) -> str:
    tried_families = {_family_of(model, task) for model in tried_models}
    available_families = {str(entry["family"]) for entry in available_models(task)}
    for family in FAMILY_ROTATION:
        if family in available_families and family not in tried_families:
            return family
    return "gbdt"


def _candidates_for(family: str, task: str | None = None) -> list[str]:
    matches = [
        str(entry["id"])
        for entry in sorted(available_models(task), key=lambda item: int(item["cost"]))
        if entry["family"] == family
    ]
    return matches or [DEFAULT_MODEL]
