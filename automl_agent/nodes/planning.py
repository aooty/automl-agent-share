"""Planning agent: the entry point of reasoning, and the node that reads history.

Roles:

* Plan schema: the JSON schema a plan must follow, per task.
* Planning node: get a plan and start a new iteration.
* Plan validation: keep runnable candidates and executor keys only.
* Rule-based planner: plan from the Critic's verdict, no LLM.
* Novelty guard: block exact repeats of earlier attempts.
* Lookups: small helpers over the registry and card.
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

# --- Role: plan schema ----------------------------------------------------------------


def families(task: str | None = None) -> list[str]:
    """families | Plan schema: family names in this task's registry."""
    return sorted({str(entry["family"]) for entry in registry(task)})


def plan_schema(task: str | None = None) -> dict[str, Any]:
    """Build the plan schema per task, so no enum names another task's model.

    Why per call:
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
            # Overrides ``preprocessing`` when given
            "pipeline": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            # A lever, not a cut value
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


# Family order tried when the family is wrong.
FAMILY_ROTATION = ("gbdt", "bagging", "linear", "neural", "tree", "kernel", "instance")
RESOURCE_FAILURES = {"oom", "too_slow"}
# Executor settings outside the registry menu
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


# --- Role: planning node --------------------------------------------------------------


def planning(state: AutoMLState, *, config: RunConfig) -> dict:
    """Make the next plan and advance the iteration counter.

    ``plan["source"]`` is ``llm``, ``fallback``, or ``rules``.
    """
    iteration = state_int(state, "iteration") + 1
    critic_verdict = dict(state.get("critic") or {})
    task = task_of_state(state)

    variables = {
        "dataset_card": state.get("dataset_card") or {},
        # Limits the card numbers cannot express.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        "goal": state.get("goal") or {},
        # Same goal as prose
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "iteration": str(iteration),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "history": _history_digest(state) or "(no previous attempt — this is the first plan)",
        "best": state.get("best") or "(no successful attempt yet)",
        "critic": critic_verdict or "(no critic verdict yet — this is the first attempt)",
        "available_models": available_models(task),
        "executor_capabilities": describe_capabilities(task),
        # Row count after the split
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
    # Who wrote this plan; three values
    plan["source"] = "llm" if proposed else ("fallback" if config.use_llm else "rules")
    plan = enforce_novelty(plan, state, task, config.seed)
    plan["iteration"] = iteration
    if plan.get("unsupported_claims"):
        # Flag only, never reject: the detector is rough.
        print(
            f"  [planning] 계획이 실행기에 없는 기능을 전제한 것으로 보입니다 — "
            f"{explain_claims(list(plan['unsupported_claims']))}. 그 부분은 실행되지 않습니다"
        )
    return {"plan": plan, "iteration": iteration}


# --- Role: plan validation ------------------------------------------------------------


def validate_plan(plan: dict[str, Any] | None, task: str | None = None) -> dict[str, Any] | None:
    """Keep only this task's runnable candidates; return ``None`` if none are left."""
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
        # Keys not named here are dropped
        "pipeline": plan.get("pipeline") if isinstance(plan.get("pipeline"), list) else [],
        "tune_threshold": plan.get("tune_threshold") is True,
        "changes_from_last": changes,
        "rationale": rationale,
        # Checks the prose; bad keys show in ``dropped_hyperparams``.
        "unsupported_claims": unsupported_claims(strategy, changes, rationale),
        "source": "llm",
    }


# --- Role: rule-based planner ---------------------------------------------------------


def fallback_plan(
    state: AutoMLState,
    verdict: dict[str, Any],
    iteration: int,
    task: str | None = None,
) -> dict[str, Any]:
    """Make a rule-based plan that follows the Critic's direction.

    Same rules as the prompt: resource failures shrink, every plan changes something.
    """
    failure_type = str(verdict.get("failure_type") or "")
    changes = dict(verdict.get("concrete_changes") or {})
    previous_hyperparams = dict(state.get("hyperparams") or {})
    tried_models = [str(item.get("model") or "") for item in (state.get("history") or [])]
    # Family kept when the diagnosis is about tuning.
    same_family = _family_of(state.get("model"), task) or "gbdt"

    if not state.get("history"):
        family = "gbdt"
        hyperparams: dict[str, Any] = {"max_iter": 150, "learning_rate": 0.1}
        strategy = "baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다"
        change_note = "첫 시도이므로 비교 대상 없음"
    elif failure_type in RESOURCE_FAILURES:
        # On a resource failure, only shrink, never grow.
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
            # No ``class_weight`` for regression
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

    # The Critic's concrete numbers win over the rule defaults.
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
        # The quoted Critic ``direction`` may claim missing features.
        "unsupported_claims": unsupported_claims(strategy, change_note, rationale),
        "source": "heuristic",
    }


# --- Role: novelty guard --------------------------------------------------------------


def enforce_novelty(
    plan: dict[str, Any], state: AutoMLState, task: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Change the plan in place if it exactly repeats an earlier attempt.

    Switches to an untried model, else changes hyperparams
    """
    history = list(state.get("history") or [])
    if not history:
        return plan
    # Keys the executor really applied
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
                    # Requested against requested
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
        # Every model was tried: change the hyperparameters instead.
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
    """_signature | Novelty guard: fingerprint what the executor uses, minus what it ignores."""
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
        # Only step order matters, not key order
        parts.append("spec=" + json.dumps(spec, sort_keys=True, default=str))
    if tune_threshold is True:
        # Only when asked, so old signatures stay
        parts.append("cut=tuned")
    return ",".join(parts)


def _canonical(value: Any) -> str:
    """_canonical | Novelty guard: write a value the same, with or without JSON round trip."""
    if isinstance(value, Mapping):
        return json.dumps({str(key): value[key] for key in value}, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _requested_preprocessing(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """_requested_preprocessing | Novelty guard: preprocessing asked for, else what ran."""
    plan = attempt.get("plan")
    if isinstance(plan, Mapping) and isinstance(plan.get("preprocessing"), Mapping):
        return dict(plan["preprocessing"])
    applied = (attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


# --- Role: lookups --------------------------------------------------------------------


def _grouped_by(state: AutoMLState) -> str | None:
    """_grouped_by | Lookups: the group column the split kept whole, or ``None``."""
    protocol = ((state.get("dataset_card") or {}).get("baseline") or {}).get("protocol")
    column = protocol.get("grouped_by") if isinstance(protocol, Mapping) else None
    return str(column) if column else None


def _entry_of(model: Any, task: str | None = None) -> dict[str, Any] | None:
    """_entry_of | Lookups: this task's registry entry for ``model``, or ``None``."""
    return next((entry for entry in registry(task) if entry["id"] == model), None)


def _family_of(model: Any, task: str | None = None) -> str | None:
    """_family_of | Lookups: family of ``model``, or ``None`` if unknown."""
    entry = _entry_of(model, task)
    return None if entry is None else str(entry["family"])


def _cost_of(model: Any, task: str | None = None) -> int:
    """_cost_of | Lookups: cost of ``model``, or 3 if unknown."""
    entry = _entry_of(model, task)
    return 3 if entry is None else int(entry["cost"])


def _next_family(tried_models: list[str], task: str | None = None) -> str:
    """_next_family | Lookups: first untried family in :data:`FAMILY_ROTATION`, else ``gbdt``."""
    tried_families = {_family_of(model, task) for model in tried_models}
    available_families = {str(entry["family"]) for entry in available_models(task)}
    for family in FAMILY_ROTATION:
        if family in available_families and family not in tried_families:
            return family
    return "gbdt"


def _candidates_for(family: str, task: str | None = None) -> list[str]:
    """_candidates_for | Lookups: models of a family, cheapest first, else default."""
    matches = [
        str(entry["id"])
        for entry in sorted(available_models(task), key=lambda item: int(item["cost"]))
        if entry["family"] == family
    ]
    return matches or [DEFAULT_MODEL]
