"""Planning agent: the reasoning entry point, and the node that consumes history.

On a retry this node must produce a plan that differs from every earlier attempt.
That is why the prompt carries the full attempt digest plus the Critic's verdict —
the shared state is what makes the second attempt smarter than the first, not the
model's memory.

This node also owns the iteration counter: entering planning *is* a new iteration.
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
from ..state import AutoMLState, drop_pinned_seed
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
    """Family names this task's registry offers, for the plan schema's enum."""
    return sorted({str(entry["family"]) for entry in registry(task)})


def plan_schema(task: str | None = None) -> dict[str, Any]:
    """The plan schema for this task, so neither enum can name the other task's models.

    Per call rather than per module, for the reason
    :func:`automl_agent.nodes.model_selection.selection_schema` is: the enum is what stops a
    cross-task proposal before it costs an attempt, and one module-level constant can only
    hold one task's menu.
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
            # The ordered form of the same subject. A plan gives one or the other: the executor
            # ignores ``preprocessing`` when a spec arrives, because two descriptions of one
            # pipeline leave nothing able to say which of them ran. Items are unconstrained
            # objects for the reason ``preprocessing`` is — the executor validates every key
            # against the thing it is about to build and reports the result in
            # ``applied_pipeline``, and a second copy of those lists here is the copy that drifts.
            "pipeline": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            # A boolean and not a number, because the planner has never seen this model's
            # probabilities and a hand-picked cut would be a guess about their distribution.
            # The executor accepts an explicit float in a hand-written config; what the plan
            # gets is the lever, not the value. Optional: absent is the default 0.5 rule, which
            # is what every plan written before this key existed means.
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


# Order the deterministic planner walks when told the family itself is wrong.
FAMILY_ROTATION = ("gbdt", "bagging", "linear", "neural", "tree", "kernel", "instance")
RESOURCE_FAILURES = {"oom", "too_slow"}
# Settings the executor honours on top of the registry's own params, so that a retune of only
# these still reads as a new plan. They are here rather than on the menu on purpose: the
# registry is rendered into the prompt, and widening it widens what the LLM is invited to tune.
#
# The last two are the early-stopping pair. Neither is advertised, and both change what the fit
# sees once the stop is on — ``validation_fraction`` decides how much of train is held back
# (``capabilities`` quotes the number to the plan when it announces the ``'auto'`` boundary) and
# ``n_iter_no_change`` is the patience that decides when it bites. Without them a plan whose one
# change is "hold back less" fingerprints identically to the attempt it is trying to differ
# from, and the novelty guard sends it to a family swap it did not ask for.
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
    """Return ``{"plan": ..., "iteration": n + 1}``."""
    iteration = int(state.get("iteration", 0) or 0) + 1
    critic_verdict = dict(state.get("critic") or {})
    task = task_of_state(state)

    variables = {
        "dataset_card": state.get("dataset_card") or {},
        # Constraints on the plan that the aggregates above cannot express —
        # automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        "goal": state.get("goal") or {},
        # The same dict in prose, because two of its keys are obligations and neither reads as one
        # in JSON. Handed ``"exceeds_ranking_ceiling": true`` and ``"passable_margin"`` as bare
        # keys, a run can spend its whole budget on tuning and family swaps against a bar no
        # threshold over that ranking could reach; the
        # sentence explaining exactly that already existed in ``goal.describe`` and went
        # only to the console and the report.
        "goal_note": describe_goal(dict(state.get("goal") or {})),
        "iteration": str(iteration),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "history": _history_digest(state) or "(no previous attempt — this is the first plan)",
        "best": state.get("best") or "(no successful attempt yet)",
        "critic": critic_verdict or "(no critic verdict yet — this is the first attempt)",
        "available_models": available_models(task),
        "executor_capabilities": describe_capabilities(task),
        # The card's row count multiplied through the split, and what that product decides —
        # automl_agent.capabilities.describe_row_budget. Only the planning prompt gets this:
        # from iteration 2 on the executor has reported the real counts in the result's
        # ``internal_validation``, and the Critic reads the result. Iteration 1 has no result,
        # which is where every attempt in bench/ chose its early-stopping setting.
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
    # Which of the two wrote this plan. Three states and not two, because "we asked and did not
    # get a usable plan" and "we never asked" are different facts about the same run: the first
    # is a call that was paid for and produced nothing, the second is the rule-based arm running
    # as designed. Before this, both looked identical in the artifacts — the console said so and
    # nothing else did, so an operator reading ``history.json`` afterwards could not tell which
    # iterations the proposer actually decided. That matters most for a run whose proposer is a
    # local model: a silent fallback rate is the difference between measuring that model and
    # measuring the rules while believing otherwise.
    plan["source"] = "llm" if proposed else ("fallback" if config.use_llm else "rules")
    plan = enforce_novelty(plan, state, task, config.seed)
    plan["iteration"] = iteration
    if plan.get("unsupported_claims"):
        # Flagged, not rejected: rejecting costs the same iteration, and the detector is
        # a substring match. Saying it out loud is what keeps the report from quoting an
        # unexecuted step as part of the winning configuration.
        print(
            f"  [planning] 계획이 실행기에 없는 기능을 전제한 것으로 보입니다 — "
            f"{explain_claims(list(plan['unsupported_claims']))}. 그 부분은 실행되지 않습니다"
        )
    return {"plan": plan, "iteration": iteration}


def validate_plan(plan: dict[str, Any] | None, task: str | None = None) -> dict[str, Any] | None:
    """Keep only runnable candidates; reject the plan outright if none survive.

    "Runnable" is task-relative, so a plan whose whole candidate list belongs to the other
    task is rejected here and the deterministic planner takes over — which is the right
    outcome: a list of classifiers for a continuous target is not a plan with a bad first
    choice, it is a plan built against the wrong menu.
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
        # The two levers that are *keys* rather than prose, carried for the same reason
        # ``preprocessing`` is: this function rebuilds the plan from a fixed list, so a key it
        # does not name is dropped before any node reads it.
        #
        # Both were missing here for one real run, and the run is what found it. The schema
        # advertised them, the plan used them correctly — it asked for `tune_threshold` and for a
        # spec naming the three high-missing columns, with the mechanism argued in its own
        # rationale — and the executor received `decision: None, pipeline: null`. Silently: no
        # ``dropped_hyperparams``, no log line, and nothing in ``history`` either, because the
        # plan recorded there is the one this function returns. That is the shape
        # ``docs/REGISTRY-GAP.md`` is about, one layer up: the reasoning side asked for something
        # the executor can do, and the harness threw it away on the way.
        "pipeline": plan.get("pipeline") if isinstance(plan.get("pipeline"), list) else [],
        "tune_threshold": plan.get("tune_threshold") is True,
        "changes_from_last": changes,
        "rationale": rationale,
        # The prose is where an unavailable capability hides: as a hyperparameter key it
        # would already show up in the result's ``dropped_hyperparams``.
        "unsupported_claims": unsupported_claims(strategy, changes, rationale),
        "source": "llm",
    }


# --------------------------------------------------------------------------- #
# Deterministic planner (--dry-run and LLM-unavailable fallback)
# --------------------------------------------------------------------------- #


def fallback_plan(
    state: AutoMLState,
    verdict: dict[str, Any],
    iteration: int,
    task: str | None = None,
) -> dict[str, Any]:
    """Rule-based replanning that still honours the Critic's direction.

    Mirrors the prompt's rules: a resource failure always shrinks the footprint, and
    every plan changes something relative to the last attempt.
    """
    failure_type = str(verdict.get("failure_type") or "")
    changes = dict(verdict.get("concrete_changes") or {})
    previous_hyperparams = dict(state.get("hyperparams") or {})
    tried_models = [str(item.get("model") or "") for item in (state.get("history") or [])]
    # The family to stay in when the diagnosis is about *tuning* rather than about the family.
    # Hoisted because four branches below wanted the same lookup, and a registry scan repeated
    # per branch is one more place for the default to drift.
    same_family = _family_of(state.get("model"), task) or "gbdt"

    if not state.get("history"):
        family = "gbdt"
        hyperparams: dict[str, Any] = {"max_iter": 150, "learning_rate": 0.1}
        strategy = "baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다"
        change_note = "첫 시도이므로 비교 대상 없음"
    elif failure_type in RESOURCE_FAILURES:
        # Shrink, never grow, in response to a resource failure.
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
            # There is no imbalance lever on a continuous target, so prescribing
            # ``class_weight`` here would put a key in the plan that the executor drops —
            # the exact defect ``capabilities`` exists to prevent, arriving from our own
            # fallback rather than from the LLM. What is left that the executor really does
            # is a family less moved by a heavy tail, so the plan says that instead.
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

    # The Critic's concrete numbers win over the rule's defaults.
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
        # Run the same detector rather than hardcoding an empty list: the rule-based
        # prose is written here, but it quotes the Critic's ``direction``, which is not.
        "unsupported_claims": unsupported_claims(strategy, change_note, rationale),
        "source": "heuristic",
    }


def enforce_novelty(
    plan: dict[str, Any], state: AutoMLState, task: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Guarantee the plan is not a byte-for-byte repeat of an earlier attempt.

    A repeated attempt burns an iteration for no information, so a top candidate and hyperparameters
    matching an earlier try rotate to an untried model.

    **The fingerprint has to see everything that reaches the executor.** When it saw only the
    advertised hyperparameters, a Critic prescribing a preprocessing step alone produced a plan that
    read as a byte-for-byte repeat — the pipeline, which decided the whole difference, was not in the
    signature at all, and this guard would have rewritten it into a family swap.
    Rationale: ``docs/rationale.md``.
    """
    history = list(state.get("history") or [])
    if not history:
        return plan
    # What the executor is known to apply to each model, from its own report rather than from
    # the advertised menu. ``registry`` publishes a curated list — widening it would also widen
    # what the prompt invites the LLM to tune — and keys off it read as a repeat when they are
    # the only change. ``EXECUTOR_PARAMS`` covers the ones worth naming ahead of time;
    # ``applied_hyperparams`` covers the rest, because it is the executor saying what it took.
    # ``min_child_weight`` is the example that only this half catches. (``early_stopping`` used
    # to be one too — it is on the menu now, for the reason ``docs/REGISTRY-GAP.md`` records.)
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
                    # The plan's own key, like the pipeline beside it: requested against
                    # requested. An attempt that asked for a cut and had it declined (too few
                    # held-back rows) still reads as having asked, which is the right side to
                    # fail on — the alternative rewrites a legitimate retry into a family swap.
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
        # Every model has been tried: perturb the hyperparameters instead.
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
    """Fingerprint everything that reaches the executor, and nothing it would ignore.

    Changing ``learning_rate`` on ``logreg`` looks like a new plan and produces a byte-identical run,
    so an unfiltered signature lets the loop burn iterations on no-op variations.

    The params list is task-relative — ``linreg`` consumes *nothing*, so the empty registry entry is
    what makes the loop see that every proposal on it is a no-op. ``extra_keys`` is the other
    direction, read off ``applied_hyperparams`` so it stays a fact about what ran rather than a second
    list to keep in sync.

    **The pipeline is in the fingerprint, compared requested-against-requested** — the only symmetric
    option, since the plan's *applied* pipeline does not exist yet. So two requests the executor
    reduces to the same pipeline read as different, and a plan that drops an already-refused key
    passes as novel and costs an iteration. **That is the direction to fail in**: the other one
    silently rewrites a legitimate single-lever plan into a family swap.
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
        # The declared pipeline, canonicalised the same way the flag block is and for the same
        # reason: it reaches the executor, so a replan that moves only a step's column list is a
        # different attempt and must not be read as a repeat. Rendered through ``json.dumps`` with
        # sorted keys so the fingerprint does not depend on the order the LLM happened to emit the
        # keys of a step in — only on the order of the *steps*, which is the part that runs.
        parts.append("spec=" + json.dumps(spec, sort_keys=True, default=str))
    if tune_threshold is True:
        # Appended only when asked for, so every signature computed before this lever existed is
        # unchanged and a resumed run's history still matches itself. It has to be in here at
        # all for the reason the pipeline does: it is a third axis that reaches the executor, and
        # a replan that moves only the cut would otherwise read as a byte-for-byte repeat and be
        # rewritten into a family swap — the exact defect ``docs/REGISTRY-GAP.md`` closed for
        # ``early_stopping`` and ``scale_pos_weight``.
        parts.append("cut=tuned")
    return ",".join(parts)


def _canonical(value: Any) -> str:
    """A value's spelling made independent of the JSON round trip it did or did not make.

    ``sanitise_hyperparams`` gives a weight map integer keys; the same map read back from the
    executor's ``train_result.json`` has string ones. Comparing those two spellings has let a genuine
    repeat through the novelty guard, and a guard whose verdict depends on which side of a subprocess
    boundary a dict came from is not a guard.
    """
    if isinstance(value, Mapping):
        return json.dumps({str(key): value[key] for key in value}, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _requested_preprocessing(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """The pipeline an earlier attempt *asked* for, falling back to the one that ran.

    The plan is the right side of the comparison — see :func:`_signature` — and it is present
    on every entry :func:`automl_agent.state.build_attempt` makes. The fallback is for records
    that predate the plan being kept, where the applied pipeline is the only thing on file.
    """
    plan = attempt.get("plan")
    if isinstance(plan, Mapping) and isinstance(plan.get("preprocessing"), Mapping):
        return dict(plan["preprocessing"])
    applied = (attempt.get("result") or {}).get("applied_preprocessing")
    return dict(applied) if isinstance(applied, Mapping) else {}


def _grouped_by(state: AutoMLState) -> str | None:
    """The column the split kept whole, or ``None`` on the row-level path.

    Read from the card's published protocol rather than from the private ``data`` block,
    which :func:`automl_agent.privacy.public_card` has already stripped by the time this node
    runs. A card built with ``--no-baseline`` has no protocol block, and ``None`` is then the
    honest answer for the only thing the caller does with it — deciding whether to call the
    row counts approximate.
    """
    protocol = ((state.get("dataset_card") or {}).get("baseline") or {}).get("protocol")
    column = protocol.get("grouped_by") if isinstance(protocol, Mapping) else None
    return str(column) if column else None


def _entry_of(model: Any, task: str | None = None) -> dict[str, Any] | None:
    """This task's registry entry for ``model``, or ``None`` when its menu has no such id.

    One lookup for the three callers that each wanted a different field off the same entry. Each of
    them used to walk ``registry(task)`` itself, which is three places for "what counts as a match"
    to drift apart.
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
