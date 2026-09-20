"""Model selection: 판단은 LLM의 것, 확정은 코드의 것.

LLM은 이 시스템이 실제로 돌릴 수 있는 모델 registry에서 고른다. 그러면 이 모듈이 식별자를 검증하고
하이퍼파라미터를 clamp한다. 돌릴 수 없는 선택은 학습 시점에 발견되는 대신 여기서 고쳐진다.

registry가 task마다 하나씩 둘인 이유는 모델 이름이 *정답 열에 대해서만* 돌릴 수 있는 것이기 때문이다.
연속 열에 대한 ``logreg``는 나쁜 선택이 아니라 선택이 아니고, executor가 대놓고 거절한다
(``scripts/train.py::build_estimator``). 그것을 여기서 고치는 것은 이 모듈이 잘못 쓴 id에 대해 이미 하는
그 일을 한 층 앞에서 하는 것이다: LLM에게 보이는 메뉴, 그 답을 검증하는 스키마, 폴백 모두가 해당 task의
registry에서 세워지므로 task를 건너뛴 이름은 애초에 제안되지 않아야 한다. executor의 거절은 그 뒤의
backstop으로 남는다.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from typing import Any

from ..config import RunConfig
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION, card_task
from ..state import AutoMLState, state_int

# --------------------------------------------------------------------------- #
# Registry: executor가 세울 줄 아는 모델의 전부
# --------------------------------------------------------------------------- #

MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "logreg",
        "family": "linear",
        "cost": 1,
        "params": ["C", "max_iter", "class_weight"],
        "notes": "fast, low-capacity baseline; features are scaled automatically",
    },
    {
        "id": "decision_tree",
        "family": "tree",
        "cost": 1,
        "params": ["max_depth", "min_samples_leaf", "class_weight"],
        "notes": "interpretable, overfits easily",
    },
    {
        "id": "knn",
        "family": "instance",
        "cost": 2,
        "params": ["n_neighbors", "weights"],
        "notes": "no training cost, slow at prediction, sensitive to dimensionality",
    },
    {
        "id": "hist_gbdt",
        "family": "gbdt",
        "cost": 3,
        "params": [
            "max_iter",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "max_leaf_nodes",
            "l2_regularization",
            "class_weight",
            "early_stopping",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of "
        "max_iter. `early_stopping` defaults to 'auto', which is *on* above 10k rows, so "
        "saying nothing about it does not mean fitting on every train row",
    },
    {
        "id": "random_forest",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf", "class_weight"],
        "notes": "robust, memory-hungry with many trees",
    },
    {
        "id": "extra_trees",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf", "class_weight"],
        "notes": "more randomised than random_forest, often better with noisy labels",
    },
    {
        "id": "gradient_boosting",
        "family": "gbdt",
        "cost": 5,
        "params": ["n_estimators", "learning_rate", "max_depth", "subsample"],
        "notes": "sequential and slow; prefer hist_gbdt unless small data",
    },
    {
        "id": "xgboost",
        "family": "gbdt",
        "cost": 4,
        "params": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_lambda",
            "subsample",
            "scale_pos_weight",
            "early_stopping_rounds",
        ],
        "notes": "requires the xgboost package. `early_stopping_rounds` needs nothing else "
        "from the plan — the executor holds its own stopping slice back and supplies the eval "
        "set; `eval_set` and `callbacks` are the two keys it refuses. `scale_pos_weight` is "
        "**binary only** — xgboost ignores it on a multiclass objective, so above two classes "
        "the executor drops it and the attempt records that in `dropped_hyperparams`",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": [
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "early_stopping",
        ],
        "notes": "highest capacity available here, and the most likely to hit the memory "
        "budget. Its `early_stopping` is a boolean only — the `'auto'` spelling belongs to "
        "hist_gbdt, and the executor drops it here rather than letting `fit` raise on it",
    },
    {
        "id": "svc",
        "family": "kernel",
        "cost": 5,
        "params": ["C", "kernel", "gamma", "class_weight"],
        "notes": "quadratic in rows; unusable above roughly 20k rows",
    },
)

# 같은 목록의 회귀 쪽. 계열이 양쪽에 다 있는 곳에서는 일부러 *같은 id*를 쓴다 — ``hist_gbdt``가
# 여기서는 HistGradientBoostingRegressor를, 저기서는 HistGradientBoostingClassifier를 세운다. 어휘가
# 하나면 계획이 어느 카드에 대해서도 같게 읽히고 ``capabilities``가 펴낼 목록도 하나다. 이름이 달라지는
# 것은 sklearn의 추정기가 정말로 다른 곳뿐이다: ``logreg``→``ridge``, ``svc``→``svr``.
#
# ``class_weight``가 모든 항목에 없는 이유는 가중할 클래스가 없기 때문이다. 여기서 그저 쓸모없는 것이
# 아니다 — 불균형 이야기 전체가 세워진 하나뿐인 지렛대이므로, 펴낸 params에 남겨 두면 executor가
# 떨어뜨릴 처방을 부르는 일이 된다.
#
# 비용이 같아도 ``ridge``가 ``linreg``보다 앞인 것은 일부러다: "이 계열에서 가장 싼 것" 조회는 모두 비용
# 으로 정렬하고 동점은 이 순서로 갈리므로, ``linear``라는 계열은 정규화된 쪽으로 해소된다. 그것은
# 프로파일러 자신의 baseline 모델이기도 하고, 둘 중 처방이 실제로 작용할 수 있는 쪽이다 — "정규화를 더
# 세게"라는 판정이 ``linreg``에 내리면 그것이 지목하는 모든 손잡이가 떨어진다.
REGRESSION_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "ridge",
        "family": "linear",
        "cost": 1,
        "params": ["alpha", "max_iter"],
        "notes": "l2-regularised linear fit, and the profiler's own baseline model; "
        "features are scaled automatically. `alpha` is the regularisation strength, so it "
        "runs the *opposite* way from logreg's `C`",
    },
    {
        "id": "linreg",
        "family": "linear",
        "cost": 1,
        # 일부러 비워 둔다: OLS는 닫힌 해가 있고 조율할 것이 없다. 그것이 무시하는 손잡이를 적으면
        # 루프가 바이트까지 같은 실행에 반복 하나를 쓰게 된다 — ``planning._signature``가 이 목록을
        # 읽는 이유가 정확히 그것을 막으려는 것이다.
        "params": [],
        "notes": "unregularised least squares; the reference point, not a tuning target",
    },
    {
        "id": "elasticnet",
        "family": "linear",
        "cost": 2,
        "params": ["alpha", "l1_ratio", "max_iter"],
        "notes": "l1 plus l2; drives coefficients to zero, so it is the nearest thing here "
        "to feature selection, which the executor will not do separately",
    },
    {
        "id": "decision_tree",
        "family": "tree",
        "cost": 1,
        "params": ["max_depth", "min_samples_leaf"],
        "notes": "interpretable, overfits easily; predicts a constant per leaf",
    },
    {
        "id": "knn",
        "family": "instance",
        "cost": 2,
        "params": ["n_neighbors", "weights"],
        "notes": "no training cost, slow at prediction, sensitive to dimensionality",
    },
    {
        "id": "hist_gbdt",
        "family": "gbdt",
        "cost": 3,
        "params": [
            "max_iter",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "max_leaf_nodes",
            "l2_regularization",
            "early_stopping",
        ],
        "notes": "strong default for tabular data; also accepts n_estimators as an alias of "
        "max_iter. `early_stopping` defaults to 'auto', which is *on* above 10k rows, so "
        "saying nothing about it does not mean fitting on every train row",
    },
    {
        "id": "random_forest",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf"],
        "notes": "robust, memory-hungry with many trees",
    },
    {
        "id": "extra_trees",
        "family": "bagging",
        "cost": 4,
        "params": ["n_estimators", "max_depth", "min_samples_leaf"],
        "notes": "more randomised than random_forest, often better with a noisy target",
    },
    {
        "id": "gradient_boosting",
        "family": "gbdt",
        "cost": 5,
        "params": ["n_estimators", "learning_rate", "max_depth", "subsample"],
        "notes": "sequential and slow; prefer hist_gbdt unless small data",
    },
    {
        "id": "xgboost",
        "family": "gbdt",
        "cost": 4,
        "params": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_lambda",
            "subsample",
            "early_stopping_rounds",
        ],
        "notes": "requires the xgboost package. `early_stopping_rounds` needs nothing else "
        "from the plan — the executor holds its own stopping slice back and supplies the eval "
        "set; `eval_set` and `callbacks` are the two keys it refuses",
    },
    {
        "id": "mlp",
        "family": "neural",
        "cost": 5,
        "params": [
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "early_stopping",
        ],
        "notes": "highest capacity available here, and the most likely to hit the memory "
        "budget. Its `early_stopping` is a boolean only — the `'auto'` spelling belongs to "
        "hist_gbdt, and the executor drops it here rather than letting `fit` raise on it",
    },
    {
        "id": "svr",
        "family": "kernel",
        "cost": 5,
        "params": ["C", "kernel", "gamma", "epsilon"],
        "notes": "quadratic in rows; unusable above roughly 20k rows. `epsilon` is a "
        "tolerance in the target's own units, so its useful size depends on the target's scale",
    },
)

# task마다 registry 하나, 그리고 모든 조회는 모듈 수준 이름을 만지는 대신 :func:`registry`를 지난다.
# 그래서 새 task는 grep이 아니라 여기 항목 하나다.
REGISTRIES: dict[str, tuple[dict[str, Any], ...]] = {
    TASK_CLASSIFICATION: MODEL_REGISTRY,
    TASK_REGRESSION: REGRESSION_REGISTRY,
}

# 두 task 모두 ``hist_gbdt``를 가지므로 기본값에 분기가 필요 없다 — id를 같게 둔 이유 중 하나다.
DEFAULT_MODEL = "hist_gbdt"

# LLM이 무엇을 제안하든 거기 적용되는 clamp: 제안은 조언이고 명령이 아니다.
LIMITS: dict[str, tuple[float, float]] = {
    "max_iter": (1, 3000),
    "n_estimators": (1, 2000),
    "learning_rate": (1e-4, 1.0),
    "learning_rate_init": (1e-5, 1.0),
    "max_depth": (1, 64),
    "max_leaf_nodes": (2, 1024),
    "min_samples_leaf": (1, 1000),
    "l2_regularization": (0.0, 100.0),
    "reg_lambda": (0.0, 100.0),
    # 추정기 둘 몫의 범위: MLP의 weight decay는 아래쪽에 살고, ridge와 elasticnet의 정규화 강도는
    # 위쪽을 원한다.
    "alpha": (1e-8, 1000.0),
    # elasticnet의 l1/l2 배합. sklearn은 [0, 1] 밖을 대놓고 거절하므로, clamp되지 않은 제안은 시도
    # 전체를 쓴다.
    "l1_ratio": (0.0, 1.0),
    # SVR의 허용 폭. 부호만 지킨다: 쓸모 있는 크기는 정답 열의 단위에 있고 여기서 그것을 아는 것이
    # 없으며, 그 단위의 상한은 이 저장소가 기본 `mae` 바에 대해 하기를 거절하는 바로 그 짐작이다.
    "epsilon": (0.0, 1e9),
    "C": (1e-4, 1e4),
    # 불균형 지렛대의 xgboost 쪽 절반. 아래 ``class_weight`` 맵과 같은 조건으로 묶는다 — 다른 추정기에
    # 대한 같은 요청이고, 묶이지 않은 것은 모든 지표를 퇴화시킨다. 하한이 0이 아니라 1e-3인 이유는 0이
    # 양성 클래스를 *빼* 버리는데, 그것은 청한 것의 약한 판본이 아니기 때문이다.
    "scale_pos_weight": (1e-3, 1000.0),
    # 0은 "early stopping 없음"이고 ``fit_estimator``가 로그에서 그렇게 말한다. 음수 라운드 수도 같은
    # 뜻이므로 놀라움이 아니라 거기로 떨어진다. 상한은 폭주 가드일 뿐이다: 라운드 수보다 큰 인내는
    # 물지 않는다.
    "early_stopping_rounds": (0, 1000),
    # early stopping이 켜졌을 때 스스로 검증하는 추정기가 *train*의 얼마를 떼어 두는지. sklearn은
    # (0, 1) 안이면 다 받으므로, clamp되지 않은 0.9는 로그가 알린 행의 10분의 1에 적합한다. 상한이
    # 절반인 이유: 그것을 넘으면 멈춤이 적합이 보는 것보다 많은 행에서 결정하는데, 여기 어느 카드도
    # 정당화하지 않는 조율 선택이다.
    "validation_fraction": (0.01, 0.5),
    "batch_size": (1, 8192),
    "n_neighbors": (1, 200),
    "subsample": (0.05, 1.0),
    "train_subsample": (0.01, 1.0),
    "gamma": (1e-6, 100.0),
}

ALLOWED_STRINGS = {"class_weight", "weights", "kernel", "precision", "solver", "penalty"}

# 문자열 형태에서 뜻이 있는 표기가 정확히 하나뿐인 키. 키 목록은 이런 것에 너무 거칠다. 어느
# 추정기가 실제로 그 문자열을 받는지는 그 추정기에 대한 사실이므로 executor가
# 정하고(``scripts/train.py::STRING_EARLY_STOPPING``), 이 맵은 어디서도 뜻이 없는 표기만 막는다.
ALLOWED_STRING_VALUES: dict[str, frozenset[str]] = {"early_stopping": frozenset({"auto"})}

# ``class_weight``는 맵으로도 올 수 있는 하나뿐인 키다.
WEIGHT_MAP_KEYS = {"class_weight"}
# 가중치는 클래스 코드마다 하나이므로 그럴듯한 맵은 작다. 경계는 폭주한 값(1e9는 모든 지표를
# 퇴화시킨다)이 추정기에 닿는 것을 막으려고만 있다.
WEIGHT_RANGE = (1e-3, 1000.0)
MAX_WEIGHT_ENTRIES = 32


def registry(task: str | None = None) -> tuple[dict[str, Any], ...]:
    """``task``의 registry, 기본은 분류.

    ``None`` — task를 선언하지 않은 카드, 또는 이 빌드가 모르는 라벨을 가진 카드 — 은 raise가 아니라
    분류로 읽힌다. :func:`automl_agent.scoring.metrics.card_task`와 같은 조건이다: 이것은 검사가 아니라
    메뉴이고, 자기가 무엇인지 말하지 못할 만큼 오래된 카드도 돌릴 수 있는 메뉴는 받아야 한다.
    """
    return REGISTRIES.get(task or TASK_CLASSIFICATION, MODEL_REGISTRY)


def selection_schema(task: str | None = None) -> dict[str, Any]:
    """이 task의 응답 스키마. 그래서 enum이 다른 task의 모델을 내놓을 수 없다.

    import 시점에 한 번이 아니라 호출마다 세운다: 여기서 가장 센 가드가 그 enum이고 — 그 밖의 이름은
    시도 하나를 쓰기 전에 API 층이 거절한다 — 모듈 수준 스키마 하나는 한 task의 메뉴만 담을 수 있다.
    """
    return {
        "type": "object",
        "properties": {
            "model": {"type": "string", "enum": [entry["id"] for entry in registry(task)]},
            "hyperparams": {"type": "object", "additionalProperties": True},
            "rationale": {"type": "string"},
        },
        "required": ["model", "hyperparams", "rationale"],
        "additionalProperties": False,
    }


def available_models(task: str | None = None) -> list[dict[str, Any]]:
    """``task``의 registry 항목 중 뒤를 받치는 패키지를 지금 import할 수 있는 것."""
    entries = []
    for entry in registry(task):
        if entry["id"] == "xgboost" and importlib.util.find_spec("xgboost") is None:
            continue
        entries.append(entry)
    return entries


def available_ids(task: str | None = None) -> set[str]:
    return {entry["id"] for entry in available_models(task)}


def task_of_state(state: AutoMLState) -> str:
    """이 실행의 카드가 어느 registry를 고르는지. 두 노드를 위해 그것을 읽는 한 곳."""
    return card_task(dict(state.get("dataset_card") or {})) or TASK_CLASSIFICATION


# --------------------------------------------------------------------------- #
# 노드
# --------------------------------------------------------------------------- #


def model_selection(state: AutoMLState, *, config: RunConfig) -> dict:
    """돌릴 수 있는 ``(model, hyperparams)`` 짝 하나로 확정한다."""
    plan = dict(state.get("plan") or {})
    task = task_of_state(state)
    variables = {
        "plan": plan,
        "dataset_card": state.get("dataset_card") or {},
        "available_models": available_models(task),
        "history": _history_digest(state),
        "critic": state.get("critic") or "(no critic verdict yet — this is the first attempt)",
    }

    iteration = state_int(state, "iteration") or None
    choice: dict[str, Any] | None = None
    if not config.use_llm:
        archive_prompt_only(
            config,
            f"model_selection_iter{iteration or 0}",
            render_prompt("model_selection", variables),
        )
    else:
        try:
            choice = LLMClient(config, proposer=True).complete_json(
                "model_selection", variables, selection_schema(task), iteration=iteration
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [model_selection] LLM 호출 실패({exc}) — 계획의 후보 모델로 폴백합니다")

    proposed = bool(choice)
    if not choice:
        choice = fallback_selection(plan, task)

    model = normalise_model(choice.get("model"), plan, task)
    hyperparams = sanitise_hyperparams(
        {**(plan.get("hyperparams") or {}), **(choice.get("hyperparams") or {})}
    )
    # ``plan["source"]``와 같은 세 값, 같은 이유 — ``nodes/planning.py`` 참고.
    source = "llm" if proposed else ("fallback" if config.use_llm else "rules")
    return {"model": model, "hyperparams": hyperparams, "selection_source": source}


def fallback_selection(plan: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    """결정적인 대역: 계획의 첫 번째 돌릴 수 있는 후보를 쓴다."""
    candidates = plan.get("candidate_models") or []
    if isinstance(candidates, str):
        candidates = [candidates]
    ids = available_ids(task)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip().lower() in ids:
            return {
                "model": candidate.strip().lower(),
                "hyperparams": dict(plan.get("hyperparams") or {}),
                "rationale": "deterministic fallback: first runnable candidate from the plan",
            }
    return {
        "model": DEFAULT_MODEL,
        "hyperparams": dict(plan.get("hyperparams") or {}),
        "rationale": "deterministic fallback: no runnable candidate in the plan, using the default",
    }


def normalise_model(proposed: Any, plan: dict[str, Any], task: str | None = None) -> str:
    """돌릴 수 없는 식별자를, 학습이 거기서 실패하게 두는 대신 고친다.

    "돌릴 수 없음"에는 이제 *다른 task에서는 돌릴 수 있음*도 들어간다: ``logreg``는 이 실행이 쓸 수 없는
    실재하는 id이고, 고치는 방식은 오타가 받는 것과 같다 — 계획에 계열이 있으면 그 계열, 없으면 기본값.
    그 이름이 어느 계열이었는지는 교체를 넘어 살아남지 않고, 그래야 한다: ``linear``는 여기서 ridge이고
    저기서 logreg다.
    """
    ids = available_ids(task)
    if isinstance(proposed, str) and proposed.strip().lower() in ids:
        return proposed.strip().lower()

    # 계획의 계열을 먼저, 그다음 기본값으로 떨어진다.
    family = str(plan.get("model_family") or "").strip().lower()
    if family:
        for entry in sorted(available_models(task), key=lambda item: int(item["cost"])):
            if entry["family"] == family:
                return str(entry["id"])
    return DEFAULT_MODEL


def _weight_map(value: Mapping[Any, Any]) -> dict[int, float] | None:
    """``{클래스 코드: 가중치}`` 맵, 제안이 그것이 아니면 ``None``.

    전부 아니면 전무: 반쪽 가중치 맵은 실제로 한 요청과 다른 요청이고, 어차피 sklearn은 클래스마다 정확히
    하나의 가중치를 원한다. 계획이 JSON으로 돌아오므로 코드는 문자열로 올 수 있고, config를 쓸 때 다시
    문자열로 나간다 — 이 고침의 나머지 절반은 ``scripts/train.py::weight_map``.
    """
    if not value or len(value) > MAX_WEIGHT_ENTRIES:
        return None
    low, high = WEIGHT_RANGE
    clean: dict[int, float] = {}
    for raw_code, raw_weight in value.items():
        if isinstance(raw_code, bool) or isinstance(raw_weight, bool):
            return None
        if not isinstance(raw_weight, (int, float)):
            return None
        try:
            code = int(str(raw_code).strip())
        except (TypeError, ValueError):
            return None
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0 or code < 0 or code in clean:
            return None
        clean[code] = min(max(weight, low), high)
    return clean


def _string_allowed(key: str, value: str) -> bool:
    """``value``가 이 키가 실어 나를 수 있는 문자열인지.

    목록이 둘이고 순서대로 본다. 서로 다른 질문에 답하기 때문이다: :data:`ALLOWED_STRING_VALUES`의 키는
    열거된 선택 집합을 갖고 그 밖은 떨어지며, :data:`ALLOWED_STRINGS`의 키는 아무 문자열이나 받는다. 첫
    목록의 키는 두 번째에 대고 *묻지 않는다* — 그것이 선택 집합을 힌트가 아니라 whitelist로 만든다.
    """
    if key in ALLOWED_STRING_VALUES:
        return value in ALLOWED_STRING_VALUES[key]
    return key in ALLOWED_STRINGS


def _clamped_number(key: str, value: int | float) -> int | float:
    """``value``를 이 키의 :data:`LIMITS` 안으로. 아직 ``int``인 것은 ``int``로 남긴다."""
    low, high = LIMITS.get(key, (-1e12, 1e12))
    clamped = min(max(float(value), low), high)
    if isinstance(value, int) and float(clamped).is_integer():
        return int(clamped)
    return clamped


def sanitise_hyperparams(raw: Any) -> dict[str, Any]:
    """executor가 쓸 수 있는 값만 제정신인 범위로 clamp해 남기고, 나머지는 떨어뜨린다.

    값 *타입*마다 ``elif`` 하나, 각 갈래는 대입 하나 — 중첩된 검사가 필요했던 두 갈래는 그것을
    넘긴다(:func:`_string_allowed`, :func:`_weight_map`). 그래서 이 루프가 결정 나무가 아니라 실제 그것인
    whitelist로 읽힌다.
    """
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, (int, float)):
            cleaned[key] = _clamped_number(key, value)
        elif isinstance(value, str) and _string_allowed(key, value):
            cleaned[key] = value
        elif isinstance(value, dict) and key in WEIGHT_MAP_KEYS:
            mapping = _weight_map(value)
            if mapping is not None:
                cleaned[key] = mapping
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            cleaned[key] = value  # e.g. hidden_layer_sizes
        elif value is None and key in ALLOWED_STRINGS:
            cleaned[key] = None
    return cleaned


def _history_digest(state: AutoMLState) -> list[dict[str, Any]]:
    """프롬프트용 압축 history: 반복을 피할 만큼 충분하고, 읽을 만큼 작다."""
    return [digest_attempt(attempt) for attempt in state.get("history") or []]


def digest_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """프롬프트가 보는 대로의 시도 하나.

    ``report``도 자기 마지막 시도를 이것으로 압축한다. 전에는 이 모양의 두 번째 사본을 갖고 있었고, 사본이
    어긋났다: 마지막 반복의 ``dropped_hyperparams``는 Critic이 어쩌다 인용했을 때만 보고서에 닿았다.
    """
    result = attempt.get("result") or {}
    metrics = result.get("metrics") or {}
    plan = attempt.get("plan") or {}
    return {
        "iteration": attempt.get("iteration"),
        "model": attempt.get("model"),
        # 이미 적용된 집합이다(``build_attempt``가 ``effective_hyperparams``를 쓴다). 아래 두 키가
        # 청했는데 일어나지 않은 것을 말하고, 그것이 독자가 점수를 그것 덕으로 돌리지 않게 한다.
        "hyperparams": attempt.get("hyperparams"),
        "dropped_hyperparams": result.get("dropped_hyperparams") or [],
        # executor가 세운 파이프라인이고, 계획이 청한 블록이 아니다 — 청한 전략이 낮춰졌을 때마다 둘은
        # 다르다. executor가 그것을 보고하기 전에 기록된 실행에는 없고, 그래서 보고서 프롬프트에
        # 폴백이 있다.
        "preprocessing": result.get("applied_preprocessing") or {},
        # 시도가 spec을 선언했을 때 위 줄의 순서 있는 형태: 렌더된 단계 하나가 한 줄씩, 순서대로,
        # 각자 건드린 열과 함께. 둘 다 있는 이유는 ``applied_preprocessing``이 spec이 더하는 것의 어느
        # 절반도 표현할 수 없기 때문이다 — ``ColumnTransformer``에 대해 ``impute: per_column``을
        # 보고하고, 순서에는 자리가 아예 없다. 플래그를 쓴 시도에는 없고, 그것이 플래그를 썼다고
        # 말하는 것이다.
        "applied_pipeline": result.get("applied_pipeline") or [],
        # ``preprocessing`` 옆인 이유도 같다: 청한 것이 아니라 executor가 한 것이다. 비어 있으면
        # 추정기가 행을 떼어 두지 않은 것이고, executor가 그것을 보고하기 전에 기록된 실행에는 없다.
        "internal_validation": result.get("internal_validation") or {},
        "unsupported_claims": plan.get("unsupported_claims") or [],
        # 누가 이 시도를 결정했는지. run config만이 아니라 digest에 있는 이유는 답이 *반복마다* 다를
        # 수 있기 때문이다: iteration 3에서 스키마 검증에 실패하고 1에서는 아닌 제안자는 한 실행 안에
        # 서로 다른 두 팔을 만들었고, 그 반복들을 견주는 독자에게 그것을 볼 다른 길이 없다.
        "plan_source": plan.get("source") or "",
        "selection_source": attempt.get("selection_source") or "",
        "status": result.get("status"),
        "error_type": result.get("error_type"),
        # ``result["paired"]``는 일부러 없다. 이 digest는 Planner와 보고서를 먹이고, 짝지은 판정은
        # Critic의 ledger가 자기가 추적하는 baseline에 대고 렌더하는 조종 계기다(``nodes/critic.py``).
        # digest로 복사되면 그 baseline 없이 도착하고, 보고서에서는 떼어 둔 점수 옆에 앉아 둘이 같은
        # 질문에 답하는 것처럼 보인다 — 왜 그렇지 않은지는 ``nodes/holdout.py``.
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "train_time_sec": result.get("train_time_sec"),
        "critic": attempt.get("critic"),
    }
