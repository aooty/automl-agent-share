"""지표 registry: 실행이 겨냥할 수 있는 모든 지표의 선언 한 벌.

**목록은 여기 하나뿐이고 나머지는 여기서 읽는다.** 세 군데가 각자 목록을 갖고 있었고 어긋남이
조용히 실패했다 (``docs/rationale.md``).

**일부러 의존성이 없다** — 이름과 성질뿐이고, sklearn도 없고 패키지의 나머지에서 오는 import도 없다.
오케스트레이터는 pandas나 sklearn을 결코 import해선 안 되고, 고정된 스크립트들은 import되는 대신
*파일*로 돌므로, 양쪽이 나눠 쓰는 것은 어느 방향에서든 아무것도 딸려 오지 않고 import될 수 있어야
한다.

지표마다 자기 ``task``를 선언한다. 실행의 과제는 호출자의 성질이 아니라 *타깃 열*의 성질이기
때문이다. ``direction``이 여기 있는 것도 같은 이유다: ``f1``을 최소화하는 것은 선호가 아니라
실수다 (``docs/rationale.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

# 타깃 열이 무엇인가. 프로파일러가 어느 쪽인지 정하고(타깃의 dtype과 서로 다른 값의 수에서) 카드에
# 기록한다. 모든 소비자는 거기서 읽는다.
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASKS: tuple[str, ...] = (TASK_CLASSIFICATION, TASK_REGRESSION)

MAXIMIZE = "maximize"
MINIMIZE = "minimize"

# 카드 자신의 ``task`` 라벨. 위의 둘보다 일부러 촘촘하다: 계획 프롬프트는
# "binary_classification"을 읽고 양성 클래스가 하나라는 것을 아는데, "classification"만으로는 그것을
# 말하지 않는다. 이것이 카드의 어휘를 지표의 어휘로 옮겨서, 어느 쪽도 자기 해상도를 포기하지 않고
# 둘을 비교할 수 있게 한다.
CARD_TASKS: dict[str, str] = {
    "binary_classification": TASK_CLASSIFICATION,
    "multiclass_classification": TASK_CLASSIFICATION,
    "regression": TASK_REGRESSION,
}


@dataclass(frozen=True)
class MetricSpec:
    """모든 소비자가 지표 하나에 대해 알아야 하는 것.

    ``needs_proba``와 ``binary_only``는 실행이 청했는데도 지표가 결과에서 빠질 수 있는 정당한 이유
    둘이다. 여기 선언하는 것이 학습기와 프로파일러가 각자 정하고 갈라지는 대신 같은 이유로 같은
    지표를 건너뛰게 한다.
    """

    needs_proba: bool
    binary_only: bool
    # 남은 여유를 재고 댈 최선값 1이 있어서 "남은 것의 얼마를 메운다"가 뜻을 갖는다. [0, 1] 위의
    # 분류 지표들과 ``r2``에 대해 True다 — ``r2``는 *아래로* 열려 있지만 1에서 멈추고, 여유가 세는
    # 것이 그 끝이다.
    bounded: bool
    # 호출자가 숫자를 대지 않았을 때 ``fixed`` 모드가 쓰는 바, 그리고 유도할 기준선을 카드가 싣지
    # 않았을 때 ``auto``의 마지막 수단. 타깃 자신의 단위를 쓰는 지표에는 ``None``이고, 그 지표들은
    # ``--threshold``나 측정된 기준선을 요구한다 (``docs/rationale.md``).
    fallback: float | None
    task: str = TASK_CLASSIFICATION
    # 어느 쪽이 더 좋은가. 지표의 성질이고 결코 호출자의 성질이 아니다.
    direction: str = MAXIMIZE


METRICS: dict[str, MetricSpec] = {
    "f1": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "accuracy": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    # accuracy보다 낮은 기본값: 불균형한 타깃에서는 이것이 더 어려운 숫자이고, 거기서 0.85 바는
    # accuracy 0.85가 청하는 것보다 훨씬 많이 청한다.
    "balanced_accuracy": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.80),
    "precision": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "recall": MetricSpec(needs_proba=False, binary_only=False, bounded=True, fallback=0.85),
    "roc_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.90),
    # precision-recall 곡선 아래 면적. 우연 수준이 0.5가 아니라 양성 비율이므로, 0.90 같은 기본값은
    # 양성이 드문 타깃에서 닿을 수 없다.
    "pr_auc": MetricSpec(needs_proba=True, binary_only=True, bounded=True, fallback=0.70),
    # -- 회귀 ---------------------------------------------------------------- #
    # 들고 다닐 수 있는 바를 가진 유일한 회귀 지표: 1이 완벽한 적합이고 0이 평균을 예측해서 받는
    # 점수인데, 모든 데이터셋에서 모든 단위에서 그렇다. 그래서 ``auto``의 여유 마진과 ``fixed``의
    # 기본값이 여기서 분류 지표에서와 같은 것을 뜻한다.
    "r2": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=True,
        fallback=0.80,
        task=TASK_REGRESSION,
    ),
    # 타깃의 단위라서 기본 바가 없다 — 위의 ``fallback``을 보라. ``auto`` 모드는 같은 단위인 측정된
    # 기준선에서 하나를 유도한다.
    "mae": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=False,
        fallback=None,
        task=TASK_REGRESSION,
        direction=MINIMIZE,
    ),
    "rmse": MetricSpec(
        needs_proba=False,
        binary_only=False,
        bounded=False,
        fallback=None,
        task=TASK_REGRESSION,
        direction=MINIMIZE,
    ),
}

# 같은 숫자에 대한 sklearn의 이름. 지표 이름을 읽는 모든 곳에서 받아들여지고, 결과에서 정규 키와
# 나란히 나가서 어느 이름으로 쓰인 카드나 보고서도 계속 읽힌다. *같은* 양에 대한 *같은* 부호의
# 이름만 여기 속한다: ``neg_mean_absolute_error``는 일부러 없는데, 부호가 뒤집힌 별칭은 한 숫자를
# 읽는 사람 절반에게 ``direction``을 거짓으로 만들기 때문이다.
ALIASES: dict[str, str] = {
    "average_precision": "pr_auc",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
}

# ``--metric``이 받아들이는 것, 그리고 설정이 검증되는 대상 집합.
GOAL_METRICS: tuple[str, ...] = tuple(METRICS)

# 호출자가 *다른* 과제에 속한 지표를 댔을 때 그 과제를 무엇으로 채점할지. :func:`substitute_metric`이
# 쓰고, 답이 거절이 아니라 대체인 이유는 거기 있다.
#
# 회귀에 ``rmse``가 아니라 ``r2``인 것은 일부러다: 들고 다닐 수 있는 바(``fallback=0.80``)를 가진
# 유일한 회귀 지표이므로, 대체된 실행에는 판단받을 목표가 있다 (``docs/rationale.md``).
DEFAULT_METRICS: dict[str, str] = {
    TASK_CLASSIFICATION: "f1",
    TASK_REGRESSION: "r2",
}


def substitute_metric(task: str, metric: str) -> str | None:
    """``metric`` 대신 ``task``를 채점할 지표, 또는 그대로 두라는 ``None``.

    ``None``은 고칠 것이 없거나 알 수 있는 것이 없다는 뜻이다: 지표가 이미 과제에 맞거나, 과제 라벨을
    이 빌드가 모르거나, 이름이 registry 지표가 아니다(``RunConfig``가 그것들을 거절한다). 대체에는
    두 사실이 모두 알려지고 서로 어긋나는 것이 필요하다.

    거절이 아니라 대체인 이유, 그리고 그 어긋남에 닿는 두 경로는 ``docs/rationale.md``. 대체가
    안전한 것은 **결코 조용하지 않기 때문뿐이다**: 호출자가 그것을 말한다
    (:mod:`automl_agent.nodes.profiling`). 아무도 청하지 않은 지표로 판단된 실행은 바뀐 것이 화면에
    있을 때만 정직하다.
    """
    wanted = task_of(metric)
    if wanted is None or task not in DEFAULT_METRICS or wanted == task:
        return None
    return DEFAULT_METRICS[task]


def canonical(name: str) -> str:
    """별칭을 registry 키로. 모르는 이름은 그대로 지나간다.

    raise하는 대신 지나가게 하는 것이 유도 경로를 너그럽게 지킨다: 이 harness가 계산할 수 없는 지표를
    댄 손으로 쓴 카드도 fallback 임계값을 받고, 거절은 ``RunConfig``에서 한 번 일어난다.
    """
    return ALIASES.get(name, name)


def spec(name: str) -> MetricSpec | None:
    """``name``의 spec (별칭을 안다), 이 harness에 그런 지표가 없으면 None."""
    return METRICS.get(canonical(name))


def metrics_for(task: str) -> tuple[str, ...]:
    """``task``에 속한 registry 키들, registry 순서대로.

    채점기가 순회하는 것. 회귀 시도를 분류 목록으로 채점하면 아무것도 안 나오는 데서 끝나지 않는다 —
    연속값에 대한 ``f1_score``는 raise하고, 그 결과는 애초에 해당되지 않은 지표가 아니라 이 예측이
    "지원할 수 없는" 지표로 읽힌다.
    """
    return tuple(name for name, item in METRICS.items() if item.task == task)


def card_task(card: dict[str, object]) -> str | None:
    """데이터셋 카드가 :data:`TASKS` 중 무엇에 대한 것인지, 말하지 않으면 ``None``.

    ``None``은 필드가 있기 전에 쓰인 카드와 이 빌드가 라벨을 모르는 카드를 함께 덮고, 호출자는 둘을
    같게 다룬다: 주장이 없으니 검사도 없다. 짐작된 과제는 없는 과제보다 나쁘다 — 아무도 쓰지 않은
    라벨을 근거로 지표를 거절하게 된다.
    """
    label = card.get("task")
    return CARD_TASKS.get(str(label)) if label else None


def task_of(name: str) -> str | None:
    """``name``이 채점하는 과제 (별칭을 안다), 모르는 지표면 ``None``."""
    found = spec(name)
    return None if found is None else found.task


def direction_of(name: str, default: str = MAXIMIZE) -> str:
    """``name``에 대해 어느 쪽이 더 좋은가. 모르는 지표는 ``default``를 지킨다.

    묻는 대신 읽는다. 왜 물어선 안 되는지는 ``docs/rationale.md``.
    """
    found = spec(name)
    return default if found is None else found.direction
