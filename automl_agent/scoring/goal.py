"""목표 임계값, 두 모드로: ``auto``(데이터셋 상대)와 ``fixed``.

**모드가 둘인 이유, 그리고 margin이 점수가 아니라 남은 여유에 붙는 이유는 ``docs/rationale.md``.**

``auto``(기본)는 ``scripts/profile``이 같은 데이터·같은 holdout 분할에서 측정한 기준선에 상대적으로
바를 놓는다::

    target = baseline + (1 - baseline) * margin        (bounded, maximize)
    target = baseline * (1 - margin)                   (minimize)

두 식 모두 :data:`MIN_LIFT`에 걸린다: **상수 예측기가 이미 넘는 목표는 목표가 아니다.**

``fixed``는 호출자의 숫자이거나 지표별 기본값이다. 데이터의 무엇도 그것을 움직이지 못한다.

**어느 모드도 LLM에게 목표가 얼마여야 하는지 묻지 않는다** — 자기 바를 세우는 모델은 그것을 낮춰
성공을 선언할 수 있다.

**도달 가능성은 밝히고, 강제하지 않는다.** 유도된 바는 기준선의 *랭킹*이 어느 임계값에서도 허락하지
않는 곳에 앉을 수 있고, 실제 실행 둘이 그것을 알아내는 데 반복 예산 전부를 썼다. ``auto``는 바를 그
상한과 대조해 ``reference``에 적고 :func:`describe`가 찍는다. 바를 낮추지도(결과에 맞춰 움직이는 바는
목표가 아니다), 실행을 거절하지도 않는다 — 상한은 *기준선*의 랭킹에 속하고 더 나은 계열은 그것을
넘는다. 그래서 경고는 구조적으로 낮은 쪽으로 틀리고, 그것이 말하는 "랭킹 자체가 올라야 한다"는 그런
실행이 실제로 하는 일이다. :mod:`automl_agent.scoring.ranking`을 보라.

**도달 가능성 검사 하나는 바를 움직인다, 위로.** ``baseline``은 0.5 컷에서 측정되므로 그 컷이 나쁘게
대하는 지표는 유도에 깎인 출발점을 건네고 바가 그 손해를 물려받는다 — 기준선 자신의 랭킹이 *어떤*
컷에서 이미 넘는 바는 아무것도 구분하지 않는다. ``auto``는 거기서 바닥을 치고, 그것은
:data:`MIN_LIFT`의 예외를 한 걸음 더 읽은 것이다. 이것이 필요한 항등식을 가진 것은
``balanced_accuracy``뿐이다. :func:`_raise_to_ranking_floor`를 보라.

**분해 가능성도 밝히고, 바를 움직이지 않는다.** 자기 95% 구간이 margin이 청하는 것보다 넓은 기준선에서
유도한 바는 슬라이스가 분해할 수 없는 차이를 청한다. ``inside_baseline_ci``로 적힌다.
:mod:`automl_agent.scoring.intervals`를 보라.
"""

from __future__ import annotations

from typing import Any

from .intervals import CI_LEVEL, contains
from .metrics import ALIASES, METRICS, MINIMIZE, card_task, direction_of, spec, substitute_metric
from .ranking import card_ceiling, passable_margin, required_ks

# 두 모드. ``auto``는 데이터셋을 재고, ``fixed``는 호출자의 숫자를 받는다.
MODE_AUTO = "auto"
MODE_FIXED = "fixed"
GOAL_MODES = (MODE_AUTO, MODE_FIXED)
DEFAULT_MODE = MODE_AUTO

# 에이전트가 닫으라고 청받는 남은 여유의 비율. ``auto``에만.
DEFAULT_MARGIN = 0.25

# [0, 1]에 사는 지표들, 곧 "남은 여유"가 뜻이 있는 곳. 다시 적는 대신 registry에서 유도한다: 이
# 저장소가 계산할 수 없는 지표는 바를 받아서는 안 된다. sklearn의 이름으로 쓴 카드가 같은 임계값을
# 유도하도록 alias도 포함한다.
BOUNDED_METRICS = frozenset(
    {name for name, spec in METRICS.items() if spec.bounded}
    | {alias for alias, target in ALIASES.items() if METRICS[target].bounded}
)

# 완벽한 점수는 결코 청하지 않는다: 실제 데이터에서 그것은 목표가 도달 불가라는 뜻이고, 실행은 항상
# 반복 예산에서 끝나므로 진짜 정체가 가려진다.
CEILING = 0.99
# 목표는 chance 수준보다 최소 이만큼 위에 앉아야 한다. 아니면 다수 클래스 예측기가 그것을 만족한다.
#
# chance가 지표 자신의 단위로 측정되므로 두 가지로 읽는다. bounded maximize 지표에서는 절대 거리
# (``chance + 0.02``), 오차 지표에서는 chance 오차의 *비율*(``chance * 0.98``)이다 — 절대 0.02는
# 일 단위 타깃과 달러 단위 타깃에서 다른 것을 뜻하게 되고, 타깃의 단위로 된 바가 결코 해서는 안 되는
# 것이 그것이다. 같은 상수, 같은 의도, 지표가 갖고 온 단위로 적은 것.
MIN_LIFT = 0.02

# ``fixed`` 모드의 기본 바, 그리고 카드에 기준선이 아예 없을 때(손으로 쓴 카드, 또는
# ``--no-baseline``으로 돈 프로파일링) ``auto``의 마지막 수단. 위와 같은 유도이므로 두 상수가 서로
# 다른 지표를 덮는 일은 있을 수 없다.
#
# 타깃 자신의 단위로 나오는 지표는 항목이 없다. 이식 가능한 기본값이 없기 때문이다 — registry가
# ``mae``/``rmse``에 ``fallback=None``을 선언하고 이 dict는 그것을 그냥 건너뛴다. 여기 없다는 것은
# "이 지표는 측정된 기준선이나 명시적 --threshold가 필요하다"이고, 그것은 "모르는 지표"와 다른 상태다.
# :func:`default_bar`를 보라.
_BARS = {name: item.fallback for name, item in METRICS.items() if item.fallback is not None}
FALLBACK_THRESHOLDS: dict[str, float] = {
    **_BARS,
    **{alias: _BARS[target] for alias, target in ALIASES.items() if target in _BARS},
}
FALLBACK_DEFAULT = 0.85


def default_bar(metric: str) -> float | None:
    """아무것도 측정되지 않았을 때 ``metric``에 쓸 바, 또는 그런 것이 없으면 ``None``.

    세 경우이고, 뒤 두 개의 차이가 중요하다:

    * 이식 가능한 기본값이 있는 registry 지표 → 그 숫자;
    * 타깃의 단위로 된 registry 지표(``mae``, ``rmse``) → ``None``. 여기 어떤 숫자든 모델이 아니라
      그 열의 척도에 대한 주장이 되기 때문이다;
    * 이 harness가 모르는 이름 → :data:`FALLBACK_DEFAULT`. 유도 경로를 손으로 쓴 카드에 관대하게
      둔다. 모르는 이름을 거절하는 곳은 ``RunConfig``다.
    """
    found = spec(metric)
    if found is None:
        return FALLBACK_DEFAULT
    return found.fallback


def _reference(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """카드의 baseline 블록에서 ``metric``에 대한 ``(baseline, chance)``를 꺼낸다."""
    baseline_block = card.get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None

    def score(section: str) -> float | None:
        scores = baseline_block.get(section)
        raw = scores.get(metric) if isinstance(scores, dict) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        return float(raw)

    return score("scores"), score("chance")


def _target(baseline: float, metric: str, direction: str, margin: float) -> float:
    if direction == MINIMIZE:
        # 오차 지표는 아래로 0에 막혀 있으므로 margin이 점수에서 빠진다.
        return max(0.0, baseline * (1.0 - margin))
    if metric in BOUNDED_METRICS:
        return baseline + (1.0 - baseline) * margin
    # 상한 없는 maximize(throughput 같은 지표): 척도 없이 남는 것은 비례 상승뿐이다.
    return baseline * (1.0 + margin)


def derive_threshold(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[float | None, dict[str, Any]]:
    """이 카드와 지표에 대한 ``(threshold, reference)``.

    ``reference``는 그 숫자에 이른 방식을 적는다. 그래서 보고서와 Critic 프롬프트가 맨 상수를
    내놓는 대신 "0.865 = logreg의 0.821 더하기 남은 여유의 4분의 1"이라고 말할 수 있다.

    ``direction``은 주지 않으면 지표 registry에서 읽는다. 호출자가 그것을 줄 이유는 없다 — 어느 쪽이
    더 좋은지는 지표에 속한다.

    임계값이 ``None``인 경우는 정확히 하나다 — 유도할 측정된 기준선이 없는, 타깃 자신의 단위로 된
    지표(``mae``, ``rmse``). 거기에 정직한 숫자는 없으므로 호출자는 추측 대신 아무것도 받지 않고,
    :mod:`automl_agent.nodes.profiling`이 실행을 거절한다.
    """
    direction = direction or direction_of(metric)
    baseline, chance = _reference(card or {}, metric)
    if baseline is None:
        bar = default_bar(metric)
        source = "fallback" if bar is not None else "unset"
        return bar, {"source": source, "metric": metric, "margin": margin}

    threshold = _target(baseline, metric, direction, margin)
    reference: dict[str, Any] = {
        "source": "baseline",
        "metric": metric,
        "baseline": round(baseline, 4),
        "margin": margin,
    }

    if direction == MINIMIZE:
        # 아래 chance 규칙의 거울이고 그만큼 필요하다: Ridge 기준선은 평균 예측보다 *나쁠* 수 있고
        # (음수 r2에는 chance보다 높은 mae가 따라온다) 그러면 ``baseline * (1 - margin)``은 평균
        # 예측기가 이미 넘는 바를 낸다. 조이는 것 — 바는 내려가고 결코 올라가지 않는다 — 이 두 방향이
        # 공유하는 하나뿐인 불변식을 지킨다: 보정은 목표를 더 어렵게만 만든다.
        if chance is not None and threshold > chance * (1.0 - MIN_LIFT):
            threshold = chance * (1.0 - MIN_LIFT)
            reference["lowered_to_clear_chance"] = True
        if threshold <= 0.0:
            # ``--margin 1.0``, 또는 이미 완벽한 0을 받는 기준선. ``exceeds_ceiling``과 같은 조건으로
            # 밀지 않고 밝힌다: 바가 정확한 예측을 청하고 있고, 어떤 시도도 그것을 보고하지 않으므로
            # 실행은 정체처럼 보이는 모습으로 반복 예산에서 끝날 것이다.
            reference["demands_zero_error"] = True
    elif metric in BOUNDED_METRICS:
        # 클램프 먼저, 그다음 chance — 반대 순서는 안 된다. 클램프를 나중에 하면 상한이 바를 chance
        # *아래로* 되당길 수 있었고(다수 클래스 99.5% 코호트가 ``accuracy >= 0.99``를 받았다), 그때
        # 실행은 아무것도 배우지 않은 채 성공을 보고했다.
        threshold = min(threshold, CEILING)
        if chance is not None and threshold < chance + MIN_LIFT:
            # 아니면 chance 이하의 기준선이 다수 클래스 예측기가 이미 넘는 목표를 낸다.
            threshold = chance + MIN_LIFT  # chance 우선 — CEILING으로 되돌리지 않는다
            reference["raised_to_clear_chance"] = True
        threshold = _raise_to_ranking_floor(reference, card or {}, metric, threshold, baseline)
        if threshold > CEILING:
            # 조용히 낮추는 대신 정직하게 둔다: 이 지표는 이 데이터셋에서 실제 모델을 다수 클래스와
            # 구분할 수 없고, ``describe``가 그렇게 말한다.
            reference["exceeds_ceiling"] = True
        # 공개 전에 반올림한다, 후가 아니다: 실행이 판정받는 바는 반올림된 쪽이고 ``required_ks``는
        # 항등식을 역으로 푼다. 반올림 안 한 중간값에서 유도한 KS는 화면의 바로 되돌아가지 않으므로
        # 독자가 검산할 수 없는 숫자가 된다.
        threshold = round(threshold, 4)
        _note_ranking_ceiling(reference, card or {}, metric, threshold, baseline)
    _note_baseline_interval(reference, card or {}, metric, round(threshold, 4))
    if chance is not None:
        reference["chance"] = round(chance, 4)
    return round(threshold, 4), reference


def _raise_to_ranking_floor(
    reference: dict[str, Any],
    card: dict[str, Any],
    metric: str,
    threshold: float,
    baseline: float,
) -> float:
    """바가 기준선 랭킹 자신의 최적 컷 아래에 앉을 때, 거기까지 바를 올린다.

    :func:`_note_ranking_ceiling`의 거울이다. 그쪽은 한 방향이었다 — 기준선 랭킹이 *어느* 컷에서도
    닿지 못하는 바는 밝혔고, 그 랭킹이 *어떤* 컷에서 이미 넘는 바에는 아무 말도 하지 않았다. 같은
    숫자의 같은 결함이다: ``baseline``은 0.5 컷에서 측정되므로(``profile.py``도 모든 시도처럼
    ``predict()``를 부른다) 그 컷이 기준선에 물리는 값이 그대로 거기서 유도한 바에서 빠진다.

    bench 카드 넷에서 그 값이 margin이 붙이는 것을 넘어서는 지표는 ``balanced_accuracy``뿐이다. 네
    카드의 표와 그 이유는 ``docs/goal.md``. 요약하면, 두 오류율을 같은 무게로 평균하는 지표라서 KS
    항등식이 여기서만 성립하고 불균형에서 0.5 컷이 가장 비싼 것도 여기다. 그래서
    ``card_ceiling``이 다른 곳에서 ``None``을 돌려주는 것은 빈틈이 아니라 맞는 답이다.

    **이것은 바를 움직이고, 이 모듈의 다른 무엇도 그러지 않는다.** :data:`MIN_LIFT`가 이미 만드는
    같은 예외, 같은 방향이다: 상수 예측기가 이미 넘는 목표가 목표가 아닌 것처럼, 기준선 자신의 랭킹을
    다시 자른 것이 이미 넘는 목표도 목표가 아니다. 둘 다 바를 더 어렵게만 만든다. 대가는 ``--margin``이
    바닥 아래에서 바를 못 움직이게 되는 것이므로, 바닥에 걸린 margin을 밝혀 :func:`describe`가 찍는다:
    자기 플래그를 무시하는 바는 그렇다고 말해야 한다.

    :data:`CEILING`에서 클램프하고 그것을 넘게 두지 않는다 — KS가 0.98을 넘는 랭킹이라면 바를 클램프
    밖으로 밀고 ``exceeds_ceiling`` 공개를 받아 갈 텐데, 그 문구는 ``chance``를 탓하므로 여기서는
    거짓이 된다.
    """
    _ks, ceiling = card_ceiling(card, metric)
    if ceiling is None or threshold >= ceiling:
        return threshold
    reference["raised_to_ranking_floor"] = True
    reference["ranking_floor"] = ceiling
    # 이 카드의 margin이 실제로 낸 바. 공개 안에서 독자가 나머지로 재계산할 수 없는 유일한 숫자이고,
    # 바가 움직여진 실행은 무엇에서 움직여졌는지 말할 수 있어야 하므로 남긴다.
    reference["margin_bar"] = round(threshold, 4)
    floored = passable_margin(baseline, ceiling)
    if floored is not None:
        reference["floored_margin"] = floored
    return min(ceiling, CEILING)


def _note_ranking_ceiling(
    reference: dict[str, Any],
    card: dict[str, Any],
    metric: str,
    threshold: float,
    baseline: float,
) -> None:
    """바가 기준선 랭킹의 어느 컷도 닿을 수 없는 곳에 앉는지 적는다.

    보정이 아니라 *공개*다. 바는 그것에 맞춰 낮춰지지 않는다. 결과에 맞춰 움직이는 바는 목표이기를
    그치고, 상한은 더 나은 계열이 자유롭게 넘을 수 있는 기준선의 랭킹에 속하기 때문이다
    (:mod:`automl_agent.scoring.ranking`에 수가 있다). 양방향으로 조언일 뿐이다: KS가 없는 카드는
    그냥 아무 note도 받지 않으므로, 이 검사가 자기가 잘못 판단한 실행을 막는 일은 있을 수 없다.

    발사할 때 바는 *두 번* 적힌다. 한 번은 호출자가 청한 지표에, 한 번은 그 부족분을 소유한 랭킹 축에
    (기준선의 ``ks``에 대고 ``required_ks``). 하나의 바, 두 개의 요구, 각각 크기를 갖는다 — 그것이
    "랭킹이 올라야 한다"와 계열 교체에 대고 비교할 수 있는 숫자 사이의 차이 전부다. 그 비교가 없을 때
    무엇을 물었는지는 :func:`automl_agent.scoring.ranking.required_ks`를 보라.
    """
    ks, ceiling = card_ceiling(card, metric)
    if ks is None or ceiling is None:
        return
    reference["ks"] = round(ks, 4)
    reference["ranking_ceiling"] = ceiling
    if threshold <= ceiling:
        return
    reference["exceeds_ranking_ceiling"] = True
    demanded = required_ks(threshold)
    if demanded is not None:
        reference["required_ks"] = demanded
        reference["ks_shortfall"] = round(demanded - round(ks, 4), 4)
    margin = passable_margin(baseline, ceiling)
    if margin is not None:
        reference["passable_margin"] = margin


def _note_baseline_interval(
    reference: dict[str, Any], card: dict[str, Any], metric: str, threshold: float
) -> None:
    """바가 기준선 점수 자신의 신뢰구간 안에 앉는지 적는다.

    위의 랭킹 상한처럼 *공개*이고, 이유도 같다: margin은 호출자가 고르는 것이고 측정치에 맞춰 움직이는
    바는 목표가 아니다.

    발사할 때의 뜻: 기준선에서 바까지의 거리가 기준선 측정 자체의 폭보다 작으므로, 이 검증 슬라이스는
    그만한 차이를 분해할 수 없다. *아닌* 것: 바를 넘긴 시도가 아무것도 배우지 않았다는 뜻은 아니다 —
    시도는 기준선과 같은 행에서 채점되므로 두 오차가 함께 움직이고 짝지은 비교는 두 구간 어느 것보다
    좁다. 정직한 독법은 "이 margin은 이 슬라이스의 잡음 바닥에 있다"이고, 그것을 정하는 숫자는 떼어 둔
    test 점수다 (:mod:`automl_agent.nodes.holdout`).

    양방향으로 조언일 뿐이다: 구간이 있기 전에 쓰인 카드나 검증 슬라이스가 구간에 너무 작았던 카드는
    그냥 아무 note도 받지 않는다.
    """
    block = card.get("baseline")
    interval = (block or {}).get("ci") if isinstance(block, dict) else None
    scores = (interval or {}).get("scores") if isinstance(interval, dict) else None
    bound = scores.get(metric) if isinstance(scores, dict) else None
    if not isinstance(bound, dict):
        return
    low, high = bound.get("low"), bound.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return
    reference["baseline_ci"] = [round(float(low), 4), round(float(high), 4)]
    # ``unit``이 독법에 중요하다: 그룹 단위 구간이 넓고 정직한 쪽이며, 군집된 데이터에서 바가 구간 안에
    # 들어갈 수 있는 이유가 그 폭이다.
    if isinstance(interval, dict) and interval.get("unit"):
        reference["baseline_ci_unit"] = str(interval["unit"])
    if contains((float(low), float(high)), threshold):
        reference["inside_baseline_ci"] = True


def derive_goal(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    mode: str = DEFAULT_MODE,
    threshold: float | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """두 모드 중 하나에 대한 ``goal`` 채널을 세운다.

    ``threshold``를 대는 것이 ``fixed``를 고르는 것*이므로*, 모드와 충돌하는 대신 모드를 함의한다 —
    숫자를 건넨 호출자가 그것이 조용히 무시되는 일을 겪지 않는다. CLI는
    ``--goal-mode auto --threshold``를 아예 거절하므로, 그 추론은 의도가 모호하지 않은 곳에만 걸린다.

    ``auto``에서 이것은 ``--data`` 경로에서 두 번 불린다: 한 번은 ``initial_state``에서(아직 카드가
    없으므로 지표별 기본값), 한 번은 카드가 생긴 뒤 ``profiling`` 노드에서. 두 번째가 첫 번째를 덮어쓰고,
    그래서 이 채널이 누적형이 아니라 맨 값이다.

    ``direction``은 호출자가 덮어쓰지 않으면 지표 registry에서 온다. 그래서 ``threshold``는 항상
    "그 바"를 뜻하고 "누군가 청한 방향으로 읽은 바"를 뜻하지 않는다. ``threshold``는 ``None``으로
    돌아올 수 있다 — :func:`derive_threshold`를 보라.
    """
    if threshold is not None:
        mode = MODE_FIXED
    direction = direction or direction_of(metric)
    goal: dict[str, Any] = {"metric": metric, "direction": direction, "mode": mode}

    if mode == MODE_FIXED:
        goal["threshold"] = float(threshold) if threshold is not None else default_bar(metric)
        if threshold is not None:
            goal["source"] = "fixed"
        elif goal["threshold"] is not None:
            goal["source"] = "fixed_default"
        else:
            # 숫자 없는 ``fixed``, 단위 때문에 모든 숫자가 임의가 되는 지표에 대해. 메시지가 측정된
            # 대안을 댈 수 있도록 여기가 아니라 profiling 노드가 거절한다.
            goal["source"] = "unset"
        return goal

    value, reference = derive_threshold(card, metric=metric, direction=direction, margin=margin)
    goal["threshold"] = value
    goal["source"] = {"baseline": "derived", "unset": "unset"}.get(
        str(reference.get("source")), "fallback"
    )
    goal["reference"] = reference
    return goal


def resolve_goal(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    mode: str = DEFAULT_MODE,
    threshold: float | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[dict[str, Any], str | None]:
    """:func:`derive_goal`, 지표가 이 카드의 타깃을 채점할 수 없을 때 그것을 바꿔서.

    ``(goal, note)``를 돌려준다. ``note``는 교체를 설명하는 한글 한 줄이고, 평범한 경로에서는
    ``None``이다. ``goal`` 채널의 모든 구성이 여기를 지나는 이유는 교체가 정확히 한 곳에서 일어나야
    하기 때문이다 — ``goal["metric"]``은 ``training``부터 ``holdout``까지 모든 노드가 읽고, 하류에서
    대체하면 한 실행 안에 지표가 둘 남는다.

    셋이 함께 움직이고, 잊기 쉬운 둘이 이것이 호출부마다 두 줄이 아니라 함수인 이유다:

    * ``direction``을 새 지표에서 다시 읽는다. 옛것을 들고 가는 것은 그것이 고치는 어긋남만큼 큰
      버그다 — ``f1``에 ``rmse``의 ``minimize``는 실행을 뒤집어 *최악*의 시도를 ``best``로 지키고
      바 아래에서 ``goal_met``을 발사한다 (``RunConfig.__post_init__``이 같은 함정을 손으로 막는다).
    * 명시적 ``threshold``는 버린다: 옛 지표의 단위로 된 값이고, ``rmse``의 ``--threshold 3.2``를
      ``f1`` 바로 다시 쓰면 호출자의 숫자를 입은 도달 불가 목표가 된다. *모드*는 지키므로 ``fixed``
      실행은 새 지표의 기본값을 받고 ``source``는 ``fixed_default``로 읽힌다.

    ``substituted_from``은 goal에 적히므로 교체가 체크포인트와 보고서까지 살아남는다.
    :func:`describe`가 goal이 찍히는 모든 곳에서 그것을 찍는다.
    """
    task = card_task(card or {})
    swap = substitute_metric(task, metric) if task else None
    if swap is None:
        return derive_goal(
            card, metric=metric, direction=direction, mode=mode, threshold=threshold, margin=margin
        ), None

    note = (
        f"이 데이터의 task는 {(card or {}).get('task')}인데 목표 지표 {metric}는 다른 task의 "
        f"지표입니다 — 이 정답 열에서는 계산되지 않으므로, 그대로 두면 모든 시도가 '목표 미달'로 "
        f"기록되고 반복 예산만 소모됩니다. {swap}로 바꿔 실행합니다"
    )
    if threshold is not None:
        # 조용히 흡수하지 않고 말한다: 호출자가 숫자를 줬고 이 실행은 그것으로 판정되지 않는다.
        # 그 숫자는 옛 지표의 단위이므로 들고 갈 것이 없다.
        note += (
            f". --threshold {threshold}는 {metric}의 단위로 준 값이라 {swap}의 바로 쓸 수 없어 "
            f"버립니다 — {swap} 기준으로 다시 지정하려면 --metric {swap} --threshold <값>"
        )
    goal = derive_goal(card, metric=swap, direction=None, mode=mode, threshold=None, margin=margin)
    goal["substituted_from"] = metric
    return goal, note


def goal_threshold(goal: dict[str, Any], default: float) -> float:
    """goal의 바를 float으로, 없으면 ``default``로.

    ``goal.get("threshold", default)``는 이 일을 하지 않는다: 바를 유도할 수 없었던 지표에서 키는
    *있고* 값이 ``None``이므로 기본값이 결코 걸리지 않고 호출자는 ``float(None)``에서 ``TypeError``를
    받는다. 루프 안의 모든 노드는 ``profiling``이 그 경우를 거절한 뒤에 돌므로, 이것은 코드 경로가
    아니라 가드다.
    """
    raw = goal.get("threshold")
    return default if raw is None else float(raw)


def missing_bar_message(metric: str) -> str:
    """이 실행이 시작할 수 없는 이유, 그리고 바를 주는 두 방법.

    설정되지 않은 임계값을 거절하는 모든 호출자가 공유하므로, 어느 경로가 그것을 찾았든(시작 시
    ``--dataset-card``, 프로파일링 뒤 ``--data``) 운영자는 같은 두 선택지를 읽는다.
    """
    return (
        f"{metric}는 정답 열의 단위로 나오는 지표라서 이식 가능한 기본 목표값이 없습니다 — "
        f"0.85 같은 숫자를 넣으면 모델이 아니라 그 열의 단위에 대한 바가 됩니다.\n"
        f"  - 요구 수준을 알고 있다면: --goal-mode fixed --threshold <{metric} 값>\n"
        "  - 데이터에서 도출하려면: 기준선이 측정된 카드로 auto 모드를 쓰십시오 "
        "(`profile`을 --no-baseline 없이 실행)"
    )


def describe(goal: dict[str, Any]) -> str:
    """한글 한 줄 설명: 어느 모드, 어떤 바, 그리고 그것이 어디서 왔는지."""
    metric = str(goal.get("metric", "metric"))
    threshold = goal.get("threshold")
    direction = "이하" if str(goal.get("direction")) == "minimize" else "이상"
    mode = str(goal.get("mode") or DEFAULT_MODE)
    head = f"{mode} 모드 — {metric} {threshold} {direction}"
    if goal.get("substituted_from"):
        # 시작 시 한 번이 아니라 바와 함께 찍는다: 이 실행은 아무도 청하지 않은 지표로 판정되고,
        # goal을 보여 주는 모든 곳 — 콘솔 헤더, Critic의 프롬프트, report.md — 이 그것을 함께 날라야
        # 한다. 콘솔 출력 한 줄에만 나오고 다른 데 없는 교체는 보고서의 독자가 볼 수 없는 교체다.
        head += f" (요청한 지표 {goal['substituted_from']}는 이 task의 지표가 아니라 대체됨)"
    source = str(goal.get("source") or "")
    reference = goal.get("reference") or {}

    if source == "unset" or threshold is None:
        # 찍을 숫자가 없으므로 "None"을 찍는 대신 무엇이 없는지 말한다. profiling 노드의 검사를
        # 무언가 건너뛰지 않았다면 실행은 여기까지 오지 않는다.
        return f"{mode} 모드 — {metric} 목표값 미정 (이 지표는 기본 바가 없습니다)"
    if source == "fixed":
        return f"{head} (직접 지정)"
    if source == "fixed_default":
        return f"{head} (지표 기본값)"
    if source == "derived" and isinstance(reference, dict):
        baseline = reference.get("baseline")
        margin = reference.get("margin", DEFAULT_MARGIN)
        if str(goal.get("direction")) == MINIMIZE:
            # 여기서 "남은 여유"는 거짓이 된다: margin이 오차 자체에서 빠지고, 독자는 눈앞의 숫자를
            # 낸 산술이 어느 것인지 봐야 한다.
            detail = f"기준선 {baseline}에서 {float(margin):.0%} 감소"
        else:
            detail = f"기준선 {baseline} + 남은 여유의 {float(margin):.0%}"
        if reference.get("raised_to_clear_chance"):
            detail += ", chance 수준을 넘도록 상향"
        if reference.get("raised_to_ranking_floor"):
            detail += ", 기준선 랭킹의 최적 컷을 넘도록 상향"
        if reference.get("lowered_to_clear_chance"):
            detail += ", chance보다 낮도록 하향"
        chance = reference.get("chance")
        if chance is not None:
            detail += f" (chance {chance})"
        line = f"{head} ← {detail}"
        if reference.get("exceeds_ceiling"):
            line += (
                f" (도달 불가) — chance가 {chance}이라 목표가 상한 {CEILING}을 넘습니다. "
                "--metric balanced_accuracy 또는 --metric pr_auc처럼 다수 클래스에 "
                "덜 휘둘리는 지표로 바꾸십시오"
            )
        if reference.get("demands_zero_error"):
            # 위 줄의 minimize 짝이고, 자기 문구가 필요하다: 여기서 문제는 지표가 아니라(mae는 재기에
            # 맞는 것이다) margin이다.
            line += (
                f" (도달 불가) — 오차 0을 요구하는 바입니다. --margin을 1보다 작게 두거나, "
                f"실제 허용 오차를 알고 있다면 --goal-mode fixed --threshold <{metric} 값>으로 "
                "직접 지정하십시오"
            )
        if reference.get("raised_to_ranking_floor"):
            # 아래 ``exceeds_ranking_ceiling`` 줄의 짝이고, 독자가 재계산할 수 없는 두 숫자를 날라야
            # 한다: margin이 낸 바, 그리고 넘긴 플래그가 왜 무의미해졌는지. ``--margin``을 조용히
            # 무시하는 것이 이 문구가 막으려고 있는 실패 양태다.
            line += (
                f" — margin이 낸 바는 {reference.get('margin_bar')}였지만, 같은 기준선의"
                f" 랭킹을 최적 컷에서 자르면 {reference.get('ranking_floor')}"
                f" (KS {reference.get('ks')})입니다: 기준선을 다시 자른 것이 이미 넘는 점수는"
                " 목표가 아니므로 그 값으로 올렸습니다"
            )
            floored = reference.get("floored_margin")
            if floored is not None:
                line += (
                    f". 이 카드에서 --margin {floored} 이하는 모두 같은 바를 냅니다 —"
                    " 더 어려운 목표를 원하면 그보다 크게 주십시오"
                )
        if reference.get("exceeds_ranking_ceiling"):
            # 위와는 다른 종류의 도달 불가: 지표 자신의 한계가 아니라 이 랭킹의 한계다. 그렇게 이름
            # 붙일 값이 있다 — 고칠 것은 더 나은 랭킹이고, 예전 문구는 호출자를 지표를 바꾸는 쪽으로
            # 보냈을 텐데 그것이 여기서 도움이 안 되는 하나뿐인 일이다.
            # "이 랭킹으로는"이 지지대다: 상한은 기준선의 것이고 더 나은 계열은 그것을 넘는다 —
            # 자기 카드의 상한 위에 앉은 바를 넘긴 실행이 있었다. 단서 없는 "도달할 수 없다"는 그
            # 실행에 대한 판정으로 읽힐 것이다.
            passable = reference.get("passable_margin")
            line += (
                f" — 이 바는 기준선 랭킹의 상한 {reference.get('ranking_ceiling')}를 넘습니다"
                f" (KS {reference.get('ks')}). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니"
                " 랭킹 자체를 올려야 합니다 — 모델 family나 특성"
            )
            # 레버가 놓인 축에서의 크기. 그것 없이 "랭킹을 올려야 한다"는 척도 없는 방향이고, 계열
            # 교체가 그럴듯한 답으로 읽힌다 — 실행들이 반복 하나를 써서 그렇지 않다는 것을 알아낸 것이
            # 그것이다.
            demanded = reference.get("required_ks")
            if demanded is not None:
                line += (
                    f". 크기로 말하면 이 바가 요구하는 KS가 {demanded}이고 기준선은"
                    f" {reference.get('ks')}이므로 랭킹이 {reference.get('ks_shortfall')}만큼"
                    " 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오"
                )
            if passable is not None:
                line += f". 지금 기준선에서 이 상한 안에 드는 margin은 {passable} 이하입니다"
        if reference.get("inside_baseline_ci"):
            # 위 둘과 독립인, 바가 나쁜 바일 수 있는 세 번째 방식: 도달 불가가 아니라 출발점과 구분이
            # 안 되는 것. 문구는 겸손하게 둔다 — 시도는 기준선과 같은 행에서 채점되므로 이것은
            # 슬라이스의 분해력에 대한 진술이고 바를 넘긴 시도에 대한 판정이 아니다.
            interval = reference.get("baseline_ci") or []
            unit = "그룹" if reference.get("baseline_ci_unit") == "group" else "행"
            line += (
                f" — 이 바는 기준선 자체의 {int(CI_LEVEL * 100)}% CI"
                f"({interval[0]}~{interval[1]}, {unit} 단위) 안에 있습니다: "
                "val 슬라이스가 이만한 차이를 분해하지 못한다는 뜻이므로, 바를 넘겼는지보다 "
                "홀드아웃 점수를 보십시오. 더 큰 --margin이 더 정직한 목표입니다"
            )
        return line
    return f"{head} (카드에 기준선이 없어 지표 기본값 사용)"
