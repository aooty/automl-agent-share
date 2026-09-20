"""랭킹 천장: ``max_cut balanced_accuracy = (1 + KS) / 2``, 항등식.

그 위의 바는 어떤 컷으로도 닿을 수 없으므로 *랭킹*이 나아져야 한다 — 권고이고 거부가 아니다.
import이 가볍다: 측정 함수만, 그것도 지연해서 sklearn을 건드린다.
"""

from __future__ import annotations

import math
from typing import Any

from .intervals import as_number

# 동일 가중 오류율만이므로 ``f1``과 ``accuracy``는 빠진다. ``critic.SYMMETRIC_METRICS``와 합치지 않는다:
# 같은 이름, 다른 주장.
SYMMETRIC_METRICS = frozenset({"balanced_accuracy"})


def ks_statistic(y_true: Any, proba: Any) -> float | None:
    """이진 랭킹의 ``max(TPR - FPR)``, 또는 분할이 그것을 받칠 수 없으면 ``None``.

    ``None``이 덮는 것은 단일 클래스 holdout(ROC 곡선 없음)과 유한하지 않은 최댓값이다 —
    ``roc_curve``는 거기서 raise하는 대신 경고하고, NaN은 카드를 유효하지 않은 JSON으로 만든다.
    """
    if proba is None:
        return None
    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, proba)
    except (ValueError, ImportError, IndexError):
        return None
    if len(fpr) == 0:
        return None
    value = float((tpr - fpr).max())
    return value if math.isfinite(value) else None


def best_cut_ceiling(ks: float | None) -> float | None:
    """``(1 + KS) / 2`` — 이 랭킹의 어떤 컷이든 허용하는 최고 ``balanced_accuracy``."""
    value = as_number(ks)
    if value is None or not 0.0 <= value <= 1.0:
        return None
    return round((1.0 + value) / 2.0, 4)


def required_ks(threshold: float | None) -> float | None:
    """``2 * threshold - 1`` — 같은 항등식을 거꾸로 읽어 바를 손잡이의 축에 올린 것.

    1.0을 넘으면 ``None``: 그런 바는 이미 :data:`automl_agent.scoring.goal.CEILING` 초과로 공개된다.
    """
    number = as_number(threshold)
    if number is None:
        return None
    value = 2.0 * number - 1.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 4)


def passable_margin(baseline: float, ceiling: float) -> float | None:
    """바가 아직 ``ceiling`` 아래에 드는 가장 큰 ``auto`` 여유.

    ``bar = baseline + (1 - baseline) * margin``(:func:`automl_agent.scoring.goal._target`)을
    뒤집으므로, 공개가 호출자에게 아래로 짐작하라고 하는 대신 여유를 직접 이름 부른다.
    """
    headroom = 1.0 - float(baseline)
    if headroom <= 0:
        return None
    margin = (float(ceiling) - float(baseline)) / headroom
    if margin <= 0:
        return None
    # 반올림이 아니라 버림: 올려 반올림한 여유는 바를 다시 천장 위로 올린다.
    return max(0.0, float(int(margin * 1000)) / 1000)


def card_ceiling(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """카드의 baseline 블록에서 읽은 ``metric``의 ``(ks, ceiling)``, 또는 ``(None, None)``.

    :data:`SYMMETRIC_METRICS`만: 그 밖에는 항등식이 없고, 없는 상한을 발명하는 것은 입 다무는 것보다
    나쁘다.
    """
    if metric not in SYMMETRIC_METRICS:
        return None, None
    baseline_block = (card or {}).get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None
    value = as_number(baseline_block.get("ks"))
    if value is None:
        return None, None
    return value, best_cut_ceiling(value)
