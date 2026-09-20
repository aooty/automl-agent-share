"""예측된 확률이 확률로서 얼마나 값이 있는지.

여기 다른 모든 지표는 *랭킹*이나 *라벨*에 대한 것이고, **모델은 완벽하게 순위를 매기면서 체계적으로
과신할 수 있고 그러면 그 전부가 만족한다.** 어느 것도 운영자가 출력 CSV에 묻는 것에 답하지 않는다:
"이 행에 0.7이라고 적혀 있는데, 무엇의 70%가 실제로 일어나는가?"

**재고, 보고하고, 모델은 건드리지 않는다.** 모델을 자기 행에 적용하는 호출자는 자기 컷을 갖고 오므로
그들이 행동의 근거로 삼는 것은 확률이다.

두 수이고, 교환 가능하지 않다:

``brier``
    양성 클래스 확률의 평균제곱오차, ``mean((p - y)**2)``. proper scoring rule이다: 참 확률에서만
    최소화되므로 calibration과 discrimination을 함께 값 매긴다. 어느 표본 크기에서도 불편이고, 그래서
    40행 배치에서도 보고되고 다른 하나는 안 된다.

``calibration_error``
    Expected calibration error — 행을 예측 확률로 구간에 넣고, 각 구간의 평균 예측과 관측된 발생률
    사이 거리를 개수 가중 평균한다. 운영자의 질문에 직접 답하는 쪽이고, *적은 행에서 위로 편향된다*:
    열 구간에 마흔 행이면 한 구간이 네 행이고 그 관측률은 0, 0.25, 0.5, 0.75, 1 중 하나밖에 될 수
    없다. 그래서 :data:`MIN_CALIBRATION_ROWS` 위에서만 보고되고, 그 부재는 조용하지 않고 밝혀진다.

**둘 다 진단이고 어느 것도 registry에 없다.** 그래서 이들에 대고 목표를 세울 수도, 이들로 시도를 고를
수도 없다 — ``brier``를 최적화한 실행은 운영자가 청한 것과 다른 실행이다. ``specificity``와
``balanced_accuracy_cut_headroom``과 같은 조건이다.

측정 함수만 numpy를 건드리므로 오케스트레이터는 숫자를 *서술*하려고 이것을 import할 수 있다.
"""

from __future__ import annotations

import math
from typing import Any

# 이 모듈이 metrics dict에 넣는 두 키. 여기서 이름 붙이는 이유는 그것을 쓰는 스크립트와 되읽는 노드가
# 오타로 어긋날 수 없게 하려고.
BRIER_KEY = "brier"
CALIBRATION_KEY = "calibration_error"

# [0, 1] 위의 등폭 구간. 열은 관례이고, 50행 분할이 대부분의 구간에 한 행만 두지 않고 채울 수 있는
# 최대이기도 하다.
N_BINS = 10

# 이 아래에서는 ``calibration_error``를 보고하지 않는다. 50행은 평균적으로 구간당 다섯 행이고, 관측된
# 발생률이 거친 분수이기를 그치는 지점이 거기다.
MIN_CALIBRATION_ROWS = 50


def brier_score(y_true: Any, proba: Any) -> float | None:
    """양성 클래스 확률에 대한 ``mean((p - y)**2)``. 쓸 수 없으면 ``None``.

    회귀 타깃, ``predict_proba`` 없는 모델, 빈 분할에 예외가 아니라 ``None``인 이유: 모든 호출자가
    없는 값을 "측정되지 않음"으로 다루므로, 퇴화한 분할은 실행이 아니라 공개 한 줄을 문다.
    """
    import numpy as np

    if proba is None:
        return None
    try:
        p = np.asarray(proba, dtype="float64").ravel()
        y = np.asarray(y_true, dtype="float64").ravel()
    except (TypeError, ValueError):
        return None
    if len(p) == 0 or len(p) != len(y):
        return None
    value = float(np.mean((p - y) ** 2))
    return round(value, 6) if math.isfinite(value) else None


def reliability(y_true: Any, proba: Any, bins: int = N_BINS) -> list[dict[str, Any]]:
    """비지 않은 확률 구간마다 한 항목: ``{low, high, rows, predicted, observed}``.

    빈 구간은 0으로 보고하는 대신 버린다 — 아무 행도 떨어지지 않은 구간은 모델이 틀린 구간이 아니다.
    집계뿐이다: 행 수, 평균 예측, 관측된 발생률. 데이터셋 카드가 열에 대해 공개하는 것과 같은 모양의
    사실이다.
    """
    import numpy as np

    if proba is None:
        return []
    try:
        p = np.asarray(proba, dtype="float64").ravel()
        y = np.asarray(y_true, dtype="float64").ravel()
    except (TypeError, ValueError):
        return []
    if len(p) == 0 or len(p) != len(y):
        return []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # 첫 경계를 접어 넣은 ``right=True``. 그래서 0.0은 첫 구간에, 1.0은 마지막 구간에 떨어지고 각자
    # 자기 구간을 갖지 않는다.
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, int(bins) - 1)
    table: list[dict[str, Any]] = []
    for position in range(int(bins)):
        hit = index == position
        rows = int(hit.sum())
        if not rows:
            continue
        table.append(
            {
                "low": round(float(edges[position]), 4),
                "high": round(float(edges[position + 1]), 4),
                "rows": rows,
                "predicted": round(float(p[hit].mean()), 4),
                "observed": round(float(y[hit].mean()), 4),
            }
        )
    return table


def calibration_error(y_true: Any, proba: Any, bins: int = N_BINS) -> float | None:
    """예측과 관측 발생률 사이 거리의 개수 가중 평균. :data:`MIN_CALIBRATION_ROWS` 미만에서는
    ``None`` — 그 아래에서는 구간 나누기를 재는 것이 되므로."""
    table = reliability(y_true, proba, bins)
    total = sum(int(item["rows"]) for item in table)
    if total < MIN_CALIBRATION_ROWS:
        return None
    weighted = sum(
        int(item["rows"]) * abs(float(item["predicted"]) - float(item["observed"]))
        for item in table
    )
    value = weighted / total
    return round(value, 6) if math.isfinite(value) else None


def measure(y_true: Any, proba: Any, bins: int = N_BINS) -> dict[str, float]:
    """두 진단 키, 측정할 수 없었던 쪽은 빼고.

    ``None``이 아니라 빼는 이유: 이들은 모든 값이 숫자라고 가정하는 소비자를 가진 metrics dict에 실려
    간다(``privacy.public_result``는 숫자 값을 지키고 나머지를 버린다). 그래서 때때로 null인 키는
    때때로 실패하는 지표로 읽힐 것이다.
    """
    found: dict[str, float] = {}
    score = brier_score(y_true, proba)
    if score is not None:
        found[BRIER_KEY] = score
    error = calibration_error(y_true, proba, bins)
    if error is not None:
        found[CALIBRATION_KEY] = error
    return found


def describe(metrics: Any, rows: int | None = None) -> str:
    """한글 한 줄, 또는 말할 만큼 측정된 것이 없으면 "".

    일부러 판정이 아니다. Brier 점수가 "좋다"고 할 바는 없다 — 기저율에 달려 있어서 0.09는 양성 2%
    타깃에서 나쁘고 균형 잡힌 타깃에서 훌륭하다 — 그리고 하나를 발명하는 것은 harness가 운영자의
    결정을 대신 하는 일이 된다. 이 줄이 말하는 것은 그 숫자가 *무엇에 대한* 것인지, 그리고 확률이
    라벨이 아니라는 것이다.
    """
    values = dict(metrics or {})
    score = values.get(BRIER_KEY)
    error = values.get(CALIBRATION_KEY)
    if not isinstance(score, (int, float)) and not isinstance(error, (int, float)):
        return ""
    parts: list[str] = []
    if isinstance(score, (int, float)):
        parts.append(f"brier={float(score):.4f}")
    if isinstance(error, (int, float)):
        parts.append(f"확률 오차={float(error):.4f}")
    line = "확률 품질(진단, 목표로 삼을 수 없음): " + ", ".join(parts)
    if isinstance(error, (int, float)):
        line += (
            f" — 예측 확률과 실제 발생률의 차이가 평균 {float(error) * 100:.1f}%p입니다"
            f" ({N_BINS}개 구간, 개수 가중)"
        )
    elif rows is not None and rows < MIN_CALIBRATION_ROWS:
        line += (
            f" — 행이 {rows}개뿐이라 구간별 확률 오차는 측정하지 않았습니다 "
            f"({MIN_CALIBRATION_ROWS}행 이상 필요)"
        )
    return line


def describe_table(table: list[dict[str, Any]]) -> list[str]:
    """reliability 표를 줄로, 운영자의 기계에 머무는 콘솔을 위해.

    구간별 개수이므로 프롬프트가 닿을 수 있는 어디에도 찍히지 않는다 — 호출자는 둘,
    ``scripts/predict.py``의 stdout과 로컬 보고서 파일이다.
    """
    if not table:
        return []
    lines = [f"확률 구간별 실제 발생률 ({N_BINS}구간, 빈 구간 제외):"]
    for item in table:
        lines.append(
            f"  {float(item['low']):.1f}~{float(item['high']):.1f}: "
            f"{int(item['rows'])}행, 예측 평균 {float(item['predicted']):.3f}, "
            f"실제 {float(item['observed']):.3f}"
        )
    return lines
