"""점수의 얼마가 모델이고 얼마가 그것이 측정된 행인지.

여기 모든 수는 한 슬라이스 위의 점추정이고, 모든 비교는 그 둘 사이의 것이다. 그래서 채점된 분할마다
목표 지표에 95% percentile bootstrap 구간을 공개하고, 비교하는 소비자들이 자기 차이가 그 안에 드는지
말한다.

**두 반쪽, 서로 다른 두 결정을 위해.** 짝지은 쪽이 왜 필요한지, 다중성을 교정하지 않는 이유, train
분할에 구간을 안 붙이는 이유는 ``docs/rationale.md``.

*주변*(:func:`bootstrap_interval`) — 한 점수의 폭. **겹치는 구간은 "이 행들로는 이 모델들을 가를 수
없다"는 뜻이고 "모델이 같다"가 아니다.** 그리고 그것은 표집 잡음이며, 다섯 중 최고를 취하는
선택 편향이 **아니다**: 그 오차는 한 방향을 가리키고, 그것을 재는 것은 떼어 둔 test 분할이다
(:mod:`automl_agent.nodes.holdout`).

*짝지은 쪽*(:func:`paired_delta`) — 차이 자체를 재표집한다. 더 좁은 질문이다: "*이 변경*이 이 행들에서
점수를 움직였는가". **교란은 고치지 못한다** — 레버를 둘씩 움직인 두 시도는 잘 분해된 차이를 갖고도
귀속은 없다.

**그룹 분할은 그룹 전체를 재표집한다.** 환자당 5회 방문에 대한 행 단위 bootstrap은 데이터에
``n_patients``가 있을 때 ``5 * n_patients`` 표본에 대한 구간을 보고하고, 그 좁아짐은 군집 크기와 함께
자란다. ``unit``이 어느 쪽을 했는지 말한다.

**짝지은 반쪽은 민감함에 맞춰진 조종 계기이고, test 분할이 승인 시험이다.** 둘을 합치면 모든 조종
결정이 승인만큼 비싸진다.

모듈 수준에 numpy도 sklearn도 import하지 않는다: 오케스트레이터 프로세스는 공개된 구간을 *읽으려고*
이 모듈을 import하고, 측정하는 것은 고정된 스크립트뿐이다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .metrics import MINIMIZE

# 설정 불가. 서로 다른 수준에서 구간을 계산한 두 실행은 비교 가능해 보이면서 그렇지 않은 숫자를 찍고,
# 여기에는 90%나 99% 구간이 더 잘 답하는 질문이 없다.
CI_LEVEL = 0.95

# 각 2.5% 꼬리가 1개가 아니라 ~10개 재표집에서 추정되기에 충분하고, 조건 없이 켜 둘 만큼 싸다:
# 10,000행 분할의 400회 재표집에 지표 하나는 1초에 훨씬 못 미치는데 적합 하나는 그 수십 배다. 더 싸야
# 하는 호출자는 ``resamples=0``을 넘겨 나쁜 구간이 아니라 구간 없음을 받는다.
DEFAULT_RESAMPLES = 400

# 이 아래에는 재표집할 것이 없다. 15행에서 나온 구간은 넓고 옳고 쓸모없다. 환자 4명에서 나온 *그룹*
# 구간은 서로 다른 값 넷에서 뽑으므로 그 백분위는 그 값들 자체다. 재표집 단위 — 행, 또는 그룹 분할일
# 때는 그룹 — 로 센다. 표본 크기가 실제로 그것이기 때문이다.
MIN_UNITS = 20

# 재표집의 5분의 1보다 많은 곳에서 정의되지 않는 지표는 공개할 만한 구간이 없다: 양성 3% 분할의
# 단일 클래스 재표집은 ``roc_auc``를 정의되지 않게 하고, 어쩌다 된 재표집만 조용히 지키는 것은 운이
# 좋았다는 조건 아래의 구간을 보고하는 일이다.
MIN_VALID_SHARE = 0.8

# 재표집될 수 있는 것. 두 return 자리에 인라인으로 적는 대신 이름을 붙인 이유는
# :func:`automl_agent.privacy.public_result`가 아무 문자열이나 전달하는 대신 이 필드를 닫힌 집합에 대고
# 검사하기 때문이고, 쓰는 쪽과 체가 따로 적는 집합은 어긋날 수 있다.
UNIT_ROW = "row"
UNIT_GROUP = "group"
RESAMPLE_UNITS: tuple[str, ...] = (UNIT_ROW, UNIT_GROUP)


@dataclass(frozen=True)
class Interval:
    """percentile bootstrap 구간, 그리고 그것을 얻으려고 무엇을 재표집했는지."""

    low: float
    high: float
    resamples: int
    # :data:`RESAMPLE_UNITS` 중 하나. 지표로 공개되지 않는다 — 규약의 ``grouped_by``가 이미 함의한다 —
    # 그러나 로그에 남긴다. 잘못된 방식으로 측정된 구간은 없는 것이 아니라 좁은 것이고, 기록의 다른
    # 무엇도 그것을 말해 주지 않기 때문이다.
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    @property
    def width(self) -> float:
        return round(self.high - self.low, 6)

    def flatten(self, metric: str) -> dict[str, float]:
        """``{"<metric>_ci_low": ..., "<metric>_ci_high": ...}``.

        한 쌍이 아니라 float 둘인 이유: :func:`automl_agent.privacy.public_result`는 숫자인 지표 값만
        지키므로, tuple은 보고서와 Critic이 읽는 state 채널로 가는 길에 버려진다. 그 채널이 구간이
        도착해야 하는 하나뿐인 곳이다.
        """
        return {f"{metric}_ci_low": self.low, f"{metric}_ci_high": self.high}


def _sample_units(y_true: Any, groups: Any, resamples: int) -> tuple[Any, list[Any] | None, int, str] | None:
    """``(y 배열, 단위별 행 인덱스, 단위 수, 단위 이름)``, 또는 재표집할 것이 없으면 ``None``.

    두 측정 함수가 같은 일곱 줄로 시작했고, 그 순서가 계약의 일부다: ``resamples``와 빈 분할을 먼저 보고,
    그다음에야 :func:`_units`가 어긋난 group 배열에 raise한다.
    """
    import numpy as np

    if resamples <= 0:
        return None
    y_arr = np.asarray(y_true)
    n_rows = int(len(y_arr))
    if n_rows == 0:
        return None
    rows_by_unit, unit = _units(groups, n_rows)
    n_units = n_rows if rows_by_unit is None else len(rows_by_unit)
    if n_units < MIN_UNITS:
        return None
    return y_arr, rows_by_unit, n_units, unit


def _resample(
    measure: Callable[[Any], float | None],
    rows_by_unit: list[Any] | None,
    n_units: int,
    seed: int,
    resamples: int,
) -> list[float] | None:
    """재표집 인덱스마다 ``measure``를 불러 모은 값들, 또는 쓸 만한 것이 덜 모이면 ``None``.

    두 측정 함수가 공유하는 루프다. 한쪽은 점수를, 다른 쪽은 두 점수의 차이를 재지만 뽑는 방식은 같아야
    한다 — 같은 ``seed``에서 같은 인덱스가 나오는 것이 두 반복의 구간을 비교할 수 있게 만드는 것이다.

    퇴화한 재표집은 ``None``을 돌려주거나 raise할 수 있다. 둘 다 "여기서는 정의되지 않음"이고 실패한
    실행이 아니다. 그렇게 버려진 것이 :data:`MIN_VALID_SHARE`를 넘으면 구간 대신 ``None``이 된다.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n_units, n_units)
        index = picks if rows_by_unit is None else np.concatenate([rows_by_unit[p] for p in picks])
        try:
            value = measure(index)
        except (ValueError, IndexError, ZeroDivisionError):
            continue
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))
    if len(values) < MIN_VALID_SHARE * resamples:
        return None
    return values


def _percentiles(values: list[float]) -> tuple[float, float]:
    """:data:`CI_LEVEL`의 양끝 백분위. 꼬리를 한 곳에서만 계산하려고 있다."""
    import numpy as np

    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = (float(x) for x in np.percentile(values, [tail, 100.0 - tail]))
    return low, high


def bootstrap_interval(
    score: Callable[[Any, Any, Any], float | None],
    y_true: Any,
    pred: Any,
    proba: Any = None,
    *,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> Interval | None:
    """채점된 분할을 ``resamples``번 재표집해 백분위 구간을 돌려준다.

    ``score``는 ``(y_true, pred, proba)`` 위의 callable이므로 이 모듈은 어느 지표인지도 어느
    라이브러리인지도 모른다 — 고정된 스크립트 둘이 자기 scorer를 넘기고 sklearn의 유일한 import자로
    남는다.

    공개할 만한 구간이 없을 때(단위가 너무 적음, ``resamples``가 0, 지표가 너무 자주 정의되지 않음)
    **예외가 아니라 ``None``**. 호출자들이 그것을 "측정되지 않음"으로 읽으므로, 퇴화한 분할은 시도가
    아니라 공개 한 줄을 문다.

    **``seed``에 대해 결정적** — 그것이 두 반복의 구간을 비교할 수 있게 만드는 것이다.
    """
    import numpy as np

    prepared = _sample_units(y_true, groups, resamples)
    if prepared is None:
        return None
    y_arr, rows_by_unit, n_units, unit = prepared

    pred_arr = np.asarray(pred)
    proba_arr = None if proba is None else np.asarray(proba)

    def measure(index: Any) -> float | None:
        return score(
            y_arr[index],
            pred_arr[index],
            None if proba_arr is None else proba_arr[index],
        )

    values = _resample(measure, rows_by_unit, n_units, seed, resamples)
    if values is None:
        return None
    low, high = _percentiles(values)
    return Interval(low=round(low, 6), high=round(high, 6), resamples=len(values), unit=unit)


def _units(groups: Any, n_rows: int) -> tuple[list[Any] | None, str]:
    """``(재표집 단위별 행 인덱스, 단위 이름)``.

    행 단위 경로에서는 ``None``이다. 거기서 단위는 행 *자체*이고, 1개짜리 인덱스 배열의 목록을 세우는
    것은 메모리만 쓴다. 그 밖에는 서로 다른 그룹마다 인덱스 배열 하나이고, 그것이 재표집을 환자 단위로
    뽑게 만드는 것이다.

    길이가 어긋난 group 배열에는 둘 중 짧은 쪽으로 맞추는 대신 raise한다: 엉뚱한 그룹을 재표집하면
    좁은 구간이 나오고, 그것은 버그가 아니라 자신 있는 측정으로 읽힌다 — ``split_three_way``가 대는
    것과 같은 논거다.
    """
    import numpy as np

    if groups is None:
        return None, UNIT_ROW
    group_arr = np.asarray(groups)
    if len(group_arr) != n_rows:
        raise ValueError(
            f"group 배열의 길이가 행 수와 다릅니다 — groups={len(group_arr)}, rows={n_rows}"
        )
    _, inverse = np.unique(group_arr, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return [np.flatnonzero(inverse == code) for code in range(int(inverse.max()) + 1)], UNIT_GROUP


# --------------------------------------------------------------------------- #
# 짝지은 반쪽: 두 점수가 아니라 차이를 재표집한다
# --------------------------------------------------------------------------- #

# 짝지은 비교가 ``result.json``에서 앉는 곳. ``metrics``에 키 넷을 더하는 대신 자기 블록인 이유는
# 이들이 이 행들 위의 모델이 아니라 *비교*의 성질이기 때문이다: metrics dict에서 ``roc_auc`` 옆에 앉은
# ``p_better``는 보고서의 지표 표로 실려 가 점수로 읽힌다. ``metrics``에 사는 진단의 선례
# — ``balanced_accuracy_cut_headroom`` — 는 단일 모델 측정이고 이것은 그렇지 않다.
PAIRED_KEY = "paired"

# 네 숫자, 쓰는 쪽과 모든 읽는 쪽이 어긋날 수 없게 한 번만 이름 붙인다. :meth:`Interval.flatten`과 같은
# 논거다: 쌍 더하기 스칼라가 아니라 float 넷인 이유는
# :func:`automl_agent.privacy.public_result`가 숫자를 지키고 구조를 버리기 때문이다.
PAIRED_FIELDS: tuple[str, ...] = ("delta_vs_best", "delta_ci_low", "delta_ci_high", "p_better")

PAIRED_MEASURED = "measured"
PAIRED_SKIPPED = "skipped"
# :data:`RESAMPLE_UNITS`와 같은 논거: 체는 ``status``를 ``str``이 아니라 이 모듈이 쓰는 단어들에 대고
# 검사해야 한다.
PAIRED_STATUSES: tuple[str, ...] = (PAIRED_MEASURED, PAIRED_SKIPPED)

# 비교가 이뤄지지 않은 이유. 없는 채로 두는 대신 공개하는 이유는 "짝지은 판정 없음"과 "짝지은 판정이
# 아무것도 못 찾음"이 다른 사실이고, 앞의 것만 가진 독자는 침묵을 뒤의 것으로 받아들이기 때문이다.
PAIRED_REASONS: dict[str, str] = {
    "no_baseline": "짝지을 직전 최고가 없습니다 — 첫 측정입니다",
    "baseline_missing": "직전 최고의 행별 예측 파일이 없습니다",
    "split_changed": "이 시도의 val 행이 직전 최고의 val 행과 다릅니다 — 짝지을 수 없습니다",
    "degenerate": "차이를 재표집할 수 없었습니다 — 행이 너무 적거나 지표가 축퇴했습니다",
}


@dataclass(frozen=True)
class PairedDelta:
    """같은 행에서 채점된 두 시도 사이의 재표집된 *차이*."""

    # 지표가 어느 쪽으로 가든 항상 candidate 빼기 baseline. 지표의 방향에 따라 뒤집히는 부호 있는 차이는
    # ledger의 "직전 최고 대비 -0.0500"과 이 숫자가 ``rmse``에서 부호로 어긋나게 만든다 — 방향은
    # ``p_better``에만, 그것이 속한 곳에만 적용된다.
    delta: float
    low: float
    high: float
    # candidate가 *더 좋았던* 재표집의 비율. 그래서 minimize 지표에서도 같은 방식으로 읽힌다. 동점은
    # 더 좋지 않은 것으로 센다: 비트까지 같은 예측을 낸 두 시도는 0.0을 받고, 그것이 "어떤 재표집도
    # 개선을 보이지 않았다"의 정직한 독법이다.
    p_better: float
    resamples: int
    unit: str

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    @property
    def resolved(self) -> bool:
        """차이의 구간이 0을 배제하는지.

        :func:`contains`와 정확히 같게 좁다: 0을 배제한다는 것은 이 행들이 두 시도를 실제로 갈랐다는
        뜻이다. *무엇이* 그들을 갈랐는지는 아무 말도 하지 않는다 — 레버를 둘 움직인 전이는 잘 분해된
        차이를 갖고도 귀속이 없다.
        """
        return not (self.low <= 0.0 <= self.high)

    def flatten(self) -> dict[str, float]:
        """공개되는 네 숫자, :data:`PAIRED_FIELDS`를 키로."""
        return dict(
            zip(PAIRED_FIELDS, (self.delta, self.low, self.high, self.p_better), strict=True)
        )


def paired_delta(
    score: Callable[[Any, Any, Any], float | None],
    y_true: Any,
    candidate: tuple[Any, Any],
    baseline: tuple[Any, Any],
    *,
    direction: str,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> PairedDelta | None:
    """같은 행에 대한 두 예측 사이의 *차이*를 재표집한다.

    **재표집마다 동일한 뽑기 위에서 둘 다 채점한다** — 그것이 요점 전부이고,
    :func:`bootstrap_interval` 두 번으로 이것을 조립할 수 없는 이유다.

    **``direction``은 필수이고 기본값이 없다**: ``delta``의 어느 부호가 ``p_better``에 세어지는지를
    그것이 정하고, 잘못된 기본값은 ``rmse``의 악화를 개선 확률 0.99로 보고한다.

    :func:`bootstrap_interval`과 같은 세 조건에서 ``None``. **길이가 *어긋난* baseline은 raise한다** —
    다른 행 수의 예측은 이 분할의 것이 아니고, 그것을 짝지으면 무의미한 숫자 주위의 좁은 구간이 나온다.
    """
    import numpy as np

    prepared = _sample_units(y_true, groups, resamples)
    if prepared is None:
        return None
    y_arr, rows_by_unit, n_units, unit = prepared
    n_rows = int(len(y_arr))

    sides = [_side(part, n_rows) for part in (candidate, baseline)]
    (a_pred, a_proba), (b_pred, b_proba) = sides

    def difference(index: Any) -> float | None:
        y_slice = y_arr[index]
        a = score(y_slice, a_pred[index], None if a_proba is None else a_proba[index])
        b = score(y_slice, b_pred[index], None if b_proba is None else b_proba[index])
        if a is None or b is None:
            return None
        return float(a) - float(b)

    try:
        observed = difference(np.arange(n_rows))
    except (ValueError, IndexError, ZeroDivisionError):
        observed = None
    if observed is None or not np.isfinite(observed):
        return None

    values = _resample(difference, rows_by_unit, n_units, seed, resamples)
    if values is None:
        return None
    deltas = np.asarray(values)
    low, high = _percentiles(values)
    better = deltas < 0.0 if direction == MINIMIZE else deltas > 0.0
    return PairedDelta(
        delta=round(float(observed), 6),
        low=round(low, 6),
        high=round(high, 6),
        p_better=round(float(better.mean()), 4),
        resamples=len(values),
        unit=unit,
    )


def _side(part: tuple[Any, Any], n_rows: int) -> tuple[Any, Any]:
    """한 시도의 ``(pred, proba)``를 배열로, 행 수에 대고 검사해서.

    :func:`_units`와 같은 이유로 자르는 대신 raise한다: 길이 어긋남은 이것이 호출자가 생각하는 행이
    아니라는 뜻이고, 짐작의 대가는 엉뚱한 비교에 대한 자신 있는 숫자다.
    """
    import numpy as np

    pred, proba = part
    pred_arr = np.asarray(pred)
    if len(pred_arr) != n_rows:
        raise ValueError(
            f"짝지을 예측의 길이가 행 수와 다릅니다 — pred={len(pred_arr)}, rows={n_rows}"
        )
    if proba is None:
        return pred_arr, None
    proba_arr = np.asarray(proba)
    if len(proba_arr) != n_rows:
        raise ValueError(
            f"짝지을 확률의 길이가 행 수와 다릅니다 — proba={len(proba_arr)}, rows={n_rows}"
        )
    return pred_arr, proba_arr


# --------------------------------------------------------------------------- #
# 하나를 되읽기
# --------------------------------------------------------------------------- #


def interval_of(metrics: Mapping[str, Any] | None, metric: str) -> tuple[float, float] | None:
    """``metric``에 공개된 양끝, 또는 이 결과에 그것이 없으면 ``None``.

    :meth:`Interval.flatten`의 읽는 반쪽. 모든 소비자가 두 키 이름을 적는 대신 이것을 지나는 이유는
    구간이 있기 전에 쓰인 결과 — 또는 분할이 구간에 너무 작았던 결과 — 가 모든 곳에서 "측정되지 않음"으로
    읽혀야 하기 때문이다.
    """
    if not metrics:
        return None
    low = as_number(metrics.get(f"{metric}_ci_low"))
    high = as_number(metrics.get(f"{metric}_ci_high"))
    if low is None or high is None or high < low:
        return None
    return low, high


def contains(bounds: tuple[float, float] | None, value: Any) -> bool:
    """``value``가 구간 안에 드는지. 어느 쪽이든 없으면 ``False``.

    소비자들이 하는 하나뿐인 비교이고, 그 뜻은 일부러 좁다: 구간 안의 값은 *이 행들이 측정된 점수와
    구분할 수 없는* 값이다. 가설 검정이 아니고, 두 숫자가 같다고 말하지도 않는다.
    """
    number = as_number(value)
    if bounds is None or number is None:
        return False
    return bounds[0] <= number <= bounds[1]


def describe_interval(metric: str, score: Any, bounds: tuple[float, float] | None) -> str:
    """한글 조각 하나 — ``f1=0.7412 (95% CI 0.7108~0.7702, 폭 0.0594)``.

    말할 것이 없으면 빈 문자열이므로 호출자가 조건 없이 이어 붙일 수 있다.
    """
    value = as_number(score)
    if bounds is None:
        return f"{metric}={value:.4f}" if value is not None else ""
    low, high = bounds
    head = f"{metric}={value:.4f} " if value is not None else f"{metric} "
    return f"{head}({int(CI_LEVEL * 100)}% CI {low:.4f}~{high:.4f}, 폭 {high - low:.4f})"


def paired_of(
    result: Mapping[str, Any] | None, *, baseline_iteration: Any
) -> dict[str, Any] | None:
    """이 결과의 짝지은 비교 — 단, 호출자가 보고하는 그 짝일 때만.

    ``baseline_iteration``은 필터가 아니라 불변식이다. 공개되는 필드 이름이 ``delta_vs_best``와
    ``p_better``이므로 둘 다 자기가 렌더되는 문장과 *같은* baseline에 대한 것이어야 한다 — iteration 1에
    대고 빼면서 iteration 4에 대고 계산된 P를 찍으면, 모든 숫자가 진짜이고 주장은 아닌 줄이 나온다.
    블록과 호출자의 baseline이 둘 다 손에 있는 곳은 여기뿐이라 검사가 선택이 아니라 필수이고, 어긋남은
    조용히 렌더되는 대신 "측정되지 않음"으로 읽힌다.

    건너뛴 비교, 그리고 :data:`PAIRED_FIELDS` 중 하나라도 없는 블록에도 ``None``이다 — 이것이 있기 전에
    쓰인 결과가 그렇게 생겼다.
    """
    block = (result or {}).get(PAIRED_KEY)
    if not isinstance(block, Mapping) or block.get("status") != PAIRED_MEASURED:
        return None
    if _iteration(block.get("baseline_iteration")) != _iteration(baseline_iteration):
        return None
    values = {field: as_number(block.get(field)) for field in PAIRED_FIELDS}
    if any(value is None for value in values.values()):
        return None
    return {**dict(block), **values}


def describe_paired(block: Mapping[str, Any] | None) -> str:
    """ledger 한 행을 위한 한글 조각, 또는 말할 것이 없으면 ``""``.

    일부러 판정을 말하고 멈춘다. "0과 구분되지 않음"이 차이의 구간이 뒷받침하는 주장 전부다 — 변경이
    아무것도 안 했다는 것도, 그 전이의 레버 중 어느 것 탓인지도 아니다.
    """
    if not isinstance(block, Mapping):
        return ""
    if block.get("status") == PAIRED_SKIPPED:
        reason = PAIRED_REASONS.get(str(block.get("reason") or ""))
        return f"[짝지은 검정 없음: {reason}]" if reason else ""
    delta, low, high, p_better = (as_number(block.get(field)) for field in PAIRED_FIELDS)
    if delta is None or low is None or high is None or p_better is None:
        return ""
    verdict = "0과 구분됨" if not low <= 0.0 <= high else "이 행들로는 0과 구분되지 않음"
    against = block.get("baseline_iteration")
    head = f"짝지은 Δ(iteration {against} 대비)" if against is not None else "짝지은 Δ"
    text = (
        f"[{head} {delta:+.4f}, {int(CI_LEVEL * 100)}% CI {low:+.4f}~{high:+.4f}, "
        f"P(개선) {p_better:.3f} — {verdict}]"
    )
    if block.get("threads_changed") is True:
        # 판정에 접어 넣는 대신 덧붙인다. 구간은 여전히 그 구간이기 때문이다: 두 예측 벡터는 정말 그만큼
        # 다르다. 이 행이 더 이상 말할 수 없는 것은 계획이 그 이유라는 것이다 — 스레드 수를 넘나드는 같은
        # 설정이 이 구간 하나의 폭만큼 벌어지므로(:mod:`automl_agent.threads`), 환경은 주장되는 것 옆에서
        # 반올림 오차가 아니다.
        text += " [baseline과 스레드 상태가 다릅니다 — 이 Δ에는 환경 차이가 섞여 있습니다]"
    return text


def _iteration(value: Any) -> int | None:
    """반복 번호를 ``int``로, 또는 ``None``. ``bool``은 반복 번호가 아니다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def resolution_note(
    metrics: Mapping[str, Any] | None,
    metric: str,
    *,
    others: Mapping[str, Any] | None = None,
) -> str:
    """이 시도의 슬라이스가 무엇을 분해할 수 있고 무엇을 못 하는지, 프롬프트용 한 단락으로.

    ``others``는 이 시도가 비교될 숫자에 라벨을 붙인 것이고 — 바, 이전 점수들 — 구간 안에 드는 것마다
    이름이 불린다. 그것이 Critic의 실제 실패 양태에 답한다: 점추정 둘을 건네받고 왜 두 번째가 더 낮은지
    물으면, 행들이 세운 적 없는 차이에 대해 원인을 진단했다.

    **일부러 그것에 대해 무엇을 할지는 말하지 않는다.** 겹침은 손잡이가 아무것도 안 했다는 뜻일 수도,
    슬라이스가 그것이 한 일을 보기에 너무 작다는 뜻일 수도 있다. 그 둘을 가르는 것은 나머지 증거뿐이고,
    여기서의 처방은 진단을 앞질러 버린다.

    **없는 구간은 침묵이 아니라 자기 문장을 받는다** — 절은 어느 쪽이든 존재하고, 빈 절은 "불확실성
    없음"으로 읽힌다.
    """
    bounds = interval_of(metrics, metric)
    if bounds is None:
        return (
            f"{metric}의 신뢰구간이 이 시도에는 없습니다 — 슬라이스가 너무 작거나 지표가 "
            "축퇴했습니다. 차이의 크기를 노이즈와 견줄 근거가 없으니 작은 변화를 "
            "개선이나 악화로 읽지 마십시오."
        )
    line = describe_interval(metric, (metrics or {}).get(metric), bounds)
    collisions = [
        f"{label}({as_number(value):.4f})"
        for label, value in (others or {}).items()
        if contains(bounds, value)
    ]
    if not collisions:
        return (
            f"{line}. 비교 대상 중 이 구간 안에 들어오는 값은 없으므로, 위 점수와의 차이는 "
            "이 슬라이스가 실제로 구분해 낸 차이입니다."
        )
    return (
        f"{line}. 다음 값들이 이 구간 안에 있습니다: {', '.join(collisions)} — 이 행들로는 "
        "위 점수와 구분되지 않는 값들이므로, 그 차이를 원인이 있는 변화로 진단하지 "
        "마십시오. 구간 밖의 차이만 증거로 쓸 수 있습니다."
    )


def as_number(value: Any) -> float | None:
    """실수면 ``float``, 아니면 ``None``. 여기서 ``bool``은 숫자가 아니다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
