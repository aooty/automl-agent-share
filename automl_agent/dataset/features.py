"""특성 열: 실행기가 쓸 수 있는 것, 그리고 나머지가 숫자가 되는 방식.

**구현이 둘이 아니라 모듈이 하나다.** 프로파일러의 기준선과 모든 학습 시도가 *같은* 특성 행렬을 봐야
한다 — 아니면 한쪽에서 유도된 바가 다른 쪽에 대고 잰 점수와 비교되지 않는다. 둘 다
:func:`encode_features`를 부르고, 어느 쪽도 열 정책을 소유하지 않는다.

열 규칙 셋, 각각이 가드다:

- **one-hot이고 결코 ordinal이 아니다**.
- :data:`MAX_ONEHOT_CARDINALITY` 위는 **떨어지고, 떨어질 때 이름이 불린다.** 자유 텍스트 열은 수천
  열로 one-hot된다. 떨어뜨리는 것은 되돌릴 수 있고 발표되지만, 침묵은 그렇지 않다.
- **결측은 자기 level이다**.

**level 집합은 일부러 모든 행에서 적합된다** — 특성 값만, 타깃은 결코.

**적합과 변환이 갈라진 이유는 "파일의 순수 함수"가 바로 적합된 모델을 두 번째 파일에서 쓸 수 없게
만드는 성질이기 때문이다.** 새 행을 다시 인코딩하면 level 하나가 없을 때 너비가 달라지고, 더 나쁘게는
같은 너비인데 열들이 다른 것을 뜻하며, 아무것도 그것을 보고하지 않는다. 그래서
:func:`build_schema`는 배치를 적합하고, :func:`encode_with_schema`는 있는 배치를 적용하고,
:func:`encode_features`는 자기 파일을 소유한 호출자를 위해 둘 다 한다.

**스키마는 너비 검사가 볼 수 없는 것도 기록한다**: 모델을 unpickle할 라이브러리 버전
(:func:`environment_drift`)과 열이 담고 있던 것 (:func:`column_stats`). 둘 다 **보고되고 결코
작동되지 않는다** — 거절은 틀린 답이 아니면 조용히 계산될 자리에 남고, 이것들은 사람이 재야 하는
사실이다.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import 비용만이고, pandas는 subprocess 전용이다
    import pandas as pd

# 이보다 서로 다른 값이 많은 열은 one-hot되는 대신 떨어진다.
MAX_ONEHOT_CARDINALITY = 50

# 생성된 열 이름에서 원본 열과 그 level을 가르는 글자. 명시적으로 두는 이유는
# ``dropped_hyperparams``나 로그 한 줄을 읽는 사람이 인코딩된 열과 파일에서 그렇게 이름 붙은 열을
# 가릴 수 있어야 하기 때문이다.
LEVEL_SEPARATOR = "="

# 결측이 되는 level. 괄호를 쓴 철자라서 진짜로 "nan"이라는 텍스트를 담은 열과 혼동될 수 없다 —
# ``get_dummies``라면 그것을 그렇게 이름 붙인다.
MISSING_LEVEL = "<missing>"

# 스키마가 읽는 사람이 검사할 수 있는 무언가를 얻을 때 올린다. :func:`encode_with_schema`가 읽고,
# 거기 규칙은 일부러 비대칭이다: 이 상수보다 *새* 버전은 거절되고, *옛* 버전은 받아들여지되 그것이
# 지원할 수 없는 검사의 이름이 불린다.
SCHEMA_VERSION = 2

# 이 빌드가 아직 읽을 수 있는 가장 오래된 버전. 옛 스키마를 그저 얇게가 아니라 *틀리게* 만드는
# 변경 — 뜻이 바뀐 키 — 에만 올린다. ``0``은 필드의 부재이고, 그것은 옛 스키마가 아니라 스키마가 아닌
# 파일이다.
MIN_SCHEMA_VERSION = 1

# 각 버전이 더한 검사, 더한 버전으로 키를 잡아서. 옛 스키마도 읽히고, 그 독자에게 —
# ``describe_drift``를 거쳐 — "드리프트 없음"이 보이는 것보다 좁은 땅을 덮었다고 말하는 것이 이것이다.
# 거기서의 침묵은 이 모듈 전체가 대비하는 실패다: 돌지 않은 비교에 대한 안심시키는 보고.
SCHEMA_CHECKS_ADDED: dict[int, tuple[str, ...]] = {
    2: ("environment", "column_stats"),
}

# fit과 새 배치 사이에서 비율이 얼마나 움직여야 보고되는지. 조건이 둘이고 둘 다 필요하다: 이 바닥값,
# 그리고 *이 배치의* 행 수에서 두 비율 중 큰 쪽의 표준오차 2배. 두 번째가 어떤 행 하한 아래에서
# 건너뛰는 대신 짧은 배치에서도 비교가 돌게 한다 — 조용히 건너뛰는 것이 여기서의 실패 양식이고,
# 배치가 짧아질수록 넓어지는 바는 규칙 하나로 같은 말을 한다.
#
# 두 바닥값은 다르다 — 결측 코드 쪽이 명목값이고, 잡음 항이 일을 한다.
MISSING_RATE_SHIFT = 0.05
SENTINEL_RATE_SHIFT = 0.01


class FeatureSchemaMismatch(ValueError):
    """적합된 모델이 요구하는 방식으로 새 행을 인코딩할 수 없다.

    우회하는 대신 raise한다. 대안이 모두 조용하기 때문이다.
    """


# --------------------------------------------------------------------------- #
# 배치가 적합된 환경
# --------------------------------------------------------------------------- #
#
# 스키마는 행렬의 어느 열이 무엇이었는지를 박는다. 그 행렬을 예측으로 바꾸는 코드는 박지 않는데, 그
# 코드도 디스크에 있다: ``model.joblib``은 sklearn 객체의 pickle이고, 열릴 때 설치돼 있는 sklearn이
# 복원한다. sklearn 자신은 경고만 하고(``InconsistentVersionWarning``, stderr로, subprocess 로그에서
# 쉽게 묻힌다), 그 경고가 말하는 실패는 크래시가 아니다 — 버전 사이에 옮겨진 속성이 기본값으로
# 복원되고, 모델은 조용히, 다르게 예측한다.
#
# 그래서 버전은 적합 때 기록되고 재생 때 비교된다. 패키지를 import하는 대신
# ``importlib.metadata``로 이름으로 기록하는 이유는 이것이 pandas를 import해선 안 되는 오케스트레이터
# 프로세스에서도 불릴 수 있어야 하기 때문이다.

ENVIRONMENT_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "joblib")


def current_environment() -> dict[str, str]:
    """이 프로세스가 모델을 적합하거나 재생할 버전들.

    설치되지 않은 패키지는 부재로 기록되는 대신 빠진다: 비교는 같은 모양인 두 기록 사이에서 이뤄지고,
    그러지 않으면 "여기 없음"이 변경으로 읽힌다. ``python``은 언제나 있다.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {"python": platform.python_version()}
    for name in ENVIRONMENT_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - all four are hard dependencies
            continue
    return found


def _same_version(package: str, fit: str, now: str) -> bool:
    """두 버전 문자열이 한 줄을 쓸 만하지 않을 정도로 가까운지.

    라이브러리는 정확히 — sklearn 자신의 호환 바가 정확하고, pandas의 패치 릴리스가 dtype 기본값을
    바꾼 적이 있다. ``python``은 major-minor로: 패치 릴리스는 pickle 프로토콜도 ABI도 바꾸지 않으므로,
    보고하면 평범한 인터프리터 업그레이드에 한 줄이 나고 읽는 사람은 이 블록을 건너뛰는 것을 배운다.
    """
    if fit == now:
        return True
    if package == "python":
        return fit.split(".")[:2] == now.split(".")[:2]
    return False


def environment_drift(recorded: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """기록된 버전 중 지금 도는 것과 다른 것들. 맞으면 빈 목록.

    보고하고 결코 거절하지 않는다.
    """
    if not recorded:
        return []
    now = current_environment()
    changed: list[dict[str, str]] = []
    for name, fitted in recorded.items():
        package = str(name)
        running = now.get(package)
        if running is None or _same_version(package, str(fitted), running):
            continue
        changed.append({"package": package, "fit": str(fitted), "now": running})
    return changed


def missing_checks(version: int) -> list[str]:
    """이 ``version``의 스키마가 돌릴 것을 아무것도 싣지 않은 검사 이름들."""
    return [
        name
        for added, names in sorted(SCHEMA_CHECKS_ADDED.items())
        if version < added
        for name in names
    ]


def is_numeric_column(series: pd.Series) -> bool:
    """추정기가 아무 인코딩 없이 먹는 dtype에 대해 True."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series)) or bool(
        pd.api.types.is_bool_dtype(series)
    )


def is_text_like_column(series: pd.Series) -> bool:
    """level 집합을 읽어 낼 수 있는 dtype에 대해 True. 오늘 pandas가 그것을 뭐라고 부르든.

    버전을 건너 같은 것에 이름이 셋이다: pandas 2는 문자열 열에 ``object``를, pandas 3은 ``str``을
    주고, 명시적으로 변환된 것은 ``CategoricalDtype``이다. datetime은 제외된다 — 타임스탬프 위의
    one-hot은 무엇의 인코딩도 아니고, 상한은 서로 다른 날짜가 적은 열을 잡지 못한다.
    """
    import pandas as pd

    dtype = series.dtype
    if is_numeric_column(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return False
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def is_encodable_column(series: pd.Series, distinct: int) -> bool:
    """one-hot할 만큼 좁은 범주 열에 대해 True.

    ``distinct``는 있는 값에서 셈되므로, level 둘에 결측이 있는 열은 셋이 아니라 둘이다 — 결측 level은
    인코더가 더하는 것이고 상한에 들어가지 않는다.
    """
    return is_text_like_column(series) and 0 < distinct <= MAX_ONEHOT_CARDINALITY


# --------------------------------------------------------------------------- #
# 배치가 적합될 때 열이 어떻게 보였는가
# --------------------------------------------------------------------------- #
#
# 배치는 *모양*이 바뀐 열을 잡는다. 모양을 지킨 채 뜻이 바뀐 열은 잡을 수 없고, 그렇게 되는 가장 흔한
# 방식은 결측 관례다: 학습 추출본은 측정이 없는 자리에 ``-9999``를 적었는데 다음 달 추출본은 빈 칸을
# 적는다. 둘 다 같은 이름의 수치 열이고, 둘 다 같은 자리로 인코딩되고, 여기까지 아무것도 보고하지
# 않는다.
#
# 그래서 적합은 열마다 얼마나 자주 결측이었는지와 어떤 관례적 결측 코드를 실었는지를 기록하고, 재생은
# 비교한다. 탐지하고, 보고하고, 결코 변환하지 않는다 — :mod:`automl_agent.dataset.sentinels`가 말하는
# 같은 규칙이고 같은 이유다: ``-1``이 코드인지 측정치인지는 추론할 수 없고, 조용히 고쳐 쓰면 모델이
# 파일에 없는 행을 먹는다는 뜻이 된다.


def column_stats(features: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """인코딩되는 열에 대한 열별 결측률과 의심되는 결측 코드.

    셈이 아니라 비율이라 900행 적합과 40행 배치가 비교되기라도 한다. 여기 나타날 수 있는 값은
    :data:`automl_agent.dataset.sentinels.NUMERIC_CODES`에 있는 것뿐이고, 그것이 "스키마가 이제 임의의
    셀 값을 싣는다"는 반론 밖에 이것을 두게 한다.
    """
    stats: dict[str, Any] = {}
    for name in columns:
        series = features[name]
        entry: dict[str, Any] = {"missing_rate": round(float(series.isna().mean()), 4)}
        found = _numeric_sentinels(series)
        if found:
            entry["sentinels"] = found
        stats[str(name)] = entry
    return stats


def rate_changed(fit: float, now: float, rows: int, floor: float = MISSING_RATE_SHIFT) -> bool:
    """``rows`` 행에 걸친 두 비율이 잡음보다 크게, 그리고 뜻이 있을 만큼 다른지.

    바의 양쪽 절반과 ``floor``가 상수 하나가 아니라 파라미터인 이유는 :data:`MISSING_RATE_SHIFT`.
    """
    if rows <= 0:
        return False
    gap = abs(float(now) - float(fit))
    if gap < floor:
        return False
    p = min(max(max(float(fit), float(now)), 0.0), 1.0)
    return gap >= 2.0 * ((p * (1.0 - p) / float(rows)) ** 0.5)


def _sentinel_rates(series: pd.Series, recorded: list[dict[str, Any]]) -> dict[float, float]:
    """코드마다 이 배치의 비율. 다시 탐지하는 대신 값으로 센다.

    fit이 이미 이름 부른 코드에 ``detect_sentinels``를 쓰지 않는 것은 일부러다
    . fit이 값을 이름 부른 뒤로는 세는 것이 정확하다.
    """
    if not recorded:
        return {}
    rows = int(len(series))
    if rows == 0:
        return {}
    present = series if is_numeric_column(series) else _coerce(series)
    rates: dict[float, float] = {}
    for item in recorded:
        value = float(item["value"])
        rates[value] = float((present == value).sum()) / rows
    return rates


def _coerce(series: pd.Series) -> pd.Series:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _numeric_sentinels(series: pd.Series, known: Container[float] = ()) -> list[dict[str, Any]]:
    """숫자 코드 발견만, 스키마가 싣는 모양으로. ``known``에 있는 값은 빠진다.

    탐지이므로 하한이 있다: 짧은 배치에서 덜 보고하고 — 고립 검사가 뜻을 갖기까지 코드 하나에 20행과
    서로 다른 값 5개가 필요하다 — 여기서는 그쪽이 틀릴 옳은 방향이다. 대안은 12행 파일의 최솟값이
    낮다는 이유로 관례가 바뀌었다고 알리는 것이다.
    """
    from automl_agent.dataset.sentinels import KIND_NUMERIC_CODE, detect_sentinels

    if not is_numeric_column(series):
        return []
    return [
        {"value": float(item["value"]), "rate": float(item["rate"])}
        for item in detect_sentinels(series)
        if item.get("kind") == KIND_NUMERIC_CODE and float(item["value"]) not in known
    ]


def _compare_column_stats(
    features: pd.DataFrame,
    recorded: Mapping[str, Any] | None,
    required: list[str],
    drift: dict[str, Any],
) -> None:
    """``missing_rate_shift``와 ``sentinel_shift``를 제자리에서 채운다.

    스키마에 ``column_stats``가 없으면 아무것도 하지 않는다 — version 1 스키마이고, 그 독자는 합의처럼
    읽히는 빈 발견 목록이 아니라 ``missing_checks``로 그 사실을 듣는다.
    """
    if not recorded:
        return
    rows = int(len(features))
    for name in required:
        entry = dict(recorded.get(name) or {})
        if not entry:
            continue
        series = features[name]
        fit_missing = float(entry.get("missing_rate") or 0.0)
        now_missing = float(series.isna().mean()) if rows else 0.0
        if rate_changed(fit_missing, now_missing, rows):
            drift["missing_rate_shift"].append(
                {
                    "column": name,
                    "fit": round(fit_missing, 4),
                    "now": round(now_missing, 4),
                }
            )
        codes = [dict(item) for item in entry.get("sentinels") or []]
        rates = _sentinel_rates(series, codes)
        for item in codes:
            value = float(item["value"])
            fit_rate = float(item.get("rate") or 0.0)
            now_rate = float(rates.get(value, 0.0))
            if rate_changed(fit_rate, now_rate, rows, SENTINEL_RATE_SHIFT):
                drift["sentinel_shift"].append(
                    {
                        "column": name,
                        "value": value,
                        "fit": round(fit_rate, 4),
                        "now": round(now_rate, 4),
                    }
                )
        # 비율 바에 걸리지 않는다. 그 바는 비율이 너무 적은 행에서 읽히는 것을 막으려 있고, fit이
        # 아예 기록하지 않은 코드는 비율 질문이 아니다 — ``detect_sentinels``가 답하기 전에 이미
        # 자기 하한을 걸었다.
        for item in _numeric_sentinels(series, {float(code["value"]) for code in codes}):
            drift["sentinel_shift"].append(
                {
                    "column": name,
                    "value": float(item["value"]),
                    "fit": 0.0,
                    "now": round(float(item["rate"]), 4),
                }
            )


def build_schema(features: pd.DataFrame) -> dict[str, Any]:
    """배치를 적합한다: 어느 열이 추정기에 닿는지, 어느 열로서, 어느 순서로.

    일부러 JSON 직렬화 가능하다 — 적합된 모델 옆에 디스크로 쓰이고 다른 프로세스가 읽어 간다. 범주
    *level*을 싣고 그것은 셀 값이므로, ``model.joblib``과 행별 예측과 같은 경계의 비공개 쪽에 산다:
    ``artifacts/`` 안, 상태 채널에는 결코, 프롬프트에는 결코.

    ``columns``는 ``numeric`` 더하기 ``one_hot``과 중복인데도 저장된다. 한 번 적힌 행렬 배치라서,
    :func:`encode_with_schema`가 같은 순서의 두 유도가 맞기를 믿는 대신 자기가 만든 것을 단언할 수 있다.
    """
    numeric: list[str] = []
    one_hot: list[dict[str, Any]] = []
    too_wide: list[str] = []
    unsupported: list[str] = []
    for column in features.columns:
        series = features[column]
        if is_numeric_column(series):
            numeric.append(str(column))
        elif is_encodable_column(series, int(series.dropna().nunique())):
            one_hot.append(
                {
                    "column": str(column),
                    # level을 정렬해서 생성된 열 순서가 level 이름의 함수이고 행 순서의 함수가
                    # 아니게 — 같은 level을 가진 두 파일이 똑같이 인코딩된다.
                    "levels": sorted(str(value) for value in series.dropna().unique()),
                    # 결측을 결측으로 지켜서 인코더가 전부 0인 행 대신 자기 열을 줄 수 있게.
                    # 기록할 것이 있을 때만: 전부 0인 열은 상수 특성이 되고 메모리 가드가 여전히
                    # 값을 매긴다.
                    "missing_level": bool(series.isna().any()),
                }
            )
        elif is_text_like_column(series):
            # 이름을 부를 수 있고 행동할 수 있다: 호출자가 버킷으로 묶어서 다시 돌릴 수 있다.
            too_wide.append(str(column))
        else:
            unsupported.append(str(column))

    columns = list(numeric)
    for spec in one_hot:
        columns += _level_columns(spec)
    sources = numeric + [str(spec["column"]) for spec in one_hot]
    return {
        "version": SCHEMA_VERSION,
        "numeric": numeric,
        "one_hot": one_hot,
        "dropped_high_cardinality": too_wide,
        "dropped_unsupported_dtype": unsupported,
        "columns": columns,
        "max_cardinality": MAX_ONEHOT_CARDINALITY,
        # 배치의 일부가 아니다: 그것을 재생할 코드. ``environment_drift``를 보라.
        "environment": current_environment(),
        # 이것도 배치의 일부가 아니다: 열이 담고 있던 것. ``column_stats``를 보라.
        "column_stats": column_stats(features, sources),
    }


def _level_columns(spec: Mapping[str, Any]) -> list[str]:
    """원본 열 하나가 되는 인코딩된 열 이름들, 순서대로."""
    column = str(spec["column"])
    names = [f"{column}{LEVEL_SEPARATOR}{level}" for level in spec.get("levels") or []]
    if spec.get("missing_level"):
        names.append(f"{column}{LEVEL_SEPARATOR}{MISSING_LEVEL}")
    return names


def encode_with_schema(
    features: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """있는 배치를 새 행에 적용한다. ``(matrix, drift)``를 돌려준다.

    행렬은 이 프레임이 무엇을 담고 있든 스키마의 열을, 스키마의 순서로 갖는다 — 그것이 요점 전부다.
    이 프레임이 담고 있는 것은 그 대신 보고된다:

    ``extra_columns``     학습 파일에 아예 없던 열. 무시되는데, 그것 없이 적합된 모델이 쓸 수 없기
                          때문이고, 무시됐다는 것을 호출자가 알도록 이름이 불린다.
    ``ignored_at_fit``    학습 파일에 있었고 거기서 스키마가 기록한 이유로 떨어진 열(서로 다른 값이
                          너무 많거나, 아무것도 인코딩할 수 없는 dtype). 위와 갈라 놓은 이유는 다른
                          실수이기 때문이다: 하나는 당신이 더한 열, 다른 하나는 당신이 쓸 수 없었던 열.
    ``unseen_levels``     fit이 본 적 없는 범주. 그 열의 그룹에서 전부 0이고, 그것은 fit이 열을 갖지
                          않은 level에 대해 만들었을 바로 그 행이다 — 그래서 실패가 아니라 발표다.
    ``unmatched_missing`` fit이 NaN을 보지 못한 열의 NaN이라, 넣을 결측 열이 없다. 이것도 전부 0이고
                          한 줄을 쓸 값이 있다: 새 행에 학습 행이 갖지 않던 빈틈이 있다는 뜻이다.
    ``coerced_numeric``   수치 열에 있는, 여기서는 숫자가 아닌 값. imputer를 위해 NaN으로 밀리고,
                          셈된다 — 조용히 텍스트가 된 열은 구분자나 export 버그이고 "전부 결측"으로
                          읽히기 때문이다.
    ``missing_rate_shift`` 결측률이 :data:`MISSING_RATE_SHIFT`를 지나 움직인 열. 인코딩에 틀린 것은
                          없다. 바뀐 것은 추정기의 imputer가 열의 얼마를 발명하고 있는가이고, 어떤
                          너비 검사도 그것을 볼 수 없다.
    ``sentinel_shift``    fit이 기록했고 이 배치는 갖지 않은 관례적 결측 코드, 또는 그 반대. ``-9999``를
                          빈 칸으로 바꾼 배치는 완벽하게 인코딩되고 다른 행에서 예측한다.

    ``environment_changed``와 ``missing_checks``는 행에 대한 것이 아니다 — 라이브러리 버전, 그리고 옛
    스키마가 비교할 것을 싣지 않은 것. **이 dict에 얹혀 오는 이유는 이것이 남의 스키마를 읽는 유일한
    함수이기 때문이고**, 두 번째 호출을 기억해야 하는 호출자는 언젠가 그것을 하지 않는 호출자다.

    스키마가 필요로 하는데 이 프레임에 없는 원본 열은 :class:`FeatureSchemaMismatch`를 낸다 — **한
    번에 전부**, 그래서 한 번 돌리면 고칠 것을 다 말한다.
    """
    import pandas as pd

    version = int(schema.get("version") or 0)
    if not MIN_SCHEMA_VERSION <= version <= SCHEMA_VERSION:
        raise FeatureSchemaMismatch(
            f"feature schema version {version} cannot be read by this build "
            f"(it reads {MIN_SCHEMA_VERSION}..{SCHEMA_VERSION}). "
            + (
                "This schema was written by a newer build; guessing at a layout it describes "
                "and this one does not is how a model gets scored on misaligned columns."
                if version > SCHEMA_VERSION
                else "Re-run training to write a current schema."
            )
        )
    numeric = [str(name) for name in schema.get("numeric") or []]
    one_hot = [dict(spec) for spec in schema.get("one_hot") or []]
    required = numeric + [str(spec["column"]) for spec in one_hot]
    have = {str(name) for name in features.columns}
    absent = [name for name in required if name not in have]
    if absent:
        raise FeatureSchemaMismatch(
            f"{len(absent)} feature column(s) the fitted model needs are not in this file: "
            + ", ".join(repr(name) for name in absent[:20])
            + (f" (+{len(absent) - 20} more)" if len(absent) > 20 else "")
        )

    dropped_at_fit = {
        str(name)
        for key in ("dropped_high_cardinality", "dropped_unsupported_dtype")
        for name in schema.get(key) or []
    }
    seen_at_fit = set(required) | dropped_at_fit
    drift: dict[str, Any] = {
        "rows": int(len(features)),
        "extra_columns": [str(name) for name in features.columns if str(name) not in seen_at_fit],
        "ignored_at_fit": [str(name) for name in features.columns if str(name) in dropped_at_fit],
        "unseen_levels": [],
        "unmatched_missing": [],
        "coerced_numeric": [],
        "missing_rate_shift": [],
        "sentinel_shift": [],
        "schema_version": version,
        "missing_checks": missing_checks(version),
        "environment_changed": environment_drift(schema.get("environment")),
    }
    _compare_column_stats(features, schema.get("column_stats"), required, drift)

    blocks: dict[str, Any] = {}
    for name in numeric:
        series = features[name]
        if is_numeric_column(series):
            blocks[name] = series
            continue
        # 적합 때는 숫자였는데 여기서는 아니다. raise가 아니라 ``coerce``인 이유: 백만 행 export의
        # 파싱 안 되는 셀 하나는 거절이 아니라 결측이 되어야 한다. 그것을 이제 전부 텍스트인 열과
        # 가르는 것이 셈이다.
        converted = pd.to_numeric(series, errors="coerce")
        failed = int((converted.isna() & series.notna()).sum())
        if failed:
            drift["coerced_numeric"].append({"column": name, "rows": failed})
        blocks[name] = converted

    for spec in one_hot:
        name = str(spec["column"])
        series = features[name]
        isna = series.isna()
        # level이 기록된 방식대로 텍스트로 비교한다. 그래서 정수 코드와 같은 코드의 문자열이 둘이
        # 아니라 한 level이다.
        as_object = series.astype("object")
        as_text = as_object.where(isna, as_object.astype(str))
        levels = [str(level) for level in spec.get("levels") or []]
        known = pd.Series(False, index=features.index)
        for level in levels:
            hit = as_text == level
            known = known | hit
            blocks[f"{name}{LEVEL_SEPARATOR}{level}"] = hit.astype("float64")
        if spec.get("missing_level"):
            blocks[f"{name}{LEVEL_SEPARATOR}{MISSING_LEVEL}"] = isna.astype("float64")
        elif bool(isna.any()):
            drift["unmatched_missing"].append({"column": name, "rows": int(isna.sum())})
        stranded = ~known & ~isna
        if bool(stranded.any()):
            drift["unseen_levels"].append(
                {
                    "column": name,
                    "levels": sorted({str(value) for value in as_text[stranded].tolist()}),
                    "rows": int(stranded.sum()),
                }
            )

    order = [str(name) for name in schema.get("columns") or []]
    if sorted(order) != sorted(blocks):
        # 파일이 아니라 스키마가 자기와 어긋난 것이다. 치명적인 이유는 둘이 같은 함수가 쓴 것이기
        # 때문이다: 여기 닿았다면 파일이 손으로 고쳐졌거나 잘렸다.
        raise FeatureSchemaMismatch(
            f"feature schema is internally inconsistent: it lists {len(order)} encoded "
            f"column(s) but its own column specs produce {len(blocks)}"
        )
    encoded = (
        pd.DataFrame(blocks, index=features.index, columns=order)
        if order
        else features.iloc[:, :0]
    )
    return encoded, drift


def encode_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """이 프레임에 배치를 적합하고 적용한다. ``(수치 프레임, report)``.

    자기 파일을 소유한 두 호출자 — 기준선을 재는 프로파일러, 그리고 매 학습 시도 — 를 위한 것으로,
    거기서는 배치를 적합하는 것과 쓰는 것이 한 행위다. 모델이 적합되지 *않은* 행에 점수를 매기는 것은
    :func:`encode_with_schema`와 그 모델의 저장된 스키마를 원한다.

    ``report``는 집계 더하기 열 *이름*이고, 그것은 원본 데이터가 아니다 (카드가 이미 모든 이름과
    dtype과 결측률을 발표한다). 실행이 자기 열 열둘 중 아홉에서 0.71을 냈다고 호출자에게 말하는 것이
    이것이다. 일부러 스키마가 아니다: 이 dict는 카드로 얹혀 들어가고(``privacy.CARD_KEYS``에
    ``encoding``이 있다) 거기서 모든 프롬프트로 가는데, 스키마는 셀 값을 싣는다.
    """
    schema = build_schema(features)
    encoded, _drift = encode_with_schema(features, schema)
    report = {
        "numeric": len(schema["numeric"]),
        "one_hot_columns": len(schema["one_hot"]),
        "one_hot_levels": int(encoded.shape[1]) - len(schema["numeric"]),
        "dropped_high_cardinality": schema["dropped_high_cardinality"],
        "dropped_unsupported_dtype": schema["dropped_unsupported_dtype"],
        "max_cardinality": MAX_ONEHOT_CARDINALITY,
    }
    return encoded, report


# 한 줄에 이름을 몇 개까지 적는지. 나머지는 개수로 접힌다.
SHOWN_NAMES = 5


def _shown(names: list[str]) -> str:
    """이름 목록을 한 줄에 넣을 만한 길이로. 호출자 셋이 같은 표현을 갖고 있었다."""
    more = f" 외 {len(names) - SHOWN_NAMES}개" if len(names) > SHOWN_NAMES else ""
    return ", ".join(names[:SHOWN_NAMES]) + more


def describe_drift(drift: Mapping[str, Any]) -> list[str]:
    """드리프트 보고를 사람을 향한 한글 줄로. 새 행이 맞았으면 빈 목록.

    한 단락이 아니라 발견마다 한 줄인 이유는 이것들이 독립된 사실이고 호출자가 하나에는 행동하고 다른
    하나는 받아들일 수 있기 때문이다. level 이름이 들어간다: 이것은 프롬프트를 렌더하지 않는
    ``scripts/predict.py``에서 돌고, 줄을 행동할 수 있게 만드는 것이 그 이름이다.
    """
    lines: list[str] = []
    for item in drift.get("coerced_numeric") or []:
        lines.append(
            f"수치 컬럼 '{item['column']}'의 {item['rows']}행이 숫자로 읽히지 않아 결측으로 처리했습니다"
        )
    for item in drift.get("unseen_levels") or []:
        lines.append(
            f"'{item['column']}'에 학습 때 없던 범주 {len(item['levels'])}개"
            f"({_shown(item['levels'])}) — {item['rows']}행이 이 컬럼에서 전부 0으로 인코딩됩니다"
        )
    for item in drift.get("unmatched_missing") or []:
        lines.append(
            f"'{item['column']}'의 {item['rows']}행이 결측인데 학습 데이터에는 결측이 없어 "
            "결측 전용 열이 없습니다 — 이 행들도 전부 0입니다"
        )
    for item in drift.get("missing_rate_shift") or []:
        lines.append(
            f"'{item['column']}'의 결측률이 학습 때 {item['fit']:.1%}에서 이 배치 "
            f"{item['now']:.1%}로 바뀌었습니다 — 인코딩은 맞지만 모델이 보는 값의 출처가 "
            "달라졌습니다 (결측 대치를 쓰는 파이프라인이면 그만큼이 학습 때의 대치값입니다)"
        )
    for item in drift.get("sentinel_shift") or []:
        value = f"{item['value']:g}"
        if float(item["now"]) > float(item["fit"]):
            lines.append(
                f"'{item['column']}'에 결측 코드로 의심되는 {value}이 학습 때 "
                f"{item['fit']:.1%}에서 이 배치 {item['now']:.1%}로 늘었습니다 — 이 값은 결측이 "
                f"아니라 숫자 {value}로 모델에 들어갑니다"
            )
        else:
            lines.append(
                f"'{item['column']}'의 결측 코드 {value}가 학습 때 {item['fit']:.1%}였는데 이 "
                f"배치는 {item['now']:.1%}입니다 — 결측 표기 방식이 바뀐 것으로 보입니다. "
                "같은 결측이 학습 때는 극단값으로, 지금은 빈 칸으로 들어가고 있습니다"
            )
    for key, why in (
        ("extra_columns", "학습 파일에 없던 컬럼"),
        ("ignored_at_fit", "학습 때도 인코딩되지 않아 제외된 컬럼"),
    ):
        names = [str(name) for name in drift.get(key) or []]
        if not names:
            continue
        lines.append(f"{why} {len(names)}개는 무시했습니다 ({_shown(names)})")
    lines += _describe_schema_drift(drift)
    return lines


# 줄을 읽는 사람에게 각 검사 이름이 뜻하는 것. ``SCHEMA_CHECKS_ADDED``가 아니라 한글 텍스트 옆에
# 두어서 그 상수가 버전에 대한 사실로 남게.
CHECK_LABELS: dict[str, str] = {
    "environment": "라이브러리 버전 대조",
    "column_stats": "컬럼 결측률·결측 코드 대조",
}


def _describe_schema_drift(drift: Mapping[str, Any]) -> list[str]:
    """행이 아니라 스키마와 인터프리터에 대한 두 발견.

    블록의 마지막인 이유는 이것들이 모든 행에 똑같이 참이기 때문이다: "내 행 중 어느 것이 영향을
    받았나"를 훑는 독자는 그쪽 줄을 먼저 만나야 한다.
    """
    lines: list[str] = []
    changed = list(drift.get("environment_changed") or [])
    if changed:
        shown = ", ".join(f"{item['package']} {item['fit']} → {item['now']}" for item in changed)
        lines.append(
            f"학습 때와 라이브러리 버전이 다릅니다 ({shown}) — 저장된 모델은 pickle이라 "
            "다른 버전에서 열면 예측이 조용히 달라질 수 있습니다. 같은 버전으로 맞추거나, "
            "학습 때의 홀드아웃 점수를 이 배치에서 다시 확인하십시오"
        )
    absent = [str(name) for name in drift.get("missing_checks") or []]
    if absent:
        shown = ", ".join(CHECK_LABELS.get(name, name) for name in absent)
        lines.append(
            f"이 스키마는 version {drift.get('schema_version')}이라 {shown}를 하지 못했습니다 "
            f"(현재 version {SCHEMA_VERSION}) — 위에 없는 항목은 이상이 없다는 뜻이 아니라 "
            "검사하지 않았다는 뜻입니다. 같은 데이터로 다시 학습하면 켜집니다"
        )
    return lines


def describe_encoding(report: dict[str, Any]) -> str:
    """로그나 콘솔용 한 줄. 사람이 읽으므로 한글."""
    line = (
        f"features: 수치 {report.get('numeric')}개"
        f" + 범주형 {report.get('one_hot_columns')}개를 one-hot {report.get('one_hot_levels')}열로"
    )
    for key, why in (
        ("dropped_high_cardinality", f"고유값 {report.get('max_cardinality')}개 초과로"),
        ("dropped_unsupported_dtype", "인코딩 불가 dtype으로"),
    ):
        names = [str(name) for name in report.get(key) or []]
        if not names:
            continue
        line += f"; {why} 제외 {len(names)}개 ({_shown(names)})"
    return line


# --------------------------------------------------------------------------- #
# 결측을 열로
# --------------------------------------------------------------------------- #
#
# 이 둘이 쓰이는 곳인 ``scripts/train.py``가 아니라 *여기* 사는 이유는 하나다:
# ``FunctionTransformer``는 함수를 ``module.qualname``으로 pickle하고, ``scripts/train.py``는 운영에서
# 늘 ``__main__``으로 돈다 (``nodes/training.py``가 스크립트로 띄운다). 거기 정의되면 적합된
# 파이프라인은 ``__main__.append_missing_count``를 기록하는데, 그 이름은 마침 같은 ``__main__``을 가진
# 다른 프로세스 안에서만 해석된다. ``--score-model``이 그렇기 때문에 결함이 잠복해 있었고,
# ``model.joblib``을 로드하는 그 밖의 무엇은 ``AttributeError``를 받는다. 여기 정의되면 기록되는 이름은
# 어디서든 import할 수 있는 ``automl_agent.dataset.features.append_missing_count``다.
# ``test_the_appenders_do_not_live_in_the_script_that_runs_as_main``이 선을 지킨다.
#
# 둘 다 상태가 없어서 출력 너비가 입력 너비의 순수 함수다. 그것이 train/validation 분할 어느 쪽에서도
# 안전하게 만들고, 둘 중 어느 것도 ``MissingIndicator(features="missing-only")``가 아닌 이유다: 그것은
# 열 집합을 적합하므로, validation 행에 NaN이 없는 열이 행렬 너비를 조용히 바꾼다.


def _missing_mask(x: Any) -> Any:
    import numpy as np

    return np.isnan(np.asarray(x, dtype=float))


def append_missing_indicator(x: Any, positions: Any = None) -> Any:
    """선택된 입력 열마다 0/1 열 하나를 뒤에 덧붙인다. 상태가 없어서 이름으로 pickle된다.

    ``positions``는 인코딩된 행렬 안 인덱스로 어느 열이 표시를 받는지 이름 부른다. ``None``은 모든
    열이고, :mod:`automl_agent.dataset.pipeline`이 있기 전의 동작이며 이미 pickle된
    ``FunctionTransformer``가 재생하는 것이다. 두 번째 함수가 아니라 기본값 있는 키워드인 이유는 있는
    ``model.joblib``(이 이름과 ``kw_args=None``을 기록한)이 그대로 로드되고 불리게 하려고.

    상태가 없다는 것이 분할 어느 쪽에서도 안전하게 지킨다: 출력 너비는 입력 너비와 이 인수의 순수
    함수이고, 본 행에서 어느 열이 마침 NaN을 가졌는지의 함수가 결코 아니다. 그래서
    ``MissingIndicator(features="missing-only")``가 아니다.

    손을 뻗기 전에: NaN으로 기본 분기하는 계열(``hist_gbdt``나 ``xgboost``의 ``impute: none``)에 대고는
    *정확히* 중복이다 — 표시가 만드는 분할은 트리가 이미 가진 NaN 가지뿐이다. 이것이 *대치하는* 경로에서
    무엇을 사는지는 ``capabilities._MISSINGNESS``를 먼저 읽으라.
    """
    import numpy as np

    mask = _missing_mask(x).astype(float)
    if positions is not None:
        mask = mask[:, list(positions)]
    return np.hstack([x, mask])


def append_missing_count(x: Any) -> Any:
    """열 하나: 이 행의 필드 중 몇 개가 측정되지 않았는가.

    NaN을 기본으로 다루는 계열이 이미 만드는 NaN 분할로는 표현되지 않는다 — 그쪽은 열별이고 이것은 열을
    건너 집계한다. 임상 행에서는 환자가 얼마나 검사를 받았는지를 대신하고, 그래서 자기 열로 *쓸 수
    있다*. 그것이 한 열의 값을 한다는 것과 같지는 않다.
    """
    import numpy as np

    return np.hstack([x, _missing_mask(x).sum(axis=1, keepdims=True).astype(float)])
