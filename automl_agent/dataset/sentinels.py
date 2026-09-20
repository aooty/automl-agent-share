"""센티넬 코드: "결측"을 뜻하지만 측정치처럼 생겨서 도착하는 값들.

**이것이 틀렸을 때 실패하는 것은 없다. 숫자가 그냥 틀린다**.

**탐지하고, 경고하고, 결코 변환하지 않는다.** 변환은 호출자의 일이다
(``pd.read_csv(na_values=[...])``). **조용히 고쳐 쓰면 카드가 파일에 없는 행을 설명하게 되고**,
카드→실행기 계약이 서 있는 성질이 바로 그것이다.

**발표 정책.** 카드는 셀 값을 실을 수 없고 센티넬은 셀 값*이다*. **탐지는 아래 상수 목록에 있는
코드만 알아본다** — 그래서 발표된 값은 데이터가 아니라 *이 파일*에서 나온 것이다. 데이터가 기여하는
것은 "있다, 이 비율로"뿐이다. 목록에 **없는** 반복된 극단값은 아예 보고되지 않는다.

**게이트.** 코드는 그것이 열 자신의 min 또는 max이고 *동시에* 가장 가까운 다른 값에서 떨어져 있을
때만 보고된다 — :data:`GAP_MULTIPLE` IQR만큼, 또는 그 부호를 다른 어떤 행도 갖지 않을 때는
:data:`SIGN_GAP_MULTIPLE`만큼.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from automl_agent.dataset.features import is_numeric_column, is_text_like_column

if TYPE_CHECKING:  # pragma: no cover - import 비용만이고, pandas는 subprocess 전용이다
    import pandas as pd

# 발견의 ``kind``. 소비자가 셋(:mod:`automl_agent.dataset.caveats`,
# :mod:`automl_agent.dataset.features`, 아래의 콘솔 경고)이고, 오타는 실패하는 대신 발견 0개로
# 조용히 읽힌다 — 이 모듈이 경고하는 바로 그 실패 양식이다.
KIND_NUMERIC_CODE = "numeric_code"
KIND_TEXT_MARKER = "text_marker"

# 관례적인 숫자 결측 코드. 넓은 것부터 — 그래야 ``-9999``/``-999`` 짝이 읽는 사람이 기대하는
# 순서로 보고된다. 이 목록에 있는 값만이 카드에 닿을 수 있다.
NUMERIC_CODES: tuple[float, ...] = (
    -9999999.0,
    -999999.0,
    -99999.0,
    -9999.0,
    -999.0,
    -99.0,
    -9.0,
    -1.0,
    9999999.0,
    999999.0,
    99999.0,
    9999.0,
    999.0,
    99.0,
)

# 범주 열이 같은 목적으로 쓰는 문자열. casefold하고 strip해서 비교하므로 ``" N/A "``도 걸린다.
TEXT_MARKERS: tuple[str, ...] = (
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "nil",
    "unknown",
    "unk",
    "missing",
    "not available",
    "not recorded",
    "?",
    "-",
    ".",
)

# 이보다 행이 적으면 간격 검사가 분포가 아니라 잡음을 재고 있다.
MIN_ROWS = 20
# 플래그나 두 수준 코드에는 말할 만한 "고립된 극단"이 없고, 그 낮은 값이 으레 진짜 수준으로서의
# -1이나 99다.
MIN_DISTINCT = 5
# 극단값이 한 번 나온 것은 코드보다 특이한 레코드 하나일 가능성이 높다 — 그리고 그것을 보고하는
# 것은 그 레코드를 보고하는 것이다.
MIN_COUNT = 2
# 코드가 사분위 범위 밖으로 얼마나 떨어져 앉아야 하는지.
GAP_MULTIPLE = 3.0
# 열의 다른 어떤 행도 갖지 않은 *부호*를 가진 코드에 적용되는 완화된 바. 완화를 정당화하는 것은
# 거리가 아니라 불가능성이다. 간격은 여전히 요구된다 — 그것이 진짜
# ``-1 … 5`` 평가 척도를 조용하게 둔다.
SIGN_GAP_MULTIPLE = 1.0


def _finding(kind: str, value: Any, count: int, n_rows: int) -> dict[str, Any]:
    """발견 하나. ``rate``는 NaN 셀까지 포함한 열 전체에 대한 비율이라는 계약이 여기 한 곳에 산다."""
    return {"kind": kind, "value": value, "rate": round(count / n_rows, 4)}


def detect_sentinels(series: pd.Series) -> list[dict[str, Any]]:
    """``series``에서 "결측"을 뜻하는 것처럼 보이는 코드들. 아무것도 변환하지 않는다.

    각 발견은 ``{"kind", "value", "rate"}``이고, ``rate``가 열 전체에 대한 비율이므로 변환 없이도
    ``missing_rate`` 옆에서 읽힌다.
    """
    import pandas as pd

    n_rows = int(len(series))
    if n_rows < MIN_ROWS:
        return []
    if is_text_like_column(series):
        return _text_findings(series, n_rows)
    if not is_numeric_column(series) or bool(pd.api.types.is_bool_dtype(series)):
        return []
    return _numeric_findings(series, n_rows)


def _text_findings(series: pd.Series, n_rows: int) -> list[dict[str, Any]]:
    counts = series.dropna().astype(str).str.strip().str.casefold().value_counts()
    findings: list[dict[str, Any]] = []
    for marker in TEXT_MARKERS:
        count = int(counts.get(marker, 0))
        if count >= MIN_COUNT:
            findings.append(_finding(KIND_TEXT_MARKER, marker, count, n_rows))
    return findings


def _numeric_findings(series: pd.Series, n_rows: int) -> list[dict[str, Any]]:
    import numpy as np

    present = series.dropna()
    if len(present) < MIN_ROWS:
        return []
    values = present.to_numpy(dtype="float64", copy=False)
    uniques, counts = np.unique(values, return_counts=True)
    frequency = dict(zip(uniques.tolist(), counts.tolist(), strict=True))

    findings: list[dict[str, Any]] = []
    excluded: list[float] = []
    working = uniques
    # 한 번 훑는 대신 반복한다 — ``-999`` 옆에 앉은 ``-9999``가 그것을 가리지 않게. 바깥 코드가
    # 셈에 들어가면 다음 것이 열의 극단이 된다.
    while len(working) >= MIN_DISTINCT:
        position = _next_code(working, values, frequency, excluded)
        if position is None:
            break
        code = float(working[position])
        findings.append(_finding(KIND_NUMERIC_CODE, code, frequency[code], n_rows))
        excluded.append(code)
        working = np.delete(working, position)
    # 넓은 코드부터, 그리고 열의 양쪽 끝이 예측 가능하게 묶이도록.
    findings.sort(key=lambda item: NUMERIC_CODES.index(item["value"]))
    return findings


def _next_code(
    working: Any, values: Any, frequency: dict[float, int], excluded: list[float]
) -> int | None:
    """코드 자격이 되는 다음 극단의 ``working`` 안 인덱스, 없으면 None."""
    import numpy as np

    for position in (0, len(working) - 1):
        code = float(working[position])
        if code not in NUMERIC_CODES or frequency.get(code, 0) < MIN_COUNT:
            continue
        rest = values[~np.isin(values, [*excluded, code])]
        if len(rest) < MIN_ROWS:
            continue
        q1, q3 = (float(x) for x in np.percentile(rest, [25, 75]))
        # 가운데 절반이 한 값일 때는 전체 범위로 물러난다. 그것은 포기할 이유가 아니라
        # 몰려 있지만 상수는 아닌 열이다.
        scale = q3 - q1 or float(rest.max() - rest.min())
        if scale <= 0:
            continue
        neighbour = float(working[1] if position == 0 else working[-2])
        gap = abs(neighbour - code)
        if gap >= GAP_MULTIPLE * scale:
            return position
        # 열의 나머지가 갖지 않은 부호: 그저 먼 것이 아니라 불가능하다.
        one_sided = code < 0 <= float(rest.min()) or code > 0 >= float(rest.max())
        if one_sided and gap >= SIGN_GAP_MULTIPLE * scale:
            return position
    return None


def describe_sentinels(by_column: dict[str, list[dict[str, Any]]]) -> str:
    """콘솔 경고. 할 말이 없으면 빈 문자열이라, 호출자가 조건 없이 print해도 된다."""
    if not by_column:
        return ""
    lines = ["경고: 결측 코드로 의심되는 값이 있습니다 — 자동 변환하지 않습니다."]
    for name, findings in by_column.items():
        for item in findings:
            if item["kind"] == KIND_NUMERIC_CODE:
                why = f"분포에서 {GAP_MULTIPLE:g}×IQR 이상 떨어진 관례적 결측 코드"
                shown = f"{item['value']:g}"
            else:
                why = "결측 표기로 자주 쓰이는 값"
                shown = repr(item["value"])
            lines.append(f"  {name}: {shown} 이 {item['rate'] * 100:.1f}% ({why})")
    lines.append(
        "  결측이 맞다면 pd.read_csv(..., na_values=[...])로 바꿔 저장한 뒤 profile을 다시 "
        "돌리세요. 그대로 두면 이 카드의 magnitude·skew·outlier_rate·target_corr와 아래 "
        "기준선이 모두 이 값을 실제 측정치로 셉니다."
    )
    return "\n".join(lines)
