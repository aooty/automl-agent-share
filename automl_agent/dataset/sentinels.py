"""Sentinel codes: values that mean "missing" but look like measurements.

Roles:

* Detection — find suspected missing-value codes in one column; convert nothing.
* Console warning — describe the findings for the profiler's user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from automl_agent.dataset.features import is_numeric_column, is_text_like_column

if TYPE_CHECKING:  # pragma: no cover - types only; pandas is loaded only in subprocesses
    import pandas as pd

# Shared names: a typo would silently mean zero findings.
KIND_NUMERIC_CODE = "numeric_code"
KIND_TEXT_MARKER = "text_marker"

# Widest first. Only these values can reach the card.
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

# Matched after strip and casefold
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

# Fewer rows: the gap test measures noise.
MIN_ROWS = 20
# Few-level columns: a -1 or 99 is a real level.
MIN_DISTINCT = 5
# Seen once: one odd record, not a code.
MIN_COUNT = 2
# Gap from nearest value, in IQRs
GAP_MULTIPLE = 3.0
# Smaller gap when no other row has that sign
SIGN_GAP_MULTIPLE = 1.0


# --- Role: detection --------------------------------------------------------------


def _finding(kind: str, value: Any, count: int, n_rows: int) -> dict[str, Any]:
    """_finding | Detection: build one finding; ``rate`` is over all rows, NaN included."""
    return {"kind": kind, "value": value, "rate": round(count / n_rows, 4)}


def detect_sentinels(series: pd.Series) -> list[dict[str, Any]]:
    """Find values in one column that look like "missing"; change nothing.

    Returns ``{"kind", "value", "rate"}`` findings; empty for short, boolean, or other columns."""
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
    """_text_findings | Detection: count the text markers in a text column."""
    counts = series.dropna().astype(str).str.strip().str.casefold().value_counts()
    findings: list[dict[str, Any]] = []
    for marker in TEXT_MARKERS:
        count = int(counts.get(marker, 0))
        if count >= MIN_COUNT:
            findings.append(_finding(KIND_TEXT_MARKER, marker, count, n_rows))
    return findings


def _numeric_findings(series: pd.Series, n_rows: int) -> list[dict[str, Any]]:
    """_numeric_findings | Detection: find the numeric codes at the ends of a column."""
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
    # Loop: removing -9999 lets the -999 behind it show.
    while len(working) >= MIN_DISTINCT:
        position = _next_code(working, values, frequency, excluded)
        if position is None:
            break
        code = float(working[position])
        findings.append(_finding(KIND_NUMERIC_CODE, code, frequency[code], n_rows))
        excluded.append(code)
        working = np.delete(working, position)
    # NUMERIC_CODES order: widest first, stable for both ends.
    findings.sort(key=lambda item: NUMERIC_CODES.index(item["value"]))
    return findings


def _next_code(
    working: Any, values: Any, frequency: dict[float, int], excluded: list[float]
) -> int | None:
    """_next_code | Detection: index of the next extreme that is a code, or None."""
    import numpy as np

    for position in (0, len(working) - 1):
        code = float(working[position])
        if code not in NUMERIC_CODES or frequency.get(code, 0) < MIN_COUNT:
            continue
        rest = values[~np.isin(values, [*excluded, code])]
        if len(rest) < MIN_ROWS:
            continue
        q1, q3 = (float(x) for x in np.percentile(rest, [25, 75]))
        # Zero IQR: bunched, not constant, so use full range.
        scale = q3 - q1 or float(rest.max() - rest.min())
        if scale <= 0:
            continue
        neighbour = float(working[1] if position == 0 else working[-2])
        gap = abs(neighbour - code)
        if gap >= GAP_MULTIPLE * scale:
            return position
        # A sign no other value has: impossible, not just far.
        one_sided = code < 0 <= float(rest.min()) or code > 0 >= float(rest.max())
        if one_sided and gap >= SIGN_GAP_MULTIPLE * scale:
            return position
    return None


# --- Role: console warning --------------------------------------------------------


def describe_sentinels(by_column: dict[str, list[dict[str, Any]]]) -> str:
    """Build the console warning for the findings; empty string when there are none."""
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
