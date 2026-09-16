"""Sentinel codes: values that mean "missing" but arrive looking like measurements.

**Nothing fails when this goes wrong; the numbers are simply wrong.** An extract with zero NaN and
tens of thousands of ``-9999`` cells makes every aggregate the card publishes — magnitude, skew,
outlier rate, target correlation — and the baseline the bar is *derived from* count that code as a
measurement. The run then chases a bar built on it (``FINDINGS-mimic.md``).

**Detect, warn, never convert.** Conversion is the caller's (``pd.read_csv(na_values=[...])``): it is
not reliably inferable — ``-1`` is a code in "days since discharge" and data in "temperature delta" —
and **a silent rewrite would mean the card describes rows the file does not contain**, which is the
one property the card→executor contract rests on.

Emission policy
---------------
A card may not carry cell values, and a sentinel *is* one. **What keeps this in policy: detection only
ever recognises codes from the constant lists below**, so the published value came out of *this file*,
not out of the data. The data contributes only "present, at this rate" — a column aggregate like
``missing_rate``. A repeated extreme **not** on the list is not reported at all.

Gates
-----
``-1`` and ``99`` are on the numeric list, which alone would flag half the columns in a normal table.
**A code is reported only when it is the column's own min or max *and* isolated from the nearest other
value** — by :data:`GAP_MULTIPLE` IQRs, or by :data:`SIGN_GAP_MULTIPLE` when its sign is one no other
row has. So ``-1`` among ``-3 … 5`` stays silent and ``-1`` among ages ``40 … 69`` is a code. Ages
``0 … 99`` with ``99`` meaning "unknown" is **not catchable and that is correct** — ``98`` sits right
next to it, so nothing in the distribution distinguishes the two readings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from automl_agent.dataset.features import is_numeric_column, is_text_like_column

if TYPE_CHECKING:  # pragma: no cover - import-time cost only, and pandas is subprocess-only
    import pandas as pd

# Conventional numeric missing codes, widest first so a ``-9999``/``-999`` pair is reported
# in the order a reader expects. Only values on this list can ever reach a card.
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

# Strings that a categorical column uses for the same purpose. Less destructive than a
# numeric code — the encoder gives them their own one-hot level rather than folding them
# into a mean — but they still split what is one fact ("absent") across two
# representations, and they are invisible in ``missing_rate``. Compared casefolded and
# stripped, so ``" N/A "`` matches. ``none`` is deliberately included even though it is a
# legitimate category on, say, a medication column: the point is to ask, not to decide.
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

# Below this many rows the gap test is measuring noise, not a distribution.
MIN_ROWS = 20
# A flag or a two-level code has no "isolated extreme" to speak of, and its low value is
# routinely -1 or 99 as a genuine level.
MIN_DISTINCT = 5
# One occurrence of an extreme value is more likely to be one unusual record than a code —
# and reporting it would be reporting that record.
MIN_COUNT = 2
# How far outside the interquartile range a code has to sit. 3.0 is deliberately well past
# the 1.5 the outlier rate already uses: a heavy tail should not read as a sentinel.
GAP_MULTIPLE = 3.0
# The relaxed bar for a code whose *sign* no other row in the column has. ``-1`` among ages
# ``40 … 69`` is 2.7 IQRs below the nearest age — under the bar above, and obviously a code.
# What licenses the relaxation is not the distance but the impossibility: a column where
# every other value is non-negative has no room for a negative measurement. The gap is still
# required, and it is what keeps a genuine ``-1 … 5`` rating scale quiet, since there ``-1``
# is one step from the rest rather than an interquartile range away.
SIGN_GAP_MULTIPLE = 1.0


def detect_sentinels(series: pd.Series) -> list[dict[str, Any]]:
    """Codes in ``series`` that look like they mean "missing". Never converts anything.

    Each finding is ``{"kind", "value", "rate"}``, where ``rate`` is over the whole column
    including its NaN cells, so it reads next to ``missing_rate`` without conversion.
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
            findings.append(
                {"kind": "text_marker", "value": marker, "rate": round(count / n_rows, 4)}
            )
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
    # Loop rather than a single pass so a ``-9999`` sitting next to a ``-999`` does not hide
    # it: once the outer code is accounted for, the next one becomes the column's extreme.
    while len(working) >= MIN_DISTINCT:
        position = _next_code(working, values, frequency, excluded)
        if position is None:
            break
        code = float(working[position])
        findings.append(
            {
                "kind": "numeric_code",
                "value": code,
                "rate": round(frequency[code] / n_rows, 4),
            }
        )
        excluded.append(code)
        working = np.delete(working, position)
    # Widest code first, and the two ends of the column grouped predictably.
    findings.sort(key=lambda item: NUMERIC_CODES.index(item["value"]))
    return findings


def _next_code(
    working: Any, values: Any, frequency: dict[float, int], excluded: list[float]
) -> int | None:
    """Index in ``working`` of the next extreme that qualifies as a code, or None."""
    import numpy as np

    for position in (0, len(working) - 1):
        code = float(working[position])
        if code not in NUMERIC_CODES or frequency.get(code, 0) < MIN_COUNT:
            continue
        rest = values[~np.isin(values, [*excluded, code])]
        if len(rest) < MIN_ROWS:
            continue
        q1, q3 = (float(x) for x in np.percentile(rest, [25, 75]))
        # Falls back to the full range when the middle half is a single value, which is a
        # concentrated-but-not-constant column rather than a reason to give up.
        scale = q3 - q1 or float(rest.max() - rest.min())
        if scale <= 0:
            continue
        neighbour = float(working[1] if position == 0 else working[-2])
        gap = abs(neighbour - code)
        if gap >= GAP_MULTIPLE * scale:
            return position
        # A sign the rest of the column does not have: not merely far, but impossible.
        one_sided = code < 0 <= float(rest.min()) or code > 0 >= float(rest.max())
        if one_sided and gap >= SIGN_GAP_MULTIPLE * scale:
            return position
    return None


def describe_sentinels(by_column: dict[str, list[dict[str, Any]]]) -> str:
    """The console warning. Korean, because the person who has to decide reads it.

    Empty string when there is nothing to say, so the caller can print unconditionally.
    """
    if not by_column:
        return ""
    lines = ["경고: 결측 코드로 의심되는 값이 있습니다 — 자동 변환하지 않습니다."]
    for name, findings in by_column.items():
        for item in findings:
            if item["kind"] == "numeric_code":
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
