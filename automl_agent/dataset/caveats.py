"""The card's channel for facts the aggregates do not show.

Roles:

* Caveat sources — caveat lines from the profiler's own checks.
* Caveat list — merge, clean, and cap the card's caveats.
* Prompt rendering — turn the card's caveats into a prompt block.
"""

from __future__ import annotations

from typing import Any

from .sentinels import KIND_NUMERIC_CODE

# Shared by privacy allowlist, profiler, nodes, and prompts.
CAVEATS_KEY = "caveats"

# Capped: shown in every prompt
MAX_CAVEATS = 20
MAX_CAVEAT_CHARS = 400


# --- Role: caveat sources ---------------------------------------------------------


def sentinel_caveats(by_column: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Write one caveat per suspected missing-value code.

    Each says the card stats and baseline counted that value as real."""
    lines: list[str] = []
    for name, findings in by_column.items():
        for item in findings:
            if item.get("kind") == KIND_NUMERIC_CODE:
                shown = f"{float(item['value']):g}"
                what = "결측 코드로 의심되는 값"
            else:
                shown = repr(item.get("value"))
                what = "결측 표기로 자주 쓰이는 범주값"
            rate = float(item.get("rate") or 0.0) * 100
            lines.append(
                f"{name}: {what} {shown} 이 {rate:.1f}%. 변환하지 않았으므로 이 열의 "
                "magnitude·skew·outlier_rate·target_corr와 기준선 점수는 이 값을 실제 "
                "측정치로 계산한 결과입니다. 이 열에 의존하는 계획은 그 점을 감안하세요."
            )
    return lines


def grouping_caveats(group_column: str | None, n_groups: int | None = None) -> list[str]:
    """Write the grouped-split caveat; empty list when there is no group column.

    Repeats ``baseline.protocol.grouped_by`` on purpose"""
    if not group_column:
        return []
    counted = f" ({n_groups}개 그룹)" if n_groups else ""
    return [
        f"split은 {group_column} 단위로 나눴습니다{counted} — 같은 {group_column}의 행은 "
        "train/val/test 중 한 곳에만 들어갑니다. 따라서 모든 점수는 처음 보는 "
        f"{group_column}에 대한 성능이며, 행 단위로 무작위 분할했을 때보다 낮게 나오는 것이 "
        f"정상입니다. {group_column} 자체는 특성에서 제외됐으니 그것을 쓰는 계획은 세울 수 "
        "없습니다."
    ]


# --- Role: caveat list ------------------------------------------------------------


def merge_caveats(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    """Merge caveat groups into one clean, capped list, keeping order.

    Drops empty and exact duplicates. Pass machine caveats first, human ones after."""
    merged: list[str] = []
    for group in groups:
        for item in group or ():
            text = " ".join(str(item).split())[:MAX_CAVEAT_CHARS]
            if text and text not in merged:
                merged.append(text)
    return merged[:MAX_CAVEATS]


def card_caveats(card: dict[str, Any] | None) -> list[str]:
    """Read the cleaned caveat list from a card; a bare string also works."""
    raw = (card or {}).get(CAVEATS_KEY)
    if isinstance(raw, str):
        return merge_caveats([raw])
    if isinstance(raw, (list, tuple)):
        return merge_caveats(list(raw))
    return []


# --- Role: prompt rendering -------------------------------------------------------


def describe_caveats(card: dict[str, Any] | None) -> str:
    """Render the card's caveats as bullet lines for a prompt.

    Never empty: says "none recorded", so silence is not read as "nothing known"."""
    items = card_caveats(card)
    if not items:
        return "(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)"
    return "\n".join(f"- {item}" for item in items)
