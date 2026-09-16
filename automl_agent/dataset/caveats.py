"""The card's channel for "things about this data the aggregates do not show".

The card's ``missing`` block is three numbers and the per-column profiles are buckets, so everything
else a human learns from the file has nowhere to go — and :data:`CARD_KEYS` is a strict allowlist,
leaving no slot even by hand. What that costs: missingness can be the strongest signal in a table
*and* entangled with row order, so a plan that leans on it partly learns where in the file a row sat,
and nothing in the card could say so (``FINDINGS-mimic.md``).

Two sources, both landing in the same list:

- **The profiler's own checks.** Sentinel codes today (:mod:`automl_agent.dataset.sentinels`).
  These are worth putting in the LLM's field of view precisely because the profiler cannot
  act on them: it will not convert a code, so the aggregates below it are measured with the
  code counted as a measurement, and a plan that leans on that column is leaning on a
  number nobody should trust.
- **The operator.** ``--caveat "..."``, repeatable. This is the only channel in the system
  that carries a *human's* knowledge of the raw data into the prompts.

**The operator's text reaches a prompt as written, so it is the one place a cell value can get there
by hand.** Deliberate, and the same shape as ``--name``: what this repo enforces mechanically is that
no *code path* carries rows into a prompt. **The bound below is on length and count, not content** —
it protects the prompt budget, not privacy.
"""

from __future__ import annotations

from typing import Any

# The card key. Named here rather than spelled in five places, since the allowlist in
# :mod:`automl_agent.privacy`, the profiler, the nodes and the prompts all reference it.
CAVEATS_KEY = "caveats"

# Bounds, because this text is rendered into every reasoning prompt of every iteration. A
# caveat that does not fit in a couple of sentences is a document, and belongs next to the
# card rather than inside it.
MAX_CAVEATS = 20
MAX_CAVEAT_CHARS = 400


def sentinel_caveats(by_column: dict[str, list[dict[str, Any]]]) -> list[str]:
    """One caveat per suspected missing-code, phrased for a reader who plans with it.

    Says both halves: which value is suspect, and that the column's own statistics — the
    ones the card publishes two blocks up — were measured with it left in place.
    """
    lines: list[str] = []
    for name, findings in by_column.items():
        for item in findings:
            if item.get("kind") == "numeric_code":
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
    """The one caveat about the *split* rather than about a column.

    Grouping is published in ``baseline.protocol.grouped_by``, but a card built with
    ``--no-baseline`` has no protocol block at all, and even when it does the fact is buried
    in a nested dict a reasoning prompt may not read closely. It changes what every score in
    the run means — the numbers are out-of-group, so they are lower than a row-level split
    would report and they are the honest ones — so it says that in the section the prompts
    are told to treat as constraints.
    """
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


def merge_caveats(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    """Flatten, trim, de-duplicate and bound. Order is preserved: machine first, then human.

    De-duplication is by exact text, which is enough for the case it exists for — the same
    ``--caveat`` passed twice, or an operator repeating what the profiler already found.
    """
    merged: list[str] = []
    for group in groups:
        for item in group or ():
            text = " ".join(str(item).split())[:MAX_CAVEAT_CHARS]
            if text and text not in merged:
                merged.append(text)
    return merged[:MAX_CAVEATS]


def card_caveats(card: dict[str, Any] | None) -> list[str]:
    """The caveat list of a card, tolerant of a hand-written one that used a bare string."""
    raw = (card or {}).get(CAVEATS_KEY)
    if isinstance(raw, str):
        return merge_caveats([raw])
    if isinstance(raw, (list, tuple)):
        return merge_caveats(list(raw))
    return []


def describe_caveats(card: dict[str, Any] | None) -> str:
    """The prompt block. Never empty, so a missing section cannot read as "none known"."""
    items = card_caveats(card)
    if not items:
        return "(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)"
    return "\n".join(f"- {item}" for item in items)
