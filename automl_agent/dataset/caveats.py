"""카드의 "집계가 보여 주지 않는, 이 데이터에 대한 것들" 채널.

출처는 둘이고, 같은 목록에 얹힌다:

- **프로파일러 자신의 검사.** 지금은 센티넬 코드 (:mod:`automl_agent.dataset.sentinels`).
- **운영자.** ``--caveat "..."``, 반복 가능. 시스템에서 *사람*의 원본 데이터 지식을 프롬프트로
  나르는 유일한 채널이다.

**운영자의 텍스트는 적힌 그대로 프롬프트에 닿는다 — 셀 값이 손으로 거기 갈 수 있는 유일한 자리다.**
의도한 것이다. **아래 상한은 내용이 아니라 길이와 개수에 걸린다** — 지키는
것은 프롬프트 예산이고 비공개가 아니다.
"""

from __future__ import annotations

from typing import Any

from .sentinels import KIND_NUMERIC_CODE

# 카드 키. :mod:`automl_agent.privacy`의 allowlist, 프로파일러, 노드, 프롬프트가 모두 이것을
# 참조하므로 다섯 군데에 적는 대신 여기서 이름을 갖는다.
CAVEATS_KEY = "caveats"

# 이 텍스트는 매 반복의 모든 추론 프롬프트에 렌더되므로 상한이 있다.
MAX_CAVEATS = 20
MAX_CAVEAT_CHARS = 400


def sentinel_caveats(by_column: dict[str, list[dict[str, Any]]]) -> list[str]:
    """의심되는 결측 코드마다 주의사항 하나. 그것으로 계획하는 독자를 향해 쓴다.

    양쪽 절반을 다 말한다: 어느 값이 의심스러운지, 그리고 그 열의 통계 — 카드가 두 블록 위에
    발표하는 그것 — 가 그 값을 그대로 둔 채 측정됐다는 사실.
    """
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
    """열이 아니라 *분할*에 대한 유일한 주의사항.

    그룹 분할은 ``baseline.protocol.grouped_by``에도 발표되지만, 여기 한 번 더 적는다.
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
    """평평하게, 다듬고, 중복을 걷고, 상한을 건다. 순서는 유지된다: 기계 먼저, 사람 나중.

    중복 판정은 글자 그대로의 일치다. 있는 이유가 되는 경우 — 같은 ``--caveat``을 두 번 주거나,
    운영자가 프로파일러가 이미 찾은 것을 되풀이하는 것 — 에는 그것으로 충분하다.
    """
    merged: list[str] = []
    for group in groups:
        for item in group or ():
            text = " ".join(str(item).split())[:MAX_CAVEAT_CHARS]
            if text and text not in merged:
                merged.append(text)
    return merged[:MAX_CAVEATS]


def card_caveats(card: dict[str, Any] | None) -> list[str]:
    """카드의 주의사항 목록. 손으로 쓴 카드가 맨 문자열을 쓴 경우도 받아 준다."""
    raw = (card or {}).get(CAVEATS_KEY)
    if isinstance(raw, str):
        return merge_caveats([raw])
    if isinstance(raw, (list, tuple)):
        return merge_caveats(list(raw))
    return []


def describe_caveats(card: dict[str, Any] | None) -> str:
    """프롬프트 블록. 결코 비지 않는다 — 빈 절이 "알려진 것 없음"으로 읽히지 않게."""
    items = card_caveats(card)
    if not items:
        return "(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)"
    return "\n".join(f"- {item}" for item in items)
