"""Split rules: which rows train, which rows tune, and which rows nobody touches.

Roles:

* Split constants — fractions and fold counts of the split.
* Protocol record — describe the split for cards and logs.
* Row counts — predict how many rows each set gets.
* Splitting — cut rows into train, validation, and test.
* Split checks — fingerprint val rows, compare card protocols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# --- Role: split constants --------------------------------------------------------

# Share of all rows, scored once at the end.
TEST_FRACTION = 0.2

# Share of the pool after the test cut (0.2 overall).
VAL_FRACTION_OF_POOL = 0.25

TRAIN_SHARE = round((1 - TEST_FRACTION) * (1 - VAL_FRACTION_OF_POOL), 4)  # 0.6
VAL_SHARE = round((1 - TEST_FRACTION) * VAL_FRACTION_OF_POOL, 4)  # 0.2

# Grouped path: one of ``k`` folds keeps ``1/k``.
TEST_FOLDS = round(1 / TEST_FRACTION)  # 5
VAL_FOLDS = round(1 / VAL_FRACTION_OF_POOL)  # 4


@dataclass(frozen=True)
class Splits:
    """Three row sets that do not overlap, read by attribute name.

    ``groups_*`` are ``None`` on the row path.
    """

    x_train: Any
    y_train: Any
    x_val: Any
    y_val: Any
    x_test: Any
    y_test: Any
    groups_train: Any = None
    groups_val: Any = None
    groups_test: Any = None

    @property
    def sizes(self) -> dict[str, int]:
        """sizes | Role: row count of each set: ``{"train", "val", "test"}``."""
        return {
            "train": int(len(self.y_train)),
            "val": int(len(self.y_val)),
            "test": int(len(self.y_test)),
        }


# --- Role: protocol record --------------------------------------------------------


def protocol(seed: int, group_column: str | None = None, stratified: bool = True) -> dict[str, Any]:
    """protocol | Role: the split as a card field, enough to rebuild the row sets."""
    return {
        "train_fraction": TRAIN_SHARE,
        "val_fraction": VAL_SHARE,
        "test_fraction": TEST_FRACTION,
        "stratified": bool(stratified),
        "grouped_by": str(group_column) if group_column else None,
        "seed": int(seed),
    }


def describe_protocol(declared: dict[str, Any] | None = None, seed: int = 42) -> str:
    """describe_protocol | Role: one-line split summary; ``declared`` or the default."""
    block = declared or protocol(seed)
    grouped = block.get("grouped_by")
    grouping = f", grouped_by={grouped}" if grouped else ""
    stratifying = "stratified" if block.get("stratified", True) else "not stratified"
    return (
        f"split: train {float(block.get('train_fraction', TRAIN_SHARE)):.0%}"
        f" / val {float(block.get('val_fraction', VAL_SHARE)):.0%}"
        f" / test {float(block.get('test_fraction', TEST_FRACTION)):.0%}"
        f" ({stratifying}{grouping}, seed={block.get('seed', seed)}) —"
        " test는 반복 밖에서 저장된 모델로 1회만 채점합니다"
    )


# --- Role: row counts -------------------------------------------------------------


def row_counts(n_rows: int) -> dict[str, int]:
    """row_counts | Role: rows per set, rounded like the split; close only when grouped.

    Raises ``ValueError`` below 3 rows.
    """
    total = int(n_rows)
    n_test = math.ceil(TEST_FRACTION * total) if total > 0 else 0
    pool = total - n_test
    n_val = math.ceil(VAL_FRACTION_OF_POOL * pool) if pool > 0 else 0
    if pool - n_val <= 0:
        raise ValueError(
            f"3행 미만은 train/val/test로 나눌 수 없습니다 — n_rows={n_rows}, "
            f"train={pool - n_val}"
        )
    return {"train": pool - n_val, "val": n_val, "test": n_test}


# --- Role: splitting --------------------------------------------------------------


def split_three_way(
    x_arr: Any, y_arr: Any, seed: int, groups: Any = None, stratify: bool = True
) -> Splits:
    """split_three_way | Role: cut test rows first, then val rows from the rest.

    Raises ``ValueError`` on bad ``groups`` length or too few groups.
    """
    if groups is None:
        from sklearn.model_selection import train_test_split

        x_pool, x_test, y_pool, y_test = train_test_split(
            x_arr, y_arr, test_size=TEST_FRACTION, random_state=seed,
            stratify=y_arr if stratify else None,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_pool, y_pool, test_size=VAL_FRACTION_OF_POOL, random_state=seed,
            stratify=y_pool if stratify else None,
        )
        return Splits(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

    import numpy as np

    group_arr = np.asarray(groups)
    if len(group_arr) != len(y_arr):
        raise ValueError(
            f"group 배열의 길이가 행 수와 다릅니다 — groups={len(group_arr)}, rows={len(y_arr)}"
        )

    pool_index, test_index = _grouped_fold(x_arr, y_arr, group_arr, TEST_FOLDS, seed, stratify)
    x_pool, y_pool, pool_groups = x_arr[pool_index], y_arr[pool_index], group_arr[pool_index]
    train_index, val_index = _grouped_fold(x_pool, y_pool, pool_groups, VAL_FOLDS, seed, stratify)
    return Splits(
        x_train=x_pool[train_index],
        y_train=y_pool[train_index],
        x_val=x_pool[val_index],
        y_val=y_pool[val_index],
        x_test=x_arr[test_index],
        y_test=y_arr[test_index],
        groups_train=pool_groups[train_index],
        groups_val=pool_groups[val_index],
        groups_test=group_arr[test_index],
    )


def _grouped_fold(
    x_arr: Any, y_arr: Any, groups: Any, folds: int, seed: int, stratify: bool = True
) -> tuple[Any, Any]:
    """_grouped_fold | Role: ``(rest, one fold)`` indices without splitting a group."""
    import numpy as np

    distinct = int(len(np.unique(groups)))
    # Name both numbers; sklearn's error names neither.
    if distinct < folds:
        raise ValueError(
            f"그룹 수가 fold 수보다 적어 그룹 단위로 나눌 수 없습니다 — 그룹 {distinct}개, "
            f"필요 {folds}개. 그룹 열이 행마다 고유한 값이 아닌지, 또는 데이터가 너무 "
            "작은지 확인하세요."
        )
    if not stratify:
        from sklearn.model_selection import GroupShuffleSplit

        # Same ``1/folds`` share, no classes to balance.
        shuffler = GroupShuffleSplit(n_splits=1, test_size=1 / folds, random_state=seed)
        return next(iter(shuffler.split(x_arr, y_arr, groups=groups)))

    from sklearn.model_selection import StratifiedGroupKFold

    # First fold is the held-out part.
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return next(iter(splitter.split(x_arr, y_arr, groups=groups)))


# --- Role: split checks -----------------------------------------------------------


def val_fingerprint(x_val: Any, y_val: Any) -> str:
    """val_fingerprint | Role: 32-char hash of the exact val rows a trial was scored on.

    Catches accidents, not attacks; compared only within one run.
    """
    import hashlib

    import numpy as np

    digest = hashlib.blake2b(digest_size=16)
    for part in (x_val, y_val):
        arr = np.asarray(part)
        digest.update(f"{arr.dtype}|{arr.shape}|".encode())
        # Object ``tobytes`` hashes addresses, so use ``repr``.
        if arr.dtype == object:
            digest.update(repr(arr.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def protocol_mismatch(
    declared: Any, seed: int, group_column: str | None = None, stratified: bool = True
) -> str | None:
    """protocol_mismatch | Role: Korean list of keys differing from this run, or ``None``.

    Missing blocks and missing keys are accepted
    """
    if not isinstance(declared, dict) or not declared:
        return None
    expected = protocol(seed, group_column, stratified)
    differences = [
        f"{key}: 카드 {declared.get(key)!r} vs 이번 실행 {expected[key]!r}"
        for key in ("test_fraction", "val_fraction", "seed", "stratified", "grouped_by")
        if key in declared and declared.get(key) != expected[key]
    ]
    if not differences:
        return None
    return (
        "카드의 split 프로토콜이 이번 실행과 다릅니다 — "
        + "; ".join(differences)
        + ". 카드의 기준선은 다른 행 집합에서 측정됐으므로 그 기준선에서 유도한 목표는 "
        "이번 실행의 점수와 비교할 수 없습니다. --seed와 --group-column을 카드와 맞추거나 "
        "카드를 다시 생성하세요."
    )
