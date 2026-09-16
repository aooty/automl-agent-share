"""The split protocol: which rows train, which rows tune, and which rows nobody touches.

**A third split exists because the loop selects on the set it reports from.** ``best`` is the
maximum of several validation scores, so the winner is partly the luckiest split and the reported
figure is biased upward by an amount nobody can measure from inside the run. The *test* slice is
scored exactly once, after the loop, from the saved best model
(:mod:`automl_agent.nodes.holdout`) — the only number in the report no decision was made against.

**Carved first, before the train/validation split**, so the test rows are a function of the file and
the seed alone — not the iteration count, the model, or anything the agent proposed. That is what
makes "scored once at the end" mean anything.

**One module, because the profiler measures the baseline the bar comes from and the trainer measures
what is compared against it.** Split differently and the two are not comparable. Both call
:func:`split_three_way`, and the profiler's baseline deliberately ignores ``x_test``.

**``groups`` exists because a random split over repeated subjects is invisible to every check here.**
Several rows per patient put the same patient in train and validation; the model recognises the
patient rather than the condition, and the held-back slice is contaminated the same way — so it
*agrees* with the validation number and the selection gap looks healthy. Pass ``groups`` and no
group is divided. It arrives through the card's *private* ``data`` block, so the LLM can neither see
it nor propose changing it.

No sklearn at module level: the nodes import this for :func:`protocol`, and the process that renders
prompts stays free of the data stack. Rationale: ``docs/rationale.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Of all rows. Held back from the whole run, scored once at the end.
TEST_FRACTION = 0.2

# Of what remains after the test slice, so 0.25 * 0.8 = 0.2 of all rows. Kept as a
# fraction *of the pool* because that is the argument ``train_test_split`` takes, and
# stating it any other way invites an off-by-a-multiplication.
VAL_FRACTION_OF_POOL = 0.25

# What the three numbers come to, for documentation and for the card.
TRAIN_SHARE = round((1 - TEST_FRACTION) * (1 - VAL_FRACTION_OF_POOL), 4)  # 0.6
VAL_SHARE = round((1 - TEST_FRACTION) * VAL_FRACTION_OF_POOL, 4)  # 0.2

# Fold counts for the grouped path, derived from the fractions above rather than written
# out: taking one fold of ``k`` as the held-out part reproduces a ``1/k`` share, and both
# fractions happen to invert exactly (5 folds, then 4). If someone changes a fraction to
# something that does not, ``round`` keeps the nearest honest split instead of silently
# ignoring the edit — and the sizes assertion in the tests will say what it cost.
TEST_FOLDS = round(1 / TEST_FRACTION)  # 5
VAL_FOLDS = round(1 / VAL_FRACTION_OF_POOL)  # 4


@dataclass(frozen=True)
class Splits:
    """Three disjoint row sets. Attribute access, because six positional arrays is a bug.

    ``x_test``/``y_test`` are handed out by the same function that produces the other
    two on purpose: a caller that must not look at them (the profiler's baseline)
    ignores them, and there is only one implementation of the protocol to keep in step.

    The three ``groups_*`` arrays are the group labels of each set's rows, or ``None`` on
    the row-level path. They are carried rather than recomputed because a caller that needs
    them cannot rebuild them: the split shuffles, so there is no way back from a returned
    ``x_val`` to the group each of its rows came from. :mod:`automl_agent.scoring.intervals` is the
    caller — a bootstrap over a split holding five visits per patient has to resample
    patients, or it reports an interval about five times as many observations as exist.
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
        return {
            "train": int(len(self.y_train)),
            "val": int(len(self.y_val)),
            "test": int(len(self.y_test)),
        }


def protocol(seed: int, group_column: str | None = None, stratified: bool = True) -> dict[str, Any]:
    """The protocol as a card/result field: enough to reproduce the exact row sets.

    Published so a reader can tell which number was selected on and which was not, and so a card
    measured under a different protocol is *refused* rather than silently compared against
    (:func:`automl_agent.nodes.profiling.assert_protocol_matches`).

    ``grouped_by`` names the column no group of which was split across two sets, ``None`` for a
    row-level split. The one protocol field whose absence changes what the scores *mean* rather than
    only which rows they came from — hence published by name, so a reader comparing two cards can see
    that one held patients together and the other did not.

    ``stratified`` is ``False`` on a regression target (no strata to balance). Published rather than
    assumed because it is checked: a card measured on a classification target and a run scoring a
    continuous one disagree here, and that disagreement is the useful half of the message.
    """
    return {
        "train_fraction": TRAIN_SHARE,
        "val_fraction": VAL_SHARE,
        "test_fraction": TEST_FRACTION,
        "stratified": bool(stratified),
        "grouped_by": str(group_column) if group_column else None,
        "seed": int(seed),
    }


def describe_protocol(declared: dict[str, Any] | None = None, seed: int = 42) -> str:
    """One Korean line for a log, a console summary or the report."""
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


def row_counts(n_rows: int) -> dict[str, int]:
    """How many rows each set holds, computed the way the split actually rounds.

    Not ``n * share``. ``train_test_split`` takes the *ceiling* of a float ``test_size`` and gives
    the whole remainder to the other side, and :func:`split_three_way` does it twice — so the product
    is off by up to two rows per set. The rounding matters because this number meets a threshold:
    sklearn resolves ``early_stopping='auto'`` at ``n_samples > 10_000``, and a two-row error lands
    on the wrong side of it for a file near that size.

    Exact on the row-level path (it reproduces all five recorded train sizes in ``bench/``),
    approximate on the grouped one, where a group cannot be divided to make a share come out even.
    Callers publishing these numbers must say which — :func:`automl_agent.capabilities.describe_row_budget`
    does.

    Derived from this module's constants, not from a declared protocol block, because the rounding is
    a property of *this* implementation. A card measured under different fractions is refused, not
    reinterpreted (:func:`automl_agent.nodes.profiling.assert_protocol_matches`).

    Refuses wherever the split would. Under three rows both ceilings take everything and nothing is
    left to train on; ``train_test_split`` raises there too ("the resulting train set will be
    empty"), and a forecast of ``train: 0`` would read as an answer.
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


def split_three_way(
    x_arr: Any, y_arr: Any, seed: int, groups: Any = None, stratify: bool = True
) -> Splits:
    """Carve test off the whole set, then validation off what is left.

    Both splits are stratified on the label, so a rare class is represented in all three
    sets rather than by luck. ``random_state`` is the run's seed in both calls, which is
    what makes the test rows identical across every iteration of a run — and identical
    between the profiler's process and the trainer's.

    With ``groups`` given, every row sharing a group value lands in the same set. The
    splitter is ``StratifiedGroupKFold`` rather than ``GroupShuffleSplit`` because giving
    up stratification to gain group integrity trades one unmeasurable problem for another:
    on the 12%-positive cohorts this repo is aimed at, an unstratified validation slice can
    hold almost no positives and its ``balanced_accuracy`` then describes the split. Taking
    the first fold as the held-out part is what makes the *k*-fold splitter produce a single
    split; the shares are approximate as a result, because a group cannot be divided to
    make them exact.

    ``stratify=False`` is for a continuous target, where stratification is not a choice
    the caller makes but an impossibility: every value is its own stratum, so sklearn
    refuses with "the least populated class has only 1 member". The callers do not pass a
    flag they decided on — they pass what :func:`automl_agent.dataset.targets.detect_task` said
    about the column, which is also what the card publishes.
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
    """Indices of ``(the rest, one fold)`` under a group-respecting split.

    ``StratifiedGroupKFold`` needs at least as many distinct groups as folds, and its own
    error names neither number. Refusing here with both counts is the difference between
    "fix the column" and "read the sklearn source". The count matters for the unstratified
    splitter too — ``GroupShuffleSplit`` would hand back an empty held-out set instead of
    complaining — so the guard is shared.

    Without stratification the splitter is ``GroupShuffleSplit`` at ``test_size = 1/folds``:
    the same share, expressed the only way available when there are no strata to balance.
    """
    import numpy as np

    distinct = int(len(np.unique(groups)))
    if distinct < folds:
        raise ValueError(
            f"그룹 수가 fold 수보다 적어 그룹 단위로 나눌 수 없습니다 — 그룹 {distinct}개, "
            f"필요 {folds}개. 그룹 열이 행마다 고유한 값이 아닌지, 또는 데이터가 너무 "
            "작은지 확인하세요."
        )
    if not stratify:
        from sklearn.model_selection import GroupShuffleSplit

        shuffler = GroupShuffleSplit(n_splits=1, test_size=1 / folds, random_state=seed)
        return next(iter(shuffler.split(x_arr, y_arr, groups=groups)))

    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return next(iter(splitter.split(x_arr, y_arr, groups=groups)))


def val_fingerprint(x_val: Any, y_val: Any) -> str:
    """A short digest of the exact validation rows an attempt was scored on.

    Paired comparison (:func:`automl_agent.scoring.intervals.paired_delta`) needs two attempts scored
    on the *same rows*, and "same seed" is not that: the seed splits whatever matrix it is handed, and
    what reaches it moves without it — ``--on-missing-target drop`` removes a different count, a
    re-extracted file has different rows in the same shape, a target now reading as regression is
    split unstratified. So the rows themselves are hashed and the digest travels with the predictions,
    turning the premise from something the code asserts into something a later attempt can check.

    Both arrays, not just labels: ``y_val`` alone is 0/1 on a binary task, so two different slices
    with the same labels in the same order would collide.

    ``blake2b`` at 16 bytes, not a cryptographic width — this catches an accident, not an attack, and
    the digest sits in a file beside the predictions it describes. Not portable across machines by
    design: it is only ever compared against another iteration of the same run.
    """
    import hashlib

    import numpy as np

    digest = hashlib.blake2b(digest_size=16)
    for part in (x_val, y_val):
        arr = np.asarray(part)
        digest.update(f"{arr.dtype}|{arr.shape}|".encode())
        # ``tobytes`` on an object array would hash pointer addresses, which differ between
        # processes — so the two iterations being compared would never agree and every
        # comparison would read as "split changed". The loader hands back float64, so this
        # branch is for a caller that does not.
        if arr.dtype == object:
            digest.update(repr(arr.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def protocol_mismatch(
    declared: Any, seed: int, group_column: str | None = None, stratified: bool = True
) -> str | None:
    """Why ``declared`` cannot be compared against a run at this ``seed``, or ``None``.

    A card carries the protocol its baseline was measured under. Disagreement means the derived
    threshold was computed against different rows than the attempts are scored on — a reported
    comparison that is not one. Same stance as ``--on-missing-target``: refuse rather than quietly
    measure two different things.

    No protocol block = a card predating this field, accepted. Its baseline used the old two-way
    split, a difference in the *bar* rather than a leak, and stopping a resumed run over it would be
    worse than the incomparability.

    ``grouped_by`` is the disagreement most worth catching: a bar measured with patients held
    together, compared against scores measured with them split, is not stricter or looser but
    meaningless — and the inflated side is the one the run reports. Cards predating the field are
    still accepted, by the same ``key in declared`` rule.

    ``stratified`` is what *this run* will do, following from the target's task. The caller reads it
    off the card rather than choosing it, so a regression card is not flagged against a
    stratification a continuous target cannot have.
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
