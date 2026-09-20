"""분할 규약: 어느 행이 train, 어느 행이 tune, 그리고 아무도 손대지 않는 행.

**세 번째 분할이 있는 이유는 루프가 수를 보고하는 집합에서 선택하기 때문이다** —
``docs/rationale.md``. *test* 슬라이스는 루프 뒤에 저장된 최적 모델로 정확히 한 번 채점된다
(:mod:`automl_agent.nodes.holdout`). 보고서에서 어떤 결정도 그것에 대고 내리지 않은 유일한 수다.

**train/validation 분할보다 먼저 뗀다.** 그래서 test 행은 파일과 시드만의 함수이고, 반복 횟수·모델·
에이전트가 제안한 무엇과도 무관하다. 그것이 "끝에 한 번 채점한다"를 의미 있게 만든다.

**한 모듈인 이유는 프로파일러가 바가 나오는 기준선을 재고 학습기가 그것과 비교되는 것을 재기
때문이다.** 다르게 나누면 둘은 비교할 수 없다. 둘 다 :func:`split_three_way`를 부르고, 프로파일러의
기준선은 ``x_test``를 일부러 무시한다.

``groups``를 주면 어떤 그룹도 두 집합으로 갈라지지 않는다. 그것이 파라미터인 이유는 반복되는 피험자에
대한 무작위 분할이 여기 어느 검사에도 안 보이기 때문이다 (``docs/rationale.md``). 카드의 *private*
``data`` 블록으로 도착하므로 LLM은 그것을 볼 수도, 바꾸자고 제안할 수도 없다.

모듈 수준에 sklearn이 없다: 노드들은 :func:`protocol` 때문에 이것을 import하고, 프롬프트를 렌더하는
프로세스는 데이터 스택에서 자유롭게 남는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# 전체 행 중. 실행 전체에서 떼어 두고 끝에 한 번 채점한다.
TEST_FRACTION = 0.2

# test 슬라이스를 뗀 나머지 중. 따라서 0.25 * 0.8 = 전체 행의 0.2. *pool에 대한* 분수로 두는 이유는
# 그것이 ``train_test_split``이 받는 인자이기 때문이고, 다르게 적으면 곱셈 하나만큼 틀리기를 청하는
# 셈이다.
VAL_FRACTION_OF_POOL = 0.25

# 세 수가 실제로 얼마인지. 문서와 카드를 위해.
TRAIN_SHARE = round((1 - TEST_FRACTION) * (1 - VAL_FRACTION_OF_POOL), 4)  # 0.6
VAL_SHARE = round((1 - TEST_FRACTION) * VAL_FRACTION_OF_POOL, 4)  # 0.2

# 그룹 경로의 fold 수. 적어 두는 대신 위 분수에서 유도한다: ``k``개 중 한 fold를 떼면 ``1/k`` 몫이
# 재현되고, 두 분수는 정확히 역수다 (5, 그다음 4). 누가 분수를 역수가 아닌 값으로 바꾸면 ``round``가
# 편집을 조용히 무시하는 대신 가장 가까운 정직한 분할을 지키고, 테스트의 sizes 단정이 그 대가를
# 말해 준다.
TEST_FOLDS = round(1 / TEST_FRACTION)  # 5
VAL_FOLDS = round(1 / VAL_FRACTION_OF_POOL)  # 4


@dataclass(frozen=True)
class Splits:
    """서로 겹치지 않는 세 행 집합. 위치 인자 여섯 개는 버그이므로 속성 접근.

    ``x_test``/``y_test``를 다른 둘을 만드는 함수가 함께 내주는 것은 일부러다: 그것을 보면 안 되는
    호출자(프로파일러의 기준선)는 무시하고, 맞춰 둘 규약 구현은 하나뿐이다.

    ``groups_*`` 셋은 각 집합 행의 그룹 라벨이고, 행 단위 경로에서는 ``None``이다. 다시 계산하지
    않고 실어 나르는 이유는 호출자가 그것을 다시 만들 수 없기 때문이다 — 분할이 섞으므로 돌려받은
    ``x_val``에서 각 행이 온 그룹으로 되돌아갈 길이 없다. 호출자는
    :mod:`automl_agent.scoring.intervals`이고, 환자당 다섯 번의 방문을 담은 분할에 대한 bootstrap은
    환자를 재표집해야 한다 — 아니면 실제 있는 것보다 다섯 배 많은 관측에 대한 구간을 보고한다.
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
    """규약을 카드/결과 필드로: 정확한 행 집합을 재현할 만큼.

    공개하는 이유는 독자가 어느 수를 선택에 썼고 어느 수를 안 썼는지 가를 수 있게, 그리고 다른
    규약에서 측정된 카드가 조용히 비교되는 대신 *거절*되게 하려고
    (:func:`automl_agent.nodes.profiling.assert_protocol_matches`).

    ``grouped_by``는 어떤 그룹도 두 집합으로 갈라지지 않은 열의 이름이고, 행 단위 분할에서는
    ``None``이다. ``stratified``는 회귀 타깃에서 ``False``다 — 균형 잡을 층이 없다. 둘 다 가정하지
    않고 이름으로 공개하는 이유는 ``docs/rationale.md``.
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
    """로그·콘솔 요약·보고서에 쓰는 한 줄."""
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
    """각 집합이 몇 행을 갖는지, 분할이 실제로 반올림하는 방식대로.

    ``n * share``가 아니다. ``train_test_split``은 float ``test_size``의 *천장*을 취하고 나머지
    전부를 다른 쪽에 주며, :func:`split_three_way`는 그것을 두 번 한다 — 그래서 곱은 집합마다 최대
    두 행 틀린다. 반올림이 중요한 이유는 이 수가 임계값을 만나기 때문이다: sklearn은
    ``early_stopping='auto'``를 ``n_samples > 10_000``에서 결정하고, 그 크기 근처의 파일에서는 두 행
    오차가 임계값의 반대편에 떨어진다.

    행 단위 경로에서는 정확하고 (``bench/``에 기록된 train 크기들을 재현한다), 그룹 경로에서는
    근사다 — 몫을 딱 맞추자고 그룹을 가를 수는 없다. 이 수를 공개하는 호출자는 어느 쪽인지 말해야
    한다. :func:`automl_agent.capabilities.describe_row_budget`가 그렇게 한다.

    선언된 규약 블록이 아니라 이 모듈의 상수에서 유도한다 (``docs/rationale.md``).

    분할이 거절할 곳에서 함께 거절한다. 세 행 미만에서는 두 천장이 전부를 가져가고 학습할 것이 남지
    않는다. ``train_test_split``도 거기서 raise하고("the resulting train set will be empty"),
    ``train: 0``이라는 예보는 답처럼 읽힐 것이다.
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
    """전체에서 test를 떼고, 남은 것에서 validation을 뗀다.

    두 분할 모두 라벨에 층화하므로 드문 클래스가 운이 아니라 세 집합 모두에 대표된다.
    ``random_state``는 두 호출 모두 실행의 시드다. 그것이 test 행을 실행의 모든 반복에서 —
    그리고 프로파일러의 프로세스와 학습기의 프로세스 사이에서 — 동일하게 만든다.

    ``groups``가 있으면 같은 그룹 값을 가진 행은 모두 같은 집합에 떨어진다. 분할기가
    ``GroupShuffleSplit``이 아니라 ``StratifiedGroupKFold``인 이유는 ``docs/rationale.md``. 첫
    fold를 떼어 둔 쪽으로 취하는 것이 *k*-fold 분할기에서 단일 분할을 얻는 방법이고, 그 결과 몫은
    근사다 — 몫을 딱 맞추자고 그룹을 가를 수는 없다.

    ``stratify=False``는 연속 타깃용이다. 거기서 층화는 호출자가 하는 선택이 아니라 불가능이다:
    모든 값이 자기만의 층이므로 sklearn이 "the least populated class has only 1 member"로 거절한다.
    호출자들은 자기가 정한 플래그를 넘기지 않는다 — :func:`automl_agent.dataset.targets.detect_task`가
    그 열에 대해 말한 것을 넘기고, 그것이 카드가 공개하는 것과 같다.
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
    """그룹을 지키는 분할에서 ``(나머지, fold 하나)``의 인덱스.

    ``StratifiedGroupKFold``는 fold 수만큼의 서로 다른 그룹이 필요하고, 자기 오류 메시지는 두 수
    어느 것도 대지 않는다. 여기서 둘을 다 적어 거절하는 것이 "열을 고쳐라"와 "sklearn 소스를
    읽어라"의 차이다. 그 수는 층화 없는 분할기에도 중요하다 — ``GroupShuffleSplit``은 불평하는 대신
    빈 held-out 집합을 돌려줄 것이다 — 그래서 가드를 공유한다.

    층화 없이는 분할기가 ``test_size = 1/folds``의 ``GroupShuffleSplit``이다: 같은 몫을, 균형 잡을
    층이 없을 때 쓸 수 있는 유일한 방식으로 적은 것.
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
    """한 시도가 채점된 정확한 validation 행들의 짧은 요약값.

    짝지은 비교(:func:`automl_agent.scoring.intervals.paired_delta`)는 *같은 행*에서 채점된 두 시도를
    필요로 하고, "같은 시드"는 그것이 아니다 (``docs/rationale.md``). 그래서 행 자체를 해시하고 그
    요약값이 예측과 함께 여행한다. 전제가 코드가 단정하는 것에서 나중 시도가 검사할 수 있는 것으로
    바뀐다.

    라벨만이 아니라 두 배열 다: ``y_val`` 하나는 이진 과제에서 0/1이므로, 같은 라벨이 같은 순서로
    있는 서로 다른 두 슬라이스가 충돌할 것이다.

    ``blake2b`` 16바이트, 암호학적 폭이 아니다 — 이것이 잡는 것은 사고이고 공격이 아니며, 요약값은
    자기가 서술하는 예측 옆 파일에 앉는다. 기계 사이로 이식되지 않는 것은 설계다: 같은 실행의 다른
    반복에 대고만 비교된다.
    """
    import hashlib

    import numpy as np

    digest = hashlib.blake2b(digest_size=16)
    for part in (x_val, y_val):
        arr = np.asarray(part)
        digest.update(f"{arr.dtype}|{arr.shape}|".encode())
        # object 배열에 대한 ``tobytes``는 포인터 주소를 해시하고 그것은 프로세스마다 다르다 — 그러면
        # 비교되는 두 반복은 결코 일치하지 않고 모든 비교가 "분할이 바뀌었다"로 읽힌다. 로더는
        # float64를 돌려주므로 이 분기는 그러지 않는 호출자를 위한 것이다.
        if arr.dtype == object:
            digest.update(repr(arr.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def protocol_mismatch(
    declared: Any, seed: int, group_column: str | None = None, stratified: bool = True
) -> str | None:
    """``declared``를 이 ``seed``의 실행과 비교할 수 없는 이유, 또는 ``None``.

    카드는 자기 기준선이 측정된 규약을 들고 다닌다. 어긋남은 유도된 임계값이 시도들이 채점되는 행과
    다른 행에서 계산됐다는 뜻이다 — 비교가 아닌 비교의 보고. ``--on-missing-target``과 같은 태도다:
    두 가지 다른 것을 조용히 재는 대신 거절한다.

    규약 블록이 없으면 = 이 필드보다 앞선 카드이고, 받아들인다 (``docs/rationale.md``). 같은
    ``key in declared`` 규칙으로 ``grouped_by``보다 앞선 카드도 받아들인다.

    ``stratified``는 *이번 실행이* 할 것이고, 타깃의 과제에서 따라온다. 호출자는 그것을 고르지 않고
    카드에서 읽으므로, 회귀 카드가 연속 타깃이 가질 수 없는 층화에 대고 플래그되지 않는다.
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
