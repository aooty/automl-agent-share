"""목표 열 인코딩, 그것이 함의하는 task, 그리고 결측 라벨을 어떻게 할지.

**두 고정 스크립트가 공유하고, 둘은 같은 답을 내야 한다**. 결측 라벨
정책은 둘:

``reject``  기본값 — 멈추고 몇 개가 비었는지 말한다.
``drop``    떨어뜨리고 몇 개였는지 기록한다. 두 스크립트가 똑같이 적용한다.

**task도 이 열에서 나오고, 호출자가 고를 것이 아니다** — :func:`detect_task`가 판정하고 답은
카드에 적힌다.

pandas는 함수 안에서 import한다. 그래야 프롬프트를 렌더하는 프로세스가 그것을 결코 올리지 않는다.
"""

from __future__ import annotations

from typing import Any

from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION

POLICY_REJECT = "reject"
POLICY_DROP = "drop"
TARGET_MISSING_POLICIES: tuple[str, ...] = (POLICY_REJECT, POLICY_DROP)
DEFAULT_TARGET_MISSING_POLICY = POLICY_REJECT

# 정수값 목표가 여전히 클래스로 읽히면서 담을 수 있는 서로 다른 값의 수. 이 위는 개수나 나이이고
# 라벨 집합이 아니다.
CLASSIFICATION_MAX_DISTINCT = 20


class TargetMissingError(ValueError):
    """목표 열에 결측 라벨이 있고 정책이 ``reject``다."""


class TargetUnusableError(ValueError):
    """목표 열이 애초에 목표가 될 수 없다 — 값이 하나이거나, 없다."""


def detect_task(series: Any) -> str:
    """``series``가 분류 목표인지 회귀 목표인지.

    규칙은 라벨이 *있는* 행에 대해, 이 순서로 — 순서가 중요하다:

    * 숫자가 아닌 열(문자열, 범주)은 분류;
    * boolean은 분류 — 숫자 규칙이 0/1 정수로 보기 전에;
    * 서로 다른 값 둘은 dtype이 무엇이든 분류;
    * 정수가 아닌 값을 담은 실수 열은 회귀;
    * 그 밖에는 :data:`CLASSIFICATION_MAX_DISTINCT`에서 개수가 정한다.

    서로 다른 라벨이 둘 미만인 열에는 :class:`TargetUnusableError`를 낸다 — task가 아니라 거절이다.
    """
    import pandas as pd

    present = series.dropna()
    distinct = int(present.nunique())
    if distinct < 2:
        raise TargetUnusableError(
            f"target column {str(series.name)!r} has {distinct} distinct value(s) in "
            f"{int(len(present))} labelled rows, so there is nothing to predict. Check "
            "that the right column was named, and that the rows were not filtered down "
            "to a single outcome."
        )
    if pd.api.types.is_bool_dtype(present) or not pd.api.types.is_numeric_dtype(present):
        return TASK_CLASSIFICATION
    if distinct == 2:
        return TASK_CLASSIFICATION
    if pd.api.types.is_float_dtype(present) and not bool((present % 1 == 0).all()):
        return TASK_REGRESSION
    return TASK_CLASSIFICATION if distinct <= CLASSIFICATION_MAX_DISTINCT else TASK_REGRESSION


def encode_target(
    series: Any, policy: str = DEFAULT_TARGET_MISSING_POLICY, task: str | None = None
) -> tuple[Any, Any, int]:
    """목표 열을 인코딩한다. NaN에도 실수에도 클래스를 만들어 주지 않는다.

    ``(values, keep, n_missing)``을 낸다:

    ``values``     분류면 라벨이 있는 행의 정수 클래스 코드 — 결코 ``-1``이 아니다. 회귀면 그
                   행들의 숫자를 ``float64``로, 손대지 않고: 스케일링도 비닝도 없으므로 목표
                   단위의 점수(``mae``)가 파일이 쓰는 단위 그대로다.
    ``keep``       원래 행에 대한 boolean 마스크. 특징 프레임을 ``values``가 덮는 행으로 정확히
                   걸러내는 데 쓴다.
    ``n_missing``  라벨이 몇 개 비어 있었는지 (정책이 ``reject``면 언제나 0 — 0이 아니면 raise).

    ``task``를 주지 않으면 열에서 판정한다. 이미 판정한 호출자(두 스크립트 모두 카드에 적으려고
    그렇게 한다)는 같은 열에 ``nunique``를 두 번 물지 않도록 그것을 넘긴다.
    """
    n_rows = int(len(series))
    n_missing = int(series.isna().sum())
    if n_missing and policy != POLICY_DROP:
        raise TargetMissingError(
            f"target column {str(series.name)!r} has {n_missing} missing values out of "
            f"{n_rows} rows. Pass --on-missing-target drop to train on the remaining "
            f"{n_rows - n_missing} rows, or fix the labels."
        )
    keep = series.notna()
    present = series[keep] if n_missing else series
    if (task or detect_task(series)) == TASK_REGRESSION:
        return present.astype("float64"), keep, n_missing
    codes = present.astype("category").cat.codes
    return codes, keep, n_missing


def target_classes(series: Any, task: str | None = None) -> list[Any] | None:
    """:func:`encode_target`의 각 클래스 코드가 무엇을 뜻하는지. 회귀 목표면 ``None``.

    구성상 그 코드들과 인덱스가 맞는다 (둘 다 ``astype("category")``를 지나고, 그 categories는
    정렬된 서로 다른 라벨이다). ``test_the_class_labels_line_up_with_the_codes``가 그 결합을 묶는다.

    값은 JSON 스칼라로 정규화한다 — 디스크에 쓰여 다른 프로세스가 읽기 때문이다. 별난 라벨 타입에는
    단방향이다 (Timestamp는 그 문자열 형태가 된다). 그래서 *출력에 라벨을 붙이는* 용도이고, 입력을
    다시 인코딩하는 데는 결코 쓰지 않는다.
    """
    import pandas as pd

    present = series.dropna()
    if (task or detect_task(series)) == TASK_REGRESSION:
        return None
    categories = pd.Series(present).astype("category").cat.categories
    return [_json_scalar(value) for value in categories.tolist()]


def _json_scalar(value: Any) -> Any:
    """``json.dumps``가 받는 무엇으로서의 라벨. 라벨이 말하는 바는 되도록 바꾸지 않는다."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)
