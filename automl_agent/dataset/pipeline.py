"""특성 공간 변환의 순서 있는 목록. JSON으로 선언되고 whitelist에서 해석된다.

**``exec``되는 것은 없고 설정에서 이름으로 import되는 것도 없다.** :data:`STEPS`는 닫힌 집합이고,
이 모듈이 모르는 단계는 해석되는 대신 이유와 함께 떨어진다.

**whitelist가 짧은 것은 재 봤기 때문이다**. 후보 일곱 중 둘만 남았고
**다섯은 일부러 없으며 각 부재가 하나의 측정이다**. 남은 둘이 합성되는 것이 이것을 플래그 둘이 아니라
순서 있는 목록으로 만든다.

**열 동일성은 이름으로, 적합된 스키마에 대고 해석되고, 그 이름은 인코딩된 쪽이다** — one-hot 원본
열은 여럿이 되었고, ``city``를 대는 계획은 그 전부를 뜻한다. 이 모듈이
:mod:`automl_agent.dataset.features`를 아는 이유는 그 해석뿐이다.

import가 가볍다: 오케스트레이터는 subprocess가 생기기 전에 계획을 allowlist하려고 :data:`STEPS`를
import하고, sklearn은 :func:`build_steps` 안에서 import한다.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .features import LEVEL_SEPARATOR

# 단계가 가질 수 있는 이름. 닫힌 집합이다 — 해석기는 설정에서 아무것도 이름으로 해석하지 않으므로,
# 모르는 단계는 떨어진 단계이고 import가 아니다.
STEP_IMPUTE = "impute"
STEP_INTERACTIONS = "interactions"
STEP_SCALE = "scale"
STEP_MISSING_INDICATOR = "missing_indicator"
STEP_MISSING_COUNT = "missing_count"
STEPS: tuple[str, ...] = (
    STEP_IMPUTE,
    STEP_INTERACTIONS,
    STEP_SCALE,
    STEP_MISSING_INDICATOR,
    STEP_MISSING_COUNT,
)

# 대체하는 대신 덧붙이는 두 단계. 따라서 어떤 imputer보다 앞에 와야 한다 — 대치 뒤에는 표시할 NaN이
# 남아 있지 않다. 집합으로 이름을 갖는 이유는 해석기가 명세가 imputer를 선언하지 않았을 때 기본
# imputer를 놓을 자리를 이것으로 정하기 때문이다.
APPENDING_STEPS = frozenset({STEP_MISSING_INDICATOR, STEP_MISSING_COUNT})

# 단계가 만드는 열의 접미사. 사람과 tracker 자신의 테스트 말고는 아무도 읽지 않는다 — 이름을 두는
# 요점은 연달아 오는 두 단계를 로그에서 가를 수 있다는 것이다.
INDICATOR_SUFFIX = "__missing"
COUNT_COLUMN = "missing_count"
# 공백. ``PolynomialFeatures`` 자신이 곱을 적는 방식이고, 읽기 더 좋은 ``*``가 아니다.
INTERACTION_JOIN = " "

IMPUTE_STRATEGIES = ("median", "mean", "most_frequent", "constant")
# SimpleImputer의 전략이 아니다 — 열을 그대로 두어서 NaN으로 분기하는 계열이 쓸 수 있게 한다.
# *remainder* 전략으로서, 그리고 그룹 안에서 인정된다.
IMPUTE_NONE = "none"
DEFAULT_IMPUTE = "median"

# 차수 2뿐이다. 파라미터가 있는 이유는 자기가 뜻하는 차수를 대는 계획이
# 조용히 다른 것을 받는 대신 어느 것이 돌았는지 듣게 하려고.
INTERACTION_DEGREE = 2
# 입력 열이 이보다 많으면 interaction 단계를 거절한다. ``guard_memory``가 통과한 *뒤* 넓은 인코딩과
# out-of-memory kill 사이에 서 있는 것은 이것뿐이다.
INTERACTION_MAX_COLUMNS = 50


def encoded_positions(
    columns: list[str], names: Any
) -> tuple[list[int], list[str]]:
    """인코딩된 배치 안에서 원본 열 ``names``에 대한 ``(positions, unknown)``.

    이름은 인코딩된 열과 그대로 맞거나, one-hot 원본이 낸 인코딩된 열 전부와 맞는다 — ``city``는
    ``city=seoul``·``city=busan``·``city=<missing>``을 고른다. ``city``를 대치하라는 계획은 카드에서
    본 그 열을 뜻하고 그 수준 하나를 뜻하지 않기 때문이다. 수준 이름도 개별로 받는다 — 로그가
    되돌려 찍는 것이 그것이므로.

    ``unknown``은 raise하지 않고 돌려준다. 파일에 없는 열을 대는 계획은 ``applied_pipeline``에
    보고할 값이 있는 실수이고, 반복 하나를 쓸 값은 없다 — 단계의 나머지는 여전히 실행기가 할 수
    있는 무엇을 서술한다.

    위치는 정렬되고 중복이 걷힌다. 그래서 ``["city", "city=seoul"]``은 한 선택이고 같은 열을 두 번
    적은 것이 아니다 — 후자는 ``ColumnTransformer``에서 적합된 중복이 된다.
    """
    index = {name: position for position, name in enumerate(columns)}
    positions: set[int] = set()
    unknown: list[str] = []
    for raw in names if isinstance(names, (list, tuple)) else ():
        name = str(raw)
        if name in index:
            positions.add(index[name])
            continue
        prefix = f"{name}{LEVEL_SEPARATOR}"
        levels = [position for column, position in index.items() if column.startswith(prefix)]
        if levels:
            positions.update(levels)
        else:
            unknown.append(name)
    return sorted(positions), unknown


def _rest(columns: list[str], taken: Any) -> list[int]:
    """``taken``에 없는 위치들, 원래 순서대로 — ``ColumnTransformer``의 remainder가 나오는 순서."""
    return [position for position in range(len(columns)) if position not in taken]


def _partial(
    name: str, transformer: Any, positions: list[int], columns: list[str], appended: Any = ()
) -> tuple[Any, list[str]]:
    """일부 열에만 도는 변환을 ``ColumnTransformer``로 싸고, 그 출력 열 이름을 함께 낸다.

    ``ColumnTransformer``는 변환기의 열을 먼저, 그다음 remainder를 원래 순서로 낸다. ``appended``는
    변환기가 *덧붙인* 열의 이름이고, 그 둘 사이에 들어간다.

    ``sparse_threshold=0.0``: 이 모듈이 출력 이름을 직접 추적하므로 sparse 출력은 추적할 이름이
    없고, 여기 estimator들은 어차피 dense 입력을 받는다.
    """
    from sklearn.compose import ColumnTransformer

    step = ColumnTransformer([(name, transformer, positions)], remainder="passthrough", sparse_threshold=0.0)
    names = [columns[position] for position in positions]
    names += list(appended)
    names += [columns[position] for position in _rest(columns, positions)]
    return step, names


def _selection(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[list[int], str]:
    """단계의 ``columns`` 키에 대한 ``(positions, label)``. 없으면 모든 열이다.

    label은 echo와 로그가 찍는 것이고, ``all``은 우연히 전부를 대는 목록과는 다른 답이다 —
    ``all``로 쓰인 명세는 다음 파일에 열이 하나 더 있어도 계속 전부를 뜻한다.
    """
    raw = spec.get("columns")
    if raw is None:
        return list(range(len(columns))), "all"
    positions, unknown = encoded_positions(columns, raw)
    if unknown:
        log.write(f"{spec.get('step')}: no such column(s) {sorted(unknown)}, ignored")
    return positions, ", ".join(columns[position] for position in positions) or "none"


def _impute_step(
    spec: dict[str, Any], columns: list[str], native_nan: bool, log: Any
) -> tuple[Any, list[str], str] | None:
    """``impute`` 단계 하나: 기본 전략 하나에 이름 붙은 열 그룹 몇 개.

    그룹이 없으면 맨 ``SimpleImputer``이고 열 이름은 그대로다 — 플래그 버전의 동작 그대로. 그룹이
    있으면 ``ColumnTransformer``이고, 그러면 출력 순서는 변환기 순서 뒤에 remainder다. 그래서 이름은
    가정되지 않고 추적되며, :func:`build_steps`의 호출자가 그 추적을 sklearn 자신의
    ``get_feature_names_out``에 대고 묶는 테스트를 갖고 있다.

    NaN을 받을 수 없는 계열에 온 ``strategy: none``은 치명적이 아니라 강등된다. 플래그 버전과 같은
    조건이다 — 계열에 대해 잘못 짐작한 계획은 로그 한 줄을 물어야 하고 반복 하나가 아니다.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    def strategy_of(source: dict[str, Any], default: str) -> str:
        raw = source.get("strategy")
        if raw is None:
            return default
        name = str(raw)
        if name in (*IMPUTE_STRATEGIES, IMPUTE_NONE):
            return name
        log.write(f"impute: unknown strategy {raw!r}, using {default}")
        return default

    remainder_strategy = strategy_of(spec, DEFAULT_IMPUTE)
    if remainder_strategy == IMPUTE_NONE and not native_nan:
        log.write(
            f"impute: strategy={IMPUTE_NONE} needs a family that splits on NaN; "
            f"using {DEFAULT_IMPUTE}"
        )
        remainder_strategy = DEFAULT_IMPUTE

    def make(strategy: str, fill: Any) -> Any:
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=float(fill))
        return SimpleImputer(strategy=strategy)

    transformers: list[tuple[str, Any, list[int]]] = []
    taken: set[int] = set()
    labels: list[str] = []
    raw_groups = spec.get("groups")
    for number, group in enumerate(raw_groups if isinstance(raw_groups, list) else []):
        if not isinstance(group, dict):
            continue
        positions, _label = _selection({**group, "step": "impute"}, columns, log)
        positions = [position for position in positions if position not in taken]
        if not positions:
            continue
        strategy = strategy_of(group, remainder_strategy)
        if strategy == IMPUTE_NONE and not native_nan:
            log.write(f"impute group {number}: strategy={IMPUTE_NONE} needs a NaN-splitting family; skipped")
            continue
        taken.update(positions)
        transformers.append(
            (
                f"g{number}",
                "passthrough" if strategy == IMPUTE_NONE else make(strategy, group.get("fill_value", 0.0)),
                positions,
            )
        )
        labels.append(f"{strategy}: {', '.join(columns[position] for position in positions)}")

    if not transformers:
        if remainder_strategy == IMPUTE_NONE:
            # 세운 것도 없고 덮은 것도 없다: 모든 열이 갖고 있던 NaN을 그대로 지킨다.
            return None
        return make(remainder_strategy, 0.0), list(columns), remainder_strategy

    step = ColumnTransformer(
        transformers,
        remainder="passthrough" if remainder_strategy == IMPUTE_NONE else make(remainder_strategy, 0.0),
        # :func:`_partial`과 같은 이유로 끈다.
        sparse_threshold=0.0,
    )
    # ColumnTransformer는 각 변환기의 열을 변환기 순서로, 그다음 remainder를 원래 순서로 낸다.
    # 가정하지 않고 추적한다 — 이 모듈의 docstring을 보라.
    names = [columns[position] for _name, _transformer, group in transformers for position in group]
    names += [columns[position] for position in _rest(columns, taken)]
    return step, names, f"{remainder_strategy} ({'; '.join(labels)})"


def _leaves_nan(spec: dict[str, Any], native_nan: bool) -> bool:
    """이 ``impute`` 단계가 일부러 어딘가에 NaN을 남기는지.

    세워진 ``ColumnTransformer``가 아니라 *명세*에서 읽는다. 질문이 무엇을 청했는가에 대한 것이고,
    답은 다음 단계를 세우기 전에 필요하기 때문이다. 남기는 방법은 둘 — ``none`` remainder, 또는
    ``strategy: none``인 그룹. 둘 다 NaN으로 기본적으로 분기하는 계열에서만 인정된다. 그 밖에서는
    강등되고, 그러면 남는 것이 없다.
    """
    if not native_nan:
        return False
    if str(spec.get("strategy") or DEFAULT_IMPUTE) == IMPUTE_NONE:
        return True
    groups = spec.get("groups")
    return any(
        str(group.get("strategy") or "") == IMPUTE_NONE
        for group in (groups if isinstance(groups, list) else [])
        if isinstance(group, dict)
    )


def _interactions_step(
    spec: dict[str, Any], columns: list[str], log: Any, nan_possible: bool = False
) -> tuple[Any, list[str], str] | None:
    """이름 붙은 열들의 차수 2 쌍별 곱, 그 열들 뒤에 덧붙여서.

    입력이 :data:`INTERACTION_MAX_COLUMNS`를 넘으면 거절한다 — preflight 가드가 볼 수 없는
    out-of-memory kill로 펼쳐지게 두는 대신. 가드는 도착하는 행렬의 값을 매기고, 이 단계는 다른
    행렬을 만드는 쪽이다.

    행렬에 NaN이 아직 있을 수 있는 동안에도 거절한다. 그것은 반복 하나를 치르고 찾았다.
    """
    from sklearn.preprocessing import PolynomialFeatures

    if nan_possible:
        log.write(
            "interactions: skipped because a NaN may still be in the matrix — no impute step "
            "has covered every column yet, and PolynomialFeatures refuses a NaN. Put an "
            "impute step that covers everything before it, or drop one of the two"
        )
        return None
    degree = spec.get("degree", INTERACTION_DEGREE)
    if not isinstance(degree, int) or isinstance(degree, bool) or degree != INTERACTION_DEGREE:
        log.write(
            f"interactions: degree={degree!r} is not available; only degree "
            f"{INTERACTION_DEGREE} (pairwise, no squares) is"
        )
        return None
    positions, label = _selection(spec, columns, log)
    if len(positions) < 2:
        log.write(f"interactions: needs at least two columns, got {len(positions)}; skipped")
        return None
    if len(positions) > INTERACTION_MAX_COLUMNS:
        log.write(
            f"interactions: {len(positions)} columns would expand to "
            f"{len(positions) + len(positions) * (len(positions) - 1) // 2}, over the "
            f"{INTERACTION_MAX_COLUMNS}-column limit; skipped"
        )
        return None
    poly = PolynomialFeatures(degree=INTERACTION_DEGREE, interaction_only=True, include_bias=False)
    pairs = [f"{columns[a]}{INTERACTION_JOIN}{columns[b]}" for a, b in combinations(positions, 2)]
    if len(positions) == len(columns):
        # 모든 열이므로 ColumnTransformer도, 놓을 remainder도 없다. PolynomialFeatures는 입력을
        # 순서대로, 그다음 쌍들을 ``combinations`` 순서로 낸다.
        return poly, list(columns) + pairs, label
    step, names = _partial("interact", poly, positions, columns, pairs)
    return step, names, label


def _scale_step(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[Any, list[str], str] | None:
    from sklearn.preprocessing import StandardScaler

    positions, label = _selection(spec, columns, log)
    if not positions:
        return None
    if len(positions) == len(columns):
        return StandardScaler(), list(columns), label
    step, names = _partial("scale", StandardScaler(), positions, columns)
    return step, names, label


def _appender_step(
    name: str, spec: dict[str, Any], columns: list[str], log: Any
) -> tuple[Any, list[str], str] | None:
    """``missing_indicator`` 또는 ``missing_count`` — 열을 덧붙이는 두 단계.

    둘 다 :mod:`automl_agent.dataset.features`의 함수 위에 놓인 ``FunctionTransformer``다. 그 함수가
    거기 살아야 하는 이유: 그 모듈은 안정된 이름으로 import되는데, ``scripts/train.py``는
    ``__main__``으로 돌기 때문에 거기 정의된 함수는 ``__main__.append_missing_indicator``로 pickle되고
    다른 어떤 프로세스에서도 해석되지 않는다. ``positions``는 ``kw_args``에 실려 가고, 그것은
    데이터로 pickle된다.
    """
    from sklearn.preprocessing import FunctionTransformer

    from .features import append_missing_count, append_missing_indicator

    if name == STEP_MISSING_COUNT:
        return (
            FunctionTransformer(append_missing_count),
            [*columns, COUNT_COLUMN],
            "all",
        )
    positions, label = _selection(spec, columns, log)
    if not positions:
        return None
    every = len(positions) == len(columns)
    return (
        # 모든 열일 때는 전체 목록이 아니라 ``None``. 그래야 ``all``이라고 쓴 명세가 플래그 버전과
        # 같은 방식으로 pickle되고 계속 전부를 뜻한다.
        FunctionTransformer(append_missing_indicator, kw_args=None if every else {"positions": positions}),
        [*columns, *(f"{columns[position]}{INDICATOR_SUFFIX}" for position in positions)],
        label,
    )


def build_steps(
    spec: Any, columns: list[str], log: Any, *, native_nan: bool = False
) -> tuple[list[tuple[str, Any]], list[str], list[str]]:
    """선언된 명세에 대한 ``(steps, names, applied)``. ``steps``는 그대로 Pipeline에 들어간다.

    ``applied``는 echo다: 실제로 돈 단계마다 렌더된 한 줄, 돈 순서대로. 그래서 보고서는 청한 것이
    아니라 일어난 것을 인용한다. 떨어진 단계는 거기 없고 그 이유는 로그에 있다 —
    ``dropped_hyperparams``와 같은 계약이다.

    ``names``는 마지막 단계 뒤의 인코딩된 열 이름이다. 돌려주는 이유는 호출자가 그것을 적합된
    파이프라인에서 sklearn 자신의 ``get_feature_names_out``에 대고 묶는 검사를 갖고 있기 때문이다.
    이 모듈은 배치를 적합 전에 계산하고, sklearn과 조용히 어긋난 tracker는 뒤 단계의 ``columns``를
    엉뚱한 열로 보낸다.
    """
    steps: list[tuple[str, Any]] = []
    applied: list[str] = []
    names = list(columns)
    seen: dict[str, int] = {}
    # 행렬 어딘가에 NaN이 아직 있을 수 있는지. ``interactions``가 그것을 받을 수 없는 유일한 단계이고,
    # ``fit`` 안에서 알아내는 대신 이것을 읽는다.
    #
    # 시작값이 ``True``가 아니라 ``native_nan``인 것이 미묘한 지점 전부다.
    #
    # 이것을 지우는 것은 실제 전략으로 *모든* 열을 덮는 ``impute`` 단계뿐이다. passthrough 그룹이
    # 있거나 remainder가 ``none``인 것은 NaN을 지키자고 청하는 것이다.
    nan_possible = native_nan
    for entry in spec if isinstance(spec, list) else []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("step") or "").strip().lower()
        if name not in STEPS:
            log.write(f"pipeline: unknown step {entry.get('step')!r}, dropped")
            continue
        if name == STEP_IMPUTE:
            built = _impute_step(entry, names, native_nan, log)
            if built is not None and not _leaves_nan(entry, native_nan):
                nan_possible = False
        elif name == STEP_INTERACTIONS:
            built = _interactions_step(entry, names, log, nan_possible)
        elif name == STEP_SCALE:
            built = _scale_step(entry, names, log)
        else:
            built = _appender_step(name, entry, names, log)
        if built is None:
            continue
        transformer, names, label = built
        count = seen.get(name, 0)
        seen[name] = count + 1
        # 첫 등장은 맨 이름을 지키고 되풀이는 번호가 붙는다.
        steps.append((name if not count else f"{name}_{count + 1}", transformer))
        applied.append(f"{name}({label})")
    return steps, names, applied
