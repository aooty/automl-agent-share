"""Ordered feature steps, declared in JSON, built from a closed whitelist.

Roles:

* Column selection — map plan column names to encoded positions.
* Step builders — build one sklearn step, or drop it with a log.
* Pipeline assembly — declared list to Pipeline steps, names, and echo.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .features import LEVEL_SEPARATOR

# Closed set: an unknown step is dropped, never imported.
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

# Must run before any imputer: imputing leaves no NaN.
APPENDING_STEPS = frozenset({STEP_MISSING_INDICATOR, STEP_MISSING_COUNT})

INDICATOR_SUFFIX = "__missing"
COUNT_COLUMN = "missing_count"
# A space, as ``PolynomialFeatures`` spells it
INTERACTION_JOIN = " "

IMPUTE_STRATEGIES = ("median", "mean", "most_frequent", "constant")
# Keep NaN, for NaN-splitting families
IMPUTE_NONE = "none"
# Also used when the named strategy cannot be used.
DEFAULT_IMPUTE = "median"

# Only degree offered; others drop the step
INTERACTION_DEGREE = 2
# Above this the step is dropped, avoiding OOM
INTERACTION_MAX_COLUMNS = 50


# --- Role: column selection -------------------------------------------------------


def encoded_positions(
    columns: list[str], names: Any
) -> tuple[list[int], list[str]]:
    """Return ``(positions, unknown)`` for plan names; ``city`` picks all its levels.

    Unknown names are returned, not raised"""
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
    """_rest | Column selection: positions not in ``taken``, in order."""
    return [position for position in range(len(columns)) if position not in taken]


def _partial(
    name: str, transformer: Any, positions: list[int], columns: list[str], appended: Any = ()
) -> tuple[Any, list[str]]:
    """_partial | Column selection: run a transform on some columns; name the outputs."""
    from sklearn.compose import ColumnTransformer

    # Order: selected, then ``appended``, then the remainder.
    step = ColumnTransformer([(name, transformer, positions)], remainder="passthrough", sparse_threshold=0.0)
    names = [columns[position] for position in positions]
    names += list(appended)
    names += [columns[position] for position in _rest(columns, positions)]
    return step, names


def _selection(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[list[int], str]:
    """_selection | Column selection: a step's columns as positions and a label."""
    raw = spec.get("columns")
    if raw is None:
        return list(range(len(columns))), "all"
    positions, unknown = encoded_positions(columns, raw)
    if unknown:
        log.write(f"{spec.get('step')}: no such column(s) {sorted(unknown)}, ignored")
    return positions, ", ".join(columns[position] for position in positions) or "none"


# --- Role: step builders ---------------------------------------------------------


def _impute_step(
    spec: dict[str, Any], columns: list[str], native_nan: bool, log: Any
) -> tuple[Any, list[str], str] | None:
    """_impute_step | Step builders: one ``impute`` step, a default plus column groups."""
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
            # Nothing to build: every column keeps its NaN.
            return None
        return make(remainder_strategy, 0.0), list(columns), remainder_strategy

    step = ColumnTransformer(
        transformers,
        remainder="passthrough" if remainder_strategy == IMPUTE_NONE else make(remainder_strategy, 0.0),
        # Dense output, as in :func:`_partial`.
        sparse_threshold=0.0,
    )
    # Transformer columns in order, then the remainder.
    names = [columns[position] for _name, _transformer, group in transformers for position in group]
    names += [columns[position] for position in _rest(columns, taken)]
    return step, names, f"{remainder_strategy} ({'; '.join(labels)})"


def _leaves_nan(spec: dict[str, Any], native_nan: bool) -> bool:
    """_leaves_nan | Step builders: whether this ``impute`` spec keeps NaN on purpose."""
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
    """_interactions_step | Step builders: add pairwise products of the named columns."""
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
        # All columns: inputs, then pairs in ``combinations`` order.
        return poly, list(columns) + pairs, label
    step, names = _partial("interact", poly, positions, columns, pairs)
    return step, names, label


def _scale_step(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[Any, list[str], str] | None:
    """_scale_step | Step builders: standard-scale the named columns."""
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
    """_appender_step | Step builders: build ``missing_indicator`` or ``missing_count``."""
    from sklearn.preprocessing import FunctionTransformer

    # Live in ``features`` so pickles load anywhere
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
        # ``None`` keeps ``all`` meaning all columns later.
        FunctionTransformer(append_missing_indicator, kw_args=None if every else {"positions": positions}),
        [*columns, *(f"{columns[position]}{INDICATOR_SUFFIX}" for position in positions)],
        label,
    )


# --- Role: pipeline assembly -----------------------------------------------------


def build_steps(
    spec: Any, columns: list[str], log: Any, *, native_nan: bool = False
) -> tuple[list[tuple[str, Any]], list[str], list[str]]:
    """Return ``(steps, names, applied)`` for a declared step list.

    Unusable steps are dropped with a reason in ``log``; ``applied`` lists what ran."""
    steps: list[tuple[str, Any]] = []
    applied: list[str] = []
    names = list(columns)
    seen: dict[str, int] = {}
    # Starts at ``native_nan``, not ``True``
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
        # Repeats get a number
        steps.append((name if not count else f"{name}_{count + 1}", transformer))
        applied.append(f"{name}({label})")
    return steps, names, applied
