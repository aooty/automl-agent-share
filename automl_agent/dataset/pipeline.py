"""An ordered list of feature-space transforms, declared as JSON and interpreted from a whitelist.

The four flags it replaces say *what* and never *where* or *in which order* — one imputation strategy
for the whole matrix, ``missing_indicator`` over every column or none. So a step here carries a column
list and the list carries an order.

**Nothing is ``exec``'d and nothing is imported by name from the config.** :data:`STEPS` is a closed
set, and a step this module does not know is dropped with a reason rather than resolved.

**The whitelist is short because it was measured** (``docs/PIPELINE-STEPS.md``): of seven candidates,
two moved the ranking with the interval clear of zero and **five are deliberately absent, each absence
a measurement** — monotone transforms cannot change a tree's splits, feature selection and PCA lose
ranking on every family tried, ``VarianceThreshold`` moves nothing because no encoded column is
constant. Putting them in the registry would invite an iteration on a lever measured to lose.

**The two that survive compose, and that is why this is an ordered list and not two more flags**: the
interaction step consumes the indicator columns the imputation step appended, so together they beat
their own sum. Both gains are on the *ranking* axis, which a fixed 0.5 cut does not show — what they
raise is the ceiling, and ``tune_threshold`` collects it.

**Column identity is by name, resolved against the fitted schema, and the names are the encoded
ones** — a one-hot source column became several, and a plan naming ``city`` means all of them. That
resolution is the only reason this module knows about :mod:`automl_agent.dataset.features`.

Import-light: the orchestrator imports :data:`STEPS` to allowlist a plan before the subprocess exists;
sklearn is imported inside :func:`build_steps`. Rationale: ``docs/rationale.md``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .features import LEVEL_SEPARATOR

# What a step may be called. A closed set: the interpreter resolves nothing by name from the
# config, so an unknown step is a dropped step and not an import.
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

# The two steps that append rather than replace, and therefore have to come before any imputer:
# after imputation there is no NaN left to mark. Named as a set because the interpreter uses it
# to place a default imputer when a spec declares none.
APPENDING_STEPS = frozenset({STEP_MISSING_INDICATOR, STEP_MISSING_COUNT})

# Suffixes for the columns a step creates. Read by nothing but a human and the tracker's own
# test — the point of naming them is that two steps in sequence can be told apart in the log.
INDICATOR_SUFFIX = "__missing"
COUNT_COLUMN = "missing_count"
# A space, which is ``PolynomialFeatures``' own spelling of a product and not the ``*`` that
# reads better. Agreeing with sklearn is what lets the tracker's test be exact equality against
# ``get_feature_names_out`` with nothing normalised away — and a join this module invented would
# be a second spelling of the same column for no one's benefit.
INTERACTION_JOIN = " "

IMPUTE_STRATEGIES = ("median", "mean", "most_frequent", "constant")
# Not a SimpleImputer strategy — it leaves the columns alone so a NaN-splitting family can use
# them. Honoured as the *remainder* strategy and inside a group, which is the combination the
# flag version could not express: keep the NaN where missingness is informative and put an
# explicit 0 plus an indicator where it is not.
IMPUTE_NONE = "none"
DEFAULT_IMPUTE = "median"

# Only degree 2. Degree 3 on 14 encoded columns is 470 columns and on 60 it is 37,880, and no
# measurement here justifies either; the parameter exists so a plan that names the degree it
# means is told which one ran rather than silently getting a different one.
INTERACTION_DEGREE = 2
# Above this many input columns the interaction step is declined. ``guard_memory`` prices the
# matrix that arrives, not the one a step creates, so this is the only thing standing between a
# wide encoding and an out-of-memory kill *after* the guard passed: 50 columns expand to 1,275,
# which at 30,000 rows is 306 MB of float64 — the same order as the matrices the guard already
# admits. 200 columns would be 20,100 and 4.8 GB.
INTERACTION_MAX_COLUMNS = 50


def encoded_positions(
    columns: list[str], names: Any
) -> tuple[list[int], list[str]]:
    """``(positions, unknown)`` for source column ``names`` inside the encoded layout.

    A name matches an encoded column outright, or matches every encoded column a one-hot source
    produced — ``city`` selects ``city=seoul``, ``city=busan`` and ``city=<missing>``, because a
    plan asking to impute ``city`` means the column it saw in the card and not one of its levels.
    Level names are still accepted individually, since they are what the log prints back.

    ``unknown`` is returned rather than raised. A plan naming a column the file does not have is
    a mistake worth reporting in ``applied_pipeline``, not worth an iteration: the rest of the
    step still describes something the executor can do.

    Positions are sorted and de-duplicated, so ``["city", "city=seoul"]`` is one selection and
    not a column listed twice — which in a ``ColumnTransformer`` would be a fitted duplicate.
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


def _selection(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[list[int], str]:
    """``(positions, label)`` for a step's ``columns`` key. Absent means every column.

    The label is what the echo and the log print, and ``all`` is a distinct answer from a list
    that happens to name everything: a spec written as ``all`` keeps meaning all when the next
    file has a column more.
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
    """One ``impute`` step: a default strategy plus any number of named column groups.

    With no groups this is a plain ``SimpleImputer`` and the column names are unchanged, which is
    the flag version's behaviour exactly. With groups it is a ``ColumnTransformer``, and then the
    output order is the transformer order followed by the remainder — so the names are tracked
    rather than assumed, and :func:`build_steps`' caller has a test pinning the tracking against
    sklearn's own ``get_feature_names_out``.

    ``strategy: none`` on a family that cannot take a NaN is downgraded rather than fatal, on the
    same terms the flag version used: a plan guessing wrong about a family should cost a log
    line, not an iteration.
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
            # Nothing built and nothing covered: every column keeps whatever NaN it had.
            return None
        return make(remainder_strategy, 0.0), list(columns), remainder_strategy

    rest = [position for position in range(len(columns)) if position not in taken]
    step = ColumnTransformer(
        transformers,
        remainder="passthrough" if remainder_strategy == IMPUTE_NONE else make(remainder_strategy, 0.0),
        # Off, because this module tracks the output names itself: sparse output has no names to
        # track and the estimators here take dense input anyway.
        sparse_threshold=0.0,
    )
    # ColumnTransformer emits each transformer's columns in transformer order, then the
    # remainder in the original order. Tracked, not assumed — see this module's docstring.
    names = [columns[position] for _name, _transformer, group in transformers for position in group]
    names += [columns[position] for position in rest]
    return step, names, f"{remainder_strategy} ({'; '.join(labels)})"


def _leaves_nan(spec: dict[str, Any], native_nan: bool) -> bool:
    """Whether this ``impute`` step deliberately leaves a NaN somewhere.

    Read off the *spec* rather than the built ``ColumnTransformer``, because the question is about
    what was asked for and the answer is needed before the next step is built. Two ways to leave
    one: a ``none`` remainder, or a group with ``strategy: none``. Both are only honoured on a
    family that splits on NaN natively — off it they are downgraded, and then nothing is left.
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
    """Degree-2 pairwise products of the named columns, appended to them.

    Declined above :data:`INTERACTION_MAX_COLUMNS` inputs rather than allowed to expand into an
    out-of-memory kill the pre-flight guard cannot see — the guard prices the matrix that
    arrives, and this step is what makes a different one.

    Also declined while a NaN may still be in the matrix, and that one cost a real iteration to
    find. ``PolynomialFeatures`` is the one step in this registry that refuses a NaN — a
    ``StandardScaler`` disregards them in fit and maintains them in transform — so a plan that
    asked for ``strategy: none`` to let the tree split on missingness and *then* for interactions
    built a pipeline that raised inside ``fit``. That is the one thing this executor is not
    allowed to do with a request it cannot honour: narrowing costs a log line, dying costs the
    attempt.
    """
    from sklearn.compose import ColumnTransformer
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
    if len(positions) == len(columns):
        # Every column, so no ColumnTransformer and no remainder to place. PolynomialFeatures
        # emits the inputs in order, then the pairs in ``combinations`` order.
        names = list(columns) + [
            f"{columns[a]}{INTERACTION_JOIN}{columns[b]}" for a, b in combinations(positions, 2)
        ]
        return poly, names, label
    step = ColumnTransformer([("interact", poly, positions)], remainder="passthrough", sparse_threshold=0.0)
    names = [columns[position] for position in positions]
    names += [f"{columns[a]}{INTERACTION_JOIN}{columns[b]}" for a, b in combinations(positions, 2)]
    names += [columns[position] for position in range(len(columns)) if position not in positions]
    return step, names, label


def _scale_step(spec: dict[str, Any], columns: list[str], log: Any) -> tuple[Any, list[str], str] | None:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    positions, label = _selection(spec, columns, log)
    if not positions:
        return None
    if len(positions) == len(columns):
        return StandardScaler(), list(columns), label
    step = ColumnTransformer(
        [("scale", StandardScaler(), positions)], remainder="passthrough", sparse_threshold=0.0
    )
    names = [columns[position] for position in positions]
    names += [columns[position] for position in range(len(columns)) if position not in positions]
    return step, names, label


def _appender_step(
    name: str, spec: dict[str, Any], columns: list[str], log: Any
) -> tuple[Any, list[str], str] | None:
    """``missing_indicator`` or ``missing_count`` — the two steps that append columns.

    Both are ``FunctionTransformer`` over a function in
    :mod:`automl_agent.dataset.features`, which is where they have to live: that module is
    importable under a stable name, and ``scripts/train.py`` runs as ``__main__``, so a function
    defined there pickles as ``__main__.append_missing_indicator`` and resolves in no other
    process. ``positions`` rides along in ``kw_args``, which is pickled as data.
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
        # ``None`` rather than the full list when it is every column, so a spec that says
        # ``all`` pickles the same way the flag version did and keeps meaning all.
        FunctionTransformer(append_missing_indicator, kw_args=None if every else {"positions": positions}),
        [*columns, *(f"{columns[position]}{INDICATOR_SUFFIX}" for position in positions)],
        label,
    )


def build_steps(
    spec: Any, columns: list[str], log: Any, *, native_nan: bool = False
) -> tuple[list[tuple[str, Any]], list[str], list[str]]:
    """``(steps, names, applied)`` for a declared spec. ``steps`` goes straight into a Pipeline.

    ``applied`` is the echo: one rendered line per step that really ran, in the order it ran, so
    a report quotes what happened rather than what was asked for. A step that was dropped is
    absent from it and its reason is in the log — the same contract ``dropped_hyperparams`` has.

    ``names`` is the encoded column names after the last step. Returned because the caller has a
    check that pins it against sklearn's own ``get_feature_names_out`` on a fitted pipeline: this
    module computes the layout ahead of the fit, and a tracker that silently disagreed with
    sklearn would send a later step's ``columns`` at the wrong columns.
    """
    steps: list[tuple[str, Any]] = []
    applied: list[str] = []
    names = list(columns)
    seen: dict[str, int] = {}
    # Whether a NaN may still be anywhere in the matrix. ``interactions`` is the one step that
    # cannot take one, and it reads this rather than finding out inside ``fit``.
    #
    # The start value is ``native_nan`` and not ``True``, which is the whole subtlety. Off a
    # NaN-splitting family an imputer is *guaranteed* to precede everything: either the spec
    # declared one, or :func:`automl_agent.scripts.train._wrap_declared` inserts one, and it
    # inserts it after the appenders — which is before any interactions step. On a NaN-splitting
    # family nothing is inserted, because the NaN is the point, so a spec that never covers the
    # columns leaves one for ``PolynomialFeatures`` to refuse.
    #
    # Only an ``impute`` step that covers *every* column with a real strategy clears it: one with
    # a passthrough group, or a ``none`` remainder, is asking to keep the NaN.
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
        # The first occurrence keeps the bare name, so ``describe_preprocessing`` still finds
        # ``impute`` and ``scale`` and ``applied_preprocessing`` keeps its shape for every reader
        # written before this module. Repeats are numbered, because a Pipeline needs unique
        # step names and a spec is allowed to impute twice.
        steps.append((name if not count else f"{name}_{count + 1}", transformer))
        applied.append(f"{name}({label})")
    return steps, names, applied
