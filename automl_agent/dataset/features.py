"""Feature columns: which the executor can use, and how the rest become numbers.

**Why this is one module and not two implementations.** The baseline in
:mod:`automl_agent.scripts.profile` and every attempt in :mod:`automl_agent.scripts.train`
have to see the *same* feature matrix, or the bar derived from one is not comparable to the
scores measured against the other — the same argument
:func:`automl_agent.scoring.splits.protocol_mismatch` makes about the rows. Both scripts call
:func:`encode_features` and neither owns a column policy of its own.

**What changed and why.** Until now both scripts did
``select_dtypes(include=["number", "bool"])``, so every string column was dropped. On the
MIMIC sample that cost nothing (all 14 features are numeric) and on ordinary business
tables it costs most of the signal. Now a low-cardinality categorical column is one-hot
encoded and a high-cardinality one is dropped and *named*, because:

- **One-hot, not ordinal.** Ordinal codes invent an order the data does not have. Trees can
  partially work around it; the linear baseline cannot, and the baseline is what the goal
  threshold is derived from — so an ordinal code would move the bar for a reason that is an
  artefact of the encoding.
- **Dropped above** :data:`MAX_ONEHOT_CARDINALITY`. A free-text or identifier column
  one-hots into thousands of columns, which is a memory-guard failure at best and pure
  overfitting surface at worst. Dropping is recoverable — the caller can bucket the column
  and re-run — and it is disclosed rather than silent.
- **Missing becomes its own level.** For a categorical, "absent" is usually a fact about
  the record rather than noise, and the alternative (an all-zero row) is indistinguishable
  from a category the encoder never saw. This is the opposite of the rule
  :func:`automl_agent.dataset.targets.encode_target` applies to the *target*, where an invented
  class would be a label nobody assigned.

**On fitting the vocabulary over all rows.** The level set is learned from the whole column
rather than from the training split alone. That is deliberate and bounded: it uses feature
values only and never the target, so unlike choosing a decision threshold on the holdout it
cannot inflate a score. What it buys is that the encoded matrix is a pure function of the
file, so the profiler's baseline and the trainer's attempts index the same columns. The
statistics that *would* leak — the impute median and the scaler's mean and variance — stay
inside the estimator pipeline and are still fitted on training rows only.

**Why fitting and transforming are two functions.** "A pure function of the file" is exactly
what makes a fitted model unusable on a *second* file. The level set, the column order and
whether a column got its own missing indicator were all decided by the rows in front of the
encoder, so re-encoding new rows produces a different matrix — a different width if a level
is absent, or, worse, the same width with the columns meaning different things. A model
scored through that is not wrong in a way anything reports. So :func:`build_schema` fits the
layout, :func:`encode_with_schema` applies a layout that already exists, and
:func:`encode_features` is the two of them in a row for the callers that legitimately want
both (the profiler and each training attempt, which own their file). The schema is written
next to ``model.joblib`` and is what :mod:`automl_agent.scripts.predict` replays.

**What the schema records beyond the layout.** A correct layout is not the same as a
comparable batch, and the two ways that gap opens are both invisible to a width check: the
library versions that will unpickle the model (:func:`environment_drift`) and what the
columns actually contained — how much was missing, and which conventional missing codes were
present (:func:`column_stats`). Both are recorded at fit time, compared at replay time, and
*reported*, never acted on. The refusals stay where a wrong answer would otherwise be
computed in silence; these are facts a human has to weigh.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cost only, and pandas is subprocess-only
    import pandas as pd

# Above this many distinct values a column is dropped instead of one-hot encoded. 50 is
# chosen against what it protects: the memory guard works on the encoded matrix, so the
# cap has to keep a wide table from exploding before that guard can price it. A 12-level
# month, a 50-state code and an ICD chapter all fit; a free-text note or an account id
# does not, and neither is something one-hot encoding was going to help with.
MAX_ONEHOT_CARDINALITY = 50

# The suffix separating a source column from its level in the generated column name. Kept
# explicit so a reader of ``dropped_hyperparams`` or a log line can tell an encoded column
# from a column that was named that way in the file.
LEVEL_SEPARATOR = "="

# The level a missing value becomes. Spelled with brackets so it cannot be confused with a
# column that genuinely contains the text "nan", which is what ``get_dummies`` would name it.
MISSING_LEVEL = "<missing>"

# Bumped when a schema gains something a reader can check. Read by
# :func:`encode_with_schema`, and the rule there is asymmetric on purpose:
#
# * a version *newer* than this constant is refused, because an unknown layout can only be
#   guessed at, and guessing is how a model gets scored on misaligned columns;
# * a version *older* than it is accepted, and the checks it cannot support are named. The
#   alternative — refusing — would mean yesterday's model stops being usable the moment
#   this file is edited, which is a self-inflicted outage rather than a safety property. The
#   layout keys have never changed meaning; every bump so far has only *added* keys.
SCHEMA_VERSION = 2

# The oldest version this build can still read. Raise it only for a change that makes an old
# schema wrong rather than merely thin — a key whose meaning changed. ``0`` is the absence of
# the field, which is not an old schema but a file that is not one.
MIN_SCHEMA_VERSION = 1

# Which checks each version added, keyed by the version that added them. An older schema is
# read, and this is what tells its reader — through ``describe_drift`` — that "no drift found"
# covered less ground than it looks like it did. Silence there would be the failure this whole
# module is arranged against: a reassuring report about a comparison that never ran.
SCHEMA_CHECKS_ADDED: dict[int, tuple[str, ...]] = {
    2: ("environment", "column_stats"),
}

# How far a rate has to move between the fit and a new batch before it is reported. Two
# conditions, both required: this floor, and two standard errors of the larger of the two rates
# at *this batch's* row count. The second is what lets the comparison run on a short batch at
# all instead of being skipped below some row floor — skipping silently is the failure mode
# here, and a bar that widens as the batch shrinks says the same thing without a second rule.
#
# The floors differ because the two findings are different kinds of claim.
#
# A **missing rate** moving is partly ordinary: collection changes, a lab is slow this month.
# Five points is a twentieth of the column — enough that the imputer is now inventing a
# visibly different share of the input, not enough to fire on the wobble a real export has
# every month.
#
# A **missing code's** share moving is not ordinary at all. ``-9999`` is either this extract's
# convention for "absent" or it is not; a batch where its share moved is a batch that made a
# different choice, and the value goes into the model as the number -9999 either way. So the
# floor is nominal and the noise term does the work: on the 400-row probe this is what catches
# an 8% ``-9999`` column becoming an 8% empty column, which the missing-rate floor alone
# suppressed — and that case is not hypothetical (see ``automl_agent.dataset.sentinels``).
MISSING_RATE_SHIFT = 0.05
SENTINEL_RATE_SHIFT = 0.01


class FeatureSchemaMismatch(ValueError):
    """New rows cannot be encoded the way the fitted model requires.

    Raised rather than worked around. Every alternative is silent: a fabricated all-NaN
    column lets the imputer invent a value for a feature the file does not have, and
    reordering to whatever the new file offers produces a full-width matrix whose columns
    mean something else. Both score, and neither reports anything.
    """


# --------------------------------------------------------------------------- #
# The environment the layout was fitted in
# --------------------------------------------------------------------------- #
#
# A schema pins which column of the matrix was which. It does not pin the code that turns that
# matrix into a prediction, and that code is on disk too: ``model.joblib`` is a pickle of
# sklearn objects, restored by whatever sklearn is installed when it is opened. sklearn itself
# only warns (``InconsistentVersionWarning``, to stderr, easily lost in a subprocess log), and
# the failure it warns about is not a crash — an attribute that moved between versions is
# restored as a default, and the model predicts, differently, in silence. Same shape of defect
# as a misaligned column, one layer down.
#
# So the versions are recorded at fit time and compared at replay time. Recorded by name
# through ``importlib.metadata`` rather than by importing the packages, so this stays callable
# from the orchestrator process, which must not import pandas.

ENVIRONMENT_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "joblib")


def current_environment() -> dict[str, str]:
    """The versions this process would fit or replay a model with.

    A package that is not installed is omitted rather than recorded as absent: the comparison
    is between two recordings of the same shape, and "missing here" would otherwise read as a
    change. ``python`` is always present.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {"python": platform.python_version()}
    for name in ENVIRONMENT_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - all four are hard dependencies
            continue
    return found


def _same_version(package: str, fit: str, now: str) -> bool:
    """Whether two version strings are close enough not to be worth a line.

    Exact for the libraries, because sklearn's own compatibility bar is exact and a patch
    release of pandas has changed a dtype default before. Major-minor for ``python``: patch
    releases do not change the pickle protocol or any ABI, so reporting them would produce a
    line on an ordinary interpreter upgrade and teach the reader to skip the block.
    """
    if fit == now:
        return True
    if package == "python":
        return fit.split(".")[:2] == now.split(".")[:2]
    return False


def environment_drift(recorded: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Recorded versions that differ from the ones running now. Empty when they agree.

    Reported, never refused. A version difference is not proof the predictions moved — most
    of the time nothing moves — and refusing would make a routine ``pip install`` an outage
    for every model already saved. What it is proof of is that this run cannot be compared to
    the fit's own measurements without checking, which is a sentence a human has to read.
    """
    if not recorded:
        return []
    now = current_environment()
    changed: list[dict[str, str]] = []
    for name, fitted in recorded.items():
        package = str(name)
        running = now.get(package)
        if running is None or _same_version(package, str(fitted), running):
            continue
        changed.append({"package": package, "fit": str(fitted), "now": running})
    return changed


def missing_checks(version: int) -> list[str]:
    """Check names a schema of this ``version`` carries nothing to run."""
    return [
        name
        for added, names in sorted(SCHEMA_CHECKS_ADDED.items())
        if version < added
        for name in names
    ]


def is_numeric_column(series: pd.Series) -> bool:
    """True for the dtypes the estimators consume without any encoding."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series)) or bool(
        pd.api.types.is_bool_dtype(series)
    )


def is_text_like_column(series: pd.Series) -> bool:
    """True for the dtypes a level set can be read off, whatever pandas calls them today.

    Three names for the same thing across versions: pandas 2 gives a column of strings
    ``object``, pandas 3 gives it ``str``, and an explicitly converted one is
    ``CategoricalDtype``. Datetimes are excluded — one-hot over timestamps is not an
    encoding of anything, and the cap would not catch a column with few distinct dates.
    """
    import pandas as pd

    dtype = series.dtype
    if is_numeric_column(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return False
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def is_encodable_column(series: pd.Series, distinct: int) -> bool:
    """True for a categorical column narrow enough to one-hot.

    ``distinct`` is counted over present values, so a column of two levels plus missing is
    two, not three — the missing level is added by the encoder and does not count against
    the cap.
    """
    return is_text_like_column(series) and 0 < distinct <= MAX_ONEHOT_CARDINALITY


# --------------------------------------------------------------------------- #
# What the columns looked like when the layout was fitted
# --------------------------------------------------------------------------- #
#
# The layout catches a column that changed *shape*. It cannot catch a column that kept its
# shape and changed meaning, and the commonest way that happens is the missing convention:
# the training extract wrote ``-9999`` where a measurement was absent, the next month's
# extract writes an empty cell. Both are numeric columns of the same name, both encode into
# the same slot, and nothing so far reports anything — the model was fitted with -9999 as a
# real value and is now handed a median, or the reverse. On the MIMIC sample this is not
# hypothetical: 91,364 cells of ``-9999`` and zero NaN.
#
# So the fit records, per column, how often it was missing and which conventional missing
# codes it carried, and the replay compares. Detected, reported, never converted — the same
# rule ``automl_agent.dataset.sentinels`` states and for the same reason: whether ``-1`` is a code or a
# measurement is not inferable, and rewriting it silently would mean the model is fed rows the
# file does not contain.


def column_stats(features: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Per-column missing rate and suspected missing codes, for the columns that get encoded.

    Rates rather than counts, so a 900-row fit and a 40-row batch are comparable at all. Only
    values on :data:`automl_agent.dataset.sentinels.NUMERIC_CODES` can appear here, which is what keeps
    this out of the "the schema now holds arbitrary cell values" objection — it already holds
    category levels, and this adds no new kind of thing.
    """
    from automl_agent.dataset.sentinels import detect_sentinels

    stats: dict[str, Any] = {}
    for name in columns:
        series = features[name]
        entry: dict[str, Any] = {"missing_rate": round(float(series.isna().mean()), 4)}
        if is_numeric_column(series):
            found = [
                {"value": float(item["value"]), "rate": float(item["rate"])}
                for item in detect_sentinels(series)
                if item.get("kind") == "numeric_code"
            ]
            if found:
                entry["sentinels"] = found
        stats[str(name)] = entry
    return stats


def rate_changed(fit: float, now: float, rows: int, floor: float = MISSING_RATE_SHIFT) -> bool:
    """Whether two rates over ``rows`` rows differ by more than noise and by enough to matter.

    See :data:`MISSING_RATE_SHIFT` for both halves of the bar and for why ``floor`` is a
    parameter rather than one constant.
    """
    if rows <= 0:
        return False
    gap = abs(float(now) - float(fit))
    if gap < floor:
        return False
    p = min(max(max(float(fit), float(now)), 0.0), 1.0)
    return gap >= 2.0 * ((p * (1.0 - p) / float(rows)) ** 0.5)


def _sentinel_rates(series: pd.Series, recorded: list[dict[str, Any]]) -> dict[float, float]:
    """This batch's rate for each code, counted by value rather than re-detected.

    Deliberately not ``detect_sentinels`` for the codes the fit already named: detection is a
    distribution test with row and distinctness floors, so on a short batch it would answer
    "no code here" for a column that plainly has one, and the reader would be told the
    convention changed when it did not. Once the fit has named the value, counting it is exact.
    """
    if not recorded:
        return {}
    rows = int(len(series))
    if rows == 0:
        return {}
    present = series if is_numeric_column(series) else _coerce(series)
    rates: dict[float, float] = {}
    for item in recorded:
        value = float(item["value"])
        rates[value] = float((present == value).sum()) / rows
    return rates


def _coerce(series: pd.Series) -> pd.Series:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _new_sentinels(series: pd.Series, known: set[float]) -> list[dict[str, Any]]:
    """Codes this batch carries that the fit did not record. Detection, so it has floors.

    Under-reports on a short batch — a code needs 20 rows and 5 distinct values before the
    isolation test means anything — and that is the right direction to be wrong in here: the
    alternative is announcing a changed convention because a 12-row file has a low minimum.
    """
    from automl_agent.dataset.sentinels import detect_sentinels

    if not is_numeric_column(series):
        return []
    return [
        {"value": float(item["value"]), "rate": float(item["rate"])}
        for item in detect_sentinels(series)
        if item.get("kind") == "numeric_code" and float(item["value"]) not in known
    ]


def _compare_column_stats(
    features: pd.DataFrame,
    recorded: Mapping[str, Any] | None,
    required: list[str],
    drift: dict[str, Any],
) -> None:
    """Fill ``missing_rate_shift`` and ``sentinel_shift`` in place.

    A no-op when the schema has no ``column_stats`` — a version 1 schema, whose reader is told
    so by ``missing_checks`` rather than by an empty finding list that reads like agreement.
    """
    if not recorded:
        return
    rows = int(len(features))
    for name in required:
        entry = dict(recorded.get(name) or {})
        if not entry:
            continue
        series = features[name]
        fit_missing = float(entry.get("missing_rate") or 0.0)
        now_missing = float(series.isna().mean()) if rows else 0.0
        if rate_changed(fit_missing, now_missing, rows):
            drift["missing_rate_shift"].append(
                {
                    "column": name,
                    "fit": round(fit_missing, 4),
                    "now": round(now_missing, 4),
                }
            )
        codes = [dict(item) for item in entry.get("sentinels") or []]
        rates = _sentinel_rates(series, codes)
        for item in codes:
            value = float(item["value"])
            fit_rate = float(item.get("rate") or 0.0)
            now_rate = float(rates.get(value, 0.0))
            if rate_changed(fit_rate, now_rate, rows, SENTINEL_RATE_SHIFT):
                drift["sentinel_shift"].append(
                    {
                        "column": name,
                        "value": value,
                        "fit": round(fit_rate, 4),
                        "now": round(now_rate, 4),
                    }
                )
        # Not subject to the rate bar. That bar exists to keep a rate from being read off too
        # few rows; a code the fit never recorded at all is not a rate question, and
        # ``detect_sentinels`` has already applied its own floors before answering.
        for item in _new_sentinels(series, {float(code["value"]) for code in codes}):
            drift["sentinel_shift"].append(
                {
                    "column": name,
                    "value": float(item["value"]),
                    "fit": 0.0,
                    "now": round(float(item["rate"]), 4),
                }
            )


def build_schema(features: pd.DataFrame) -> dict[str, Any]:
    """Fit the layout: which columns reach the estimator, as which columns, in which order.

    JSON-serialisable on purpose — this is written to disk beside the fitted model and read
    back by another process. It holds category *levels*, which are cell values, so it lives
    on the private side of the boundary with ``model.joblib`` and the per-row predictions:
    inside ``artifacts/``, never in a state channel, never in a prompt.

    ``columns`` is redundant with ``numeric`` plus ``one_hot`` and is stored anyway. It is the
    matrix layout stated once, so :func:`encode_with_schema` can assert what it produced
    instead of two derivations of the same order being trusted to agree.
    """
    numeric: list[str] = []
    one_hot: list[dict[str, Any]] = []
    too_wide: list[str] = []
    unsupported: list[str] = []
    for column in features.columns:
        series = features[column]
        if is_numeric_column(series):
            numeric.append(str(column))
        elif is_encodable_column(series, int(series.dropna().nunique())):
            one_hot.append(
                {
                    "column": str(column),
                    # Levels sorted so the generated column order is a function of the level
                    # names and not of the row order — two files with the same levels encode
                    # identically.
                    "levels": sorted(str(value) for value in series.dropna().unique()),
                    # Missing kept as missing so the encoder can give it its own column rather
                    # than an all-zero row. Only when there is something to record: an all-zero
                    # column would be a constant feature and would still be priced by the
                    # memory guard.
                    "missing_level": bool(series.isna().any()),
                }
            )
        elif is_text_like_column(series):
            # Nameable and actionable: the caller can bucket it and run again.
            too_wide.append(str(column))
        else:
            unsupported.append(str(column))

    columns = list(numeric)
    for spec in one_hot:
        columns += _level_columns(spec)
    sources = numeric + [str(spec["column"]) for spec in one_hot]
    return {
        "version": SCHEMA_VERSION,
        "numeric": numeric,
        "one_hot": one_hot,
        "dropped_high_cardinality": too_wide,
        "dropped_unsupported_dtype": unsupported,
        "columns": columns,
        "max_cardinality": MAX_ONEHOT_CARDINALITY,
        # Not part of the layout: the code that will replay it. See ``environment_drift``.
        "environment": current_environment(),
        # Not part of the layout either: what the columns contained. See ``column_stats``.
        "column_stats": column_stats(features, sources),
    }


def _level_columns(spec: Mapping[str, Any]) -> list[str]:
    """The encoded column names one source column becomes, in order."""
    column = str(spec["column"])
    names = [f"{column}{LEVEL_SEPARATOR}{level}" for level in spec.get("levels") or []]
    if spec.get("missing_level"):
        names.append(f"{column}{LEVEL_SEPARATOR}{MISSING_LEVEL}")
    return names


def encode_with_schema(
    features: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply an existing layout to new rows. Returns ``(matrix, drift)``.

    The matrix has the schema's columns, in the schema's order, whatever this frame happens to
    contain — which is the whole point. What this frame contains is reported instead:

    ``extra_columns``     not in the training file at all. Ignored, because a model fitted
                          without them cannot use them, and named so the caller knows they were.
    ``ignored_at_fit``    in the training file, and dropped there for a reason the schema
                          records (too many distinct values, or a dtype nothing can encode).
                          Separated from the above because they are different mistakes: one is
                          a column you added, the other is a column you never got to use.
    ``unseen_levels``     a category the fit never saw. All-zero across that column's group,
                          which is the same row the fit would have produced for a level it had
                          no column for — so this is a disclosure, not a failure.
    ``unmatched_missing`` NaN in a column the fit saw no NaN in, so there is no missing column
                          to put it in. Also all-zero, and worth a line: it means the new rows
                          have a gap the training rows did not.
    ``coerced_numeric``   values in a numeric column that are not numbers here. Forced to NaN
                          for the imputer, and counted, because a column that has silently
                          become text is a delimiter or export bug and reads as "all missing".
    ``missing_rate_shift`` a column whose missing rate has moved past
                          :data:`MISSING_RATE_SHIFT`. Nothing about the encoding is wrong;
                          what has changed is how much of the column the estimator's imputer
                          is inventing, which no width check can see.
    ``sentinel_shift``    a conventional missing code the fit recorded and this batch does not
                          have, or the reverse. The batch that switched ``-9999`` for an empty
                          cell encodes perfectly and predicts from different rows.

    Two further keys are not about the rows at all. ``environment_changed`` is the library
    versions the layout was fitted under against the ones running now, and ``missing_checks``
    is what an older schema carries nothing to compare. They ride in this dict because this is
    the one function that reads a foreign schema, and a caller who has to remember a second
    call is a caller who will one day not make it — the same argument the module docstring
    makes about there being one encoder rather than two.

    A source column the schema needs and this frame does not have raises
    :class:`FeatureSchemaMismatch` — every one of them at once, so one run says what to fix.
    """
    import pandas as pd

    version = int(schema.get("version") or 0)
    if not MIN_SCHEMA_VERSION <= version <= SCHEMA_VERSION:
        raise FeatureSchemaMismatch(
            f"feature schema version {version} cannot be read by this build "
            f"(it reads {MIN_SCHEMA_VERSION}..{SCHEMA_VERSION}). "
            + (
                "This schema was written by a newer build; guessing at a layout it describes "
                "and this one does not is how a model gets scored on misaligned columns."
                if version > SCHEMA_VERSION
                else "Re-run training to write a current schema."
            )
        )
    numeric = [str(name) for name in schema.get("numeric") or []]
    one_hot = [dict(spec) for spec in schema.get("one_hot") or []]
    required = numeric + [str(spec["column"]) for spec in one_hot]
    have = {str(name) for name in features.columns}
    absent = [name for name in required if name not in have]
    if absent:
        raise FeatureSchemaMismatch(
            f"{len(absent)} feature column(s) the fitted model needs are not in this file: "
            + ", ".join(repr(name) for name in absent[:20])
            + (f" (+{len(absent) - 20} more)" if len(absent) > 20 else "")
        )

    dropped_at_fit = {
        str(name)
        for key in ("dropped_high_cardinality", "dropped_unsupported_dtype")
        for name in schema.get(key) or []
    }
    seen_at_fit = set(required) | dropped_at_fit
    drift: dict[str, Any] = {
        "rows": int(len(features)),
        "extra_columns": [str(name) for name in features.columns if str(name) not in seen_at_fit],
        "ignored_at_fit": [str(name) for name in features.columns if str(name) in dropped_at_fit],
        "unseen_levels": [],
        "unmatched_missing": [],
        "coerced_numeric": [],
        "missing_rate_shift": [],
        "sentinel_shift": [],
        "schema_version": version,
        "missing_checks": missing_checks(version),
        "environment_changed": environment_drift(schema.get("environment")),
    }
    _compare_column_stats(features, schema.get("column_stats"), required, drift)

    blocks: dict[str, Any] = {}
    for name in numeric:
        series = features[name]
        if is_numeric_column(series):
            blocks[name] = series
            continue
        # Not a number here even though it was at fit time. ``coerce`` rather than raise: one
        # unparseable cell in a million-row export should become a missing value, not a refusal.
        # The count is what distinguishes that from a column that is now entirely text.
        converted = pd.to_numeric(series, errors="coerce")
        failed = int((converted.isna() & series.notna()).sum())
        if failed:
            drift["coerced_numeric"].append({"column": name, "rows": failed})
        blocks[name] = converted

    for spec in one_hot:
        name = str(spec["column"])
        series = features[name]
        isna = series.isna()
        # Compared as text, the way the levels were recorded, so an integer code and the string
        # of the same code are one level rather than two.
        as_object = series.astype("object")
        as_text = as_object.where(isna, as_object.astype(str))
        levels = [str(level) for level in spec.get("levels") or []]
        known = pd.Series(False, index=features.index)
        for level in levels:
            hit = as_text == level
            known = known | hit
            blocks[f"{name}{LEVEL_SEPARATOR}{level}"] = hit.astype("float64")
        if spec.get("missing_level"):
            blocks[f"{name}{LEVEL_SEPARATOR}{MISSING_LEVEL}"] = isna.astype("float64")
        elif bool(isna.any()):
            drift["unmatched_missing"].append({"column": name, "rows": int(isna.sum())})
        stranded = ~known & ~isna
        if bool(stranded.any()):
            drift["unseen_levels"].append(
                {
                    "column": name,
                    "levels": sorted({str(value) for value in as_text[stranded].tolist()}),
                    "rows": int(stranded.sum()),
                }
            )

    order = [str(name) for name in schema.get("columns") or []]
    if sorted(order) != sorted(blocks):
        # The schema disagreeing with itself, not with the file. Fatal because the two are
        # written by the same function: reaching here means the file was edited or truncated.
        raise FeatureSchemaMismatch(
            f"feature schema is internally inconsistent: it lists {len(order)} encoded "
            f"column(s) but its own column specs produce {len(blocks)}"
        )
    encoded = (
        pd.DataFrame(blocks, index=features.index, columns=order)
        if order
        else features.iloc[:, :0]
    )
    return encoded, drift


def encode_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a layout on this frame and apply it. ``(numeric frame, report)``.

    For the two callers that own their file — the profiler measuring a baseline and each
    training attempt — where fitting the layout and using it are the same act. Anything
    scoring rows a model was *not* fitted on wants :func:`encode_with_schema` and the schema
    that model was saved with.

    ``report`` is aggregate plus column *names*, which are not raw data: the card already
    publishes every column's name, dtype and missing rate. It is what tells a caller that
    the run scored 0.71 on nine of their twelve columns. Deliberately not the schema — this
    dict rides into the dataset card (``privacy.CARD_KEYS`` has ``encoding``) and from there
    into every prompt, and the schema holds cell values.
    """
    schema = build_schema(features)
    encoded, _drift = encode_with_schema(features, schema)
    report = {
        "numeric": len(schema["numeric"]),
        "one_hot_columns": len(schema["one_hot"]),
        "one_hot_levels": int(encoded.shape[1]) - len(schema["numeric"]),
        "dropped_high_cardinality": schema["dropped_high_cardinality"],
        "dropped_unsupported_dtype": schema["dropped_unsupported_dtype"],
        "max_cardinality": MAX_ONEHOT_CARDINALITY,
    }
    return encoded, report


def describe_drift(drift: Mapping[str, Any]) -> list[str]:
    """The drift report as Korean lines for a human. Empty when the new rows matched.

    One line per finding rather than one paragraph, because these are independent facts and a
    caller may act on one and accept another. Level names are included: this runs in
    ``scripts/predict.py``, which renders no prompt, and the names are what make the line
    actionable.
    """
    lines: list[str] = []
    for item in drift.get("coerced_numeric") or []:
        lines.append(
            f"수치 컬럼 '{item['column']}'의 {item['rows']}행이 숫자로 읽히지 않아 결측으로 처리했습니다"
        )
    for item in drift.get("unseen_levels") or []:
        shown = ", ".join(item["levels"][:5])
        more = f" 외 {len(item['levels']) - 5}개" if len(item["levels"]) > 5 else ""
        lines.append(
            f"'{item['column']}'에 학습 때 없던 범주 {len(item['levels'])}개({shown}{more}) — "
            f"{item['rows']}행이 이 컬럼에서 전부 0으로 인코딩됩니다"
        )
    for item in drift.get("unmatched_missing") or []:
        lines.append(
            f"'{item['column']}'의 {item['rows']}행이 결측인데 학습 데이터에는 결측이 없어 "
            "결측 전용 열이 없습니다 — 이 행들도 전부 0입니다"
        )
    for item in drift.get("missing_rate_shift") or []:
        lines.append(
            f"'{item['column']}'의 결측률이 학습 때 {item['fit']:.1%}에서 이 배치 "
            f"{item['now']:.1%}로 바뀌었습니다 — 인코딩은 맞지만 모델이 보는 값의 출처가 "
            "달라졌습니다 (결측 대치를 쓰는 파이프라인이면 그만큼이 학습 때의 대치값입니다)"
        )
    for item in drift.get("sentinel_shift") or []:
        value = f"{item['value']:g}"
        if float(item["now"]) > float(item["fit"]):
            lines.append(
                f"'{item['column']}'에 결측 코드로 의심되는 {value}이 학습 때 "
                f"{item['fit']:.1%}에서 이 배치 {item['now']:.1%}로 늘었습니다 — 이 값은 결측이 "
                f"아니라 숫자 {value}로 모델에 들어갑니다"
            )
        else:
            lines.append(
                f"'{item['column']}'의 결측 코드 {value}가 학습 때 {item['fit']:.1%}였는데 이 "
                f"배치는 {item['now']:.1%}입니다 — 결측 표기 방식이 바뀐 것으로 보입니다. "
                "같은 결측이 학습 때는 극단값으로, 지금은 빈 칸으로 들어가고 있습니다"
            )
    for key, why in (
        ("extra_columns", "학습 파일에 없던 컬럼"),
        ("ignored_at_fit", "학습 때도 인코딩되지 않아 제외된 컬럼"),
    ):
        names = [str(name) for name in drift.get(key) or []]
        if not names:
            continue
        shown = ", ".join(names[:5])
        more = f" 외 {len(names) - 5}개" if len(names) > 5 else ""
        lines.append(f"{why} {len(names)}개는 무시했습니다 ({shown}{more})")
    lines += _describe_schema_drift(drift)
    return lines


# What each check name means to the person reading the line. Kept beside the Korean text
# rather than in ``SCHEMA_CHECKS_ADDED`` so the constant stays a fact about versions.
CHECK_LABELS: dict[str, str] = {
    "environment": "라이브러리 버전 대조",
    "column_stats": "컬럼 결측률·결측 코드 대조",
}


def _describe_schema_drift(drift: Mapping[str, Any]) -> list[str]:
    """The two findings that are about the schema and the interpreter, not about the rows.

    Last in the block, because they are true of every row equally: a reader scanning for
    "which of my rows is affected" should hit those lines first.
    """
    lines: list[str] = []
    changed = list(drift.get("environment_changed") or [])
    if changed:
        shown = ", ".join(f"{item['package']} {item['fit']} → {item['now']}" for item in changed)
        lines.append(
            f"학습 때와 라이브러리 버전이 다릅니다 ({shown}) — 저장된 모델은 pickle이라 "
            "다른 버전에서 열면 예측이 조용히 달라질 수 있습니다. 같은 버전으로 맞추거나, "
            "학습 때의 홀드아웃 점수를 이 배치에서 다시 확인하십시오"
        )
    absent = [str(name) for name in drift.get("missing_checks") or []]
    if absent:
        shown = ", ".join(CHECK_LABELS.get(name, name) for name in absent)
        lines.append(
            f"이 스키마는 version {drift.get('schema_version')}이라 {shown}를 하지 못했습니다 "
            f"(현재 version {SCHEMA_VERSION}) — 위에 없는 항목은 이상이 없다는 뜻이 아니라 "
            "검사하지 않았다는 뜻입니다. 같은 데이터로 다시 학습하면 켜집니다"
        )
    return lines


def describe_encoding(report: dict[str, Any]) -> str:
    """One line for a log or the console. Korean, because a human reads it."""
    line = (
        f"features: 수치 {report.get('numeric')}개"
        f" + 범주형 {report.get('one_hot_columns')}개를 one-hot {report.get('one_hot_levels')}열로"
    )
    for key, why in (
        ("dropped_high_cardinality", f"고유값 {report.get('max_cardinality')}개 초과로"),
        ("dropped_unsupported_dtype", "인코딩 불가 dtype으로"),
    ):
        names = [str(name) for name in report.get(key) or []]
        if not names:
            continue
        shown = ", ".join(names[:5])
        more = f" 외 {len(names) - 5}개" if len(names) > 5 else ""
        line += f"; {why} 제외 {len(names)}개 ({shown}{more})"
    return line


# --------------------------------------------------------------------------- #
# Missingness as columns
# --------------------------------------------------------------------------- #
#
# These two live *here* rather than in ``scripts/train.py``, where they are used, for one
# reason: ``FunctionTransformer`` pickles a function by ``module.qualname``, and
# ``scripts/train.py`` always runs as ``__main__`` in production (``nodes/training.py``
# spawns it as a script). Defined there, a fitted pipeline records
# ``__main__.append_missing_count`` — a name that resolves only inside another process that
# happens to have the same ``__main__``. ``--score-model`` does, so the defect stayed latent;
# anything else that loads ``model.joblib`` gets an ``AttributeError``. Defined here, the
# recorded name is ``automl_agent.dataset.features.append_missing_count``, which is importable from
# anywhere. ``test_the_appenders_do_not_live_in_the_script_that_runs_as_main`` holds the line.
#
# Both are stateless, so the output width is a pure function of the input width. That is what
# makes them safe either side of a train/validation split, and it is why neither is
# ``MissingIndicator(features="missing-only")``: that one fits the column set, so a column
# with no NaN among the validation rows would silently change the matrix width.


def _missing_mask(x: Any) -> Any:
    import numpy as np

    return np.isnan(np.asarray(x, dtype=float))


def append_missing_indicator(x: Any) -> Any:
    """One 0/1 column per input column, appended. Stateless, hence picklable by name.

    Worth knowing before reaching for it: against a family that splits on NaN natively
    (``impute: none`` on ``hist_gbdt`` or ``xgboost``) this is *exactly* redundant. The
    indicator's only split is the NaN branch the tree already has, so it never wins on gain.
    Measured on the MIMIC sample: 14 extra columns reached the model and the predictions were
    bit-identical to not passing them at all. That part reproduced under a second, harder
    split, because it is structural rather than an effect size. Whatever it is worth on the
    imputed path did *not* reproduce — see ``capabilities._MISSINGNESS`` before treating a
    number there as what this buys.
    """
    import numpy as np

    return np.hstack([x, _missing_mask(x).astype(float)])


def append_missing_count(x: Any) -> Any:
    """One column: how many of this row's fields were not measured.

    Not expressible by the NaN splits a native-NaN family already makes — those are per
    column, and this aggregates across them. On clinical rows it stands in for how much
    workup a patient received, which is why it is *available* as a column of its own. Not
    why it is worth one: on the MIMIC sample it changed the predictions (Δp 0.281) and moved
    ``roc_auc`` by 0.0003, and by 0.0000 on top of the indicator.
    """
    import numpy as np

    return np.hstack([x, _missing_mask(x).sum(axis=1, keepdims=True).astype(float)])
