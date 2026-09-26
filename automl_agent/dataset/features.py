"""Feature columns: which ones are used, and how they become numbers.

Roles:

* Environment record — library versions at fit, compared at replay.
* Column kinds — numeric, text-like, or narrow enough to one-hot.
* Column stats — missing rates and codes at fit, compared later.
* Schema — fit the column layout, apply it, or both.
* Drift messages — drift and encoding reports as lines for people.
* Missing-value columns — stateless missing indicator and missing count steps.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - types only; pandas is loaded only in subprocesses
    import pandas as pd

# Wider text columns are dropped, not one-hot
MAX_ONEHOT_CARDINALITY = 50

# As in ``city=seoul``.
LEVEL_SEPARATOR = "="

# Brackets keep it apart from a real text "nan".
MISSING_LEVEL = "<missing>"

# Newer refused, older read
SCHEMA_VERSION = 2

# Raise only when a key changes meaning
MIN_SCHEMA_VERSION = 1

# Tells readers of older schemas which checks did not run.
SCHEMA_CHECKS_ADDED: dict[int, tuple[str, ...]] = {
    2: ("environment", "column_stats"),
}

# Smallest reported rate change; two floors differ
MISSING_RATE_SHIFT = 0.05
SENTINEL_RATE_SHIFT = 0.01


class FeatureSchemaMismatch(ValueError):
    """Raised when new rows cannot be encoded as the fitted model needs.

    Raises because every workaround is silent"""


# --- Role: environment record -----------------------------------------------------
# Pickles can predict differently under other versions

ENVIRONMENT_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "joblib")


def current_environment() -> dict[str, str]:
    """Map package name to the version running now; always has ``python``.

    Packages not installed are left out, so records compare cleanly."""
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
    """_same_version | Environment record: are two versions close enough to stay quiet?"""
    if fit == now:
        return True
    if package == "python":
        return fit.split(".")[:2] == now.split(".")[:2]
    return False


def environment_drift(recorded: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """List ``{"package", "fit", "now"}`` for each changed version.

    Reports only; never refuses"""
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
    """List the checks a schema of this ``version`` has no data for."""
    return [
        name
        for added, names in sorted(SCHEMA_CHECKS_ADDED.items())
        if version < added
        for name in names
    ]


# --- Role: column kinds -----------------------------------------------------------


def is_numeric_column(series: pd.Series) -> bool:
    """Tell whether the column is numeric or boolean, needing no encoding."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series)) or bool(
        pd.api.types.is_bool_dtype(series)
    )


def is_text_like_column(series: pd.Series) -> bool:
    """Tell whether the column is text or categorical, across pandas versions.

    Datetimes are excluded: one-hot on timestamps means nothing."""
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
    """Tell whether a text column is narrow enough to one-hot.

    ``distinct`` counts non-missing values; the missing level does not count."""
    return is_text_like_column(series) and 0 < distinct <= MAX_ONEHOT_CARDINALITY


# --- Role: column stats -----------------------------------------------------------
# Catches meaning changes; report, never convert


def column_stats(features: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Record each column's missing rate and suspected missing-value codes.

    Rates, so big and small batches compare; codes only from ``NUMERIC_CODES``."""
    stats: dict[str, Any] = {}
    for name in columns:
        series = features[name]
        entry: dict[str, Any] = {"missing_rate": round(float(series.isna().mean()), 4)}
        found = _numeric_sentinels(series)
        if found:
            entry["sentinels"] = found
        stats[str(name)] = entry
    return stats


def rate_changed(fit: float, now: float, rows: int, floor: float = MISSING_RATE_SHIFT) -> bool:
    """Tell whether two rates differ by at least ``floor`` and two standard errors.

    ``False`` when ``rows`` is 0 or less."""
    if rows <= 0:
        return False
    gap = abs(float(now) - float(fit))
    if gap < floor:
        return False
    p = min(max(max(float(fit), float(now)), 0.0), 1.0)
    return gap >= 2.0 * ((p * (1.0 - p) / float(rows)) ** 0.5)


def _sentinel_rates(series: pd.Series, recorded: list[dict[str, Any]]) -> dict[float, float]:
    """_sentinel_rates | Column stats: this batch's rate for each recorded code."""
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
    """_coerce | Column stats: parse a column as numbers, NaN on failure."""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _numeric_sentinels(series: pd.Series, known: Container[float] = ()) -> list[dict[str, Any]]:
    """_numeric_sentinels | Column stats: numeric code findings, minus ``known``."""
    from automl_agent.dataset.sentinels import KIND_NUMERIC_CODE, detect_sentinels

    if not is_numeric_column(series):
        return []
    return [
        {"value": float(item["value"]), "rate": float(item["rate"])}
        for item in detect_sentinels(series)
        if item.get("kind") == KIND_NUMERIC_CODE and float(item["value"]) not in known
    ]


def _compare_column_stats(
    features: pd.DataFrame,
    recorded: Mapping[str, Any] | None,
    required: list[str],
    drift: dict[str, Any],
) -> None:
    """_compare_column_stats | Column stats: fill the two shift lists in ``drift``."""
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
        # New codes skip the rate test
        for item in _numeric_sentinels(series, {float(code["value"]) for code in codes}):
            drift["sentinel_shift"].append(
                {
                    "column": name,
                    "value": float(item["value"]),
                    "fit": 0.0,
                    "now": round(float(item["rate"]), 4),
                }
            )


# --- Role: schema -----------------------------------------------------------------


def build_schema(features: pd.DataFrame) -> dict[str, Any]:
    """Fit the layout: which columns reach the estimator, encoded how, in what order.

    Holds cell values (levels): keep in ``artifacts/``, never in prompts."""
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
                    # Sorted: order follows level names, not row order.
                    "levels": sorted(str(value) for value in series.dropna().unique()),
                    # Only with NaN at fit; else all zeros.
                    "missing_level": bool(series.isna().any()),
                }
            )
        elif is_text_like_column(series):
            # Dropped and named, so the caller can bucket it.
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
        # Not layout: library versions, see ``environment_drift``.
        "environment": current_environment(),
        # Not layout: what the columns held.
        "column_stats": column_stats(features, sources),
    }


def _level_columns(spec: Mapping[str, Any]) -> list[str]:
    """_level_columns | Schema: encoded names for one one-hot column, in order."""
    column = str(spec["column"])
    names = [f"{column}{LEVEL_SEPARATOR}{level}" for level in spec.get("levels") or []]
    if spec.get("missing_level"):
        names.append(f"{column}{LEVEL_SEPARATOR}{MISSING_LEVEL}")
    return names


def encode_with_schema(
    features: pd.DataFrame, schema: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a saved layout; return ``(matrix, drift)`` with the schema's exact columns.

    Anything else is reported in ``drift``; raises FeatureSchemaMismatch if unusable."""
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
        # Coerce, not raise: one bad cell becomes missing.
        converted = pd.to_numeric(series, errors="coerce")
        failed = int((converted.isna() & series.notna()).sum())
        if failed:
            drift["coerced_numeric"].append({"column": name, "rows": failed})
        blocks[name] = converted

    for spec in one_hot:
        name = str(spec["column"])
        series = features[name]
        isna = series.isna()
        # As text, so ``3`` and ``"3"`` are one level.
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
        # Self-inconsistent schema: edited by hand or cut short.
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
    """Fit a layout on this frame and apply it; return ``(matrix, report)``.

    ``report`` has counts and names only: it reaches the card and prompts."""
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


# --- Role: drift messages ---------------------------------------------------------

SHOWN_NAMES = 5


def _shown(names: list[str]) -> str:
    """_shown | Drift messages: names on one line, the rest as a count."""
    more = f" 외 {len(names) - SHOWN_NAMES}개" if len(names) > SHOWN_NAMES else ""
    return ", ".join(names[:SHOWN_NAMES]) + more


def describe_drift(drift: Mapping[str, Any]) -> list[str]:
    """Turn a drift report into Korean lines, one per finding.

    Level names are fine: this runs in ``predict.py``, with no prompt."""
    lines: list[str] = []
    for item in drift.get("coerced_numeric") or []:
        lines.append(
            f"수치 컬럼 '{item['column']}'의 {item['rows']}행이 숫자로 읽히지 않아 결측으로 처리했습니다"
        )
    for item in drift.get("unseen_levels") or []:
        lines.append(
            f"'{item['column']}'에 학습 때 없던 범주 {len(item['levels'])}개"
            f"({_shown(item['levels'])}) — {item['rows']}행이 이 컬럼에서 전부 0으로 인코딩됩니다"
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
        lines.append(f"{why} {len(names)}개는 무시했습니다 ({_shown(names)})")
    lines += _describe_schema_drift(drift)
    return lines


CHECK_LABELS: dict[str, str] = {
    "environment": "라이브러리 버전 대조",
    "column_stats": "컬럼 결측률·결측 코드 대조",
}


def _describe_schema_drift(drift: Mapping[str, Any]) -> list[str]:
    """_describe_schema_drift | Drift messages: lines on versions and skipped checks."""
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
    """Describe an encoding report as one Korean line."""
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
        line += f"; {why} 제외 {len(names)}개 ({_shown(names)})"
    return line


# --- Role: missing-value columns --------------------------------------------------
# Not in train.py: ``__main__`` breaks pickles


def _missing_mask(x: Any) -> Any:
    """_missing_mask | Missing-value columns: boolean NaN mask of the matrix."""
    import numpy as np

    return np.isnan(np.asarray(x, dtype=float))


def append_missing_indicator(x: Any, positions: Any = None) -> Any:
    """Append a 0/1 NaN column per selected column; stateless

    ``positions=None`` means all columns, which saved models still replay."""
    import numpy as np

    mask = _missing_mask(x).astype(float)
    if positions is not None:
        mask = mask[:, list(positions)]
    return np.hstack([x, mask])


def append_missing_count(x: Any) -> Any:
    """Append one column: the NaN count per row; stateless"""
    import numpy as np

    return np.hstack([x, _missing_mask(x).sum(axis=1, keepdims=True).astype(float)])
