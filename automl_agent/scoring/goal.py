"""Goal threshold in two modes: ``auto`` (relative to the dataset) and ``fixed``.

Roles:

* Goal constants — modes, default margin, limits on every bar.
* Bar derivation — card baseline to bar, with notes.
* Goal building — build the ``goal`` channel, swap unfit metrics.
* Descriptions — Korean lines about the goal or refusals.
"""

from __future__ import annotations

from typing import Any

from .intervals import CI_LEVEL, as_number, contains
from .metrics import ALIASES, METRICS, MINIMIZE, card_task, direction_of, spec, substitute_metric
from .ranking import card_ceiling, passable_margin, required_ks

# --- Role: goal constants ---------------------------------------------------------

MODE_AUTO = "auto"
MODE_FIXED = "fixed"
GOAL_MODES = (MODE_AUTO, MODE_FIXED)
DEFAULT_MODE = MODE_AUTO

# Share of remaining room to close; ``auto`` only.
DEFAULT_MARGIN = 0.25

# Aliases included so sklearn names get the same bar.
BOUNDED_METRICS = frozenset(
    {name for name, spec in METRICS.items() if spec.bounded}
    | {alias for alias, target in ALIASES.items() if METRICS[target].bounded}
)

# Never ask for a perfect score.
CEILING = 0.99
# Gap above chance; a share for error metrics
MIN_LIFT = 0.02

# No entry for unit-bound metrics (mae, rmse).
_BARS = {name: item.fallback for name, item in METRICS.items() if item.fallback is not None}
FALLBACK_THRESHOLDS: dict[str, float] = {
    **_BARS,
    **{alias: _BARS[target] for alias, target in ALIASES.items() if target in _BARS},
}
# Bar for metric names this harness does not know.
FALLBACK_DEFAULT = 0.85


# --- Role: bar derivation ---------------------------------------------------------


def default_bar(metric: str) -> float | None:
    """default_bar | Role: bar when nothing was measured; ``None`` for mae/rmse.

    Unknown names get :data:`FALLBACK_DEFAULT`.
    """
    found = spec(metric)
    if found is None:
        return FALLBACK_DEFAULT
    return found.fallback


def _reference(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """_reference | Role: ``(baseline, chance)`` for ``metric`` from the card."""
    baseline_block = card.get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None

    def score(section: str) -> float | None:
        scores = baseline_block.get(section)
        return as_number(scores.get(metric) if isinstance(scores, dict) else None)

    return score("scores"), score("chance")


def _target(baseline: float, metric: str, direction: str, margin: float) -> float:
    """_target | Role: apply ``margin`` to ``baseline`` as the metric allows."""
    if direction == MINIMIZE:
        # Error stops at 0: margin comes off the score.
        return max(0.0, baseline * (1.0 - margin))
    if metric in BOUNDED_METRICS:
        return baseline + (1.0 - baseline) * margin
    # No scale here, so only a proportional rise.
    return baseline * (1.0 + margin)


def derive_threshold(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[float | None, dict[str, Any]]:
    """derive_threshold | Role: ``(threshold, reference)`` for the ``auto`` bar.

    ``threshold`` is ``None`` only for unit-bound metrics with no baseline.
    """
    direction = direction or direction_of(metric)
    baseline, chance = _reference(card or {}, metric)
    if baseline is None:
        bar = default_bar(metric)
        source = "fallback" if bar is not None else "unset"
        return bar, {"source": source, "metric": metric, "margin": margin}

    threshold = _target(baseline, metric, direction, margin)
    reference: dict[str, Any] = {
        "source": "baseline",
        "metric": metric,
        "baseline": round(baseline, 4),
        "margin": margin,
    }

    if direction == MINIMIZE:
        # Mirror of the chance rule; only lowers
        if chance is not None and threshold > chance * (1.0 - MIN_LIFT):
            threshold = chance * (1.0 - MIN_LIFT)
            reference["lowered_to_clear_chance"] = True
        if threshold <= 0.0:
            # Bar asks for exact predictions; reported, not changed.
            reference["demands_zero_error"] = True
    elif metric in BOUNDED_METRICS:
        # Clamp first, then chance, never reversed
        threshold = min(threshold, CEILING)
        if chance is not None and threshold < chance + MIN_LIFT:
            # Else the majority-class predictor already passes.
            threshold = chance + MIN_LIFT  # chance wins; not pulled back to CEILING
            reference["raised_to_clear_chance"] = True
        threshold = _raise_to_ranking_floor(reference, card or {}, metric, threshold, baseline)
        if threshold > CEILING:
            # Not lowered; ``describe`` says it cannot be reached.
            reference["exceeds_ceiling"] = True
        # Round before notes so required_ks matches the screen.
        threshold = round(threshold, 4)
        _note_ranking_ceiling(reference, card or {}, metric, threshold, baseline)
    _note_baseline_interval(reference, card or {}, metric, round(threshold, 4))
    if chance is not None:
        reference["chance"] = round(chance, 4)
    return round(threshold, 4), reference


def _raise_to_ranking_floor(
    reference: dict[str, Any],
    card: dict[str, Any],
    metric: str,
    threshold: float,
    baseline: float,
) -> float:
    """_raise_to_ranking_floor | Role: raise a bar below the baseline's best cut to it.

    Capped at :data:`CEILING`; only makes goals harder.
    """
    _ks, ceiling = card_ceiling(card, metric)
    if ceiling is None or threshold >= ceiling:
        return threshold
    reference["raised_to_ranking_floor"] = True
    reference["ranking_floor"] = ceiling
    # Reader cannot recompute this one, so keep it.
    reference["margin_bar"] = round(threshold, 4)
    floored = passable_margin(baseline, ceiling)
    if floored is not None:
        reference["floored_margin"] = floored
    return min(ceiling, CEILING)


def _note_ranking_ceiling(
    reference: dict[str, Any],
    card: dict[str, Any],
    metric: str,
    threshold: float,
    baseline: float,
) -> None:
    """_note_ranking_ceiling | Role: note a bar no baseline cut reaches; bar unchanged."""
    ks, ceiling = card_ceiling(card, metric)
    if ks is None or ceiling is None:
        return
    reference["ks"] = round(ks, 4)
    reference["ranking_ceiling"] = ceiling
    if threshold <= ceiling:
        return
    reference["exceeds_ranking_ceiling"] = True
    demanded = required_ks(threshold)
    if demanded is not None:
        reference["required_ks"] = demanded
        reference["ks_shortfall"] = round(demanded - round(ks, 4), 4)
    margin = passable_margin(baseline, ceiling)
    if margin is not None:
        reference["passable_margin"] = margin


def _note_baseline_interval(
    reference: dict[str, Any], card: dict[str, Any], metric: str, threshold: float
) -> None:
    """_note_baseline_interval | Role: note a bar inside the baseline's CI; bar unchanged."""
    block = card.get("baseline")
    interval = (block or {}).get("ci") if isinstance(block, dict) else None
    scores = (interval or {}).get("scores") if isinstance(interval, dict) else None
    bound = scores.get(metric) if isinstance(scores, dict) else None
    if not isinstance(bound, dict):
        return
    low, high = as_number(bound.get("low")), as_number(bound.get("high"))
    if low is None or high is None:
        return
    reference["baseline_ci"] = [round(low, 4), round(high, 4)]
    # Group-level intervals are wider; the unit matters.
    if isinstance(interval, dict) and interval.get("unit"):
        reference["baseline_ci_unit"] = str(interval["unit"])
    if contains((low, high), threshold):
        reference["inside_baseline_ci"] = True


# --- Role: goal building ----------------------------------------------------------


def derive_goal(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    mode: str = DEFAULT_MODE,
    threshold: float | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """derive_goal | Role: build the ``goal`` channel; a ``threshold`` means ``fixed``."""
    if threshold is not None:
        mode = MODE_FIXED
    direction = direction or direction_of(metric)
    goal: dict[str, Any] = {"metric": metric, "direction": direction, "mode": mode}

    if mode == MODE_FIXED:
        goal["threshold"] = float(threshold) if threshold is not None else default_bar(metric)
        if threshold is not None:
            goal["source"] = "fixed"
        elif goal["threshold"] is not None:
            goal["source"] = "fixed_default"
        else:
            # Profiling refuses this, offering the measured option.
            goal["source"] = "unset"
        return goal

    value, reference = derive_threshold(card, metric=metric, direction=direction, margin=margin)
    goal["threshold"] = value
    goal["source"] = {"baseline": "derived", "unset": "unset"}.get(
        str(reference.get("source")), "fallback"
    )
    goal["reference"] = reference
    return goal


def resolve_goal(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    mode: str = DEFAULT_MODE,
    threshold: float | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[dict[str, Any], str | None]:
    """resolve_goal | Role: ``(goal, note)`` like derive_goal, swapping an unfit metric.

    ``note`` is a Korean line, or ``None`` when nothing was swapped.
    """
    task = card_task(card or {})
    swap = substitute_metric(task, metric) if task else None
    if swap is None:
        return derive_goal(
            card, metric=metric, direction=direction, mode=mode, threshold=threshold, margin=margin
        ), None

    note = (
        f"이 데이터의 task는 {(card or {}).get('task')}인데 목표 지표 {metric}는 다른 task의 "
        f"지표입니다 — 이 정답 열에서는 계산되지 않으므로, 그대로 두면 모든 시도가 '목표 미달'로 "
        f"기록되고 반복 예산만 소모됩니다. {swap}로 바꿔 실행합니다"
    )
    if threshold is not None:
        # Old metric's units; say so, never drop quietly.
        note += (
            f". --threshold {threshold}는 {metric}의 단위로 준 값이라 {swap}의 바로 쓸 수 없어 "
            f"버립니다 — {swap} 기준으로 다시 지정하려면 --metric {swap} --threshold <값>"
        )
    goal = derive_goal(card, metric=swap, direction=None, mode=mode, threshold=None, margin=margin)
    goal["substituted_from"] = metric
    return goal, note


def goal_threshold(goal: dict[str, Any], default: float) -> float:
    """goal_threshold | Role: the bar as float; ``default`` when missing or ``None``."""
    raw = goal.get("threshold")
    return default if raw is None else float(raw)


# --- Role: descriptions -----------------------------------------------------------


def missing_bar_message(metric: str) -> str:
    """missing_bar_message | Role: Korean refusal naming the two ways to give a bar."""
    return (
        f"{metric}는 정답 열의 단위로 나오는 지표라서 이식 가능한 기본 목표값이 없습니다 — "
        f"0.85 같은 숫자를 넣으면 모델이 아니라 그 열의 단위에 대한 바가 됩니다.\n"
        f"  - 요구 수준을 알고 있다면: --threshold <{metric} 값>\n"
        "  - 데이터에서 도출하려면: 기준선이 측정된 카드로 auto 모드를 쓰십시오 "
        "(`profile`을 --no-baseline 없이 실행)"
    )


def describe(goal: dict[str, Any]) -> str:
    """describe | Role: one Korean line: mode, bar, and where it came from."""
    metric = str(goal.get("metric", "metric"))
    threshold = goal.get("threshold")
    direction = "이하" if str(goal.get("direction")) == "minimize" else "이상"
    mode = str(goal.get("mode") or DEFAULT_MODE)
    head = f"{mode} 모드 — {metric} {threshold} {direction}"
    if goal.get("substituted_from"):
        # Printed with the bar, so every view shows it.
        head += f" (요청한 지표 {goal['substituted_from']}는 이 task의 지표가 아니라 대체됨)"
    source = str(goal.get("source") or "")
    reference = goal.get("reference") or {}

    if source == "unset" or threshold is None:
        # Say what is missing, not "None".
        return f"{mode} 모드 — {metric} 목표값 미정 (이 지표는 기본 바가 없습니다)"
    if source == "fixed":
        return f"{head} (직접 지정)"
    if source == "fixed_default":
        return f"{head} (지표 기본값)"
    if source == "derived" and isinstance(reference, dict):
        baseline = reference.get("baseline")
        margin = reference.get("margin", DEFAULT_MARGIN)
        if str(goal.get("direction")) == MINIMIZE:
            # Margin comes off the error, not remaining room.
            detail = f"기준선 {baseline}에서 {float(margin):.0%} 감소"
        else:
            detail = f"기준선 {baseline} + 남은 여유의 {float(margin):.0%}"
        if reference.get("raised_to_clear_chance"):
            detail += ", chance 수준을 넘도록 상향"
        if reference.get("raised_to_ranking_floor"):
            detail += ", 기준선 랭킹의 최적 컷을 넘도록 상향"
        if reference.get("lowered_to_clear_chance"):
            detail += ", chance보다 낮도록 하향"
        chance = reference.get("chance")
        if chance is not None:
            detail += f" (chance {chance})"
        line = f"{head} ← {detail}"
        if reference.get("exceeds_ceiling"):
            line += (
                f" (도달 불가) — chance가 {chance}이라 목표가 상한 {CEILING}을 넘습니다. "
                "--metric balanced_accuracy 또는 --metric pr_auc처럼 다수 클래스에 "
                "덜 휘둘리는 지표로 바꾸십시오"
            )
        if reference.get("demands_zero_error"):
            # Minimize partner: here the fix is the margin.
            line += (
                f" (도달 불가) — 오차 0을 요구하는 바입니다. --margin을 1보다 작게 두거나, "
                f"실제 허용 오차를 알고 있다면 --threshold <{metric} 값>으로 "
                "직접 지정하십시오"
            )
        if reference.get("raised_to_ranking_floor"):
            # Explain why ``--margin`` had no effect.
            line += (
                f" — margin이 낸 바는 {reference.get('margin_bar')}였지만, 같은 기준선의"
                f" 랭킹을 최적 컷에서 자르면 {reference.get('ranking_floor')}"
                f" (KS {reference.get('ks')})입니다: 기준선을 다시 자른 것이 이미 넘는 점수는"
                " 목표가 아니므로 그 값으로 올렸습니다"
            )
            floored = reference.get("floored_margin")
            if floored is not None:
                line += (
                    f". 이 카드에서 --margin {floored} 이하는 모두 같은 바를 냅니다 —"
                    " 더 어려운 목표를 원하면 그보다 크게 주십시오"
                )
        if reference.get("exceeds_ranking_ceiling"):
            # Limit of this ranking, not a verdict
            passable = reference.get("passable_margin")
            line += (
                f" — 이 바는 기준선 랭킹의 상한 {reference.get('ranking_ceiling')}를 넘습니다"
                f" (KS {reference.get('ks')}). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니"
                " 랭킹 자체를 올려야 합니다 — 모델 family나 특성"
            )
            # Gap size on the KS axis, comparable to levers.
            demanded = reference.get("required_ks")
            if demanded is not None:
                line += (
                    f". 크기로 말하면 이 바가 요구하는 KS가 {demanded}이고 기준선은"
                    f" {reference.get('ks')}이므로 랭킹이 {reference.get('ks_shortfall')}만큼"
                    " 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오"
                )
            if passable is not None:
                line += f". 지금 기준선에서 이 상한 안에 드는 margin은 {passable} 이하입니다"
        if reference.get("inside_baseline_ci"):
            # Too close to the start for the slice.
            interval = reference.get("baseline_ci") or []
            unit = "그룹" if reference.get("baseline_ci_unit") == "group" else "행"
            line += (
                f" — 이 바는 기준선 자체의 {int(CI_LEVEL * 100)}% CI"
                f"({interval[0]}~{interval[1]}, {unit} 단위) 안에 있습니다: "
                "val 슬라이스가 이만한 차이를 분해하지 못한다는 뜻이므로, 바를 넘겼는지보다 "
                "홀드아웃 점수를 보십시오. 더 큰 --margin이 더 정직한 목표입니다"
            )
        return line
    return f"{head} (카드에 기준선이 없어 지표 기본값 사용)"
