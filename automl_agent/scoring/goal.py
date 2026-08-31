"""The goal threshold, in two modes: ``auto`` (dataset-relative) and ``fixed``.

**Why two.** A fixed ``f1 >= 0.85`` says nothing portable. On an 11%-positive clinical
cohort it is unreachable no matter how good the model is; on a well-separated synthetic
card it is free. Either way the loop's stopping condition stops meaning "this model is
good" and starts meaning "this dataset is easy". But a fixed bar is exactly right when
the number comes from outside the data — a deployment requirement, a paper to beat, a
regulatory floor. Those are different jobs, so they are different modes rather than one
heuristic trying to serve both.

``auto`` — the default. The bar is set *relative to a reference baseline* that
:mod:`automl_agent.scripts.profile` measures on the same data, with the same holdout
split, using one deliberately unglamorous model:

    target = baseline + (1 - baseline) * margin        (bounded, maximize)
    target = baseline * (1 - margin)                   (minimize)

Which makes the goal mean the same thing across datasets: *beat a competent default by a
meaningful margin*. The margin is on the remaining headroom, not on the score, because
the last 0.05 of roc_auc is far harder to win than the first.

For an error metric the headroom is the error itself — zero is the perfect score — so the
margin comes off the baseline proportionally. That is the only scale-free reading available:
``mae`` is in the target's units, and "close a quarter of the remaining distance" would
otherwise need a number nobody can supply. Both formulas are then held to the same floor:
whatever the margin produces, the bar has to beat the trivial predictor
(:data:`MIN_LIFT`), because a goal a constant predictor already meets is not a goal.

``fixed`` — the bar is the number the caller gave, or the per-metric default if they
gave none. Nothing about the data can move it.

Both modes are deterministic, and in neither is the LLM asked what the target should be:
letting the model set its own bar would let it declare success by lowering it.

**Reachability is disclosed, never enforced.** A derived bar can sit above anything the
baseline's *ranking* permits at any decision threshold, and two real runs spent their
whole iteration budget finding that out. ``auto`` now compares the bar against that
ceiling and records the comparison in ``reference`` for :func:`describe` to print, along
with the margin that would have fitted. It does not lower the bar — a bar that moves to
meet the result is not a goal — and it does not refuse to run, because the ceiling
belongs to the *baseline's* ranking and a better family beats it: ``m-llm7``'s bar of
0.7842 is over the 0.7584 its card allowed and that run cleared it at 0.7847. So the
warning errs low by construction, and what it says — "the ranking itself has to improve" —
is what that run did. See :mod:`automl_agent.scoring.ranking`.

**One reachability check does move the bar, upward.** The ceiling comparison above was
one-directional — it caught a bar above what the baseline's ranking permits at any cut, and
said nothing about a bar that ranking already clears at *some* cut. The second case is a
real defect in the same number: ``baseline`` is measured at the 0.5 cut, so a metric the
0.5 cut treats badly hands the derivation a deflated starting point and the bar inherits the
deficit. Measured on the bench cards, ``bank-marketing``'s ``balanced_accuracy`` bar of
0.7465 sat under the 0.8400 that re-cutting the *same logreg* reaches — and both the
rule-based fallback arm (val 0.7833) and the LLM arm (val 0.8780) cleared it, so the bar
separated nothing. Validation scores because that is the channel ``route()`` compares the
bar against; ``bench/runs/artifacts/{nollm,llm}-bank-marketing-seed42/history.json``.
``auto`` now floors the bar at that number, which is the exception
:data:`MIN_LIFT` already makes read one step further: a goal the baseline's own ranking
already meets at some cut is not a goal either. Only ``balanced_accuracy`` has the identity
this needs, and measurement says it is also the only metric whose cut deficit outruns the
margin — see :func:`_raise_to_ranking_floor` for the table.

**Resolution is disclosed too, on the same terms.** A bar can also be bad by being too
*close*: derived from a baseline whose own 95% interval is wider than the margin, so the
validation slice cannot resolve the difference the goal asks for. ``auto`` records that
comparison as ``inside_baseline_ci`` and :func:`describe` prints it, without moving the bar.
See :mod:`automl_agent.scoring.intervals` for what that width does and does not license.
"""

from __future__ import annotations

from typing import Any

from .intervals import CI_LEVEL, contains
from .metrics import ALIASES, METRICS, MINIMIZE, card_task, direction_of, spec, substitute_metric
from .ranking import card_ceiling, passable_margin, required_ks

# The two modes. ``auto`` measures the dataset; ``fixed`` takes the caller's number.
MODE_AUTO = "auto"
MODE_FIXED = "fixed"
GOAL_MODES = (MODE_AUTO, MODE_FIXED)
DEFAULT_MODE = MODE_AUTO

# Fraction of the remaining headroom the agent is asked to close. ``auto`` only.
DEFAULT_MARGIN = 0.25

# Metrics living on [0, 1], where "remaining headroom" is meaningful. Derived from the
# registry rather than restated: a metric this repo cannot compute must not get a bar.
# Aliases are included so a card written with sklearn's name derives the same threshold.
BOUNDED_METRICS = frozenset(
    {name for name, spec in METRICS.items() if spec.bounded}
    | {alias for alias, target in ALIASES.items() if METRICS[target].bounded}
)

# Never ask for a perfect score: on real data it means the target is unreachable and
# the run always ends on the iteration budget, which hides genuine stalls.
CEILING = 0.99
# A target must sit at least this far above chance level, or a majority-class
# predictor would satisfy it.
#
# Read two ways, because chance is measured in the metric's own terms. On a bounded
# maximizing metric it is an absolute distance (``chance + 0.02``). On an error metric it
# is a *fraction* of the chance error (``chance * 0.98``), since an absolute 0.02 would mean
# something different for a target in days than for one in dollars — the one thing a bar in
# the target's units must never do. Same constant, same intent, stated in whichever unit the
# metric brought with it.
MIN_LIFT = 0.02

# ``fixed`` mode's default bar, and ``auto``'s last resort when the card carries no
# baseline at all (a hand-written card, or a profiling run with ``--no-baseline``). Same
# derivation as above, so the two constants can never cover different metrics.
#
# A metric in the target's own units has no entry, because it has no portable default: the
# registry declares ``fallback=None`` for ``mae``/``rmse`` and this dict simply skips it.
# Missing here means "this metric needs a measured baseline or an explicit --threshold",
# which is a different state from "unknown metric" — see :func:`default_bar`.
_BARS = {name: item.fallback for name, item in METRICS.items() if item.fallback is not None}
FALLBACK_THRESHOLDS: dict[str, float] = {
    **_BARS,
    **{alias: _BARS[target] for alias, target in ALIASES.items() if target in _BARS},
}
FALLBACK_DEFAULT = 0.85


def default_bar(metric: str) -> float | None:
    """The bar to use for ``metric`` when nothing was measured, or ``None`` if there is none.

    Three cases, and the difference between the last two matters:

    * a registry metric with a portable default → that number;
    * a registry metric in the target's units (``mae``, ``rmse``) → ``None``, because any
      number here would be a claim about the column's scale rather than about the model;
    * a name this harness does not know → :data:`FALLBACK_DEFAULT`, keeping the derivation
      path tolerant of hand-written cards. ``RunConfig`` is where unknown names are refused.
    """
    found = spec(metric)
    if found is None:
        return FALLBACK_DEFAULT
    return found.fallback


def _reference(card: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    """Pull ``(baseline, chance)`` for ``metric`` out of the card's baseline block."""
    baseline_block = card.get("baseline")
    if not isinstance(baseline_block, dict):
        return None, None

    def score(section: str) -> float | None:
        scores = baseline_block.get(section)
        raw = scores.get(metric) if isinstance(scores, dict) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        return float(raw)

    return score("scores"), score("chance")


def _target(baseline: float, metric: str, direction: str, margin: float) -> float:
    if direction == MINIMIZE:
        # Error metrics are bounded below by zero, so the margin comes off the score.
        return max(0.0, baseline * (1.0 - margin))
    if metric in BOUNDED_METRICS:
        return baseline + (1.0 - baseline) * margin
    # Unbounded and maximizing (a throughput-like metric): proportional lift is the
    # only thing left that is scale-free.
    return baseline * (1.0 + margin)


def derive_threshold(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[float | None, dict[str, Any]]:
    """Return ``(threshold, reference)`` for this card and metric.

    ``reference`` records how the number was reached, so the report and the Critic
    prompt can say "0.865 = logreg's 0.821 plus a quarter of the headroom" instead of
    presenting a bare constant.

    ``direction`` is read from the metric registry when not given, and no caller has a
    reason to give it: which way is better belongs to the metric.

    The threshold is ``None`` in exactly one case — a metric in the target's own units
    (``mae``, ``rmse``) with no measured baseline to derive from. There is no honest number
    there, so the caller gets nothing rather than a guess, and
    :mod:`automl_agent.nodes.profiling` refuses the run.
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
        # The mirror of the chance rule below, and every bit as necessary: a Ridge baseline
        # can score *worse* than predicting the mean (a negative r2 comes with an mae above
        # chance), and ``baseline * (1 - margin)`` then produces a bar the mean predictor
        # already clears. Tightening it — the bar moves down, never up — keeps the one
        # invariant both directions share: an adjustment may only make the goal harder.
        if chance is not None and threshold > chance * (1.0 - MIN_LIFT):
            threshold = chance * (1.0 - MIN_LIFT)
            reference["lowered_to_clear_chance"] = True
        if threshold <= 0.0:
            # ``--margin 1.0``, or a baseline that already scores a perfect zero. Disclosed
            # rather than nudged, on the same terms as ``exceeds_ceiling``: the bar asks for
            # exact prediction, which no attempt will report, and the run would end on the
            # iteration budget looking like a stall.
            reference["demands_zero_error"] = True
    elif metric in BOUNDED_METRICS:
        # Clamp first, then chance — never the other way round. Clamping last let the
        # ceiling pull the bar back *below* chance (a 99.5%-majority cohort got
        # ``accuracy >= 0.99``, which a constant predictor already beats), and the run
        # then reported success without learning anything.
        threshold = min(threshold, CEILING)
        if chance is not None and threshold < chance + MIN_LIFT:
            # A baseline at or below chance would otherwise produce a target a
            # majority-class predictor already clears.
            threshold = chance + MIN_LIFT  # chance 우선 — CEILING으로 되돌리지 않는다
            reference["raised_to_clear_chance"] = True
        threshold = _raise_to_ranking_floor(reference, card or {}, metric, threshold, baseline)
        if threshold > CEILING:
            # Kept honest rather than quietly lowered: this metric cannot separate a real
            # model from the majority class on this dataset, and ``describe`` says so.
            reference["exceeds_ceiling"] = True
        # Rounded before the disclosure, not after: the bar the run is judged by is the
        # rounded one, and ``required_ks`` inverts an identity. A KS derived from the
        # unrounded intermediate does not invert back to the bar on screen, which makes it a
        # number the reader cannot check — 0.6077 against a printed 0.8038.
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
    """Raise the bar to the baseline ranking's own best cut, when the bar sits under it.

    The mirror of :func:`_note_ranking_ceiling`, which was one-directional: it disclosed a
    bar the baseline's ranking cannot reach at *any* cut and said nothing about a bar that
    ranking already reaches at *some* cut. Both are the same defect in the same number,
    because ``baseline`` is measured at the 0.5 cut — ``profile.py`` calls ``predict()``, like
    every attempt does — so whatever the cut costs the baseline is subtracted from the bar
    derived from it.

    Measured on the four bench cards, that cost is ``balanced_accuracy``'s alone. The
    threshold-dependent metrics all lose something at 0.5, but only here does the loss
    outrun what the margin adds:

    ======================  ==============  ============  ===========================
    bank-marketing          0.5 cut         best cut       margin 0.25 adds
    ======================  ==============  ============  ===========================
    ``f1``                  0.4552          0.5834         0.1362 — covers the 0.1282
    ``accuracy``            0.9026          0.9051         0.0244 — covers the 0.0025
    ``balanced_accuracy``   0.6620          0.8400         0.0845 — under the 0.1780
    ======================  ==============  ============  ===========================

    The 0.5-cut column and the last one are recomputable from
    ``bench/cards/bank-marketing-seed42.json`` — ``baseline.scores`` and
    ``(1 - score) * 0.25``. The ``balanced_accuracy`` best cut is in the same card
    (``baseline.balanced_accuracy_at_best_cut``, the ``(1 + ks) / 2`` this module reads). The
    other two best cuts are **not**, and this module cannot derive them: ``card_ceiling``
    returns ``None`` off ``SYMMETRIC_METRICS`` precisely because no identity exists there.
    They were measured once off the baseline's own validation probabilities, and a reader who
    wants to check them has to redo that rather than read a card.

    Not a coincidence of this dataset. ``balanced_accuracy`` weights the two error rates
    equally, which is both why the KS identity holds for it (see
    :mod:`automl_agent.scoring.ranking`) and why the 0.5 cut costs most under imbalance — the
    recall collapses to 0.3478 and half of that lands in the score. ``accuracy`` is carried
    by the majority class, ``f1`` never counts the negatives. So the metric this fires for is
    the metric that needs it, and ``card_ceiling`` returning ``None`` for the rest is the
    right answer rather than a gap. ``precision`` and ``recall`` are excluded for a further
    reason: their best cut is 1.0000 by predicting nothing or everything, so a floor there
    would be degenerate rather than demanding.

    **This moves the bar, which the rest of this module does not do.** The exception is the
    one :data:`MIN_LIFT` already makes, in the same direction and for the same reason: a goal
    a constant predictor already meets is not a goal, and neither is one that re-cutting the
    baseline's own ranking already meets. Both only ever make the bar harder. What it costs
    is that ``--margin`` stops moving the bar below the floor — 0.25 through 0.526 all
    produce 0.8400 on bank-marketing — so the floored margin is disclosed and
    :func:`describe` prints it, because a bar that ignores the flag it was given has to say so.

    Clamped at :data:`CEILING` rather than allowed past it: a ranking with KS above 0.98
    would otherwise push the bar over the clamp and collect the ``exceeds_ceiling``
    disclosure, whose wording blames ``chance`` and would be false here.
    """
    _ks, ceiling = card_ceiling(card, metric)
    if ceiling is None or threshold >= ceiling:
        return threshold
    reference["raised_to_ranking_floor"] = True
    reference["ranking_floor"] = ceiling
    # The bar this card's margin actually produced, kept because it is the only number in the
    # disclosure the reader cannot recompute from the others — and because a run whose bar was
    # moved should be able to say what it was moved from.
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
    """Record whether the bar sits above what any cut of the baseline ranking can reach.

    A *disclosure*, never an adjustment. The bar is not lowered to meet it, because a bar
    that moves to meet the result stops being a goal — and because the ceiling belongs to
    the baseline's ranking, which a better model family is free to beat
    (:mod:`automl_agent.scoring.ranking` has the numbers). Advisory both ways: a card with no KS
    simply gets no note, so the check can never block a run it misjudged.

    When it does fire, the bar is recorded *twice*: once on the metric the caller asked for
    and once on the ranking axis that owns the shortfall (``required_ks`` against the
    baseline's ``ks``). One bar, two requirements, each with a size — which is the whole
    difference between "the ranking has to improve" and a number a family swap can be
    compared against. See :func:`automl_agent.scoring.ranking.required_ks` for what that comparison
    cost when it was missing.
    """
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
    """Record whether the bar sits inside the baseline score's own confidence interval.

    A *disclosure*, like the ranking ceiling above, and for the same reason: the margin is
    the caller's to choose and a bar that moves to meet the measurement is not a goal.

    What it means when it fires: the distance from the baseline to the bar is smaller than
    the spread of the baseline measurement itself, so this validation slice cannot resolve
    a difference that size. What it does *not* mean is that an attempt clearing the bar
    learned nothing — attempts are scored on the same rows as the baseline, so the two
    errors move together and a paired comparison is tighter than either interval. The
    honest reading is "this margin is at the noise floor of this slice", and the number that
    settles it is the held-back test score (:mod:`automl_agent.nodes.holdout`).

    Advisory both ways: a card written before intervals existed, or one whose validation
    slice was too small for one, simply gets no note.
    """
    block = card.get("baseline")
    interval = (block or {}).get("ci") if isinstance(block, dict) else None
    scores = (interval or {}).get("scores") if isinstance(interval, dict) else None
    bound = scores.get(metric) if isinstance(scores, dict) else None
    if not isinstance(bound, dict):
        return
    low, high = bound.get("low"), bound.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return
    reference["baseline_ci"] = [round(float(low), 4), round(float(high), 4)]
    # ``unit`` matters to the reading: a grouped interval is the wide, honest one, and its
    # width is the reason a bar can land inside it at all on clustered data.
    if isinstance(interval, dict) and interval.get("unit"):
        reference["baseline_ci_unit"] = str(interval["unit"])
    if contains((float(low), float(high)), threshold):
        reference["inside_baseline_ci"] = True


def derive_goal(
    card: dict[str, Any],
    *,
    metric: str,
    direction: str | None = None,
    mode: str = DEFAULT_MODE,
    threshold: float | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Build the ``goal`` channel for one of the two modes.

    Naming a ``threshold`` *is* choosing ``fixed``, so it implies the mode rather than
    conflicting with it — a caller who hands over a number never has it silently
    ignored. The CLI refuses ``--goal-mode auto --threshold`` outright, so that
    inference only ever applies where the intent is unambiguous.

    In ``auto`` this is called twice on the ``--data`` path: once in ``initial_state``
    (no card yet, so the per-metric default applies) and again by the ``profiling`` node
    once the card exists. The second call overwrites the first, which is why the channel
    is a plain value and not an accumulating one.

    ``direction`` comes from the metric registry unless a caller overrides it, so
    ``threshold`` always means "the bar", never "the bar, in whichever direction someone
    asked for". ``threshold`` can come back ``None`` — see :func:`derive_threshold`.
    """
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
            # ``fixed`` with no number, for a metric whose units make every number
            # arbitrary. Refused by the profiling node rather than here, so the message
            # can name the measured alternative.
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
    """:func:`derive_goal`, with the metric swapped when it cannot score this card's target.

    Returns ``(goal, note)``. ``note`` is the Korean line explaining the swap, and it is
    ``None`` on the ordinary path — so a caller prints it when there is one and says nothing
    when there is not. Every construction of the ``goal`` channel goes through here, because
    the swap has to happen in exactly one place: ``goal["metric"]`` is what every node from
    ``training`` to ``holdout`` reads, and a substitution applied anywhere downstream would
    leave two different metrics in one run.

    Three things move together, and the two that are easy to forget are the reason this is a
    function rather than two lines at each call site:

    * ``direction`` is re-read from the new metric. Carrying the old one over is not a smaller
      bug than the mismatch it fixes — ``rmse``'s ``minimize`` applied to ``f1`` inverts the
      whole run, keeping the *worst* attempt as ``best`` and firing ``goal_met`` on any score
      below the bar (see ``RunConfig.__post_init__`` for the same trap by hand).
    * an explicit ``threshold`` is dropped. It was a number in the old metric's units, and
      ``--threshold 3.2`` for ``rmse`` reused as an ``f1`` bar is not a translation of the
      caller's intent, it is an unreachable goal wearing their number. The mode is kept, so a
      ``fixed`` run gets the new metric's default bar and ``source`` says ``fixed_default``.

    ``substituted_from`` is recorded on the goal so the swap survives into the checkpoint and
    the report; :func:`describe` prints it wherever the goal is printed.
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
        # Said, not silently absorbed: the caller gave a number and this run will not be
        # judged by it. The number is in the old metric's units, so there is nothing to carry.
        note += (
            f". --threshold {threshold}는 {metric}의 단위로 준 값이라 {swap}의 바로 쓸 수 없어 "
            f"버립니다 — {swap} 기준으로 다시 지정하려면 --metric {swap} --threshold <값>"
        )
    goal = derive_goal(card, metric=swap, direction=None, mode=mode, threshold=None, margin=margin)
    goal["substituted_from"] = metric
    return goal, note


def goal_threshold(goal: dict[str, Any], default: float) -> float:
    """The goal's bar as a float, falling back to ``default`` when it has none.

    ``goal.get("threshold", default)`` does not do this: the key is *present* and ``None``
    for a metric whose bar could not be derived, so the default never applies and the
    caller gets a ``TypeError`` from ``float(None)``. Every node inside the loop runs after
    ``profiling`` has refused that case, so this is a guard rather than a code path.
    """
    raw = goal.get("threshold")
    return default if raw is None else float(raw)


def missing_bar_message(metric: str) -> str:
    """Why this run cannot start, and the two ways to give it a bar.

    Shared by every caller that refuses an unset threshold, so the operator reads the same
    two options whichever path found it (``--dataset-card`` at startup, ``--data`` after
    profiling).
    """
    return (
        f"{metric}는 정답 열의 단위로 나오는 지표라서 이식 가능한 기본 목표값이 없습니다 — "
        f"0.85 같은 숫자를 넣으면 모델이 아니라 그 열의 단위에 대한 바가 됩니다.\n"
        f"  - 요구 수준을 알고 있다면: --goal-mode fixed --threshold <{metric} 값>\n"
        "  - 데이터에서 도출하려면: 기준선이 측정된 카드로 auto 모드를 쓰십시오 "
        "(`profile`을 --no-baseline 없이 실행)"
    )


def describe(goal: dict[str, Any]) -> str:
    """One-line Korean explanation: which mode, what bar, and where it came from."""
    metric = str(goal.get("metric", "metric"))
    threshold = goal.get("threshold")
    direction = "이하" if str(goal.get("direction")) == "minimize" else "이상"
    mode = str(goal.get("mode") or DEFAULT_MODE)
    head = f"{mode} 모드 — {metric} {threshold} {direction}"
    if goal.get("substituted_from"):
        # Printed with the bar rather than only once at startup: this run is judged by a
        # metric nobody asked for, and every place that shows the goal — the console header,
        # the Critic's prompt, report.md — has to carry that with it. A swap that appears in
        # one line of console output and nowhere else is a swap the reader of the report
        # cannot see.
        head += f" (요청한 지표 {goal['substituted_from']}는 이 task의 지표가 아니라 대체됨)"
    source = str(goal.get("source") or "")
    reference = goal.get("reference") or {}

    if source == "unset" or threshold is None:
        # No number to print, so the line says what is missing instead of printing "None".
        # The run does not get this far unless something skipped the profiling node's check.
        return f"{mode} 모드 — {metric} 목표값 미정 (이 지표는 기본 바가 없습니다)"
    if source == "fixed":
        return f"{head} (직접 지정)"
    if source == "fixed_default":
        return f"{head} (지표 기본값)"
    if source == "derived" and isinstance(reference, dict):
        baseline = reference.get("baseline")
        margin = reference.get("margin", DEFAULT_MARGIN)
        if str(goal.get("direction")) == MINIMIZE:
            # "남은 여유" would be a lie here: the margin comes off the error itself, and the
            # reader needs to see which arithmetic produced the number in front of them.
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
            # The minimize counterpart of the line above, and it needs its own wording: the
            # metric is not the problem here (mae is the right thing to measure), the margin is.
            line += (
                f" (도달 불가) — 오차 0을 요구하는 바입니다. --margin을 1보다 작게 두거나, "
                f"실제 허용 오차를 알고 있다면 --goal-mode fixed --threshold <{metric} 값>으로 "
                "직접 지정하십시오"
            )
        if reference.get("raised_to_ranking_floor"):
            # The counterpart of the ``exceeds_ranking_ceiling`` line below, and it has to
            # carry two numbers the reader cannot recompute: what the margin would have
            # produced, and why the flag they passed stopped mattering. Silently ignoring
            # ``--margin`` is the failure mode this wording exists to prevent.
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
            # A different kind of out of reach from the one above: not the metric's own
            # limit but this ranking's. Worth naming as such — the fix is a better
            # ranking, and the previous wording would have sent the caller to change the
            # metric, which is the one thing that does not help here.
            # "이 랭킹으로는" is load-bearing: the ceiling is the baseline's, and a better
            # family beats it — m-llm7 cleared a bar that was over its card's ceiling. An
            # unqualified "도달할 수 없다" would read as a verdict on the run.
            passable = reference.get("passable_margin")
            line += (
                f" — 이 바는 기준선 랭킹의 상한 {reference.get('ranking_ceiling')}를 넘습니다"
                f" (KS {reference.get('ks')}). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니"
                " 랭킹 자체를 올려야 합니다 — 모델 family나 특성"
            )
            # The size, on the axis the lever is on. Without it "랭킹을 올려야 한다" is a
            # direction with no scale, and a family swap reads as a plausible answer to it —
            # which is exactly what mv-llm-4·5·6 each spent an iteration finding out.
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
            # A third way a bar can be a bad bar, independent of the two above: not out of
            # reach but indistinguishable from where it started. Wording stays modest —
            # attempts are scored on the same rows as the baseline, so this is a statement
            # about the slice's resolution, not a verdict on an attempt that clears it.
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
