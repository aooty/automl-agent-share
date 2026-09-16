"""Profiling node: turns the private data reference into a public dataset card.

Execution node — no LLM, and the graph's entry point. It reads the one channel the
reasoning nodes never touch (``data_ref``) and writes the one they all read
(``dataset_card``), so the card is the *only* thing that crosses from data to
reasoning. That crossing is a node boundary, not a convention.

Like training, the work happens in a subprocess (``scripts/profile.py``): pandas is
never imported into the process that renders prompts, so a data row cannot end up in
one by accident.

Unlike training, a failure here is *not* handed to the Critic. A missing card is a
setup error, not an experimental outcome — there is nothing to diagnose and no plan
worth making without it — so this node raises and the run stops.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import PROFILE_SCRIPT, PROFILE_TIMEOUT_SEC, RunConfig, decode_output, utf8_env
from ..privacy import CardSchemaError, public_card, register_private, validate_card
from ..scoring.goal import describe, missing_bar_message, resolve_goal
from ..scoring.metrics import TASK_REGRESSION, card_task, metrics_for, task_of
from ..scoring.splits import protocol_mismatch
from ..state import AutoMLState


class ProfilingFailed(RuntimeError):
    """The card could not be built, so there is nothing to plan against."""


def profiling(state: AutoMLState, *, config: RunConfig) -> dict:
    """Return ``{"dataset_card": ..., "goal": ...}``, or nothing when a card is present."""
    reference = dict(state.get("data_ref") or {})
    if state.get("dataset_card"):
        # A hand-written card (or a resumed run) wins: profiling never overwrites it.
        card = dict(state["dataset_card"])
        assert_protocol_matches(card, config, reference)
        # The goal on this path was derived in ``initial_state``, from this same card —
        # there is nothing left to measure, so it is checked rather than recomputed.
        assert_goal_is_usable(card, dict(state.get("goal") or {}), config)
        return {}

    path = reference.get("path")
    if not path:
        # No real data and no card: the executor will synthesise from the card's
        # declared shape, and there is nothing here to profile.
        return {}

    register_private(path)
    card = run_profiler(
        Path(str(path)),
        str(reference.get("target_column") or "target"),
        config,
    )
    # In "auto" mode the goal is derived here, from the freshly measured reference
    # baseline, because this is the first moment the baseline exists — ``initial_state``
    # could only guess. In "fixed" mode this returns the same bar it was seeded with.
    #
    # And this is also the first moment the *task* is known on the ``--data`` path, so it is
    # where a metric belonging to the other task gets swapped for one that can score this
    # target (:func:`automl_agent.scoring.goal.resolve_goal`). ``initial_state`` had no card and could
    # not have caught it.
    goal, substitution = resolve_goal(
        card,
        metric=config.metric,
        direction=config.direction,
        mode=config.goal_mode,
        threshold=config.threshold,
        margin=config.goal_margin,
    )
    if substitution:
        print(f"  [profiling] {substitution}")
    assert_goal_is_usable(card, goal, config)
    if goal != dict(state.get("goal") or {}):
        print(f"  [profiling] 목표: {describe(goal)}")
    try:
        # The card comes from our own fixed script, so a violation here means that script
        # regressed — exactly the case where stopping beats forwarding. A setup error,
        # like the rest of this node's failures, so it is not handed to the Critic.
        validate_card(card)
    except CardSchemaError as exc:
        raise ProfilingFailed(str(exc)) from exc
    # Our own script just wrote this one, so a mismatch here means the protocol changed
    # under a resumed run rather than that the operator supplied a foreign card.
    assert_protocol_matches(card, config, reference)
    return {"dataset_card": public_card(card), "goal": goal}


def assert_protocol_matches(
    card: dict[str, Any], config: RunConfig, reference: dict[str, Any] | None = None
) -> None:
    """Stop when the card's baseline was measured over different rows than this run uses.

    The goal threshold comes from that baseline, so a protocol mismatch means the bar and the scores
    it is compared against come from two different splits — a comparison that reads as a result and
    is not one. Same stance as ``--on-missing-target``: a silently incomparable number is worse than
    a stopped run. Cards with no ``protocol`` block predate the field and are accepted
    (:func:`automl_agent.scoring.splits.protocol_mismatch`).

    Group column read off ``data_ref``, not ``config``, because that is where the two sources are
    already resolved: a card profiled with ``--group-column`` names it in its private ``data`` block,
    so a ``--dataset-card`` run inherits it and is *not* refused for omitting the flag — while an
    explicit flag disagreeing with the card still is.

    Stratification read off the card's ``task`` for the same reason: a continuous target cannot be
    stratified, so ``stratified: false`` on a regression card agrees with this run rather than
    differing from it.
    """
    group_column = dict(reference or {}).get("group_column") or config.group_column
    message = protocol_mismatch(
        (card.get("baseline") or {}).get("protocol"),
        config.seed,
        str(group_column) if group_column else None,
        stratified=card_task(card) != TASK_REGRESSION,
    )
    if message:
        raise ProfilingFailed(message)


def assert_goal_is_usable(
    card: dict[str, Any], goal: dict[str, Any], config: RunConfig
) -> None:
    """Stop when the metric cannot score this target, or when the bar could not be derived.

    Both are setup errors, not experimental outcomes, and both are cheap here and expensive later.
    ``f1`` on a continuous target does not produce a bad score, it produces *no* score — the loop
    would spend its whole budget reporting "목표 미달" for a number never computed. And a ``None`` bar
    compares against nothing: ``goal_met`` is False for every attempt by construction.

    Metric read off the **goal**, not the config — ``goal["metric"]`` is what the run is judged by,
    and :func:`automl_agent.scoring.goal.resolve_goal` may already have replaced the config's metric
    with one this target has. Reading ``config.metric`` would refuse the very run substitution just
    made runnable. What is left here is the case substitution cannot reach: a goal channel written by
    something else (hand-edited checkpoint, future caller that forgets). A backstop, not the first
    line.

    No ``task`` (or a label this build does not know) predates the field and is accepted, same terms
    as a missing ``protocol`` block — this catches mismatches, it does not reject cards it cannot
    judge.

    An *empty* ``goal`` is accepted too, and is not the same as a derived bar of ``None``: it means
    no goal yet, not that deriving failed. Only ``mae``/``rmse`` produce a ``None`` bar (no portable
    default), so treating "no goal" as "no bar" would print a missing-units message about ``f1`` —
    which has a default — and refuse a run that was about to work.
    """
    metric = str(goal.get("metric") or config.metric)
    declared = card_task(card)
    wanted = task_of(metric)
    if declared and wanted and declared != wanted:
        raise ProfilingFailed(
            f"이 데이터의 task는 {card.get('task')}인데 --metric {metric}는 {wanted} "
            f"지표입니다. 이 정답 열에서는 그 지표가 계산되지 않으므로, 어떤 시도를 해도 목표에 "
            f"도달할 수 없습니다.\n  - 쓸 수 있는 지표: {', '.join(metrics_for(declared))}"
        )
    if goal and goal.get("threshold") is None:
        raise ProfilingFailed(missing_bar_message(str(goal.get("metric") or config.metric)))


def run_profiler(data_path: Path, target_column: str, config: RunConfig) -> dict[str, Any]:
    """Spawn ``scripts/profile.py`` and read back the card it wrote."""
    config.ensure_dirs()
    card_path = config.run_dir / "dataset_card.json"
    log_path = config.run_dir / "profile.log"

    command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--data",
        str(data_path),
        "--target",
        target_column,
        "--out",
        str(card_path),
        # So the baseline's holdout is the very split train.py will use.
        "--seed",
        str(config.seed),
    ]
    if config.on_missing_target:
        # Left to the script's own default when unset, so the two defaults cannot drift.
        command += ["--on-missing-target", config.on_missing_target]
    for note in config.caveats:
        # The operator's knowledge of the raw data, on its way to becoming a card field the
        # reasoning nodes read — automl_agent.dataset.caveats.
        command += ["--caveat", note]
    if config.group_column:
        # So the baseline is measured under the same group-aware split train.py will use;
        # without this the bar would be a row-level number and every attempt an out-of-group
        # one, which is the incomparability protocol_mismatch exists to catch.
        command += ["--group-column", config.group_column]

    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            command,
            capture_output=True,
            text=True,
            # Pinned, not left to the locale: the child prints Korean, and on a
            # cp949 console ``text=True`` would raise UnicodeDecodeError inside
            # subprocess' reader thread and lose the whole log.
            encoding="utf-8",
            errors="replace",
            env=utf8_env(),
            timeout=PROFILE_TIMEOUT_SEC,
            check=False,
        )
        console = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        console = decode_output(exc.stdout) + decode_output(exc.stderr)
        returncode = -9
    except OSError as exc:
        console = f"failed to spawn the profiling subprocess: {exc}"
        returncode = -1

    # Local artifact only: the child's stderr may quote the data, so it is written to
    # disk for a human and never returned into a state channel.
    with contextlib.suppress(OSError):
        log_path.write_text(console, encoding="utf-8")

    if returncode != 0 or not card_path.exists():
        raise ProfilingFailed(
            f"데이터셋 카드 생성이 실패했습니다 (returncode={returncode}). "
            f"자세한 로그: {log_path}"
        )
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfilingFailed(f"생성된 카드를 읽을 수 없습니다 ({card_path}): {exc}") from exc
    if not isinstance(card, dict):
        raise ProfilingFailed(f"생성된 카드가 JSON 객체가 아닙니다: {card_path}")
    print(f"  [profiling] 데이터셋 카드 생성 완료 → {card_path}")
    return card
