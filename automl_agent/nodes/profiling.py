"""Profiling node: turns the private data reference into the public dataset card.

Roles:

* Card building: run ``scripts/profile.py`` and read the card.
* Goal check: stop if the metric or bar is unusable.
* Protocol check: stop if the baseline used different rows.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from ..config import PROFILE_SCRIPT, PROFILE_TIMEOUT_SEC, RunConfig, run_fixed_script
from ..dataset.source import as_source
from ..privacy import CardSchemaError, public_card, register_private, validate_card
from ..scoring.goal import describe, missing_bar_message, resolve_goal
from ..scoring.metrics import TASK_REGRESSION, card_task, metrics_for, task_of
from ..scoring.splits import protocol_mismatch
from ..state import AutoMLState

# --- Role: card building --------------------------------------------------------------


class ProfilingFailed(RuntimeError):
    """Raised when the card could not be built or checked."""


def profiling(state: AutoMLState, *, config: RunConfig) -> dict:
    """Build the dataset card and goal, or check an existing card.

    Returns ``{}`` when a card exists or there is no data; raises ``ProfilingFailed``.
    """
    reference = dict(state.get("data_ref") or {})
    if state.get("dataset_card"):
        # An existing card wins; never overwrite it.
        card = dict(state["dataset_card"])
        assert_protocol_matches(card, config, reference)
        # ``initial_state`` already built this goal; only check it.
        assert_goal_is_usable(card, dict(state.get("goal") or {}), config)
        return {}

    path = reference.get("path")
    if not path:
        # No data: the executor makes synthetic data instead.
        return {}

    register_private(path)
    card = run_profiler(
        as_source(path),
        str(reference.get("target_column") or "target"),
        config,
        table=reference.get("table"),
        query=reference.get("query"),
    )
    # Goal and metric swap happen here
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
        # Our own script wrote it, so an error means a bug.
        validate_card(card)
    except CardSchemaError as exc:
        raise ProfilingFailed(str(exc)) from exc
    # A mismatch here means a resumed run changed protocol.
    assert_protocol_matches(card, config, reference)
    return {"dataset_card": public_card(card), "goal": goal}


# --- Role: protocol check -------------------------------------------------------------


def assert_protocol_matches(
    card: dict[str, Any], config: RunConfig, reference: dict[str, Any] | None = None
) -> None:
    """Raise ``ProfilingFailed`` if the card's baseline used other rows than this run.

    Cards with no ``protocol`` block pass
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


# --- Role: goal check -----------------------------------------------------------------


def assert_goal_is_usable(
    card: dict[str, Any], goal: dict[str, Any], config: RunConfig
) -> None:
    """Raise ``ProfilingFailed`` if the metric cannot score the target, or no bar exists.

    Reads the metric from ``goal``, not ``config``
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


# --- Role: card building (subprocess) -------------------------------------------------


def run_profiler(
    data_path: Path | str,
    target_column: str,
    config: RunConfig,
    *,
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Run ``scripts/profile.py`` and return the card it wrote.

    Raises ``ProfilingFailed`` if the script fails or the card is bad.
    """
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
        # Same split as train.py for the baseline.
        "--seed",
        str(config.seed),
    ]
    if config.on_missing_target:
        # Unset: use the script default, so they agree.
        command += ["--on-missing-target", config.on_missing_target]
    for note in config.caveats:
        # Operator notes become a card field (dataset.caveats).
        command += ["--caveat", note]
    if config.group_column:
        # Same group split as train.py, so scores compare.
        command += ["--group-column", config.group_column]
    # From ``data_ref``, which already merged card and CLI.
    if table:
        command += ["--table", str(table)]
    if query:
        command += ["--query", str(query)]

    returncode, console = run_fixed_script(command, timeout=PROFILE_TIMEOUT_SEC, label="profiling")

    # Stderr may quote data: disk only, never state.
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
