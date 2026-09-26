"""Graph wiring and ``stop_condition``: the fixed, rule-based loop controller.

Roles:

* Stop decision — decide in code, never by the LLM.
* Checkpointing — open the SQLite store for pause and resume.
* Graph wiring — bind nodes to the config and add edges.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import (
    CHECKPOINT_DB,
    CHECKPOINT_TIMEOUT_SEC,
    DEFAULT_MAX_ITERATIONS,
    STALL_LIMIT,
    RunConfig,
)
from .state import AutoMLState, accrue_budget, goal_met, loop_budget_exhausted, state_int

Route = Literal["report", "critic"]
StopReason = Literal["goal_reached", "max_iterations", "stalled", "out_of_time"]


# --- Role: stop decision --------------------------------------------------------------


def stop_condition(
    state: AutoMLState, stall_limit: int = STALL_LIMIT, search_past_goal: bool = False
) -> StopReason | None:
    """Return the first stop condition that holds, or ``None`` to keep going.

    Order sets the reported name; ``out_of_time`` is last
    """
    if not search_past_goal and goal_met(state.get("result", {}) or {}, state.get("goal", {}) or {}):
        return "goal_reached"

    if state_int(state, "iteration") >= state_int(state, "max_iterations", DEFAULT_MAX_ITERATIONS):
        return "max_iterations"

    if state_int(state, "stall_count") >= stall_limit:
        return "stalled"

    if loop_budget_exhausted(state):
        return "out_of_time"

    return None


def route(state: AutoMLState, stall_limit: int = STALL_LIMIT, search_past_goal: bool = False) -> Route:
    """Return ``"report"`` when :func:`stop_condition` holds, else ``"critic"``."""
    return "report" if stop_condition(state, stall_limit, search_past_goal) else "critic"


# --- Role: checkpointing --------------------------------------------------------------


def make_checkpointer(db_path: Path = CHECKPOINT_DB) -> BaseCheckpointSaver:
    """Open the SQLite checkpointer that pauses and resumes runs by ``thread_id``.

    Not a context manager on purpose
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=CHECKPOINT_TIMEOUT_SEC)
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA journal_mode = WAL")
    return SqliteSaver(conn)


# --- Role: graph wiring ---------------------------------------------------------------


class NodeFn(Protocol):
    """A bound node with one parameter named ``state``.

    LangGraph matches parameters by name, so a plain ``Callable`` does not fit.
    """

    def __call__(self, state: AutoMLState) -> dict: ...


def _bind(node: Callable[..., dict], config: RunConfig) -> NodeFn:
    """_bind | Graph wiring: bind ``config`` and add the node's run time to ``budget``.

    Not a plain ``partial``;
    """

    def wrapped(state: AutoMLState) -> dict:
        started = time.monotonic()
        update = node(state, config=config)
        if isinstance(update, dict) and "budget" not in update:
            update["budget"] = accrue_budget(
                state.get("budget"), time.monotonic() - started, config.time_budget_sec
            )
        return update

    wrapped.__name__ = getattr(node, "__name__", "node")
    return wrapped


def build_state_graph(config: RunConfig) -> StateGraph:
    """Build the uncompiled graph; kept apart so the ``graph`` command can draw it."""
    from .nodes.critic import critic
    from .nodes.evaluate import evaluate
    from .nodes.holdout import holdout
    from .nodes.model_selection import model_selection
    from .nodes.planning import planning
    from .nodes.profiling import profiling
    from .nodes.report import report
    from .nodes.training import training

    g: StateGraph = StateGraph(AutoMLState)
    # Edges use these names; renaming a node breaks wiring silently.
    for node in (profiling, planning, model_selection, training, evaluate, holdout, critic, report):
        g.add_node(node.__name__, _bind(node, config))

    # Entry point only; the data card is built once.
    g.set_entry_point("profiling")
    g.add_edge("profiling", "planning")
    g.add_edge("planning", "model_selection")
    g.add_edge("model_selection", "training")
    g.add_edge("training", "evaluate")
    # Holdout runs after the stop decision
    g.add_conditional_edges(
        "evaluate",
        partial(route, stall_limit=config.stall_limit, search_past_goal=config.search_past_goal),
        {"report": "holdout", "critic": "critic"},
    )
    g.add_edge("holdout", "report")
    g.add_edge("critic", "planning")  # the cycle: re-plan with the critic's verdict
    g.add_edge("report", END)
    return g


def build_graph(
    config: RunConfig,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the graph with a checkpointer (a new one when omitted) for resume."""
    saver = checkpointer if checkpointer is not None else make_checkpointer()
    return build_state_graph(config).compile(checkpointer=saver)
