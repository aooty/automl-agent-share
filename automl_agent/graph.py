"""Graph wiring plus ``route`` — the deterministic loop controller.

Loop control lives here, in code. The LLM is never asked "should we stop now?":
``route`` and the graph edges are the only things that decide continuation.

Node modules are imported lazily inside :func:`build_graph` so that ``route`` can
be imported (and unit-tested) without pulling in scikit-learn or the LLM SDK.
"""

from __future__ import annotations

import contextlib
import sqlite3
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
from .state import AutoMLState, goal_met

Route = Literal["report", "critic"]


def route(
    state: AutoMLState, stall_limit: int = STALL_LIMIT, search_past_goal: bool = False
) -> Route:
    """Decide whether to write the report or to critique and replan.

    Terminates on any of three conditions:

    1. the goal metric is reached — unless ``search_past_goal``,
    2. the iteration budget is exhausted,
    3. the run has stalled (``stall_count`` consecutive non-improving iterations).

    Anything else continues the loop through the critic. Conditions 2 and 3 are what make an
    infinite loop impossible, and they hold with or without ``search_past_goal`` — which is why
    that flag can only ever cost iterations, never unbound them.

    On condition 1 being checked first, and what ``search_past_goal`` changes. In ``auto`` mode
    the bar is derived from the card's baseline, so a first attempt that clears it ends the run
    at iteration 1 and the Critic never runs — four of the five datasets in ``bench/RESULTS.md``
    ended exactly that way, which means the diagnose-and-replan path the benchmark was comparing
    did not execute on either arm. ``search_past_goal`` is how a run keeps going anyway: it does
    not raise the bar and it does not change which attempt wins (``best`` is still val-best), it
    only spends the remaining iterations. Off by default, because every number recorded under the
    old behaviour was measured with a budget this flag changes — see
    :attr:`automl_agent.config.RunConfig.search_past_goal`.

    ``goal_met`` is still evaluated on this iteration's result and still reaches the report:
    ``report.stop_reason`` recomputes it, so a run that cleared the bar at iteration 1 and then
    spent four more says ``goal_reached`` regardless of what the last attempt scored.
    """
    if not search_past_goal and goal_met(
        state.get("result", {}) or {}, state.get("goal", {}) or {}
    ):
        return "report"

    iteration = int(state.get("iteration", 0) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS)
    if iteration >= max_iterations:
        return "report"

    if int(state.get("stall_count", 0) or 0) >= stall_limit:
        return "report"

    return "critic"


def make_checkpointer(db_path: Path = CHECKPOINT_DB) -> BaseCheckpointSaver:
    """Open the SQLite checkpointer used for interrupt/resume by ``thread_id``.

    The connection is deliberately *not* wrapped in a context manager: the
    compiled app must outlive any single ``with`` block so the CLI can keep
    streaming. ``check_same_thread=False`` because LangGraph may touch it from a
    worker thread.

    One database holds every ``thread_id``, so two commands on the same machine share it —
    ``show`` or ``predict`` read it while a ``run`` writes. The two settings below are what
    make that ordinary instead of an error:

    * ``timeout`` is sqlite's busy handler. The stdlib default is 5 seconds, which a
      checkpoint write can exceed while another process holds the lock; a run that has
      already spent an hour training should wait, not abort.
    * WAL lets readers proceed during a write at all. It is a property of the file, so it
      is set once and survives; if the filesystem cannot support it (a network share), the
      journal mode is left as it was — this is a concurrency improvement, not a
      requirement, and refusing to open the database over it would be worse.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=CHECKPOINT_TIMEOUT_SEC)
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA journal_mode = WAL")
    return SqliteSaver(conn)


class NodeFn(Protocol):
    """A bound node as LangGraph wants it: one parameter, and it must be named ``state``.

    LangGraph's own node protocol matches on the parameter *name*, so a bare
    ``Callable[[AutoMLState], dict]`` (positional-only) would not satisfy it.
    """

    def __call__(self, state: AutoMLState) -> dict: ...


def _bind(node: Callable[..., dict], config: RunConfig) -> NodeFn:
    """Bind ``config`` into a node, leaving a single-parameter ``(state)`` signature.

    A plain ``partial(node, config=config)`` would still expose a parameter named
    ``config``, which LangGraph reads as a request for its own ``RunnableConfig``
    and warns about. Wrapping keeps the node's own naming intact.
    """

    def wrapped(state: AutoMLState) -> dict:
        return node(state, config=config)

    wrapped.__name__ = getattr(node, "__name__", "node")
    return wrapped


def build_state_graph(config: RunConfig) -> StateGraph:
    """Wire the nodes and edges. Separated from ``compile`` so ``graph`` can draw it."""
    from .nodes.critic import critic
    from .nodes.evaluate import evaluate
    from .nodes.holdout import holdout
    from .nodes.model_selection import model_selection
    from .nodes.planning import planning
    from .nodes.profiling import profiling
    from .nodes.report import report
    from .nodes.training import training

    g: StateGraph = StateGraph(AutoMLState)
    g.add_node("profiling", _bind(profiling, config))
    g.add_node("planning", _bind(planning, config))
    g.add_node("model_selection", _bind(model_selection, config))
    g.add_node("training", _bind(training, config))
    g.add_node("evaluate", _bind(evaluate, config))
    g.add_node("holdout", _bind(holdout, config))
    g.add_node("critic", _bind(critic, config))
    g.add_node("report", _bind(report, config))

    # profiling is the entry point, not part of the cycle: the card is built once from
    # the raw data, and every later pass through planning reuses it.
    g.set_entry_point("profiling")
    g.add_edge("profiling", "planning")
    g.add_edge("planning", "model_selection")
    g.add_edge("model_selection", "training")
    g.add_edge("training", "evaluate")
    # ``route`` still answers "report" or "critic" — the stopping decision is unchanged.
    # The exit path just goes through ``holdout`` first, which scores the best saved model
    # on the rows the loop never saw. It is placed *after* the decision on purpose: a
    # number that could change the loop's behaviour would no longer be a held-back one.
    g.add_conditional_edges(
        "evaluate",
        partial(
            route, stall_limit=config.stall_limit, search_past_goal=config.search_past_goal
        ),
        {"report": "holdout", "critic": "critic"},
    )
    g.add_edge("holdout", "report")
    g.add_edge("critic", "planning")  # the cycle: replan with the critic's verdict
    g.add_edge("report", END)
    return g


def build_graph(
    config: RunConfig,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the graph with a checkpointer so runs are resumable."""
    saver = checkpointer if checkpointer is not None else make_checkpointer()
    return build_state_graph(config).compile(checkpointer=saver)
