"""Graph wiring plus ``route`` — the deterministic loop controller.

Loop control lives here, in code. The LLM is never asked "should we stop now?":
``route`` and the graph edges are the only things that decide continuation.

Node modules are imported lazily inside :func:`build_graph` so that ``route`` can
be imported (and unit-tested) without pulling in scikit-learn or the LLM SDK.
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
from .state import AutoMLState, accrue_budget, goal_met, loop_budget_exhausted

Route = Literal["report", "critic"]


def route(
    state: AutoMLState, stall_limit: int = STALL_LIMIT, search_past_goal: bool = False
) -> Route:
    """Decide whether to write the report or to critique and replan.

    Terminates on any of four conditions:

    1. the goal metric is reached — unless ``search_past_goal``,
    2. the iteration budget is exhausted,
    3. the run has stalled (``stall_count`` consecutive non-improving iterations),
    4. the time budget is exhausted (``--time-budget-sec``, less holdout's reserved share).

    Anything else continues through the critic. Conditions 2 and 3 make an infinite loop
    impossible and hold with or without ``search_past_goal`` — which is why that flag can only cost
    iterations, never unbound them.

    **Condition 4 is checked last, and the order is the point.** The first three are conditions the
    loop reached on its own terms — found what it wanted, spent what it was given, stopped moving —
    and any of them would have ended the run with the clock stopped. The budget is the reason a run
    ended only when it cut short a loop that was otherwise still going, so ``out_of_time`` names
    exactly that. Checked here at all because it was checked nowhere: ``--time-budget-sec`` went to
    each training subprocess as its own timeout and nothing else read it, so five iterations at the
    3600s default bounded the run at 21,600 seconds.

    **Condition 1 first, and what ``search_past_goal`` changes.** In ``auto`` mode the bar comes
    from the card's baseline, so a first attempt that clears it ends the run at iteration 1 and the
    Critic never runs — four of the five datasets in ``docs/RESULTS.md`` ended that way, meaning
    the diagnose-and-replan path the benchmark compared did not execute on either arm.
    ``search_past_goal`` keeps a run going anyway: it does not raise the bar and does not change
    which attempt wins (``best`` is still val-best), it only spends the remaining iterations. Off by
    default, because every number recorded under the old behaviour was measured with a budget this
    flag changes (:attr:`automl_agent.config.RunConfig.search_past_goal`).

    ``goal_met`` is still evaluated on this iteration's result and still reaches the report —
    ``report.stop_reason`` recomputes it, so a run that cleared the bar at iteration 1 and spent
    four more says ``goal_reached`` whatever the last attempt scored.
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

    if loop_budget_exhausted(state):
        return "report"

    return "critic"


def make_checkpointer(db_path: Path = CHECKPOINT_DB) -> BaseCheckpointSaver:
    """Open the SQLite checkpointer used for interrupt/resume by ``thread_id``.

    Deliberately *not* in a context manager — the compiled app must outlive any single ``with``
    block so the CLI can keep streaming. ``check_same_thread=False`` because LangGraph may touch it
    from a worker thread.

    One database holds every ``thread_id``, so ``show`` or ``predict`` can read while a ``run``
    writes. Two settings make that ordinary rather than an error:

    * ``timeout`` — sqlite's busy handler. The stdlib default of 5s is short for a checkpoint write
      waiting on another process's lock, and a run an hour into training should wait, not abort.
    * WAL — lets readers proceed during a write at all. A property of the file, so set once and it
      survives; if the filesystem cannot support it (a network share) the journal mode is left as it
      was. This is a concurrency improvement, not a requirement, and refusing to open the database
      over it would be worse.
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

    A plain ``partial(node, config=config)`` would still expose a parameter named ``config``, which
    LangGraph reads as a request for its own ``RunnableConfig`` and warns about. Wrapping keeps the
    node's naming intact.

    The run's clock lives here because this is the only place every node passes through. A budget
    that counts the nodes which remembered to report their own time is not a budget, and the easiest
    time to forget is the reasoning nodes' — an LLM call retrying through the backoff can cost
    minutes while measuring nothing. So the wrapper times the call and folds the seconds into
    ``budget``, and a node stays a function of ``state`` that knows nothing about the clock.

    ``monotonic``, not wall clock: the accumulated number is a duration, and a system clock
    adjustment mid-fit must not hand the run more budget or less. The total survives in the
    checkpoint, so ``resume`` continues the budget rather than restarting it — or counting the hours
    the process was not running.

    A node returning its own ``budget`` keeps it. Nothing does; the check exists so that if
    something ever needs to (correcting for time it knows was not spent), the wrapper does not
    silently overwrite it.
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
