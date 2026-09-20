"""그래프 배선과 ``stop_condition`` — 결정론적인 루프 제어기.

루프 제어는 여기, 코드 안에 있다. LLM에게 "이제 멈춰야 하나?"를 묻는 일은 없다:
``stop_condition``과 그래프의 간선만이 계속할지를 정한다.

노드 모듈은 :func:`build_state_graph` 안에서 lazy하게 import한다. 그래서 루프 제어기는
scikit-learn이나 LLM SDK 없이 import하고 단위 테스트할 수 있다.
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


def stop_condition(
    state: AutoMLState, stall_limit: int = STALL_LIMIT, search_past_goal: bool = False
) -> StopReason | None:
    """네 가지 종료 조건 중 무엇이 성립하는지, 계속 돌아야 하면 ``None``.

    1. 목표 지표에 도달했다 — ``search_past_goal``이 아닌 한,
    2. 반복 예산을 다 썼다,
    3. 실행이 정체됐다 (개선 없는 반복이 ``stall_count``회 연속),
    4. 시간 예산을 다 썼다 (``--time-budget-sec``에서 holdout이 떼어 둔 몫을 뺀 것).

    2와 3은 무한 루프를 불가능하게 하고, ``search_past_goal``이 있든 없든 성립한다 — 그래서 그
    플래그는 반복을 더 쓰게 할 수는 있어도 풀어놓을 수는 없다. 플래그가 실제로 바꾸는 것:
    :attr:`automl_agent.config.RunConfig.search_past_goal`.

    **순서를 바꾸면 보고서가 멈춤을 부르는 이름이 바뀐다** — 네 조건 모두 같은 노드로 가고,
    ``out_of_time``이 마지막인 데는 이유가 있다 (``docs/rationale.md``).
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
    """보고서를 쓸지, 비평하고 다시 계획할지. 멈춤에 못 미치는 것은 모두 계속이다.

    결정은 :func:`stop_condition`이고 그것뿐이다. 그래서 ``report.stop_reason``은 이 조건들의 두
    번째 사본을 맞춰 둘 필요 없이 보고서를 위해 이유를 부를 수 있다.
    """
    return "report" if stop_condition(state, stall_limit, search_past_goal) else "critic"


def make_checkpointer(db_path: Path = CHECKPOINT_DB) -> BaseCheckpointSaver:
    """``thread_id``로 중단/재개하는 데 쓰는 SQLite 체크포인터를 연다.

    일부러 context manager가 *아니다* — 컴파일된 app은 어떤 ``with`` 블록보다 오래 살아야 CLI가
    계속 스트리밍할 수 있다. ``check_same_thread=False``인 이유는 LangGraph가 worker 스레드에서
    건드릴 수 있기 때문이다.

    데이터베이스 하나가 모든 ``thread_id``를 담으므로 ``run``이 쓰는 동안 ``show``나 ``predict``가
    읽는다. 그것을 오류가 아니라 평범한 일로 만드는 것이 ``timeout``(sqlite busy handler)과
    WAL이고, WAL은 설정이 실패해도 넘어간다 — 논증은 ``docs/rationale.md``.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=CHECKPOINT_TIMEOUT_SEC)
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA journal_mode = WAL")
    return SqliteSaver(conn)


class NodeFn(Protocol):
    """LangGraph가 원하는 형태로 묶인 노드: 파라미터 하나, 이름은 ``state``여야 한다.

    LangGraph의 노드 protocol은 파라미터 *이름*으로 맞추므로 맨 ``Callable[[AutoMLState], dict]``
    (위치 전용)은 그것을 만족하지 않는다.
    """

    def __call__(self, state: AutoMLState) -> dict: ...


def _bind(node: Callable[..., dict], config: RunConfig) -> NodeFn:
    """``config``를 노드에 묶어 ``(state)`` 파라미터 하나만 남긴다.

    그냥 ``partial(node, config=config)``로 하면 ``config``라는 이름의 파라미터가 그대로 보이는데,
    LangGraph는 그것을 자기 ``RunnableConfig``를 달라는 요청으로 읽고 경고한다. 감싸면 노드의
    이름 짓기가 그대로 남는다.

    실행의 시계도 여기서 접어 넣는다 — 모든 노드가 지나가는 자리가 여기뿐이다. 자기 ``budget``을
    낸 노드는 그것을 지킨다 (지금 그런 노드는 없다). 노드마다 재지 않는 이유와 wall clock이 아니라
    ``monotonic``인 이유: ``docs/rationale.md``.
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
    """노드와 간선을 배선한다. ``graph`` 명령이 그림을 그릴 수 있도록 ``compile``과 분리했다."""
    from .nodes.critic import critic
    from .nodes.evaluate import evaluate
    from .nodes.holdout import holdout
    from .nodes.model_selection import model_selection
    from .nodes.planning import planning
    from .nodes.profiling import profiling
    from .nodes.report import report
    from .nodes.training import training

    g: StateGraph = StateGraph(AutoMLState)
    # 노드 이름은 함수 이름이다. 아래 간선들이 그 문자열을 쓰므로 둘을 갈라 놓으면 배선이
    # 조용히 어긋난다.
    for node in (profiling, planning, model_selection, training, evaluate, holdout, critic, report):
        g.add_node(node.__name__, _bind(node, config))

    # profiling은 진입점이고 순환의 일부가 아니다: 카드는 원본 데이터에서 한 번 만들어지고,
    # 이후 planning을 지나는 모든 경로가 그것을 재사용한다.
    g.set_entry_point("profiling")
    g.add_edge("profiling", "planning")
    g.add_edge("planning", "model_selection")
    g.add_edge("model_selection", "training")
    g.add_edge("training", "evaluate")
    # ``holdout``이 결정 뒤 탈출 경로에 있는 것은 일부러다: 루프가 본 적 없는 행으로 저장된
    # 최고 모델을 채점하는데, 루프의 행동을 바꿀 수 있는 숫자라면 그것은 더 이상 떼어 둔
    # 숫자가 아니다.
    g.add_conditional_edges(
        "evaluate",
        partial(route, stall_limit=config.stall_limit, search_past_goal=config.search_past_goal),
        {"report": "holdout", "critic": "critic"},
    )
    g.add_edge("holdout", "report")
    g.add_edge("critic", "planning")  # 순환: critic의 판정을 갖고 다시 계획한다
    g.add_edge("report", END)
    return g


def build_graph(
    config: RunConfig,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """실행을 재개할 수 있도록 체크포인터와 함께 그래프를 컴파일한다."""
    saver = checkpointer if checkpointer is not None else make_checkpointer()
    return build_state_graph(config).compile(checkpointer=saver)
