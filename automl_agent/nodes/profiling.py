"""Profiling 노드: 비공개 데이터 참조를 공개 데이터셋 카드로 바꾼다.

실행 노드 — LLM 없음, 그리고 그래프의 진입점. 추론 노드가 절대 건드리지 않는 하나의 채널(``data_ref``)을
읽어 그들이 모두 읽는 하나의 채널(``dataset_card``)에 쓰므로, 카드는 데이터에서 추론으로 건너오는
*유일한* 것이다. 그 건넘은 관례가 아니라 노드 경계다.

training처럼 작업은 서브프로세스(``scripts/profile.py``)에서 일어난다: pandas는 프롬프트를 렌더하는
프로세스에 한 번도 import되지 않으므로, 데이터 행이 실수로 프롬프트에 들어갈 수 없다.

training과 달리 여기서의 실패는 Critic에게 넘기지 *않는다*. 없는 카드는 실험 결과가 아니라 설정 오류다 —
진단할 것도, 그것 없이 세울 만한 계획도 없다 — 그래서 이 노드는 raise하고 실행은 멈춘다.
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


class ProfilingFailed(RuntimeError):
    """카드를 세울 수 없었으므로 계획을 세울 대상이 없다."""


def profiling(state: AutoMLState, *, config: RunConfig) -> dict:
    """``{"dataset_card": ..., "goal": ...}``, 또는 카드가 이미 있으면 아무것도."""
    reference = dict(state.get("data_ref") or {})
    if state.get("dataset_card"):
        # 손으로 쓴 카드(또는 재개된 실행)가 이긴다: profiling은 그것을 절대 덮어쓰지 않는다.
        card = dict(state["dataset_card"])
        assert_protocol_matches(card, config, reference)
        # 이 경로의 목표는 같은 카드에서 ``initial_state``가 유도했다 — 더 잴 것이 없으므로 다시
        # 계산하는 대신 검사한다.
        assert_goal_is_usable(card, dict(state.get("goal") or {}), config)
        return {}

    path = reference.get("path")
    if not path:
        # 실제 데이터도 카드도 없다: executor가 카드가 선언한 모양에서 합성할 것이고, 여기에는 프로파일할
        # 것이 없다.
        return {}

    register_private(path)
    card = run_profiler(
        as_source(path),
        str(reference.get("target_column") or "target"),
        config,
        table=reference.get("table"),
        query=reference.get("query"),
    )
    # "auto" 모드에서 목표는 방금 측정된 기준 baseline에서 여기서 유도된다. baseline이 존재하는 첫
    # 순간이기 때문이다 — ``initial_state``는 짐작밖에 할 수 없었다. "fixed" 모드에서는 씨앗으로 받은
    # 바를 그대로 돌려준다.
    #
    # 그리고 ``--data`` 경로에서 *task*가 알려지는 첫 순간도 여기이므로, 다른 task에 속한 지표가 이 정답
    # 열을 채점할 수 있는 지표로 바뀌는 곳도 여기다(:func:`automl_agent.scoring.goal.resolve_goal`).
    # ``initial_state``에는 카드가 없어 잡아낼 수 없었다.
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
        # 카드는 우리 자신의 고정된 스크립트에서 왔으므로, 여기서의 위반은 그 스크립트가 퇴행했다는
        # 뜻이다 — 넘기기보다 멈추는 것이 나은 바로 그 경우다. 이 노드의 다른 실패들처럼 설정 오류이므로
        # Critic에게 넘기지 않는다.
        validate_card(card)
    except CardSchemaError as exc:
        raise ProfilingFailed(str(exc)) from exc
    # 방금 우리 스크립트가 이것을 썼으므로, 여기서의 어긋남은 운영자가 남의 카드를 넣었다는 것이 아니라
    # 재개된 실행 아래에서 규약이 바뀌었다는 뜻이다.
    assert_protocol_matches(card, config, reference)
    return {"dataset_card": public_card(card), "goal": goal}


def assert_protocol_matches(
    card: dict[str, Any], config: RunConfig, reference: dict[str, Any] | None = None
) -> None:
    """카드의 baseline이 이 실행이 쓰는 것과 다른 행에서 측정되었으면 멈춘다.

    목표 문턱값이 그 baseline에서 오므로, 규약 어긋남은 바와 그것에 견줄 점수가 서로 다른 두 분할에서
    왔다는 뜻이다 — 결과로 읽히면서 결과가 아닌 비교다. ``--on-missing-target``과 같은 태도: 조용히
    비교 불가능한 수는 멈춘 실행보다 나쁘다. ``protocol`` 블록이 없는 카드는 그 필드보다 앞서므로
    받아들인다(:func:`automl_agent.scoring.splits.protocol_mismatch`).

    group 열을 ``config``가 아니라 ``data_ref``에서 읽는 이유는 두 출처가 이미 거기서 해소되기 때문이다:
    ``--group-column``으로 프로파일된 카드는 그것을 비공개 ``data`` 블록에 적어 두므로,
    ``--dataset-card`` 실행은 그것을 물려받고 플래그를 뺐다고 거부당하지 *않는다* — 카드와 어긋나는
    명시적 플래그는 여전히 거부된다.

    층화도 같은 이유로 카드의 ``task``에서 읽는다: 연속 정답은 층화할 수 없으므로, 회귀 카드의
    ``stratified: false``는 이 실행과 어긋나는 것이 아니라 맞는 것이다.
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
    """지표가 이 정답 열을 채점할 수 없거나, 바를 유도할 수 없었으면 멈춘다.

    둘 다 실험 결과가 아니라 설정 오류이고, 둘 다 여기서는 싸고 나중에는 비싸다. 연속 정답에 대한 ``f1``은
    나쁜 점수를 내지 않고 점수를 *못* 낸다 — 루프는 계산된 적 없는 수에 대해 "목표 미달"을 보고하며
    예산 전부를 쓸 것이다. 그리고 ``None`` 바는 아무것에도 견주지 않는다: ``goal_met``이 구조적으로 모든
    시도에 False다.

    지표를 config가 아니라 **goal**에서 읽는다 — 실행이 판정받는 것은 ``goal["metric"]``이고,
    :func:`automl_agent.scoring.goal.resolve_goal`이 config의 지표를 이 정답 열이 가진 것으로 이미 바꿨을
    수 있다. ``config.metric``을 읽으면 방금 치환이 실행 가능하게 만든 바로 그 실행을 거부하게 된다.
    여기 남는 것은 치환이 닿지 못하는 경우다: 다른 무엇이 쓴 goal 채널(손으로 고친 체크포인트, 잊어버린
    미래의 호출자). 첫 줄이 아니라 뒤를 받치는 것.

    ``task``가 없거나 이 빌드가 모르는 라벨이면 그 필드보다 앞서므로 받아들인다. ``protocol`` 블록이 없는
    경우와 같은 조건이다 — 이것은 어긋남을 잡는 것이고, 판정할 수 없는 카드를 거부하는 것이 아니다.

    *빈* ``goal``도 받아들이고, 유도된 바가 ``None``인 것과 같지 않다: 아직 목표가 없다는 뜻이지 유도가
    실패했다는 뜻이 아니다. ``None`` 바를 내는 것은 ``mae``/``rmse``뿐이므로(이식 가능한 기본값이 없다),
    "목표 없음"을 "바 없음"으로 다루면 기본값이 *있는* ``f1``에 대해 단위 없음 메시지를 찍고 곧 돌아갈
    실행을 거부하게 된다.
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


def run_profiler(
    data_path: Path | str,
    target_column: str,
    config: RunConfig,
    *,
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """``scripts/profile.py``를 띄우고 그것이 쓴 카드를 되읽는다."""
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
        # baseline의 holdout이 train.py가 쓸 바로 그 분할이 되도록.
        "--seed",
        str(config.seed),
    ]
    if config.on_missing_target:
        # 설정되지 않았으면 스크립트 자신의 기본값에 맡긴다. 두 기본값이 어긋날 수 없게.
        command += ["--on-missing-target", config.on_missing_target]
    for note in config.caveats:
        # 원본 데이터에 대한 운영자의 지식이 추론 노드가 읽는 카드 필드가 되어 가는 길 —
        # automl_agent.dataset.caveats.
        command += ["--caveat", note]
    if config.group_column:
        # baseline이 train.py가 쓸 같은 group 인지 분할 아래에서 측정되도록. 이것이 없으면 바는 행 단위
        # 수가 되고 모든 시도는 그룹 밖 수가 되는데, 그것이 protocol_mismatch가 잡으려고 존재하는 바로
        # 그 비교 불가능성이다.
        command += ["--group-column", config.group_column]
    # DB 출처에서 어느 행을 프로파일하는가. ``config``가 아니라 인자로 받는 이유는 ``group_column``과
    # 같다 — 두 출처(카드의 비공개 블록과 CLI)가 이미 ``data_ref``에서 해소된다.
    if table:
        command += ["--table", str(table)]
    if query:
        command += ["--query", str(query)]

    returncode, console = run_fixed_script(command, timeout=PROFILE_TIMEOUT_SEC, label="profiling")

    # 로컬 artifact일 뿐이다: 자식의 stderr가 데이터를 인용할 수 있으므로, 사람을 위해 디스크에 쓰고
    # state 채널로는 절대 돌려보내지 않는다.
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
