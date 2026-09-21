"""Report 에이전트: 모든 종료 경로에서 나오는 최종 보고서.

목표를 놓친 실행도 보고서를 받는다. ``best`` 스냅샷에서 쓴다 — 루프의 요점은 그것이 모은 증거이고,
통과 그 자체가 아니다.

이 노드는 마지막 반복을 ``history``에 덧붙이고(Critic은 그것을 본 적이 없다) 실행의 artifact
디렉터리에 ``report.md``와 ``history.json``을 쓴다.
"""

from __future__ import annotations

import json
from typing import Any

from ..capabilities import describe as describe_capabilities
from ..config import RunConfig, file_size_text
from ..dataset.caveats import describe_caveats
from ..graph import stop_condition
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.goal import describe, goal_threshold
from ..scoring.intervals import as_number, contains, describe_interval, interval_of
from ..state import AutoMLState, build_attempt, describe_budget, goal_met, state_int
from .holdout import describe as describe_holdout
from .model_selection import _history_digest, digest_attempt, task_of_state

STOP_REASON_LABELS = {
    "goal_reached": "목표 지표 달성",
    "max_iterations": "최대 반복 횟수 도달",
    "stalled": "연속 미개선(정체)으로 조기 종료",
    # 그 밖에는 계속 갈 루프를 시계가 끝냈을 때만 쓴다 — ``automl_agent.graph.stop_condition``의
    # 순서 메모 참고.
    "out_of_time": "시간 예산 소진으로 조기 종료",
    "unknown": "알 수 없는 사유",
}

# 보고서는 실행의 유일한 장문 출력이고 딱 한 번 생성되므로, 추론 노드의 구조화된 응답보다 몫을
# 넉넉히 받는다. 적응적 사고와 산문이 같은 몫에서 나오기 때문이다.
REPORT_MAX_TOKENS = 24000

# 모델이 끊겼을 때 덧붙인다. 콘솔만이 아니라 보고서 자체에 넣는 이유는 나중에 읽히고 전달되는
# artifact가 ``report.md``이고, 문장 중간에 멈춘 글은 그 표시가 없으면 그저 무언가를 생략한 완결된
# 결론으로 읽히기 때문이다.
TRUNCATION_NOTE = (
    "> **주의 — 이 보고서는 완결되지 않았습니다.** 생성 중 출력 상한"
    f"({REPORT_MAX_TOKENS} 토큰)에 닿아 마지막 문장이 끊겼고, 뒤에 올 절이 빠져 있을 수 "
    "있습니다. 위에 적힌 내용 자체는 유효하지만 **결론으로 읽지 마십시오.** 모든 시도의 지표는 "
    "`history.json`에, 프롬프트와 응답 원본은 `llm/` 아래에 그대로 남아 있습니다."
)


def report(state: AutoMLState, *, config: RunConfig) -> dict:
    """``{"report": markdown, "history": [마지막 시도]}``."""
    reason = stop_reason(state, config)
    reached = run_met_goal(state)
    final_attempt = build_attempt(state, None)
    # 마지막 시도는 아직 state["history"]에 없으므로 보고서를 위해 여기서 넣는다.
    full_history = [*_history_digest(state), digest_attempt(final_attempt)]

    variables = {
        "goal": state.get("goal") or {},
        "goal_met": "yes" if reached else "no",
        "iterations": str(state.get("iteration") or 0),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "stop_reason": f"{reason} ({STOP_REASON_LABELS.get(reason, reason)})",
        # 아래 시도 중 몇 개가 진단되었는지. 문장으로 건네는 이유는 보고서 프롬프트가 "Critic 판정
        # 전반의 패턴"을 청하는데, 판정이 0개면 그 지시가 패턴을 발명하라고 부추기기 때문이다 —
        # describe_replanning 참고.
        "replanning": describe_replanning(full_history),
        "best": state.get("best") or "(성공한 시도 없음)",
        "history": full_history,
        "dataset_card": state.get("dataset_card") or {},
        # 카드의 집계가 보여 줄 수 없는 것, 그리고 보고서의 모든 권고가 살아남아야 하는 것 —
        # automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # 어떤 결정도 여기에 대고 선택되지 않은 하나뿐인 수. dict가 아니라 문장으로 주므로, 다른
        # 시도의 지표로 오인되어 함께 평균되는 일이 없다.
        "holdout": describe_holdout(dict(state.get("holdout") or {})),
        # 보고서가 "계획이 청한 것"과 "실제로 돈 것"을 가를 수 있게. 실행되지 않은 전략이
        # 최고 성능 구성 아래 실리지 않도록.
        "executor_capabilities": describe_capabilities(task_of_state(state)),
    }

    text = ""
    truncated = False
    if not config.use_llm:
        archive_prompt_only(config, "report", render_prompt("report", variables))
    else:
        try:
            text, truncated = LLMClient(config).complete_text(
                "report",
                variables,
                # ``max()``인 이유: 상한을 일부러 올린 config가 여기서 낮춰지지 않게. 하한은 이
                # 노드의 것이고 선택은 호출자에게 남는다.
                max_tokens=max(REPORT_MAX_TOKENS, config.llm_max_tokens),
            )
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [report] LLM 보고서 생성 실패({exc}) — 템플릿 보고서로 폴백합니다")

    if truncated:
        print(
            f"  [report] 보고서가 출력 상한({max(REPORT_MAX_TOKENS, config.llm_max_tokens)} "
            "토큰)에서 끊겼습니다 — report.md가 완결되지 않았습니다"
        )
    if not text.strip():
        # 사고 중에 몫이 떨어져 산문이 아예 안 돌아온 경우도 포함한다. 템플릿 보고서는 그 자체로
        # 완결이므로 끊김 표시가 필요 없다 — 장문 보고서를 잃었다고 말하는 것은 위의 콘솔 줄이다.
        text = fallback_report(state, config, reason, reached, full_history)
    elif truncated:
        text = f"{text.rstrip()}\n\n{TRUNCATION_NOTE}\n"

    _write_artifacts(state, config, text, full_history)
    return {"report": text, "history": [final_attempt]}


def describe_replanning(history: list[dict[str, Any]]) -> str:
    """루프가 실제로 얼마나 돌았는지, 한 문장으로.

    마지막 시도에 ``critic``이 없는 것은 구조적이다(루프가 그것을 평가한 뒤 멈추므로 그 판정은 일어나지
    않는 재계획을 먹일 것이다) — 그래서 문장이 그 시도를 지목한다. 독자가 빈 행을 스스로 설명하게 두지
    않으려고.
    """
    attempts = len(history)
    diagnosed = sum(1 for item in history if item.get("critic"))
    if not attempts:
        return "시도가 없어 재계획에 대해 말할 것이 없습니다."
    if not diagnosed:
        return (
            f"critic이 한 번도 실행되지 않았습니다 (시도 {attempts}회, 진단 0회). 진단·재계획 "
            "경로는 이 결과에 기여하지 않았습니다 — 점수는 첫 계획 하나가 낸 것입니다. 그 경로가 "
            "도움이 된다는 증거도, 해가 된다는 증거도 이 실행에는 없습니다."
        )
    return (
        f"critic이 {diagnosed}회 실행되어 그만큼 재계획했습니다 (시도 {attempts}회 — 마지막 "
        "시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다)."
    )


def run_met_goal(state: AutoMLState) -> bool:
    """이 실행이 바를 넘는 모델을 냈는지 — *마지막* 것이 넘었는지가 아니다.

    마지막 결과와 함께 ``best``로도 판정한다. ``best``가 실행의 답이기 때문이다 — ``holdout``이 채점한
    모델이고 ``predict``가 찾아가는 모델이다. 기본 라우팅에서는 두 질문의 답이 같다(시도가 바를 넘는
    순간 루프가 멈추고, ``best``가 그 시도다). :attr:`automl_agent.config.RunConfig.search_past_goal`
    아래에서는 갈라진다: 실행이 계속 가고, 이긴 모델이 ``best``에 그대로 앉아 있는데도 뒤의 더 나쁜
    시도가 "달성"을 "미달성"으로 바꿀 것이다.

    어느 쪽이든 만족시키고, ``goal_met``은 없거나 숫자가 아닌 점수를 거절한다 — 그래서 빈 ``best``(성공한
    적합 없음)는 raise가 아니라 False다.
    """
    goal = dict(state.get("goal") or {})
    return goal_met(dict(state.get("result") or {}), goal) or goal_met(dict(state.get("best") or {}), goal)


def stop_reason(state: AutoMLState, config: RunConfig) -> str:
    """루프가 왜 끝났는지의 이름 — ``route``가 한 그 판단 그대로이고, 다시 계산한 것이 아니다.

    ``unknown``은 여기로 라우팅된 적이 없는 state를 위한 것이다: 테스트에서 손으로 세운 것, 또는 채널이
    생기기 전에 쓰인 체크포인트. ``search_past_goal`` 아래에서는 실행이 반복 예산에서 멈추면서 *동시에*
    목표를 달성했을 수 있고, 그것은 :func:`run_met_goal`이 따로 말한다.
    """
    return stop_condition(state, config.stall_limit, config.search_past_goal) or "unknown"


# --------------------------------------------------------------------------- #
# 결정적인 보고서 (--dry-run, 그리고 LLM을 못 쓸 때의 폴백)
# --------------------------------------------------------------------------- #


def fallback_report(
    state: AutoMLState,
    config: RunConfig,
    reason: str,
    reached: bool,
    history: list[dict[str, Any]],
) -> str:
    """축적된 state만으로 세우는 템플릿 보고서."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    threshold = goal_threshold(goal, config.fallback_threshold)
    best = dict(state.get("best") or {})
    best_score = as_number(best.get("score"))
    iterations = state_int(state, "iteration")
    max_iterations = state_int(state, "max_iterations", config.max_iterations)

    verdict_line = (
        f"목표 `{metric} >= {threshold}`를 **달성**했습니다."
        if reached
        else f"목표 `{metric} >= {threshold}`를 **달성하지 못했습니다**."
    )
    # 인용되는 점수의 구간, 실행이 그것을 쟀을 때. 설정 블록만이 아니라 요약 문장에 적는 이유는
    # 독자가 다른 곳에 인용하는 것이 이 문장이고, 맨 0.7503은 정확한 값처럼 돌아다니기 때문이다.
    bounds = interval_of(dict(best.get("metrics") or {}), metric)
    best_line = (
        f"최고 성능은 iteration {best.get('iteration')}의 `{best.get('model')}`이며 "
        f"{describe_interval(metric, best_score, bounds)}입니다."
        if best_score is not None
        else "성공적으로 학습을 마친 시도가 없어 유효한 최고 성능이 없습니다."
    )
    if contains(bounds, threshold):
        best_line += (
            f" 다만 목표 {threshold}가 이 구간 안에 있어, 이 검증 슬라이스는 달성 여부를 "
            "판정할 만큼 두 값을 구분하지 못합니다 — 아래 최종 검증 점수를 보십시오."
        )

    lines: list[str] = [
        f"# AutoML 실행 보고서 — `{config.thread_id}`",
        "",
        "## 요약",
        "",
        f"{verdict_line} {best_line} 총 {iterations}회 시도했고(최대 {max_iterations}회), "
        f"종료 사유는 **{STOP_REASON_LABELS.get(reason, reason)}**입니다. "
        # 자원 절이 아니라 반복 횟수 옆에 두는 이유는 두 예산이 함께 읽히기 때문이다: "5회 중
        # 2회"만으로는 나머지 세 번이 왜 안 쓰였는지 말하지 못한다.
        f"시간 예산 사용: {describe_budget(dict(state.get('budget') or {}))}.",
        "",
        # 원인 분석만이 아니라 요약에도 넣는 이유: iteration 1의 "목표 지표 달성"은 루프가
        # 통했다는 뜻으로 읽히는데, iteration 1에서 루프는 돌지 않았다.
        f"재계획: {describe_replanning(history)}",
        "",
        # 바가 어디서 왔는지. 이것 없이 "0.872 달성"이라고 적은 보고서는 그 수가 이 데이터셋의
        # 것이 아니라 보편 기준인 것처럼 읽힌다.
        f"목표 기준: {describe(goal)}",
        "",
        # 요약 바로 아래인 이유: 그 밑 표의 모든 수는 선택에 쓰였고 이 수는 아니다.
        f"최종 검증: {describe_holdout(dict(state.get('holdout') or {}))}",
        "",
        "## 시도별 경과",
        "",
        "| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for attempt in history:
        metrics = attempt.get("metrics") or {}
        score = as_number(metrics.get(metric))
        if attempt.get("status") == "ok" and score is not None:
            outcome = f"{metric}={score:.4f}"
        else:
            outcome = f"실패 (`{attempt.get('error_type') or 'unknown'}`)"
        critic_verdict = attempt.get("critic") or {}
        diagnosis = f"`{critic_verdict.get('failure_type')}`" if critic_verdict else "—"
        lines.append(
            f"| {attempt.get('iteration')} | `{attempt.get('model')}` | "
            f"{_format_hyperparams(attempt.get('hyperparams') or {})} | {outcome} | {diagnosis} |"
        )

    lines += [
        "",
        "## 최고 성능 구성",
        "",
    ]
    if best:
        lines += [
            f"- model: `{best.get('model')}`",
            f"- hyperparams: {_format_hyperparams(best.get('hyperparams') or {})}",
            f"- preprocessing: {_format_hyperparams(best.get('preprocessing') or {})}",
            f"- {metric}: {best_score:.4f}" if best_score is not None else f"- {metric}: 없음",
            f"- 전체 지표: {_format_hyperparams(best.get('metrics') or {})}",
            f"- 학습 시간: {best.get('train_time_sec')}초",
            f"- 재현: `seed={config.seed}`, iteration {best.get('iteration')}",
        ]
    else:
        lines.append("유효한 최고 성능 구성이 없습니다.")

    lines += ["", "## 원인 분석", ""]
    diagnoses = [
        (attempt.get("critic") or {}).get("failure_type") for attempt in history if attempt.get("critic")
    ]
    if diagnoses:
        counts: dict[str, int] = {}
        for name in diagnoses:
            counts[str(name)] = counts.get(str(name), 0) + 1
        dominant = max(counts, key=lambda key: counts[key])
        area = "자원 제약" if dominant in {"oom", "too_slow"} else "모델/데이터 적합도"
        lines.append(
            f"Critic 진단 분포: {counts}. 가장 빈번한 원인은 `{dominant}`이며, {area} 쪽 문제로 수렴했습니다."
        )
        last_direction = next(
            (
                (attempt.get("critic") or {}).get("direction")
                for attempt in reversed(history)
                if attempt.get("critic")
            ),
            None,
        )
        if last_direction:
            lines += ["", f"마지막 Critic 방향: {last_direction}"]
    else:
        # 없다는 것이 *무엇을 뜻하는지*. 셈은 위 요약이 이미 준다. 이것이 없으면 이 절은 실행을
        # 무엇에 인용할 수 있는지에 대한 한계가 아니라 글의 빈틈으로 읽힌다.
        lines.append(
            "Critic이 실행되지 않아 축적된 진단이 없습니다. 이 실행이 잰 것은 첫 계획의 "
            "품질이고, 진단·재계획 경로는 여기서 평가되지 않았습니다 — 이 결과를 그 경로의 "
            "근거로 인용할 수 없습니다."
        )

    lines += ["", "## 다음 단계 제안", ""]
    lines += [f"- {item}" for item in _next_steps(reason, reached, diagnoses, metric, threshold)]
    mode = "--dry-run 모의 실행" if config.dry_run else "실제 실행"
    lines += ["", "---", "", f"*이 보고서는 {mode} 결과입니다.*", ""]
    return "\n".join(lines)


def _next_steps(
    reason: str,
    reached: bool,
    diagnoses: list[Any],
    metric: str,
    threshold: float,
) -> list[str]:
    if reached:
        return [
            "동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.",
            "홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.",
            "학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.",
        ]
    steps = [
        f"`--max-iterations`를 늘려 탐색 예산을 확대한다 "
        f"(현재 예산 안에서는 {metric} {threshold}에 도달하지 못했다).",
    ]
    names = [str(item) for item in diagnoses]
    if "oom" in names or "too_slow" in names:
        steps.append(
            "메모리·시간 예산(`constraints.memory_limit_mb`, `--time-budget-sec`)을 올리고 재실행한다."
        )
    if reason == "stalled":
        steps.append(
            "동일 계열 안에서의 미세 조정이 정체됐으므로 특성 공학 또는 데이터 품질 개선을 먼저 시도한다."
        )
    steps += [
        "데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — "
        "계획 품질은 카드 정확도에 종속된다.",
        "목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.",
    ]
    return steps


def _format_hyperparams(hyperparams: dict[str, Any]) -> str:
    if not hyperparams:
        return "—"
    parts = []
    for key, value in sorted(hyperparams.items()):
        parts.append(f"`{key}`={round(value, 4) if isinstance(value, float) else value}")
    return ", ".join(parts)


def prune_models(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """아무것도 더는 가리킬 수 없는 적합된 모델을 지우고, 어느 것인지 말한다.

    여기서 도는 이유는 이 노드가 실행의 마지막이고, 그래프 안에서 모델 파일을 읽는 유일한 소비자인
    ``holdout``이 이미 이긴 것을 채점했기 때문이다. 남는 것은 ``predict``가 찾아가는 모델이므로 실행은
    계속 적용 가능하다. 가는 것은 진 시도들의 것이고, 그것에는 ``--iteration`` 없이 닿는 명령이 없다.

    치명적이지 않고, 조용하지도 않다. 나중에 설명할 수 없는 삭제는 그것이 아끼는 디스크보다 나쁘므로,
    반환값이 반복 번호와 바이트 수와 그것을 남겼을 플래그를 적고, 찍기만 하지 않고 ``history.json``에도
    쓴다.
    """
    freed = 0
    removed: list[int] = []
    kept = dict(state.get("best") or {}).get("iteration")
    if config.keep_models == "all" or not isinstance(kept, int):
        # 최고 반복이 없다는 것은 성공한 적합이 없다는 뜻이므로, 남길 승자도 없고 나머지를 패자라
        # 부를 근거도 없다 — 그 실행의 파일이 그것이 가진 증거 전부다.
        return {"mode": config.keep_models, "kept_iteration": kept, "removed": [], "freed_bytes": 0}
    for iteration in range(1, state_int(state, "iteration") + 1):
        if iteration == kept:
            continue
        path = config.model_path(iteration)
        try:
            size = path.stat().st_size
        except OSError:
            continue  # 적합된 적 없음, 이미 없음, 또는 못 읽음 — 회수할 것이 없다
        try:
            path.unlink()
        except OSError as exc:
            print(f"  [report] iteration {iteration}의 모델을 지우지 못했습니다: {exc}")
            continue
        freed += size
        removed.append(iteration)
    if removed:
        print(
            f"  [report] iteration {', '.join(str(item) for item in removed)}의 모델 파일을 "
            f"지웠습니다 ({file_size_text(freed)} 회수) — iteration {kept}의 모델은 남아 있어 "
            "predict가 그대로 동작합니다. 진 시도의 모델까지 남기려면 --keep-models all"
        )
    return {
        "mode": config.keep_models,
        "kept_iteration": kept,
        "removed": removed,
        "freed_bytes": freed,
    }


def _write_artifacts(
    state: AutoMLState,
    config: RunConfig,
    text: str,
    history: list[dict[str, Any]],
) -> None:
    # try 앞이고 summary 리터럴 안이 아닌 이유: ``report.md`` 쓰기 실패가 디스크 회수 여부를 정해서는
    # 안 되고, 무엇을 지웠다는 기록은 삭제가 먼저 일어났을 때만 믿을 수 있다.
    models = prune_models(state, config)
    try:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        (config.run_dir / "report.md").write_text(text, encoding="utf-8")
        summary = {
            "thread_id": config.thread_id,
            "goal": state.get("goal"),
            "iterations": state.get("iteration"),
            "stop_reason": stop_reason(state, config),
            # Critic이 몇 번의 시도를 진단했는지. 문장이 아니라 수인 이유는 이 파일을 읽어 실행을
            # 집계하는 것들 때문이고, 그것이 빠져 있던 칸이었다: 다섯 데이터셋
            # 비교는 모든 시도의 ``critic``에서 다시 유도하지 않고는 그중 넷이 루프에 들어간 적조차
            # 없다는 것을 말할 수 없었다.
            "critic_runs": sum(1 for item in history if item.get("critic")),
            # 실행이 받은 것에 대고 실제로 쓴 비용. 기록하는 이유는 ``--time-budget-sec``가 이제
            # 루프가 따르는 한계이고, 나중에 아무도 읽을 수 없는 한계는 무시된 한계와 구분되지
            # 않기 때문이다: 5회 중 2회에서 멈춘 실행은 시계가 그랬는지 정체 가드가 그랬는지
            # 말하려면 이것이 필요하다.
            "budget": state.get("budget"),
            "best": state.get("best"),
            "holdout": state.get("holdout"),
            # 이 실행이 아직 어느 모델 파일을 갖고 있는지. 이것 없이는 나중의 "iteration 3의 모델이
            # 없다"가 기록된 결정이 아니라 사고로 읽힌다.
            "models": models,
            "history": history,
        }
        (config.run_dir / "history.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as exc:
        print(f"  [report] 아티팩트 저장 실패: {exc}")
