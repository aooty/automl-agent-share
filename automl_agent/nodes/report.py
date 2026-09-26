"""Report node: writes the final report of every run, even one that missed the goal.

Roles:

* Report node: prompt the LLM and record the last attempt.
* Template report: state-only report for --dry-run or LLM failure.
* Artifacts: prune losing models, write report and history.
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

# Report label for each stop reason.
STOP_REASON_LABELS = {
    "goal_reached": "목표 지표 달성",
    "max_iterations": "최대 반복 횟수 도달",
    "stalled": "연속 미개선(정체)으로 조기 종료",
    # Only when the clock ends the loop (see ``stop_condition``).
    "out_of_time": "시간 예산 소진으로 조기 종료",
    "unknown": "알 수 없는 사유",
}

# Thinking and text share this limit
REPORT_MAX_TOKENS = 24000

# Written into report.md itself when the output was cut.
TRUNCATION_NOTE = (
    "> **주의 — 이 보고서는 완결되지 않았습니다.** 생성 중 출력 상한"
    f"({REPORT_MAX_TOKENS} 토큰)에 닿아 마지막 문장이 끊겼고, 뒤에 올 절이 빠져 있을 수 "
    "있습니다. 위에 적힌 내용 자체는 유효하지만 **결론으로 읽지 마십시오.** 모든 시도의 지표는 "
    "`history.json`에, 프롬프트와 응답 원본은 `llm/` 아래에 그대로 남아 있습니다."
)


# --- Role: report node ------------------------------------------------------------


def report(state: AutoMLState, *, config: RunConfig) -> dict:
    """Write the final report and record the last attempt.

    Uses :func:`fallback_report` when the LLM is off, fails, or returns nothing.
    """
    reason = stop_reason(state, config)
    reached = run_met_goal(state)
    final_attempt = build_attempt(state, None)
    # The last attempt is not in history yet.
    full_history = [*_history_digest(state), digest_attempt(final_attempt)]

    variables = {
        "goal": state.get("goal") or {},
        "goal_met": "yes" if reached else "no",
        "iterations": str(state.get("iteration") or 0),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "stop_reason": f"{reason} ({STOP_REASON_LABELS.get(reason, reason)})",
        # Stated, so the model does not invent Critic patterns.
        "replanning": describe_replanning(full_history),
        "best": state.get("best") or "(성공한 시도 없음)",
        "history": full_history,
        "dataset_card": state.get("dataset_card") or {},
        # Data warnings the card numbers cannot show.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # A sentence, so it is not mixed with attempts.
        "holdout": describe_holdout(dict(state.get("holdout") or {})),
        # Tells what the plan asked from what ran.
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
                # max() keeps a limit the user raised on purpose.
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
        # Also when thinking used all tokens; no cut note.
        text = fallback_report(state, config, reason, reached, full_history)
    elif truncated:
        text = f"{text.rstrip()}\n\n{TRUNCATION_NOTE}\n"

    _write_artifacts(state, config, text, full_history)
    return {"report": text, "history": [final_attempt]}


def describe_replanning(history: list[dict[str, Any]]) -> str:
    """Say in one Korean sentence how often the Critic ran and the loop re-planned.

    Why this is spelled out:
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
    """True if the last result or ``best`` met the goal"""
    goal = dict(state.get("goal") or {})
    return goal_met(dict(state.get("result") or {}), goal) or goal_met(dict(state.get("best") or {}), goal)


def stop_reason(state: AutoMLState, config: RunConfig) -> str:
    """Name why the loop ended, with the same check ``route`` uses.

    Returns ``unknown`` for a state that never went through routing.
    """
    return stop_condition(state, config.stall_limit, config.search_past_goal) or "unknown"


# --- Role: template report (--dry-run, or when the LLM cannot be used) ------------


def fallback_report(
    state: AutoMLState,
    config: RunConfig,
    reason: str,
    reached: bool,
    history: list[dict[str, Any]],
) -> str:
    """Build a markdown report from the saved state alone, with no LLM."""
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
    # Interval in the summary: a bare score looks exact.
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
        # Says why unused iterations were not used.
        f"시간 예산 사용: {describe_budget(dict(state.get('budget') or {}))}.",
        "",
        # A goal at iteration 1 means the loop never ran.
        f"재계획: {describe_replanning(history)}",
        "",
        # Where the goal came from; not a universal standard.
        f"목표 기준: {describe(goal)}",
        "",
        # The only score not used to choose a model.
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
        # Say what no diagnoses means for the result.
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
    """_next_steps | Template report: list next-step suggestions for the report."""
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
    """_format_hyperparams | Template report: render a dict as sorted `key`=value pairs."""
    if not hyperparams:
        return "—"
    parts = []
    for key, value in sorted(hyperparams.items()):
        parts.append(f"`{key}`={round(value, 4) if isinstance(value, float) else value}")
    return ", ".join(parts)


# --- Role: artifacts --------------------------------------------------------------


def prune_models(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Delete losing attempts' model files and return what was removed.

    Keeps the best model; never fails the run
    """
    freed = 0
    removed: list[int] = []
    kept = dict(state.get("best") or {}).get("iteration")
    if config.keep_models == "all" or not isinstance(kept, int):
        # No winner means no losers: keep every file.
        return {"mode": config.keep_models, "kept_iteration": kept, "removed": [], "freed_bytes": 0}
    for iteration in range(1, state_int(state, "iteration") + 1):
        if iteration == kept:
            continue
        path = config.model_path(iteration)
        try:
            size = path.stat().st_size
        except OSError:
            continue  # never fitted, already gone, or unreadable: nothing to free
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
    """_write_artifacts | Artifacts: prune models, then write report.md and history.json."""
    # Prune first, so a failed write cannot skip it.
    models = prune_models(state, config)
    try:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        (config.run_dir / "report.md").write_text(text, encoding="utf-8")
        summary = {
            "thread_id": config.thread_id,
            "goal": state.get("goal"),
            "iterations": state.get("iteration"),
            "stop_reason": stop_reason(state, config),
            # A number for tools like ``bench/``
            "critic_runs": sum(1 for item in history if item.get("critic")),
            # Shows if the clock stopped the run
            "budget": state.get("budget"),
            "best": state.get("best"),
            "holdout": state.get("holdout"),
            # So a missing model file reads as a choice.
            "models": models,
            "history": history,
        }
        (config.run_dir / "history.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as exc:
        print(f"  [report] 아티팩트 저장 실패: {exc}")
