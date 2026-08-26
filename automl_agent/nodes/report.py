"""Report agent: the final write-up, produced on every exit path.

A run that missed its target still gets a report, written from the ``best``
snapshot — the point of the loop is the evidence it accumulated, not just a pass.

This node appends the final iteration to ``history`` (the Critic never saw it) and
writes ``report.md`` plus ``history.json`` into the run's artifact directory.
"""

from __future__ import annotations

import json
from typing import Any

from ..capabilities import describe as describe_capabilities
from ..config import RunConfig, file_size_text
from ..dataset.caveats import describe_caveats
from ..llm.client import LLMClient, LLMUnavailable, archive_prompt_only, render_prompt
from ..scoring.goal import describe, goal_threshold
from ..scoring.intervals import contains, describe_interval, interval_of
from ..state import AutoMLState, build_attempt, goal_met
from .holdout import describe as describe_holdout
from .model_selection import _history_digest, digest_attempt, task_of_state

STOP_REASON_LABELS = {
    "goal_reached": "목표 지표 달성",
    "max_iterations": "최대 반복 횟수 도달",
    "stalled": "연속 미개선(정체)으로 조기 종료",
    "unknown": "알 수 없는 사유",
}


def report(state: AutoMLState, *, config: RunConfig) -> dict:
    """Return ``{"report": markdown, "history": [final attempt]}``."""
    reason = stop_reason(state, config)
    reached = goal_met(dict(state.get("result") or {}), dict(state.get("goal") or {}))
    final_attempt = build_attempt(state, None)
    # The final attempt is not in state["history"] yet, so add it for the write-up.
    full_history = [*_history_digest(state), digest_attempt(final_attempt)]

    variables = {
        "goal": state.get("goal") or {},
        "goal_met": "yes" if reached else "no",
        "iterations": str(state.get("iteration") or 0),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "stop_reason": f"{reason} ({STOP_REASON_LABELS.get(reason, reason)})",
        "best": state.get("best") or "(성공한 시도 없음)",
        "history": full_history,
        "dataset_card": state.get("dataset_card") or {},
        # What the card's aggregates cannot show, and what every recommendation in the
        # write-up has to survive — automl_agent.dataset.caveats.
        "caveats": describe_caveats(dict(state.get("dataset_card") or {})),
        # The one number no decision was selected against. Given to the writer as a
        # sentence rather than a dict so it cannot be mistaken for another attempt's
        # metrics and averaged in with them.
        "holdout": describe_holdout(dict(state.get("holdout") or {})),
        # So the write-up can separate "what the plan asked for" from "what ran" instead
        # of listing an unexecuted strategy under 최고 성능 구성.
        "executor_capabilities": describe_capabilities(task_of_state(state)),
    }

    text = ""
    if not config.use_llm:
        archive_prompt_only(config, "report", render_prompt("report", variables))
    else:
        try:
            text = LLMClient(config).complete_text("report", variables, max_tokens=8000)
        except (LLMUnavailable, KeyError, OSError) as exc:
            print(f"  [report] LLM 보고서 생성 실패({exc}) — 템플릿 보고서로 폴백합니다")

    if not text.strip():
        text = fallback_report(state, config, reason, reached, full_history)

    _write_artifacts(state, config, text, full_history)
    return {"report": text, "history": [final_attempt]}


def stop_reason(state: AutoMLState, config: RunConfig) -> str:
    """Recompute why the loop ended, in the same order ``route`` decided it."""
    if goal_met(dict(state.get("result") or {}), dict(state.get("goal") or {})):
        return "goal_reached"
    if int(state.get("iteration", 0) or 0) >= int(state.get("max_iterations", 0) or 0):
        return "max_iterations"
    if int(state.get("stall_count", 0) or 0) >= config.stall_limit:
        return "stalled"
    return "unknown"


# --------------------------------------------------------------------------- #
# Deterministic report (--dry-run and LLM-unavailable fallback)
# --------------------------------------------------------------------------- #


def fallback_report(
    state: AutoMLState,
    config: RunConfig,
    reason: str,
    reached: bool,
    history: list[dict[str, Any]],
) -> str:
    """Template report built purely from the accumulated state."""
    goal = dict(state.get("goal") or {})
    metric = str(goal.get("metric", config.metric))
    threshold = goal_threshold(goal, config.fallback_threshold)
    best = dict(state.get("best") or {})
    best_score = best.get("score")
    iterations = int(state.get("iteration", 0) or 0)
    max_iterations = int(state.get("max_iterations", config.max_iterations) or config.max_iterations)

    verdict_line = (
        f"목표 `{metric} >= {threshold}`를 **달성**했습니다."
        if reached
        else f"목표 `{metric} >= {threshold}`를 **달성하지 못했습니다**."
    )
    # The interval of the score being quoted, where the run measured one. Written into the
    # summary sentence rather than only the configuration block because this is the sentence
    # a reader quotes elsewhere, and a bare 0.7503 travels as if it were exact.
    bounds = interval_of(dict(best.get("metrics") or {}), metric)
    best_line = (
        f"최고 성능은 iteration {best.get('iteration')}의 `{best.get('model')}`이며 "
        f"{describe_interval(metric, best_score, bounds)}입니다."
        if isinstance(best_score, (int, float))
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
        f"종료 사유는 **{STOP_REASON_LABELS.get(reason, reason)}**입니다.",
        "",
        # Where the bar came from. A report that states "0.872 달성" without this reads
        # as though the number were a universal standard rather than this dataset's.
        f"목표 기준: {describe(goal)}",
        "",
        # Immediately under the summary, because every number in the table below it was
        # selected on and this one was not.
        f"최종 검증: {describe_holdout(dict(state.get('holdout') or {}))}",
        "",
        "## 시도별 경과",
        "",
        "| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for attempt in history:
        metrics = attempt.get("metrics") or {}
        score = metrics.get(metric)
        if attempt.get("status") == "ok" and isinstance(score, (int, float)):
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
            f"- {metric}: {best_score:.4f}" if isinstance(best_score, (int, float)) else f"- {metric}: 없음",
            f"- 전체 지표: {_format_hyperparams(best.get('metrics') or {})}",
            f"- 학습 시간: {best.get('train_time_sec')}초",
            f"- 재현: `seed={config.seed}`, iteration {best.get('iteration')}",
        ]
    else:
        lines.append("유효한 최고 성능 구성이 없습니다.")

    lines += ["", "## 원인 분석", ""]
    diagnoses = [
        (attempt.get("critic") or {}).get("failure_type")
        for attempt in history
        if attempt.get("critic")
    ]
    if diagnoses:
        counts: dict[str, int] = {}
        for name in diagnoses:
            counts[str(name)] = counts.get(str(name), 0) + 1
        dominant = max(counts, key=lambda key: counts[key])
        area = "자원 제약" if dominant in {"oom", "too_slow"} else "모델/데이터 적합도"
        lines.append(
            f"Critic 진단 분포: {counts}. 가장 빈번한 원인은 `{dominant}`이며, "
            f"{area} 쪽 문제로 수렴했습니다."
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
        lines.append("Critic이 개입하기 전에 종료되어 축적된 진단이 없습니다.")

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
            "동일 계열 안에서의 미세 조정이 정체됐으므로 특성 공학 또는 데이터 품질 개선을 "
            "먼저 시도한다."
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
    """Delete the fitted models nothing can still be pointed at, and say which.

    Runs here because this node is the run's last one and ``holdout`` — the only consumer that
    reads a model file inside the graph — has already scored the winner by now. What is left
    behind is the model ``predict`` resolves to, so the run stays applicable; what goes is the
    losing attempts', which no command reaches without ``--iteration``.

    Never fatal, and never silent. A deletion that cannot be explained afterwards is worse
    than the disk it saves, so the return value names the iterations and the bytes and the
    flag that would have kept them, and it is written into ``history.json`` as well as printed.
    """
    freed = 0
    removed: list[int] = []
    kept = dict(state.get("best") or {}).get("iteration")
    if config.keep_models == "all" or not isinstance(kept, int):
        # No best iteration means no successful fit, so there is no winner to keep and no
        # basis for calling the others losers — that run's files are all the evidence it has.
        return {"mode": config.keep_models, "kept_iteration": kept, "removed": [], "freed_bytes": 0}
    for iteration in range(1, int(state.get("iteration", 0) or 0) + 1):
        if iteration == kept:
            continue
        path = config.model_path(iteration)
        try:
            size = path.stat().st_size
        except OSError:
            continue  # never fitted, already gone, or unreadable — nothing to reclaim
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
    # Before the try, and not inside the summary literal: a failed ``report.md`` write must
    # not decide whether the disk gets reclaimed, and the record of what was deleted is only
    # trustworthy if the deletion happened first.
    models = prune_models(state, config)
    try:
        config.run_dir.mkdir(parents=True, exist_ok=True)
        (config.run_dir / "report.md").write_text(text, encoding="utf-8")
        summary = {
            "thread_id": config.thread_id,
            "goal": state.get("goal"),
            "iterations": state.get("iteration"),
            "stop_reason": stop_reason(state, config),
            "best": state.get("best"),
            "holdout": state.get("holdout"),
            # Which model files this run still has. Without it, "iteration 3의 모델이 없다"
            # later reads as an accident rather than a recorded decision.
            "models": models,
            "history": history,
        }
        (config.run_dir / "history.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as exc:
        print(f"  [report] 아티팩트 저장 실패: {exc}")
