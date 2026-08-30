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

# The write-up is the run's only long-form output and is generated exactly once, so it gets a
# larger allowance than the reasoning nodes' structured replies. It used to be handed the
# 8000 that was also ``DEFAULT_LLM_MAX_TOKENS``, which ``test-1`` spent in full — adaptive
# thinking and the prose draw on the same allowance, so the report was competing with the
# model's own reasoning for room, and lost silently. Raised rather than removed: an unbounded
# request cannot fail loudly either.
REPORT_MAX_TOKENS = 24000

# Appended when the model was cut off. In the report itself and not only on the console,
# because ``report.md`` is the artifact that gets read later and forwarded on, and a write-up
# that stops mid-sentence otherwise reads as a finished conclusion that simply omitted things.
TRUNCATION_NOTE = (
    "> **주의 — 이 보고서는 완결되지 않았습니다.** 생성 중 출력 상한"
    f"({REPORT_MAX_TOKENS} 토큰)에 닿아 마지막 문장이 끊겼고, 뒤에 올 절이 빠져 있을 수 "
    "있습니다. 위에 적힌 내용 자체는 유효하지만 **결론으로 읽지 마십시오.** 모든 시도의 지표는 "
    "`history.json`에, 프롬프트와 응답 원본은 `llm/` 아래에 그대로 남아 있습니다."
)


def report(state: AutoMLState, *, config: RunConfig) -> dict:
    """Return ``{"report": markdown, "history": [final attempt]}``."""
    reason = stop_reason(state, config)
    reached = run_met_goal(state)
    final_attempt = build_attempt(state, None)
    # The final attempt is not in state["history"] yet, so add it for the write-up.
    full_history = [*_history_digest(state), digest_attempt(final_attempt)]

    variables = {
        "goal": state.get("goal") or {},
        "goal_met": "yes" if reached else "no",
        "iterations": str(state.get("iteration") or 0),
        "max_iterations": str(state.get("max_iterations") or config.max_iterations),
        "stop_reason": f"{reason} ({STOP_REASON_LABELS.get(reason, reason)})",
        # How many of the attempts below were diagnosed. Handed over as a sentence because the
        # report prompt asks for "the pattern across the Critic's verdicts", and with zero
        # verdicts that instruction invites a pattern to be invented — see describe_replanning.
        "replanning": describe_replanning(full_history),
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
    truncated = False
    if not config.use_llm:
        archive_prompt_only(config, "report", render_prompt("report", variables))
    else:
        try:
            text, truncated = LLMClient(config).complete_text(
                "report",
                variables,
                # ``max()`` so a config that deliberately raised the ceiling is not lowered
                # here — the floor is this node's, the choice stays the caller's.
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
        # Includes the case where the allowance ran out during thinking and no prose came back
        # at all. The template report is complete on its own, so it needs no truncation note —
        # the console line above is what says the long-form write-up was lost.
        text = fallback_report(state, config, reason, reached, full_history)
    elif truncated:
        text = f"{text.rstrip()}\n\n{TRUNCATION_NOTE}\n"

    _write_artifacts(state, config, text, full_history)
    return {"report": text, "history": [final_attempt]}


def describe_replanning(history: list[dict[str, Any]]) -> str:
    """How much of the loop actually looped, as one sentence.

    ``goal_reached`` is checked before anything else in :func:`automl_agent.graph.route`, and the
    bar in ``auto`` mode is derived from the baseline — so a first attempt that clears it ends the
    run at iteration 1 and the Critic never runs at all. Four of the five datasets in
    ``bench/RESULTS.md`` ended that way. Nothing said so: the report's stop reason read
    "목표 지표 달성", the attempt table's ``critic 진단`` column read "—", and a reader comparing
    the LLM arm against the rule-based arm had no way to see that the whole diagnose-and-replan
    path — the thing the comparison was about — had not executed on either side.

    So the count is stated rather than left to be inferred from an empty column, and it is stated
    in both directions: zero verdicts is evidence about the *planner*, and it is evidence for
    neither side about the loop.

    ``critic`` is absent from the last attempt by construction — the loop stops after evaluating
    it, so its verdict would only have fed a replan that never happens — which is why the
    sentence names that attempt instead of leaving a reader to explain the missing row.
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
    """Whether this run produced a model that clears the bar — not whether its *last* one did.

    Judged on ``best`` as well as on the final result, because ``best`` is the run's answer: it
    is the model ``holdout`` scored and the one ``predict`` resolves to. Under the default
    routing the two questions have the same answer, since the loop stops the instant an attempt
    clears the bar and ``best`` is then that attempt. Under
    :attr:`automl_agent.config.RunConfig.search_past_goal` they come apart — the run keeps going,
    and a later attempt that scores worse would otherwise turn "달성" into "미달성" while the
    winning model sits unchanged in ``best``.

    Either shape satisfies it, and ``goal_met`` refuses a missing or non-numeric score, so an
    empty ``best`` (no successful fit) answers False rather than raising.
    """
    goal = dict(state.get("goal") or {})
    return goal_met(dict(state.get("result") or {}), goal) or goal_met(
        dict(state.get("best") or {}), goal
    )


def stop_reason(state: AutoMLState, config: RunConfig) -> str:
    """Recompute why the loop ended, in the same order ``route`` decided it.

    The ``search_past_goal`` guard mirrors ``route``'s, and it has to: under that flag clearing
    the bar is not a stop condition at all, so ``goal_reached`` would name a reason the loop did
    not act on. Such a run stops on iterations or on the stall guard, and *also* met its goal —
    which is what ``goal_met`` in the write-up says, separately from this.
    """
    if not config.search_past_goal and goal_met(
        dict(state.get("result") or {}), dict(state.get("goal") or {})
    ):
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
        # In the summary and not only in 원인 분석, because "목표 지표 달성" at iteration 1 is
        # read as the loop having worked — and at iteration 1 the loop did not run.
        f"재계획: {describe_replanning(history)}",
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
        # What the absence *means*, since the summary above already gives the count. Without
        # this the section reads as a gap in the write-up rather than as a limit on what the
        # run can be cited for.
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
            # How many attempts the Critic diagnosed. A count rather than the sentence, because
            # this file is read by ``bench/`` and by anything else aggregating runs — and it was
            # the missing column: the five-dataset comparison could not tell that four of its
            # runs never entered the loop without re-deriving it from every attempt's ``critic``.
            "critic_runs": sum(1 for item in history if item.get("critic")),
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
