"""CLI entry point: ``profile`` / ``run`` / ``resume`` / ``show`` / ``predict`` / ``graph``.

All console output is Korean; identifiers, metric names and error types stay in
English so they match the artifacts and the code.

``profile`` is the card-building step run on its own, for when a human wants to read
(and edit) the card before spending a run on it. ``run --data`` does the same work
inside the graph, via the ``profiling`` node. Either way the split is the same: the
path goes into the private ``data_ref`` channel, the aggregates into
``dataset_card``.

``predict`` is the step after the loop: the model the run selected, applied to rows nobody
has labels for. It is a command rather than a node because it is not part of choosing a
model — and because it is the only path here whose *output* is per-row, so it never enters
the graph's state at all.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import (
    API_KEY_ENV,
    ARTIFACTS_ROOT,
    BEDROCK_FLAG_ENV,
    DEFAULT_KEEP_MODELS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_METRIC,
    DEFAULT_TIME_BUDGET_SEC,
    DIRECTIONS,
    DRY_RUN_SCENARIOS,
    KEEP_MODELS_MODES,
    RunConfig,
    bedrock_signing_available,
    has_llm_credentials,
    use_bedrock,
)
from .dataset.caveats import CAVEATS_KEY, card_caveats, merge_caveats
from .dataset.targets import DEFAULT_TARGET_MISSING_POLICY, TARGET_MISSING_POLICIES
from .graph import build_graph, build_state_graph, make_checkpointer

# Costs nothing extra: ``.graph`` above already imports every node module.
from .nodes.holdout import describe as describe_holdout
from .nodes.profiling import ProfilingFailed, assert_goal_is_usable
from .privacy import CardSchemaError, data_ref, public_card, register_private, validate_card
from .scoring.goal import (
    DEFAULT_MARGIN,
    DEFAULT_MODE,
    GOAL_MODES,
    MODE_AUTO,
    MODE_FIXED,
    describe,
    resolve_goal,
)
from .scoring.metrics import GOAL_METRICS
from .state import describe_budget
from .threads import thread_state

RUN_CONFIG_FILE = "run_config.json"


# --------------------------------------------------------------------------- #
# Console reporting
# --------------------------------------------------------------------------- #


class ConsoleReporter:
    """Prints one summary line per iteration, completed by the critic's verdict."""

    def __init__(self, goal: dict[str, Any]) -> None:
        self.goal = goal
        self._model = ""
        self._pending: str | None = None

    def on_update(self, node: str, update: dict[str, Any]) -> None:
        if node == "profiling":
            # The profiler measures the reference baseline, so the bar it reports
            # supersedes whatever could be derived before the card existed.
            goal = update.get("goal")
            if isinstance(goal, dict) and goal:
                self.goal = goal
        elif node == "planning":
            plan = update.get("plan") or {}
            strategy = str(plan.get("strategy") or "")
            print(f"[iter {update.get('iteration')}] 계획: {strategy}")
        elif node == "model_selection":
            self._model = str(update.get("model") or "")
        elif node == "evaluate":
            self._pending = self._summary_line(update.get("evaluation") or {})
        elif node == "critic":
            verdict = update.get("critic") or {}
            self._flush(f" → critic: {verdict.get('failure_type')}")
            direction = str(verdict.get("direction") or "").strip()
            if direction:
                print(f"          방향: {direction}")
        elif node == "holdout":
            # Flush first: the pending line is the iteration this model came from, and the
            # final measurement only makes sense read after it.
            self._flush("")
            print(f"  [holdout] {describe_holdout(dict(update.get('holdout') or {}))}")
        elif node == "report":
            self._flush("")
            print("보고서 작성 완료")

    def _summary_line(self, evaluation: dict[str, Any]) -> str:
        metric = str(evaluation.get("metric") or self.goal.get("metric", "metric"))
        threshold = self.goal.get("threshold")
        iteration = evaluation.get("iteration")
        if evaluation.get("status") == "ok" and isinstance(evaluation.get("score"), (int, float)):
            outcome = f"{metric}={float(evaluation['score']):.4f}"
        else:
            outcome = f"실패({evaluation.get('error_type') or 'unknown'})"
        marker = " [목표 달성]" if evaluation.get("goal_met") else ""
        return f"[iter {iteration}] model={self._model} {outcome} (goal {threshold}){marker}"

    def _flush(self, suffix: str) -> None:
        if self._pending is not None:
            print(self._pending + suffix)
            self._pending = None

    def finish(self) -> None:
        self._flush("")


# --------------------------------------------------------------------------- #
# Config persistence (so ``resume`` needs only --thread-id)
# --------------------------------------------------------------------------- #


def save_run_config(config: RunConfig) -> None:
    config.ensure_dirs()
    payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in dataclasses.asdict(config).items()
    }
    # Not a ``RunConfig`` field, and written here anyway: it is the environment the run
    # happened in rather than a setting the run was given, and ``load_run_config`` keeps only
    # declared fields, so a resumed run never reads it back as one. It is here because this is
    # the one file per run directory that says what the run *was*, and thread count moves the
    # scores in it by more than some of the differences they are used to argue
    # (:mod:`automl_agent.threads`). A resume overwrites this file, so it describes the shell
    # of the most recent leg — the per-attempt copies in ``result.json`` are the ones that stay
    # matched to their own numbers.
    payload["threads"] = thread_state()
    path = config.run_dir / RUN_CONFIG_FILE
    # Written to a sibling and renamed, because ``os.replace`` is atomic on both platforms:
    # every command except ``run`` rebuilds its settings from this file, so a write cut off
    # halfway — Ctrl-C, a full disk — used to leave truncated JSON, and truncated JSON is
    # the one state where the run is intact on disk but nothing can read it back. The
    # rename either happened or did not; there is no half-file to load.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_run_config(thread_id: str, artifacts_root: Path | None = None) -> RunConfig:
    """Rebuild the config of a finished or interrupted run from its own directory.

    ``artifacts_root`` is re-anchored to where this file was actually found, and the saved
    value is discarded. It is an absolute path written on the machine that ran the run, and
    every command that loads a config then derives the things it needs from it — the
    checkpoint database, ``history.json``, ``model.joblib``. Copy the run directory to
    another machine (or just move the repository) and the saved root points at nothing, so
    the derived paths do too, and what the operator gets is not an error about a moved
    directory: ``show`` and ``resume`` report "저장된 체크포인트가 없습니다" and ``predict``
    reports "선택된 최고 시도가 없습니다" — three messages that describe a run that failed,
    for a run that succeeded. The file is *in* the root, so the root is a fact this function
    already has and does not need to be told.

    ``data_path`` and ``dataset_card_path`` are left exactly as saved: they point outside
    ``artifacts/``, so there is nothing here to re-anchor them against, and inventing a
    location for the raw data would be worse than a clear failure. ``command_resume``
    checks the one that a resumed run actually reads.
    """
    base = artifacts_root or ARTIFACTS_ROOT
    path = base / thread_id / RUN_CONFIG_FILE
    if not path.exists():
        raise SystemExit(
            f"오류: thread_id '{thread_id}'의 실행 설정을 찾을 수 없습니다 ({path}).\n"
            f"먼저 `python -m automl_agent.main run --thread-id {thread_id} ...`를 실행하세요."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"오류: 실행 설정을 읽을 수 없습니다 ({path}) — {exc}") from exc
    except json.JSONDecodeError as exc:
        # ``save_run_config`` renames into place, so this is a file from before that change,
        # a hand edit, or a damaged disk. The traceback it used to raise pointed at json's
        # internals; what the operator needs is the path and the fact that the rest of the
        # run directory is untouched.
        raise SystemExit(
            f"오류: 실행 설정이 올바른 JSON이 아닙니다 ({path}) — {exc}\n"
            "  쓰는 중에 중단된 파일일 수 있습니다. 같은 --thread-id로 `run`을 다시 실행하면 "
            "이 파일이 새로 쓰이고, 체크포인트가 남아 있으면 그 지점부터 이어집니다 — "
            f"artifacts/{thread_id}의 나머지 파일은 그대로입니다"
        ) from exc
    if not isinstance(raw, dict):
        raise SystemExit(
            f"오류: 실행 설정이 JSON 객체가 아닙니다 ({path}, 최상위가 "
            f"{type(raw).__name__}입니다)"
        )
    fields = {field.name for field in dataclasses.fields(RunConfig)}
    kwargs: dict[str, Any] = {key: value for key, value in raw.items() if key in fields}
    for key in ("dataset_card_path", "data_path"):
        kwargs[key] = Path(str(kwargs[key])) if kwargs.get(key) else None
    kwargs["artifacts_root"] = base
    return RunConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def load_dataset_card(path: Path) -> dict[str, Any]:
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"오류: 데이터셋 카드를 읽을 수 없습니다 — {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"오류: 데이터셋 카드가 올바른 JSON이 아닙니다 — {exc}") from exc
    try:
        # Before anything else touches it: a card that cannot be published stops the run
        # here, where the message is about the file the user just named.
        return validate_card(card)
    except CardSchemaError as exc:
        raise SystemExit(f"{exc}\n  - 카드 경로: {path}") from exc


def config_goal(
    config: RunConfig, card: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str | None]:
    """The ``goal`` channel for this config and card, and the substitution note if there is one.

    Same call the profiling node makes, including the metric substitution — which is why it
    returns a pair. A caller that has a card in hand can find out here that the run will be
    judged by a different metric than the flags asked for, and the display-only callers pass
    no card, so for them the note is always ``None``.
    """
    return resolve_goal(
        card or {},
        metric=config.metric,
        direction=config.direction,
        mode=config.goal_mode,
        threshold=config.threshold,
        margin=config.goal_margin,
    )


def initial_state(card: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    """Seed the blackboard, splitting the card into its public and private halves.

    This is where the boundary is drawn for the whole run: ``public_card`` drops the
    ``data`` block (and any example rows a hand-written card smuggled in), while
    ``data_ref`` keeps the path in a channel only the execution nodes read. The path is
    also registered with the prompt guard, so a regression that puts it back into the
    card stops the run instead of shipping it to the API.
    """
    reference = data_ref(card, config.data_path, config.target_column, config.group_column)
    if reference.get("path"):
        register_private(reference["path"])
    return {
        "dataset_card": public_card(card),
        "data_ref": reference,
        # Derived from whatever card we have now; on the ``--data`` path there is no
        # card yet, so this is the per-metric fallback and the profiling node
        # overwrites it with a baseline-derived bar on the first tick.
        "goal": config_goal(config, card)[0],
        "plan": {},
        "model": "",
        "hyperparams": {},
        "result": {},
        "critic": {},
        "history": [],
        "iteration": 0,
        "max_iterations": config.max_iterations,
        "stall_count": 0,
        "best": {},
        "report": "",
        "evaluation": {},
        "holdout": {},
    }


def check_credentials(config: RunConfig) -> None:
    """Fail before spending time when a live run cannot possibly call the LLM."""
    if not config.use_llm:
        return
    if use_bedrock() and not bedrock_signing_available():
        # The SDK imports botocore lazily, when it signs the first request — without
        # this check the failure surfaces as a traceback from inside the planning node.
        raise SystemExit(
            "오류: Bedrock 경로는 요청 서명(SigV4)에 botocore가 필요하지만 설치되어 있지 않습니다.\n"
            '  - 설치: python -m pip install "anthropic[bedrock]"\n'
            f"  - 또는 {BEDROCK_FLAG_ENV}를 해제하고 {API_KEY_ENV}로 직접 호출하세요.\n"
            "  - 자격 증명 없이 돌리려면 --no-llm 또는 --dry-run 을 사용하세요."
        )
    if has_llm_credentials():
        return
    route = "Bedrock" if use_bedrock() else "Anthropic API"
    raise SystemExit(
        f"오류: {route} 자격 증명이 없어 LLM을 호출할 수 없습니다.\n"
        f"  - 직접 호출: {API_KEY_ENV} 환경변수를 설정하세요.\n"
        "  - Bedrock 경로: AUTOML_USE_BEDROCK=1 과 AWS_REGION 을 설정하세요.\n"
        "  - 실제 학습은 하되 추론만 규칙 기반으로 돌리려면 --no-llm 을 사용하세요.\n"
        "  - LLM과 학습을 모두 모킹하려면 --dry-run 을 사용하세요."
    )


def open_app(config: RunConfig) -> tuple[Any, BaseCheckpointSaver, dict[str, Any]]:
    """Compile the graph over this run's checkpoint file and build its runtime config."""
    saver = make_checkpointer(config.checkpoint_db)
    app = build_graph(config, checkpointer=saver)
    runtime = {
        "configurable": {"thread_id": config.thread_id},
        # Ample headroom for max_iterations * 4 nodes plus the report.
        "recursion_limit": config.max_iterations * 6 + 12,
    }
    return app, saver, runtime


def run_goal(
    app: Any,
    runtime: dict[str, Any],
    config: RunConfig,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The goal the run is working against: seeded state, else checkpoint, else config.

    Only for display. The nodes read ``state["goal"]`` directly, and on the ``--data``
    path the profiling node replaces it once the baseline has been measured.
    """
    goal = dict((payload or {}).get("goal") or {})
    if not goal:
        snapshot = app.get_state(runtime)
        values = dict(snapshot.values) if snapshot and snapshot.values else {}
        goal = dict(values.get("goal") or {})
    return goal or config_goal(config)[0]


def stream_graph(
    app: Any,
    runtime: dict[str, Any],
    config: RunConfig,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run (or resume) the graph, printing progress as nodes report."""
    reporter = ConsoleReporter(run_goal(app, runtime, config, payload))
    try:
        for chunk in app.stream(payload, config=runtime, stream_mode="updates"):
            for node, update in chunk.items():
                if isinstance(update, dict):
                    reporter.on_update(node, update)
    except ProfilingFailed as exc:
        # A setup error, not an experimental outcome (see nodes/profiling.py): the operator
        # has to change something before any run of this data can work. Printed as a
        # message rather than a traceback, because the message *is* the fix.
        reporter.finish()
        raise SystemExit(f"오류: {exc}") from exc
    reporter.finish()
    snapshot = app.get_state(runtime)
    return dict(snapshot.values) if snapshot and snapshot.values else {}


def print_outcome(state: dict[str, Any], config: RunConfig) -> None:
    from .nodes.report import STOP_REASON_LABELS, stop_reason

    best = dict(state.get("best") or {})
    reason = stop_reason(state, config)  # type: ignore[arg-type]
    print("")
    print("=" * 70)
    print(f"종료 사유: {STOP_REASON_LABELS.get(reason, reason)}")
    score = best.get("score")
    if isinstance(score, (int, float)):
        print(
            f"최고 성능: {best.get('metric')}={score:.4f} "
            f"(iteration {best.get('iteration')}, model={best.get('model')})"
        )
    else:
        print("최고 성능: 성공한 시도가 없습니다.")
    # Printed right under the best validation score, because that is the number a reader
    # would otherwise walk away with, and it was selected on.
    print(describe_holdout(dict(state.get("holdout") or {})))
    goal = dict(state.get("goal") or {}) or config_goal(config)[0]
    print(f"목표: {describe(goal)}")
    print(f"총 반복: {state.get('iteration')} / {state.get('max_iterations')}")
    # Next to the iteration count, because they are the same kind of fact and one of them used
    # to be unenforced: --time-budget-sec was a per-fit timeout, so the seconds a run spent were
    # nowhere on this screen and nowhere in its artifacts.
    print(f"시간 예산: {describe_budget(dict(state.get('budget') or {}))}")
    print(f"아티팩트: {config.run_dir}")
    report_path = config.run_dir / "report.md"
    if report_path.exists():
        print(f"보고서: {report_path}")
    print("=" * 70)


def command_profile(args: argparse.Namespace) -> int:
    """Build a dataset card from raw data, as a standalone step.

    Runs the same fixed script the ``profiling`` node runs, in the same kind of
    subprocess. The point of exposing it separately is review: a human can read the
    card, correct a mislabelled column, and only then spend a run on it.
    """
    from .scripts.profile import main as profile_main

    out_path = Path(args.out)
    code = profile_main(
        [
            "--data",
            str(args.data),
            "--target",
            str(args.target),
            "--out",
            str(out_path),
            "--seed",
            str(args.seed),
            "--on-missing-target",
            str(args.on_missing_target),
            *(["--name", args.name] if args.name else []),
            *[item for note in (args.caveats or []) for item in ("--caveat", str(note))],
            *(["--group-column", str(args.group_column)] if args.group_column else []),
            *(["--no-baseline"] if args.no_baseline else []),
        ]
    )
    if code != 0:
        raise SystemExit("오류: 데이터셋 카드 생성에 실패했습니다 (위 stderr 참고).")
    print("")
    print(f"카드를 저장했습니다: {out_path}")
    # Both bars this card implies, before a run is spent on it — the point of the
    # standalone command is choosing. Same derivation the profiling node performs.
    card = load_dataset_card(out_path)
    print(f"이 카드로 {args.metric} 를 목표로 실행하면:")
    for mode in GOAL_MODES:
        # ``resolve_goal``, not ``derive_goal``: the point of this preview is to show the bar
        # a real run would be judged against, and a run against this card would substitute
        # the metric if it belongs to the other task. Previewing ``rmse`` bars for a
        # classification card would preview a run that cannot happen.
        goal, substitution = resolve_goal(card, metric=args.metric, mode=mode, margin=args.margin)
        print(f"  {describe(goal)}")
        if substitution and mode == GOAL_MODES[0]:
            # Once, not per mode: the substitution is a property of the card and the metric,
            # and the same sentence under both bars would read as two different findings.
            print(f"  경고: {substitution}")
    print("이 카드의 'data' 블록만 원본 경로를 담고 있고, 실행 시 프롬프트에서 제외됩니다.")
    print(f"실행: python -m automl_agent.main run --dataset-card {out_path} --thread-id <id> ...")
    return 0


def resolve_goal_mode(args: argparse.Namespace) -> str:
    """Settle the goal mode from the flags, refusing combinations that contradict.

    A silently ignored ``--threshold`` (or ``--margin``) is the worst outcome here: the
    run would report against a bar the caller did not ask for and never be told.
    """
    # ``--goal-mode`` defaults to None rather than "auto" so that "not specified" stays
    # distinguishable from "explicitly auto": naming a threshold and nothing else is
    # choosing fixed, but naming both is a contradiction worth stopping for.
    mode = args.goal_mode or (MODE_FIXED if args.threshold is not None else MODE_AUTO)
    if mode == MODE_AUTO and args.threshold is not None:
        raise SystemExit(
            "오류: auto 모드는 임계값을 카드의 기준선에서 스스로 도출하므로 "
            "--threshold 와 함께 쓸 수 없습니다.\n"
            "  - 값을 못박으려면: --goal-mode fixed --threshold 0.9\n"
            "  - 요구 수준만 조절하려면: --margin 0.5  (기준선에서 남은 여유의 50%)"
        )
    if mode == MODE_FIXED and args.margin is not None:
        raise SystemExit(
            "오류: --margin 은 기준선에서 목표를 도출하는 auto 모드에서만 의미가 있습니다.\n"
            "  - fixed 모드에서 값을 바꾸려면 --threshold 를 쓰세요."
        )
    return str(mode)


def command_run(args: argparse.Namespace) -> int:
    if not args.dataset_card and not args.data:
        raise SystemExit(
            "오류: --dataset-card 또는 --data 중 하나는 필요합니다.\n"
            "  - 카드가 이미 있으면: --dataset-card <card.json>\n"
            "  - 원본 CSV에서 시작하려면: --data <csv> --target <컬럼>  (profiling 노드가 카드를 만듭니다)"
        )
    if args.data and not args.target:
        raise SystemExit("오류: --data 를 쓸 때는 --target 으로 정답 컬럼을 지정해야 합니다.")
    goal_mode = resolve_goal_mode(args)

    config = RunConfig(
        thread_id=args.thread_id,
        metric=args.metric,
        goal_mode=goal_mode,
        threshold=args.threshold,
        goal_margin=DEFAULT_MARGIN if args.margin is None else args.margin,
        direction=args.direction,
        max_iterations=args.max_iterations,
        time_budget_sec=args.time_budget_sec,
        search_past_goal=args.search_past_goal,
        dry_run=args.dry_run,
        dry_run_scenario=args.scenario,
        no_llm=args.no_llm,
        seed=args.seed,
        llm_model=args.model,
        dataset_card_path=Path(args.dataset_card) if args.dataset_card else None,
        data_path=Path(args.data) if args.data else None,
        target_column=args.target,
        on_missing_target=args.on_missing_target,
        caveats=tuple(args.caveats or ()),
        group_column=args.group_column,
        artifacts_root=Path(args.artifacts_root) if args.artifacts_root else None,
        keep_models=args.keep_models,
    )
    check_credentials(config)
    # No card means the profiling node builds one from data_ref on the first tick.
    card = load_dataset_card(config.dataset_card_path) if config.dataset_card_path else {}
    if card and config.caveats:
        # Appended in memory only: the card file on disk is the profiler's artifact, and a
        # run that edited it would change what a later run of a different thread reads.
        # On the ``--data`` path the profiling node passes these to the script instead.
        card[CAVEATS_KEY] = merge_caveats(card_caveats(card), config.caveats)
    if card:
        # A metric that cannot score this card's target, or a bar that could not be derived
        # from it: with the card already in hand there is nothing left to measure, so the
        # answer will not change by starting. Refused before a checkpoint, an artifact
        # directory or a run_config.json exists. The ``--data`` path has no card yet, so
        # there the same check runs in the profiling node.
        goal, substitution = config_goal(config, card)
        if substitution:
            # Before the run header rather than inside it: the operator is being told the run
            # will be judged by a metric they did not name, and the moment to read that is
            # before waiting for an iteration. ``initial_state`` recomputes the same goal, so
            # this only prints — it does not decide anything.
            print(f"경고: {substitution}")
        try:
            assert_goal_is_usable(card, goal, config)
        except ProfilingFailed as exc:
            raise SystemExit(f"오류: {exc}") from exc

    app, saver, runtime = open_app(config)
    existing = app.get_state(runtime)
    if existing and existing.values:
        # Seeding a fresh state onto a used thread_id would *append* to the old
        # history through the reducer, silently mixing two runs together.
        if not args.force:
            raise SystemExit(
                f"오류: thread_id '{config.thread_id}'에 이미 체크포인트가 있습니다 "
                f"(iteration {existing.values.get('iteration')}).\n"
                "  - 이어서 실행: `resume --thread-id ...`\n"
                "  - 결과 확인: `show --thread-id ...`\n"
                "  - 같은 id로 처음부터 다시 실행: `--force`\n"
                "  - 두 실행을 모두 남기려면 다른 --thread-id 를 쓰세요."
            )
        print(f"--force: thread_id '{config.thread_id}'의 기존 체크포인트를 삭제하고 새로 시작합니다")
        saver.delete_thread(config.thread_id)

    # Written only once this run is actually going to happen. Saving before the
    # checkpoint check overwrote the *previous* run's run_config.json even when this
    # invocation was rejected — and a later ``resume`` then read the new settings against
    # the old checkpoint, reporting a goal and a metric the recorded iterations never used.
    save_run_config(config)

    if config.dry_run:
        mode = f"--dry-run (scenario={config.dry_run_scenario})"
    elif config.no_llm:
        mode = "실제 학습 + 규칙 기반 추론 (--no-llm)"
    else:
        mode = "실제 실행 (LLM + 실제 학습)"
    print(f"AutoML 실행 시작 — thread_id={config.thread_id}, {mode}")
    if card:
        print(f"데이터셋 카드: {config.dataset_card_path}")
    else:
        print(f"데이터셋 카드: profiling 노드가 {config.data_path} 에서 생성합니다")

    seed_state = initial_state(card, config)
    goal = dict(seed_state["goal"])
    print(f"목표: {describe(goal)}")
    if config.goal_mode == MODE_AUTO and goal["source"] == "fallback":
        # An auto-mode bar that had no baseline to derive from. Say which way it will go
        # instead of leaving a per-metric default looking like a measured number.
        if card:
            print(
                "  경고: 이 카드에는 기준선(baseline)이 없어 auto 모드가 데이터셋에 맞출 수 "
                "없습니다. `profile` 로 카드를 다시 만들거나 --goal-mode fixed 를 쓰세요."
            )
        else:
            print("  profiling이 기준선을 측정하면 다시 도출됩니다")
    print(f"최대 {config.max_iterations}회 반복")
    print("-" * 70)

    state = stream_graph(app, runtime, config, seed_state)
    print_outcome(state, config)
    return 0


def assert_resumable_data(reference: dict[str, Any], thread_id: str) -> None:
    """Refuse to resume a run whose data file is not where the checkpoint says it is.

    Checked here rather than left to the training subprocess, and only for a run that is going
    to train again. The path in the checkpoint is absolute and was written on the machine that
    started the run, so a copied artifacts directory resumes into a file that is not there —
    and the way *that* surfaces is one recorded attempt failure per remaining iteration, a
    critic reasoning about the failures as if they were about the models, and a report
    concluding the run could not reach its goal. Refusing costs one message.

    A run with no data reference at all is the synthetic path, which has nothing to check.
    """
    path = str(reference.get("path") or "")
    if not path or Path(path).exists():
        return
    raise SystemExit(
        f"오류: 이 실행이 학습에 쓰던 데이터 파일이 없습니다 ({path}).\n"
        "  체크포인트에 저장된 경로는 실행을 시작한 기계의 절대 경로입니다 — artifacts "
        "디렉터리만 옮겨 왔다면 데이터 파일도 같은 경로에 있어야 재개할 수 있습니다.\n"
        f"  - 결과만 보려면: `show --thread-id {thread_id}` (데이터 파일이 필요하지 않습니다)\n"
        f"  - 저장된 모델을 새 데이터에 쓰려면: `predict --thread-id {thread_id} --data <csv>`"
    )


def command_resume(args: argparse.Namespace) -> int:
    config = load_run_config(
        args.thread_id, Path(args.artifacts_root) if args.artifacts_root else None
    )
    check_credentials(config)
    app, _saver, runtime = open_app(config)
    snapshot = app.get_state(runtime)
    if not snapshot or not snapshot.values:
        raise SystemExit(f"오류: thread_id '{config.thread_id}'에 저장된 체크포인트가 없습니다.")
    # Arm the prompt guard before the first resumed node runs: on this path the data
    # reference comes from the checkpoint, not from the CLI arguments.
    resumed_reference = dict(snapshot.values.get("data_ref") or {})
    if resumed_reference.get("path"):
        register_private(resumed_reference["path"])
    if not snapshot.next:
        print(f"thread_id '{config.thread_id}'는 이미 완료된 실행입니다. `show`로 결과를 확인하세요.")
        print_outcome(dict(snapshot.values), config)
        return 0

    assert_resumable_data(resumed_reference, config.thread_id)
    print(f"체크포인트에서 재개 — thread_id={config.thread_id}, 다음 노드={snapshot.next}")
    print(f"저장된 진행 상황: iteration {snapshot.values.get('iteration')}")
    print("-" * 70)
    # Passing None resumes from the checkpoint instead of seeding a new state.
    state = stream_graph(app, runtime, config, None)
    print_outcome(state, config)
    return 0


def command_show(args: argparse.Namespace) -> int:
    config = load_run_config(
        args.thread_id, Path(args.artifacts_root) if args.artifacts_root else None
    )
    app, _saver, _runtime = open_app(config)
    snapshot = app.get_state({"configurable": {"thread_id": config.thread_id}})
    if not snapshot or not snapshot.values:
        raise SystemExit(f"오류: thread_id '{config.thread_id}'에 저장된 체크포인트가 없습니다.")

    state = dict(snapshot.values)
    print(f"thread_id: {config.thread_id}")
    print(f"상태: {'완료' if not snapshot.next else f'미완료 (다음 노드={snapshot.next})'}")
    print_outcome(state, config)

    print("")
    print("시도 이력:")
    for attempt in state.get("history") or []:
        result = attempt.get("result") or {}
        metrics = result.get("metrics") or {}
        score = metrics.get(config.metric)
        outcome = (
            f"{config.metric}={float(score):.4f}"
            if isinstance(score, (int, float))
            else f"실패({result.get('error_type') or 'unknown'})"
        )
        verdict = attempt.get("critic") or {}
        suffix = f" → critic: {verdict.get('failure_type')}" if verdict else ""
        print(f"  [iter {attempt.get('iteration')}] model={attempt.get('model')} {outcome}{suffix}")

    if args.report:
        report_text = str(state.get("report") or "")
        if not report_text:
            path = config.run_dir / "report.md"
            report_text = path.read_text(encoding="utf-8") if path.exists() else ""
        print("")
        print(report_text or "(보고서가 아직 없습니다)")
    return 0


def best_iteration(config: RunConfig) -> int | None:
    """Which iteration this run selected, from the checkpoint or from ``history.json``.

    Two sources because they fail differently. The checkpoint is authoritative and is what
    ``show`` reads, but it is a SQLite file that a half-finished run may not have flushed a
    ``best`` into; ``history.json`` is written by the report node at the end and survives the
    checkpoint being deleted. ``None`` when neither names one, which means no attempt
    succeeded — there is no model to predict with.
    """
    try:
        app, _saver, _runtime = open_app(config)
        snapshot = app.get_state({"configurable": {"thread_id": config.thread_id}})
    except Exception:  # noqa: BLE001 - a missing or unreadable checkpoint is not fatal here
        snapshot = None
    if snapshot and snapshot.values:
        iteration = (dict(snapshot.values).get("best") or {}).get("iteration")
        if isinstance(iteration, int):
            return iteration
    path = config.run_dir / "history.json"
    if path.exists():
        digest = json.loads(path.read_text(encoding="utf-8"))
        iteration = (dict(digest.get("best") or {})).get("iteration")
        if isinstance(iteration, int):
            return iteration
    return None


def command_predict(args: argparse.Namespace) -> int:
    """Apply a finished run's chosen model to a new CSV.

    The iteration is resolved rather than asked for, because "the model this run produced"
    is the thing a caller has in mind and picking the wrong directory out of ``train/`` is
    silent: every iteration has a ``model.joblib`` and the worst of them loads just as well.
    ``--iteration`` is still there for scoring a specific attempt on purpose.

    Both halves of the artifact are checked here, before the script starts, so a run that
    cannot be applied says so in terms of what the operator did — a ``--dry-run`` or
    synthetic run never fits an encoding, and one from before schemas were written has a
    model but no way to replay it.
    """
    from .scripts.predict import main as predict_main

    config = load_run_config(
        args.thread_id, Path(args.artifacts_root) if args.artifacts_root else None
    )
    iteration = args.iteration
    if iteration is None:
        iteration = best_iteration(config)
        if iteration is None:
            raise SystemExit(
                f"오류: thread_id '{config.thread_id}'에는 선택된 최고 시도가 없습니다 "
                "(성공한 학습이 없거나 실행이 report까지 도달하지 않았습니다).\n"
                "  - 상태 확인: `show --thread-id ...`\n"
                "  - 특정 시도를 지정하려면: --iteration <번호>"
            )
        print(f"이 실행이 선택한 iteration {iteration}의 모델을 사용합니다")

    model_path = config.model_path(iteration)
    schema_path = config.schema_path(iteration)
    if not model_path.exists():
        raise SystemExit(
            f"오류: iteration {iteration}의 모델 파일이 없습니다 ({model_path}).\n"
            "  - --dry-run 실행은 모델을 저장하지 않습니다.\n"
            f"  - 이 실행이 고른 최고 시도가 아니라면, 끝날 때 정리된 것일 수 있습니다 "
            f"(기본 --keep-models {DEFAULT_KEEP_MODELS}). 어느 iteration의 모델이 남았는지는 "
            "history.json의 models 블록에 적혀 있고, 진 시도까지 남기려면 다시 실행할 때 "
            "--keep-models all 을 주십시오.\n"
            "  - 다른 시도를 지정하려면: --iteration <번호>"
        )
    if not schema_path.exists():
        raise SystemExit(
            f"오류: iteration {iteration}의 인코딩 스키마가 없습니다 ({schema_path}).\n"
            "  이 파일 없이는 모델을 새 데이터에 적용할 수 없습니다 — 학습 당시 어떤 컬럼이 "
            "행렬의 몇 번째였는지, 범주 레벨이 무엇이었는지 남아 있지 않기 때문입니다. "
            "폭이 우연히 맞으면 어긋난 컬럼으로 예측이 나오고 아무것도 그것을 알려주지 않습니다.\n"
            "  - 합성 데이터로 돌린 실행에는 인코딩 자체가 없습니다 (--data 없이 카드만으로 실행한 경우).\n"
            "  - 이 파일이 저장되기 전 버전에서 만든 실행이면, 같은 데이터로 다시 학습해야 합니다."
        )

    out_path = (
        Path(args.out)
        if args.out
        else config.run_dir / "predict" / f"{Path(args.data).stem}_predictions.csv"
    )
    report_path = Path(args.report) if args.report else out_path.with_suffix(".report.json")
    code = predict_main(
        [
            "--model",
            str(model_path),
            "--schema",
            str(schema_path),
            "--data",
            str(args.data),
            "--out",
            str(out_path),
            "--report",
            str(report_path),
            *(["--id-column", str(args.id_column)] if args.id_column else []),
            *(["--label-column", str(args.label_column)] if args.label_column else []),
        ]
    )
    if code != 0:
        raise SystemExit("오류: 예측에 실패했습니다 (위 stderr 참고).")
    print(f"실행 요약: {report_path}")
    print(
        "출력 파일은 행 단위 예측이므로 원본 데이터와 같은 취급을 하십시오 — "
        "커밋하지 말고, 프롬프트에 넣지 마십시오."
    )
    return 0


def command_graph(args: argparse.Namespace) -> int:
    # Drawing needs no checkpointer, and shouldn't create an artifacts directory.
    config = RunConfig(thread_id="graph-render", dry_run=True)
    graph = build_state_graph(config).compile().get_graph()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".png":
        try:
            out_path.write_bytes(graph.draw_mermaid_png())
            print(f"그래프 이미지를 저장했습니다: {out_path}")
            return 0
        except Exception as exc:  # noqa: BLE001 - PNG rendering needs network access
            fallback = out_path.with_suffix(".mmd")
            fallback.write_text(graph.draw_mermaid(), encoding="utf-8")
            print(f"PNG 렌더링 실패({exc}) — Mermaid 소스로 저장했습니다: {fallback}")
            return 0
    out_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"Mermaid 소스를 저장했습니다: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m automl_agent.main",
        description="LangGraph 기반 AutoML 에이전트: 계획 → 모델 선택 → 학습 → 평가 → 재계획 루프",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser(
        "profile", help="원본 데이터에서 데이터셋 카드를 생성합니다 (집계만, 원본 행 없음)"
    )
    profile_parser.add_argument("--data", required=True, help="프로파일링할 CSV 경로")
    profile_parser.add_argument("--target", required=True, help="정답(label) 컬럼 이름")
    profile_parser.add_argument("--out", required=True, help="생성할 카드 JSON 경로")
    profile_parser.add_argument(
        "--name",
        default=None,
        help="카드에 기록할 데이터셋 이름 (기본: '<target>-prediction'. 파일 이름은 "
        "프롬프트에 데이터 출처를 남기게 되므로 일부러 쓰지 않습니다)",
    )
    profile_parser.add_argument(
        "--seed", type=int, default=42, help="기준선 holdout 분할 시드 (기본: 42, run의 --seed와 맞추세요)"
    )
    profile_parser.add_argument(
        "--metric",
        choices=list(GOAL_METRICS),
        default=DEFAULT_METRIC,
        help="이 카드로 실행할 때의 목표 지표. 도출될 임계값을 미리 보여주기 위한 것으로, "
        f"카드에는 기록되지 않습니다 (기본: {DEFAULT_METRIC})",
    )
    profile_parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"auto 모드 미리보기에 쓸 margin (기본: {DEFAULT_MARGIN}). run의 --margin과 같은 "
        "의미이며, 그 값으로 실행했을 때의 임계값을 미리 보여주기만 합니다",
    )
    profile_parser.add_argument(
        "--on-missing-target",
        choices=list(TARGET_MISSING_POLICIES),
        default=DEFAULT_TARGET_MISSING_POLICY,
        help="target(정답)이 비어 있는 행의 처리 방식 (기본: reject). reject = 개수를 알리고 "
        "중단, drop = 그 행들을 제외하고 몇 행을 뺐는지 카드에 기록",
    )
    profile_parser.add_argument(
        "--caveat",
        action="append",
        default=[],
        dest="caveats",
        metavar="문장",
        # ``%%`` and not ``%``: argparse runs every help string through ``%``-formatting, so a
        # bare percent sign followed by anything that is not a format character raises
        # ValueError — from inside ``--help``, which is where nobody has a workaround.
        help="집계만으로는 드러나지 않는 이 데이터의 주의사항. 카드에 기록되고 실행 시 모든 추론 "
        "프롬프트에 실립니다. 여러 번 쓸 수 있습니다. 원본을 직접 본 사람의 지식이 들어오는 "
        "유일한 통로입니다 — 예: \"결측이 0%%인 플래그인데도 뜻이 행 순서에 따라 바뀌니(앞 2%%, "
        "뒤 43%%), 모델이 중증도가 아니라 차팅 체계를 학습할 수 있습니다\"",
    )
    profile_parser.add_argument(
        "--group-column",
        default=None,
        metavar="컬럼",
        help="한 값에 속한 행들이 train/val/test로 흩어지면 안 되는 컬럼 — 한 행이 방문 1건이고 "
        "한 환자가 여러 행을 갖는 데이터의 환자 ID 같은 것. 지정하지 않으면 같은 환자가 학습과 "
        "검증 양쪽에 들어가고, 모델은 상태 대신 환자를 외워서 모든 점수가 부풀려집니다 — 이 "
        "부풀림은 홀드아웃도 같이 오염되므로 실행 안에서는 탐지할 방법이 없습니다. 이 컬럼은 "
        "특성에서 제외되고, 카드의 비공개 data 블록에 기록되므로 LLM은 보거나 바꿀 수 없습니다.",
    )
    profile_parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="기준선(logreg) 측정을 건너뜁니다. 이 카드로 실행하면 목표 임계값은 지표별 기본값이 됩니다.",
    )
    profile_parser.set_defaults(func=command_profile)

    run_parser = subparsers.add_parser("run", help="새 실행을 시작합니다")
    run_parser.add_argument(
        "--dataset-card", default=None, help="데이터셋 카드 JSON 경로 (--data 와 둘 중 하나 필수)"
    )
    run_parser.add_argument(
        "--data",
        default=None,
        help="원본 CSV 경로. 카드 없이 주면 profiling 노드가 카드를 만듭니다. "
        "이 경로는 data_ref 채널에만 들어가고 프롬프트에는 포함되지 않습니다.",
    )
    run_parser.add_argument(
        "--target", default=None, help="정답 컬럼 이름 (--data 를 쓸 때 필수)"
    )
    # choices, not a free string: a metric the trainer cannot emit would leave goal_met
    # False for the whole run with nothing in the output explaining why.
    run_parser.add_argument(
        "--metric",
        choices=list(GOAL_METRICS),
        default=DEFAULT_METRIC,
        help=f"목표 지표 이름 (기본: {DEFAULT_METRIC}). 이 지표가 정답 열의 task에 없으면 "
        "(분류 타겟에 rmse 등) 그 task의 기본 지표로 바꿔 실행하고, 바꿨다는 사실을 실행 시작 "
        "시점과 report에 적습니다",
    )
    run_parser.add_argument(
        "--goal-mode",
        choices=list(GOAL_MODES),
        default=None,
        help=f"목표 임계값을 정하는 방식 (기본: {DEFAULT_MODE}, --threshold 를 주면 fixed). "
        "auto = 카드의 기준선(logreg)에서 데이터셋마다 도출 (--margin 으로 조절). "
        "fixed = --threshold 로 준 값을 그대로, 안 주면 지표별 기본값.",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="fixed 모드의 목표 임계값. 이 값을 주면 --goal-mode fixed 로 간주합니다. "
        "auto 모드와 함께 쓸 수 없습니다.",
    )
    run_parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help=f"auto 모드에서 기준선의 남은 여유 중 목표로 삼을 비율 (기본: {DEFAULT_MARGIN}). "
        "fixed 모드에서는 의미가 없습니다.",
    )
    run_parser.add_argument(
        "--direction",
        choices=list(DIRECTIONS),
        default=None,
        help="지표 최적화 방향. 생략하면 지표에서 정해집니다 (mae·rmse는 minimize, 나머지는 "
        "maximize). 지표와 다른 방향을 주면 거부합니다 — 방향은 취향이 아닙니다.",
    )
    run_parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="최대 반복 횟수 (기본: 5)"
    )
    run_parser.add_argument(
        "--time-budget-sec",
        type=int,
        default=DEFAULT_TIME_BUDGET_SEC,
        help=(
            "실행 하나가 일하는 초의 상한 (기본: 3600). 10%%는 holdout 몫으로 예약되고, "
            "적합 하나의 몫은 남은 시간 ÷ 남은 반복 수입니다. 몫을 넘긴 적합은 too_slow, "
            "예산을 다 쓰고 끊긴 실행은 out_of_time으로 기록됩니다."
        ),
    )
    run_parser.add_argument(
        "--search-past-goal",
        action="store_true",
        help="목표를 넘어도 --max-iterations 까지 계속 탐색합니다. 기본값은 목표 달성 즉시 "
        "종료이고, auto 모드에서는 첫 시도가 바를 넘는 일이 흔하므로 그때 critic은 한 번도 "
        "실행되지 않습니다 — 진단·재계획 경로를 실제로 돌리려면 이 플래그입니다. 바를 올리지도, "
        "어느 시도가 이기는지를 바꾸지도 않고(승자는 여전히 val 최고), 남은 예산을 쓸 뿐입니다. "
        "쓴 예산이 달라지므로 이 플래그를 켠 실행은 끄고 잰 실행과 같은 조건이 아닙니다",
    )
    run_parser.add_argument("--thread-id", required=True, help="체크포인트 식별자")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="같은 thread_id의 기존 체크포인트를 삭제하고 처음부터 다시 실행합니다",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="LLM과 학습을 모두 모킹합니다")
    run_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="학습은 실제로 하되 추론 노드는 규칙 기반 폴백으로 실행합니다 (API 키 불필요)",
    )
    run_parser.add_argument(
        "--scenario",
        choices=list(DRY_RUN_SCENARIOS),
        default="success",
        help="--dry-run에서 모의 학습이 따라갈 시나리오 (기본: success)",
    )
    run_parser.add_argument(
        "--on-missing-target",
        choices=list(TARGET_MISSING_POLICIES),
        default=None,
        help="target(정답)이 비어 있는 행의 처리 방식. 생략하면 카드에 기록된 정책을 따르고, "
        "그것도 없으면 reject (중단). drop 을 주면 그 행들을 제외하고 학습합니다",
    )
    run_parser.add_argument(
        "--caveat",
        action="append",
        default=[],
        dest="caveats",
        metavar="문장",
        help="집계만으로는 드러나지 않는 이 데이터의 주의사항. 모든 추론 프롬프트에 실립니다. "
        "여러 번 쓸 수 있습니다. --data 경로에서는 카드에 기록되고, --dataset-card 경로에서는 "
        "카드가 이미 담고 있는 항목에 덧붙습니다 (원본 카드 파일은 바뀌지 않습니다)",
    )
    run_parser.add_argument(
        "--group-column",
        default=None,
        metavar="컬럼",
        help="한 값에 속한 행들이 train/val/test로 흩어지면 안 되는 컬럼 (예: 환자 ID). 지정하면 "
        "그 컬럼은 특성에서 제외되고 모든 점수는 처음 보는 그룹에 대한 성능이 됩니다. "
        "--dataset-card 경로에서는 생략하면 카드가 기록한 값을 그대로 따르므로 보통 다시 줄 "
        "필요가 없고, 카드와 다른 값을 주면 기준선과 비교할 수 없다는 이유로 실행이 중단됩니다.",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    run_parser.add_argument("--model", default="claude-opus-5", help="사용할 Claude 모델 ID")
    run_parser.add_argument("--artifacts-root", default=None, help="아티팩트 루트 디렉터리 재지정")
    run_parser.add_argument(
        "--keep-models",
        choices=list(KEEP_MODELS_MODES),
        default=DEFAULT_KEEP_MODELS,
        help=f"실행이 끝난 뒤 어느 iteration의 model.joblib을 남길지 (기본: {DEFAULT_KEEP_MODELS}). "
        "기본값은 predict가 실제로 쓰는 최고 시도의 모델만 남기고 나머지를 지웁니다 — 적합된 "
        "모델은 하이퍼파라미터에 따라 한 개가 수백 MB가 되고 아무것도 그것을 제한하지 않습니다. "
        "`predict --iteration <다른 번호>`로 진 시도를 쓰려면 all",
    )
    run_parser.set_defaults(func=command_run)

    resume_parser = subparsers.add_parser("resume", help="중단된 실행을 thread_id로 재개합니다")
    resume_parser.add_argument("--thread-id", required=True)
    resume_parser.add_argument("--artifacts-root", default=None)
    resume_parser.set_defaults(func=command_resume)

    show_parser = subparsers.add_parser("show", help="저장된 실행 상태를 출력합니다")
    show_parser.add_argument("--thread-id", required=True)
    show_parser.add_argument("--artifacts-root", default=None)
    show_parser.add_argument("--report", action="store_true", help="최종 보고서 전문까지 출력합니다")
    show_parser.set_defaults(func=command_show)

    predict_parser = subparsers.add_parser(
        "predict", help="끝난 실행이 선택한 모델을 새 CSV에 적용합니다"
    )
    predict_parser.add_argument("--thread-id", required=True, help="어느 실행의 모델을 쓸지")
    predict_parser.add_argument("--data", required=True, help="예측할 CSV 경로")
    predict_parser.add_argument(
        "--out",
        default=None,
        help="예측 결과 CSV 경로 (기본: 실행 디렉터리 아래 predict/<입력파일이름>_predictions.csv). "
        "행 단위 예측이므로 원본 데이터와 같은 등급으로 다루십시오",
    )
    predict_parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="특정 시도의 모델을 쓸 때만 지정합니다. 생략하면 이 실행이 선택한 최고 시도를 씁니다 — "
        "학습 디렉터리를 직접 골라 잘못된 시도의 모델을 쓰는 일은 아무 오류도 내지 않습니다",
    )
    predict_parser.add_argument(
        "--id-column",
        default=None,
        metavar="컬럼",
        help="출력에 그대로 실어 보낼 식별자 컬럼. 행 순서가 아니라 키로 원본과 다시 붙일 수 있게 "
        "합니다. 이 컬럼은 특성으로 쓰이지 않습니다",
    )
    predict_parser.add_argument(
        "--label-column",
        default=None,
        metavar="컬럼",
        help="입력 CSV에 정답 라벨이 이미 있을 때 그 컬럼 이름. 주면 예측에 더해 이 배치를 "
        "채점합니다 — 실행이 목표로 삼았던 지표로, 홀드아웃을 채점한 것과 같은 코드로. 이 점수는 "
        "실행의 채점 프로토콜이 아니며(이 파일이 어떤 행으로 이루어졌는지는 알 수 없습니다) "
        "출력에 그렇게 적힙니다. 이 컬럼은 특성으로 쓰이지 않습니다",
    )
    predict_parser.add_argument(
        "--report",
        default=None,
        help="실행 요약 JSON 경로 (기본: 예측 CSV 옆). 몇 행을 예측했는지, 학습 때의 인코딩과 "
        "어디가 어긋났는지, --label-column을 주면 채점 결과까지 담습니다",
    )
    predict_parser.add_argument("--artifacts-root", default=None, help="아티팩트 루트 디렉터리 재지정")
    predict_parser.set_defaults(func=command_predict)

    graph_parser = subparsers.add_parser("graph", help="그래프 구조를 이미지/Mermaid로 저장합니다")
    graph_parser.add_argument("--out", default="graph.png", help="출력 경로 (.png 또는 .mmd)")
    graph_parser.set_defaults(func=command_graph)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n중단되었습니다. 동일한 --thread-id로 `resume` 하면 이어서 실행됩니다.")
        return 130
    except sqlite3.OperationalError as exc:
        # The checkpoint database, always: it is the only sqlite file here. Caught at the
        # top because the failure surfaces from inside LangGraph's writer, so the traceback
        # blames a library the operator did not call, and the actual cause is another
        # process on the same machine. ``graph.make_checkpointer`` already waits
        # CHECKPOINT_TIMEOUT_SEC before this can happen.
        print(f"\n오류: 체크포인트 데이터베이스를 쓸 수 없습니다 — {exc}")
        if "locked" in str(exc).lower():
            print(
                "  같은 artifacts 디렉터리를 쓰는 다른 실행이 남아 있습니다 (다른 터미널의 "
                "`run`/`resume`, 또는 죽은 프로세스). 그 프로세스를 끝낸 뒤 같은 --thread-id로 "
                "`resume` 하면 마지막 체크포인트부터 이어집니다 — 지금까지의 학습은 남아 "
                "있습니다. 두 실행을 나란히 돌려야 한다면 한쪽에 --artifacts-root를 다르게 "
                "주십시오"
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
