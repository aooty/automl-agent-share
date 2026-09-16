"""Training node: a thin subprocess wrapper. Execution node — no LLM here.

Why a separate process, always:

* an OOM or a hard CUDA fault kills the child, not the orchestrator;
* process exit reclaims all memory, so a long replan loop cannot accumulate it.

This node therefore does only four things: write the config, spawn
``scripts/train.py``, parse ``result.json``, and turn every failure into a normal
result the Critic can reason about. It never raises on training failure.

It is also the second half of the raw-data boundary. The path comes from the private
``data_ref`` channel, and what goes back into state is ``privacy.public_result`` —
metrics, status, timings and one scrubbed exception line. The full log, which can
quote cell values inside an exception message, stays on disk under
``artifacts/<thread_id>/train/iter_NN/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..config import TRAIN_SCRIPT, RunConfig, decode_output, read_json_object, utf8_env
from ..dataset.pipeline import STEPS as PIPELINE_STEPS
from ..dataset.targets import DEFAULT_TARGET_MISSING_POLICY, TARGET_MISSING_POLICIES
from ..privacy import public_result, register_private
from ..scoring.goal import goal_threshold
from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION, card_task
from ..state import AutoMLState, fit_share_sec


def training(state: AutoMLState, *, config: RunConfig) -> dict:
    """Run one training attempt and return ``{"result": ...}`` (public fields only)."""
    iteration = int(state.get("iteration", 0) or 0)
    if config.dry_run:
        # Through the same filter as a real result, so the mocked path cannot drift
        # into a different shape than the one the Critic sees in production.
        return {"result": public_result(_mocked_result(state, config, iteration))}

    # What this fit is allowed: its slice of the run's remaining time, not the whole budget.
    # ``None`` when no budget is being accounted for, and then the old ceiling applies.
    share = fit_share_sec(state)
    if share is not None and share <= 0:
        # Nothing is spawned. ``route`` checks the budget between iterations, but the planning
        # and model-selection calls after its decision cost time too, so the budget can run out
        # in the gap. Spending a subprocess to have it killed a second later would record the
        # spawn as the slow thing; this records the reason, and ``route`` ends the run next.
        return {"result": public_result(_out_of_time(iteration))}
    timeout = config.train_timeout_sec if share is None else min(share, config.train_timeout_sec)

    work_dir = config.iteration_dir(iteration)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "train_config.json"
    result_path = work_dir / "result.json"
    log_path = work_dir / "train.log"

    train_config = build_train_config(state, config)
    try:
        config_path.write_text(
            json.dumps(train_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        # Nothing was spawned, so there is no log and no result.json to read a failure out
        # of. Unguarded, this raised out of the node and took the orchestrator with it —
        # losing the checkpointed run to a traceback about a file, which is the one failure
        # mode this node exists to prevent.
        return {"result": public_result(_unwritable(iteration, config_path, exc))}

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--out",
        str(result_path),
    ]

    started = time.perf_counter()
    timed_out = False
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            command,
            capture_output=True,
            text=True,
            # Pinned rather than locale-derived: an exception message can quote a
            # non-ASCII column name, and on a cp949 console the default decoding
            # would raise inside subprocess' reader thread and lose the log.
            encoding="utf-8",
            errors="replace",
            env=utf8_env(),
            timeout=timeout,
            check=False,
        )
        console = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        console = decode_output(exc.stdout) + decode_output(exc.stderr)
    except OSError as exc:  # spawning itself failed
        console = f"failed to spawn the training subprocess: {exc}"
    elapsed = time.perf_counter() - started

    try:
        log_path.write_text(console, encoding="utf-8")
    except OSError as exc:
        # The archive copy, not the evidence: ``console`` is already in memory and
        # ``log_tail`` is cut from it below. So this attempt keeps its result and only the
        # file a human would have read afterwards is missing — said out loud, because a
        # silently absent train.log looks like an attempt that never ran.
        print(f"  [training] iteration {iteration}의 {log_path.name}을 저장하지 못했습니다: {exc}")

    result = read_json_object(result_path)

    if timed_out:
        # The time budget is enforced here, by the orchestrator, not by the child.
        result = {
            "metrics": (result or {}).get("metrics", {}),
            "train_time_sec": round(elapsed, 3),
            "status": "error",
            "error_type": "too_slow",
            "log_tail": _tail(console) or f"exceeded this fit's share of the time budget ({timeout:.0f}s)",
        }
    elif result is None:
        # No parsable result file: the child died before it could write one.
        returncode = completed.returncode if completed is not None else -1
        result = {
            "metrics": {},
            "train_time_sec": round(elapsed, 3),
            "status": "error",
            "error_type": "crash" if returncode != 0 else "no_result",
            "log_tail": _tail(console) or "the training subprocess produced no result.json",
        }

    result.setdefault("status", "error")
    result.setdefault("metrics", {})
    result["wall_time_sec"] = round(elapsed, 3)
    result["returncode"] = completed.returncode if completed is not None else (-9 if timed_out else -1)

    # The boundary: log_tail and the artifact paths are dropped here rather than
    # filtered downstream, so ``result`` is prompt-safe by construction and no later
    # node can leak them even by mistake. The files themselves stay in work_dir, which
    # is ``config.iteration_dir(iteration)`` — derivable, so it need not be in state.
    return {"result": public_result(result)}


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #


def build_train_config(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Translate the card, the private data reference, the plan and the model into a config."""
    card = dict(state.get("dataset_card") or {})
    plan = dict(state.get("plan") or {})
    constraints = dict(card.get("constraints") or {})
    reference = dict(state.get("data_ref") or {})
    if reference.get("path"):
        register_private(reference["path"])

    train_config: dict[str, Any] = {
        "model": str(state.get("model") or "hist_gbdt"),
        "hyperparams": dict(state.get("hyperparams") or {}),
        "preprocessing": preprocessing_block(plan, card),
        "target_missing": {"policy": target_missing_policy(card, config)},
        "data": data_block(card, reference),
        "metric": str((state.get("goal") or {}).get("metric", config.metric)),
        # Which family of estimators the model name is resolved in, and — on the synthetic
        # path, where there is no column to read — which generator runs. On real data the
        # script re-reads the target column itself and follows *that*: the card describes the
        # column, the column is the authority.
        "task": card_task(card) or TASK_CLASSIFICATION,
        "seed": config.seed,
    }
    steps = pipeline_block(plan)
    if steps:
        # Replaces ``preprocessing`` in the executor rather than joining it — the executor
        # ignores the flags when a spec arrives, and sending both would put two descriptions of
        # one pipeline in the config with nothing able to say which ran. The flags stay in the
        # file because the card's defaults still live there and a spec of only unknown steps
        # falls back to nothing rather than to them.
        train_config["pipeline"] = steps
    decision = decision_block(plan)
    if decision:
        # Only when something asked for it, so a run that does not tune writes the same
        # ``train_config.json`` it always did — which is what lets an earlier attempt's config be
        # replayed against a later one and differ only where the plan differed.
        train_config["decision"] = decision
    baseline = paired_baseline(state, config)
    if baseline:
        train_config["paired_baseline"] = baseline
    if constraints.get("memory_limit_mb"):
        train_config["memory_limit_mb"] = constraints["memory_limit_mb"]
    if card.get("simulate"):
        # Failure-injection drill declared by the card, not by the code.
        train_config["simulate"] = card["simulate"]
    return train_config


def paired_baseline(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Which earlier attempt this one should be compared against, row by row.

    The run's best so far — the comparison every consumer already makes (the ledger's "직전 최고
    대비", ``evaluate``'s ``improved``, the report's headline). ``best`` is owned by
    :mod:`automl_agent.nodes.evaluate`, which runs *after* training, so here it holds the best of
    iterations 1..N-1: exactly the baseline the subtraction uses.

    Into the config goes an iteration number and a path, never predictions. The path is a pure
    function of the number, so no file contents need remembering in state, and the file is read by the
    training subprocess — already on the data side of the boundary.

    ``{}`` when there is no baseline yet, or that iteration left no predictions file (errored, or the
    write failed). The executor turns both into a ``skipped`` block with a reason, not silence.
    """
    iteration = (state.get("best") or {}).get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        return {}
    if iteration == int(state.get("iteration", 0) or 0):
        # Cannot happen through the graph — ``evaluate`` has not run for this iteration yet —
        # but a resumed run rebuilds state from a checkpoint, and an attempt paired against
        # itself would publish a delta of exactly 0 and a P of 0, which reads as a measured
        # non-improvement rather than as a bookkeeping error.
        return {}
    path = config.predictions_path(iteration)
    if not path.exists():
        return {}
    return {"iteration": iteration, "path": str(path)}


def target_missing_policy(card: dict[str, Any], config: RunConfig) -> str:
    """Which unlabelled-row policy this attempt follows.

    The flag wins when it was given; otherwise the card's, because a card built with
    ``--on-missing-target drop`` already describes the reduced row set, and training under
    a different policy would score a different dataset than the baseline was measured on.
    Absent both, ``reject``: an unlabelled row is a data-preparation bug by default.
    """
    if config.on_missing_target:
        return config.on_missing_target
    declared = (card.get("target_missing") or {}).get("policy")
    if isinstance(declared, str) and declared in TARGET_MISSING_POLICIES:
        return declared
    return DEFAULT_TARGET_MISSING_POLICY


# Kept in step with ``scripts.train.PREPROCESSING_ALIASES``, which cannot be imported here:
# that module pulls in sklearn, and the orchestrator process does not.
_PREPROCESSING_ALIASES: dict[str, str] = {
    "add_missing_indicators": "missing_indicator",
    "add_missing_indicator": "missing_indicator",
    "missing_indicators": "missing_indicator",
    "add_indicator": "missing_indicator",
    "missing_counts": "missing_count",
    "n_missing": "missing_count",
}


def preprocessing_block(plan: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """Allowlist the preprocessing settings the executor actually honours.

    ``plan.preprocessing`` is free-form LLM output. ``train.py`` only implements one
    imputation strategy for the whole matrix plus scaling for scale-sensitive estimators, so
    passing the rest through would put unvalidated model-authored keys into the executor's
    config file and let a report claim transformations that never ran.

    ``none`` is allowed through without checking the model family, because the family check
    belongs to the executor: ``_wrap_preprocessing`` is where every path arrives, including
    a hand-written config this node never saw, and it downgrades the request when the family
    cannot take a NaN.
    """
    raw = plan.get("preprocessing") or card.get("preprocessing") or {}
    if not isinstance(raw, dict):
        return {}
    # Aliases are normalised here as well as in the executor. Deliberate repetition, for the
    # same reason the strategy list is repeated: this node has to admit the key before the
    # executor ever sees it, and the executor has to admit it in a hand-written config this
    # node never touched.
    raw = {_PREPROCESSING_ALIASES.get(str(name), str(name)): value for name, value in raw.items()}
    block: dict[str, Any] = {}
    impute = raw.get("impute")
    if isinstance(impute, str) and impute in {"median", "mean", "most_frequent", "none"}:
        block["impute"] = impute
    for flag in ("scale", "missing_indicator", "missing_count"):
        if isinstance(raw.get(flag), bool):
            block[flag] = raw[flag]
    return block


# Kept in step with ``scripts.train.DECISION_TUNED``, restated here for the same reason
# ``_PREPROCESSING_ALIASES`` is: this node has to build the value before the executor sees it.
_DECISION_TUNED = "tuned"


def decision_block(plan: dict[str, Any]) -> dict[str, Any]:
    """Turn the plan's ``tune_threshold`` into the executor's ``decision`` config.

    Boolean in, string out, and the asymmetry is the point: the executor's key takes ``"tuned"`` *or*
    an explicit cut, because a hand-written config or a test has reason to name one. A *plan* does
    not — the planner has never seen this model's probabilities, so a number from it is a guess about
    their distribution dressed as a decision.

    Only ``True`` counts. ``False`` and absent are the same request (the default 0.5 rule) and both
    give ``{}``, so the config file is unchanged rather than carrying a key meaning "as before".
    """
    if plan.get("tune_threshold") is True:
        return {"threshold": _DECISION_TUNED}
    return {}


def pipeline_block(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Allowlist the ordered pipeline spec the executor will interpret.

    ``plan.pipeline`` is free-form LLM output and this is its gate, same terms as
    :func:`preprocessing_block`: an unknown step name never reaches the config file, so the executor
    is never asked to resolve a name it does not know and no report can claim a transform with no
    implementation. Names come from :data:`automl_agent.dataset.pipeline.STEPS`, imported rather than
    restated — that module reaches no further than the stdlib at import time, which is why the
    registry lives there and not in ``scripts/train.py``.

    Keys *inside* a step are deliberately not filtered here. The executor validates each against the
    thing it is about to build (a strategy against the imputer's own list, a column against the fitted
    schema, a degree against the one that exists) and reports what it did in ``applied_pipeline``. A
    second validation here would need a second copy of all three lists, and the copy is what drifts.

    ``columns`` is sorted where present, so two spellings of one selection are one signature to
    :func:`automl_agent.nodes.planning._signature` rather than two plans.
    """
    raw = plan.get("pipeline")
    if not isinstance(raw, list):
        return []
    steps: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or str(entry.get("step") or "") not in PIPELINE_STEPS:
            continue
        step = {key: value for key, value in entry.items() if isinstance(key, str)}
        for holder in (step, *(g for g in step.get("groups") or [] if isinstance(g, dict))):
            names = holder.get("columns")
            if isinstance(names, list):
                holder["columns"] = sorted({str(name) for name in names if isinstance(name, str)})
        steps.append(step)
    return steps


def data_block(card: dict[str, Any], reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map the private data reference onto ``train.py``'s data contract.

    A reference with a ``path`` trains on real rows; without one the card's declared
    shape is synthesised, so a card alone is still enough to run the whole loop. Note
    that the path comes from ``data_ref`` and never from the card: the card that
    reaches this node has already had its private ``data`` block stripped.
    """
    declared = dict(reference or {})
    if declared.get("path"):
        block = {
            "path": declared["path"],
            "target_column": declared.get("target_column") or card.get("target_column") or "target",
        }
        if declared.get("group_column"):
            # Forwarded, never defaulted: the split protocol the card's baseline was
            # measured under has to be the one train.py reproduces.
            block["group_column"] = str(declared["group_column"])
        return block

    difficulty = dict(card.get("difficulty") or {})
    balance = card.get("class_balance")
    synthetic: dict[str, Any] = {
        "n_samples": int(card.get("n_rows", 5000) or 5000),
        "n_features": int(card.get("n_features", 20) or 20),
    }
    if card_task(card) == TASK_REGRESSION:
        # No classes, no separation, no label flips — the continuous counterpart of all three
        # is one number, the irreducible noise ``make_regression`` adds.
        synthetic["noise"] = float(difficulty.get("noise", 10.0) or 10.0)
    else:
        synthetic.update(
            {
                "n_classes": int(card.get("n_classes", 2) or 2),
                "class_weights": list(balance) if isinstance(balance, (list, tuple)) else None,
                "class_sep": float(difficulty.get("class_sep", 0.9) or 0.9),
                "flip_y": float(difficulty.get("label_noise", 0.03) or 0.0),
            }
        )
    if card.get("n_informative"):
        synthetic["n_informative"] = int(card["n_informative"])
    return {"path": None, "target_column": declared.get("target_column", "target"), "synthetic": synthetic}


# --------------------------------------------------------------------------- #
# Result parsing helpers
# --------------------------------------------------------------------------- #


def _unwritable(iteration: int, path: Path, exc: OSError) -> dict[str, Any]:
    """The attempt that could not start, shaped like every other failed attempt.

    A full disk or a read-only ``artifacts/`` is not a modelling failure, and no change the
    Critic can propose will fix it — so the console line says that outright instead of
    leaving the operator to infer it from three identical iterations. The result still goes
    through the normal channel, because the alternative (raising) discards the run.

    ``error_type`` is a name of its own rather than the nearest existing one:
    ``config_error`` maps to ``data_issue`` in ``critic.ERROR_TYPE_MAP``, which would have
    the Critic prescribing column fixes for a disk that is out of space.
    """
    print(
        f"  [training] iteration {iteration}의 {path.name}을 쓸 수 없습니다: {exc}\n"
        "    학습이 실패한 것이 아니라 디스크나 권한 문제입니다 — 제안을 바꿔도 다음 "
        "iteration은 같은 지점에서 멈춥니다. artifacts 디렉터리의 남은 공간과 쓰기 권한을 "
        "확인하고 같은 --thread-id로 `resume` 하십시오"
    )
    return {
        "metrics": {},
        "train_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "status": "error",
        "error_type": "write_failed",
        "returncode": -1,
        "log_tail": f"could not write {path.name}: {exc}",
    }


def _out_of_time(iteration: int) -> dict[str, Any]:
    """The attempt the run budget had no room for, shaped like every other failed attempt.

    ``too_slow`` and not a name of its own, because from the loop's side it is the same fact as a
    fit that overran: this iteration produced no score and the reason is the clock.
    ``critic.ERROR_TYPE_MAP`` already routes that to a diagnosis about cost, and the Critic will
    not run again anyway — ``route`` ends the run on the same budget this checked.

    ``train_time_sec`` is 0.0 and that is the honest value: nothing was fitted. An attempt that
    reads as zero-cost and failed is exactly what happened, and it is how a reader tells this
    apart from the timeout case, where the seconds were really spent.
    """
    print(
        f"  [training] iteration {iteration}: 시간 예산이 이 시도를 시작하기 전에 소진됐습니다 "
        "— 학습을 시작하지 않았습니다.\n"
        "    --time-budget-sec를 올리거나 --max-iterations를 줄이십시오 (반복마다 남은 시간을 "
        "나눠 씁니다)"
    )
    return {
        "metrics": {},
        "train_time_sec": 0.0,
        "wall_time_sec": 0.0,
        "status": "error",
        "error_type": "too_slow",
        "returncode": -1,
        "log_tail": "the run's time budget was exhausted before this fit started; nothing was spawned",
    }


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


# --------------------------------------------------------------------------- #
# --dry-run trainer
# --------------------------------------------------------------------------- #


def _mocked_result(state: AutoMLState, config: RunConfig, iteration: int) -> dict[str, Any]:
    """Deterministic stand-in for training, selected by ``--scenario``.

    Trajectories are keyed off the iteration so every scenario terminates through
    a different branch of ``route``: goal reached, iteration budget, or stall.
    """
    goal = dict(state.get("goal") or {})
    threshold = goal_threshold(goal, config.fallback_threshold)
    metric = str(goal.get("metric", config.metric))
    scenario = config.dry_run_scenario

    def ok(score: float, seconds: float = 12.0) -> dict[str, Any]:
        score = max(0.0, min(1.0, round(score, 4)))
        return {
            "metrics": {
                metric: score,
                "accuracy": round(min(1.0, score + 0.03), 4),
                f"train_{metric}": round(min(1.0, score + 0.06), 4),
                "train_val_gap": 0.06,
            },
            # No estimator was built, so nothing was narrowed and nothing was downgraded:
            # the proposal *is* what ran. Present all the same, so the mocked path has the
            # real path's shape.
            "applied_hyperparams": dict(state.get("hyperparams") or {}),
            "dropped_hyperparams": [],
            "applied_preprocessing": preprocessing_block(
                dict(state.get("plan") or {}), dict(state.get("dataset_card") or {})
            ),
            "train_time_sec": seconds,
            "status": "ok",
            "error_type": None,
            "log_tail": f"[dry-run:{scenario}] iteration {iteration} finished",
            "dry_run": True,
        }

    def failed(error_type: str, seconds: float = 3.0) -> dict[str, Any]:
        return {
            "metrics": {},
            "train_time_sec": seconds,
            "status": "error",
            "error_type": error_type,
            "log_tail": f"[dry-run:{scenario}] iteration {iteration} failed with {error_type}",
            "dry_run": True,
        }

    if scenario == "success":
        # Reaches the goal on the third attempt.
        return ok(threshold - 0.09 + 0.05 * (iteration - 1))
    if scenario == "fail":
        # Improves too slowly to ever clear the bar: ends on the iteration budget.
        return ok(threshold - 0.20 + 0.02 * (iteration - 1))
    if scenario == "oom":
        # First attempt blows up; the Critic's shrink advice lets it recover.
        if iteration <= 1:
            return failed("oom")
        return ok(threshold - 0.05 + 0.06 * (iteration - 2))
    if scenario == "slow":
        if iteration <= 1:
            return failed("too_slow", seconds=config.train_timeout_sec)
        return ok(threshold - 0.04 + 0.05 * (iteration - 2))
    if scenario == "crash":
        if iteration <= 1:
            return failed("crash")
        return ok(threshold - 0.06 + 0.07 * (iteration - 2))
    # "stall": the same score forever, so stall_count ends the run.
    return ok(threshold - 0.10)
