"""Training node: a thin subprocess wrapper. An execution node, no LLM.

Roles:

* Training run: spawn ``scripts/train.py``; failures become normal results.
* Config building: pass only settings the executor supports.
* Failure results: results for attempts that could not start.
* Dry-run trainer: fixed stand-in under ``--dry-run``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from ..config import TRAIN_SCRIPT, RunConfig, read_json_object, run_fixed_script
from ..dataset.pipeline import STEPS as PIPELINE_STEPS
from ..dataset.targets import DEFAULT_TARGET_MISSING_POLICY, TARGET_MISSING_POLICIES
from ..privacy import public_result, register_private
from ..scoring.goal import goal_threshold
from ..scoring.metrics import TASK_CLASSIFICATION, TASK_REGRESSION, card_task
from ..state import AutoMLState, fit_share_sec, state_int

# --- Role: training run ---------------------------------------------------------------


def training(state: AutoMLState, *, config: RunConfig) -> dict:
    """Run one training attempt and return its public result. Never raises on failure."""
    iteration = state_int(state, "iteration")
    if config.dry_run:
        # Same filter as a real result, so they match.
        return {"result": public_result(_mocked_result(state, config, iteration))}

    # This fit's share of time; ``None`` means no budget.
    share = fit_share_sec(state)
    if share is not None and share <= 0:
        # Out of time: spawn nothing
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
        # Return a failed result, never raise
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
    returncode, console = run_fixed_script(command, timeout=timeout, label="training")
    elapsed = time.perf_counter() - started
    timed_out = returncode == -9

    try:
        log_path.write_text(console, encoding="utf-8")
    except OSError as exc:
        # Warn: a missing log looks like no attempt ran.
        print(f"  [training] iteration {iteration}의 {log_path.name}을 저장하지 못했습니다: {exc}")

    result = read_json_object(result_path)

    if timed_out:
        # The orchestrator enforces the time limit, not the child.
        result = {
            "metrics": (result or {}).get("metrics", {}),
            "train_time_sec": round(elapsed, 3),
            "status": "error",
            "error_type": "too_slow",
            "log_tail": _tail(console) or f"exceeded this fit's share of the time budget ({timeout:.0f}s)",
        }
    elif result is None:
        # The child died before writing its result.
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
    result["returncode"] = returncode

    # Boundary: drops log_tail and paths before state.
    return {"result": public_result(result)}


# --- Role: config building ------------------------------------------------------------


def build_train_config(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Build the ``train_config.json`` dict from card, data reference, plan, and model."""
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
        # Real data: the script re-reads the task from target.
        "task": card_task(card) or TASK_CLASSIFICATION,
        "seed": config.seed,
    }
    steps = pipeline_block(plan)
    if steps:
        # Replaces ``preprocessing``, not merged
        train_config["pipeline"] = steps
    decision = decision_block(plan)
    if decision:
        # Only when asked for
        train_config["decision"] = decision
    baseline = paired_baseline(state, config)
    if baseline:
        train_config["paired_baseline"] = baseline
    if constraints.get("memory_limit_mb"):
        train_config["memory_limit_mb"] = constraints["memory_limit_mb"]
    if card.get("simulate"):
        # Failure drill declared by the card, not code.
        train_config["simulate"] = card["simulate"]
    return train_config


def paired_baseline(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """Pick the current best as the row-by-row baseline, as iteration and path.

    Returns ``{}`` with no baseline or no predictions file
    """
    iteration = (state.get("best") or {}).get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        return {}
    if iteration == state_int(state, "iteration"):
        # Resumed runs only; self-pairing fakes a zero delta.
        return {}
    path = config.predictions_path(iteration)
    if not path.exists():
        return {}
    return {"iteration": iteration, "path": str(path)}


def target_missing_policy(card: dict[str, Any], config: RunConfig) -> str:
    """Choose the missing-label policy: flag, then card, then ``reject``.

    Why this order:
    """
    if config.on_missing_target:
        return config.on_missing_target
    declared = (card.get("target_missing") or {}).get("policy")
    if isinstance(declared, str) and declared in TARGET_MISSING_POLICIES:
        return declared
    return DEFAULT_TARGET_MISSING_POLICY


# Copy of ``scripts.train.PREPROCESSING_ALIASES``; importing loads sklearn.
_PREPROCESSING_ALIASES: dict[str, str] = {
    "add_missing_indicators": "missing_indicator",
    "add_missing_indicator": "missing_indicator",
    "missing_indicators": "missing_indicator",
    "add_indicator": "missing_indicator",
    "missing_counts": "missing_count",
    "n_missing": "missing_count",
}


def preprocessing_block(plan: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """Pass through only preprocessing settings the executor supports.

    Plan first, card as fallback
    """
    raw = plan.get("preprocessing") or card.get("preprocessing") or {}
    if not isinstance(raw, dict):
        return {}
    # Also done in the executor, on purpose
    raw = {_PREPROCESSING_ALIASES.get(str(name), str(name)): value for name, value in raw.items()}
    block: dict[str, Any] = {}
    impute = raw.get("impute")
    if isinstance(impute, str) and impute in {"median", "mean", "most_frequent", "none"}:
        block["impute"] = impute
    for flag in ("scale", "missing_indicator", "missing_count"):
        if isinstance(raw.get(flag), bool):
            block[flag] = raw[flag]
    return block


# Copy of ``scripts.train.DECISION_TUNED``, same reason.
_DECISION_TUNED = "tuned"


def decision_block(plan: dict[str, Any]) -> dict[str, Any]:
    """Turn ``tune_threshold is True`` into ``{"threshold": "tuned"}``, else ``{}``.

    A plan never names a cut
    """
    if plan.get("tune_threshold") is True:
        return {"threshold": _DECISION_TUNED}
    return {}


def _named_step(entry: dict[str, Any]) -> dict[str, Any]:
    """_named_step | Config building: unwrap ``{"impute": {...}}`` into ``{"step": "impute", ...}``."""
    if "step" in entry or len(entry) != 1:
        return entry
    name, body = next(iter(entry.items()))
    if name not in PIPELINE_STEPS or not isinstance(body, dict):
        return entry
    return {**body, "step": name}


def pipeline_block(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Pass through only pipeline steps with known names, columns sorted.

    Keys inside a step are checked by the executor
    """
    raw = plan.get("pipeline")
    if not isinstance(raw, list):
        return []
    steps: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        entry = _named_step(entry)
        if str(entry.get("step") or "") not in PIPELINE_STEPS:
            continue
        step = {key: value for key, value in entry.items() if isinstance(key, str)}
        for holder in (step, *(g for g in step.get("groups") or [] if isinstance(g, dict))):
            names = holder.get("columns")
            if isinstance(names, list):
                holder["columns"] = sorted({str(name) for name in names if isinstance(name, str)})
        steps.append(step)
    return steps


def data_block(card: dict[str, Any], reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the ``data`` block: real rows from ``data_ref``, else synthetic from the card.

    The path comes only from ``data_ref``, never the card.
    """
    declared = dict(reference or {})
    if declared.get("path"):
        block = {
            "path": declared["path"],
            "target_column": declared.get("target_column") or card.get("target_column") or "target",
        }
        if declared.get("group_column"):
            # No default: repeat the baseline's split.
            block["group_column"] = str(declared["group_column"])
        for key in ("table", "query"):
            # Same rows as profiling read.
            if declared.get(key):
                block[key] = str(declared[key])
        return block

    difficulty = dict(card.get("difficulty") or {})
    balance = card.get("class_balance")
    synthetic: dict[str, Any] = {
        "n_samples": int(card.get("n_rows", 5000) or 5000),
        "n_features": int(card.get("n_features", 20) or 20),
    }
    if card_task(card) == TASK_REGRESSION:
        # Regression has only ``make_regression`` noise.
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


# --- Role: failure results ------------------------------------------------------------


def _unwritable(iteration: int, path: Path, exc: OSError) -> dict[str, Any]:
    """_unwritable | Failure results: result when the config could not be written."""
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
    """_out_of_time | Failure results: result when no time is left."""
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
    """_tail | Training run: keep the last ``limit`` characters of output."""
    return text[-limit:]


# --- Role: dry-run trainer ------------------------------------------------------------


def _mocked_result(state: AutoMLState, config: RunConfig, iteration: int) -> dict[str, Any]:
    """_mocked_result | Dry-run trainer: fixed fake result for ``--dry-run <scenario>``."""
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
            # Nothing dropped; kept for the real result's shape.
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
        # Too slow to pass: ends on iteration budget.
        return ok(threshold - 0.20 + 0.02 * (iteration - 1))
    if scenario == "oom":
        # First attempt OOMs, then the Critic's advice recovers.
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
    # "stall": same score forever, so stall_count ends it.
    return ok(threshold - 0.10)
