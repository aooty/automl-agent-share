"""Holdout node: scores the best model once on rows the loop never saw. No LLM.

Roles:

* Final scoring: score the winning model on the test slice.
* Selection gap: compare test and validation scores of it.
* Console line: :func:`describe` gives one Korean line.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any

from ..config import TRAIN_SCRIPT, RunConfig, read_json_object, run_fixed_script
from ..privacy import public_result
from ..scoring.intervals import CI_LEVEL, as_number, contains, interval_of
from ..scoring.metrics import MINIMIZE, direction_of
from ..scoring.splits import TEST_FRACTION
from ..state import AutoMLState, holdout_share_sec

# Why no score was taken, by reason key.
SKIP_REASONS = {
    "dry_run": "--dry-run이므로 저장된 모델이 없습니다",
    "no_best": "성공한 시도가 없어 채점할 모델이 없습니다",
    "no_model": "최고 성능 iteration의 모델 파일을 찾을 수 없습니다",
    "failed": "테스트 채점 자체가 실패했습니다",
}


# --- Role: final scoring --------------------------------------------------------------


def holdout(state: AutoMLState, *, config: RunConfig) -> dict:
    """Score the best model on the test split, or record why not.

    Runs the train script in a subprocess; writes ``holdout.json`` and ``holdout.log``.
    """
    best = dict(state.get("best") or {})
    iteration = best.get("iteration")

    if config.dry_run:
        return {"holdout": _skipped("dry_run")}
    if not isinstance(iteration, int):
        return {"holdout": _skipped("no_best")}

    model_path = config.model_path(iteration)
    config_path = config.iteration_dir(iteration) / "train_config.json"
    if not model_path.exists() or not config_path.exists():
        return {"holdout": _skipped("no_model")}

    out_path = config.run_dir / "holdout.json"
    log_path = config.run_dir / "holdout.log"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        # Reuse the winner's config so the same rows stay held out.
        str(config_path),
        "--out",
        str(out_path),
        "--score-model",
        str(model_path),
    ]

    started = time.perf_counter()
    returncode, console = run_fixed_script(
        command,
        # Time kept aside from the run budget
        timeout=holdout_share_sec(state) or config.train_timeout_sec,
        label="holdout",
    )

    # Log is for people only, not the result.
    with contextlib.suppress(OSError):
        log_path.write_text(console, encoding="utf-8")

    payload = read_json_object(out_path)
    if returncode != 0 or payload is None or payload.get("status") != "ok":
        return {"holdout": _skipped("failed", iteration=iteration)}

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return {"holdout": _skipped("failed", iteration=iteration)}

    result = public_result(payload)
    metric = str((state.get("goal") or {}).get("metric", config.metric))
    holdout_block: dict[str, Any] = {
        "status": "ok",
        "split": "test",
        "test_fraction": TEST_FRACTION,
        "iteration": iteration,
        "model": best.get("model", ""),
        "metric": metric,
        "score": result["metrics"].get(metric),
        "val_score": best.get("score"),
        "metrics": result["metrics"],
        "wall_time_sec": round(time.perf_counter() - started, 3),
    }
    score = as_number(holdout_block["score"])
    val_score = as_number(best.get("score"))
    if score is not None and val_score is not None:
        # Positive gap always means validation was optimistic
        gap = val_score - score
        holdout_block["selection_gap"] = round(-gap if direction_of(metric) == MINIMIZE else gap, 6)
        # A gap inside the test interval looks like 0.
        bounds = interval_of(result["metrics"], metric)
        if bounds is not None:
            holdout_block["score_ci"] = [bounds[0], bounds[1]]
            holdout_block["val_inside_ci"] = contains(bounds, best.get("score"))
    return {"holdout": holdout_block}


def _skipped(reason: str, iteration: int | None = None) -> dict[str, Any]:
    """_skipped | Final scoring: build the ``skipped`` block for one reason key."""
    block: dict[str, Any] = {"status": "skipped", "reason": reason, "split": "test"}
    if iteration is not None:
        block["iteration"] = iteration
    return block


# --- Role: console line ---------------------------------------------------------------


def describe(block: dict[str, Any]) -> str:
    """Describe the final test score in one Korean line, with interval and gap."""
    if not block:
        return "최종 테스트 채점: 수행되지 않았습니다"
    if block.get("status") != "ok":
        return f"최종 테스트 채점 생략 — {SKIP_REASONS.get(str(block.get('reason')), '사유 불명')}"
    metric = block.get("metric", "metric")
    score = as_number(block.get("score"))
    if score is None:
        return f"최종 테스트 채점: iteration {block.get('iteration')} 모델, {metric} 값 없음"
    line = (
        f"최종 테스트({int(float(block.get('test_fraction') or TEST_FRACTION) * 100)}%, "
        f"반복 중 한 번도 쓰이지 않은 행): iteration {block.get('iteration')}의 "
        f"{block.get('model')} → {metric}={score:.4f}"
    )
    ci = block.get("score_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        line += f" ({int(CI_LEVEL * 100)}% CI {float(ci[0]):.4f}~{float(ci[1]):.4f})"
    gap = as_number(block.get("selection_gap"))
    if gap is not None:
        val = as_number(block.get("val_score"))
        line += (
            f" (검증 {val:.4f} 대비 {gap:+.4f}" if val is not None else f" (검증 대비 {gap:+.4f}"
        )
        line += " — 이 차이가 선택 편향의 크기입니다"
        # Else readers take any gap as real bias.
        if block.get("val_inside_ci"):
            line += ", 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다"
        line += ")"
    return line
