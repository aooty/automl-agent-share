"""Holdout node: score the best model once, on rows the loop never saw. No LLM.

Runs after ``route`` has decided to stop, between ``evaluate`` and ``report``. Everything
before it — every plan, every hyperparameter, and the choice of ``best`` itself — was
decided against validation scores. This node is the one measurement that was not: it
loads the model file the winning iteration saved and scores it on the test slice
:mod:`automl_agent.scoring.splits` carved off before the first fit.

Why that matters enough to be its own node: with five attempts, ``best`` is the maximum
of five noisy validation numbers, so quoting it as the run's result overstates the model
by an amount the run itself cannot measure. The test score can be quoted without that
caveat, and the gap between the two *is* the caveat, in the metric's own units.

The test score arrives with its own bootstrap interval (:mod:`automl_agent.scoring.intervals`),
and this node reports whether the validation score falls inside it. That is the second
question about the same gap and it is not the same as the first: a gap of 0.004 on a slice
whose interval spans 0.06 is a gap this measurement cannot distinguish from zero, and
printing the number without the width invites reading it as a measured amount of bias.

**This is the acceptance channel, and it has to stay one.** Two different decisions are made
about differences in this system and they carry different costs. Steering — what to prescribe
next — is decided inside the loop on the paired interval
(:func:`automl_agent.scoring.intervals.paired_delta`), where being sensitive is the point and a false
positive costs one iteration. Acceptance — the claim that the run improved on something — is
decided here, once, on rows no plan and no diagnosis ever saw.

So this number must never be promoted into a gate on which iteration wins. The moment
``best`` is chosen, confirmed or overruled by a test score, the test 20% is part of the
selection set, and the promise ``capabilities._CLASSIFICATION_SCORING`` makes to every plan —
carved off first, scored exactly once after the loop ends, never visible to a plan or a
diagnosis — is false from that instant, with nothing left in the run able to measure what it
cost. A gate needs a fourth split or repeated cross-validation; it does not need this one.

``selection_gap`` is also not a substitute for the multiplicity argument in
:mod:`automl_agent.scoring.intervals`. The 0.0078 measured on the MIMIC sample is the *size* of one
realisation of the selection effect; multiplicity is a *rate*. A small gap on one run says
nothing about how often a sensitive decision line fires on noise, and the two numbers cannot
stand in for each other in either direction.

Never fatal, and never a route decision. If there is no saved model — a dry run, a run
whose every attempt failed, an old run from before models were persisted — the node
records why and the report says so. Refusing to write a report because the final
measurement was unavailable would throw away the evidence the run did accumulate.

Third member of the raw-data boundary, on the same terms as ``training``: the path comes
from the private ``data_ref`` channel, the work happens in a subprocess, and what returns
to state goes through :func:`automl_agent.privacy.public_result`.

Nothing here prints. :func:`describe` renders the one line and ``main.ConsoleReporter``
calls it *after* flushing the iteration summary that ``evaluate`` left buffered — a print
from inside this function would land before that line, since the console only sees a node's
state update once the node has already returned.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from typing import Any

from ..config import TRAIN_SCRIPT, RunConfig, decode_output, read_json_object, utf8_env
from ..privacy import public_result
from ..scoring.intervals import CI_LEVEL, contains, interval_of
from ..scoring.metrics import MINIMIZE, direction_of
from ..scoring.splits import TEST_FRACTION
from ..state import AutoMLState, holdout_share_sec

# What the report prints when there is nothing to measure. Keys, not sentences, so the
# report can phrase them and the CLI can branch on them.
SKIP_REASONS = {
    "dry_run": "--dry-run이므로 저장된 모델이 없습니다",
    "no_best": "성공한 시도가 없어 채점할 모델이 없습니다",
    "no_model": "최고 성능 iteration의 모델 파일을 찾을 수 없습니다",
    "failed": "테스트 채점 자체가 실패했습니다",
}


def holdout(state: AutoMLState, *, config: RunConfig) -> dict:
    """Return ``{"holdout": {...}}`` — the test-split score, or why there is none."""
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
        # The winning iteration's own config, not a rebuilt one: the split is a function
        # of the data path, the target-missing policy, the group column and the seed, so
        # reusing the file the fit ran from is what guarantees the same rows are held back.
        str(config_path),
        "--out",
        str(out_path),
        "--score-model",
        str(model_path),
    ]

    started = time.perf_counter()
    console = ""
    returncode = -1
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=utf8_env(),
            # One pass over 20% of the rows with an already-fitted model, so the share held
            # back for it is generous rather than tight — and it is held back on purpose. The
            # loop is stopped before it can spend this, because a run whose budget deleted its
            # own held-out score would report the numbers it selected on and nothing else.
            timeout=holdout_share_sec(state) or config.train_timeout_sec,
            check=False,
        )
        console = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        console = decode_output(exc.stdout) + decode_output(exc.stderr)
        returncode = -9
    except OSError as exc:
        console = f"failed to spawn the holdout subprocess: {exc}"

    # The log is a convenience for a human, not the result.
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
        # Which model was scored, so the report can say "iteration 3's hist_gbdt", and so
        # a reader can tell the number was not produced by a fresh fit.
        "iteration": iteration,
        "model": best.get("model", ""),
        "metric": metric,
        "score": result["metrics"].get(metric),
        # The validation score of the same model, for the one comparison this node exists
        # to make: how much of ``best`` was selection.
        "val_score": best.get("score"),
        "metrics": result["metrics"],
        "wall_time_sec": round(time.perf_counter() - started, 3),
    }
    score = holdout_block["score"]
    if isinstance(score, (int, float)) and isinstance(best.get("score"), (int, float)):
        # Signed so that positive always means "validation was optimistic", whichever way the
        # metric runs. ``best`` is the *minimum* of the noisy validation numbers on an error
        # metric, so ``best - test`` there is negative exactly when the selection effect is
        # present — and ``describe`` calls this number the size of that effect.
        gap = float(best["score"]) - float(score)
        holdout_block["selection_gap"] = round(-gap if direction_of(metric) == MINIMIZE else gap, 6)
        # Whether the gap is bigger than what these rows can resolve. The test split is
        # ~20% of the file, so its own interval is the widest in the run, and a gap inside
        # it is a gap this measurement cannot tell from zero — reporting "선택 편향
        # 0.004" off a slice with a ±0.03 interval was the number this fixes.
        bounds = interval_of(result["metrics"], metric)
        if bounds is not None:
            holdout_block["score_ci"] = [bounds[0], bounds[1]]
            holdout_block["val_inside_ci"] = contains(bounds, best.get("score"))
    return {"holdout": holdout_block}


def _skipped(reason: str, iteration: int | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"status": "skipped", "reason": reason, "split": "test"}
    if iteration is not None:
        block["iteration"] = iteration
    return block


def describe(block: dict[str, Any]) -> str:
    """One Korean line about the final measurement, for a console or a report."""
    if not block:
        return "최종 테스트 채점: 수행되지 않았습니다"
    if block.get("status") != "ok":
        return f"최종 테스트 채점 생략 — {SKIP_REASONS.get(str(block.get('reason')), '사유 불명')}"
    metric = block.get("metric", "metric")
    score = block.get("score")
    if not isinstance(score, (int, float)):
        return f"최종 테스트 채점: iteration {block.get('iteration')} 모델, {metric} 값 없음"
    line = (
        f"최종 테스트({int(float(block.get('test_fraction') or TEST_FRACTION) * 100)}%, "
        f"반복 중 한 번도 쓰이지 않은 행): iteration {block.get('iteration')}의 "
        f"{block.get('model')} → {metric}={score:.4f}"
    )
    ci = block.get("score_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        line += f" ({int(CI_LEVEL * 100)}% CI {float(ci[0]):.4f}~{float(ci[1]):.4f})"
    gap = block.get("selection_gap")
    if isinstance(gap, (int, float)):
        val = block.get("val_score")
        line += (
            f" (검증 {float(val):.4f} 대비 {gap:+.4f}"
            if isinstance(val, (int, float))
            else f" (검증 대비 {gap:+.4f}"
        )
        line += " — 이 차이가 선택 편향의 크기입니다"
        # Said out loud, because the gap is the headline number of this node and a reader
        # who is not told will read any nonzero value as a measured amount of bias.
        if block.get("val_inside_ci"):
            line += ", 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다"
        line += ")"
    return line
