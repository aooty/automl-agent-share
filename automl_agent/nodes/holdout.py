"""Holdout 노드: 최고 모델을 루프가 한 번도 못 본 행에서 한 번 채점한다. LLM 없음.

``route``가 멈추기로 정한 뒤, ``evaluate``와 ``report`` 사이에 돈다. 그 전의 모든 것 — 모든 계획, 모든
하이퍼파라미터, ``best``의 선택 자체 — 은 검증 점수에 대고 결정되었다. 이 노드는 그렇지 않은 하나뿐인
측정이다: 이긴 iteration이 저장한 모델 파일을 읽어, :mod:`automl_agent.scoring.splits`가 첫 적합 전에
떼어 둔 test 슬라이스에서 채점한다.

``best``는 잡음 있는 검증 수 여럿의 최대이므로 그것을 인용하면 모델을 과장한다. test 점수는 그 유보 없이
인용할 수 있고, **둘 사이의 격차가 곧 그 유보다**. 점수는 자기 구간과 함께 오고, 이 노드는 검증 점수가
그 안에 드는지 보고한다 — 구간보다 작은 격차는 이 측정이 0과 구분하지 못하는 격차다.

**승인 채널이고 하나로 남아야 한다**, 그리고 **이 수는 어느 반복이 이기는지에 대한 게이트가 되면 안
된다**. 게이트가 필요하면 네 번째 분할이나 반복 CV가 필요하다.

``selection_gap``은 :mod:`automl_agent.scoring.intervals`의 **다중성 논거를 대신하지 않는다**: 하나는
*크기*, 다른 하나는 *비율*이다.

**치명적이지 않고, route 결정도 아니다.** 저장된 모델이 없으면 그렇게 기록하고 보고서에서 말한다 —
최종 측정을 할 수 없었다고 보고서 쓰기를 거부하면 실행이 모은 증거를 버리는 일이 된다.

원본 데이터 경계의 세 번째 구성원, ``training``과 같은 조건이다: 경로는 비공개 ``data_ref`` 채널에서,
작업은 서브프로세스에서, 복귀는 :func:`automl_agent.privacy.public_result`를 통해. 여기서는 아무것도
찍지 않는다 — ``main.ConsoleReporter``가 버퍼된 반복 요약을 비운 *뒤* :func:`describe`를 부른다.
콘솔은 노드가 반환한 다음에야 그 갱신을 보기 때문이다.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any

from ..config import TRAIN_SCRIPT, RunConfig, read_json_object, run_fixed_script
from ..privacy import public_result
from ..scoring.intervals import CI_LEVEL, contains, interval_of
from ..scoring.metrics import MINIMIZE, direction_of
from ..scoring.splits import TEST_FRACTION
from ..state import AutoMLState, holdout_share_sec

# 잴 것이 없을 때 보고서가 찍는 것. 문장이 아니라 키인 이유는 보고서가 표현을 정하고 CLI가 여기에 대고
# 분기할 수 있게 하려고.
SKIP_REASONS = {
    "dry_run": "--dry-run이므로 저장된 모델이 없습니다",
    "no_best": "성공한 시도가 없어 채점할 모델이 없습니다",
    "no_model": "최고 성능 iteration의 모델 파일을 찾을 수 없습니다",
    "failed": "테스트 채점 자체가 실패했습니다",
}


def holdout(state: AutoMLState, *, config: RunConfig) -> dict:
    """``{"holdout": {...}}`` — test 분할 점수, 또는 그것이 없는 이유."""
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
        # 다시 조립한 것이 아니라 이긴 iteration 자신의 config. 분할은 데이터 경로, 정답 결측 정책, group
        # 열, 시드의 함수이므로, 적합이 돌았던 파일을 그대로 쓰는 것이 같은 행이 떼어져 있음을 보장한다.
        str(config_path),
        "--out",
        str(out_path),
        "--score-model",
        str(model_path),
    ]

    started = time.perf_counter()
    returncode, console = run_fixed_script(
        command,
        # 이미 적합된 모델로 행 20%를 한 번 지나는 것이므로, 여기 떼어 둔 몫은 빡빡하기보다 넉넉하다 —
        # 그리고 일부러 떼어 둔다. 루프는 이것을 쓸 수 있기 전에 멈춰지는데, 예산이 자기 holdout 점수를
        # 지워 버린 실행은 자기가 선택에 쓴 수만 보고하고 그 밖에는 아무것도 보고하지 못하기 때문이다.
        timeout=holdout_share_sec(state) or config.train_timeout_sec,
        label="holdout",
    )

    # 로그는 사람을 위한 편의이고 결과가 아니다.
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
        # 어느 모델을 채점했는지. 보고서가 "iteration 3의 hist_gbdt"라고 말할 수 있고, 독자가 이 수가
        # 새 적합에서 나온 것이 아님을 알 수 있게.
        "iteration": iteration,
        "model": best.get("model", ""),
        "metric": metric,
        "score": result["metrics"].get(metric),
        # 같은 모델의 검증 점수. 이 노드가 존재하는 이유인 그 하나의 비교를 위해: ``best``의 얼마가
        # 선택이었는지.
        "val_score": best.get("score"),
        "metrics": result["metrics"],
        "wall_time_sec": round(time.perf_counter() - started, 3),
    }
    score = holdout_block["score"]
    if isinstance(score, (int, float)) and isinstance(best.get("score"), (int, float)):
        # 지표가 어느 쪽으로 가든 양수가 언제나 "검증이 낙관적이었다"를 뜻하도록 부호를 맞춘다. 오류
        # 지표에서 ``best``는 잡음 있는 검증 수들의 *최소*이므로, 거기서 ``best - test``는 선택 효과가
        # 있을 때 정확히 음수다 — 그리고 ``describe``는 이 수를 그 효과의 크기라고 부른다.
        gap = float(best["score"]) - float(score)
        holdout_block["selection_gap"] = round(-gap if direction_of(metric) == MINIMIZE else gap, 6)
        # 격차가 이 행들이 분해할 수 있는 것보다 큰지. test 분할은 파일의 ~20%이므로 그 구간이 실행에서
        # 가장 넓고, 그 안에 든 격차는 이 측정이 0과 구분하지 못하는 격차다.
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
    """최종 측정에 대한 한글 한 줄, 콘솔이나 보고서용."""
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
        # 소리 내어 말한다. 격차가 이 노드의 대표 숫자이고, 듣지 못한 독자는 0이 아닌 값을 모두 측정된
        # 편향의 양으로 읽기 때문이다.
        if block.get("val_inside_ci"):
            line += ", 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다"
        line += ")"
    return line
