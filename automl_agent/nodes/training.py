"""Training 노드: 얇은 서브프로세스 래퍼. 실행 노드 — 여기에 LLM은 없다.

늘 별도 프로세스인 이유:

* OOM이나 강한 CUDA 오류가 오케스트레이터가 아니라 자식을 죽인다;
* 프로세스 종료가 메모리를 전부 회수하므로, 긴 재계획 루프가 그것을 쌓을 수 없다.

그래서 이 노드는 네 가지만 한다: config를 쓰고, ``scripts/train.py``를 띄우고, ``result.json``을
파싱하고, 모든 실패를 Critic이 추론할 수 있는 보통의 결과로 바꾼다. 학습 실패에 절대 raise하지 않는다.

원본 데이터 경계의 나머지 절반이기도 하다. 경로는 비공개 ``data_ref`` 채널에서 오고, state로 돌아가는
것은 ``privacy.public_result`` — 지표, 상태, 시간, 그리고 씻어 낸 예외 한 줄. 예외 메시지 안에 셀 값을
인용할 수 있는 전체 로그는 ``artifacts/<thread_id>/train/iter_NN/`` 아래 디스크에 남는다.
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


def training(state: AutoMLState, *, config: RunConfig) -> dict:
    """학습 시도 하나를 돌리고 ``{"result": ...}``를 돌려준다(공개 필드만)."""
    iteration = state_int(state, "iteration")
    if config.dry_run:
        # 실제 결과와 같은 체를 지난다. 모킹된 경로가 Critic이 실제로 보는 모양에서 어긋날 수
        # 없게.
        return {"result": public_result(_mocked_result(state, config, iteration))}

    # 이 적합에 허용된 것: 전체 예산이 아니라 실행의 남은 시간 중 자기 몫. 예산을 계산하지 않는
    # 중이면 ``None``이고, 그때는 예전 상한이 적용된다.
    share = fit_share_sec(state)
    if share is not None and share <= 0:
        # 아무것도 띄우지 않는다. ``route``가 반복 사이에 예산을 보지만 그 결정 뒤의 계획·모델 선택
        # 호출도 시간을 쓰므로, 예산은 그 틈에서 소진될 수 있다. 1초 뒤 죽을 서브프로세스를 띄우면
        # spawn이 느린 것으로 기록된다. 이것은 이유를 기록하고, ``route``가 다음에 실행을 끝낸다.
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
        # 아무것도 띄우지 않았으므로 실패를 읽어 낼 로그도 result.json도 없다. 가드 없이는 이것이
        # 노드 밖으로 raise되어 오케스트레이터까지 데려갔다 — 파일에 대한 traceback 하나에
        # 체크포인트된 실행을 잃는 것이고, 그것이 이 노드가 막으려고 존재하는 하나뿐인 실패 양태다.
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
        # 증거가 아니라 보관용 사본이다: ``console``은 이미 메모리에 있고 ``log_tail``은 아래에서 거기서
        # 잘라 낸다. 그래서 이 시도는 자기 결과를 지키고, 사람이 나중에 읽었을 파일만 없어진다 — 소리 내어
        # 말하는 이유는 조용히 없는 train.log가 아예 돌지 않은 시도처럼 보이기 때문이다.
        print(f"  [training] iteration {iteration}의 {log_path.name}을 저장하지 못했습니다: {exc}")

    result = read_json_object(result_path)

    if timed_out:
        # 시간 예산을 집행하는 것은 자식이 아니라 여기, 오케스트레이터다.
        result = {
            "metrics": (result or {}).get("metrics", {}),
            "train_time_sec": round(elapsed, 3),
            "status": "error",
            "error_type": "too_slow",
            "log_tail": _tail(console) or f"exceeded this fit's share of the time budget ({timeout:.0f}s)",
        }
    elif result is None:
        # 파싱할 result 파일이 없다: 자식이 그것을 쓰기 전에 죽었다.
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

    # 경계: log_tail과 artifact 경로를 하류에서 걸러 내는 대신 여기서 떨어뜨린다. 그래서
    # ``result``는 구조적으로 프롬프트에 안전하고, 뒤의 어느 노드도 실수로도 그것을 흘릴 수 없다.
    # 파일 자체는 work_dir에 남고 그것은 ``config.iteration_dir(iteration)``이다 — 유도 가능하므로
    # state에 있을 필요가 없다.
    return {"result": public_result(result)}


# --------------------------------------------------------------------------- #
# Config 조립
# --------------------------------------------------------------------------- #


def build_train_config(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """카드, 비공개 데이터 참조, 계획, 모델을 config로 옮긴다."""
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
        # 모델 이름을 어느 추정기 계열에서 해소할지, 그리고 — 읽을 열이 없는 합성 경로에서는 —
        # 어느 생성기가 도는지. 실제 데이터에서는 스크립트가 정답 열을 직접 다시 읽고 *그것*을
        # 따른다: 카드는 열을 기술하고, 권위는 열에 있다.
        "task": card_task(card) or TASK_CLASSIFICATION,
        "seed": config.seed,
    }
    steps = pipeline_block(plan)
    if steps:
        # executor에서 ``preprocessing``에 합쳐지는 것이 아니라 그것을 대체한다 — spec이 오면
        # executor는 플래그를 무시하고, 둘 다 보내면 한 파이프라인의 기술이 config에 둘 들어가면서
        # 어느 쪽이 돌았는지 말할 수 있는 것이 없어진다. 플래그가 파일에 남는 이유는 카드의 기본값이
        # 여전히 거기 살고, 모르는 단계만으로 된 spec은 그 기본값이 아니라 아무것도 없는 쪽으로
        # 떨어지기 때문이다.
        train_config["pipeline"] = steps
    decision = decision_block(plan)
    if decision:
        # 무언가 청했을 때만. 그래서 조율하지 않는 실행은 늘 쓰던 ``train_config.json``을 그대로
        # 쓴다 — 그것이 앞선 시도의 config를 뒤의 것에 대고 재생해서 계획이 다른 곳에서만 다르게
        # 만드는 것이다.
        train_config["decision"] = decision
    baseline = paired_baseline(state, config)
    if baseline:
        train_config["paired_baseline"] = baseline
    if constraints.get("memory_limit_mb"):
        train_config["memory_limit_mb"] = constraints["memory_limit_mb"]
    if card.get("simulate"):
        # 코드가 아니라 카드가 선언한 실패 주입 훈련.
        train_config["simulate"] = card["simulate"]
    return train_config


def paired_baseline(state: AutoMLState, config: RunConfig) -> dict[str, Any]:
    """이 시도를 행 단위로 견줄 앞선 시도.

    실행의 현재 최고 — 모든 소비자가 이미 하는 그 비교다(ledger의 "직전 최고 대비", ``evaluate``의
    ``improved``, 보고서의 대표 줄). ``best``는 :mod:`automl_agent.nodes.evaluate`의 것이고 그것은
    training *뒤에* 도므로, 여기서는 iteration 1..N-1의 최고를 담는다: 뺄셈이 쓰는 바로 그 baseline이다.

    config에 들어가는 것은 반복 번호와 경로이고, 예측은 절대 아니다. 경로는 번호의 순함수이므로 state가
    파일 내용을 기억할 필요가 없고, 파일을 읽는 것은 학습 서브프로세스다 — 이미 경계의 데이터 쪽이다.

    아직 baseline이 없거나 그 반복이 예측 파일을 남기지 않았으면(오류, 또는 쓰기 실패) ``{}``. executor가
    둘 다 침묵이 아니라 이유가 붙은 ``skipped`` 블록으로 바꾼다.
    """
    iteration = (state.get("best") or {}).get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
        return {}
    if iteration == state_int(state, "iteration"):
        # 그래프를 통해서는 일어날 수 없다 — ``evaluate``가 이 반복에 대해 아직 안 돌았다 — 그러나
        # 재개된 실행은 체크포인트에서 state를 다시 세우고, 자신과 짝지어진 시도는 정확히 0인 델타와
        # 0인 P를 공개하는데, 그것은 기록 오류가 아니라 측정된 미개선으로 읽힌다.
        return {}
    path = config.predictions_path(iteration)
    if not path.exists():
        return {}
    return {"iteration": iteration, "path": str(path)}


def target_missing_policy(card: dict[str, Any], config: RunConfig) -> str:
    """이 시도가 따르는 라벨 없는 행 정책.

    플래그가 주어졌으면 그것이 이긴다. 그 밖에는 카드의 것인데, ``--on-missing-target drop``으로 세운
    카드는 이미 줄어든 행 집합을 기술하고 있고, 다른 정책으로 학습하면 baseline이 측정된 것과 다른
    데이터셋을 채점하게 되기 때문이다. 둘 다 없으면 ``reject``: 라벨 없는 행은 기본적으로 데이터 준비
    버그다.
    """
    if config.on_missing_target:
        return config.on_missing_target
    declared = (card.get("target_missing") or {}).get("policy")
    if isinstance(declared, str) and declared in TARGET_MISSING_POLICIES:
        return declared
    return DEFAULT_TARGET_MISSING_POLICY


# ``scripts.train.PREPROCESSING_ALIASES``와 발을 맞춰 둔다. 여기서 그것을 import할 수 없는 이유는
# 그 모듈이 sklearn을 끌어오고 오케스트레이터 프로세스는 그러지 않기 때문이다.
_PREPROCESSING_ALIASES: dict[str, str] = {
    "add_missing_indicators": "missing_indicator",
    "add_missing_indicator": "missing_indicator",
    "missing_indicators": "missing_indicator",
    "add_indicator": "missing_indicator",
    "missing_counts": "missing_count",
    "n_missing": "missing_count",
}


def preprocessing_block(plan: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """executor가 실제로 존중하는 전처리 설정만 통과시킨다.

    ``plan.preprocessing``은 자유 형식 LLM 출력이다. ``train.py``가 구현하는 것은 행렬 전체에 대한 대치
    전략 하나와 스케일에 민감한 추정기용 스케일링뿐이므로, 나머지를 흘려보내면 검증되지 않은 모델 작성
    키가 executor의 config 파일에 들어가고 보고서가 돌지 않은 변환을 주장하게 된다.

    ``none``은 모델 계열을 보지 않고 통과시킨다. 계열 검사는 executor의 몫이기 때문이다:
    ``_wrap_preprocessing``이 모든 경로가 — 이 노드가 본 적 없는 손으로 쓴 config까지 — 도착하는 곳이고,
    계열이 NaN을 받을 수 없으면 거기서 요청을 낮춘다.
    """
    raw = plan.get("preprocessing") or card.get("preprocessing") or {}
    if not isinstance(raw, dict):
        return {}
    # 별칭을 executor에서만이 아니라 여기서도 정규화한다. 전략 목록을 되적는 것과 같은 이유의 의도된
    # 반복이다: 이 노드는 executor가 보기 전에 그 키를 받아들여야 하고, executor는 이 노드가 건드린 적
    # 없는 손으로 쓴 config에서 그것을 받아들여야 한다.
    raw = {_PREPROCESSING_ALIASES.get(str(name), str(name)): value for name, value in raw.items()}
    block: dict[str, Any] = {}
    impute = raw.get("impute")
    if isinstance(impute, str) and impute in {"median", "mean", "most_frequent", "none"}:
        block["impute"] = impute
    for flag in ("scale", "missing_indicator", "missing_count"):
        if isinstance(raw.get(flag), bool):
            block[flag] = raw[flag]
    return block


# ``scripts.train.DECISION_TUNED``와 발을 맞춰 둔다. ``_PREPROCESSING_ALIASES``와 같은 이유로 여기
# 되적는다: 이 노드가 executor가 보기 전에 그 값을 세워야 한다.
_DECISION_TUNED = "tuned"


def decision_block(plan: dict[str, Any]) -> dict[str, Any]:
    """계획의 ``tune_threshold``를 executor의 ``decision`` config로 바꾼다.

    불리언이 들어가고 문자열이 나오는 비대칭이 요점이다: executor의 키는 ``"tuned"`` *또는* 명시적 컷을
    받는데, 손으로 쓴 config나 테스트에는 하나를 지목할 이유가 있다. *계획*에는 없다 — planner는 이 모델의
    확률을 본 적이 없으므로, 거기서 나온 수는 그 분포에 대한 짐작이 결정으로 분장한 것이다.

    ``True``만 센다. ``False``와 없음은 같은 요청(기본 0.5 규칙)이고 둘 다 ``{}``를 주므로, config 파일은
    "이전과 같음"을 뜻하는 키를 지니는 대신 그대로 남는다.
    """
    if plan.get("tune_threshold") is True:
        return {"threshold": _DECISION_TUNED}
    return {}


def _named_step(entry: dict[str, Any]) -> dict[str, Any]:
    """``{"impute": {...}}``를 ``{"step": "impute", ...}``로 편다.

    제안자가 실제로 쓴 모양이다. 단계 이름이 ``step``의 *값*이라는 것은 CAN 목록도 planning 프롬프트도
    글자로 보여주지 않으므로(둘 다 "a list of steps"까지만 말한다) 모델은 껍데기를 짐작해야 하고, 이름을
    키로 적는 쪽을 짐작한다. 짐작이 틀렸을 때 실제로 일어난 일: 명세 전체가 아래의 ``step`` 검사에서
    조용히 떨어지고, config에 ``pipeline`` 키가 아예 없고, executor는 플래그 경로를 돌고, 서로 다른
    전처리를 청한 세 반복이 바이트 단위로 같은 config로 같은 점수를 냈다.

    읽는 쪽에서 받는 이유는 여기가 모든 명세가 지나는 한 곳이기 때문이다. 애매하지 않을 때만 편다 —
    키가 하나뿐이고, 그것이 아는 단계 이름이고, 값이 dict일 때. 그 밖에는 손대지 않고 원래의 검사에
    보낸다. 껍데기 안의 ``step``보다 키 쪽 이름이 이기는 이유는 이 모양에서 이름을 적는 자리가
    키이기 때문이다.
    """
    if "step" in entry or len(entry) != 1:
        return entry
    name, body = next(iter(entry.items()))
    if name not in PIPELINE_STEPS or not isinstance(body, dict):
        return entry
    return {**body, "step": name}


def pipeline_block(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """executor가 해석할 순서 있는 파이프라인 spec만 통과시킨다.

    ``plan.pipeline``도 자유 형식 LLM 출력이고 이것이 그 문이다. :func:`preprocessing_block`과 같은
    조건이다: 모르는 단계 이름은 config 파일에 절대 닿지 않으므로, executor는 모르는 이름을 해소하라고
    요청받지 않고 어떤 보고서도 구현 없는 변환을 주장할 수 없다. 이름은
    :data:`automl_agent.dataset.pipeline.STEPS`에서 오고, 되적는 대신 import한다 — 그 모듈은 import 시점에
    stdlib보다 멀리 닿지 않고, 그래서 registry가 ``scripts/train.py``가 아니라 거기 산다.

    단계 *안쪽*의 키는 일부러 여기서 걸러지 않는다. executor가 각각을 곧 세울 대상에 대고 검증하고(전략은
    imputer 자신의 목록에, 열은 적합된 스키마에, 차수는 존재하는 것에) 자기가 한 일을
    ``applied_pipeline``에 보고한다. 여기서 두 번째 검증을 하려면 그 세 목록의 두 번째 사본이 필요하고,
    어긋나는 것은 그 사본이다.

    ``columns``는 있으면 정렬한다. 그래서 한 선택의 두 표기가
    :func:`automl_agent.nodes.planning._signature`에게 두 계획이 아니라 하나의 서명이 된다.
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
    """비공개 데이터 참조를 ``train.py``의 데이터 계약에 맞춘다.

    ``path``가 있는 참조는 실제 행에서 학습한다. 없으면 카드가 선언한 모양을 합성하므로, 카드만으로도
    여전히 루프 전체를 돌릴 수 있다. 경로는 ``data_ref``에서 오고 카드에서는 절대 오지 않는다는 점에
    주의: 이 노드에 닿는 카드는 이미 비공개 ``data`` 블록이 벗겨져 있다.
    """
    declared = dict(reference or {})
    if declared.get("path"):
        block = {
            "path": declared["path"],
            "target_column": declared.get("target_column") or card.get("target_column") or "target",
        }
        if declared.get("group_column"):
            # 전달만 하고 기본값을 세우지 않는다: 카드의 baseline이 측정된 분할 규약이 train.py가
            # 재현하는 것이어야 한다.
            block["group_column"] = str(declared["group_column"])
        return block

    difficulty = dict(card.get("difficulty") or {})
    balance = card.get("class_balance")
    synthetic: dict[str, Any] = {
        "n_samples": int(card.get("n_rows", 5000) or 5000),
        "n_features": int(card.get("n_features", 20) or 20),
    }
    if card_task(card) == TASK_REGRESSION:
        # 클래스도, 분리도, 라벨 뒤집기도 없다 — 그 셋의 연속 대응물은 하나의 수, 즉
        # ``make_regression``이 더하는 줄일 수 없는 잡음이다.
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
# 결과 파싱 보조
# --------------------------------------------------------------------------- #


def _unwritable(iteration: int, path: Path, exc: OSError) -> dict[str, Any]:
    """시작할 수 없었던 시도, 다른 모든 실패한 시도와 같은 모양으로.

    꽉 찬 디스크나 읽기 전용 ``artifacts/``는 모델링 실패가 아니고, Critic이 제안할 수 있는 어떤 변경도
    그것을 고치지 못한다 — 그래서 콘솔 줄이 운영자가 똑같은 세 반복에서 추론하게 두는 대신 그것을 대놓고
    말한다. 결과는 그래도 보통 채널을 지난다. 대안(raise)은 실행을 버리기 때문이다.

    ``error_type``이 가장 가까운 기존 이름이 아니라 자기 이름인 이유: ``config_error``는
    ``critic.ERROR_TYPE_MAP``에서 ``data_issue``로 가고, 그러면 Critic이 공간이 없는 디스크에 대고 열
    수정을 처방한다.
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
    """실행 예산에 자리가 없었던 시도, 다른 모든 실패한 시도와 같은 모양으로.

    자기 이름이 아니라 ``too_slow``인 이유는 루프 쪽에서 보면 시간을 넘긴 적합과 같은 사실이기 때문이다:
    이 반복은 점수를 내지 못했고 이유는 시계다. ``critic.ERROR_TYPE_MAP``은 이미 그것을 비용에 대한
    진단으로 보내고, 어차피 Critic은 다시 돌지 않는다 — ``route``가 여기서 본 것과 같은 예산으로 실행을
    끝낸다.

    ``train_time_sec``는 0.0이고 그것이 정직한 값이다: 아무것도 적합하지 않았다. 비용 0으로 읽히면서
    실패한 시도가 정확히 일어난 일이고, 독자가 이것을 초가 정말 쓰인 timeout 경우와 가르는 방법이다.
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
# --dry-run 학습기
# --------------------------------------------------------------------------- #


def _mocked_result(state: AutoMLState, config: RunConfig, iteration: int) -> dict[str, Any]:
    """``--dry-run <시나리오>``로 고르는, 학습의 결정적인 대역.

    궤적이 반복 번호에서 나오므로 모든 시나리오가 ``route``의 서로 다른 분기로 끝난다: 목표 도달, 반복
    예산, 정체.
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
            # 추정기를 세우지 않았으므로 좁혀진 것도 낮춰진 것도 없다: 제안이 곧 돌아간 것이다.
            # 그래도 넣어 둔다. 모킹된 경로가 실제 경로의 모양을 갖게.
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
        # 세 번째 시도에서 목표에 닿는다.
        return ok(threshold - 0.09 + 0.05 * (iteration - 1))
    if scenario == "fail":
        # 바를 넘기에는 너무 느리게 나아진다: 반복 예산에서 끝난다.
        return ok(threshold - 0.20 + 0.02 * (iteration - 1))
    if scenario == "oom":
        # 첫 시도가 터지고, Critic의 축소 권고가 회복시킨다.
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
    # "stall": 영원히 같은 점수이므로 stall_count가 실행을 끝낸다.
    return ok(threshold - 0.10)
