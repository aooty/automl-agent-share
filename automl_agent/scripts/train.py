"""고정 학습 스크립트. ``nodes/training.py``가 격리된 subprocess로 실행한다.

이 파일은 LLM이 생성하지 *않는다*. 주입된 config — 모델 이름, 하이퍼파라미터, 데이터 출처 — 가
행동을 전부 몬다. 그래야 실행 경로가 결정적이고 감사 가능하다.

계약
----
입력 : ``--config <path>``(JSON), ``--out <path>``(result.json을 쓸 곳),
       선택적으로 ``--score-model <model.joblib>``
출력 : ``result.json``::

    {"metrics": {...}, "split": "val" | "test", "applied_hyperparams": {...},
     "dropped_hyperparams": [...], "train_time_sec": float,
     "status": "ok" | "error", "error_type": str | null, "log_tail": str,
     "model_path": str | null}

그리고 그 옆의 ``model.joblib`` — 적합된 파이프라인이라, 최고 시도를 다시 적합하지 않고 나중에
다시 채점할 수 있다. ``model_path``는 로컬이다: ``privacy.PUBLIC_RESULT_FIELDS``에 없다. 적합된
추정기는 데이터와 동등하기 때문이다 (SVC는 support vector를 그대로 저장한다).

한 파일에 두 모드. ``--score-model`` 없이는 학습 분할에 적합하고 *검증* 점수를 보고한다. 있으면
아무것도 적합하지 않는다: 저장된 모델을 :mod:`automl_agent.scoring.splits`가 실행 전체에서 떼어
둔 test 분할에서 한 번 채점한다. 보고서에서 어떤 결정도 그것에 대고 선택되지 않은 유일한 수다.

*왜*가 그것을 쓰는 코드 옆에 사는 필드들. ``result.json``의 모양을 한자리에서 읽을 수 있도록
이름만 적는다:

``applied_hyperparams``  제안이 이 모델이 받는 것으로 좁혀진 뒤, 추정기가 실제로 무엇으로
                         구성됐는지. 보고서가 인용하는 것은 제안이 아니라 이것이다.
``internal_validation``  적합에서 빠진 학습 행. 추정기 자신이 뺐거나 harness가 뺐거나
                         (:func:`describe_internal_validation`, :func:`fit_estimator`) 결정 컷이
                         뺐다. ``fit_rows``가 이 시도가 실제로 적합된 행 수이고, 뺀 것이 없으면
                         필드가 없다.
``metrics``              예측이 지원하는, 타깃 태스크의 모든 registry 지표와 각각의 ``train_``
                         짝, 목표 지표에 대한 ``train_val_gap``, 그리고 이진 타깃에서는
                         ``specificity``·``balanced_accuracy_at_best_cut``·
                         ``balanced_accuracy_cut_headroom``. 이 넷은 진단값이고 아무것도 이것을
                         목표로 삼을 수 없다 (:mod:`automl_agent.capabilities`).
``*_ci_low`` / ``_high`` 방금 채점한 분할에 대한 목표 지표의 95% 부트스트랩 구간. 분할이 그룹
                         단위였으면 재표집도 그룹 단위다. float 둘인 것은
                         ``privacy.public_result``가 수치 지표 값만 남기기 때문이다. 폭이 무엇을
                         뜻하는지는 :mod:`automl_agent.scoring.intervals`.

학습 실패(OOM 포함)에도 종료 코드는 0이다. orchestrator가 그것을 정상 흐름으로 다루고 Critic에게
넘길 수 있도록. 0이 아닌 종료는 일부러 시뮬레이션한 하드 크래시 하나뿐이고, 그것은 orchestrator가
죽는 subprocess를 넘기고 산다는 것을 증명하려고 있다.

config 모양
-----------
    {
      "model": "hist_gbdt",
      "hyperparams": {"max_iter": 200, "learning_rate": 0.1},
      "preprocessing": {"impute": "median", "scale": true},
      "target_missing": {"policy": "reject" | "drop"},
      "data": {"path": null, "target_column": "target",
               "synthetic": {"n_samples": 5000, "n_features": 20, ...}},
      "metric": "f1",
      "seed": 42,
      "memory_limit_mb": 512,
      "bootstrap": {"resamples": 400},
      "paired_baseline": {"iteration": 3, "path": ".../iter_03/val_predictions.npz"},
      "simulate": {"fail": "oom" | "crash", "sleep_sec": 0}
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 학습 노드가 이 파일을 패키지 모듈이 아니라 경로로 실행하니 상대 import가 불가능하고, repo
# 루트를 먼저 sys.path에 얹어야 한다. registry를 공유하는 것이 요점이다: 목표가 유도되는 지표와
# 이 스크립트가 내는 지표가 손으로 맞춘 두 목록이 아니라 같은 하나의 목록이 된다. 아래 import들에
# 붙은 ``E402`` 무시도 모두 이 수정 때문이다.
if __package__ in (None, ""):  # pragma: no cover - 파일로 실행할 때만
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.config import (  # noqa: E402
    DECISION_FILENAME,
    MODEL_FILENAME,
    PREDICTIONS_FILENAME,
    SCHEMA_FILENAME,
    file_size_text,
)
from automl_agent.dataset.features import (  # noqa: E402
    append_missing_count,
    append_missing_indicator,
    build_schema,
    describe_drift,
    describe_encoding,
    encode_features,
    encode_with_schema,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    APPENDING_STEPS as PIPELINE_APPENDING_STEPS,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    STEP_IMPUTE as PIPELINE_STEP_IMPUTE,
)
from automl_agent.dataset.pipeline import (  # noqa: E402
    STEP_SCALE as PIPELINE_STEP_SCALE,
)
from automl_agent.dataset.pipeline import build_steps  # noqa: E402
from automl_agent.dataset.targets import (  # noqa: E402
    DEFAULT_TARGET_MISSING_POLICY,
    detect_task,
    encode_target,
    target_classes,
)
from automl_agent.scoring.calibration import (  # noqa: E402
    measure as calibration_measure,
)
from automl_agent.scoring.intervals import (  # noqa: E402
    DEFAULT_RESAMPLES,
    PAIRED_KEY,
    PAIRED_MEASURED,
    PAIRED_SKIPPED,
    as_number,
    bootstrap_interval,
    describe_interval,
    describe_paired,
    paired_delta,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES as METRIC_ALIASES,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    DEFAULT_METRICS,
    METRICS,
    MINIMIZE,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    canonical,
    direction_of,
    substitute_metric,
)
from automl_agent.scoring.ranking import (  # noqa: E402
    best_cut_ceiling,
    ks_statistic,
)
from automl_agent.scoring.splits import (  # noqa: E402
    describe_protocol,
    protocol,
    split_three_way,
    val_fingerprint,
)
from automl_agent.threads import (  # noqa: E402
    describe_thread_state,
    thread_state,
    thread_state_changed,
)

LOG_TAIL_CHARS = 4000

# 이 크기를 넘으면 저장된 모델이 로그에 자기 줄을 받는다. 거절은 아니다: 파일을 쓸 때쯤이면
# 점수는 이미 얻은 것이고, 잘 채점하고 나서 자기 모델을 지운 실행이 더 나쁜 실패다. ``save_model``
# 참고.
LARGE_MODEL_BYTES = 100 * 1024**2

# 모든 시도가 *학습 분할* 짝을 함께 보고하는 지표 둘, 태스크마다. 목표 지표가 무엇이든 Critic이
# 과소적합과 과적합을 구분할 수 있도록. 양쪽 다 유계 지표 하나와 목표가 보통 그것으로 적히는
# 지표 하나다.
TRAIN_METRICS: dict[str, tuple[str, ...]] = {
    TASK_CLASSIFICATION: ("f1", "accuracy"),
    TASK_REGRESSION: ("r2", "mae"),
}

# 이 랭킹의 가장 좋은 컷, 그리고 얻은 점수가 그것에서 얼마나 떨어져 있는지. 호출처 한 곳에
# 리터럴로 쓰지 않고 여기서 이름 짓는 것은, 능력 목록이 랭킹 부족과 운영점 부족을 구분하는 방법으로
# 계획하는 쪽에 두 이름을 인용하기 때문이다 — 테스트가 이 튜플에 묶어 둔 ``capabilities._LEVER_AXES``
# 참고. 둘 다 목표로 삼을 수 없다: 실행기가 적용하지 않을 컷에서 재는 값이다.
CUT_DIAGNOSTICS: tuple[str, str] = ("balanced_accuracy_at_best_cut", "balanced_accuracy_cut_headroom")


class LogBuffer:
    """진행 줄을 모은다. 꼬리는 result.json 안에 실려 돌아간다.

    ``echo=False``는 찍지 않고 모은다. ``scripts/predict.py``는 라벨 붙은 배치를 채점하면서
    ``evaluate_split``의 건너뛴 이유("roc_auc skipped: only one class present")를 위에 흩뿌리지
    않고 그것이 설명하는 점수 옆에 두고 싶어 하므로, stdout으로 보내지 않고 나중에 :attr:`lines`를
    읽는다.
    """

    def __init__(self, echo: bool = True) -> None:
        self.lines: list[str] = []
        self._echo = echo

    def write(self, message: str) -> None:
        self.lines.append(message)
        if self._echo:
            print(message, flush=True)

    def tail(self, limit: int = LOG_TAIL_CHARS) -> str:
        return "\n".join(self.lines)[-limit:]


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #


def goal_metric(cfg: dict[str, Any], task: str, log: LogBuffer | None = None) -> str:
    """이 시도가 조종당하는 지표: 설정된 것, 아니면 이 태스크의 기본값.

    닿을 수 있는 어긋남이 둘이고 둘 다 위쪽에서 잡히지 않는다: ``RunConfig``는 프로파일링이 타깃
    열을 알기 전에 이름을 registry에 대고만 검사하고, orchestrator의 대체
    (:func:`automl_agent.scoring.goal.resolve_goal`)는 *카드*를 따르는데 이 스크립트는 열을 따른다 —
    그래서 분류 열 위에 ``regression``을 선언한 카드는 일관돼 보인다.

    그냥 두면 피해가 흩어지고 조각마다 다른 것처럼 보인다: 부트스트랩 구간 건너뜀("split too
    small, or metric degenerate"), 짝지은 비교가 ``degenerate``로 건너뜀, 아무도 고르지 않은
    지표에서 잰 과적합 격차, 어떤 라벨 배치로도 채점할 수 없는 목표 지표를 기록한 스키마. 여기서
    한 번 대체하면 넷 모두 한 숫자에 머문다.

    ``f1``이 아니라 이 태스크의 기본값으로 떨어진다 — 옛 리터럴 자체가 분류 지표였으니, 회귀
    타깃에서는 그것이 같은 어긋남이었다.
    """
    configured = canonical(str(cfg.get("metric") or "")) or DEFAULT_METRICS[task]
    swap = substitute_metric(task, configured)
    if swap is None:
        return configured if configured in METRICS else DEFAULT_METRICS[task]
    if log is not None:
        log.write(
            f"metric {configured} is not defined for a {task} target: steering by {swap} "
            f"instead, and recording {swap} in the schema so a labelled batch is scored on "
            "the same number"
        )
    return swap


def load_data(
    cfg: dict[str, Any], log: LogBuffer, schema: dict[str, Any] | None = None
) -> tuple[Any, Any, int, Any, str, dict[str, Any] | None]:
    """CSV 경로에서 ``(X, y, n_classes, groups, task, schema)``를 돌려주거나, 합성한다.

    ``path``가 없으면 카드가 선언한 모양에서 합성한다. 그래야 원본 데이터 없이도 파이프라인을 끝에서
    끝까지 돌릴 수 있다.

    ``groups``  돌려주는 행에 맞춘 ``data.group_column``의 값. **피처에서 먼저 뺀다** — 추정기가
                읽을 수 있는 식별자는 라벨로 가는 지름길이고, 수치형이면 그러지 않으면
                :func:`encode_features`를 피처로 통과한다.
    ``task``    :func:`automl_agent.dataset.targets.detect_task`가 열에서 읽는다. 프로파일러가 쓴
                것과 같은 함수라, 두 번 듣지 않고도 두 프로세스가 합의한다. 회귀에서
                ``n_classes``는 ``0``이다.
    ``schema``  ``X``를 낸 인코딩. 모델 옆에 저장하려고 돌려준다. 합성 경로에서는 ``None``.

    **``schema``를 넣으면 방향이 뒤집힌다**: 파일이 자기 혼자 함축하는 배치가 아니라 *그* 배치로
    인코딩된다. 저장된 모델을 ``--score-model`` 경로에서 채점할 수 있게 만드는 것이 이것이다 —
    대안은 오류를 내지 않고 조용히 어긋난다.
    """
    import numpy as np

    data = dict(cfg.get("data") or {})
    path = data.get("path")
    seed = int(cfg.get("seed", 42))
    group_column = data.get("group_column")
    declared_task = str(cfg.get("task") or TASK_CLASSIFICATION)

    if path:
        import pandas as pd

        target = str(data.get("target_column") or "target")
        frame = pd.read_csv(path)
        if target not in frame.columns:
            raise ValueError(f"target_column {target!r} not found in {path}")
        y_series = frame[target]
        group_series = None
        drop = [target]
        if group_column:
            group_column = str(group_column)
            if group_column not in frame.columns:
                raise ValueError(f"group_column {group_column!r} not found in {path}")
            if group_column == target:
                raise ValueError(f"group_column {group_column!r} is the target column")
            group_series = frame[group_column]
            drop.append(group_column)
        # 프로파일러가 기준선을 재기 전에 적용한 것과 같은 인코딩, 같은 함수에서 — 한 피처
        # 행렬에서 유도한 바와 다른 행렬에서 잰 점수는 비교 가능하지 않다.
        raw_features = frame.drop(columns=drop)
        if schema is None:
            features, encoding = encode_features(raw_features)
            log.write(describe_encoding(encoding))
            fitted_schema: dict[str, Any] | None = build_schema(raw_features)
        else:
            features, drift = encode_with_schema(raw_features, schema)
            for line in describe_drift(drift):
                log.write(line)
            fitted_schema = dict(schema)
        if features.shape[1] == 0:
            raise ValueError("no usable feature columns remain after dropping the target")
        # 카드가 기록한 것과 같은 정책. 그래야 프로파일러의 기준선과 이 점수가 같은 행에서
        # 재어진다. NaN이 자기 클래스가 되는 일은 없다.
        policy = str((cfg.get("target_missing") or {}).get("policy") or DEFAULT_TARGET_MISSING_POLICY)
        task = detect_task(y_series)
        if task != declared_task:
            # 치명적이지 않다: 권위는 열에 있고 카드는 그것에 대한 기술이다. 한 줄 적을 값은
            # 있다 — 카드 아래에서 파일이 바뀌었고, 바가 나온 기준선이 이 열의 다른 읽기에서
            # 재어졌다는 뜻이니까.
            log.write(
                f"target column reads as {task}, but the config declares {declared_task}; "
                "following the column"
            )
        codes, keep, n_missing = encode_target(y_series, policy, task)
        if n_missing:
            features = features[keep]
            if group_series is not None:
                group_series = group_series[keep]
            log.write(f"dropped {n_missing} rows with a missing target (policy={policy})")
        log.write(f"loaded {path}: {features.shape[0]} rows x {features.shape[1]} encoded features")
        groups = None
        if group_series is not None:
            # 텍스트로. 같은 환자의 float id와 int id가 한 그룹이 되고, NaN이 NaN마다 한 행인
            # 그룹으로 조용히 갈라지지 않도록.
            groups = group_series.astype("string").fillna("<missing>").to_numpy()
            log.write(f"grouped by {group_column}: {len(set(groups.tolist()))} distinct groups")
        y_codes = codes.to_numpy()
        x_arr = np.asarray(features.to_numpy(), dtype="float64")
        goal = goal_metric(cfg, task, log)
        if fitted_schema is not None:
            # ``predict``가 필요로 하지만 추정기 안에는 없는 것 전부: 행렬의 어느 열이 어느
            # 것이었는지, 거기서 나오는 수가 무엇을 뜻하는지, 그리고 적합이 피처로 본 적 없어서
            # 파일이 공급하게 두면 안 되는 열이 어느 것인지.
            fitted_schema.update(
                {
                    "task": task,
                    "target_column": target,
                    "group_column": group_column,
                    "classes": target_classes(y_series, task),
                    "n_features": int(x_arr.shape[1]),
                    "seed": seed,
                    # 이 실행이 조종당한 지표. 나중의 라벨 배치가 채점하는 사람이 마침 고른
                    # 것이 아니라 이 실행이 최적화한 그 숫자로 채점되도록. 이것이 없으면
                    # "홀드아웃 0.71, 여기 0.64"가 서로 다른 두 측정일 수 있다.
                    "metric": goal,
                }
            )
        return (
            x_arr,
            y_codes,
            0 if task == TASK_REGRESSION else int(len(set(y_codes.tolist()))),
            groups,
            task,
            fitted_schema,
        )

    syn = dict(data.get("synthetic") or {})
    n_samples = int(syn.get("n_samples", 5000))
    n_features = int(syn.get("n_features", 20))
    n_informative = int(syn.get("n_informative", max(2, n_features // 2)))
    n_informative = min(n_informative, n_features)

    if declared_task == TASK_REGRESSION:
        from sklearn.datasets import make_regression

        x_arr, y_arr = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            # ``flip_y``의 회귀 쪽 짝: 줄일 수 없는 오차. 그래야 합성 실행이 r2=1.0에 닿지
            # 못하고 루프에 실제로 수렴할 것이 생긴다.
            noise=float(syn.get("noise", 10.0)),
            random_state=seed,
        )
        log.write(
            f"synthesised regression dataset from card: {n_samples} rows x {n_features} "
            f"features, noise={syn.get('noise', 10.0)}"
        )
        return x_arr, y_arr, 0, None, TASK_REGRESSION, None

    from sklearn.datasets import make_classification

    n_classes = int(syn.get("n_classes", 2))
    weights = syn.get("class_weights")
    x_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, min(n_features - n_informative, int(syn.get("n_redundant", 2)))),
        n_classes=n_classes,
        n_clusters_per_class=int(syn.get("n_clusters_per_class", 2)),
        weights=list(weights) if weights else None,
        class_sep=float(syn.get("class_sep", 0.8)),
        flip_y=float(syn.get("flip_y", 0.03)),
        random_state=seed,
    )
    log.write(
        f"synthesised dataset from card: {n_samples} rows x {n_features} features, "
        f"{n_classes} classes, class_sep={syn.get('class_sep', 0.8)}, flip_y={syn.get('flip_y', 0.03)}"
    )
    return x_arr, y_arr, n_classes, None, TASK_CLASSIFICATION, None


def guard_memory(x_arr: Any, cfg: dict[str, Any], log: LogBuffer) -> None:
    """작업 집합이 선언된 예산을 넘을 때 ``MemoryError``를 던진다.

    실제 GPU 실행이라면 여기서 CUDA OOM이 드러날 것이다. CPU/표 데이터에서는 대신 명시적 예산을
    강제한다. 그래야 ``oom`` 분기가 가정이 아니라 닿을 수 있고 시험할 수 있는 것이 된다.
    ``batch_size``가 추정치를 줄이는데, Critic의 "reduce batch size" 조언이 실제로 값을 하게
    만드는 것이 그것이다.
    """
    limit_mb = cfg.get("memory_limit_mb")
    if not limit_mb:
        return
    rows, cols = int(x_arr.shape[0]), int(x_arr.shape[1])
    hyperparams = dict(cfg.get("hyperparams") or {})
    batch = hyperparams.get("batch_size")
    working_rows = min(rows, int(batch)) if isinstance(batch, int) and batch > 0 else rows
    # 트리 앙상블과 MLP는 작업 집합의 복사본을 여러 개 들고 있다.
    copies = 4 if cfg.get("model") in {"random_forest", "extra_trees", "gradient_boosting"} else 2
    if str(hyperparams.get("precision", "")).lower() in {"fp16", "float16", "half"}:
        copies = max(1, copies // 2)
    estimate_mb = working_rows * cols * 8 * copies / (1024 * 1024)
    log.write(f"memory estimate {estimate_mb:.1f} MB vs budget {float(limit_mb):.1f} MB")
    if estimate_mb > float(limit_mb):
        raise MemoryError(
            f"out of memory: estimated {estimate_mb:.1f} MB exceeds budget {float(limit_mb):.1f} MB"
        )


# --------------------------------------------------------------------------- #
# 모델 registry
# --------------------------------------------------------------------------- #

# LLM이 쓸 법한 하이퍼파라미터 이름을 sklearn 이름에 대응시킨다.
ALIASES: dict[str, dict[str, str]] = {
    "hist_gbdt": {"n_estimators": "max_iter", "reg_lambda": "l2_regularization"},
    "mlp": {"learning_rate": "learning_rate_init", "hidden_size": "hidden_layer_sizes"},
    "logreg": {"reg_strength": "C", "epochs": "max_iter"},
    "xgboost": {"lr": "learning_rate", "l2": "reg_lambda"},
    # Ridge의 정규화는 LogisticRegression의 것과 *반대* 방향이다: alpha는 벌점이고 C는 그
    # 역수다. 그래서 ``reg_strength``가 여기서는 ``alpha``로, 거기서는 ``C``로 간다. 어느 어휘로
    # 쓴 제안이든 맞는 손잡이에 내린다.
    "ridge": {"reg_strength": "alpha", "l2": "alpha", "lambda": "alpha"},
    "elasticnet": {"reg_strength": "alpha", "l1_ratio": "l1_ratio"},
}

SCALE_SENSITIVE = {"logreg", "mlp", "svc", "knn", "ridge", "linreg", "elasticnet", "svr"}

# NaN을 자기 분기로 흘려보내는 계열. 그래서 대치를 건너뛸 수 있는 것은 이들뿐이다. 자유 선택이
# 아니라 계열 검사인 것은, 선형 모델에 ``impute: none``은 설정이 아니라 ``fit``에서의 크래시이기
# 때문이고 — 또 대치기가 한때 무조건이었기 때문이다. 그때는 NaN 기본 처리를 근거로 삼은 모든 계획이
# 자기가 무엇을 받을지에 대해 틀렸다.
NATIVE_NAN = {"hist_gbdt", "xgboost"}

# 추정기에는 유효하지만 이 harness의 ``fit`` 시점에서 깨지는 생성자 인자. ``set_params``는
# 받아들이고 실행은 나중에 죽는다.
#
# ``early_stopping_rounds``가 한때 이 목록에 있었다. Pipeline이 ``eval_set``을 전달할 수 없기
# 때문이다. 버리면 크래시는 멈췄지만 더 나쁜 문제가 남았다: xgboost가 늘 ``n_estimators`` 끝까지만
# 돌 수 있었고, 그렇게 생긴 손실이 ``overfitting``으로 보고됐다 — harness의 성질이 아니라 계열의
# 성질인 것처럼 (``test-1`` 반복 2에서 또). 지금은
# :func:`fit_estimator`가 eval set을 대신 공급하니 파라미터가 존중되고, 만들 수 없는 두 객체만
# 거절로 남는다: 이 스크립트가 분할하지 않은 행을 지목하는 eval set, 그리고 도착할 JSON 형태가 없는
# 콜백 목록.
UNSUPPORTED_BY_HARNESS: dict[str, tuple[str, ...]] = {
    "xgboost": ("eval_set", "callbacks"),
}

# ``early_stopping``은 양쪽이 다 받는 키인데 문자열로 받는 것은 한쪽뿐이다. sklearn의
# HistGradientBoosting 짝은 ``'auto'``를 행 수에 대고 해석하고, MLP의 제약은 boolean이라 같은 표기가
# ``fit`` 안에서 raise한다 — 파이프라인을 세우고 메모리 가드를 통과한 뒤라, 반복 하나를 다 쓴다.
# 프롬프트가 ``'auto'``를 인용하므로(``capabilities``가 자기가 정하는 경계를 렌더한다) 계획이 그 값을
# 되받는 것은 평범한 일이고, traceback이 아니라 ``dropped_hyperparams``에 속한다. 키 단위 목록으로는
# 이것을 표현할 수 없어서 자기 이름을 갖는다.
STRING_EARLY_STOPPING = frozenset({"hist_gbdt"})

# 뭐라도 하려면 *목적함수*가 이진이어야 하는 손잡이들. xgboost는 ``multi:softprob``에 대고
# ``scale_pos_weight``를 받아들이고 나서 stderr에 "Parameters: { "scale_pos_weight" } are not
# used"라고 말한다 — 그러면서 ``get_params``는 여전히 그것을 보고하니, 이것이 없으면 시도가 그것을
# ``applied_hyperparams``에 기록하고 계획이 움직인 적 없는 컷을 자기 공으로 돌릴 수 있다. registry의
# 주석이 들어오는 길에서 같은 말을 한다.
BINARY_ONLY_PARAMS = frozenset({"scale_pos_weight"})


def weight_map(value: dict[Any, Any], labels: Sequence[Any] | None) -> dict[int, float] | None:
    """sklearn이 받아들일 ``class_weight`` 매핑, 받아들이지 않을 것이면 ``None``.

    수리가 둘인데 둘 다 한때 치명적이거나 조용했다:

    * **JSON에는 정수 키가 없다.** config는 ``train_config.json``을 거쳐 이 스크립트에 닿으니
      ``{0: 1, 1: 10}``이 ``{"0": 1, "1": 10}``으로 도착하고, sklearn은 정수 라벨에 대고 문자열
      키를 거절한다.
    * **데이터에 없는 라벨은 ``fit`` 안에서 raise한다.** 파이프라인을 세우고 메모리 가드를 통과한
      뒤라, 반복 하나를 다 쓴다. 있는 라벨마다 정확히 하나의 가중치를 요구하면 그것이 평범하게
      버려진 하이퍼파라미터로 바뀐다.

    키는 클래스 *코드*다: ``targets.encode_target``이 날 라벨이 아니라 학습 카테고리 코드
    (0..n_classes-1, 카드의 ``class_balance`` 순서)를 건넨다.
    """
    if not value:
        return None
    clean: dict[int, float] = {}
    for raw_code, raw_weight in value.items():
        if isinstance(raw_code, bool) or isinstance(raw_weight, bool):
            return None
        try:
            code = int(str(raw_code).strip())
            weight = float(raw_weight)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(weight) or weight <= 0 or code in clean:
            return None
        clean[code] = weight
    if labels is not None and set(clean) != {int(label) for label in labels}:
        return None
    return clean


# LLM 표기 → registry 키, 태스크별. 하나가 아니라 둘인 것은 한쪽에서는 이름이 아무것도 뜻하지 않을
# 수 있기 때문이다: 연속형 타깃에 대고 "logreg"는 조용히 Ridge로 고쳐 줄 오타가 아니라 *거절*해야
# 하는 제안이다. 그래야 실수가 Critic에게(그리고 model_selection 노드의 다음 선택에) 닿는다.
# 그러지 않으면 대체 뒤에 숨고, 보고서는 그것을 요청받은 것이라고 기술하게 된다.
MODEL_ALIASES: dict[str, dict[str, str]] = {
    TASK_CLASSIFICATION: {
        "logistic_regression": "logreg",
        "logisticregression": "logreg",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "hist_gradient_boosting": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
        "lightgbm": "hist_gbdt",  # 설치돼 있지 않다. 검증된 것 중 가장 가까운 대응물
        "xgb": "xgboost",
        "neural_net": "mlp",
        "mlp_classifier": "mlp",
    },
    TASK_REGRESSION: {
        "linear_regression": "linreg",
        "linearregression": "linreg",
        "ols": "linreg",
        "ridge_regression": "ridge",
        "ridgeregression": "ridge",
        "elastic_net": "elasticnet",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "hist_gradient_boosting": "hist_gbdt",
        "histgradientboosting": "hist_gbdt",
        "lightgbm": "hist_gbdt",
        "xgb": "xgboost",
        "neural_net": "mlp",
        "mlp_regressor": "mlp",
        "mlpregressor": "mlp",
        "svm": "svr",
    },
}


def _classifier(key: str, seed: int) -> Any:
    """이 키가 지목하는 분류기, 지목하는 것이 없으면 ``None``."""
    if key == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed)
    if key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=seed)
    if key == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=seed)
    if key == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier()
    if key == "svc":
        from sklearn.svm import SVC

        return SVC(random_state=seed)
    if key == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(max_iter=400, random_state=seed)
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - extra가 없을 때만 닿는다
            raise ValueError(f"model 'xgboost' is unavailable: {exc}") from exc

        return XGBClassifier(
            n_estimators=300, tree_method="hist", eval_metric="logloss", random_state=seed
        )
    return None


def _regressor(key: str, seed: int) -> Any:
    """이 키가 지목하는 회귀기, 지목하는 것이 없으면 ``None``.

    계열이 양쪽에 다 있는 곳에서는 일부러 :func:`_classifier`와 같은 키를 쓴다 — ``hist_gbdt``는
    양쪽에서 같은 단어이므로, 한 카드에 대고 쓴 계획이 다른 카드에 대고도 같게 읽히고,
    ``capabilities``는 어휘 하나를 공표한다.
    """
    if key == "linreg":
        from sklearn.linear_model import LinearRegression

        # ``random_state`` 없음: 닫힌 해에는 seed를 줄 것이 없다.
        return LinearRegression()
    if key == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed)
    if key == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(random_state=seed)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if key == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed)
    if key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed)
    if key == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(random_state=seed)
    if key == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor()
    if key == "svr":
        from sklearn.svm import SVR

        return SVR()
    if key == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(max_iter=400, random_state=seed)
    if key == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - extra가 없을 때만 닿는다
            raise ValueError(f"model 'xgboost' is unavailable: {exc}") from exc

        return XGBRegressor(n_estimators=300, tree_method="hist", random_state=seed)
    return None


def build_estimator(
    name: str,
    hyperparams: dict[str, Any],
    seed: int,
    log: LogBuffer,
    preprocessing: dict[str, Any] | None = None,
    labels: Sequence[Any] | None = None,
    task: str = TASK_CLASSIFICATION,
    declared: list[tuple[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """이름이 지목하는 모델을 만들고, 그것이 실제로 받는 파라미터만 적용한다.

    모르는 키는 raise하지 않고 버리며 로그에 남긴다: LLM이 제안하고, 이 코드가 검증한다.

    ``(pipeline, applied, dropped)``를 돌려준다 — 기록이 제안을 인용하는 대신 실제로 돈 것을 그것이
    돈 sklearn 이름으로 말할 수 있도록.

    ``task``가 이름을 어느 추정기 계열에서 해석할지 고른다. 다른 쪽의 이름은 raise하고,
    :func:`classify_exception`이 그것을 ``unsupported_model``로 바꾼다 — 연속형 타깃에 조용히 적합된
    분류기와 달리 진단할 수 있다.

    **``declared``가 있으면 ``preprocessing``을 아예 무시한다** — 한 파이프라인을 기술하는 면이 둘이면
    어느 쪽이 돌았는지 말할 수 있는 것이 없어진다. 안전망 둘은 그래도 적용된다. 명세를 쓰는 것은 잊을
    수 있는 것이기 때문이다 (:func:`_wrap_declared`).
    """
    key = resolve_model_key(name, task)
    model = _regressor(key, seed) if task == TASK_REGRESSION else _classifier(key, seed)
    if model is None:
        raise ValueError(f"unsupported model {name!r} for a {task} target")

    mapped: dict[str, Any] = {}
    dropped: list[str] = []
    accepted = model.get_params(deep=False)
    blocked = UNSUPPORTED_BY_HARNESS.get(key, ())
    for raw_key, value in hyperparams.items():
        param = ALIASES.get(key, {}).get(raw_key, raw_key)
        if param in blocked:
            dropped.append(raw_key)
            continue
        if param == "class_weight" and isinstance(value, dict):
            repaired = weight_map(value, labels)
            if repaired is None:
                log.write(f"dropped class_weight={value!r}: not one positive weight per class")
                dropped.append(raw_key)
                continue
            value = repaired
        if param == "early_stopping" and isinstance(value, str) and key not in STRING_EARLY_STOPPING:
            log.write(
                f"dropped early_stopping={value!r}: {key} takes this as a boolean only, and "
                "the string would raise inside fit"
            )
            dropped.append(raw_key)
            continue
        if param in BINARY_ONLY_PARAMS and labels is not None and len(set(labels)) > 2:
            log.write(
                f"dropped {raw_key}={value!r}: {len(set(labels))} classes, and this lever only "
                "acts on a binary objective"
            )
            dropped.append(raw_key)
            continue
        if param in accepted:
            if param == "hidden_layer_sizes" and isinstance(value, list):
                value = tuple(value)
            mapped[param] = value
        else:
            dropped.append(raw_key)
    if mapped:
        try:
            model.set_params(**mapped)
        except (ValueError, TypeError) as exc:
            log.write(f"rejected hyperparams {mapped}: {exc}; falling back to defaults")
            mapped = {}
    if dropped:
        log.write(f"dropped hyperparams not applicable to {key}: {sorted(dropped)}")
    log.write(f"estimator={key} applied_hyperparams={mapped}")
    built = (
        _wrap_preprocessing(key, model, preprocessing or {}, log)
        if declared is None
        else _wrap_declared(key, model, declared, log)
    )
    return built, mapped, sorted(dropped)


def resolve_model_key(name: str, task: str) -> str:
    """모델 이름이 alias 적용 뒤에 해석되는 registry 키.

    자기 함수인 것은 답이 필요한 호출자가 둘이고 그중 추정기를 세우는 것은 하나뿐이기 때문이다:
    :func:`run_training`은 ``pipeline`` 명세를 해석하기 전에 계열이 NaN을 기본으로 분기하는지 알아야
    하고, 그것이 같은 조회다. 거기서 다시 적으면 그렇게 둘이 갈라진다.
    """
    key = name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(task, {}).get(key, key)


# config가 SimpleImputer에게 요청할 수 있는 것. 계획 쪽도 이것을 검증한다
# (``nodes/training.py::preprocessing_block``). 여기서 반복하는 것이 손으로 쓴 config에 대고 이
# 스크립트를 돌려도 안전하게 만든다.
IMPUTE_STRATEGIES = ("median", "mean", "most_frequent")
DEFAULT_IMPUTE = "median"
# SimpleImputer의 전략이 아니다 — 단계를 아예 없앤다. :data:`NATIVE_NAN`에서만 존중된다. 그래야
# 카드가 어느 행이 측정됐는지를 지우는 중앙값 대신, 트리가 분기할 줄 아는 NaN을 그대로 건넬 수 있다.
IMPUTE_NONE = "none"
# 대치기가 ``pipeline`` 명세로 세운 ``ColumnTransformer``일 때 ``applied_preprocessing.impute``가
# 하는 말 — 이름 붙일 전략 하나가 없다.
PER_COLUMN_IMPUTE = "per_column"

# 결측이 될 수 있는 열 둘. 둘 다 대치기보다 *먼저* 돈다. 그것이 요점 전부다: 그 뒤에는 표시할 것이
# 남아 있지 않다. 변환기 자체는
# :func:`automl_agent.dataset.features.append_missing_indicator`와 그 짝이다 — 이 모듈에 정의하지
# 않은 것은 일부러다. 이 모듈은 ``__main__``으로 돌고, 피클된 파이프라인은 그것을 함수의 집으로
# 기록하기 때문이다. 거기 주석 참고.
MISSING_INDICATOR = "missing_indicator"
MISSING_COUNT = "missing_count"
# 정식 이름 외에 실행기가 응답하는 이름들. sklearn은 파라미터를 ``add_indicator``, 클래스를
# ``MissingIndicator``라고 부르니 계획이 어느 이름을 집을지는 동전 던지기에 가깝고, 못 알아본 키는
# 조용히 버려진다 — 그러면 계획이 존재하지 않는 열을 기술하게 된다. 하이퍼파라미터에 대한
# :data:`ALIASES`와 같은 목적이다.
PREPROCESSING_ALIASES: dict[str, str] = {
    "add_missing_indicators": MISSING_INDICATOR,
    "add_missing_indicator": MISSING_INDICATOR,
    "missing_indicators": MISSING_INDICATOR,
    "add_indicator": MISSING_INDICATOR,
    "missing_counts": MISSING_COUNT,
    "n_missing": MISSING_COUNT,
}


def _wrap_preprocessing(
    key: str, model: Any, preprocessing: dict[str, Any], log: LogBuffer
) -> Any:
    """고정된 파이프라인이 아니라 *config*가 기술하는 파이프라인을 세운다.

    config는 한때 아티팩트에 적히고 나서 여기서 무시됐다. 그래서 중앙값으로 대치한 실행에 대해
    보고서가 "mean imputation"이라고 말할 수 있었다. 없는 키는 옛 기본값을 유지한다 — 중앙값, 그리고
    추정기가 필요로 하는 곳에서만 스케일링 — 그래서 아무 말도 하지 않는 config는 전과 똑같이
    행동한다.

    ``impute: none``은 대치기를 뺀다. 유효성이 ``key``에 달린 설정은 이것 하나다: ``fit``에서 NaN을
    넘기는 것은 :data:`NATIVE_NAN` 계열뿐이다. 다른 데서 요청하면 시도를 실패시키는 대신 기본값으로
    낮춘다. 계열에 대해 잘못 짚은 계획은 반복 하나가 아니라 로그 한 줄을 값으로 내야 한다. 낮춘
    것을 로그에 남기는 이유는 적용된 하이퍼파라미터를 기록하는 이유와 같다 — 그러지 않으면 돌지
    않은 전처리를 기술하는 보고서가 나온다.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    preprocessing = {
        PREPROCESSING_ALIASES.get(str(name), str(name)): value
        for name, value in preprocessing.items()
    }
    requested = preprocessing.get("impute")
    strategy = DEFAULT_IMPUTE if requested is None else str(requested)
    if strategy == IMPUTE_NONE and key not in NATIVE_NAN:
        log.write(
            f"impute={IMPUTE_NONE} needs a family that splits on NaN "
            f"({', '.join(sorted(NATIVE_NAN))}); {key} does not, using {DEFAULT_IMPUTE}"
        )
        strategy = DEFAULT_IMPUTE
    elif strategy not in (*IMPUTE_STRATEGIES, IMPUTE_NONE):
        log.write(f"unknown impute strategy {requested!r}; using {DEFAULT_IMPUTE}")
        strategy = DEFAULT_IMPUTE

    # 명시적 ``scale``이 정한다. 없으면 전처럼 추정기가 정한다.
    scale = preprocessing.get("scale")
    should_scale = bool(scale) if isinstance(scale, bool) else key in SCALE_SENSITIVE

    steps: list[tuple[str, Any]] = []
    # 대치기보다 먼저, 이 순서로. 계수는 지시자 뒤에 돌아서 원래 열만 센다 — 덧붙은 지시자에는
    # NaN이 없으니 합은 어느 쪽이든 같지만, 순서가 그 독립성을 우연이 아니라 명시로 만든다.
    indicate = bool(preprocessing.get(MISSING_INDICATOR))
    count = bool(preprocessing.get(MISSING_COUNT))
    if indicate:
        steps.append((MISSING_INDICATOR, FunctionTransformer(append_missing_indicator)))
    if count:
        steps.append((MISSING_COUNT, FunctionTransformer(append_missing_count)))
    if strategy != IMPUTE_NONE:
        steps.append(("impute", SimpleImputer(strategy=strategy)))
    if should_scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    log.write(
        f"preprocessing applied: impute={strategy} scale={should_scale} "
        f"{MISSING_INDICATOR}={indicate} {MISSING_COUNT}={count}"
    )
    return Pipeline(steps)


def declared_steps(
    cfg: dict[str, Any],
    schema: dict[str, Any] | None,
    width: int,
    task: str,
    log: LogBuffer,
) -> tuple[list[tuple[str, Any]] | None, list[str]]:
    """이 config의 ``pipeline`` 명세에 대한 ``(steps, applied)``, 명세가 없으면 ``(None, [])``.

    ``None``은 ``[]``와 같은 답이 아니다. 없음은 *플래그 경로를 쓰라*는 뜻이고, 이 블록보다 먼저 쓰인
    모든 config가 뜻하는 것이 그것이며 그것이 그 config들의 채점을 같게 유지한다. 빈 목록은 명세가
    주어졌고 그 안에서 살아남은 것이 없다는 뜻이며, 그러면 선언 경로가 안전망만 걸친 채로 돈다. 둘을
    합치면 모르는 단계로 된 명세가 자기가 대체하려고 쓰인 플래그를 조용히 물려받게 된다.

    열 이름은 적합된 스키마에서 온다. 단계가 지목하는 이름이 그것이기 때문이다. 합성 경로에는 스키마가
    없고 — 인코딩된 것이 없으니 이름 붙일 것도 없다 — 대신 위치 이름을 받는다: 행렬 전체를 다루는
    단계로 된 명세는 거기서도 돌고, 실제 열을 지목하는 명세는 열마다 이 실행에는 그런 것이 없다는 말을
    듣는다.
    """
    spec = cfg.get("pipeline")
    if not spec:
        return None, []
    columns = [str(name) for name in (schema or {}).get("columns") or []]
    if not columns:
        columns = [f"x{position}" for position in range(int(width))]
        log.write(
            f"pipeline: this run encoded no columns (synthetic data), so steps can only "
            f"address all {len(columns)} of them or none by name"
        )
    key = resolve_model_key(str(cfg.get("model") or "hist_gbdt"), task)
    steps, _names, applied = build_steps(spec, columns, log, native_nan=key in NATIVE_NAN)
    return steps, applied


def base_step_name(name: str) -> str:
    """``missing_count_2`` -> ``missing_count``. 반복 번호를 뗀 단계의 종류.

    ``split("_")``이 아니라 끝의 ``_<숫자>``에 대한 정규식이다: 다섯 단계 이름 중 셋에 밑줄이 들어
    있어서, 쪼개면 ``missing_count``가 ``missing``이 됐고 :func:`_wrap_declared`가 끼우는 대치기가
    덧붙이는 단계 *앞에* 내렸다 — 덧붙이는 단계가 세려고 존재하는 그 NaN을 대치로 없애면서.
    """
    return re.sub(r"_\d+$", "", name)


def describe_pipeline(estimator: Any, declared: list[str]) -> list[str]:
    """실제로 세워진 파이프라인에 대고 맞춘, 선언한 것의 메아리.

    ``declared`` = :func:`automl_agent.dataset.pipeline.build_steps`가 낸 것: *명세*가 요청해서 받은
    단계 전부. 파이프라인 전체가 아니다 — :func:`_wrap_declared`가 계열이 필요로 하고 명세는 말한 적
    없는 대치기나 스케일러를 더하는데, 그것을 빼먹는 것이 ``applied_hyperparams``가 막으려고 존재하는
    그 거짓이다 (스케일링을 요청한 적 없는 계획에 스케일된 적합을 공으로 돌리는 보고서).

    그래서: 객체에서 순서대로 읽고, 요청받지 않은 것에 ``(auto)``를 표시한다. 다시 계산하지 않고 읽는
    것은 :func:`describe_preprocessing`과 같다 — config에서 두 번째로 유도하는 것은 돈 것과 어긋날
    자유가 있는 두 번째 물건이다.
    """
    steps = [name for name, _step in getattr(estimator, "steps", ()) if name != "model"]
    remaining = list(declared)
    echo: list[str] = []
    for name in steps:
        base = base_step_name(name)
        if remaining and remaining[0].split("(", 1)[0] == base:
            echo.append(remaining.pop(0))
        else:
            echo.append(f"{base}(auto)")
    return echo


def _wrap_declared(
    key: str, model: Any, declared: list[tuple[str, Any]], log: LogBuffer
) -> Any:
    """``pipeline`` 명세가 기술한 파이프라인, 더하기 그것이 깨뜨려서는 안 되는 두 가지.

    명세는 플래그보다 표현력이 커서 플래그로는 빼먹을 수 없던 것을 빼먹을 수 있다. 안전망 둘은 다
    *계열*에 관한 것이고, 그것은 계획하는 쪽이 늘 갖고 있지는 않고 실행기는 항상 갖고 있는 지식이다:

    * **계열이 NaN을 받을 수 없을 때의 대치기.** 없으면 적합이 raise하고, 명세가 자기 아이디어에
      쓰려던 그 반복을 값으로 낸다. 맨 앞이 아니라 덧붙이는 단계 뒤에 끼운다. 플래그 경로가 쓴
      자리와 같고 옳은 자리는 그것뿐이다: NaN이 사라진 뒤에 ``missing_indicator``에는 표시할 것이
      없다.
    * **그것을 필요로 하는 계열에 대한 스케일링.** 플래그 경로는 추정기가 스케일에 민감하고 반대로
      말하는 것이 없으면 늘 그것을 더한다. 그러니 스케일링을 그냥 말하지 않은 명세는 ``logreg``,
      ``svc``, ``mlp``, ``knn``에서 그것을 조용히 떨어뜨린다 — 계획으로 차려입은 퇴행이다.

    둘 다 로그에 남는다. 명세가 요청하지 않은 단계를 담은 파이프라인은 그렇다고 말해야 한다. 그러지
    않으면 ``applied_pipeline``이 ``applied_hyperparams``가 막으려고 존재하는 것과 같은 종류의
    거짓이 된다.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = list(declared)
    names = {base_step_name(name) for name, _step in steps}
    if PIPELINE_STEP_IMPUTE not in names and key not in NATIVE_NAN:
        # 마지막으로 덧붙이는 단계 뒤에. 표시가 불가능해지기 전에 표시가 되도록.
        at = 1 + max(
            (
                position
                for position, (name, _step) in enumerate(steps)
                if base_step_name(name) in PIPELINE_APPENDING_STEPS
            ),
            default=-1,
        )
        steps.insert(at, (PIPELINE_STEP_IMPUTE, SimpleImputer(strategy=DEFAULT_IMPUTE)))
        log.write(
            f"pipeline: no impute step declared and {key} cannot be fitted on a NaN; "
            f"inserted {DEFAULT_IMPUTE} imputation at position {at}"
        )
    if PIPELINE_STEP_SCALE not in names and key in SCALE_SENSITIVE:
        steps.append((PIPELINE_STEP_SCALE, StandardScaler()))
        log.write(f"pipeline: no scale step declared and {key} is scale-sensitive; appended one")
    steps.append(("model", model))
    log.write("pipeline steps: " + " -> ".join(name for name, _step in steps))
    return Pipeline(steps)


# 언제 멈출지 정하려고 떼어 두는 *train*의 비율. HistGradientBoosting 자신의
# ``validation_fraction`` 기본값과 일부러 같게 둔다. 그래야 부스팅 계열 둘이 비슷한 크기의 증거에서
# 멈추고, 계열 비교가 프로토콜 비교까지 되지 않는다.
EARLY_STOPPING_FRACTION = 0.1
# 이 행 수 아래에서는 멈춤 신호가 라운드 수를 정하는 잡음이고, 그것은 아예 멈추지 않는 것보다
# 나쁘다. ``train_subsample``이 지키는 바닥과 같은 자리수다.
MIN_EARLY_STOPPING_ROWS = 50


def held_back_indices(
    x: Any, y: Any, seed: int, stratify: bool, groups: Any, fraction: float
) -> tuple[Any, Any]:
    """val이 아니라 *train*에서 떼어낸 조각에 대한 ``(fit_idx, held_idx)``.

    어느 시도가 이기는지는 val이 정하니, 시도가 *자기를 위해* 고르는 것 — 라운드 수, 결정 컷 — 은
    다른 행에 대고 골라야 한다. 그러지 않으면 시도가 나중에 자기가 채점될 집합에서 고르는 것이 된다.
    이 프로젝트가 ``best``를 결과로 보고하는 데에 대고 하는 반론과 같은 반론이고, 한 층 아래다.

    실행에 그룹이 있으면 그룹을 존중하며, 이유는 두 호출처에서 같다: 적합되는 행과 환자를 공유하는
    떼어 둔 조각은 적합에 *대해*가 아니라 적합과 *함께* 답한다 — 나중 부스팅 라운드가 아직 나아지는
    중으로 읽히고, 컷이 학습 분포를 떠난 적도 없는데 전이되는 것으로 읽힌다.
    """
    import numpy as np

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        return next(iter(splitter.split(x, y, groups=groups)))

    from sklearn.model_selection import train_test_split

    return train_test_split(
        np.arange(len(y)),
        test_size=fraction,
        random_state=seed,
        stratify=y if stratify else None,
    )


def _early_stopping_split(
    x: Any, y: Any, seed: int, stratify: bool, groups: Any
) -> tuple[Any, Any, Any, Any]:
    """멈춤 증거 조각. ``fit_estimator``가 추정기에 건네는 네 배열의 모양으로."""
    fit_idx, stop_idx = held_back_indices(
        x, y, seed, stratify, groups, EARLY_STOPPING_FRACTION
    )
    return x[fit_idx], x[stop_idx], y[fit_idx], y[stop_idx]


def fit_estimator(
    pipeline: Any,
    x: Any,
    y: Any,
    seed: int,
    log: LogBuffer,
    *,
    stratify: bool = True,
    groups: Any = None,
) -> dict[str, Any]:
    """파이프라인을 적합한다. 추정기가 일찍 멈추겠다고 했으면 eval set을 직접 만들어서.

    ``Pipeline.fit``은 ``eval_set``을 전달할 수 없다 — 추정기는 *변환된* 행렬을 필요로 하고, 변환기는
    파이프라인의 적합이 돌기 전에는 적합되지 않는다. 그래서 마지막 단계가 ``early_stopping_rounds``를
    지고 있으면, 전처리 단계를 적합 조각에 적합하고, 두 조각을 변환하고, 추정기에 그 짝을 건넨다.
    그것은 파이프라인 자신의 단계 객체이므로, 이 함수가 돌아올 때 파이프라인은 적합돼 있고
    ``predict``/``joblib.dump``는 평범한 Pipeline을 본다.

    **전처리는 train 전체가 아니라 적합 조각만으로 적합된다** — 대치 중앙값에 기여한 멈춤 조각은
    자기가 내려고 존재하는 결정에서 떼어져 있지 않다.

    그 행들을 :func:`describe_internal_validation`의 모양으로 돌려준다. 떼어 둔 것이 없으면 ``{}``.
    개수는 이 함수가 자른 조각의 길이이고, **``n * fraction``을 다시 계산한 값이 아니다**.
    """
    final = pipeline.steps[-1][1]
    rounds = getattr(final, "early_stopping_rounds", None)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        if isinstance(rounds, int) and not isinstance(rounds, bool):
            # 아래의 행이-너무-적음 분기와 같은 말투로 소리 내어 말한다. 0인 라운드 수는
            # "early stopping 없음"을 뜻한 계획에서 여기 닿고, 음수는 ``sanitise_hyperparams``를
            # 건너뛴 config에서 닿는다 — 손으로 고친 파일이거나 무작위 탐색 스크립트다. 둘 다
            # train의 모든 행에 적합하는데, ``applied_hyperparams``가 여전히 요청받은 수를 보이므로
            # 기록이 이 사실을 실어야 한다.
            log.write(
                f"early_stopping_rounds={rounds} not applied: a round count of {rounds} is not "
                "a stop, so the fit sees every train row"
            )
            # 그리고 여기서 건너뛰는 것에 그치지 않고 추정기에서 지운다: xgboost는 0이 아닌
            # 모든 수에 대해(음수 포함) early-stopping 콜백을 달고, 그다음 ``fit`` 안에서
            # ``Must have at least 1 validation dataset for early stopping``을 raise한다 — 이
            # 분기가 이미 무시하기로 정한 수 때문에 반복 하나를 다 쓰면서. 거기서 0은 falsy라
            # 살아남겠지만, 어느 수를 한 라이브러리가 참아 주는지를 적어 넣는 두 번째 조건보다
            # 분기에 규칙 하나가 싸다.
            final.set_params(early_stopping_rounds=None)
        pipeline.fit(x, y)
        return {}

    x_fit, x_stop, y_fit, y_stop = _early_stopping_split(x, y, seed, stratify, groups)
    if len(y_stop) < MIN_EARLY_STOPPING_ROWS:
        # 우회하지 않고 끈다: 30행에서 고른 라운드 수는 보고서가 면책 문구를 붙여야 할 수이고,
        # n_estimators 전체는 적어도 정직하다.
        log.write(
            f"early_stopping_rounds={rounds} not applied: the stopping slice would hold "
            f"{len(y_stop)} rows, under the {MIN_EARLY_STOPPING_ROWS} needed"
        )
        final.set_params(early_stopping_rounds=None)
        pipeline.fit(x, y)
        return {}

    head = pipeline.steps[:-1]
    if head:
        from sklearn.pipeline import Pipeline

        pre = Pipeline(list(head))
        x_fit_t = pre.fit_transform(x_fit, y_fit)
        x_stop_t = pre.transform(x_stop)
    else:
        x_fit_t, x_stop_t = x_fit, x_stop

    final.fit(x_fit_t, y_fit, eval_set=[(x_stop_t, y_stop)], verbose=False)
    best = getattr(final, "best_iteration", None)
    reached = f", best_iteration={best}" if isinstance(best, int) else ""
    log.write(
        f"early_stopping_rounds={rounds}: fitted on {len(y_fit)} rows, stopped against "
        f"{len(y_stop)} rows held out of train{reached}"
    )
    held_back: dict[str, Any] = {
        "held_out_rows": len(y_stop),
        "fit_rows": len(y_fit),
        "validation_fraction": EARLY_STOPPING_FRACTION,
    }
    if isinstance(best, int) and not isinstance(best, bool):
        # ``best_iteration``은 0에서 센다. 이 필드는 적합된 라운드 수를 뜻하니
        # ``stopped_at_iter == max_iter`` 읽기가 여기서도 성립한다 — 멈춤이 물지 않았다는 말이다.
        held_back["stopped_at_iter"] = best + 1
    cap = getattr(final, "n_estimators", None)
    if isinstance(cap, int) and not isinstance(cap, bool):
        held_back["max_iter"] = cap
    return held_back


def describe_preprocessing(estimator: Any) -> dict[str, Any]:
    """파이프라인이 *무엇인지*. 다시 계산하지 않고 객체에서 읽는다.

    :func:`_wrap_preprocessing`은 존중할 수 없는 요청을 조용히 낮춘다 — :data:`NATIVE_NAN` 밖의
    ``impute: none``은 중앙값이 되고, 모르는 전략은 기본값이 된다 — 그래서 config는 돈 것의 기록이
    아니다. 여기서 config에서 답을 두 번째로 유도하면 첫 번째와 어긋날 자유가 있는 두 번째 코드
    경로가 된다. 세워진 파이프라인을 들여다보는 것은 그럴 수 없다.

    이것이 있는 이유는 ``applied_hyperparams``가 있는 이유와 같다. 이것이 없을 때 보고서가 인용할 수
    있는 것은 카드의 ``preprocessing`` 블록뿐이었는데, 그것은 계획의 덮어쓰기가 아니라 *카드*의
    기본값이다. 그래서 어떤 실행의 기록은 둘 다 돌지 않은 시도에 대해 중앙값 대치와 스케일링을
    충실히 알렸다.
    """
    steps = getattr(estimator, "named_steps", None)
    if not steps:
        return {}
    imputer = steps.get("impute")
    return {
        # 대치기가 ``ColumnTransformer``일 때는 ``per_column``: 전략 하나가 없고, 이것이 한때
        # ``getattr(..., "strategy", "")``를 보고했다 — 빈 문자열이라 보고서가 "impute: "로 찍고
        # 읽는 사람은 그것을 보고서의 버그로 받는다. 세부는 ``applied_pipeline``에 있다. 이 필드가
        # 빚진 것은 참인 단어 하나다.
        "impute": IMPUTE_NONE
        if imputer is None
        else str(getattr(imputer, "strategy", "") or PER_COLUMN_IMPUTE),
        "scale": "scale" in steps,
        # ``scale``처럼 켜짐/꺼짐 상관없이 보고한다: "지시자 없음"과 "이 실행은 이 필드보다
        # 먼저다"를 구분해야 하는 기록은 빈칸이 아니라 false를 필요로 한다.
        MISSING_INDICATOR: MISSING_INDICATOR in steps,
        MISSING_COUNT: MISSING_COUNT in steps,
    }


def describe_internal_validation(estimator: Any, n_train: int) -> dict[str, Any]:
    """추정기 자신의 early stopping이 떼어 둔 학습 행. 적합된 객체에서 읽는다.

    ``fitting on N rows``는 ``fit``에 *들어간* 행이고 적합된 행이 아니다 — 추정기 셋이 N에서 자기
    조각을 잘라내는데 그렇다고 말하는 것이 없었다.

    **config가 아니라 객체에서 읽는다.** config는 어느 쪽으로도 답할 수 없기 때문이다:
    ``early_stopping='auto'``는 표본 1만 개 위에서만 True로 풀리고, ``validation_fraction=None``인
    ``early_stopping=True``는 아무것도 떼어 두지 않는다. 어느 속성이 그것을 결정하는지는
    :func:`_held_rows_back`.

    **``n_train * fraction``이 아니라 ``train_test_split``에서 센다** — 두 번째 산술 경로는 한 행
    어긋날 자유가 있다. :func:`automl_agent.capabilities.describe_row_budget`이 *바로* 그 두 번째
    경로다(적합이 존재하기 전에 예측한다). 테스트가 둘을 묶어 둔다.

    ``xgboost``는 이것을 건너뛴다 — 그 ``eval_set`` 분할은 :func:`fit_estimator`의 것이고, 같은 모양을
    돌려준다. 어느 쪽이든 필드 하나. 떼어 둔 것이 없으면 ``{}``.
    """
    from sklearn.model_selection import train_test_split

    fraction = getattr(estimator, "validation_fraction", None)
    if fraction is None or not _held_rows_back(estimator):
        return {}
    held_out = len(train_test_split(list(range(n_train)), test_size=fraction)[1])
    described: dict[str, Any] = {
        "held_out_rows": held_out,
        "fit_rows": n_train - held_out,
        "validation_fraction": fraction,
    }
    # early stopping이 실제로 물었는지. ``stopped_at_iter == max_iter``는 행을 쓰고도 상한에
    # 닿았다는 뜻이다 — 설정이 받지 못한 멈춤에 값을 치렀다.
    stopped = getattr(estimator, "n_iter_", None)
    if stopped is None:
        # ``gradient_boosting``은 반복이 아니라 stage를 세고 그 수를 다른 이름으로 부른다. 같은
        # 양이다: 멈춤 전에 몇 라운드가 적합됐는지.
        stopped = getattr(estimator, "n_estimators_", None)
    cap = getattr(estimator, "max_iter", None)
    if cap is None:
        cap = getattr(estimator, "n_estimators", None)
    if isinstance(stopped, int) and not isinstance(stopped, bool):
        described["stopped_at_iter"] = stopped
    if isinstance(cap, int) and not isinstance(cap, bool):
        described["max_iter"] = cap
    return described


def _held_rows_back(estimator: Any) -> bool:
    """이 적합된 추정기가 정말로 학습 행에서 검증 조각을 잘라냈는지.

    추정기 셋이 ``validation_fraction``을 받는데 셋 모두에 답하는 속성 하나가 없다:

    * ``hist_gbdt``와 ``mlp``는 조각이 정말로 채점됐을 때만 ``validation_score_`` /
      ``validation_scores_``를 채운다. 그래서 답은 *비어 있지 않은* 쪽이다 — 있음은 답이 아니다.
      속성은 어느 쪽이든 설정되고(빈 배열 / ``None``), ``_use_validation_data``는
      ``early_stopping=False``에서도 True다.
    * ``gradient_boosting``은 둘 다 내놓지 않으면서 ``n_iter_no_change``가 None이 아니면 어쨌든
      분할한다 — sklearn 자신의 조건이고, 객체에서 읽는다. ``build_estimator``가 ``get_params``가
      받는 키를 다 전달하므로 닿을 수 있다. registry가 그것을 광고하지 않는 것이 계획이 그것을
      설정하는 것을 막지는 않는다.

    그래서 분기에는 ``hasattr``, 판정에는 값이다: ``None``인 점수 목록에서 아래로 떨어지면
    ``gradient_boosting``의 규칙을 ``mlp``에 적용하게 되는데, ``mlp``의 ``n_iter_no_change``는
    기본이 10이고 ``early_stopping=True``가 없으면 아무 뜻이 없어서, 모든 MLP 시도가 떼어 둔 적 없는
    행을 보고하게 된다.
    """
    for attribute in ("validation_score_", "validation_scores_"):
        if hasattr(estimator, attribute):
            scores = getattr(estimator, attribute)
            return scores is not None and len(scores) > 0
    return getattr(estimator, "n_iter_no_change", None) is not None


# --------------------------------------------------------------------------- #
# 채점
# --------------------------------------------------------------------------- #


def scorers(
    y_true: Any, pred: Any, proba: Any, average: str, task: str = TASK_CLASSIFICATION
) -> dict[str, Any]:
    """registry 지표마다 thunk 하나. 그래서 요청받지 않은 것은 계산되지 않는다.

    **``scripts/profile.py``도 이것을 부르고, 그것이 요점이다** — 바와 그것에 대고 비교되는 점수는 같은
    측정이어야 한다. 한때 거기에 복제본이 살았는데, 두 파일 사이에서 제곱근만큼 다른 ``rmse``는 아무도
    틀린 줄 볼 수 없는 비교다.

    회귀 쪽은 일부러 *스케일하지 않는다*: ``mae``/``rmse``를 타깃 열 자신의 단위로 낸다. 그것이 그
    값을 읽을 수 있게 만들고, 이식 가능한 기본 바를 불가능하게 만든다
    (:func:`automl_agent.scoring.goal.default_bar`).

    sklearn은 분기 안에서 import한다. 그래야 프로파일러가 다른 스크립트의 추정기 값을 치르지 않고
    이것을 부를 수 있다.
    """
    if task == TASK_REGRESSION:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "r2": lambda: r2_score(y_true, pred),
            "mae": lambda: mean_absolute_error(y_true, pred),
            # ``squared=False``가 아니다: sklearn 1.6에서 없어졌고,
            # ``root_mean_squared_error``는 1.4 전에는 없다.
            "rmse": lambda: float(mean_squared_error(y_true, pred)) ** 0.5,
        }

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "f1": lambda: f1_score(y_true, pred, average=average, zero_division=0),
        "accuracy": lambda: accuracy_score(y_true, pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, pred),
        "precision": lambda: precision_score(y_true, pred, average=average, zero_division=0),
        "recall": lambda: recall_score(y_true, pred, average=average, zero_division=0),
        "roc_auc": lambda: roc_auc_score(y_true, proba),
        "pr_auc": lambda: average_precision_score(y_true, proba),
    }


def score_split(
    names: list[str],
    y_true: Any,
    pred: Any,
    proba: Any,
    average: str,
    log: LogBuffer,
    task: str = TASK_CLASSIFICATION,
) -> dict[str, float]:
    """한 분할에서 ``names``를 채점한다. 이 예측이 지원할 수 없는 것은 건너뛴다.

    registry가 몬다, 일부러. ``--metric``이 *이 태스크에서* 받는 모든 지표가 여기서 나온다. 그래야
    그중 하나로 유도된 목표가 존재하는 수에 대고 재어진다 — 목표는 알지만 결과에는 없던 지표가 한때
    "바에 닿은 적 없음"으로 읽혔다.
    """
    thunks = scorers(y_true, pred, proba, average, task)
    scores: dict[str, float] = {}
    for name in names:
        spec = METRICS.get(name)
        if spec is None:  # pragma: no cover - RunConfig 검증이 막는다
            continue
        if spec.task != task:
            # 로그에 남길 건너뜀이 아니다: 다른 태스크의 지표는 이 결과에서 빠진 것이 아니라
            # 이 결과에 대해 정의되지 않은 것이다.
            continue
        if spec.binary_only and average != "binary":
            log.write(f"{name} skipped: defined for binary targets only (average={average})")
            continue
        if spec.needs_proba and proba is None:
            continue
        try:
            value = float(thunks[name]())
        except (ValueError, AttributeError) as exc:
            log.write(f"{name} skipped: {exc}")
            continue
        if not math.isfinite(value):
            # sklearn은 채점할 수 없는 일부 입력에 대해 raise 대신 NaN을 돌려준다 —
            # ``y_true``에 클래스가 하나인 ``roc_auc``가 닿을 수 있는 경우다. NaN은 점수가 아니고
            # 키가 없는 것보다 나쁘다: ``json.dumps``가 맨 ``NaN``으로 쓰는데 엄격한 독자에게 그것은
            # 유효한 JSON이 아니고, 그것에 대한 모든 비교가 False라 목표 검사가 "바에 닿지 못함"으로
            # 읽히며, 프롬프트나 배치 요약은 측정이 이뤄진 것처럼 그것을 찍는다.
            log.write(f"{name} skipped: not defined on these rows (returned {value})")
            continue
        scores[name] = value
    return scores


def specificity(y_true: Any, pred: Any) -> float:
    """참 음성률 — 음성 클래스를 양성으로 두고 잰 recall.

    이진 전용이고, 일부러 registry 지표가 *아니다* — 다수 클래스만 찍는 예측기가 1.0을 받으니 아무도
    이것을 목표로 삼을 수 없다. ``train_val_gap``처럼 진단값으로 낸다:
    ``balanced_accuracy = (recall + specificity) / 2``이므로, 불균형 손잡이가 어느 쪽으로 움직여야
    하는지 지목하는 것은 이 짝뿐이다. 이것이 없으면 Critic은 추측한다.
    """
    from sklearn.metrics import recall_score

    return float(recall_score(y_true, pred, pos_label=0, zero_division=0))


def bootstrap_resamples(cfg: dict[str, Any]) -> int:
    """신뢰구간이 받는 재표집 횟수. ``0``이면 끈다.

    고정하지 않고 config에서 읽는 것은 50만 행 분할을 가진 호출자에게 탈출구를 주려는 것이지만, CLI
    플래그는 없다: 구간은 "개선됨"과 "이 행들로는 구분되지 않음"의 차이이고, 손잡이가 있으면 실행을
    단호해 보이게 하려고 그것을 끄도록 초대하게 된다. 쓰레기 값은 실행을 실패시키지 않고 기본값으로
    떨어진다.
    """
    raw = as_number(dict(cfg.get("bootstrap") or {}).get("resamples", DEFAULT_RESAMPLES))
    return DEFAULT_RESAMPLES if raw is None else max(0, int(raw))


def _proba(model: Any, x_arr: Any, n_classes: int, log: LogBuffer) -> Any:
    """양성 클래스 확률, 이 모델/타깃이 그것을 줄 수 없으면 None."""
    if n_classes != 2 or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(x_arr)[:, 1]
    except (ValueError, AttributeError, IndexError) as exc:
        log.write(f"predict_proba unavailable: {exc}")
        return None


# --------------------------------------------------------------------------- #
# 결정 규칙 — 확률을 어디서 자르는가
# --------------------------------------------------------------------------- #
#
# 불변식이 둘이고, 둘 다 하중을 받는다:
#
# * **컷은 추정기가 적합되지 않았고 시도가 보고하지도 않는 행에서 고른다** — train에서 떼어낸 조각,
#   early stopping과 같은 조건이다 (:func:`held_back_indices`). 학습 분할을 외우는 계열에서는 적합된
#   행 위에 고를 것이 없다: 닿을 수 있는 최고 점수가 후보 컷의 넓은 고원 위에 도착하니, 컷은 식별되지
#   않고 검증 점수가 그 위를 오간다.
# * **모델 옆 디스크에 적는다.** 그래야 홀드아웃과 ``predict``가 검증 점수를 얻은 그 규칙으로 행에
#   라벨을 붙인다. 이 프로세스 안에만 사는 컷은 이후 모든 패스가 조용히 다른 규칙을 채점하게 만든다.
#

# ``decision.threshold``가 수 말고 받는 것: 떼어 둔 행에서 컷을 고르라.
DECISION_TUNED = "tuned"

# 컷을 고르려고 적합에서 빼 두는 *train*의 비율. ``EARLY_STOPPING_FRACTION``의 두 배인 것은 둘이
# 사는 것이 다르기 때문이다 — 아직 움직이는 곡선에서 딴 라운드 수 대(對) 적은 행에서의 잡음이 바로
# 고원을 만드는 지표 위의 argmax.
CUT_FRACTION = 0.2
# 이 행 수 아래에서는 argmax가 운영점을 고르는 잡음이고, 기본 규칙은 적어도 규칙이다.
# ``MIN_EARLY_STOPPING_ROWS``보다 높은 것은 위의 이유 때문이고, 또 이 손잡이가 대상으로 삼는 타깃이
# 불균형하기 때문이다 — 9:1 비율의 200행은 양성 20개이고, 후보 컷에서의 recall 추정으로는 이미 적다.
MIN_CUT_ROWS = 200

# 후보 컷. 일부러 출처가 둘이다. 떼어 둔 확률의 분위수는 행이 실제로 있는 곳에 점을 놓는다 —
# 양성률 9%에서 보정된 모델은 거의 모든 행을 0.2 아래에 두고, 균일 격자는 빈 공간에 자기를 다 쓴다.
# (0, 1) 위의 균일 격자는 분위수가 덮을 수 없는 것을 덮는다: 확신하는 모델의 확률은 양쪽 끝에 쌓이고,
# 그러면 분위수 집합에는 중간 근처의 후보가 하나도 없다. 0.5는 명시적으로 집합에 있다. 그래야 튜너가
# 언제나 기본 규칙을 돌려줄 수 있고, "튜닝이 아무것도 사지 못했다"가 닿을 수 없는 답이 아니라 닿을 수
# 있는 답이 된다.
CUT_QUANTILES = 99
CUT_GRID = 49


def label_at_cut(proba: Any, cut: float) -> Any:
    """양성 클래스 확률에서 ``cut``을 기준으로 뽑은 0/1 라벨.

    ``>=``이므로 관측된 확률 *바로 그 자리*의 컷은 양성이다. 0.5에서 sklearn과 같지 않다:
    ``predict``는 확률 열 둘에 argmax를 하고 ``numpy.argmax``는 정확한 동점을 앞쪽으로 깨니, 정확히
    0.500인 행은 거기서 음성이고 여기서 양성이다. 트리에서 닿을 수 있다(여러 행이 한 잎 값을 공유한다)
    — 그래서 0.5를 명시적으로 요청한 시도는 튜닝 안 한 것이 아니라 컷을 적용한 것으로 기록된다.
    """
    import numpy as np

    return (np.asarray(proba) >= float(cut)).astype(int)


def requested_cut(cfg: dict[str, Any], log: LogBuffer) -> float | str | None:
    """``config["decision"]["threshold"]``가 요청한 것: 컷, ``"tuned"``, 또는 없음.

    ``None`` — 키가 없음 — 은 기본 규칙이고, 이 블록이 있기 전에 쓰인 모든 config가 말하는 것이
    그것이다. 쓰레기는 치명적이지 않고 로그에 남기고 무시한다. 추정기가 받지 않을 하이퍼파라미터와 같은
    조건이다: 시도는 그래도 유효한 시도이고, 요청이 효력을 갖지 못했다고 말하는 것은 결과에
    ``applied_threshold``가 없다는 사실이다.
    """
    raw = (cfg.get("decision") or {}).get("threshold")
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() == DECISION_TUNED:
            return DECISION_TUNED
        log.write(
            f"decision.threshold={raw!r} ignored: expected a number in (0, 1) "
            f'or "{DECISION_TUNED}"'
        )
        return None
    cut = as_number(raw)
    if cut is None or not 0.0 < cut < 1.0:
        log.write(
            f"decision.threshold={raw!r} ignored: outside (0, 1), which labels every row "
            "the same way"
        )
        return None
    return round(cut, 6)


def tune_threshold(
    y_held: Any, proba: Any, metric: str, average: str, task: str, log: LogBuffer
) -> float | None:
    """떼어 둔 행 위에서 ``metric``을 가장 잘 받는 ``proba``의 컷.

    **언제나 ``balanced_accuracy``가 아니라 목표 지표다** — 하나를 최대화하는 컷이 다른 것을 최대화하지
    않으니, 실행이 심판받는 그 수를 쓸어 보는 것이 손잡이가 그 수를 낮추지 않게 지킨다.

    **동점은 고원의 가장자리가 아니라 가운데로 간다.** 같은 점수는 혼동행렬이 동일하다는 뜻이니, 고원은
    행이 구분할 수 없는 구간이고 그 중점이 더 나쁘다고 알려진 두 컷에서 가장 멀다.

    고를 것이 없으면 ``None``이고, 어느 경우인지는 로그가 말한다: 확률 없음, 받아들일 후보 없음, 또는
    **컷이 움직일 수 없는 지표** — ``roc_auc``와 ``pr_auc``는 ``proba``만으로 나오니 모든 후보가
    동점이고, 하나를 돌려주면 바꿀 수 없는 수를 위해 고른 컷을 기록하게 된다.
    """
    if proba is None:
        log.write("threshold tuning skipped: this model gives no positive-class probabilities")
        return None
    import numpy as np

    values = np.asarray(proba)
    grid = np.concatenate(
        [
            np.quantile(values, np.linspace(0.01, 0.99, CUT_QUANTILES)),
            np.linspace(0.02, 0.98, CUT_GRID),
            [0.5],
        ]
    )
    candidates = sorted({round(float(c), 6) for c in grid if 0.0 < float(c) < 1.0})
    if not candidates:
        log.write("threshold tuning skipped: the held-back probabilities leave no cut inside (0, 1)")
        return None
    thunk_key = canonical(metric)
    scored: list[tuple[float, float]] = []
    for cut in candidates:
        try:
            score = float(
                scorers(y_held, label_at_cut(proba, cut), proba, average, task)[thunk_key]()
            )
        except (ValueError, AttributeError, KeyError) as exc:
            log.write(f"threshold tuning skipped: {metric} not scorable on the held-back rows ({exc})")
            return None
        if math.isfinite(score):
            scored.append((score, cut))
    if not scored:
        log.write(f"threshold tuning skipped: {metric} is not defined on the held-back rows")
        return None
    # 부호는 오늘 아무것도 하지 않는다 — registry의 모든 분류 지표는 최대화한다 — 그런데도 여기
    # 있는 것은, 튜닝하지 않는 것보다 나쁜 단 하나가 *가장 나쁜* 컷으로 튜닝하는 것이고, 이쪽에
    # 최소화 지표가 더해지는 날 하드코딩된 ``max``가 조용히 그것을 할 것이기 때문이다.
    sign = -1.0 if direction_of(thunk_key) == MINIMIZE else 1.0
    best_value = max(sign * score for score, _cut in scored)
    plateau = [cut for score, cut in scored if sign * score == best_value]
    if len(plateau) == len(scored):
        log.write(
            f"threshold tuning skipped: {metric} is the same at every one of {len(scored)} "
            "candidate cuts — it is computed from the probabilities, not from the labels, so no "
            "cut changes it"
        )
        return None
    cut = plateau[len(plateau) // 2]
    spread = (
        f", indistinguishable over {len(plateau)} cuts from {plateau[0]} to {plateau[-1]}"
        if len(plateau) > 1
        else ""
    )
    log.write(
        f"decision threshold tuned on the held-back rows: {cut} "
        f"({metric} {sign * best_value:.6f} over {len(scored)} candidate cuts{spread}) — "
        "these rows were in neither the fit nor the reported split"
    )
    return cut


def save_decision(rule: dict[str, Any], path: Path, log: LogBuffer) -> str | None:
    """이 시도가 적용한 컷을 남긴다. 그 경로를 돌려주거나, 쓸 수 없으면 ``None``.

    :func:`save_model`처럼 절대 치명적이지 않다 — 이것이 돌 때쯤 검증 점수는 이미 얻은 것이다. 잃는
    것은 *합의*다: 홀드아웃이 같은 추정기를 0.5에서 채점하고, 그 하락이 아예 쓰이지 않은 파일이 아니라
    잘 전이되지 않은 컷으로 읽힌다. 그래서 로그 한 줄이 있다.

    여기 이웃 둘과 달리 이것은 집계값이다 — [0, 1] 안의 스칼라 하나, 어떤 지표의 성적 — 그래서 이
    디렉터리에서 값까지 ``applied_threshold``로 공표되는 유일한 아티팩트다.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rule, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.write(f"decision rule not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"decision rule saved to {path.name}: {json.dumps(rule, ensure_ascii=False)}")
    return str(path)


def load_decision(path: Path, log: LogBuffer) -> float | None:
    """모델 옆에 저장된 컷, 또는 ``None`` — 그것은 sklearn의 고정 0.5 규칙을 뜻한다.

    없음은 평범한 경우이고 줄을 받지 않는다: 이 파일이 있기 전에 적합된 모든 모델과, 계획이 컷을
    요청하지 않은 모든 시도가 그대로 채점된다.

    파일은 있는데 쓸 수 있는 컷이 없으면 줄을 받을 값이 *있고*, 그래도 raise하지 않고 떨어진다 —
    :func:`load_schema`와 반대이고, 일부러다. 틀린 스키마는 폭만으로 열을 줄 세우고 다른 데이터셋을
    조용히 채점한다. 없는 컷은 라벨을 정의된 규칙으로 되돌릴 수밖에 없고, 그것은
    ``applied_threshold``의 부재로 공표되며 점수에 보인다.
    """
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.write(f"{path.name} unusable ({exc}); scoring at the default rule")
        return None
    # ``float(...)``을 날값에 바로 걸지 않는다. 닿을 수 있는 모든 실패가 위의 ``except``에 있어서
    # 작동했지만 "파일을 믿고 뒷감당은 잡는다"로 읽혔고, 타입이 ``Any``라 반환값이 float임을 증명할 수
    # 있는 것이 없었다. ``as_number``는 그 둘을 한 번에 준다 — 판정과 좁혀진 타입.
    raw = loaded.get("threshold") if isinstance(loaded, dict) else None
    cut = as_number(raw)
    if cut is None:
        log.write(f"{path.name} holds no numeric threshold; scoring at the default rule")
        return None
    if not 0.0 < cut < 1.0:
        log.write(f"{path.name} holds threshold={cut}, outside (0, 1); scoring at the default rule")
        return None
    log.write(f"applying the decision rule saved with this model: threshold={cut}")
    return cut


# --------------------------------------------------------------------------- #
# 학습
# --------------------------------------------------------------------------- #


def evaluate_split(
    model: Any,
    x_eval: Any,
    y_eval: Any,
    n_classes: int,
    average: str,
    log: LogBuffer,
    *,
    task: str = TASK_CLASSIFICATION,
    interval_metric: str | None = None,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
    pred: Any = None,
    proba: Any = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """이 예측이 지원하는 모든 registry 지표, 더하기 이진 진단값.

    ``run_training``의 검증 채점과 ``score_saved_model``의 한 번짜리 test 채점이 공유한다. 그래야
    보고서가 나란히 놓는 두 수가 같은 코드에서 나온다.

    ``interval_metric``  부트스트랩 구간까지 받는 지표 하나 — 목표 지표다. 비용이 지표 × 재표집에
                         선형이기 때문이다. ``groups``는 군집 분할을 그룹 단위로 재표집한다
                         (:mod:`automl_agent.scoring.intervals`).
    ``pred``/``proba``   배열을 *저장*해야 하는 그 한 호출자가 여기서 계산하지 않고 넣어 준다:
                         **두 번 예측하면 한 파일 이름 아래에 출처가 둘이 된다**.
    ``threshold``        sklearn의 고정 0.5를 대신하고, **아래의 모든 지표가 그 컷에서 재어진다** —
                         구간과 ``specificity``도 포함. 한 규칙에서 보고하고 다른 규칙에서 진단한
                         점수는 Critic을 틀린 손잡이로 보낸다.
    """
    if pred is None:
        pred = model.predict(x_eval)
        proba = _proba(model, x_eval, n_classes, log)
    if threshold is not None:
        if proba is None:
            # 컷이 요청됐고 적용할 수 없으니, 보고돼서도 안 된다: 아래에서
            # ``applied_threshold``가 없다는 것이 그 기록 전부다.
            log.write("the saved decision rule needs probabilities this model cannot give; default rule")
            threshold = None
        else:
            # ``pred``를 믿지 않고 ``proba``에서 다시 계산한다. 그래서 이것이 멱등이다:
            # :func:`run_training`은 자기 복사본을 이미 다시 라벨링했고 — 채점한 그 배열을 저장해야
            # 한다 — 여기서 정확히 같은 라벨에 다시 내린다.
            pred = label_at_cut(proba, threshold)
    metrics = score_split(list(METRICS), y_eval, pred, proba, average, log, task)
    for alias, canonical_name in METRIC_ALIASES.items():
        if canonical_name in metrics:
            metrics[alias] = metrics[canonical_name]

    # balanced_accuracy의 나머지 절반. balanced_accuracy가 목표일 때만이 아니라 모든 이진 시도에서
    # 낸다. 채점기 호출 한 번이 값이고, "운영점이 어긋났다"를 "가중치를 올려야 한다"로 바꾸는 수가
    # 그것이기 때문이다.
    if n_classes == 2:
        metrics["specificity"] = round(specificity(y_eval, pred), 6)

    # 결정 임계값을 고르는 것이 얼마짜리였을지를 그 지표 자신의 단위로 잰 값. 실제 LLM 실행은 모두
    # 임계값 쓸기를 계획했고 실행기는 그것을 할 수 없다 — 그런데 거절의 근거는 양적인 것인데 그것을
    # 재는 것이 아무것도 없었다. ``(1 + KS) / 2``는 이 랭킹의 어떤 컷이든 허용하는 최고
    # balanced_accuracy이므로(automl_agent.scoring.ranking), 차이가 정확히 상금의 크기다 — 그것이
    # "실행기는 그것을 하지 않는다"를 "그것은 이만큼을 살 것이고, 격차의 나머지는 랭킹에 있다"로
    # 바꾼다.
    #
    # specificity와 train_val_gap처럼 진단값이다. 목표로 삼을 수 없다: 실행기가 적용하지 않을 컷에서
    # 재므로, 그것에 대고 세운 목표는 harness가 실제로 돌리는 어떤 것으로도 달성될 수 없다.
    ceiling = best_cut_ceiling(ks_statistic(y_eval, proba))
    if ceiling is not None:
        best_cut, headroom = CUT_DIAGNOSTICS
        metrics[best_cut] = ceiling
        if "balanced_accuracy" in metrics:
            metrics[headroom] = round(ceiling - metrics["balanced_accuracy"], 6)

    # 컷이 실제로 내린 자리. 가장 좋은 컷이 얼마짜리였을지 옆에 공표한다. 컷이 적용됐을 때만 쓰므로
    # 그것이 *없다*는 것이 시도가 기본 규칙을 썼다는 말이 된다 — 이 키가 있기 전의 모든 결과가 여전히
    # 옳게 읽히는 것이 그 덕이다. 위의 둘은 신탁 진단값으로 남는다: 그것들은 *이* 행에서 재고, 적용된
    # 컷은 학습 행에서 골랐으니, ``balanced_accuracy_cut_headroom``은 0이 아니라 여전히 정직한
    # 잔차다.
    if threshold is not None:
        metrics["applied_threshold"] = threshold

    # 출력 CSV의 확률이 말하는 그대로를 뜻하는지. 위의 모든 지표는 랭킹에 관한 것이거나 sklearn의
    # 고정 0.5 컷에서의 0/1 판정에 관한 것이고, 모델은 둘 다 완벽하면서도 체계적으로 과신할 수 있다 —
    # 그것이 여기서 문제가 되는 이유는 정확히, harness가 임계값 고르기를 거절하므로 호출자가 자기
    # 임계값으로 행동하는 부분이 확률이기 때문이다. 위의 둘과 같은 조건의 진단값이다: 재고, 보고하고,
    # 절대 목표로 삼을 수 없고, 재보정으로 되먹이지 않는다
    # (:mod:`automl_agent.scoring.calibration`).
    if proba is not None:
        metrics.update(calibration_measure(y_eval, proba))

    # 이 행에서 목표 지표가 얼마나 넓은지. ``--metric``은 registry 키만 받으므로
    # (``metrics.GOAL_METRICS``) 여기서 alias를 따라 적을 필요가 없다 — 구간은 목표가 적힌 그 이름
    # 아래에 공표된다.
    target = canonical(interval_metric) if interval_metric else None
    if target and target in metrics:

        def one(y_slice: Any, pred_slice: Any, proba_slice: Any) -> float:
            return float(scorers(y_slice, pred_slice, proba_slice, average, task)[str(target)]())

        interval = bootstrap_interval(
            one, y_eval, pred, proba, groups=groups, seed=seed, resamples=resamples
        )
        if interval is None:
            log.write(f"{target}: no confidence interval (split too small, or metric degenerate)")
        else:
            metrics.update(interval.flatten(target))
            log.write(
                describe_interval(target, metrics[target], interval.bounds)
                + f" — {interval.unit} 단위 재표집 {interval.resamples}회"
            )
    return metrics


def save_model(model: Any, path: Path, log: LogBuffer) -> str | None:
    """적합된 파이프라인을 남긴다. 그 경로를 돌려주거나, 쓸 수 없었으면 ``None``.

    절대 치명적이지 않다. 이것이 돌 때쯤 점수는 이미 얻은 것이고, 디스크가 찼다고 실행을 잃는 것이
    파일을 잃는 것보다 나쁜 실패다. 그 경우 마지막 홀드아웃 채점이 "skipped"로 내려간다
    (``nodes/holdout.py``).

    파일은 ``.gitignore``가 막는 실행 아티팩트 디렉터리 안에 머물고, 그 경로는
    ``privacy.PUBLIC_RESULT_FIELDS``에 들지 않는다. 정리벽이 아니라 일부러다: 적합된 ``SVC``는
    support vector를 저장하고 트리는 실제 값에서 읽은 임계값을 저장하니, 모델 파일은 데이터와
    동등하고 날 행과 같은 경계의 사적인 쪽에 속한다.
    """
    try:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except (OSError, ImportError, TypeError, ValueError) as exc:
        log.write(f"model not saved: {type(exc).__name__}: {exc}")
        return None
    size = path.stat().st_size
    line = f"model saved to {path.name} ({file_size_text(size)})"
    if size > LARGE_MODEL_BYTES:
        # 이것을 정하는 것은 반복이 받은 하이퍼파라미터이고, 그것을 보는 것은 달리 없다:
        # ``guard_memory``는 입력 행렬에 값을 매기는데, 그 크기는 추정기가 그것으로 무엇을 하든
        # 같다. 깊이 제한 없는 forest가 실제 실행의 한 반복에서 4만 7천 행 위에 486 MB를 썼으니,
        # 반복 다섯 번 루프에서 진짜 한계는 디스크다.
        line += (
            f" — over {file_size_text(LARGE_MODEL_BYTES)}, driven by this iteration's "
            "hyperparameters (tree count and depth, mostly). The run keeps it; the disk is "
            "what bounds a long loop, not memory"
        )
    log.write(line)
    return str(path)




def save_schema(schema: dict[str, Any] | None, path: Path, log: LogBuffer) -> str | None:
    """모델이 적합된 인코딩을 남긴다. 그 경로를 돌려주거나 ``None``.

    :func:`save_model`처럼 절대 치명적이지 않다. 잃는 것은 구체적이다: 모델 파일만으로는 어느 입력
    열이 어느 것이었는지 말할 수 없으니, 스키마 없이는 마침 같은 배치로 인코딩되는 행에만 적용할 수
    있고 — 그렇게 됐는지를 검사하는 것이 아무것도 없다. ``predict``는 추측하지 않고 거절하는데, 이것을
    애초에 쓰는 이유가 그것이다.

    ``None`` 스키마는 오류가 아니다 — 합성 경로는 파일을 인코딩하는 대신 행렬을 생성하니, 다시 재생할
    인코딩이 없다.

    데이터와 동등하므로(카테고리 수준과 클래스 라벨은 셀 값이다) 모델 옆 ``artifacts/``에 내리고 그
    경로는 ``privacy.PUBLIC_RESULT_FIELDS`` 밖에 머문다.
    """
    if schema is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        log.write(f"feature schema not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"feature schema saved to {path.name} ({len(schema.get('columns') or ())} columns)")
    return str(path)


def load_schema(path: Path, log: LogBuffer) -> dict[str, Any] | None:
    """모델 옆에 저장된 스키마를 읽는다. 읽을 것이 없으면 ``None``.

    ``None``은 계속 작동해야 하는 두 경우를 덮는다: 합성 실행(인코딩이 적합된 적 없음), 그리고 이
    파일이 있기 전의 실행. 둘 다 배치를 다시 유도하는 쪽으로 떨어진다.

    읽을 수 없거나 형태가 깨진 파일은 없음으로 다루지 *않는다* — 그러면 손상된 스키마가 열이 폭만으로
    줄 세워지는 채점 실행이 되고, 그것이 이 아티팩트가 막는 바로 그 실패다.
    """
    if not path.exists():
        log.write(f"no feature schema beside the model ({path.name}); deriving the encoding from the file")
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"feature schema {path} is not a JSON object")
    log.write(f"encoding from {path.name}: {len(loaded.get('columns') or ())} columns")
    return loaded


def save_predictions(
    path: Path, pred: Any, proba: Any, fingerprint: str, log: LogBuffer
) -> str | None:
    """이 시도의 검증 행 예측을 남긴다. 그래야 나중 시도가 그것에 짝지어 비교할 수 있다.

    :func:`save_model`처럼 절대 치명적이지 않다 — 이것이 돌 때쯤 점수는 얻은 것이고, 잃는 것은 나중
    시도의 짝지은 판정이며 그것은 이유를 달고 ``skipped``로 내려간다.

    **검증 행마다 값 하나이므로 ``model.joblib``과 정확히 같은 의미로 데이터와 동등하고** 경계의 같은
    쪽에 머문다: ``artifacts/`` 안, 내용은 절대 state 채널에 들어가지 않고, 경로는 state로 실려 오는
    것이 아니라 반복 번호에서 유도된다
    (:meth:`automl_agent.config.RunConfig.predictions_path`).

    ``y_val``은 일부러 없다 — 짝짓는 프로세스가 같은 행을 분할했고 라벨을 이미 갖고 있으며, 그것이
    같은 행*임을* 세우는 것은 지문이다.
    """
    try:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "pred": np.asarray(pred),
            "fingerprint": np.asarray(str(fingerprint)),
            # 이 수들이 만들어진 스레드 상태. 지문과 같은 이유로 함께 따라간다: 짝지은 비교의
            # 전제는 두 벡터가 계획이 바꾼 것만큼만 다르다는 것인데, 스레드 수는 어떤 계획보다도
            # 더 많이 그것을 바꾼다 (:mod:`automl_agent.threads`). 지문은 행을 보고 이것은 볼 수
            # 없다.
            "threads": np.asarray(json.dumps(thread_state(), ensure_ascii=False, sort_keys=True)),
        }
        if proba is not None:
            arrays["proba"] = np.asarray(proba)
        np.savez_compressed(path, **arrays)
    except (OSError, ImportError, TypeError, ValueError) as exc:
        log.write(f"val predictions not saved: {type(exc).__name__}: {exc}")
        return None
    log.write(f"val predictions saved to {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    return str(path)


def paired_against_baseline(
    cfg: dict[str, Any],
    y_val: Any,
    pred: Any,
    proba: Any,
    fingerprint: str,
    *,
    metric: str,
    average: str,
    task: str,
    log: LogBuffer,
    groups: Any = None,
    seed: int = 42,
) -> dict[str, Any]:
    """이 시도의 목표 지표를 지금까지의 최고와 대 놓고 잰, 재표집된 차이.

    결과를 읽는 노드가 아니라 여기서 계산하는 것은 이 파일이 애초에 subprocess인 이유와 같다:
    orchestrator는 numpy를 import하지 않는다. **그리고 방금 채점한 배열이 바로 여기에 있다** — 두
    번째 subprocess라면 config에서 분할을 다시 세워야 하고, 두 번째 유도는 어긋날 수 있는 두 번째
    물건이다.

    **언제나 블록을 돌려주고, 건너뛴 비교는 이유를 공표한다** — "짝지은 판정이 없음"과 "짝지은 판정이
    아무것도 찾지 못함"은 다른 사실이고, 첫 번째에 대한 침묵은 두 번째로 읽힌다.
    """
    declared = dict(cfg.get("paired_baseline") or {})
    iteration = declared.get("iteration")
    raw_path = declared.get("path")
    block: dict[str, Any] = {"metric": metric, "baseline_iteration": iteration}

    def skipped(reason: str, detail: str = "") -> dict[str, Any]:
        log.write(f"paired comparison skipped ({reason}){f': {detail}' if detail else ''}")
        return {**block, "status": PAIRED_SKIPPED, "reason": reason}

    if not raw_path or iteration is None:
        return skipped("no_baseline")

    import numpy as np

    path = Path(str(raw_path))
    try:
        # ``allow_pickle=False``: 이 파일은 이 스크립트가 쓰고 이 스크립트가 읽는다. 아티팩트
        # 디렉터리에 pickle을 실은 npz가 있는 것은 측정으로 차려입은 코드 실행이다.
        with np.load(path, allow_pickle=False) as loaded:
            stored = str(loaded["fingerprint"])
            base_pred = np.asarray(loaded["pred"])
            base_proba = np.asarray(loaded["proba"]) if "proba" in loaded.files else None
            # 이 필드가 있기 전에 쓰인 파일에는 없다. 그럴 때는 추측이 아니라 ``None``이다:
            # ``thread_state_changed``는 "기록되지 않음"을 "바뀌지 않음"이 아니라 침묵으로 바꾼다.
            # ``JSONDecodeError``는 ``ValueError``이므로, 손상된 필드가 손상된 파일과 같은 핸들러에
            # 내린다.
            base_threads = (
                json.loads(str(loaded["threads"])) if "threads" in loaded.files else None
            )
    except (OSError, ValueError, KeyError) as exc:
        return skipped("baseline_missing", f"{path.name}: {type(exc).__name__}: {exc}")

    if stored != fingerprint:
        # 비교 전체의 전제. seed에서 단정하지 않고 검사한다 — seed를 바꾸지 않고 행을 바꾸는
        # 것이 무엇인지는 :func:`automl_agent.scoring.splits.val_fingerprint` 참고.
        return skipped("split_changed", f"{stored[:8]} != {fingerprint[:8]}")

    # 두 번째 전제이고, 첫 번째와 달리 치명적이지 않다: 기준의 예측이 디스크에 있으니 두 벡터의
    # 차이는 여전히 정확히 일어난 그 차이다. 스레드 변화가 값으로 가져가는 것은 *귀속*이다 — 그러면
    # 델타의 일부가 계획이 아니라 환경이다 — 그래서 거절하지 않고 델타 옆에 공표한다. 어느 쪽이든
    # 기록되지 않았으면 ``None``이고, ``None``은 아무것도 공표하지 않는다.
    current_threads = thread_state()
    changed = thread_state_changed(base_threads, current_threads)
    if changed is not None:
        block["threads_changed"] = changed
    if changed:
        log.write(
            f"baseline iteration {iteration} was fitted in a different thread state: "
            f"[{describe_thread_state(base_threads)}] -> [{describe_thread_state(current_threads)}]"
            " — part of the delta below is the environment, not the plan"
        )

    spec = METRICS.get(metric)
    if spec is not None and spec.needs_proba and (proba is None or base_proba is None):
        return skipped("degenerate", f"{metric} needs probabilities and one side has none")

    def one(y_slice: Any, pred_slice: Any, proba_slice: Any) -> float:
        return float(scorers(y_slice, pred_slice, proba_slice, average, task)[metric]())

    delta = paired_delta(
        one,
        y_val,
        (pred, proba),
        (base_pred, base_proba),
        direction=direction_of(metric),
        groups=groups,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
    )
    if delta is None:
        return skipped("degenerate", f"{metric}: no interval over the difference")
    measured = {
        **block,
        "status": PAIRED_MEASURED,
        "resamples": delta.resamples,
        "unit": delta.unit,
        **delta.flatten(),
    }
    log.write(f"{metric} vs iteration {iteration}: {describe_paired(measured)}")
    return measured


@dataclass
class TrainingRun:
    """한 번의 적합이 낸 것. :func:`run_training`이 돌려준다.

    아홉 중 넷은 실행이 채점한 것이 아니라 무엇으로 *설정됐는지*를 말하고, 다시 유도하지 않고 실어
    내보내는 것은 추정기를 본 것이 ``run_training``뿐이기 때문이다: config는 실행기가 좁힐 수 있는
    요청이다. ``paired``는 지금까지의 최고에 대고 한 비교이고, 호출자가 요청하지 않았으면 비어 있다
    (:func:`paired_against_baseline`). ``schema_path``는 ``model_path``의 나머지 절반이다 — 그 모델이
    적합된 인코딩이고, 그것이 없으면 모델을 불러올 수는 있어도 적용할 수는 없다.
    ``internal_validation``은 early stopping이나 결정 컷이 적합에서 떼어 둔 학습 행이고
    (:func:`describe_internal_validation`, :func:`fit_estimator`), 떼어 둔 것이 없으면 비어 있다.

    한때 9-튜플이었던 것을 dataclass로 바꿨다. 위치 기반 해체는 하나를 읽으려고 모든 호출자가 아홉을
    다 이름 짓게 만들었고 — 한 테스트는 ``metrics``와 ``internal_validation``에
    닿으려고 버리는 이름 일곱을 썼다 — 그만큼 긴 튜플은 조용한 순서 바뀜의 위험이기도 하다. 인접한
    ``dict[str, Any]`` 필드 둘은 타입 오류 없이 자리를 바꾼다.
    """

    metrics: dict[str, float]
    applied: dict[str, Any]
    dropped: list[str]
    preprocessing: dict[str, Any]
    model_path: str | None
    paired: dict[str, Any]
    schema_path: str | None
    internal_validation: dict[str, Any]
    applied_pipeline: list[str] = field(default_factory=list)


@dataclass
class HeldBackCut:
    """결정 컷 요청의 결말, 그리고 그것을 고르려고 학습 행에서 무엇을 뺐는지.

    ``request``는 거절을 다 거친 뒤 남은 것이고 ``asked``는 계획이 청한 것이다. 둘을 따로 드는 이유는
    ``applied_threshold``의 부재가 컷이 효력이 없다고만 말하고 컷이 원해진 적이 있는지는 말하지 못하기
    때문이다 — ``dropped_hyperparams``가 있는 이유와 같다. ``declined``가 그 이유를 적는다.

    ``x_fit``/``y_fit``/``groups_fit``은 조각을 뺀 *뒤의* 학습 행이다. 컷이 거절됐으면 들어온 것
    그대로이니, 요청하지 않은 시도는 늘 적합돼 온 바로 그 행에 적합된다.
    """

    request: float | str | None
    asked: float | str | None
    declined: str | None
    rows: dict[str, Any]
    x_fit: Any
    y_fit: Any
    groups_fit: Any
    x_cut: Any
    y_cut: Any


def hold_back_cut(
    cfg: dict[str, Any],
    x_train: Any,
    y_train: Any,
    groups_train: Any,
    seed: int,
    *,
    task: str,
    n_classes: int,
    log: LogBuffer,
) -> HeldBackCut:
    """결정 컷을 고를 행을 적합 *전에* 학습 분할에서 빼낸다.

    추정기가 그 행을 절대 보지 않도록 여기서, 적합 전에 한다. 거절이 넷이고 모두 조각을 잘라내기
    *전에* 검사하는 것이 이 함수의 요점이다: 쓸기가 알아낼 때쯤 조각은 이미 적합에서 빠져 있고, 시도는
    고를 수도 없었던 손잡이에 학습 행의 20%를 치른 것이 된다. 어떤 실제 실행이 그렇게 5,658행을 냈고
    유일한 흔적은 ``applied_threshold``의 부재였다.
    """
    request = requested_cut(cfg, log)
    # 아래의 모든 거절이 ``request``를 비우므로 청해진 것은 따로 들고 있는다. 여기서 한 번만 잡는
    # 이유는 ``requested_cut``이 로그를 쓰기 때문이다 — 두 번 부르면 줄이 두 번 남는다.
    asked = request
    regression = task == TASK_REGRESSION
    goal = goal_metric(cfg, task)
    declined: str | None = None
    if request is not None and (METRICS.get(goal) or METRICS["f1"]).needs_proba:
        # ``roc_auc``와 ``pr_auc``는 확률만으로 계산되니 모든 후보 컷이 같은 점수를 받고
        # ``tune_threshold``는 애초에 거절할 예정이었다.
        log.write(
            f"decision.threshold={request!r} ignored: the goal metric {goal} is computed from "
            "the probabilities, not from the labels, so no cut changes it"
        )
        request, declined = None, f"the goal metric {goal} is cut-invariant"
    if request is not None and (regression or n_classes != 2):
        # 연속형 타깃에는 고를 컷이 없고 다중 클래스에는 단일 컷이 없으니 ``tune_threshold``는 여기서
        # 애초에 ``None``으로 답할 예정이었다.
        log.write(
            f"decision.threshold={request!r} ignored: "
            + (
                "the target is continuous, so there is no decision rule to move"
                if regression
                else f"the target has {n_classes} classes and a single cut is a binary rule"
            )
        )
        request = None
        declined = (
            "the target is continuous" if regression else f"the target has {n_classes} classes"
        )
    # 조각을 떼지 않은 결말. 셋이 여기로 온다: 컷을 청하지 않은 config, 위에서 거절된 요청, 그리고
    # 아래에서 조각이 너무 작다고 거절되는 것.
    kept = HeldBackCut(
        request=request,
        asked=asked,
        declined=declined,
        rows={},
        x_fit=x_train,
        y_fit=y_train,
        groups_fit=groups_train,
        x_cut=None,
        y_cut=None,
    )
    if request != DECISION_TUNED:
        return kept
    fit_idx, cut_idx = held_back_indices(
        x_train, y_train, seed, not regression, groups_train, CUT_FRACTION
    )
    if len(cut_idx) < MIN_CUT_ROWS:
        # 줄이지 않고 끈다. 크기가 모자란 멈춤 조각과 같은 조건이다: 40행에서 고른 컷은 보고서가 면책
        # 문구를 붙여야 할 수이고, 기본 규칙은 적어도 규칙이다. 아래쪽의 어떤 것도 컷이 요청됐다고
        # 믿지 않도록 ``request``를 비운다.
        log.write(
            f"decision.threshold={DECISION_TUNED!r} not applied: the slice held back to "
            f"choose the cut would hold {len(cut_idx)} rows, under the {MIN_CUT_ROWS} needed"
        )
        kept.request = None
        kept.declined = (
            f"the slice to choose the cut on would hold {len(cut_idx)} rows, "
            f"under the {MIN_CUT_ROWS} needed"
        )
        return kept
    log.write(
        f"decision.threshold={DECISION_TUNED!r}: {len(cut_idx)} of train's rows are held "
        f"out of the fit to choose the cut on, leaving {len(fit_idx)} to fit — the price "
        "of the lever, and reported in internal_validation"
    )
    return HeldBackCut(
        request=request,
        asked=asked,
        declined=None,
        rows={"cut_held_out_rows": len(cut_idx), "cut_fraction": CUT_FRACTION},
        x_fit=x_train[fit_idx],
        y_fit=y_train[fit_idx],
        groups_fit=None if groups_train is None else groups_train[fit_idx],
        x_cut=x_train[cut_idx],
        y_cut=y_train[cut_idx],
    )


def held_back_by_estimator(model: Any, n_train: int, log: LogBuffer) -> dict[str, Any]:
    """추정기 *자신의* early stopping이 떼어 간 학습 행, 적합된 객체에서 읽어서.

    ``fit_estimator``가 아무것도 말하지 않았을 때만 물을 것이다. 학습 행을 떼어 둘 수 있는 추정기가
    둘인데 그중 나중에 물어볼 수 있는 것은 하나뿐이다: harness가 직접 분할했으면(xgboost의 eval set)
    ``fit_estimator``가 개수를 돌려주고 이미 로그에 남겼고, 추정기가 자기 것을 만들었으면 적합된 객체
    말고는 아는 것이 없다. 어느 쪽이든 적합 뒤다 — 그 전에는 읽을 것이 없기 때문이다.
    """
    held = describe_internal_validation(
        (getattr(model, "named_steps", None) or {}).get("model"), n_train
    )
    if held:
        stopped = (
            f", iteration {held['stopped_at_iter']}/{held['max_iter']}에서 멈춤"
            if {"stopped_at_iter", "max_iter"} <= held.keys()
            else ""
        )
        log.write(
            f"early_stopping이 위 {n_train}행 중 {held['held_out_rows']}행을 자체 검증으로 "
            f"떼어 갔습니다 — 실제 학습은 {held['fit_rows']}행{stopped}"
        )
    return held


def record_train_val_gap(
    metrics: dict[str, float], target_metric: str, task: str, log: LogBuffer
) -> None:
    """``metrics["train_val_gap"]``을 제자리에 적는다 — 잴 두 점수가 다 있을 때만.

    태스크의 첫 기본값이 아니라 목표 지표에서 정의한다: Critic의 과적합/과소적합 분기가 읽는 것이 이 한
    수이고, 아무도 최적화하지 않는 지표에서 잰 격차는 그것에 틀린 진단을 보낸다. 대체값은 train 짝이
    없는 목표 지표를 위한 것이다(predict_proba 없는 모델에서의 확률 지표).
    """
    gap_metric = target_metric if f"train_{target_metric}" in metrics else TRAIN_METRICS[task][0]
    if gap_metric not in metrics or f"train_{gap_metric}" not in metrics:
        return
    # 언제나 "검증이 학습보다 얼마나 나쁜가"이고, 맨 뺄셈이 아니다: 최소화 지표에서는 학습 점수가
    # *더 작은* 수이므로 ``train - val``은 모델이 과적합일 때 정확히 음수가 된다. 모든 소비자가 양수
    # 격차를 과적합으로 읽으니(``nodes/critic.py``, Critic 프롬프트의 규칙) 방향은 각자에게가 아니라
    # 여기 한 곳에서 적용한다.
    train_value, val_value = metrics[f"train_{gap_metric}"], metrics[gap_metric]
    worse_by = (
        val_value - train_value
        if direction_of(gap_metric) == MINIMIZE
        else train_value - val_value
    )
    metrics["train_val_gap"] = round(worse_by, 6)
    log.write(f"train_val_gap measured on {gap_metric} ({direction_of(gap_metric)})")


def run_training(
    cfg: dict[str, Any],
    log: LogBuffer,
    model_out: Path | None = None,
    predictions_out: Path | None = None,
    schema_out: Path | None = None,
    decision_out: Path | None = None,
) -> TrainingRun:
    """적합하고 채점한다. :class:`TrainingRun`을 돌려준다.

    점수는 검증 집합 점수다: test 조각(:mod:`automl_agent.scoring.splits`)은 여기서 아예 읽지 않으니,
    루프가 그것에 대고 고르는 것이 그것을 만졌을 수 없다.
    """
    seed = int(cfg.get("seed", 42))
    x_arr, y_arr, n_classes, groups, task, schema = load_data(cfg, log)
    guard_memory(x_arr, cfg, log)
    regression = task == TASK_REGRESSION

    group_column = (cfg.get("data") or {}).get("group_column")
    splits = split_three_way(x_arr, y_arr, seed, groups=groups, stratify=not regression)
    x_train, y_train = splits.x_train, splits.y_train
    x_val, y_val = splits.x_val, splits.y_val
    log.write(
        describe_protocol(protocol(seed, group_column, stratified=not regression))
        + f" {splits.sizes}"
    )

    # ``fit_estimator``가 그것을 다시 분할할 수 있으니 x_train과 함께 실려 간다. 둘은 같은 길이를
    # 유지해야 하므로 아래의 subsample이 둘 다 자른다.
    groups_train = splits.groups_train

    subsample = as_number((cfg.get("hyperparams") or {}).get("train_subsample"))
    if subsample is not None and 0 < subsample < 1:
        keep = max(50, int(len(x_train) * subsample))
        x_train, y_train = x_train[:keep], y_train[:keep]
        if groups_train is not None:
            groups_train = groups_train[:keep]
        log.write(f"train_subsample={subsample} -> {keep} rows")

    # 결정 컷을 고를 행, 적합 전에 빼낸다 — 거절 넷과 그 이유는 ``hold_back_cut``. 지역 이름으로 풀어
    # 두는 이유는 아래에서 ``cut_declined``가 두 번 더 채워지고 ``cut_rows``가 자라기 때문이다.
    cut = hold_back_cut(cfg, x_train, y_train, groups_train, seed, task=task, n_classes=n_classes, log=log)
    request, cut_requested, cut_declined = cut.request, cut.asked, cut.declined
    x_cut, y_cut, cut_rows = cut.x_cut, cut.y_cut, cut.rows
    x_train, y_train, groups_train = cut.x_fit, cut.y_fit, cut.groups_fit

    # 선언된 파이프라인. 단계가 *열*을 지목하고 어느 열이 어느 것인지 말하는 것이 스키마이므로
    # 여기서 해석한다. ``None`` — 키가 없음 — 은 플래그 경로를 그대로 두니, 이 블록보다 먼저 쓰인
    # 모든 config가 전과 똑같이 행동한다.
    declared, applied_pipeline = declared_steps(cfg, schema, x_train.shape[1], task, log)

    model, applied, dropped = build_estimator(
        str(cfg.get("model") or "hist_gbdt"),
        dict(cfg.get("hyperparams") or {}),
        seed,
        log,
        dict(cfg.get("preprocessing") or {}),
        # ``class_weight`` 매핑이 정확히 덮어야 하는 라벨. sklearn이 그 매핑을 대고 검증할 대상인
        # 학습 분할에서 딴다. 연속형 타깃에서는 ``None``이다: 가중치를 줄 클래스가 없고, 열의 서로
        # 다른 값을 다 열거하는 것은 쓸모없는 동시에 셀 값의 집합이다.
        labels=None if regression else sorted(set(y_train.tolist())),
        task=task,
        declared=declared,
    )
    log.write(f"fitting on {len(x_train)} rows, validating on {len(x_val)} rows")
    internal_validation = fit_estimator(
        model,
        x_train,
        y_train,
        seed,
        log,
        stratify=not regression,
        groups=groups_train,
    ) or held_back_by_estimator(model, len(x_train), log)
    model_path = save_model(model, model_out, log) if model_out is not None else None
    # 적합 뒤에, 모델 옆에 쓴다. 그래야 디스크의 파일이 나중에 따로 유도된 것이 아니라 이 추정기가
    # 실제로 본 행렬의 인코딩을 기술한다.
    schema_path = save_schema(schema, schema_out, log) if schema_out is not None else None

    # 여기서는 조용하다: 위의 ``load_data``가 대체가 있었으면 이미 로그에 남겼고, 이것은 같은
    # 태스크로 하는 같은 호출이다.
    target_metric = goal_metric(cfg, task)
    average = "binary" if n_classes == 2 else "macro"
    pred_train = model.predict(x_train)
    # 여기서 한 번 예측하고, 그다음 채점도 저장도 한다. 나중 반복이 짝지어 비교할 배열은 이 반복의
    # 지표가 계산된 그 배열이어야 한다. 그러지 않으면 파일이 아무것도 채점되지 않은 패스를 기술한다.
    pred_val = model.predict(x_val)
    proba_val = _proba(model, x_val, n_classes, log)

    # 운영점. 무엇이 채점되거나 저장되기 전에 정해진다. 튜닝 경로는 위에서 잘라낸 행을 읽는다 —
    # 이 추정기가 적합된 ``x_train``도 아니고, 이제 채점될 ``x_val``도 아니다.
    # 좁혀서 돌려받지 않고 여기서 좁히는 것은, ``requested_cut``에 정직한 답이 둘이고(키워드와 수)
    # 그중 하나는 적합 뒤에만 풀 수 있기 때문이다. 분기를 명시해서 타입이 코드가 늘 해 온 것을 말하게
    # 한다: 이 지점을 지나면 임계값은 float이거나 없음이고, 절대 키워드가 아니다.
    threshold: float | None
    if request == DECISION_TUNED:
        threshold = tune_threshold(
            y_cut, _proba(model, x_cut, n_classes, log), target_metric, average, task, log
        )
    else:
        threshold = as_number(request)
    proba_train = None
    if threshold is not None:
        proba_train = _proba(model, x_train, n_classes, log)
    if threshold is not None and (proba_train is None or proba_val is None):
        log.write(f"decision.threshold={threshold} not applied: this model gives no probabilities")
        threshold, cut_declined = None, "this model gives no probabilities"
    if request is not None and threshold is None and cut_declined is None:
        # 쓸기 자신이 거절했다 — 받아들일 후보가 없거나, 더 잘 받은 후보가 없다. 그쪽 이유는 로그에
        # 있다. 이것은 적합이 이미 조각 값을 치른 뒤에 컷이 원해졌고 아무것도 나오지 않았다고 말하는
        # 한 줄이다.
        cut_declined = "the sweep found no cut worth applying"
    if threshold is not None:
        # 같은 컷으로 두 분할 다. ``pred_val``은 채점·저장·짝짓기 *전에* 다시 라벨링된다. 그래야
        # 나중 시도가 짝지어 비교할 파일이 이 시도 자신의 지표가 계산된 그 라벨을 담는다.
        # ``pred_train``도 반대쪽에서 같은 이유로 다시 라벨링되니, ``train_val_gap``이 한 규칙에서 잰
        # 두 점수를 비교한다.
        pred_train = label_at_cut(proba_train, threshold)
        pred_val = label_at_cut(proba_val, threshold)
        if decision_out is not None:
            save_decision(
                {
                    "threshold": threshold,
                    # 어느 행이 그것을 골랐는지. 그 수를 알 값이 있게 만드는 것이 그것이기
                    # 때문이다: ``held_back``은 적합과 보고 둘 다 밖의 행에서 재어졌고,
                    # ``config``는 위에서 내려왔고 아무것에서도 재어지지 않았다.
                    "chosen_on": "held_back" if request == DECISION_TUNED else "config",
                    "metric": target_metric,
                    **cut_rows,
                },
                decision_out,
                log,
            )
    if cut_requested is not None:
        # 행 옆에 둔다. 값을 읽는 사람이 바로 그 값으로 무엇을 샀는지 알아야 하는 사람이기
        # 때문이다. 컷이 요청됐을 때만 쓰므로 부재는 계속 "요청한 계획이 없었다"를 뜻한다 — 이 키가
        # 있기 전의 모든 결과가 여전히 옳게 읽히는 것이 그 덕이고, ``applied_threshold``의 관례
        # 그대로다.
        cut_rows["cut_requested"] = cut_requested
        if cut_declined is not None:
            cut_rows["cut_declined"] = cut_declined
    if cut_rows:
        # 자기 채널을 따로 두지 않고, 시도가 적합하지 않은 행을 이미 보고하는 채널에 합친다: 시도를
        # 읽는 사람에게 그것은 같은 사실이고, 키가 서로 달라서 같은 dict 안의 early-stopping 조각도
        # 여전히 그것으로 읽힌다.
        #
        # 그 행에서 컷이 나왔든 안 나왔든 보고한다. 어느 쪽이든 적합은 20% 적게 봤으니, 쓸기가
        # 거절한 시도는 — 컷 불변 목표 지표, 확률 없는 모델 — 값을 치렀는데도 그러지 않으면 train
        # 전체에 적합된 시도와 점수 비교 가능해 보인다. 이 필드가 막으려고 존재하는 읽기가 그것이다.
        internal_validation = {**internal_validation, **cut_rows}
    applied_pipeline = (
        describe_pipeline(model, applied_pipeline) if declared is not None else []
    )
    metrics = evaluate_split(
        model,
        x_val,
        y_val,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=target_metric,
        # 파일 전체가 아니라 검증 분할 자신의 그룹 라벨: 이 구간은 점수가 재어진 그 행에 관한
        # 것이다.
        groups=splits.groups_val,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
        pred=pred_val,
        proba=proba_val,
        threshold=threshold,
    )

    fingerprint = val_fingerprint(x_val, y_val)
    if predictions_out is not None:
        save_predictions(predictions_out, pred_val, proba_val, fingerprint, log)
    paired = paired_against_baseline(
        cfg,
        y_val,
        pred_val,
        proba_val,
        fingerprint,
        metric=target_metric,
        average=average,
        task=task,
        log=log,
        groups=splits.groups_val,
        seed=seed,
    )

    # 학습 점수가 있으면 Critic이 과소적합과 과적합을 구분할 수 있다. 태스크의 기본 지표 둘은 목표
    # 지표가 무엇이든 result.json의 모양에 속한다. 목표 지표를 더하는 것은, 아래의 격차가 실행이
    # 심판받는 그 수에 관한 것이어야 하기 때문이다.
    train_names = list(TRAIN_METRICS[task])
    if target_metric in METRICS and target_metric not in train_names:
        train_names.append(target_metric)
    train_proba = proba_train
    if train_proba is None and any(METRICS[name].needs_proba for name in train_names):
        train_proba = _proba(model, x_train, n_classes, log)
    for name, value in score_split(
        train_names, y_train, pred_train, train_proba, average, log, task
    ).items():
        metrics[f"train_{name}"] = value

    record_train_val_gap(metrics, target_metric, task, log)

    log.write("metrics: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return TrainingRun(
        metrics=metrics,
        applied=applied,
        dropped=dropped,
        preprocessing=describe_preprocessing(model),
        model_path=model_path,
        paired=paired,
        schema_path=schema_path,
        internal_validation=internal_validation,
        applied_pipeline=applied_pipeline,
    )


def score_saved_model(
    cfg: dict[str, Any], log: LogBuffer, model_path: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """앞선 프로세스에서 적합된 모델로 떼어 둔 test 분할을 채점한다.

    **아무것도 적합하지 않고, ``x_test`` 말고는 아무것에도 예측하지 않는다** — 이 수의 요점은 실행의
    어떤 결정도 이 행에 대고 내려지지 않았다는 것이다. 분할은 저장하지 않고 다시 세운다: 그것은 파일,
    타깃 정책, 그룹 열, seed의 순함수이고, 넷 다 적합이 쓴 그 ``train_config.json``에서 온다.

    세 가지가 config가 아니라 디스크에서 온다. config는 기록이 아니라 요청이기 때문이다:

    * **전처리**. 불러온 파이프라인에서 읽는다 — 이 프로세스는 파이프라인을 세운 적이 없다.
    * **인코딩**. 모델 옆 스키마에서. 그래야 행이 이 추정기가 적합된 배치로 내린다. 다시 유도하면
      평범한 경우에는 일치하고 잡을 값이 있는 경우에는 **조용히 어긋난다**. 스키마가 없으면(더 오래된
      실행) 옛 방식으로 채점하고 로그에 남긴다.
    * **결정 규칙**. 그리고 여기서 그것을 다시 유도하는 것은 존재하는 선택지가 아니다. 이것은 test
      행이고, 그 행에서 컷을 고르지 *않는* 것이 이 함수가 존재하는 이유이기 때문이다. 파일이 없으면
      시도가 컷을 튜닝한 적이 없다는 뜻이다.
    """
    import joblib

    seed = int(cfg.get("seed", 42))
    schema = load_schema(model_path.parent / SCHEMA_FILENAME, log)
    threshold = load_decision(model_path.parent / DECISION_FILENAME, log)
    x_arr, y_arr, n_classes, groups, task, _schema = load_data(cfg, log, schema)
    splits = split_three_way(
        x_arr, y_arr, seed, groups=groups, stratify=task != TASK_REGRESSION
    )
    model = joblib.load(model_path)
    average = "binary" if n_classes == 2 else "macro"
    log.write(
        f"scoring the held-back test split: {len(splits.y_test)} rows, "
        f"model={model_path.name}"
    )
    metrics = evaluate_split(
        model,
        splits.x_test,
        splits.y_test,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=canonical(str(cfg.get("metric") or "f1")),
        groups=splits.groups_test,
        seed=seed,
        resamples=bootstrap_resamples(cfg),
        threshold=threshold,
    )
    log.write("test metrics: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return metrics, describe_preprocessing(model)


# --------------------------------------------------------------------------- #
# 결과 배관
# --------------------------------------------------------------------------- #


def classify_exception(exc: BaseException) -> str:
    """예외를 Critic이 추론할 수 있는 거친 ``error_type``에 대응시킨다."""
    if isinstance(exc, MemoryError):
        return "oom"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(token in text for token in ("out of memory", "cuda", "cannot allocate", "alloc failed")):
        return "oom"
    if isinstance(exc, (FileNotFoundError, KeyError)) or "not found" in text:
        return "data_issue"
    if isinstance(exc, ValueError) and "unsupported model" in text:
        return "unsupported_model"
    if isinstance(exc, ValueError):
        return "data_issue"
    return "exception"


# 유한하지 않은 수가 있었다면 어디에 있었는지. 사적이다: ``privacy.PUBLIC_RESULT_FIELDS``에 없다.
# 추론 프롬프트는 이것에서 필요한 단 하나를 이미 읽기 때문이다 — 지표가 ``metrics``에 *없다*는 것이고,
# 능력 목록이 그것을 "측정되지 않음"으로 정의한다. 이 목록은 result.json과 그 옆 로그를 읽는 사람을
# 위해 있다.
NONFINITE_KEY = "nonfinite_dropped"


def drop_nonfinite(value: Any, path: str = "") -> tuple[Any, list[str]]:
    """유한하지 않은 수를 모두 없앤 ``value``, 그리고 없앤 것들의 경로.

    ``result.json``을 쓰기 전의 마지막 가드이고, 생산자 하나가 아니라 *모양*을 덮는다 —
    ``score_split``은 이미 NaN 지표를 거절하지만, 다른 어떤 키로 도착한 것은 파일에 닿고
    ``json.dumps``는 그것을 맨 ``NaN``으로 쓴다.

    **맨 ``NaN``은 JSON이 아닌데 Python 자신의 ``json.loads``는 그것을 받아들인다** — 그래서 눈에 띄지
    않았다. 그러면 그 값에 대한 모든 비교가 False이니 목표 검사는 "바에 닿지 못함"으로 읽히고 보고서는
    그것을 측정값으로 찍는다. Python 밖에서 이 파일을 읽는 것은 그냥 실패한다.

    **``null``로 바꾸지 않고 버린다**: 없음은 "측정되지 않음"을 뜻하고, 그것이 metrics dict가 다른
    곳에서 이미 뜻하는 것이다. ``null``은 모든 독자가 배워야 하는 세 번째 상태를 더한다.
    """
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        dropped: list[str] = []
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            clean, gone = drop_nonfinite(item, here)
            dropped += gone
            if gone and clean is None and not isinstance(item, (dict, list)):
                continue
            kept[str(key)] = clean
        return kept, dropped
    if isinstance(value, list):
        kept_list: list[Any] = []
        dropped = []
        for index, item in enumerate(value):
            clean, gone = drop_nonfinite(item, f"{path}[{index}]")
            dropped += gone
            # 목록에서는 위치가 중요하니 — 구간은 ``[low, high]``다 — 없앤 원소는 목록을 그
            # 주위로 줄이는 대신 ``null``이 된다.
            kept_list.append(clean)
        return kept_list, dropped
    # 값이 아니라 판정만 ``as_number``에서 빌린다. 돌려주는 것은 날값 그대로여야 한다 — ``414``는
    # ``414.0``이 아니고, 이 함수는 유한하지 않은 것만 없애는 자리다.
    number = as_number(value)
    if number is None or math.isfinite(number):
        return value, []
    return None, [path or "<root>"]


def write_result(
    out_path: Path,
    *,
    status: str,
    metrics: dict[str, Any],
    train_time_sec: float,
    error_type: str | None,
    log_tail: str,
    applied_hyperparams: dict[str, Any] | None = None,
    dropped_hyperparams: list[str] | None = None,
    applied_preprocessing: dict[str, Any] | None = None,
    applied_pipeline: list[str] | None = None,
    model_path: str | None = None,
    schema_path: str | None = None,
    paired: dict[str, Any] | None = None,
    internal_validation: dict[str, Any] | None = None,
    split: str = "val",
) -> None:
    payload = {
        "metrics": metrics,
        # 이 수들이 어느 행에 관한 것인지. 언제나 있다. 검증 점수와 test 점수를 구분할 수 없는
        # 소비자는 둘이 같은 것을 뜻하는 것처럼 비교할 것이기 때문이다.
        "split": split,
        # 추정기가 실제로 무엇으로 구성됐는지, 그리고 오는 길에 무엇이 버려졌는지. 실패에도
        # 언제나 있다(그때는 적용된 것이 없다). 그래야 orchestrator가 없는 키가 "없음"을 뜻하는지
        # "옛 결과 파일"을 뜻하는지 추측할 일이 없다.
        "applied_hyperparams": dict(applied_hyperparams or {}),
        "dropped_hyperparams": list(dropped_hyperparams or []),
        # 파이프라인이 무슨 대치와 스케일링으로 세워졌는지. 그것이 늘 config가 요청한 것은 아니다
        # (:func:`describe_preprocessing` 참고). 기술할 파이프라인이 없는 실패에서는 비어 있다.
        "applied_preprocessing": dict(applied_preprocessing or {}),
        "train_time_sec": round(float(train_time_sec), 3),
        "status": status,
        "error_type": error_type,
        "log_tail": log_tail,
        # 이 수들이 만들어진 환경. 실패한 것까지 모든 결과에 붙는다 — 파일에서 가장 싼 필드이고,
        # 두 결과가 애초에 비교 가능한지를 정하는 필드다 (:mod:`automl_agent.threads`).
        # ``privacy.PUBLIC_RESULT_FIELDS``에 일부러 없다: 환경에서 나온 자유 형식 텍스트이고,
        # 프롬프트에서 진단에 쓸 데가 없으며, 소비자가 여기서 필요한 한 비트 — 비교되는 두 시도
        # 사이에서 바뀌었는지 — 는 paired 블록 안에 boolean으로 실려 간다.
        "threads": thread_state(),
    }
    if applied_pipeline:
        # 실제로 돈 단계마다 렌더된 줄 하나, 돈 순서로. ``internal_validation``처럼 비워서 쓰지
        # 않고 뺀다: 없음은 시도가 플래그 경로를 썼다는 뜻이고, 그것은 모든 단계가 버려진 명세와는
        # 다른 것이다 — 후자는 안전망만 담은 목록으로 도착한다.
        #
        # 명세를 되돌려 보내지 않고 문자열인 것은 둘의 일이 다르기 때문이다. 명세는 요청이고 이미
        # ``train_config.json``에 디스크에 있다. 이것은 기록이고, 독자가 여기서 필요한 것은 *무엇이
        # 어느 순서로 어느 열 위에서 돌았는지*이며 그것은 각각 한 줄이다. 덕분에 이 필드는 스칼라의
        # 평평한 목록이 되어 ``privacy._public_params``가 새 필터 없이 그것을 실어 간다.
        payload["applied_pipeline"] = list(applied_pipeline)
    if paired:
        # 이 행 위의 이 모델에 관한 것이 아니라 *비교*에 관한 스칼라 블록 — 그래서 ``metrics``에
        # 접어 넣지 않는다. "skipped"라고 말할 때도 남긴다. 그래야 독자가 "비교하지 않았고 이유는
        # 이것이다"와 "비교했고 아무것도 찾지 못했다"를 구분할 수 있다.
        payload[PAIRED_KEY] = paired
    if internal_validation:
        # ``applied_hyperparams``와 달리 비워서 쓰지 않고 뺀다: 없음은 "추정기가 행을 떼어 두지
        # 않았다"를 뜻하고, 그것이 흔한 경우라 필드가 필요하지 않다. 빈 블록은 "떼어 뒀는데 양은
        # 모름"으로 읽힐 것이다.
        payload["internal_validation"] = internal_validation
    if model_path:
        # 로컬 전용. ``privacy.PUBLIC_RESULT_FIELDS``에 없으니 프롬프트가 렌더되는 state 채널에
        # 절대 들어가지 않는다 — 적합된 모델은 데이터와 동등하다.
        payload["model_path"] = model_path
    if schema_path:
        # ``model_path``와 같은 조건이고 이유도 같다: 거기 나열된 수준은 셀 값이다. result.json을
        # 읽는 사람이 "적합됐고 새 행에 적용할 수 있음"과 "적합됐고 적합된 그 파일에서만 채점
        # 가능함"을 구분할 수 있도록 기록한다.
        payload["schema_path"] = schema_path
    payload, nonfinite = drop_nonfinite(payload)
    if nonfinite:
        payload[NONFINITE_KEY] = nonfinite
        # 로그 꼬리에도 넣는다. 사람이 먼저 읽는 필드이고 실패한 반복에서 Critic이 보는 필드가
        # 그것이기 때문이다. 무엇을 대체하지 않고 덧붙인다: 예외 줄이 있다면 그쪽이 여전히 더 중요한
        # 절반이다.
        payload["log_tail"] = (
            str(payload.get("log_tail") or "")
            + f"\nnon-finite values dropped from result.json: {', '.join(nonfinite)}"
        ).strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # ``allow_nan=False``는 위의 순회가 제 일을 했다는 단정이다. 장치가 아니라 가드다: 여기에
        # 닿았다는 것은 순회가 수로 알아보지 못하는 타입으로 유한하지 않은 수가 도착했다는 뜻이고,
        # 그것은 나쁜 점수가 아니라 여기의 버그다.
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        # 수가 적게 든 유효한 파일이 읽을 수 없는 파일을 이긴다: orchestrator는 없는 result.json을
        # 크래시한 반복으로 다루고, 그러면 이것을 이 함수가 실패한 것이 아니라 학습 실행이 한 것으로
        # 보고하게 된다.
        text = json.dumps(
            {
                "metrics": {},
                "split": split,
                "applied_hyperparams": {},
                "dropped_hyperparams": list(dropped_hyperparams or []),
                "applied_preprocessing": {},
                "train_time_sec": round(float(train_time_sec), 3),
                "status": status,
                "error_type": error_type,
                "log_tail": f"{log_tail}\nresult.json could not be serialised strictly: {exc}",
                "threads": thread_state(),
                NONFINITE_KEY: nonfinite or ["<unknown>"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    out_path.write_text(text, encoding="utf-8")


def apply_simulation(cfg: dict[str, Any], log: LogBuffer) -> None:
    """결정적인 실패 주입. 테스트와 OOM/타임아웃 훈련이 쓴다."""
    sim = dict(cfg.get("simulate") or {})
    sleep_sec = float(sim.get("sleep_sec", 0) or 0)
    if sleep_sec > 0:
        log.write(f"simulate.sleep_sec={sleep_sec}: sleeping to trigger the orchestrator timeout")
        time.sleep(sleep_sec)
    failure = str(sim.get("fail") or "").lower()
    if failure == "crash":
        # result.json을 쓰지 않고 하드 종료: segfault / CUDA 하드 실패를 흉내낸다. orchestrator가
        # 죽는 subprocess를 넘기고 산다는 것을 증명한다.
        log.write("simulate.fail=crash: exiting hard without a result file")
        sys.stderr.write("simulated hard crash in training subprocess\n")
        sys.stderr.flush()
        os._exit(3)
    if failure == "oom":
        raise MemoryError("simulated out of memory during training")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed AutoML training script.")
    parser.add_argument("--config", required=True, help="path to the training config JSON")
    parser.add_argument("--out", required=True, help="path to write result.json")
    parser.add_argument(
        "--score-model",
        default=None,
        help=(
            "score the held-back test split with this saved model instead of fitting. "
            "Run once, after the loop, by nodes/holdout.py"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    log = LogBuffer()
    started = time.perf_counter()

    try:
        cfg: dict[str, Any] = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type="config_error",
            log_tail=f"failed to read config {args.config}: {exc}",
        )
        return 0

    if args.score_model:
        return _score_only(cfg, Path(args.score_model), out_path, log, started)

    try:
        apply_simulation(cfg, log)
        run = run_training(
            cfg,
            log,
            model_out=out_path.parent / MODEL_FILENAME,
            predictions_out=out_path.parent / PREDICTIONS_FILENAME,
            schema_out=out_path.parent / SCHEMA_FILENAME,
            decision_out=out_path.parent / DECISION_FILENAME,
        )
    except BaseException as exc:  # noqa: BLE001 - 모든 실패는 결과가 되어야 한다
        error_type = classify_exception(exc)
        log.write(f"training failed ({error_type}): {type(exc).__name__}: {exc}")
        log.write(traceback.format_exc(limit=6))
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type=error_type,
            log_tail=log.tail(),
        )
        return 0

    write_result(
        out_path,
        status="ok",
        metrics=run.metrics,
        train_time_sec=time.perf_counter() - started,
        error_type=None,
        log_tail=log.tail(),
        applied_hyperparams=run.applied,
        dropped_hyperparams=run.dropped,
        applied_preprocessing=run.preprocessing,
        applied_pipeline=run.applied_pipeline,
        model_path=run.model_path,
        schema_path=run.schema_path,
        paired=run.paired,
        internal_validation=run.internal_validation,
    )
    return 0


def _score_only(
    cfg: dict[str, Any], model_path: Path, out_path: Path, log: LogBuffer, started: float
) -> int:
    """``--score-model`` 경로: test 행 위로 한 번, 적합 없음, 시뮬레이션 없음.

    ``apply_simulation``은 일부러 건너뛴다. 카드의 실패 주입 훈련은 그것이 선언된 학습 시도에 속한다.
    여기서 다시 돌리면 이미 성공적으로 학습된 모델의 마지막 측정을 실패시킬 것이다.
    """
    try:
        metrics, preprocessing = score_saved_model(cfg, log, model_path)
    except BaseException as exc:  # noqa: BLE001 - 학습 경로와 같은 계약
        error_type = classify_exception(exc)
        log.write(f"test scoring failed ({error_type}): {type(exc).__name__}: {exc}")
        log.write(traceback.format_exc(limit=6))
        write_result(
            out_path,
            status="error",
            metrics={},
            train_time_sec=time.perf_counter() - started,
            error_type=error_type,
            log_tail=log.tail(),
            split="test",
        )
        return 0

    write_result(
        out_path,
        status="ok",
        metrics=metrics,
        train_time_sec=time.perf_counter() - started,
        error_type=None,
        log_tail=log.tail(),
        applied_preprocessing=preprocessing,
        split="test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
