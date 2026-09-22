"""고정 프로파일링 스크립트: 원본 데이터를 받아 dataset card를 낸다. subprocess로 돈다.

이 파일은 LLM이 생성하지 *않고*, ``scripts/train.py`` 말고 데이터 파일을 여는 유일한 곳이다.
둘 다 자식 프로세스에서 도니, 프롬프트를 렌더하는 orchestrator 프로세스는 데이터 행을 들지 않는다.

내보내기 정책
-------------
카드는 추론 노드가 보게 되는 것이므로, 아래의 모든 필드는 한 열 전체에 대한 집계이고 레코드로
되돌릴 수 없다.

내보낸다  행·열 수, 결측률, 클래스 비율, 고유값 수 bucket, 왜도·크기·상관 bucket, 이상치 비율,
          열 이름, dtype, 기준선의 홀드아웃 점수와 그 부트스트랩 구간, 기준선 랭킹의 KS,
          인코딩 보고, 분할 프로토콜, 의심되는 결측 코드, 운영자의 주의사항
절대 안 함  셀 값, 최소·최대, 평균·표준편차·분위수, 클래스 라벨, 예시 행

이 정책의 예외처럼 보이는 것이 넷이고 넷 다 예외가 아니다:

* **의심되는 sentinel**은 *값*을 내보내지만 이 repo의 상수 목록이 이미 들고 있던 값뿐이므로,
  데이터가 기여한 것은 그 비율이다 (:mod:`automl_agent.dataset.sentinels`);
* **``--caveat``**은 자유 텍스트를 내보내지만, 여기 어느 코드 경로가 행에서 들어 올린 것이 아니라
  사람이 자기 데이터에 대해 직접 쓴 문장이다 (:mod:`automl_agent.dataset.caveats`);
* **기준선 점수, KS, ``baseline.ci``**는 홀드아웃 집계다. 학습 지표가 정책에 맞는 것과 같은
  이유로 맞고, 여기 있는 것은 목표 임계값이 셋 모두에서 유도되기 때문이다: 점수가 바를 세우고
  (:mod:`automl_agent.scoring.goal`), KS가 그 랭킹의 어떤 컷도 닿을 수 있는 한계를 고정하고
  (:mod:`automl_agent.scoring.ranking`), 구간이 그 바가 자기가 나온 수의 소음 안에 앉았는지를
  말한다 (:mod:`automl_agent.scoring.intervals`);
* **``data`` 블록**은 실행기가 파일을 다시 찾도록 경로를 들고 있고, private이다 —
  :func:`automl_agent.privacy.public_card`가 카드가 state에 들어가기 전에 떼어낸다.

크기와 왜도를 수가 아니라 bucket으로 내는 것은, 계획하는 쪽이 알아야 할 것이 어떤 열이 꼬리가
두껍고 수백 단위에 산다는 사실이지 3번 환자의 creatinine이 4.1이었다는 것이 아니기 때문이다 —
그리고 최소·최대 한 쌍은 실제 환자 둘의 값 *그 자체*다.

계약
----
입력 : ``--data <csv> --target <column> --out <card.json>``
출력 : ``--out``의 카드. stdout에 사람이 읽을 요약. 성공 0, 실패 1 (호출자가 stderr를 보여주고,
       그것은 로컬에 남아 프롬프트로 가지 않는다).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

# 노드가 이 파일을 패키지 모듈이 아니라 경로로 실행하니, repo 루트가 sys.path에 없고 상대
# import가 불가능하다. 이것을 더하는 덕분에 두 프로세스가 각자 목록을 들지 않고 하나의 지표
# registry를 공유한다. 아래 import들에 붙은 ``E402`` 무시도 모두 이 수정 때문이다.
if __package__ in (None, ""):  # pragma: no cover - 파일로 실행할 때만
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.dataset.caveats import (  # noqa: E402
    CAVEATS_KEY,
    card_caveats,
    grouping_caveats,
    merge_caveats,
    sentinel_caveats,
)
from automl_agent.dataset.features import (  # noqa: E402
    describe_encoding,
    encode_features,
    is_encodable_column,
    is_numeric_column,
)
from automl_agent.dataset.sentinels import (  # noqa: E402
    describe_sentinels,
    detect_sentinels,
)
from automl_agent.dataset.source import as_source, load_frame, source_kind  # noqa: E402
from automl_agent.dataset.targets import (  # noqa: E402
    DEFAULT_TARGET_MISSING_POLICY,
    TARGET_MISSING_POLICIES,
    detect_task,
    encode_target,
)
from automl_agent.scoring.intervals import (  # noqa: E402
    CI_LEVEL,
    DEFAULT_RESAMPLES,
    bootstrap_interval,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES,
    METRICS,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from automl_agent.scoring.ranking import (  # noqa: E402
    best_cut_ceiling,
    ks_statistic,
)
from automl_agent.scoring.splits import (  # noqa: E402
    describe_protocol,
    protocol,
    split_three_way,
)

# "각 지표를 어떻게 계산하는가"의 단 하나의 정의. 이 스크립트가 재는 기준선은 학습된 모든 시도가
# 비교되는 *바*이므로 둘은 똑같이 채점해야 하고, 여기 사본을 하나 더 두면 그 비교가 조용히 표류할
# 수 있다. train.py의 모듈 수준 import는 전부 순수 파이썬 패키지 모듈이라(sklearn은 분기 안에서
# 온다) 이 스크립트가 이미 갖지 않은 import 시점 의존성은 늘지 않는다.
from automl_agent.scripts.train import scorers  # noqa: E402
from automl_agent.threads import thread_state  # noqa: E402

# bucket 경계. 레코드 하나가 bucket을 옮기지 못할 만큼 거칠게 잡았다.
DISTINCT_EDGES: tuple[tuple[int, str], ...] = ((1, "constant"), (2, "binary"), (10, "low"), (100, "medium"))
SKEW_EDGES: tuple[tuple[float, str], ...] = ((0.5, "low"), (2.0, "moderate"))
MAGNITUDE_EDGES: tuple[tuple[float, str], ...] = (
    (1.0, "sub_unit"),
    (10.0, "unit"),
    (100.0, "tens"),
    (1000.0, "hundreds"),
)
CORRELATION_EDGES: tuple[tuple[float, str], ...] = ((0.05, "none"), (0.15, "weak"), (0.30, "moderate"))

DEFAULT_MEMORY_LIMIT_MB = 2048
DEFAULT_MAX_TRAIN_TIME_SEC = 600

# 기준선. 점수가 실행끼리도 데이터셋끼리도 같은 것을 뜻하도록 고정했다: "제대로 된 기본값이 여기서
# 이미 얻는 것". 태스크마다 하나이고, 회귀 쪽이 Ridge인 것은 분류 쪽이 LogisticRegression인 것과
# 같은 이유다 — 정규화된 선형 모델은 허수아비가 아닌 것 중 가장 싼 것이라, 이기면 뜻이 있고 못
# 이기면 그것이 발견이다.
BASELINE_MODELS: dict[str, str] = {
    TASK_CLASSIFICATION: "logreg (median impute + standard scale)",
    TASK_REGRESSION: "ridge (median impute + standard scale)",
}
BASELINE_ESTIMATORS: dict[str, str] = {
    TASK_CLASSIFICATION: "LogisticRegression",
    TASK_REGRESSION: "Ridge",
}
BASELINE_NOTE = (
    "{estimator} fitted on the training split and scored on the validation split "
    "of automl_agent.scoring.splits' protocol, seed {seed} — the same split scripts/train.py "
    "uses, so the numbers are directly comparable. The test slice is not read here."
)
# 프로파일링은 루프의 시간 예산이 적용되기 전에 도니, 기준선을 파일 크기에 맡기지 않고 막아 둔다.
BASELINE_MAX_ROWS = 50_000


def _bucket(value: float, edges: tuple[tuple[float, str], ...], last: str) -> str:
    for edge, label in edges:
        if value <= edge:
            return label
    return last


# --------------------------------------------------------------------------- #
# 열 프로파일
# --------------------------------------------------------------------------- #


def profile_column(series: Any, target: Any) -> dict[str, Any]:
    """한 열의 집계 기술자. 그 열의 값은 하나도 돌려주지 않는다."""
    import numpy as np
    import pandas as pd

    n_rows = int(len(series))
    n_missing = int(series.isna().sum())
    present = series.dropna()
    distinct = int(present.nunique())
    numeric = bool(pd.api.types.is_numeric_dtype(series)) and not bool(
        pd.api.types.is_bool_dtype(series)
    )

    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "missing_rate": round(n_missing / n_rows, 4) if n_rows else 0.0,
        "distinct": _bucket(distinct, DISTINCT_EDGES, "high"),
        # 여기서 결정하지 않고 :mod:`automl_agent.dataset.features`에 묻는다: one-hot으로 펼
        # 만큼 좁은 범주 열은 이제 쓸 수 있고, 그 상한을 아는 것은 그 모듈뿐이다.
        "usable_by_executor": (
            is_numeric_column(series) or is_encodable_column(series, distinct)
        ),
    }

    if distinct <= 1:
        profile["kind"] = "constant"
    elif distinct == 2:
        profile["kind"] = "binary"
    elif not numeric:
        # 범주 *라벨*은 셀 값이므로, 이 함수를 떠나는 것은 개수뿐이다.
        profile["kind"] = "categorical"
    elif bool(pd.api.types.is_integer_dtype(series)) and distinct <= 20:
        profile["kind"] = "discrete"
    else:
        profile["kind"] = "continuous"

    # 아래 집계들보다 위에 둔 것은 일부러다: magnitude, skew, outlier_rate, target_corr 전부
    # 이 값들을 여전히 측정값으로 세면서 재고, 여기서는 아무것도 변환하지 않는다.
    # :mod:`automl_agent.dataset.sentinels` 참고.
    suspects = detect_sentinels(series)
    if suspects:
        profile["sentinel_suspects"] = suspects

    if not numeric or len(present) < 8:
        return profile

    values = present.to_numpy(dtype="float64", copy=False)
    magnitude = float(np.median(np.abs(values)))
    profile["magnitude"] = _bucket(magnitude, MAGNITUDE_EDGES, "thousands+")
    profile["skew"] = _bucket(abs(float(pd.Series(values).skew() or 0.0)), SKEW_EDGES, "high")

    q1, q3 = (float(x) for x in np.percentile(values, [25, 75]))
    spread = q3 - q1
    if spread > 0:
        outliers = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
        profile["outlier_rate"] = round(outliers / len(values), 4)

    if target is not None:
        mask = present.index
        try:
            correlation = float(pd.Series(values, index=mask).corr(target.loc[mask]))
        except (ValueError, TypeError):  # pragma: no cover - 퇴화한 열
            correlation = float("nan")
        if correlation == correlation:  # NaN이 아니다
            profile["target_corr"] = _bucket(abs(correlation), CORRELATION_EDGES, "strong")
    return profile


# --------------------------------------------------------------------------- #
# 기준 baseline
# --------------------------------------------------------------------------- #


def _mirror_aliases(scores: dict[str, float]) -> dict[str, float]:
    """점수를 alias 이름으로도 함께 낸다. 어느 철자로 읽어도 카드가 읽히도록."""
    for alias, target in ALIASES.items():
        if target in scores:
            scores[alias] = scores[target]
    return scores


def _score_all(
    y_true: Any, pred: Any, proba: Any, average: str, task: str = TASK_CLASSIFICATION
) -> dict[str, float]:
    """이 예측이 지원할 수 있는 registry 지표를 전부 채점한다.

    손으로 관리하는 목록이 아니라 registry가 몬다: 그러면 기준선이 *이 태스크에서* ``--metric``이
    받는 지표를 정확히 덮고, 그 성질이 어느 지표에서든 ``auto`` 바를 유도할 수 있게 만든다.
    """
    thunks = scorers(y_true, pred, proba, average, task)
    scores: dict[str, float] = {}
    for name, spec in METRICS.items():
        if spec.task != task:
            continue
        scorer = thunks.get(name)
        if scorer is None:  # pragma: no cover - 여기 채점기가 없는 registry 항목
            continue
        if spec.binary_only and average != "binary":
            continue
        if spec.needs_proba and proba is None:
            continue
        # 클래스가 하나뿐인 홀드아웃에는 AUC가 없다. 임계값 계열 지표는 그래도 값이 있다.
        with contextlib.suppress(ValueError):
            scores[name] = round(float(scorer()), 4)
    return _mirror_aliases(scores)


def _baseline_intervals(
    y_true: Any,
    pred: Any,
    proba: Any,
    average: str,
    scores: dict[str, float],
    *,
    task: str = TASK_CLASSIFICATION,
    groups: Any = None,
    seed: int = 42,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any] | None:
    """기준선 점수마다의 95% 부트스트랩 구간. 잴 수 있는 것이 없으면 ``None``.

    하나가 아니라 *전부*다: 카드는 ``--metric``이 고르기 전에 쓰이므로, 여기서 지표 하나를 고르면
    절반은 틀린 것을 고른다. 비용은 검증 조각에 대해 최대 지표 일곱 개 × ``resamples``이고, 이미
    모델 하나를 적합하는 프로파일링 실행에서는 몇 초다.

    구간이 값을 하는 자리가 여기다. 기준선이 곧 *바*이기 때문이다: ``auto`` 임계값이 이 수들에서
    유도되고(:mod:`automl_agent.scoring.goal`), 자기가 유도된 점수의 구간 안에 앉은 바는 이
    데이터셋 검증 조각의 소음 바닥 안에 있는 바다. ``goal.describe``가 루프가 시작하기 전에 그것을
    말하니, 나중에 알아내는 것보다 iteration 예산 하나가 싸다.
    """
    bounds: dict[str, Any] = {}
    unit, drawn = "row", 0
    for name in METRICS:
        if name not in scores:
            continue

        def one(y_slice: Any, pred_slice: Any, proba_slice: Any, metric: str = name) -> float:
            return float(scorers(y_slice, pred_slice, proba_slice, average, task)[metric]())

        interval = bootstrap_interval(
            one, y_true, pred, proba, groups=groups, seed=seed, resamples=resamples
        )
        if interval is None:
            continue
        # ``scores``·``chance``와 같이 소수 넷째 자리까지. 카드의 수들은 나란히 읽히고, 여섯째
        # 자리까지 찍은 경계는 재표집 400회가 갖지 않은 정밀도를 주장한다.
        bounds[name] = {"low": round(interval.low, 4), "high": round(interval.high, 4)}
        unit, drawn = interval.unit, interval.resamples
    if not bounds:
        return None
    # ``scores``·``chance``와 같이 alias로도 함께 낸다. 어느 철자로 읽은 카드든 같은 구간을
    # 찾도록.
    for alias, target in ALIASES.items():
        if target in bounds:
            bounds[alias] = bounds[target]
    return {"level": CI_LEVEL, "unit": unit, "resamples": drawn, "scores": bounds}


def reference_scores(
    features: Any,
    codes: Any,
    n_classes: int,
    *,
    task: str = TASK_CLASSIFICATION,
    seed: int = 42,
    groups: Any = None,
    group_column: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any] | None:
    """고정된 기준선을 적합하고 그 홀드아웃 점수를 돌려준다.

    같은 분할에서 피처를 하나도 쓰지 않는 예측기 ``chance``도 함께 돌려준다. 기준선 점수는 그것
    옆에서만 해석되기 때문이다. 양성이 11%인 코호트에서 ``accuracy 0.89``는 우연이고, 그것에서
    유도한 임계값은 거짓말이다. 회귀 쪽 짝은 *평균* 예측기이고 둘 중 더 중요하다: ``r2``가 그것에
    대해 정의되고(그래서 우연은 구성상 0.0), ``mae``는 평균을 예측하는 값이 얼마인지 알기 전까지
    아무 뜻이 없다.

    예외를 내지 않고 ``None``을 돌려준다: 여기서 산출물은 카드이고, 기준선이 다룰 수 없는
    데이터셋이야말로 에이전트가 그래도 시도해 봐야 하는 종류다.
    """
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    regression = task == TASK_REGRESSION
    if features.shape[1] == 0 or (not regression and n_classes < 2):
        return None

    x_arr = np.asarray(features.to_numpy(), dtype="float64")
    y_arr = np.asarray(codes, dtype="float64" if regression else None)
    group_arr = np.asarray(groups) if groups is not None else None
    if len(y_arr) > BASELINE_MAX_ROWS:
        x_arr, y_arr = x_arr[:BASELINE_MAX_ROWS], y_arr[:BASELINE_MAX_ROWS]
        if group_arr is not None:
            group_arr = group_arr[:BASELINE_MAX_ROWS]

    try:
        # train.py가 쓰는 것과 같은 3분할 프로토콜, 같은 함수에서. test 조각을 여기서 *무시하는*
        # 것은 일부러다: 기준선은 모든 시도와 마찬가지로 검증 집합 점수이고, test 행은 루프가
        # 끝날 때까지 읽지 않는다.
        splits = split_three_way(x_arr, y_arr, seed, groups=group_arr, stratify=not regression)
        x_train, y_train = splits.x_train, splits.y_train
        x_val, y_val = splits.x_val, splits.y_val
        if regression:
            from sklearn.dummy import DummyRegressor
            from sklearn.linear_model import Ridge

            estimator: Any = Ridge(random_state=seed)
            dummy: Any = DummyRegressor(strategy="mean")
        else:
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(max_iter=1000, random_state=seed)
            dummy = DummyClassifier(strategy="prior")
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        model.fit(x_train, y_train)
        dummy.fit(x_train, y_train)

        average = "binary" if not regression and n_classes == 2 else "macro"
        proba = (
            model.predict_proba(x_val)[:, 1] if not regression and n_classes == 2 else None
        )
        pred = model.predict(x_val)
        scores = _score_all(y_val, pred, proba, average, task)
        chance = _score_all(y_val, dummy.predict(x_val), None, average, task)
        # 사전확률만 쓰는 예측기에는 정의상 랭킹 정보가 전혀 없다 — 그래서 랭킹 지표 둘은 잴
        # 대상이 아니라 상수다. AUC는 0.5이고, 상수 ranker의 average precision은 양성 비율이다.
        # pr_auc 바를 0.5가 아니라 클래스 균형에 대고 읽어야 하는 이유가 이것이다.
        if not regression and n_classes == 2:
            chance["roc_auc"] = 0.5
            chance["pr_auc"] = round(float(np.mean(np.asarray(y_val) == 1)), 4)
            _mirror_aliases(chance)
    except (ValueError, MemoryError, ImportError) as exc:
        print(f"reference baseline skipped: {type(exc).__name__}: {exc}", flush=True)
        return None

    reference: dict[str, Any] = {
        "model": BASELINE_MODELS[task],
        "note": BASELINE_NOTE.format(estimator=BASELINE_ESTIMATORS[task], seed=seed),
        "n_rows_used": int(len(y_arr)),
        # 이 수들을 낸 행이 어느 것인지. 실행의 프로토콜이 이것과 어긋날 때 카드를 *거절*할 수
        # 있도록 낸다. 여기서 유도한 바를 다른 행에서 잰 점수와 조용히 비교하는 대신이다 —
        # automl_agent.scoring.splits.
        "protocol": protocol(seed, group_column, stratified=not regression),
        # 위의 이웃이 어느 행이 이 수들을 냈는지 말하고, 이쪽은 어느 환경이 냈는지 말한다. 둘이
        # 같은 이유로 여기 있다: ``auto`` 모드에서 이 점수가 곧 루프가 판정받는 바이고, 한 thread
        # 상태에서 잰 바를 다른 상태에서 적합된 시도에 대고 쓰면 balanced_accuracy로 최대 0.0077
        # 어긋난다(:mod:`automl_agent.threads`) — iteration 1이 이미 넘었는지를 가를 만한 크기다.
        # 값은 planning 프롬프트 네 줄이고, 그것이 ``protocol``이 치르는 값이기도 하다.
        "threads": thread_state(),
        "scores": scores,
        "chance": chance,
    }
    # 랭킹 자신의 한계. 점수가 정책에 맞는 것과 같은 이유로 맞다 — 홀드아웃 전체에 대한 통계량이다.
    # 목표 유도에게 어떤 바가 이 랭킹이 *어떤* 컷으로도 닿을 수 없는 위에 있다고 말해 주는 것이
    # 이것이다 — automl_agent.scoring.ranking 참고. 확률이 없으면 잴 랭킹도 없으니, 회귀 기준선에는
    # 두 필드가 모두 없다.
    ks = ks_statistic(y_val, proba)
    ceiling = best_cut_ceiling(ks)
    if ks is not None and ceiling is not None:
        reference["ks"] = round(ks, 4)
        reference["balanced_accuracy_at_best_cut"] = ceiling
    # 위의 각 점수 중 얼마가 모델이 아니라 검증 조각인지. 분할이 그룹 단위였으면 이것도 그룹
    # 단위다. 한 환자의 다섯 방문은 다섯 관측이 아니니까 — automl_agent.scoring.intervals.
    intervals = _baseline_intervals(
        y_val,
        pred,
        proba,
        average,
        scores,
        task=task,
        groups=splits.groups_val,
        seed=seed,
        resamples=resamples,
    )
    if intervals is not None:
        reference["ci"] = intervals
    return reference


# --------------------------------------------------------------------------- #
# 카드 조립
# --------------------------------------------------------------------------- #


def _card_task(task: str, n_classes: int | None) -> str:
    """카드 자신의 태스크 라벨. registry의 둘보다 잘게 나뉜다.

    ``binary_classification``을 읽은 계획 프롬프트는 양성 클래스가 하나라는 것을 안다.
    ``classification``만으로는 그 말이 되지 않는다. :data:`automl_agent.scoring.metrics.CARD_TASKS`가
    이것들을 registry의 짝으로 되돌리니, 늘어난 해상도가 모호함을 만들지는 않는다.
    """
    if task == TASK_REGRESSION:
        return "regression"
    return "binary_classification" if n_classes == 2 else "multiclass_classification"


def target_profile(series: Any) -> dict[str, Any]:
    """연속 타깃의 모양. 피처 열이 받는 것과 같은 bucket으로.

    피처와 같은 함수를 쓰니 타깃도 같은 어휘, 같은 프라이버시 규칙으로 기술된다: bucket과 비율,
    값은 절대. 이 블록의 값을 하는 필드는 ``magnitude``다 — ``mae`` 바 3.5는 그 열이 단위 단위로
    가는지 천 단위로 가는지 읽는 사람이 알기 전까지 아무 뜻이 없고, 카드에서 그것을 말하는 것은
    이것뿐이다.

    ``usable_by_executor``와 ``name``은 뺀다: 타깃은 피처가 아니고, 그 이름은 이미 카드의
    ``target_column``이다.
    """
    profile = profile_column(series, None)
    return {
        key: value for key, value in profile.items() if key not in ("name", "usable_by_executor")
    }


def build_card(
    data_path: Path | str,
    target_column: str,
    *,
    name: str | None = None,
    memory_limit_mb: float = DEFAULT_MEMORY_LIMIT_MB,
    max_train_time_sec: float = DEFAULT_MAX_TRAIN_TIME_SEC,
    baseline: bool = True,
    seed: int = 42,
    on_missing_target: str = DEFAULT_TARGET_MISSING_POLICY,
    caveats: tuple[str, ...] | list[str] = (),
    group_column: str | None = None,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    table: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """데이터를 읽어 카드를 돌려준다. 여기서 행을 보는 유일한 함수다."""
    frame = load_frame(data_path, table=table, query=query)
    if target_column not in frame.columns:
        # 사용자가 고른 파일의 열 이름은 원본 데이터가 아니고, 없는 것을 이름 짓는 것이 이
        # 오류의 진단 가치 전부다.
        raise ValueError(
            f"target_column {target_column!r} not found; available columns: "
            f"{sorted(str(c) for c in frame.columns)}"
        )
    if group_column:
        if group_column not in frame.columns:
            raise ValueError(
                f"group_column {group_column!r} not found; available columns: "
                f"{sorted(str(c) for c in frame.columns)}"
            )
        if group_column == target_column:
            raise ValueError(f"group_column {group_column!r} is the target column")

    target_raw = frame[target_column]
    # 분할에서만이 아니라 피처에서도 뺀다: 추정기가 읽을 수 있는 id는 라벨로 가는 지름길이고,
    # ``subject_id`` 열의 프로필은 그것이 예측변수인 것처럼 발행된다. train.py도 같은 열을 뺀다.
    group_raw = frame[group_column] if group_column else None
    features = frame.drop(columns=[target_column] + ([group_column] if group_column else []))
    # 고르지 않고 열에서 읽는다: 연속 타깃을 라벨로 범주 코딩하면 행마다 "클래스" 하나인 분류기를
    # 학습시킨다 — automl_agent.dataset.targets.detect_task 참고.
    task = detect_task(target_raw)
    regression = task == TASK_REGRESSION
    codes, keep, n_missing_target = encode_target(target_raw, on_missing_target, task)
    if n_missing_target:
        # 여기서 걸러서 아래의 모든 수 — 행 수, 결측률, 상관, 기준선 — 가 학습기가 쓸 것과 같은
        # 행을 기술하게 한다.
        features = features[keep]
        if group_raw is not None:
            group_raw = group_raw[keep]
        print(
            f"dropped {n_missing_target} rows with a missing target "
            f"({target_column!r}); {len(codes)} rows remain",
            flush=True,
        )
    # 클래스 *라벨*은 남는다. 빈도만 발행하고, 어느 라벨이 어느 것인지에 대한 정보를 순서가
    # 나르지 않도록 정렬한다. 연속 타깃에는 발행할 빈도가 없고, 그 모양이 아래 ``target`` 블록으로
    # 간다.
    balance = (
        []
        if regression
        else sorted((codes.value_counts(normalize=True)).round(4).tolist(), reverse=True)
    )
    n_classes = None if regression else int(codes.nunique())

    # 각 열 프로필의 ``target_corr``이 무엇에 대고 재는지. 연속 타깃이 이 필드가 애초에 노린 경우다.
    # 0/1로 코딩된 두 클래스는 point-biserial 상관이고 같은 수다. 코드가 셋 이상이면 순서를 매길 수
    # 없어 상관 낼 것이 없고, 필드도 없다.
    numeric_target = codes.astype("float64") if regression or n_classes == 2 else None
    profiles = [profile_column(features[column], numeric_target) for column in features.columns]
    usable = [item for item in profiles if item["usable_by_executor"]]

    # 기준선을 적합하기 전에 찍는다. 읽는 사람이 경고를 그것이 적용되는 점수 뒤가 아니라 위에서
    # 만나도록.
    suspects = {
        item["name"]: item["sentinel_suspects"]
        for item in profiles
        if item.get("sentinel_suspects")
    }
    warning = describe_sentinels(suspects)
    if warning:
        print(warning, flush=True)

    # train.py가 하는 것과 같은 인코딩, 같은 함수에서. 그래야 기준선이 에이전트의 모델들이 받을
    # 바로 그 피처 행렬에서 재어진다.
    executor_features, encoding = encode_features(features)
    print(describe_encoding(encoding), flush=True)
    # train.py가 하는 것과 같은 정규화. 그래야 "한 그룹"이 두 프로세스에서 같은 것을 뜻하고 둘이
    # 똑같이 분할한다.
    group_values = (
        group_raw.astype("string").fillna("<missing>").to_numpy() if group_raw is not None else None
    )
    if group_values is not None:
        print(
            f"grouped by {group_column}: {len(set(group_values.tolist()))} distinct groups"
            f" over {len(group_values)} rows — no group is split across train/val/test",
            flush=True,
        )
    reference = (
        reference_scores(
            executor_features,
            codes,
            n_classes or 0,
            task=task,
            seed=seed,
            groups=group_values,
            group_column=group_column,
            resamples=bootstrap_resamples,
        )
        if baseline
        else None
    )

    card: dict[str, Any] = {
        "name": name or f"{target_column}-prediction",
        # 출처의 *종류*만 적는다. 이 필드는 공개 카드에 남아 모든 추론 프롬프트에 실리므로
        # 접속 URL이 여기로 나갈 수 없다 (automl_agent.dataset.source::source_kind).
        "description": (
            f"{source_kind(data_path)} 데이터에서 자동 생성된 카드입니다. "
            "원본 행은 이 카드에 포함되지 않습니다 — 모든 수치는 열 전체에 대한 집계입니다."
        ),
        "task": _card_task(task, n_classes),
        "target_column": target_column,
        # 카드가 실제로 기술하는 행: ``drop`` 정책에서는 라벨 없는 행이 사라졌고, 실행기도 같은
        # 행을 뺀다.
        "n_rows": int(len(codes)),
        "n_features": len(usable),
        # one-hot 인코딩이 있기 전에 쓰인 카드도 그대로 읽히도록 원래 이름을 유지한다. 세는
        # 대상은 그대로다 — 실행기가 쓸 수 없는 열 — 다만 좁은 범주 열은 이제 그중 하나가 아니다.
        "n_features_dropped_non_numeric": len(profiles) - len(usable),
        # 추정기가 실제로 받는 것. 이제 열 수와 같지 않다: 수치 열 9개 더하기 수준 12개인 month는
        # 9 + 12다. 메모리 가드가 값 매기는 수이고, "당신 데이터로 채점"과 "당신 데이터의 수치
        # 절반으로 채점"을 읽는 사람이 구분하려면 필요한 수라서 밝힌다.
        "encoding": encoding,
        # 연속 타깃에서 없애지 않고 ``None``으로 둔다. 읽는 사람도 목표 유도도 "이 카드는 회귀에
        # 대한 것"과 "이 카드는 오래된 것"을 구분할 수 있도록.
        "n_classes": n_classes,
        "class_balance": balance or None,
        "imbalance_ratio": round(balance[0] / balance[-1], 2) if balance and balance[-1] else None,
        "missing": {
            "overall_rate": round(float(features.isna().to_numpy().mean()), 4),
            "columns_with_missing": sum(1 for item in profiles if item["missing_rate"] > 0),
            "worst_rate": max((item["missing_rate"] for item in profiles), default=0.0),
        },
        "features": profiles,
        # 학습기가 다시 듣지 않고 같은 정책을 적용하도록, 그리고 읽는 사람이 "8000행"과
        # "8012행 중 8000행"을 구분할 수 있도록 카드에 실어 나른다.
        "target_missing": {"policy": on_missing_target, "n_dropped": int(n_missing_target)},
        "preprocessing": {"impute": "median", "scale": True},
        "constraints": {
            "memory_limit_mb": memory_limit_mb,
            "max_train_time_sec": max_train_time_sec,
        },
        "profile": {
            "generated_by": "automl_agent.scripts.profile",
            "policy": "aggregates only — no cell values, no min/max, no class labels",
        },
        # private: 카드가 state에 들어가기 전에 privacy.public_card가 떼어낸다.
        # ``group_column``이 ``preprocessing``이 아니라 여기 사는 것은 ``path``와 같은 이유다:
        # 데이터 로딩 결정이고, LLM이 쓰는 계획이 어느 행을 떼어 두는지를 바꿀 수 있어서는
        # 안 된다. automl_agent.scoring.splits 참고.
        "data": {"path": str(data_path), "target_column": target_column},
    }
    # DB 출처는 경로만으로 어느 행인지 정해지지 않는다. 카드가 기술하는 행을 ``--resume``과
    # ``run`` 이 다시 읽을 수 있어야 하므로 질의도 같은 비공개 블록에 남긴다.
    if table:
        card["data"]["table"] = str(table)
    if query:
        card["data"]["query"] = str(query)
    if regression:
        # 분류 카드에서 ``class_balance``가 하는 일. 타깃 자기 단위의 수가 무엇을 뜻하는지
        # 말하는 유일한 블록이다. 회귀 경로에만 있는 것은, 클래스로 코딩된 타깃의 모양이 곧 그
        # 균형이고 그것은 이미 위에 발행됐기 때문이다.
        card["target"] = target_profile(codes)
    if group_column:
        card["data"]["group_column"] = group_column
    # 위의 집계가 말할 수 없는 것을, 말할 수 있는 두 출처에서: 이 스크립트 자신의 검사와 운영자의
    # ``--caveat``. 기계의 발견을 먼저 둔다 — 사람의 메모는 보통 그것에 대한 논평이기 때문이다.
    # 아무것도 없으면 비우지 않고 없앤다 — automl_agent.dataset.caveats 참고.
    notes = merge_caveats(
        sentinel_caveats(suspects),
        grouping_caveats(
            group_column, len(set(group_values.tolist())) if group_values is not None else None
        ),
        list(caveats),
    )
    if notes:
        card[CAVEATS_KEY] = notes
    if reference is not None:
        # 목표 임계값이 여기서 유도되고(automl_agent.scoring.goal), 계획하는 쪽에 첫 시도가 맨
        # 선형 모델을 실제로 이겼는지 말해 주는 것도 이것이다.
        card["baseline"] = reference
    return card


def summarise(card: dict[str, Any]) -> str:
    """콘솔용 요약. 카드와 마찬가지로 집계뿐이다."""
    missing = card.get("missing") or {}
    worst = sorted(
        (item for item in card.get("features") or [] if item.get("missing_rate")),
        key=lambda item: -float(item["missing_rate"]),
    )[:5]
    target = card.get("target") or {}
    if card.get("task") == "regression":
        # 찍을 클래스 수가 없다. 두 블록 아래의 mae·rmse를 읽으려면 읽는 사람에게 필요한 것이
        # 크기와 모양이다.
        target_line = (
            f"  target={card['target_column']}  kind={target.get('kind')}"
            f"  magnitude={target.get('magnitude')}  skew={target.get('skew')}"
            f"  outlier_rate={target.get('outlier_rate')}"
        )
    else:
        target_line = (
            f"  target={card['target_column']}  n_classes={card['n_classes']}"
            f"  balance={card['class_balance']}  imbalance_ratio={card['imbalance_ratio']}"
        )
    lines = [
        f"dataset card: {card['name']} ({card['task']})",
        f"  rows={card['n_rows']}  features={card['n_features']}"
        f"  dropped_non_numeric={card['n_features_dropped_non_numeric']}",
        target_line,
        f"  missing: overall={missing.get('overall_rate')}"
        f" columns={missing.get('columns_with_missing')} worst={missing.get('worst_rate')}",
    ]
    declared = (card.get("baseline") or {}).get("protocol")
    if isinstance(declared, dict):
        # 카드에 있는 것과 같은 이유로 기준선 위에 찍는다: 두 줄 아래의 점수는 검증 숫자이고,
        # 파일의 한 조각은 일부러 거기 들어 있지 않다.
        lines.append("  " + describe_protocol(declared))
    encoding = card.get("encoding")
    if isinstance(encoding, dict):
        # 아래 기준선을 이것에 대고 읽어야 한다: 그 점수는 한 줄 위의 열 목록이 아니라 이 행렬에서
        # 재어진다.
        lines.append("  " + describe_encoding(encoding))
    dropped = int((card.get("target_missing") or {}).get("n_dropped") or 0)
    if dropped:
        lines.append(f"  target 결측 {dropped}행 제외 (--on-missing-target drop) — 위 수치는 남은 행 기준")
    if worst:
        lines.append(
            "  결측 상위: "
            + ", ".join(f"{item['name']}={item['missing_rate']}" for item in worst)
        )
    suspects = [item for item in card.get("features") or [] if item.get("sentinel_suspects")]
    if suspects:
        # build_card가 이미 긴 형태를 찍었어도 여기서 반복한다: 사람이 실제로 훑는 것은 이 요약
        # 이고, 그것이 옆에 앉은 줄 — `결측 상위` — 이 바로 sentinel이 거짓말로 만드는 줄이다.
        lines.append(
            f"  결측 코드 의심 {len(suspects)}열: "
            + ", ".join(
                f"{item['name']}={item['sentinel_suspects'][0]['value']}" for item in suspects[:5]
            )
            + " (변환 안 함 — 위 경고 참고)"
        )
    notes = card_caveats(card)
    if notes:
        # 운영자가 자기 ``--caveat``이 도착한 것을 보고, 이 실행의 모든 추론 프롬프트가 나를
        # 문장을 그대로 읽을 수 있도록 전문을 되울린다.
        lines.append(f"  데이터 주의사항 {len(notes)}건 — 모든 추론 프롬프트에 그대로 실립니다:")
        lines.extend(f"    - {note}" for note in notes)
    reference = card.get("baseline")
    if isinstance(reference, dict):
        scores = reference.get("scores") or {}
        chance = reference.get("chance") or {}
        # 정규 이름만. 카드는 점수를 alias로도 발행해서 ``--metric``의 어느 철자든 기준선을
        # 찾지만(:func:`_mirror_aliases`), 그것들까지 찍으면 회귀 점수 셋이 여섯이 되고 ``mae``와
        # ``mean_absolute_error``가 서로 다른 두 측정인 것처럼 읽힌다.
        published = {key: value for key, value in scores.items() if key not in ALIASES}
        lines.append(f"  기준선({reference.get('model')}, {reference.get('n_rows_used')}행):")
        lines.append(
            "    "
            + "  ".join(
                f"{key}={value}(chance {chance.get(key, '-')})" for key, value in published.items()
            )
        )
        ci = reference.get("ci")
        if isinstance(ci, dict) and isinstance(ci.get("scores"), dict):
            # 점만이 아니라 폭도. f1=0.61을 보고 두 줄 아래에서 임계값 0.71을 본 사람은 폭이
            # 없으면 그 격차가 목표인지 반올림 오차인지 알 길이 없다.
            unit = "그룹" if ci.get("unit") == "group" else "행"
            lines.append(
                f"    {int(float(ci.get('level') or CI_LEVEL) * 100)}% CI "
                f"({unit} 단위 부트스트랩 {ci.get('resamples')}회): "
                + "  ".join(
                    f"{key}={bound.get('low')}~{bound.get('high')}"
                    for key, bound in ci["scores"].items()
                    if key in published
                )
            )
        ceiling = reference.get("balanced_accuracy_at_best_cut")
        if ceiling is not None:
            lines.append(
                f"    랭킹 KS={reference.get('ks')} → 어떤 컷으로도 "
                f"balanced_accuracy는 {ceiling}이 상한입니다 (이 기준선의 랭킹 기준). "
                "더 높은 바를 원하면 랭킹 자체를 올려야 합니다 — 특성이나 모델 family"
            )
    else:
        lines.append(
            "  기준선: 없음 — 목표 임계값은 지표별 기본값을 씁니다 "
            "(mae·rmse처럼 정답 열의 단위로 나오는 지표는 기본값이 없어 --threshold가 필요합니다)"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dataset card from raw data.")
    parser.add_argument(
        "--data",
        required=True,
        help="path to the CSV, path to a sqlite file (.db/.sqlite/.sqlite3), or a "
        "SQLAlchemy connection URL. A database source needs --table or --query",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="database source only: read every row of this table (shorthand for "
        '--query \'SELECT * FROM "<name>"\')',
    )
    parser.add_argument(
        "--query",
        default=None,
        help="database source only: the SELECT whose rows are the dataset",
    )
    parser.add_argument("--target", required=True, help="name of the target column")
    parser.add_argument("--out", required=True, help="where to write the dataset card JSON")
    parser.add_argument(
        "--name",
        default=None,
        help="dataset name for the card (defaults to '<target>-prediction'; the file "
        "name is deliberately not used, since it would put the data source in the prompt)",
    )
    parser.add_argument("--memory-limit-mb", type=float, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument("--max-train-time-sec", type=float, default=DEFAULT_MAX_TRAIN_TIME_SEC)
    parser.add_argument("--seed", type=int, default=42, help="seed for the baseline split")
    parser.add_argument(
        "--on-missing-target",
        choices=list(TARGET_MISSING_POLICIES),
        default=DEFAULT_TARGET_MISSING_POLICY,
        help="what to do about rows whose target is missing: 'reject' (default) fails and "
        "reports the count, 'drop' removes them and records how many",
    )
    parser.add_argument(
        "--caveat",
        action="append",
        default=[],
        metavar="TEXT",
        dest="caveats",
        help="a fact about this data the card's aggregates cannot show, carried into every "
        "reasoning prompt. Repeatable. This is the only channel for what a human learned by "
        "looking at the raw file — e.g. that a flag with no missing values changes meaning "
        "along row order, so a model leaning on it learns the charting regime, not acuity.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        metavar="COLUMN",
        help="a column whose rows must not be divided across train/val/test — a patient id "
        "when the table has one row per visit, say. Without it a random split puts the same "
        "patient on both sides and every score in the run is inflated by an amount nothing "
        "in the run can detect. The column is dropped from the features, and the choice is "
        "recorded in the card's private data block so the LLM cannot alter it.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the reference baseline fit. The card is then thresholdless, so the "
        "goal falls back to the per-metric default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        card = build_card(
            as_source(args.data),
            args.target,
            name=args.name,
            memory_limit_mb=args.memory_limit_mb,
            max_train_time_sec=args.max_train_time_sec,
            baseline=not args.no_baseline,
            seed=args.seed,
            on_missing_target=args.on_missing_target,
            caveats=list(args.caveats or []),
            group_column=args.group_column,
            table=args.table,
            query=args.query,
        )
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        sys.stderr.write(f"profiling failed: {type(exc).__name__}: {exc}\n")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summarise(card), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
