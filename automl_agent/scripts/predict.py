"""고정 예측 스크립트: 적합된 모델과 새 행을 받아 라벨이 붙은 행을 낸다.

데이터 파일을 여는 세 번째이자 마지막 파일이고, 이 실행이 학습한 적 없는 파일을 여는 유일한
파일이다. 다른 둘처럼 subprocess로 돌아서 orchestrator 프로세스는 여전히 데이터 행을 들지 않고,
다른 둘과 달리 *행 단위* 출력을 내므로 여기서 쓰는 것은 어느 state 채널에도 가지 않는다.

**``model.joblib`` 하나만으로는 쓸 수 있는 모델이 아니다.** 인코더는 수준 집합, 열 순서, 결측 열
결정을 *눈앞의 파일*에서 읽으므로 두 번째 파일은 다르게 인코딩된다. 폭이 다르면 sklearn이 예외를
낸다. **폭이 같으면 — 같은 표를 매달 내보내는 흔한 경우다 — 아무것도 나지 않고 다른 것을 뜻하는
열에서 모든 예측이 계산된다.** 출력에서도 그 위의 모든 지표에서도 보이지 않게.

그래서 인코딩은 적합 시점에 저장되고 여기서
:func:`~automl_agent.dataset.features.encode_with_schema`가 재생한다. 재생할 수 없는 것은 전부
거절하고, 재생한 것은 경고를 달아 전부 알린다:

* 적합이 필요로 하는데 이 파일에 없는 원본 열 — 거절
  (:class:`automl_agent.dataset.features.FeatureSchemaMismatch`);
* 이 빌드가 모르는 버전의 스키마 — 거절;
* 조립된 폭에 추정기가 동의하지 않는 경우 — 인코더와 스키마는 이미 일치하지만 거절한다. 함께
  저장되도록 만들어진 두 artifact가 그 짝이라는 것은 디렉터리가 증명할 수 없는 단 하나다;
* 새 범주, 학습에는 없던 결측, 텍스트가 된 열, 파일이 더한 열 — 발견마다 경고로 내보낸다. 각각
  정의된 인코딩이 있고, 아닌 척하는 것이 이 파일이 없애려고 있는 그 침묵이다.

``--label-column``은 참 라벨 컬럼을 이름 지어 실행을 backtest로 바꾼다: 플래그가 없을 때와 똑같이
예측하고, 그다음 채점한다. 그 점수에 대해 일부러 그렇게 한 것이 셋이다 — 이 실행의 프로토콜이
아니라고 말하는 것, 적합 시점에 기록된 지표(``schema["metric"]``)로 재는 것, 클래스 목록을 이
파일에서 다시 유도하지 않는 것.

계약
----
입력 : ``--model <model.joblib> --schema <feature_schema.json> --data <csv> --out <csv>``,
       배치를 채점하려면 ``--label-column <name>``
출력 : ``--out``의 예측 CSV — 입력 순서로 입력 행마다 한 줄. 요청하면 ``--report``에 JSON 요약.
       stdout에 한글 요약. 성공 0, 실패 1 (stderr는 로컬에 남고 프롬프트로 가지 않는다).

이 스크립트의 모든 출력은 행 단위 데이터다. CSV는 실행의 목적이고 호출자가 자기 데이터를 두는
곳에 속한다. report는 범주 이름을 들고 있어서 ``.gitignore``가 막는 ``artifacts/``가 기본값이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 다른 두 스크립트와 같은 이유: 경로로 실행되니 repo 루트가 sys.path에 없고 상대 import가
# 불가능하다. 이것을 더하는 덕분에 이 파일이 적합이 쓴 인코더의 사본이 아니라 *같은* 인코더를
# 재생한다. 아래 import들에 붙은 ``E402`` 무시도 모두 이 수정 때문이다.
if __package__ in (None, ""):  # pragma: no cover - only when run as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from automl_agent.config import DECISION_FILENAME  # noqa: E402
from automl_agent.dataset.features import (  # noqa: E402
    FeatureSchemaMismatch,
    describe_drift,
    encode_with_schema,
)
from automl_agent.scoring import calibration  # noqa: E402
from automl_agent.scoring.intervals import DEFAULT_RESAMPLES  # noqa: E402
from automl_agent.scoring.metrics import (  # noqa: E402
    ALIASES as METRIC_ALIASES,
)
from automl_agent.scoring.metrics import (  # noqa: E402
    TASK_REGRESSION,
    canonical,
)

# 인코더와 같은 이유로 베끼지 않고 import한다: "이 분할을 채점한다"의 두 번째 구현은 두 번째
# 답이고, 배치 점수의 존재 이유는 ``result.json``의 홀드아웃과 비교 가능하다는 것이다. train.py의
# import는 전부 순수 파이썬 패키지 모듈이라(sklearn과 pandas는 함수 안에서 import한다) 이
# 스크립트가 이미 갖지 않은 import 시점 의존성은 늘지 않는다.
from automl_agent.scripts.train import (  # noqa: E402
    LogBuffer,
    evaluate_split,
    label_at_cut,
    load_decision,
)

# 예측 라벨이 들어갈 컬럼과 클래스 확률마다 붙는 접두사. 호출자의 하류 코드가 이 이름들로 키를
# 잡으니 여기서 이름 짓는다.
PREDICTION_COLUMN = "prediction"
PROBA_PREFIX = "proba_"

# 배치 점수가 보고되는 이름. 이것과 실행의 홀드아웃을 함께 든 report에서 읽는 사람도 나중의
# 스크립트도 둘을 섞지 못하게 한다.
BATCH_SCORE_KEY = "batch_metrics"


def load_schema(path: Path) -> dict[str, Any]:
    """모델 옆에 저장된 인코딩을 읽는다.

    파일이 없으면 배치를 다시 유도하는 폴백이 아니라, 이유를 적은 거절이다. 다시 유도하는 것이
    바로 조용히 어긋난 행렬을 만드는 연산이고, 스키마가 정당하게 없는 실행 둘(합성 데이터 실행,
    스키마를 쓰기 전에 만들어진 실행)은 어차피 모델을 새 행에 적용할 수 없는 실행이다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"feature schema not found at {path}. A model saved without one cannot be applied "
            "to new rows: nothing records which column of its input was which. Re-run training "
            "on a real data file to write one — a --dry-run or synthetic run never fits an "
            "encoding, so it has no schema to save."
        )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"feature schema {path} is not a JSON object")
    return loaded


def prepare_features(
    frame: Any, schema: dict[str, Any], label_column: str | None = None
) -> tuple[Any, list[str]]:
    """애초에 피처가 아니었던 열을 빼고, 무엇을 뺐는지 말한다.

    타깃과 그룹 열은 추측이 아니라 *스키마의 이름으로* 뺀다: 새 파일은 참 라벨(backtest)이나 환자
    id를 들고 있을 수 있고 둘 다 적합 시점에 행렬에서 제외됐다. 그러지 않으면 학습 파일에 없던
    열로 보고된다 — 맞는 말이지만 쓸모가 없다.

    ``label_column``도 같은 이유로 같은 목록에 넣는다. 보통은 그것이 같은 이름의 스키마 타깃 열이라
    이미 처리되지만, 그것을 ``outcome_actual``이라 부르는 backtest 내보내기는 예상 못 한 여분의
    열로 인코더에 닿는다 — 채점 대상인 것이 drift로 보고된다.
    """
    dropped: list[str] = []
    for key in ("target_column", "group_column"):
        name = schema.get(key)
        if name and str(name) in frame.columns:
            dropped.append(str(name))
    if label_column and label_column in frame.columns and label_column not in dropped:
        dropped.append(str(label_column))
    return (frame.drop(columns=dropped) if dropped else frame), dropped


def check_width(model: Any, n_columns: int) -> None:
    """추정기가 적합되지 않은 행렬을, 그것이 조용히 받아들이기 전에 거절한다.

    인코더는 스키마의 열을 냈다고 이미 단언했으므로, 이것이 발화하는 것은 스키마와 모델이 함께
    저장된 짝이 아닐 때뿐이다 — 한 iteration 디렉터리의 파일을 다른 쪽 옆에 복사한 경우. 폭 자체는
    sklearn도 잡지만 메시지가 배열에 대한 것이고, 이쪽은 원인을 이름 짓는다.

    ``n_features_in_``은 적합되지 않은 추정기와 그것을 기록하지 않는 몇몇에 없다. 없음은
    "불일치"가 아니라 "검사할 수 없음"이다 — 거기서 거절하면 일어난 적 없는 경우를 막으려고
    작동하는 짝을 버린다.
    """
    expected = getattr(model, "n_features_in_", None)
    if expected is None or int(expected) == int(n_columns):
        return
    raise FeatureSchemaMismatch(
        f"the model expects {int(expected)} encoded feature(s) but this schema produces "
        f"{int(n_columns)}. The model and the schema are not the pair that was saved together "
        "— check that both came from the same iteration directory."
    )


def label_predictions(raw: Any, schema: dict[str, Any]) -> list[Any]:
    """클래스 코드를 파일이 쓴 라벨로 되돌린다. 회귀 값은 그대로 지나간다.

    ``encode_target``이 타깃 열을 범주 코드로 바꿨으므로, 호출자가 ``died``를 물은 자리에서 적합된
    분류기는 ``1``을 예측한다. 스키마의 ``classes`` 목록은 그 코드와 인덱스가 맞으니 이것은 추론이
    아니라 조회다. 목록 밖의 코드는 코드로 남긴다: 스키마와 모델이 라벨 집합에 대해 어긋났다는
    뜻이고, 이름을 지어내는 것은 숫자를 보여주는 것보다 나쁘다.
    """
    if schema.get("task") == TASK_REGRESSION:
        return [float(value) for value in raw.tolist()]
    classes = list(schema.get("classes") or [])
    if not classes:
        return list(raw.tolist())
    return [
        classes[int(code)] if 0 <= int(code) < len(classes) else int(code)
        for code in raw.tolist()
    ]


def probability_matrix(model: Any, matrix: Any, schema: dict[str, Any]) -> Any:
    """``predict_proba``를 한 번, 아니면 ``None``. 한 번만 계산되도록 여기서 계산한다.

    출력 컬럼과 배치 점수 둘 다 이 수를 필요로 하고, ``predict_proba``를 두 번 부르면 비결정적인
    추정기가 한 report 안에 서로 다른 두 확률 집합을 넣을 수 있다 — CSV는 0.70이라 말하고 Brier는
    0.68로 값 매겨진다.
    """
    if schema.get("task") == TASK_REGRESSION or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(matrix)
    except (AttributeError, ValueError, NotImplementedError):
        return None


def probability_columns(proba: Any, model: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """클래스별 확률. 코드가 아니라 라벨로 키를 잡는다.

    회귀 타깃에서도, ``predict_proba``가 없는 분류기에서도 비어 있다 — 0/1 확신도를 지어내지 않고
    컬럼을 없애는 쪽이다. 지어낸 값은 딱딱한 라벨을 확률이라 보고하는 것이니까.

    스키마의 목록이 아니라 ``model.classes_``로 키를 잡는다. 학습 분할에 행이 없는 클래스는 출력에
    컬럼이 없고, 그러면 그 뒤의 모든 클래스에서 두 목록이 한 칸씩 어긋난다.
    """
    if proba is None:
        return {}
    classes = list(schema.get("classes") or [])
    codes = list(getattr(model, "classes_", range(proba.shape[1])))
    columns: dict[str, Any] = {}
    for position, code in enumerate(codes):
        index = int(code)
        label = classes[index] if 0 <= index < len(classes) else index
        columns[f"{PROBA_PREFIX}{label}"] = proba[:, position]
    return columns


def positive_class_proba(proba: Any, model: Any) -> Any:
    """클래스 코드 ``1``에 속하는 ``proba``의 컬럼, 아니면 ``None``.

    ``train.py``의 ``_proba``는 ``[:, 1]``을 집는다 — 그쪽 ``classes_``는 정렬된 코드 ``[0, 1]``이라
    컬럼 1이 곧 코드 1이다. 여기서는 가정하지 않고 위치를 찾는다: 적합 시점에 두 클래스가 모두
    있었다면 같은 컬럼이고, 하나가 없었다면 ``[:, 1]``은 다른 클래스의 확률을 양성인 것처럼 채점한
    값이다. 틀린 컬럼에서 계산한 이진 지표는 이 파일 전체의 실패 양식이므로, 세 줄 아끼자고 할 일이
    아니다.
    """
    if proba is None or getattr(proba, "ndim", 0) != 2 or proba.shape[1] < 2:
        return None
    codes = [int(code) for code in getattr(model, "classes_", range(proba.shape[1]))]
    if 1 not in codes:
        return None
    return proba[:, codes.index(1)]


def encode_batch_labels(series: Any, schema: dict[str, Any]) -> tuple[Any, Any, dict[str, int]]:
    """참 라벨을 모델이 예측하는 코드로. 그리고 어느 행이 쓸 수 있는지의 마스크.

    ``(codes, keep, counts)``를 돌려주고 ``keep``은 입력 행에 대한 불리언 마스크다.

    ``schema["classes"]``의 위치로 코드를 매긴다 — ``label_predictions``가 반대 방향으로 읽는 그
    목록이다 — 절대 ``encode_target``을 거치지 않는다. 이 파일에서 범주를 다시 유도하는 것이 라벨을
    조용히 다시 번호 붙이는 일이다: 한 클래스가 빠진 배치는 남은 라벨을 ``0..n-2``로 매기므로 모델의
    ``1``이 적합이 뜻한 것과 다른 클래스에 대해 채점되고, 그 위의 모든 지표는 아무것도 아닌 것에
    대한 숫자다.

    스키마에 코드가 없는 행은 무엇으로 매핑하지 않고 빼서 센다: 빈 라벨은 ``missing``, 적합이 본 적
    없는 값은 ``unknown``. backtest 파일의 새 클래스는 진짜 소식이고, 그것의 정직한 형태는 "이
    행들은 채점할 수 없었다"이지 음성인 것처럼 계산한 점수가 아니다.
    """
    import numpy as np
    import pandas as pd

    task = schema.get("task")
    if task == TASK_REGRESSION:
        values = pd.to_numeric(series, errors="coerce")
        keep = values.notna().to_numpy()
        counts = {"missing": int((~keep).sum()), "unknown": 0}
        return values.to_numpy(dtype="float64")[keep], keep, counts

    classes = list(schema.get("classes") or [])
    if not classes:
        raise ValueError(
            "this schema records no class list, so the batch's labels cannot be coded the way "
            "the model's outputs were. Scoring would compare two different numberings."
        )
    # 양쪽을 텍스트로 본다. CSV를 왕복하면 정수 라벨 1이 문자열 "1"이 되고, JSON 스키마는 적합의
    # 열이 들고 있던 것을 그대로 저장한다. 렌더된 값으로 맞추는 것이 1과 "1"을 하나는 알려진 것
    # 하나는 모르는 것이 아니라 같은 클래스로 만든다.
    lookup = {str(label): index for index, label in enumerate(classes)}
    raw = series.astype("string")
    mapped = raw.map(lookup)
    blank = raw.isna().to_numpy()
    keep = mapped.notna().to_numpy()
    counts = {
        "missing": int(blank.sum()),
        "unknown": int((~keep & ~blank).sum()),
    }
    return np.asarray(mapped[keep].to_numpy(), dtype="int64"), keep, counts


def score_batch(
    model: Any,
    matrix: Any,
    labels: Any,
    proba: Any,
    pred: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """이 배치의 라벨 붙은 행을, 이 실행이 목표로 삼았던 지표로 채점한다.

    전부 :func:`automl_agent.scripts.train.evaluate_split`로 잰다 — ``result.json``의 홀드아웃
    숫자를 낸 그 함수다. 다른 것은 행이고, 그리고 이 행들이 어떻게 이루어졌는지 알 수 없다는 것 —
    숨기지 않고 요약에 적는다.
    """
    codes, keep, counts = encode_batch_labels(labels, schema)
    scored = int(len(codes))
    result: dict[str, Any] = {
        "label_column": None,  # 이름을 아는 호출자가 채운다
        "scored_rows": scored,
        "excluded_rows": counts,
        "metric": None,
        BATCH_SCORE_KEY: {},
        "reliability": [],
        "notes": [],
    }
    if not scored:
        result["notes"].append(
            "채점할 수 있는 행이 없습니다 — 라벨이 모두 비었거나 학습 때 없던 값입니다"
        )
        return result

    task = str(schema.get("task") or "classification")
    n_classes = 0 if task == TASK_REGRESSION else len(list(schema.get("classes") or []))
    average = "binary" if n_classes == 2 else "macro"
    metric = schema.get("metric")
    metric = canonical(str(metric)) if metric else None
    # 찍지 않고 모은다: 건너뛴 이유는 위에 흩어지는 게 아니라 그것이 설명하는 점수 옆의 요약
    # 블록에 속한다.
    log = LogBuffer(echo=False)

    kept_pred = pred[keep]
    kept_proba = None if proba is None else proba[keep]
    metrics = evaluate_split(
        model,
        matrix[keep],
        codes,
        n_classes,
        average,
        log,
        task=task,
        interval_metric=metric,
        # groups 없음: 이 파일의 군집 구조는 여기서 알 수 없다. 스키마의 그룹 열이 있을 수도
        # 있지만, *이* 행들이 대상마다 여러 번 나온 것 중 하나인지는 배치가 어떻게 이루어졌는지에
        # 대한 사실이고 그것이 이 스크립트가 모르는 것이다. 군집된 행에 대한 row 단위 재표집
        # 구간은 실제보다 좁으므로, 아닌 그룹 구간으로 꾸미지 않고 그 경고를 달아 보고한다.
        groups=None,
        seed=int(schema.get("seed") or 42),
        resamples=DEFAULT_RESAMPLES,
        pred=kept_pred,
        proba=kept_proba,
    )
    result["metric"] = metric
    result[BATCH_SCORE_KEY] = metrics
    result["resamples"] = DEFAULT_RESAMPLES
    # 구간 줄은 뺀다: ``describe_score``가 같은 경계를 지표 옆에 찍고, 같은 수를 두 번 보여주면
    # "roc_auc skipped: only one class present" 같은 줄도 든 이 블록을 건너뛰도록 읽는 사람을
    # 길들인다.
    result["notes"] = [line for line in log.lines if not line.startswith(f"{metric}=")]
    if kept_proba is not None and scored >= calibration.MIN_CALIBRATION_ROWS:
        # ``calibration_error``와 같은 하한으로 막는다. 이유도 같다: 그 아래에서는 한 bin이 행
        # 몇 개를 들고 있어서 "1행, 예측 0.79, 실제 0.000" 표가 재지 못한 곳을 틀린 모델로 읽히게
        # 한다. 요약 숫자는 참으면서 그것을 참은 이유인 bin은 찍는 것은 양쪽을 다 갖는 짓이다.
        result["reliability"] = calibration.reliability(codes, kept_proba)
    return result


def run_prediction(
    model_path: Path,
    schema_path: Path,
    data_path: Path,
    out_path: Path,
    id_column: str | None = None,
    label_column: str | None = None,
) -> dict[str, Any]:
    """``data_path``의 모든 행을 예측해 CSV로 쓴다. 요약을 돌려준다.

    ``label_column``은 스키마가 라벨을 코드로 매길 수 있는 행에 대한 점수를 더한다. 예측은 하나도
    바뀌지 않는다: 어느 쪽이든 배치는 똑같이 인코딩·검사·예측되고, 라벨은 그 뒤에야 읽는다.
    """
    import joblib
    import pandas as pd

    schema = load_schema(schema_path)
    model = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    if id_column and id_column not in frame.columns:
        raise ValueError(f"id_column {id_column!r} not found in {data_path}")
    if label_column and label_column not in frame.columns:
        raise ValueError(
            f"label_column {label_column!r} not found in {data_path}. Without it there is "
            "nothing to score against; drop the flag to predict without scoring."
        )

    features, dropped = prepare_features(frame, schema, label_column)
    encoded, drift = encode_with_schema(features, schema)
    check_width(model, int(encoded.shape[1]))

    import numpy as np

    matrix = np.asarray(encoded.to_numpy(), dtype="float64")
    raw = model.predict(matrix)
    proba = probability_matrix(model, matrix, schema)
    positive = positive_class_proba(proba, model)

    # 이 CSV의 라벨을 만드는 규칙. 모델 옆에서 읽고 여기서 고르지 않는다: 이 시도가 점수를 얻은
    # 컷은 그 시도의 학습 행에서 골라졌고, 다른 컷을 적용하는 호출자는 이 실행이 보고한 숫자가
    # 설명하지 않는 라벨을 받는다. 파일이 없는 것이 흔한 경우이고 sklearn의 고정 0.5 규칙을 뜻한다.
    decision_log = LogBuffer(echo=False)
    threshold = load_decision(model_path.parent / DECISION_FILENAME, decision_log)
    if threshold is not None and positive is None:
        decision_log.write(
            "저장된 결정 규칙이 있지만 이 모델에서는 양성 클래스 확률을 얻을 수 없어 기본 규칙으로 "
            "라벨을 만들었습니다 — 이 실행이 기록한 점수와 다른 규칙입니다"
        )
        threshold = None
    if threshold is not None:
        raw = label_at_cut(positive, threshold)

    predictions = label_predictions(raw, schema)
    # id 컬럼을 먼저 둬서 위치를 믿지 않고도 출력을 되붙일 수 있게 하고, 그다음 예측, 그다음 확률.
    # 입력의 나머지 컬럼은 일부러 베끼지 않는다: 베끼면 이 파일이 호출자 데이터의 두 번째 사본이
    # 되고, 첫 번째는 이미 그쪽에 있다.
    out: dict[str, Any] = {}
    if id_column:
        out[id_column] = frame[id_column]
    out[PREDICTION_COLUMN] = predictions
    out.update(probability_columns(proba, model, schema))
    result = pd.DataFrame(out, index=frame.index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "model": str(model_path),
        "schema": str(schema_path),
        "data": str(data_path),
        "out": str(out_path),
        "task": schema.get("task"),
        "n_features": int(encoded.shape[1]),
        "columns": [str(name) for name in result.columns],
        # 이름을 적어 둬서 backtest가 "라벨 컬럼을 제외했다"와 "라벨 컬럼을 피처로 모델에
        # 먹였다"를 구분할 수 있게 한다.
        "dropped_non_features": dropped,
        "drift": drift,
        # ``load_decision``은 적용할 규칙이 아예 없을 때 아무 말도 하지 않으므로, 말한 것이
        # 있다면 존재하지만 쓸 수 없었던 파일이라는 뜻이다 — 출력의 모든 라벨이 달라지는 일이고,
        # 읽으라고 지시받은 블록에 속한다. *적용된* 경우에는 요약 줄이 컷 자체를 찍는다.
        "warnings": describe_drift(drift) + (decision_log.lines if threshold is None else []),
    }
    if threshold is not None:
        summary["threshold"] = threshold
    if label_column:
        scored = score_batch(
            model,
            matrix,
            frame[label_column],
            positive,
            raw,
            schema,
        )
        scored["label_column"] = str(label_column)
        summary["score"] = scored
    return summary


def describe_score(scored: dict[str, Any]) -> list[str]:
    """배치 점수를 한글 줄로. 그것이 아닌 것과 함께.

    여기서 경고는 각주가 아니라 둘째 줄이다. 배치 점수의 뜻은 전부 배치가 어떻게 이루어졌는지에서
    오고 이 스크립트는 그것을 볼 수 없다 — 홀드아웃을 채점한 그 함수가 이 수를 냈지만, 행의 출처는
    harness가 아니라 호출자의 지식이다. 그 말 없이 찍힌 수는 비교가 유효한 것처럼 홀드아웃과 비교될
    수다.
    """
    metrics = dict(scored.get(BATCH_SCORE_KEY) or {})
    rows = int(scored.get("scored_rows") or 0)
    label = scored.get("label_column")
    # 컬럼 이름 뒤에 조사가 오지 않게 쓴다: 맞는 조사는 이 스크립트가 고르지 않는 이름의 마지막
    # 음절에 달려 있다.
    lines = [f"이 배치를 채점했습니다 — 라벨 컬럼 {label!r}, {rows}행"]

    excluded = dict(scored.get("excluded_rows") or {})
    missing, unknown = int(excluded.get("missing") or 0), int(excluded.get("unknown") or 0)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"라벨이 빈 행 {missing}개")
        if unknown:
            # 따로 말할 값이 있다: 모르는 라벨은 더러운 칸이 아니라 적합이 본 적 없는
            # 클래스이고, 모델에는 그것을 예측할 코드가 없다.
            parts.append(f"학습 때 없던 라벨 값을 가진 행 {unknown}개")
        lines.append(f"  채점에서 제외: {', '.join(parts)}")

    if not rows:
        lines += [f"  - {note}" for note in scored.get("notes") or []]
        return lines

    metric = scored.get("metric")
    if metric and metric in metrics:
        headline = f"  {metric}={float(metrics[metric]):.4f}"
        low, high = metrics.get(f"{metric}_ci_low"), metrics.get(f"{metric}_ci_high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            headline += (
                f" (95% 구간 {float(low):.4f}~{float(high):.4f}, "
                f"row 단위 재표집 {int(scored.get('resamples') or 0)}회)"
            )
        lines.append(headline + " ← 이 실행이 목표로 삼았던 지표")
    else:
        # 건너뛰지 않고 말한다. 이 줄이 없으면 아래 블록은 숫자 중 하나가 우연히 목표인 평범한
        # 점수표로 읽히고 — 스키마가 지표를 이름 짓지 못했으니 읽는 사람은 어느 것인지 알 길이
        # 없다. 신뢰구간을 든 것도 목표 지표뿐이라, 그것이 없으면 구간도 조용히 사라진다.
        #
        # 없음이 두 가지이고 아래 분기가 둘을 갈라 둔다. *이름이 적힌* 지표를 이 배치가 계산할
        # 수 없는 것은 평범하고 현재의 일이다: 모든 행이 같은 라벨인 곳에서 ``roc_auc``는
        # 정의되지 않고, ``score_split``은 NaN을 기록하지 않고 뺀다. *이름이 없는* 쪽은
        # ``train.goal_metric`` 전에 쓰인 스키마다 — 다른 태스크의 목표 지표가 대체되지 않고
        # ``null``로 기록됐다. 디스크의 파일이지 이 빌드가 만들 수 있는 길은 아니다.
        lines.append(
            "  이 배치에는 목표 지표가 없습니다"
            # 굴절시키지 않고 괄호에 넣는다: 지표 이름 뒤의 조사는 마지막 음절에 달려 있고
            # ("f1을"이지만 "rmse를"), 이름은 스키마에서 온다.
            + (f" — 스키마가 적은 지표({metric})는 이 행들에서 계산할 수 없었습니다" if metric else
               " — 스키마에 목표 지표가 적혀 있지 않습니다 (지표를 대체해 기록하기 전에 만들어진 "
               "스키마입니다). 이 모델을 다시 학습하면 기록됩니다")
            + ". 아래 지표는 모두 같은 채점에서 나온 값이지만, 어느 것이 이 실행이 목표로 "
            "삼았던 숫자인지는 여기서 알 수 없고 신뢰구간도 없습니다"
        )
    # 같은 채점 호출이 낸 나머지 전부. 여기서 소음이 될 넷은 뺀다: 목표 지표(위에 자기 줄이
    # 있다), 확률 진단값 둘(아래에 자기 줄이 있다), 구간 경계(그것이 감싸는 지표와 함께 이미
    # 찍혔다), 그리고 같은 수를 다른 이름으로 내는 registry의 alias들.
    hidden = {metric, calibration.BRIER_KEY, calibration.CALIBRATION_KEY, *METRIC_ALIASES}
    others = ", ".join(
        f"{name}={float(value):.4f}"
        for name, value in sorted(metrics.items())
        if name not in hidden
        and isinstance(value, (int, float))
        and not name.endswith(("_ci_low", "_ci_high"))
    )
    if others:
        lines.append(f"  그 밖의 지표: {others}")

    probability = calibration.describe(metrics, rows)
    if probability:
        lines.append(f"  {probability}")
    lines += [f"  {line}" for line in calibration.describe_table(scored.get("reliability") or [])]

    # 모듈 docstring이 논증하는 그 딱지. 점수가 찍힐 때마다 함께 찍는다.
    lines.append(
        "  이 점수는 이 실행의 채점 프로토콜이 아닙니다 — result.json의 홀드아웃은 학습 전에 "
        "떼어 둔 행을 한 번만 채점한 값이지만, 이 파일이 어떤 행으로 이루어졌는지는 여기서 알 "
        "수 없습니다. 낮게 나오는 것이 정상일 수도 있고, 위의 '확인할 점'이 그 이유일 수도 "
        "있습니다. 이 행들이 한 대상에서 여러 번 나온 것이라면 위 신뢰구간은 실제보다 좁습니다."
    )
    for note in scored.get("notes") or []:
        lines.append(f"  - {note}")
    return lines


def summarise(summary: dict[str, Any]) -> str:
    """실행을 사람이 읽을 한글 줄로. 읽을 값이 있는 부분은 확인할 점이다."""
    lines = [
        f"{summary['rows']}행을 예측해 {summary['out']}에 저장했습니다 "
        f"(인코딩된 피처 {summary['n_features']}개, task={summary['task']})"
    ]
    if summary.get("dropped_non_features"):
        names = ", ".join(summary["dropped_non_features"])
        lines.append(f"피처가 아닌 컬럼은 제외했습니다: {names}")
    if summary.get("threshold") is not None:
        # 놀라운 때만이 아니라 매번 찍는다: CSV의 라벨은 0.5에서의 ``predict_proba``가 말했을
        # 것이 아니고, 이 파일을 다른 모델의 출력과 비교하는 호출자는 비교 전에 그것을 알아야
        # 한다.
        lines.append(
            f"라벨은 이 모델과 함께 저장된 결정 규칙으로 만들었습니다 — 양성 확률 "
            f"{summary['threshold']} 이상 (sklearn 기본값 0.5가 아닙니다)"
        )
    if summary.get("score"):
        lines += describe_score(dict(summary["score"]))
    warnings = list(summary.get("warnings") or [])
    if warnings:
        # "경고"가 아니라 "확인할 점": 적합이 이미 무시한 열은 매번 보고되고 이상이 아니므로,
        # 이것들을 모두 경고라 부르면 못 본 범주 줄도 든 이 블록을 건너뛰도록 읽는 사람을
        # 길들인다.
        lines.append(f"확인할 점 {len(warnings)}건 — 예측은 나왔지만 아래를 읽으십시오:")
        lines += [f"  - {line}" for line in warnings]
    else:
        lines.append("학습 때의 인코딩과 어긋난 곳은 없었습니다")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed prediction script: apply a saved model to new rows."
    )
    parser.add_argument("--model", required=True, help="path to the saved model.joblib")
    parser.add_argument(
        "--schema",
        required=True,
        help="path to the feature_schema.json saved beside that model. Required rather than "
        "derived: re-deriving the encoding from the new file is what produces a misaligned "
        "matrix that scores without complaining",
    )
    parser.add_argument("--data", required=True, help="path to the CSV to predict on")
    parser.add_argument("--out", required=True, help="path to write the predictions CSV")
    parser.add_argument(
        "--report",
        default=None,
        help="path to write the run summary as JSON (rows, drift, warnings). Optional; the "
        "same summary is printed either way",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="a column copied through to the output so the predictions can be joined back to "
        "the input by key instead of by row position",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="a column of true labels in --data. Given one, the batch is also scored on the "
        "metric the run was steered by, using the same scoring code as the run's holdout. The "
        "score is not the run's protocol — how these rows were assembled is unknown here — and "
        "the output printed with it says so",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_prediction(
            Path(args.model),
            Path(args.schema),
            Path(args.data),
            Path(args.out),
            id_column=args.id_column,
            label_column=args.label_column,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        # FeatureSchemaMismatch는 ValueError이므로, 이 스크립트가 있는 이유인 거절들은
        # traceback이 아니라 메시지로 나온다.
        sys.stderr.write(f"prediction failed: {type(exc).__name__}: {exc}\n")
        return 1

    if args.report:
        report_path = Path(args.report)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except OSError as exc:
            # 이 시점에 예측은 이미 디스크에 있다. 막지 않았을 때는 잘못된 --report 경로가
            # 명령을 traceback으로 끝냈고, 운영자가 그것을 보고 내린 결론은 예측이 실패했다는
            # 것이었다 — 그래서 메시지는 *써진* 파일을 이름 짓고, 아래 요약은 어느 쪽이든 찍는다.
            sys.stderr.write(
                f"report not written: {type(exc).__name__}: {exc}\n"
                f"  the predictions themselves are in {args.out}\n"
            )
    print(summarise(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
