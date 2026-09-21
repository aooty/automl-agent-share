"""실행 설정.

노드는 전역값을 읽지 않는다: ``graph._bind``가 이 객체를 각 노드에 묶어 주므로
노드는 순수한 ``state -> dict`` 함수로 남는다.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset.targets import TARGET_MISSING_POLICIES

# goal.py는 패키지에서 아무것도 import하지 않으므로 이 방향은 순환이 아니다.
from .scoring.goal import DEFAULT_MARGIN, DEFAULT_MODE, GOAL_MODES
from .scoring.metrics import DEFAULT_METRICS, GOAL_METRICS, TASK_CLASSIFICATION, canonical, direction_of

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
# 그래프가 spawn하는 스크립트 둘만. ``scripts/predict.py``는 루프가 끝난 뒤 같은 프로세스에서
# 도는 CLI 명령이라 여기 경로 상수가 필요 없다.
TRAIN_SCRIPT = PACKAGE_DIR / "scripts" / "train.py"
PROFILE_SCRIPT = PACKAGE_DIR / "scripts" / "profile.py"
PROMPTS_DIR = PACKAGE_DIR / "llm" / "prompts"
# 파일 하나가 모든 thread_id를 담으므로 이름이 두 번 필요하다 — 여기 기본값과, 출력 위치를
# 옮긴 실행을 위한 ``RunConfig.checkpoint_db``.
CHECKPOINT_FILENAME = "checkpoints.sqlite"
CHECKPOINT_DB = ARTIFACTS_ROOT / CHECKPOINT_FILENAME

# 체크포인트 쓰기가 다른 프로세스의 잠금 해제를 얼마나 기다리다 포기하는지. ``show``나 두 번째
# ``run``이 잠금을 쥘 수 있다. sqlite 자체 기본값 5초는 이미 한 시간을 학습한 실행 옆에서는
# 짧다. graph.make_checkpointer 참고.
CHECKPOINT_TIMEOUT_SEC = 60.0

# 학습 한 번이 자기 디렉터리에 남기는 것. 셋 다 데이터와 동등하므로 셋 다 ``artifacts/`` 안에
# 머문다 — ``.gitignore``가 막고, 경로가 ``privacy.PUBLIC_RESULT_FIELDS``에 없고, 어떤 추론
# 노드도 내용을 읽지 않는다.
MODEL_FILENAME = "model.joblib"
PREDICTIONS_FILENAME = "val_predictions.npz"
SCHEMA_FILENAME = "feature_schema.json"

# 이 모델의 확률이 라벨이 되는 방식, 그것이 sklearn의 고정된 0.5 규칙이 아닐 때:
# ``{"threshold": 0.137, "chosen_on": "train", "metric": "balanced_accuracy"}``. 계획이 조정된
# 컷을 요청했을 때만 쓴다 — 파일이 *없다*는 것이 기본 규칙이라는 뜻이다. 위의 셋과 달리 데이터와
# 동등하지 않아서, 컷 자체는 ``metrics``의 ``applied_threshold``로 밖에 나간다.
DECISION_FILENAME = "decision_rule.json"

# 다시 적는 대신 registry에서 읽는다. 그러면 CLI 기본값과, 지표가 대체된 뒤 분류 실행이
# 되돌아가는 지표가 구조적으로 같은 이름이다.
DEFAULT_METRIC = DEFAULT_METRICS[TASK_CLASSIFICATION]
DEFAULT_THRESHOLD = 0.85
# ``direction``이 받는 값. 기본값 상수는 일부러 두지 않았다: 기본값은 지표가 선언한
# 방향(``scoring.metrics.direction_of``)이고, 하나 더 두면 진실이 둘이 된다. CLI 플래그는 없다 —
# 지표가 이미 정하므로 줄 수 있는 값이 하나뿐이었다. 남은 이유는 이것이 ``run_config.json``의 키이고,
# 손으로 고친 파일이 ``resume``에서 거절돼야 하기 때문이다.
DIRECTIONS = ("maximize", "minimize")
DEFAULT_MAX_ITERATIONS = 5
DRY_RUN_SCENARIOS = ("success", "fail", "oom", "stall", "slow", "crash")
# ``--dry-run``에 값을 주지 않았을 때. 상수인 이유는 argparse의 ``const``와 아래 필드 기본값이 같은
# 수여야 하기 때문이다.
DEFAULT_DRY_RUN_SCENARIO = DRY_RUN_SCENARIOS[0]
DEFAULT_TIME_BUDGET_SEC = 3600
STALL_LIMIT = 2  # 개선 없는 반복이 이만큼 연속되면 포기한다

# ``--time-budget-sec``에서 떼어 두는 몫. 실행이 보고하는 숫자가 자기 예산보다 오래 살아남게
# 한다. 10분의 1인 것은 ``holdout``이 이미 적합된 모델 하나를 채점하고 탐색은 하지 않기
# 때문이고, 상수가 아니라 비율인 것은 규모를 호출자가 정하기 때문이다.
HOLDOUT_RESERVE_FRACTION = 0.1
# 적합 한 번이 남은 시간에서 받는 몫의 하한. 그 몫이 subprocess timeout이 받을 수 있는 숫자로
# 남게 한다 — 1초 아래면 spawn이 대부분이다. 몫이 없어진 적합은 ``nodes/training.py``가
# 거절하므로 이것이 예산을 넘겨 쓸 수는 없다.
MIN_FIT_TIMEOUT_SEC = 1.0

# 프로파일링은 파일을 한 번 읽고 열 단위 집계를 낸다. 루프의 시간 예산이 적용되기 전에
# 돌기 때문에 학습과 따로 제한한다.
PROFILE_TIMEOUT_SEC = 900.0

# 실행이 끝날 때 ``model.joblib``을 무엇만 남길지. 긴 실행의 진짜 한계는 디스크다 — 루프 안에
# 적합된 estimator의 크기를 제한하는 것이 없다. 기본값은 ``predict``가
# 찾아가는 모델 하나를 남긴다. "all"은 ``predict --iteration <다른 번호>``가 필요로 하는
# 것이라, 정리 메시지가 그 플래그를 알려 준다.
KEEP_MODELS_MODES = ("best", "all")
DEFAULT_KEEP_MODELS = KEEP_MODELS_MODES[0]

DEFAULT_LLM_MODEL = "claude-opus-5"
DEFAULT_LLM_MAX_TOKENS = 8000
DEFAULT_LLM_TIMEOUT_SEC = 180.0
# SDK가 429/5xx에 exponential backoff로 수행하는 전송 계층 재시도. 넉넉하게 둔 이유: 일시적인
# 5xx 하나가 추론 노드 하나를 통째로 날린다 — 노드는 heuristic으로 되돌아가고, 실행은 조용히
# LLM의 판단을 잃는다.
DEFAULT_LLM_MAX_RETRIES = 4

API_KEY_ENV = "ANTHROPIC_API_KEY"
BEDROCK_FLAG_ENV = "AUTOML_USE_BEDROCK"
AWS_REGION_ENV = "AWS_REGION"


# ``__post_init__``이 값을 거절하는 두 가지 형태. ``label``은 필드 이름과 조사를 함께 담는다 —
# 메시지에서 달라지는 부분은 그것뿐이다.
def _one_of(label: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{label} {allowed} 중 하나여야 합니다 (받은 값: {value!r})")


def _at_least(label: str, value: float, floor: int) -> None:
    if value < floor:
        raise ValueError(f"{label} {floor} 이상이어야 합니다 (받은 값: {value!r})")


@dataclass(frozen=True)
class RunConfig:
    """실행 하나(``thread_id`` 하나)의 불변 설정."""

    thread_id: str
    metric: str = DEFAULT_METRIC
    # "auto": 카드의 기준 baseline에서 기준선을 끌어낸다. "fixed": ``threshold``를, 그것이
    # None이면 지표별 기본값을 쓴다. automl_agent.scoring.goal 참고.
    goal_mode: str = DEFAULT_MODE
    # "fixed" 모드에서만 읽는다. 값을 준다는 것이 그 모드를 뜻한다.
    threshold: float | None = None
    # "auto"에서만: baseline에 남은 여유의 어느 만큼을 요구할지.
    goal_margin: float = DEFAULT_MARGIN
    # 어느 쪽이 더 좋은지. ``None``은 "지표에서 읽어라"라는 뜻이고 그것이 유일하게 정직한
    # 출처다 — automl_agent.scoring.metrics.direction_of. ``__post_init__``에서 해소되므로
    # 아래쪽 독자는 모두 실제 값을 받는다.
    direction: str | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    time_budget_sec: int = DEFAULT_TIME_BUDGET_SEC
    stall_limit: int = STALL_LIMIT
    # 기준선을 넘은 것이 실행을 끝내는지 (automl_agent.graph.stop_condition). 기본은 꺼져
    # 있다: 켜면 실행이 반복 예산을 얼마나 쓰는지가 달라진다.
    search_past_goal: bool = False
    dry_run: bool = False
    # --dry-run에서 모사된 trainer가 어떤 경로를 따를지.
    dry_run_scenario: str = DEFAULT_DRY_RUN_SCENARIO
    # 실제 학습은 하고, LLM 호출 대신 규칙 기반 추론을 쓴다. 자격 증명 없이 실행 경로
    # 전체를 훑을 수 있다.
    no_llm: bool = False
    seed: int = 42
    llm_model: str = DEFAULT_LLM_MODEL
    # 계획하고 선택하는 모델, 그것이 위의 것과 다를 때. 빈 문자열은 "같은 것"을 뜻한다.
    # 움직일 수 있는 것은 제안하는 쪽 절반뿐이다: ``planning``과 ``model_selection`` 뒤에는
    # 코드 관문이 있어서(``validate_plan``, 모델 registry, 하이퍼파라미터 clamp) 약한 모델은
    # 조용히가 아니라 요란하게 실패한다.
    proposer_model: str = ""
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC
    dataset_card_path: Path | None = None
    # ``data_path``와 ``group_column``은 dataset card가 아니라 여기 둔다: 둘 다 어떤 행이
    # 존재하는지 또는 떼어 두는지를 정하므로, 카드 안에 있으면 카드와 함께 프롬프트로 실려 갈
    # 수 있고, LLM이 둘 중 어느 것도 바꾸자고 제안할 수 있어서는 안 된다.
    data_path: Path | None = None
    target_column: str | None = None
    # 라벨이 없는 행: "reject" 또는 "drop". None은 "카드를 따르고, 카드가 말이 없으면
    # reject"라는 뜻이다 — nodes/training.py::build_train_config 참고.
    on_missing_target: str | None = None
    # 원본 데이터에 대한 운영자 메모. 프로파일러로 전달되고 거기서 모든 추론 프롬프트로
    # 들어간다 (automl_agent.dataset.caveats). ``--data`` 경로에서만 쓴다: ``--dataset-card``로
    # 준 카드는 이미 자기 것을 갖고 있다.
    caveats: tuple[str, ...] = ()
    # 이 열의 어떤 그룹도 train/val/test에 걸쳐 나뉘어서는 안 된다
    # (automl_agent.scoring.splits). 여기 둔 이유는 ``data_path`` 위에 있다.
    group_column: str | None = None
    # 테스트(그리고 나란히 돌리는 실행)가 출력 위치를 전부 옮길 수 있도록 덮어쓸 수 있다.
    artifacts_root: Path | None = None
    # 실행이 끝난 뒤 어느 반복이 적합된 모델을 남길지 — KEEP_MODELS_MODES 참고.
    keep_models: str = DEFAULT_KEEP_MODELS

    def __post_init__(self) -> None:
        # 여기 있는 검사는 모두, 정상으로 *보이면서* 헛소리를 보고하는 실행을 만들던 값이다.
        _one_of("goal_mode는", self.goal_mode, GOAL_MODES)
        metric = canonical(self.metric)
        if metric not in GOAL_METRICS:
            raise ValueError(
                f"metric은 {GOAL_METRICS} 중 하나여야 합니다 (받은 값: {self.metric!r}). "
                "학습 스크립트가 만들지 않는 지표를 목표로 잡으면 그 실행은 무엇을 해도 "
                "목표를 달성할 수 없습니다"
            )
        if metric != self.metric:
            # 별칭으로 들어와 정규 이름으로 나간다. 아래쪽 독자는 모두 이름 하나만 비교한다.
            object.__setattr__(self, "metric", metric)
        implied = direction_of(metric)
        if self.direction is None:
            # 보통의 경로: ``rmse``가 어느 쪽인지 아무도 알 필요 없다.
            object.__setattr__(self, "direction", implied)
        else:
            _one_of("direction은", self.direction, DIRECTIONS)
            if self.direction != implied:
                # 지표가 이미 뜻하는 바를 확인하는 용도로만 받는다. 반대 조합은 실행 전체를
                # 뒤집는다.
                raise ValueError(
                    f"{metric}은 {implied} 지표라서 direction={self.direction!r}로 실행할 수 없습니다. "
                    "방향은 지표에서 나오므로 run_config.json의 direction을 지우거나 "
                    f"{implied}로 고치십시오"
                )
        _one_of("dry_run_scenario는", self.dry_run_scenario, DRY_RUN_SCENARIOS)
        _one_of("keep_models는", self.keep_models, KEEP_MODELS_MODES)
        if self.on_missing_target is not None:
            _one_of("on_missing_target은", self.on_missing_target, TARGET_MISSING_POLICIES)
        _at_least("max_iterations는", self.max_iterations, 1)
        _at_least("stall_limit은", self.stall_limit, 1)
        _at_least("seed는", self.seed, 0)
        if self.time_budget_sec <= 0:
            raise ValueError(f"time_budget_sec는 0보다 커야 합니다 (받은 값: {self.time_budget_sec!r})")
        if not 0.0 < self.goal_margin < 1.0:
            # 0은 baseline과 똑같이, 1은 만점을 요구한다. 둘 다 기준선을 엄격하게 만드는 게
            # 아니라 무의미하게 만든다.
            raise ValueError(f"goal_margin은 0과 1 사이여야 합니다 (받은 값: {self.goal_margin!r})")
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise ValueError(f"threshold는 유한한 숫자여야 합니다 (받은 값: {self.threshold!r})")
        if not isinstance(self.caveats, tuple):
            # ``resume``은 이 객체를 run_config.json에서 다시 만드는데, 거기서 tuple은 JSON
            # 배열로 직렬화되어 있다 — 이것이 없으면 재개된 실행의 caveats는 list가 되고
            # dataclass는 더 이상 hashable하지도 비교 가능하지도 않다.
            object.__setattr__(self, "caveats", tuple(str(item) for item in self.caveats or ()))

    # -- 파생 경로 ---------------------------------------------------------- #

    @property
    def use_llm(self) -> bool:
        return not (self.dry_run or self.no_llm)

    @property
    def artifacts_base(self) -> Path:
        return self.artifacts_root or ARTIFACTS_ROOT

    @property
    def checkpoint_db(self) -> Path:
        return self.artifacts_base / CHECKPOINT_FILENAME

    @property
    def run_dir(self) -> Path:
        return self.artifacts_base / self.thread_id

    @property
    def llm_dir(self) -> Path:
        """모든 LLM 프롬프트/응답 쌍이 보관되는 곳."""
        return self.run_dir / "llm"

    @property
    def train_dir(self) -> Path:
        """반복별 train config, 로그, result.json이 놓이는 곳."""
        return self.run_dir / "train"

    def iteration_dir(self, iteration: int) -> Path:
        return self.train_dir / f"iter_{iteration:02d}"

    # 넷 다 기억해 두는 대신 ``iteration_dir``에서 파생시킨다. 그러면 파일이 다른 적합과
    # 짝지어질 수 없다: ``predict``는 경로 둘을 받는 대신 한 디렉터리에서 모델과 스키마를 함께
    # 읽고, 예측은 state 채널이 필요 없다. 넷 다 없을 수 있다 — 그 반복이 각 파일을 남길
    # 지점까지 갔을 때만 있다.
    def model_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / MODEL_FILENAME

    def schema_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / SCHEMA_FILENAME

    def decision_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / DECISION_FILENAME

    def predictions_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / PREDICTIONS_FILENAME

    def ensure_dirs(self) -> None:
        for path in (self.artifacts_base, self.run_dir, self.llm_dir, self.train_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def fallback_threshold(self) -> float:
        """``goal`` 채널 밖에서 실수 threshold가 필요한 코드를 위한 값.

        ``goal.get("threshold", ...)`` 호출의 기본값일 뿐이다 — 그래야 호출마다 ``None`` 분기를
        적지 않는다. 그래프는 ``state["goal"]``을 읽고, 거기서 기준선은 이미 실제 숫자다.
        """
        return DEFAULT_THRESHOLD if self.threshold is None else self.threshold

    @property
    def train_timeout_sec(self) -> float:
        """학습 subprocess 하나의 상한. 넘기면 ``too_slow``로 기록된다.

        예산 전체라서, 어떤 적합도 자기 실행보다 오래 갈 수 없다는 말만 한다. 루프가 실제로
        적합에 주는 것은 남은 시간의 몫인 :func:`automl_agent.state.fit_share_sec`이고, 이것은
        state에 예산 계산이 없을 때(단위 테스트, 손으로 고친 체크포인트)의 fallback이다.
        """
        return float(self.time_budget_sec)


def file_size_text(size: float) -> str:
    """``509234754``를 ``485.6 MB``로. 1 KB 아래는 바이트, 그 위로 KB, MB, GB.

    바이트 경우는 장식이 아니다: 없으면 1 KB 아래 모든 파일이 ``0 KB``로 읽히는데, 그것은
    쓰기가 *실패한* 모습이다.

    여기 있는 이유: subprocess 경계 양쪽이 모두 artifact 크기를 출력하는데 노드에서
    ``scripts/train.py``를 import할 수 없다 — sklearn을 끌어오고, orchestrator 프로세스에는
    그것이 없다.
    """
    if size < 1024:
        return f"{size:.0f} B"
    for unit, cutoff in (("KB", 1024**2), ("MB", 1024**3)):
        if size < cutoff:
            value = size / (cutoff / 1024)
            return f"{value:.0f} {unit}" if unit == "KB" else f"{value:.1f} {unit}"
    return f"{size / 1024**3:.2f} GB"


def utf8_env() -> dict[str, str]:
    """자식 스크립트를 위한 환경. stdio를 UTF-8에 고정한다.

    자식 스크립트 둘 다 부모 쪽에서 ``encoding="utf-8"``으로 spawn된다. 이것이 없으면 자식은
    여전히 콘솔 코드 페이지(한국어 Windows에서는 cp949)로 *인코딩*해서, 프로파일러의 한글
    요약이 깨져 돌아왔다 — 부모가 디코딩을 locale에 맡긴 경우에는 subprocess의 reader 스레드
    안에서 터지면서 로그 전체를 잃었다.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def decode_output(raw: Any) -> str:
    """자식 프로세스의 스트림 하나를 텍스트로. ``subprocess``가 무엇을 돌려줬든.

    raise가 아니라 ``errors="replace"``인 이유: 이 함수는 실패 경로에서만, 운영자가 읽는
    ``log_tail``을 만들려고 호출된다. 깨끗한 UTF-8이 아닌 스트림 자체가 무엇이 잘못됐는지의
    일부다. 거기서 디코딩 오류로 로그 전체를 잃으면 tail이 나르려고 존재하는 메시지를 가린다 —
    나머지 절반은 :func:`utf8_env`.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def run_fixed_script(command: list[str], *, timeout: float, label: str) -> tuple[int, str]:
    """고정된 스크립트 하나를 띄우고 ``(returncode, 합쳐진 콘솔)``을 돌려준다.

    노드 셋(``training``, ``holdout``, ``profiling``)이 같은 열여덟 줄을 따로 갖고 있었다. spawn 자체가
    실패해도 raise하지 않는 것이 세 곳 모두의 계약이다 — 그것은 실행을 끝낼 일이 아니라 시도 하나가
    보고하는 실패다.

    두 실패에 붙는 returncode는 관례이고, 자식이 낼 수 없는 값으로 골랐다:

    ``-9``
        timeout. 시간 예산을 재는 것은 부모다.
    ``-1``
        spawn 실패. ``console``이 빈 문자열이 아니라 그 이유를 담고 돌아가므로, 호출자의
        ``log_tail``이 "왜 아무 로그도 없나"를 운영자에게 설명할 수 있다.

    ``encoding``을 locale에서 유도하지 않고 고정하는 이유는 :func:`utf8_env`와 :func:`decode_output`.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=utf8_env(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return -9, decode_output(exc.stdout) + decode_output(exc.stderr)
    except OSError as exc:
        return -1, f"failed to spawn the {label} subprocess: {exc}"
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def read_json_object(path: Path) -> dict[str, Any] | None:
    """``path``에 있는 JSON 객체, 읽을 것이 없으면 ``None``.

    실패하는 모든 방식에 ``None``이다 — 없음, 못 읽음, 형식이 깨짐, 객체가 아닌 유효한 JSON.
    호출자에게는 넷이 같은 뜻이기 때문이다: 자식 프로세스가 쓸 수 있는 artifact를 남기지
    않았고, 그것은 호출자가 이미 보고하는 시도 실패다. 운영자에게 필요한 구별은 exit code와
    로그 tail에 있고, 여기 있지 않다.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def use_bedrock() -> bool:
    return os.environ.get(BEDROCK_FLAG_ENV, "").strip().lower() in {"1", "true", "yes"}


def has_llm_credentials() -> bool:
    """실제 LLM 호출이 가능해 보일 때 True. 비밀값 자체는 절대 돌려주지 않는다."""
    if use_bedrock():
        return bool(os.environ.get(AWS_REGION_ENV))
    return bool(os.environ.get(API_KEY_ENV))


def bedrock_signing_available() -> bool:
    """Bedrock 경로가 요청에 서명할 수 있는지.

    SDK는 ``botocore``를 요청 시점에 lazy하게 import한다 — 이 preflight가 없으면 빠진 의존성이
    실행 시작 전의 분명한 메시지가 아니라 첫 노드 안에서 나온 traceback으로 드러난다.
    """
    return importlib.util.find_spec("botocore") is not None
