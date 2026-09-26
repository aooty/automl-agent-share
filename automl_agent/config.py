"""Run settings, and small helpers shared by the nodes that spawn scripts.

Roles:

* Paths and constants — file names, defaults, limits, env names.
* Run config — the frozen settings of one run, checked when built.
* Artifact paths — where one run and one iteration keep their files.
* Script running — spawn scripts, read their output, format sizes.
* LLM access — tell whether LLM calls can be made.
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

# goal.py imports nothing from the package, so no cycle.
from .scoring.goal import DEFAULT_MARGIN, DEFAULT_MODE, GOAL_MODES
from .scoring.metrics import DEFAULT_METRICS, GOAL_METRICS, TASK_CLASSIFICATION, canonical, direction_of

# --- Role: paths and constants --------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
# Only the scripts the graph spawns; predict.py runs in-process.
TRAIN_SCRIPT = PACKAGE_DIR / "scripts" / "train.py"
PROFILE_SCRIPT = PACKAGE_DIR / "scripts" / "profile.py"
PROMPTS_DIR = PACKAGE_DIR / "llm" / "prompts"
# Also used by RunConfig.checkpoint_db.
CHECKPOINT_FILENAME = "checkpoints.sqlite"
CHECKPOINT_DB = ARTIFACTS_ROOT / CHECKPOINT_FILENAME

# How long to wait for another process's lock.
CHECKPOINT_TIMEOUT_SEC = 60.0

# Data-level files: stay in artifacts/, never in a prompt.
MODEL_FILENAME = "model.joblib"
PREDICTIONS_FILENAME = "val_predictions.npz"
SCHEMA_FILENAME = "feature_schema.json"

# Tuned cut; written only when the plan asks for one.
DECISION_FILENAME = "decision_rule.json"

DEFAULT_METRIC = DEFAULT_METRICS[TASK_CLASSIFICATION]
DEFAULT_THRESHOLD = 0.85
# No default: it comes from the metric.
DIRECTIONS = ("maximize", "minimize")
DEFAULT_MAX_ITERATIONS = 5
DRY_RUN_SCENARIOS = ("success", "fail", "oom", "stall", "slow", "crash")
DEFAULT_DRY_RUN_SCENARIO = DRY_RUN_SCENARIOS[0]
DEFAULT_TIME_BUDGET_SEC = 3600
STALL_LIMIT = 2  # stop after this many non-improving iterations in a row

# Budget share kept for holdout
HOLDOUT_RESERVE_FRACTION = 0.1
# Floor for one fit's timeout
MIN_FIT_TIMEOUT_SEC = 1.0

# Profiling runs before the loop's budget starts.
PROFILE_TIMEOUT_SEC = 900.0

# Which model.joblib files survive the run.
KEEP_MODELS_MODES = ("best", "all")
DEFAULT_KEEP_MODELS = KEEP_MODELS_MODES[0]

DEFAULT_LLM_MODEL = "claude-opus-5"
DEFAULT_LLM_MAX_TOKENS = 8000
DEFAULT_LLM_TIMEOUT_SEC = 180.0
# SDK retries on 429/5xx; a miss costs a whole node.
DEFAULT_LLM_MAX_RETRIES = 4

API_KEY_ENV = "ANTHROPIC_API_KEY"
BEDROCK_FLAG_ENV = "AUTOML_USE_BEDROCK"
AWS_REGION_ENV = "AWS_REGION"


# --- Role: run config -----------------------------------------------------------------


# ``label`` carries the Korean particle, e.g. "seed는".
def _one_of(label: str, value: Any, allowed: tuple[str, ...]) -> None:
    """_one_of | Run config: raise ``ValueError`` unless ``value`` is in ``allowed``."""
    if value not in allowed:
        raise ValueError(f"{label} {allowed} 중 하나여야 합니다 (받은 값: {value!r})")


def _at_least(label: str, value: float, floor: int) -> None:
    """_at_least | Run config: raise ``ValueError`` if ``value`` is below ``floor``."""
    if value < floor:
        raise ValueError(f"{label} {floor} 이상이어야 합니다 (받은 값: {value!r})")


@dataclass(frozen=True)
class RunConfig:
    """Frozen settings of one run (one ``thread_id``).

    ``__post_init__`` checks values and fills ``direction``; ``ValueError`` if bad.
    """

    thread_id: str
    metric: str = DEFAULT_METRIC
    # "auto": bar from the baseline. "fixed": use ``threshold``.
    goal_mode: str = DEFAULT_MODE
    # Used only in "fixed" mode.
    threshold: float | None = None
    # "auto" only: share of the room above the baseline.
    goal_margin: float = DEFAULT_MARGIN
    # None: filled from the metric in __post_init__.
    direction: str | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    time_budget_sec: int = DEFAULT_TIME_BUDGET_SEC
    stall_limit: int = STALL_LIMIT
    # Keep going after the goal is met
    search_past_goal: bool = False
    dry_run: bool = False
    dry_run_scenario: str = DEFAULT_DRY_RUN_SCENARIO
    # Real training, rule-based reasoning, no credentials needed.
    no_llm: bool = False
    seed: int = 42
    llm_model: str = DEFAULT_LLM_MODEL
    # Planner and selector model; empty means ``llm_model``.
    proposer_model: str = ""
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC
    dataset_card_path: Path | None = None
    # Kept out of the card, so no prompt sees it.
    # May be a URL string, so not always a Path.
    data_path: Path | str | None = None
    # Database source: a table or a query, not both.
    data_table: str | None = None
    data_query: str | None = None
    target_column: str | None = None
    # None: follow the card, else reject.
    on_missing_target: str | None = None
    # Operator notes on the raw data; used only with --data.
    caveats: tuple[str, ...] = ()
    # No group is split across train/val/test.
    group_column: str | None = None
    # Tests move all output here.
    artifacts_root: Path | None = None
    keep_models: str = DEFAULT_KEEP_MODELS

    def __post_init__(self) -> None:
        # Each check once let a bad run look fine.
        _one_of("goal_mode는", self.goal_mode, GOAL_MODES)
        metric = canonical(self.metric)
        if metric not in GOAL_METRICS:
            raise ValueError(
                f"metric은 {GOAL_METRICS} 중 하나여야 합니다 (받은 값: {self.metric!r}). "
                "학습 스크립트가 만들지 않는 지표를 목표로 잡으면 그 실행은 무엇을 해도 "
                "목표를 달성할 수 없습니다"
            )
        if metric != self.metric:
            # Keep the main name, not the alias.
            object.__setattr__(self, "metric", metric)
        implied = direction_of(metric)
        if self.direction is None:
            object.__setattr__(self, "direction", implied)
        else:
            _one_of("direction은", self.direction, DIRECTIONS)
            if self.direction != implied:
                # Only the metric's own direction is allowed.
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
            # 0 or 1 makes the threshold meaningless.
            raise ValueError(f"goal_margin은 0과 1 사이여야 합니다 (받은 값: {self.goal_margin!r})")
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise ValueError(f"threshold는 유한한 숫자여야 합니다 (받은 값: {self.threshold!r})")
        if not isinstance(self.caveats, tuple):
            # resume reads a JSON list; a tuple keeps it hashable.
            object.__setattr__(self, "caveats", tuple(str(item) for item in self.caveats or ()))
        if self.data_table and self.data_query:
            # Reject before profiling starts.
            raise ValueError(
                "--table 과 --query 는 같이 쓸 수 없습니다 — 읽을 행을 정하는 방식이 서로 "
                "다릅니다. --table 은 SELECT * FROM <이름> 의 줄임입니다"
            )
        if (self.data_table or self.data_query) and not self.data_path:
            raise ValueError("--table/--query 는 --data 없이는 가리킬 데이터베이스가 없습니다")

    # --- Role: artifact paths ---------------------------------------------------------

    @property
    def use_llm(self) -> bool:
        """True unless the run is a dry run or ``no_llm``."""
        return not (self.dry_run or self.no_llm)

    @property
    def artifacts_base(self) -> Path:
        """Root folder for all output: ``artifacts_root``, else ``artifacts/``."""
        return self.artifacts_root or ARTIFACTS_ROOT

    @property
    def checkpoint_db(self) -> Path:
        """The checkpoint database under :attr:`artifacts_base`."""
        return self.artifacts_base / CHECKPOINT_FILENAME

    @property
    def run_dir(self) -> Path:
        """This run's folder, named by ``thread_id``."""
        return self.artifacts_base / self.thread_id

    @property
    def llm_dir(self) -> Path:
        """Folder where every LLM prompt/response pair is kept."""
        return self.run_dir / "llm"

    @property
    def train_dir(self) -> Path:
        """Folder for each iteration's train config, log, and result.json."""
        return self.run_dir / "train"

    def iteration_dir(self, iteration: int) -> Path:
        """Folder of one iteration, e.g. ``train/iter_03``."""
        return self.train_dir / f"iter_{iteration:02d}"

    # Built, not stored, so model and schema always match.
    def model_path(self, iteration: int) -> Path:
        """Path of the fitted model file of one iteration."""
        return self.iteration_dir(iteration) / MODEL_FILENAME

    def schema_path(self, iteration: int) -> Path:
        """Path of the feature schema file of one iteration."""
        return self.iteration_dir(iteration) / SCHEMA_FILENAME

    def decision_path(self, iteration: int) -> Path:
        """Path of the decision rule file of one iteration."""
        return self.iteration_dir(iteration) / DECISION_FILENAME

    def predictions_path(self, iteration: int) -> Path:
        """Path of the validation predictions file of one iteration."""
        return self.iteration_dir(iteration) / PREDICTIONS_FILENAME

    def ensure_dirs(self) -> None:
        """Create the output, run, LLM, and train folders if missing."""
        for path in (self.artifacts_base, self.run_dir, self.llm_dir, self.train_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def fallback_threshold(self) -> float:
        """Default for ``goal.get("threshold", ...)``, so callers need no ``None`` branch."""
        return DEFAULT_THRESHOLD if self.threshold is None else self.threshold

    @property
    def train_timeout_sec(self) -> float:
        """Fallback fit timeout: the whole budget

        The loop really uses ``state.fit_share_sec``; over it is ``too_slow``.
        """
        return float(self.time_budget_sec)


# --- Role: script running -------------------------------------------------------------


def file_size_text(size: float) -> str:
    """Format bytes for people, e.g. ``509234754`` as ``485.6 MB``"""
    if size < 1024:
        return f"{size:.0f} B"
    for unit, cutoff in (("KB", 1024**2), ("MB", 1024**3)):
        if size < cutoff:
            value = size / (cutoff / 1024)
            return f"{value:.0f} {unit}" if unit == "KB" else f"{value:.1f} {unit}"
    return f"{size / 1024**3:.2f} GB"


def utf8_env() -> dict[str, str]:
    """Return ``os.environ`` with child stdio fixed to UTF-8"""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def decode_output(raw: Any) -> str:
    """Turn a child stream (``None``, bytes, or str) into text; never raises."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def run_fixed_script(command: list[str], *, timeout: float, label: str) -> tuple[int, str]:
    """Spawn a fixed script; return ``(returncode, console)``. Never raises.

    ``-9`` means timeout; ``-1`` means spawn failed, with the reason in ``console``.
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
    """Read the JSON object at ``path``; ``None`` on any failure or non-object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# --- Role: LLM access -----------------------------------------------------------------


def use_bedrock() -> bool:
    """True when ``AUTOML_USE_BEDROCK`` is set to 1, true, or yes."""
    return os.environ.get(BEDROCK_FLAG_ENV, "").strip().lower() in {"1", "true", "yes"}


def has_llm_credentials() -> bool:
    """True if ``AWS_REGION`` (Bedrock) or ``ANTHROPIC_API_KEY`` is set."""
    if use_bedrock():
        return bool(os.environ.get(AWS_REGION_ENV))
    return bool(os.environ.get(API_KEY_ENV))


def bedrock_signing_available() -> bool:
    """True if ``botocore`` is installed, so Bedrock requests can be signed."""
    return importlib.util.find_spec("botocore") is not None
