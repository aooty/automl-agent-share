"""Run configuration.

Nodes never read globals: ``build_graph`` binds this object into each node with
``functools.partial``, so a node stays a pure ``state -> dict`` function.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset.targets import TARGET_MISSING_POLICIES

# goal.py imports nothing from the package, so this direction stays acyclic.
from .scoring.goal import DEFAULT_MARGIN, DEFAULT_MODE, GOAL_MODES
from .scoring.metrics import DEFAULT_METRICS, GOAL_METRICS, TASK_CLASSIFICATION, canonical, direction_of

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
TRAIN_SCRIPT = PACKAGE_DIR / "scripts" / "train.py"
PROFILE_SCRIPT = PACKAGE_DIR / "scripts" / "profile.py"
PREDICT_SCRIPT = PACKAGE_DIR / "scripts" / "predict.py"
PROMPTS_DIR = PACKAGE_DIR / "llm" / "prompts"
CHECKPOINT_DB = ARTIFACTS_ROOT / "checkpoints.sqlite"

# How long a checkpoint write waits for another process to release the database before it
# gives up. One file holds every thread_id, so ``show`` or a second ``run`` can hold the
# lock; sqlite's own default is 5 seconds, which is short next to an hour of training
# already spent. See graph.make_checkpointer.
CHECKPOINT_TIMEOUT_SEC = 60.0

# The fitted pipeline every training iteration leaves in its own directory. Named here so
# the script that writes it and the node that reads it back cannot disagree — and kept
# inside ``artifacts/``, which ``.gitignore`` blocks, because a fitted estimator is
# data-equivalent (an SVC stores its support vectors verbatim).
MODEL_FILENAME = "model.joblib"

# The validation-row predictions the same iteration leaves beside its model, so a later
# attempt can be compared against it *on the same rows* instead of point estimate against
# point estimate (:func:`automl_agent.scoring.intervals.paired_delta`). One value per row, so it is
# data-equivalent in exactly the way the model file is, and it stays on the same side of the
# boundary: no node reads its contents, only its path, and the path is derivable from the
# iteration number rather than carried in state.
PREDICTIONS_FILENAME = "val_predictions.npz"

# The encoding that produced the matrix ``model.joblib`` was fitted on: the ordered encoded
# column names, each categorical column's level set, and what each class code means. Written
# beside the model because it is half of the same artifact — the estimator alone cannot say
# which column was which, so a model without this file can only be applied to rows that
# happen to encode identically, and nothing checks that they did.
#
# Data-equivalent, on the same terms as the two files above: category levels and class labels
# are cell values. So it stays inside ``artifacts/``, its path is absent from
# ``privacy.PUBLIC_RESULT_FIELDS``, and no reasoning node reads it.
SCHEMA_FILENAME = "feature_schema.json"

# Read from the registry rather than restated, so the CLI default and the metric a
# classification run falls back to after a substitution are the same name by construction.
DEFAULT_METRIC = DEFAULT_METRICS[TASK_CLASSIFICATION]
DEFAULT_THRESHOLD = 0.85
# Accepted values for ``--direction``. There is deliberately no default constant: the
# default is whatever the metric declares (automl_agent.scoring.metrics.direction_of), and a
# module-level "maximize" would be a second source of truth for the same fact.
DIRECTIONS = ("maximize", "minimize")
DEFAULT_MAX_ITERATIONS = 5
DRY_RUN_SCENARIOS = ("success", "fail", "oom", "stall", "slow", "crash")
DEFAULT_TIME_BUDGET_SEC = 3600
STALL_LIMIT = 2  # consecutive non-improving iterations before giving up
# Profiling reads the file once and computes column aggregates; it is bounded
# separately from training because it runs before the loop's time budget applies.
PROFILE_TIMEOUT_SEC = 900.0

# What to keep of ``model.joblib`` when the run ends. Measured on this repository's own
# artifacts: 1.79 GB of 1.9 GB was fitted models, 486 MB of it a single iteration whose
# proposal happened to be a deep forest. Nothing in the loop bounds that — the memory guard
# prices the input matrix, not the fitted estimator — so a long run's real limit is the disk.
#
# The default keeps the one model that can still be used (``predict`` resolves to ``best``)
# and drops the attempts nothing points at. "all" is the older behaviour, and it is what
# ``predict --iteration <다른 번호>`` needs, so the pruning names the flag when it runs.
KEEP_MODELS_MODES = ("best", "all")
DEFAULT_KEEP_MODELS = KEEP_MODELS_MODES[0]

DEFAULT_LLM_MODEL = "claude-opus-5"
DEFAULT_LLM_MAX_TOKENS = 8000
DEFAULT_LLM_TIMEOUT_SEC = 180.0
# Transport-level retries the SDK performs on 429/5xx with exponential backoff.
# Kept generous because a single transient 5xx costs a whole reasoning node: the
# node falls back to heuristics and the run silently loses its LLM judgment.
DEFAULT_LLM_MAX_RETRIES = 4

API_KEY_ENV = "ANTHROPIC_API_KEY"
BEDROCK_FLAG_ENV = "AUTOML_USE_BEDROCK"
AWS_REGION_ENV = "AWS_REGION"


@dataclass(frozen=True)
class RunConfig:
    """Immutable settings for a single run (one ``thread_id``)."""

    thread_id: str
    metric: str = DEFAULT_METRIC
    # "auto": derive the bar from the card's reference baseline. "fixed": use
    # ``threshold``, or the per-metric default if it is None. See automl_agent.scoring.goal.
    goal_mode: str = DEFAULT_MODE
    # Only read in "fixed" mode. Setting it implies that mode.
    threshold: float | None = None
    # "auto" only: how much of the baseline's remaining headroom to ask for.
    goal_margin: float = DEFAULT_MARGIN
    # Which way is better. ``None`` means "read it off the metric", which is the only
    # honest source — see automl_agent.scoring.metrics.direction_of. Resolved in ``__post_init__``,
    # so every reader downstream gets a real value.
    direction: str | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    time_budget_sec: int = DEFAULT_TIME_BUDGET_SEC
    stall_limit: int = STALL_LIMIT
    dry_run: bool = False
    # Which trajectory the mocked trainer follows under --dry-run.
    dry_run_scenario: str = "success"
    # Real training, but rule-based reasoning instead of LLM calls. Lets the full
    # execution path be exercised without credentials.
    no_llm: bool = False
    seed: int = 42
    llm_model: str = DEFAULT_LLM_MODEL
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC
    dataset_card_path: Path | None = None
    # The raw data reference. Held here (and in the ``data_ref`` state channel) rather
    # than inside the dataset card, so it cannot ride into a prompt with the card.
    data_path: Path | None = None
    target_column: str | None = None
    # Rows whose label is missing: "reject" or "drop". None means "follow the card, and
    # reject if it says nothing" — see nodes/training.py::build_train_config.
    on_missing_target: str | None = None
    # Operator notes about the raw data, passed to the profiler and from there into every
    # reasoning prompt (automl_agent.dataset.caveats). Only used on the ``--data`` path: a card
    # supplied with ``--dataset-card`` already carries its own.
    caveats: tuple[str, ...] = ()
    # A column no group of which may be divided across train/val/test — see
    # automl_agent.scoring.splits. Held here with the other data references rather than in the
    # card's public part, for the same reason ``data_path`` is: it decides which rows are
    # held out, and the LLM must not be able to propose changing it.
    group_column: str | None = None
    # Overridable so tests (and side-by-side runs) can relocate all output.
    artifacts_root: Path | None = None
    # Which iterations keep their fitted model once the run is over — see KEEP_MODELS_MODES.
    keep_models: str = DEFAULT_KEEP_MODELS

    def __post_init__(self) -> None:
        # Checked here rather than in the node that reads it: a bad value should stop the
        # run before it spends a minute profiling, and ``resume`` rebuilds this object
        # from disk, so the same guard covers a hand-edited run_config.json.
        #
        # Every check below is a value that used to produce a run that *looked* normal and
        # reported nonsense: max_iterations=-1 skipped the loop and reported "budget
        # reached", a NaN threshold made goal_met False for any score whatsoever.
        if self.goal_mode not in GOAL_MODES:
            raise ValueError(
                f"goal_mode는 {GOAL_MODES} 중 하나여야 합니다 (받은 값: {self.goal_mode!r})"
            )
        metric = canonical(self.metric)
        if metric not in GOAL_METRICS:
            raise ValueError(
                f"metric은 {GOAL_METRICS} 중 하나여야 합니다 (받은 값: {self.metric!r}). "
                "학습 스크립트가 만들지 않는 지표를 목표로 잡으면 그 실행은 무엇을 해도 "
                "목표를 달성할 수 없습니다"
            )
        if metric != self.metric:
            # Alias in, canonical name out, so every downstream reader compares one name.
            object.__setattr__(self, "metric", metric)
        implied = direction_of(metric)
        if self.direction is None:
            # The normal path: nobody has to know which way ``rmse`` goes.
            object.__setattr__(self, "direction", implied)
        elif self.direction not in DIRECTIONS:
            raise ValueError(f"direction은 {DIRECTIONS} 중 하나여야 합니다 (받은 값: {self.direction!r})")
        elif self.direction != implied:
            # An explicit direction is accepted only as an assertion of what the metric
            # already implies. The other combination is not a preference: ``--direction
            # minimize --metric f1`` inverted the entire run — ``best`` kept the worst
            # attempt and ``goal_met`` fired on any score *below* the bar.
            raise ValueError(
                f"{metric}은 {implied} 지표라서 direction={self.direction!r}로 실행할 수 없습니다. "
                "방향은 지표에서 나오므로 --direction 은 생략하거나 지표와 같은 값을 주십시오"
            )
        if self.dry_run_scenario not in DRY_RUN_SCENARIOS:
            raise ValueError(
                f"dry_run_scenario는 {DRY_RUN_SCENARIOS} 중 하나여야 합니다 "
                f"(받은 값: {self.dry_run_scenario!r})"
            )
        if self.keep_models not in KEEP_MODELS_MODES:
            raise ValueError(
                f"keep_models는 {KEEP_MODELS_MODES} 중 하나여야 합니다 (받은 값: {self.keep_models!r})"
            )
        if self.on_missing_target is not None and self.on_missing_target not in TARGET_MISSING_POLICIES:
            raise ValueError(
                f"on_missing_target은 {TARGET_MISSING_POLICIES} 중 하나여야 합니다 "
                f"(받은 값: {self.on_missing_target!r})"
            )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations는 1 이상이어야 합니다 (받은 값: {self.max_iterations!r})")
        if self.time_budget_sec <= 0:
            raise ValueError(f"time_budget_sec는 0보다 커야 합니다 (받은 값: {self.time_budget_sec!r})")
        if self.stall_limit < 1:
            raise ValueError(f"stall_limit은 1 이상이어야 합니다 (받은 값: {self.stall_limit!r})")
        if self.seed < 0:
            raise ValueError(f"seed는 0 이상이어야 합니다 (받은 값: {self.seed!r})")
        if not 0.0 < self.goal_margin < 1.0:
            # 0 asks for exactly the baseline, 1 asks for a perfect score: both make the
            # bar meaningless rather than merely strict.
            raise ValueError(
                f"goal_margin은 0과 1 사이여야 합니다 (받은 값: {self.goal_margin!r})"
            )
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise ValueError(f"threshold는 유한한 숫자여야 합니다 (받은 값: {self.threshold!r})")
        if not isinstance(self.caveats, tuple):
            # ``resume`` rebuilds this object from run_config.json, where the tuple was
            # serialised as a JSON array — so without this a resumed run's caveats would be
            # a list and the dataclass would no longer be hashable or comparable.
            object.__setattr__(self, "caveats", tuple(str(item) for item in self.caveats or ()))

    # -- derived paths ----------------------------------------------------- #

    @property
    def use_llm(self) -> bool:
        """Whether reasoning nodes should call the API at all."""
        return not (self.dry_run or self.no_llm)

    @property
    def artifacts_base(self) -> Path:
        return self.artifacts_root or ARTIFACTS_ROOT

    @property
    def checkpoint_db(self) -> Path:
        return self.artifacts_base / "checkpoints.sqlite"

    @property
    def run_dir(self) -> Path:
        return self.artifacts_base / self.thread_id

    @property
    def llm_dir(self) -> Path:
        """Where every LLM prompt/response pair is archived."""
        return self.run_dir / "llm"

    @property
    def train_dir(self) -> Path:
        """Where per-iteration train configs, logs and result.json land."""
        return self.run_dir / "train"

    def iteration_dir(self, iteration: int) -> Path:
        return self.train_dir / f"iter_{iteration:02d}"

    def model_path(self, iteration: int) -> Path:
        """Where that iteration's fitted model was saved, if it got far enough to save one."""
        return self.iteration_dir(iteration) / MODEL_FILENAME

    def schema_path(self, iteration: int) -> Path:
        """Where that iteration's feature schema was saved, if it got as far as fitting.

        Beside its model, and derived the same way, so the pair cannot be separated by
        bookkeeping: ``predict`` reads both out of one directory rather than being told two
        paths that could belong to different fits.
        """
        return self.iteration_dir(iteration) / SCHEMA_FILENAME

    def predictions_path(self, iteration: int) -> Path:
        """Where that iteration's validation-row predictions were saved, if it scored any.

        Derived rather than remembered, for the same reason ``iteration_dir`` is: the file
        holds one value per validation row, so its *contents* must never enter a state
        channel, and a path that is a function of the iteration number needs no channel.
        """
        return self.iteration_dir(iteration) / PREDICTIONS_FILENAME

    def ensure_dirs(self) -> None:
        for path in (self.artifacts_base, self.run_dir, self.llm_dir, self.train_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def fallback_threshold(self) -> float:
        """A float threshold for code that needs one outside the ``goal`` channel.

        The graph always reads ``state["goal"]``, whose threshold is a real number by the
        time any node inside the loop runs: a metric in the target's own units can leave it
        unset, and ``profiling`` refuses the run rather than letting it start against a bar
        that does not exist. This exists for the ``goal.get("threshold", ...)`` defaults,
        which would otherwise have to spell out a ``None`` branch each.
        """
        return DEFAULT_THRESHOLD if self.threshold is None else self.threshold

    @property
    def train_timeout_sec(self) -> float:
        """subprocess timeout. Exceeding it is recorded as ``too_slow``."""
        return float(self.time_budget_sec)


def file_size_text(size: float) -> str:
    """``509234754`` as ``485.6 MB``. Bytes below a kilobyte, then KB, MB, GB.

    The byte case is not decoration: without it every file under 1 KB reads as ``0 KB``, and
    "0 KB" is what a *failed* write looks like.

    Lives here rather than beside its callers because both sides of the subprocess boundary
    print artifact sizes — the training script when it saves a model, the report node when it
    deletes one — and ``scripts/train.py`` cannot be imported from a node (it pulls in
    sklearn, and the orchestrator process does not).

    The model line used to be printed as KB unconditionally, which turned the one number that
    would have made this repository's 1.9 GB of artifacts visible ("509234 KB") into a number
    nobody reads.
    """
    if size < 1024:
        return f"{size:.0f} B"
    for unit, cutoff in (("KB", 1024**2), ("MB", 1024**3)):
        if size < cutoff:
            value = size / (cutoff / 1024)
            return f"{value:.0f} {unit}" if unit == "KB" else f"{value:.1f} {unit}"
    return f"{size / 1024**3:.2f} GB"


def utf8_env() -> dict[str, str]:
    """Environment for a child script, with its stdio pinned to UTF-8.

    Both child scripts are spawned with ``encoding="utf-8"`` on the parent side. Without
    this the child would still *encode* with the console code page (cp949 on a Korean
    Windows), so the profiler's Korean summary came back mangled — or, when the parent
    left decoding to the locale, blew up inside subprocess' reader thread and lost the
    whole log.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def decode_output(raw: Any) -> str:
    """One of a child process' streams as text, whatever ``subprocess`` handed back.

    ``errors="replace"`` rather than a raise: this is only ever called on the failure path,
    to build the ``log_tail`` an operator reads, and a stream that is not clean UTF-8 is
    itself part of what went wrong. Losing the whole log to a decode error there would hide
    the message the tail exists to carry — see :func:`utf8_env` for the other half.

    Lives here beside ``utf8_env`` because all three subprocess nodes (``profiling``,
    ``training``, ``holdout``) need it and each used to keep its own copy.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """The JSON object at ``path``, or ``None`` if there is not one to read.

    ``None`` for every way this can fail — absent, unreadable, malformed, or valid JSON that
    is not an object — because all four mean the same thing to the callers: the child process
    did not leave a usable artifact, which is an attempt failure they already report on. The
    distinction the operator needs is in the exit code and the log tail, not here.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def use_bedrock() -> bool:
    """Route LLM calls through Amazon Bedrock instead of the direct API."""
    return os.environ.get(BEDROCK_FLAG_ENV, "").strip().lower() in {"1", "true", "yes"}


def has_llm_credentials() -> bool:
    """True when a live LLM call is plausible. Never returns the secret itself."""
    if use_bedrock():
        return bool(os.environ.get(AWS_REGION_ENV))
    return bool(os.environ.get(API_KEY_ENV))


def bedrock_signing_available() -> bool:
    """Whether the Bedrock route can sign requests.

    The SDK imports ``botocore`` lazily, at request time — so without this preflight
    a missing dependency surfaces as a traceback from inside the first node instead
    of a clear message before the run starts.
    """
    return importlib.util.find_spec("botocore") is not None
