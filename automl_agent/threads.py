"""Record the thread settings a fit ran under; they can move the score.

Roles:

* Thread recording — capture thread variables and core count.
* Thread comparison — tell whether two records fit differently.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Same tuple bench/paired.py records
THREAD_ENV: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

CPU_COUNT_KEY = "cpu_count"


# --- Role: thread recording -------------------------------------------------------


def thread_state() -> dict[str, Any]:
    """Return this process's thread variables (``None`` when unset) plus core count."""
    state: dict[str, Any] = {key: os.environ.get(key) for key in THREAD_ENV}
    state[CPU_COUNT_KEY] = os.cpu_count()
    return state


def describe_thread_state(state: Mapping[str, Any] | None) -> str:
    """Format a thread record as one log line, writing ``unset`` for missing values.

    A non-mapping gives ``unrecorded``.
    """
    if not isinstance(state, Mapping):
        return "unrecorded"

    def shown(key: str, absent: str) -> str:
        value = state.get(key)
        return f"{key}={absent if value is None else value}"

    parts = [shown(key, "unset") for key in THREAD_ENV]
    parts.append(shown(CPU_COUNT_KEY, "?"))
    return " ".join(parts)


def _effective(state: Mapping[str, Any]) -> tuple[str, ...]:
    """_effective | Thread comparison: what each variable means (unset = core count)."""
    cores = state.get(CPU_COUNT_KEY)
    return tuple(
        str(state[key]) if state.get(key) is not None else f"{CPU_COUNT_KEY}:{cores}"
        for key in THREAD_ENV
    )


# --- Role: thread comparison -----------------------------------------------------


def thread_state_changed(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> bool | None:
    """Tell whether two thread records differ; ``None`` when it cannot be told.

    Callers stay silent on ``None`` and warn on ``True``.
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if not all(key in before and key in after for key in THREAD_ENV):
        return None
    return _effective(before) != _effective(after)
