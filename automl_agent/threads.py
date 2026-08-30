"""Which BLAS/OpenMP thread state a fit ran in, recorded because it moves the score.

Measured on the MIMIC card with ``tree_method="hist"``: two runs of a key-for-key identical
training config *in the same shell* agree to the last bit across all 9,429 validation rows,
and the same config across ``OMP_NUM_THREADS`` 1..20 spans balanced_accuracy 0.007691 — 0.99x
the half-width of a single paired verdict on that run — with ``max |delta proba|`` 0.34.
Histogram summation over a different number of partial buffers adds in a different order, and
floating-point addition is not associative.

So "same seed, same config" is not "same numbers", and until this module existed the harness
wrote down every input to a fit *except* the one that decides which of the two it is. A run
directory could hold two attempts fitted under different thread counts, and the ledger would
publish the difference between them as the plan's doing.

This module only *records*. It does not set the variables and does not warn about a value:
pinning a fit to one thread would make it reproducible and several times slower, and that
trade belongs to whoever is running it. What is not theirs to make is the choice to leave the
number out of the record — a measurement whose environment is unstated cannot be compared
against another one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# These three and not more. They are the ones the estimators in this repository actually
# read — OpenMP for xgboost and sklearn's histogram trees, OpenBLAS and MKL for the linear
# algebra under logreg and the MLP. ``NUMEXPR_NUM_THREADS`` is pandas' and touches loading,
# not fitting. Same tuple as ``bench/paired.py`` records, so a bench adjudication and a run's
# own artifacts can be read against each other.
THREAD_ENV: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

CPU_COUNT_KEY = "cpu_count"


def thread_state() -> dict[str, Any]:
    """The thread environment of this process, in the shape it is recorded in.

    ``None`` for an unset variable rather than a default, because unset does not mean one: it
    means the library chooses, and what it chooses is derived from the core count of the
    machine. Both facts go in the block, so a reader holding two records can tell "unset in
    both, and the same number of cores" from "unset on 4 cores, unset on 64" — which are the
    same three ``None``\\ s and different arithmetic.
    """
    state: dict[str, Any] = {key: os.environ.get(key) for key in THREAD_ENV}
    state[CPU_COUNT_KEY] = os.cpu_count()
    return state


def describe_thread_state(state: Mapping[str, Any] | None) -> str:
    """One log line: ``OMP_NUM_THREADS=1 ... MKL_NUM_THREADS=unset cpu_count=20``.

    ``unset`` spelled out rather than left blank, because a blank reads as a value of zero
    or as a truncated line, and this string is written into ``log_tail``, which is where a
    human looks first.
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
    """What each variable resolves to for the fit, unset folded into the core count.

    Comparing the raw values would call two records different when they are not and the same
    when they are not. ``OMP_NUM_THREADS=1`` on a 4-core box and on a 64-core box run the same
    arithmetic; all three unset on those two boxes do not. So the core count enters the
    comparison only where a variable is unset — which is exactly where the library reads it.
    """
    cores = state.get(CPU_COUNT_KEY)
    return tuple(
        str(state[key]) if state.get(key) is not None else f"{CPU_COUNT_KEY}:{cores}"
        for key in THREAD_ENV
    )


def thread_state_changed(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> bool | None:
    """Whether two records describe thread states that fit differently. ``None`` if unknowable.

    ``None`` — not ``False`` — for a record made before this module existed, or one missing a
    variable. "We did not check" and "we checked and they match" are different facts, and the
    absent field would be read as the second one, which is the failure this whole module is
    about. Consumers publish the ``None`` as silence and the ``True`` as a warning
    (:func:`automl_agent.scoring.intervals.describe_paired`).
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if not all(key in before and key in after for key in THREAD_ENV):
        return None
    return _effective(before) != _effective(after)
