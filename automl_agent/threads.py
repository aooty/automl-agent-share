"""적합이 어떤 BLAS/OpenMP 스레드 상태에서 돌았는지 — 점수를 움직이므로 기록한다.

**"같은 seed, 같은 설정"은 "같은 숫자"가 아니다.** 스레드 수가 다르면 부동소수점 덧셈의 순서가
달라지고 점수가 움직인다.

**이 모듈은 기록만 한다.** 변수를 설정하지 않고, 어떤 값을 두고 경고하지도 않는다.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# 이 셋이고 더는 아니다 — 저장소의 estimator가 실제로 읽는 것.
THREAD_ENV: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

CPU_COUNT_KEY = "cpu_count"


def thread_state() -> dict[str, Any]:
    """이 프로세스의 스레드 환경, 기록되는 모양 그대로.

    설정되지 않은 변수는 기본값이 아니라 ``None``이고, 코어 수가 함께 담긴다 — unset이 무엇으로
    풀리는지는 코어 수에 달려 있다.
    """
    state: dict[str, Any] = {key: os.environ.get(key) for key in THREAD_ENV}
    state[CPU_COUNT_KEY] = os.cpu_count()
    return state


def describe_thread_state(state: Mapping[str, Any] | None) -> str:
    """로그 한 줄: ``OMP_NUM_THREADS=1 ... MKL_NUM_THREADS=unset cpu_count=20``.

    없는 값은 빈칸이 아니라 ``unset``으로 적는다.
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
    """각 변수가 적합에 대해 실제로 무엇으로 풀리는지. unset은 코어 수로 접는다.

    날값을 비교하면 두 기록을 거꾸로 판정한다.
    """
    cores = state.get(CPU_COUNT_KEY)
    return tuple(
        str(state[key]) if state.get(key) is not None else f"{CPU_COUNT_KEY}:{cores}"
        for key in THREAD_ENV
    )


def thread_state_changed(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> bool | None:
    """두 기록이 다르게 적합되는 스레드 상태를 말하는지. 알 수 없으면 ``None``.

    ``None``은 ``False``가 아니다: 이 모듈이 생기기 전의 기록이거나 변수가 빠진 기록이라는 뜻이고,
    소비자는 그것을 침묵으로, ``True``를 경고로 발표한다.
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    if not all(key in before and key in after for key in THREAD_ENV):
        return None
    return _effective(before) != _effective(after)
