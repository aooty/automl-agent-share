"""그래프 노드.

추론 노드(LLM): ``planning``, ``model_selection``(판단 쪽 절반), ``critic``, ``report``.
실행 노드(결정적인 코드, LLM 없음): ``profiling``, ``training``, ``evaluate``, ``holdout``.

모든 노드는 ``(state, *, config) -> dict``이고 부분 state 갱신을 돌려준다. ``config``는
``build_graph``가 ``functools.partial``로 묶어 준다 — 전역값 없음.
"""
