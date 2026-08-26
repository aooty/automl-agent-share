"""Graph nodes.

Reasoning nodes (LLM): ``planning``, ``model_selection`` (the judgement half),
``critic``, ``report``.
Execution nodes (deterministic code, no LLM): ``training``, ``evaluate``.

Every node is ``(state, *, config) -> dict`` and returns a partial state update.
``config`` is bound by ``build_graph`` via ``functools.partial``; no globals.
"""
