"""The single place where LLM calls happen. Nodes never touch the SDK directly.

Everything lives in ``client``; import from there. This file re-exported four of its names for
a while and nothing ever used the facade — every node and every test reaches for
``automl_agent.llm.client`` directly.
"""
