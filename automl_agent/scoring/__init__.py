"""How a run is measured: which metric, on which rows, and how much of the score is noise.

``metrics``      the registry — every metric a run can target, and its properties.
``goal``         the bar the loop is trying to clear, derived or fixed.
``splits``       which rows train, which tune, and which nobody touches until the end.
``intervals``    how much of a score is the model and how much is the rows it was measured on.
``calibration``  whether a predicted probability is worth reading as a probability.
``ranking``      what the ranking alone decides, versus what the operating point can still buy.

These are the deterministic measurement rules, so they stay free of the graph: nothing here
imports a node, a prompt, or ``state``. ``metrics`` in particular is free of sklearn too, which
is what lets the orchestrator process import it without paying for the estimators — see the
module docstring there before adding an import to it.
"""
