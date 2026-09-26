"""Scoring rules: which metric, which rows, and how much is noise.

Roles:

* ``metrics`` — every metric a run can target.
* ``goal`` — the bar the loop tries to pass.
* ``splits`` — train, tune, and untouched holdout rows.
* ``intervals`` — how much of a score is noise.
* ``calibration`` — whether probabilities read as probabilities.
* ``ranking`` — what the ranking decides, what a cut gains.
"""
