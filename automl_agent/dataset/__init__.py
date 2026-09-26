"""How the table's columns are read and what each value means.

Roles:

* ``features`` — which columns are features, and how they become numbers.
* ``targets`` — target encoding, the implied task, and missing labels.
* ``sentinels`` — "missing" codes that look like measurements (``-9999``).
* ``caveats`` — facts about the data the aggregates do not show.
* ``pipeline`` — ordered feature steps, declared in JSON, from a whitelist.
"""
