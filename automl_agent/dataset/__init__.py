"""How the table's columns are read: what the executor can use, and what each value means.

``features``   which columns are usable as features, and how the rest become numbers.
``targets``    target-column encoding, the task it implies, and missing labels.
``sentinels``  values that mean "missing" but arrive looking like measurements (``-9999``).
``caveats``    the card's channel for things about the data the aggregates do not show.

**No data rows live in this package** — despite the name, these are rules and vocabularies, not
storage. Rows only ever exist inside a subprocess (``scripts/``); the boundary that keeps them
there is ``privacy``, one level up.
"""
