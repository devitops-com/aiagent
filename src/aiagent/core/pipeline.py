"""Base type for aiagent pipelines.

A *pipeline* is just a ``dspy.Module`` — DSPy already gives composition,
``forward``/``__call__``, parameter tracking, and ``save``/``load``. This thin
base adds only a ``default_alias`` convention (the model a skill prefers when the
caller doesn't pin one) so we don't re-invent any of that.
"""

from __future__ import annotations

import dspy


class Pipeline(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """Marker base for aiagent pipelines.

    Subclasses build sub-modules in ``__init__`` and implement ``forward``.
    ``default_alias`` is advisory metadata read by the CLI when routing.
    """

    default_alias: str = "default"
