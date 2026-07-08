"""Entry point for the built-in ``sentiment`` skill.

Thin shim (mirrors ``extract``): the signatures, module, and statistics live in
``aiagent.core.sentiment`` / ``aiagent.core.sentiment_stats``. ``build()``
returns the pipeline; the skill is run-only (no trainset/devset/metric), so it
works with nothing but the data sources the user supplies.
"""

from __future__ import annotations

import dspy

from aiagent.core.sentiment import SentimentModule


def build() -> dspy.Module:
    """Return the sentiment-analysis pipeline."""
    return SentimentModule()
