"""Entry points for the built-in ``extract`` skill.

Loaded dynamically by the skills engine: ``build()`` returns the DSPy module and
``metric`` is the scoring function referenced by ``metric: skill:metric`` in
SKILL.md. Datasets are plain JSONL files alongside this module.
"""

from __future__ import annotations

import dspy

from aiagent.core.extract import ExtractExpenseModule
from aiagent.metrics.extraction import field_accuracy


def build() -> dspy.Module:
    """Return the expense-extraction pipeline."""
    return ExtractExpenseModule()


# Referenced by `metric: skill:metric` in SKILL.md.
metric = field_accuracy
