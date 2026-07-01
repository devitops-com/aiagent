"""The expense-extraction signature and module (the MVP demo pipeline).

Extracts ``{merchant, date, amount}`` from a free-text expense note. The typed
``amount: float`` output is coerced by DSPy's ChatAdapter; the field descriptions
steer the model toward the metric's normalization (ISO date, bare number).
"""

from __future__ import annotations

import dspy

from aiagent.core.pipeline import Pipeline


class ExtractExpense(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Extract structured expense fields from a free-text note or receipt line."""

    text: str = dspy.InputField(
        desc="Free-text expense note, e.g. a receipt line or memo."
    )
    merchant: str = dspy.OutputField(
        desc="Merchant / vendor name only, no surrounding words."
    )
    date: str = dspy.OutputField(
        desc="Transaction date in ISO 8601 format (YYYY-MM-DD)."
    )
    amount: float = dspy.OutputField(
        desc="Total amount as a plain number, no currency symbol or separators."
    )


class ExtractExpenseModule(Pipeline):
    """ChainOfThought expense extractor.

    CoT adds a prompt-level ``reasoning`` output field; the model still runs under
    the configured ``::nothink`` policy, so no separate ``reasoning_content``
    channel pollutes typed-field parsing — the two coexist.
    """

    default_alias = "default"

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractExpense)

    def forward(self, text: str) -> dspy.Prediction:
        return self.extract(text=text)
