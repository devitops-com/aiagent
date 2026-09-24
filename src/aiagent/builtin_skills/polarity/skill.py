"""Entry point for the built-in ``polarity`` skill (self-contained; no core deps).

The System 1 distillation pilot: one ``dspy.Predict`` (``classify``) with a
single ``str`` input and a single ``Literal`` output, the shape a distilled
student can answer. The skill is run-only (no trainset/devset/metric).
"""

from __future__ import annotations

from typing import Literal

import dspy

from aiagent.core.pipeline import Pipeline


class Polarity(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Classify the overall sentiment polarity of a passage."""

    text: str = dspy.InputField(desc="A passage of text to assess.")
    polarity: Literal["negative", "neutral", "mixed", "positive"] = dspy.OutputField(
        desc="Overall sentiment polarity of the passage."
    )


class PolarityModule(Pipeline):
    """One dspy.Predict: the qualifying predictor ``classify``."""

    default_alias = "default"

    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict(Polarity)

    def forward(self, text: str) -> dspy.Prediction:
        return self.classify(text=text)


def build() -> dspy.Module:
    """Return the polarity classifier."""
    return PolarityModule()
