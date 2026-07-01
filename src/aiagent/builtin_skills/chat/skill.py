"""Entry point for the built-in ``chat`` skill (self-contained; no core deps)."""

from __future__ import annotations

import dspy


class ChatSkill(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """Multi-turn chat: a single Predict over (history, question) -> answer."""

    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.Predict("history: dspy.History, question -> answer")

    def forward(self, history: object, question: str) -> dspy.Prediction:
        return self.respond(history=history, question=question)


def build() -> dspy.Module:
    """Return the chat pipeline."""
    return ChatSkill()
