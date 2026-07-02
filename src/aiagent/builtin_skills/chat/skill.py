"""Entry point for the built-in ``chat`` skill (self-contained; no core deps).

Basic multi-turn Q&A: a single ``dspy.Predict`` that answers the user's latest
message using the prior conversation (``dspy.History``) for context. History is
optional, so the skill also works single-shot via ``aiagent run chat --text``.
The ``aiagent chat`` command persists the history so a session can be resumed.
"""

from __future__ import annotations

import dspy


class Chat(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Reply to the user's latest message, using the conversation history."""

    history: dspy.History = dspy.InputField(desc="prior turns of this conversation")
    text: str = dspy.InputField(desc="the user's latest message")
    answer: str = dspy.OutputField(desc="a helpful, direct reply")


class ChatSkill(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """Multi-turn chat: a Predict over (history, message) -> answer."""

    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.Predict(Chat)

    def forward(
        self, text: str, history: dspy.History | None = None
    ) -> dspy.Prediction:
        hist = history if history is not None else dspy.History(messages=[])
        return self.respond(history=hist, text=text)


def build() -> dspy.Module:
    """Return the chat pipeline."""
    return ChatSkill()
