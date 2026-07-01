"""Runtime helpers shared by the dspy-touching CLI commands.

Import-safe at module top (no ``dspy``); the heavy imports happen only when
``configure_lm`` is called, preserving the lazy-dspy guarantee for ``--help``.
"""

from __future__ import annotations

from typing import Any

from aiagent.config import Settings


def configure_lm(settings: Settings, model: str | None) -> None:
    """Set the global DSPy LM, optionally overriding the model alias/name."""
    import dspy

    from aiagent.llm.lm import build_lm, configure_default

    if model:
        dspy.configure(lm=build_lm(model, settings=settings))
    else:
        configure_default(settings)


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Best-effort conversion of a ``dspy.Prediction`` to a plain dict."""
    to_dict = getattr(prediction, "toDict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {"result": str(prediction)}
