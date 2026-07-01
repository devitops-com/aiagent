"""Model registry: friendly aliases -> devai model strings.

Pure and dspy-free so it is trivially testable. The single source of truth for
composing devai's control-surface model string ``<model>[@<ctx>]::<reasoning>``
(prefixed with the ``openai/`` provider so DSPy/LiteLLM uses the OpenAI-compatible
path against ``api_base``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Reasoning = Literal["think", "nothink"]


class ModelSpec(BaseModel):
    """An immutable model entry: the backend model plus devai suffixes."""

    model_config = ConfigDict(frozen=True)

    model: str
    ctx: int | None = None
    reasoning: Reasoning | None = None
    provider: str = "openai"


# Placeholder aliases — confirm the real served tags with `aiagent models list`.
# In a devai container the model name is injected by the picker and overrides
# these; the registry only matters for standalone use.
DEFAULT_REGISTRY: dict[str, ModelSpec] = {
    "default": ModelSpec(model="qwen3.5:9b-q8_0", reasoning="nothink"),
}


def get_registry(
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, ModelSpec]:
    """``DEFAULT_REGISTRY`` merged with ``overrides`` (override wins)."""
    merged = dict(DEFAULT_REGISTRY)
    for alias, raw in (overrides or {}).items():
        merged[alias] = ModelSpec(**raw)
    return merged


def resolve(alias_or_model: str, registry: dict[str, ModelSpec]) -> ModelSpec:
    """Return the alias's spec, or treat the string as a raw model name.

    devai's picker passes raw model names (e.g. ``qwen3.5:9b-q8_0``); anything not
    found in the registry is taken as such, yielding ``ModelSpec(model=...)``.
    """
    if alias_or_model in registry:
        return registry[alias_or_model]
    return ModelSpec(model=alias_or_model)


def compose_model_string(
    spec: ModelSpec,
    default_reasoning: Reasoning,
    ctx_override: int | None = None,
) -> str:
    """Build ``<provider>/<model>[@<ctx>]::<reasoning>``.

    The single place the ``::nothink`` default materialises: a spec with
    ``reasoning=None`` inherits ``default_reasoning``.
    """
    ctx = ctx_override if ctx_override is not None else spec.ctx
    reasoning = spec.reasoning or default_reasoning
    suffix = f"@{ctx}" if ctx is not None else ""
    return f"{spec.provider}/{spec.model}{suffix}::{reasoning}"


def list_model_aliases(
    registry: dict[str, ModelSpec],
    default_reasoning: Reasoning,
) -> list[tuple[str, str]]:
    """``(alias, composed_model_string)`` pairs for display."""
    return [
        (alias, compose_model_string(spec, default_reasoning))
        for alias, spec in sorted(registry.items())
    ]
