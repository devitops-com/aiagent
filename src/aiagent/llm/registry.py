"""Model registry: friendly aliases -> devai model strings.

Pure and dspy-free so it is trivially testable. The single source of truth for
composing devai's control-surface model string ``<model>::<reasoning>[@<ctx>]``
(prefixed with the ``openai/`` provider so DSPy/LiteLLM uses the OpenAI-compatible
path against ``api_base``). The router's parse is right-to-left with ``@<ctx>``
outermost, so ``@<ctx>`` must be the final token or it survives into the model
name and Ollama rejects it (issue #3).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Reasoning = Literal["think", "nothink"]

# A trailing ``@<int>`` on a model name is the gateway's context-window suffix.
_BAKED_CTX = re.compile(r"@(\d+)$")


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
    default_model: str | None = None,
) -> dict[str, ModelSpec]:
    """``DEFAULT_REGISTRY`` merged with ``overrides`` (override wins).

    When ``default_model`` is non-empty (the configured ``AIAGENT_MODEL``), the
    ``default`` alias resolves to it instead of the baked placeholder, so the
    configured model stays the single source of truth for skills and callers
    that route through the ``default`` alias (issue #4). Its ``reasoning`` is
    left unset so it inherits ``default_reasoning`` at compose time.
    """
    merged = dict(DEFAULT_REGISTRY)
    for alias, raw in (overrides or {}).items():
        merged[alias] = ModelSpec(**raw)
    if default_model:
        merged["default"] = ModelSpec(model=default_model)
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
    """Build ``<provider>/<model>::<reasoning>[@<ctx>]``.

    The single place the ``::nothink`` default materialises: a spec with
    ``reasoning=None`` inherits ``default_reasoning``. ``@<ctx>`` is appended
    **after** ``::<reasoning>`` so it is the outermost (last) token, which is
    what the devai router's right-to-left ctx parser expects (issue #3).

    A ``@<ctx>`` already baked into ``spec.model`` (a gateway that hands aiagent
    a model name with the context suffix pre-attached) is peeled off and
    re-emitted last, so this path matches the ``AIAGENT_CONTEXT`` path instead of
    producing ``<model>@<ctx>::<reasoning>`` (issue #6). An explicit ctx
    (``ctx_override`` or ``spec.ctx``) takes precedence over the baked one and is
    never duplicated.
    """
    model, baked_ctx = _split_baked_ctx(spec.model)
    ctx = ctx_override if ctx_override is not None else spec.ctx
    if ctx is None:
        ctx = baked_ctx
    reasoning = spec.reasoning or default_reasoning
    ctx_suffix = f"@{ctx}" if ctx is not None else ""
    return f"{spec.provider}/{model}::{reasoning}{ctx_suffix}"


def _split_baked_ctx(model: str) -> tuple[str, int | None]:
    """Split a trailing ``@<int>`` context suffix off a model name.

    Returns ``(bare_model, ctx)`` — ``ctx`` is ``None`` when the name has no
    numeric ``@`` suffix, leaving names like ``org/model@latest`` untouched.
    """
    match = _BAKED_CTX.search(model)
    if match is None:
        return model, None
    return model[: match.start()], int(match.group(1))


def list_model_aliases(
    registry: dict[str, ModelSpec],
    default_reasoning: Reasoning,
) -> list[tuple[str, str]]:
    """``(alias, composed_model_string)`` pairs for display."""
    return [
        (alias, compose_model_string(spec, default_reasoning))
        for alias, spec in sorted(registry.items())
    ]
