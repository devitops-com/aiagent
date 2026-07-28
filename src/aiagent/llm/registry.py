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

from pydantic import BaseModel, ConfigDict, ValidationError

from aiagent.exceptions import AiagentConfigError

Reasoning = Literal["think", "nothink"]

# A trailing ``@<int>`` on a model name is the gateway's context-window suffix.
_BAKED_CTX = re.compile(r"@(\d+)$")


class ModelSpec(BaseModel):
    """An immutable model entry: the backend model plus devai suffixes.

    ``api_base`` / ``api_key`` are optional per-model endpoint overrides: an
    alias that sets them is served by that endpoint, one that leaves them unset
    falls back to the global ``settings.api_base`` / ``settings.api_key``. That
    is what lets a single session address several backends at once — e.g. one
    alias on the router's ollama port and another on its vLLM port (issue #11).

    ``extra="forbid"`` because a mistyped or unsupported key used to be dropped
    without a word, composing the alias as though it had never been written.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    ctx: int | None = None
    reasoning: Reasoning | None = None
    provider: str = "openai"
    api_base: str | None = None
    api_key: str | None = None


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

    A malformed override raises :class:`AiagentConfigError` naming the alias
    and the keys a spec accepts.
    """
    merged = dict(DEFAULT_REGISTRY)
    for alias, raw in (overrides or {}).items():
        merged[alias] = _build_spec(alias, raw)
    if default_model:
        merged["default"] = ModelSpec(model=default_model)
    return merged


def _build_spec(alias: str, raw: dict[str, Any]) -> ModelSpec:
    """Validate one ``[registry_overrides.<alias>]`` table into a ``ModelSpec``."""
    try:
        return ModelSpec(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<value>'}: {err['msg']}"
            for err in exc.errors()
        )
        accepted = ", ".join(sorted(ModelSpec.model_fields))
        raise AiagentConfigError(
            f"invalid [registry_overrides.{alias}]: {problems} "
            f"(accepted keys: {accepted})"
        ) from exc


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

    Context precedence is ``spec.ctx`` > ``ctx_override`` > a ``@<ctx>`` baked
    into ``spec.model``. An alias that declares its own ``ctx`` wins over the
    global ``context_tokens`` because the global is a default for aliases that
    state no preference, and endpoint-specific aliases legitimately top out at
    different context windows (issue #11).

    A ``@<ctx>`` already baked into ``spec.model`` (a gateway that hands aiagent
    a model name with the context suffix pre-attached) is peeled off and
    re-emitted last, so this path matches the ``AIAGENT_CONTEXT`` path instead of
    producing ``<model>@<ctx>::<reasoning>`` (issue #6). An explicit ctx
    (``spec.ctx`` or ``ctx_override``) takes precedence over the baked one and is
    never duplicated.
    """
    model, baked_ctx = _split_baked_ctx(spec.model)
    ctx = spec.ctx if spec.ctx is not None else ctx_override
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
    ctx_override: int | None = None,
) -> list[tuple[str, str]]:
    """``(alias, composed_model_string)`` pairs for display.

    ``ctx_override`` is the configured ``context_tokens``. Passing it keeps the
    displayed string identical to the one the request actually carries; without
    it the listing silently dropped the ``@<ctx>`` suffix (issue #11).
    """
    return [
        (alias, compose_model_string(spec, default_reasoning, ctx_override))
        for alias, spec in sorted(registry.items())
    ]
