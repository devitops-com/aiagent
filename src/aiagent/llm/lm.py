"""Construct and route ``dspy.LM`` instances from settings + the registry.

This is the only module in ``llm/`` that imports ``dspy``; it is imported lazily
by the run/optimize/eval/chat paths, never by ``--help``/``doctor``/``config``.
No adapter is configured anywhere, so DSPy uses its default ChatAdapter (portable
across devai's ollama/vllm/sglang backends; no native tool-calling or server JSON
mode is assumed).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import dspy

from aiagent.config import Settings
from aiagent.llm.registry import compose_model_string, get_registry, resolve


def _model_string(settings: Settings, alias_or_model: str | None) -> str:
    """Resolve the effective model string for ``alias_or_model``.

    ``None`` -> ``settings.model`` (picker/env injected) if set, else the
    registry's ``default_alias``.
    """
    target = alias_or_model or settings.model or settings.default_alias
    registry = get_registry(settings.registry_overrides, settings.model)
    spec = resolve(target, registry)
    return compose_model_string(
        spec, settings.default_reasoning, settings.context_tokens
    )


def build_lm(
    alias_or_model: str | None = None,
    *,
    settings: Settings,
    **overrides: object,
) -> dspy.LM:
    """Build a ``dspy.LM`` for ``alias_or_model`` (or the configured default).

    The generous ``timeout`` (default 900s) absorbs devai's on-demand backend
    cold starts; combined with ``num_retries`` the worst-case wait is
    ``(num_retries + 1) * timeout``.
    """
    return dspy.LM(
        _model_string(settings, alias_or_model),
        api_base=settings.api_base,
        api_key=settings.api_key,
        model_type="chat",
        num_retries=settings.num_retries,
        cache=settings.cache,
        timeout=settings.request_timeout_s,
        **overrides,
    )


def configure_default(settings: Settings) -> dspy.LM:
    """Build the default LM and set it as the process-wide DSPy default."""
    lm = build_lm(settings=settings)
    dspy.configure(lm=lm)
    return lm


@contextmanager
def routing(
    alias_or_model: str | None = None,
    *,
    settings: Settings,
    lm: dspy.LM | None = None,
    **overrides: object,
) -> Iterator[dspy.LM]:
    """Per-step model routing: run a block under a scoped ``dspy.context(lm=...)``."""
    active = lm if lm is not None else build_lm(
        alias_or_model, settings=settings, **overrides
    )
    with dspy.context(lm=active):
        yield active
