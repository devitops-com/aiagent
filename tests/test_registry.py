"""Unit tests for the model registry (pure, no dspy)."""

from __future__ import annotations

from aiagent.llm.registry import (
    ModelSpec,
    compose_model_string,
    get_registry,
    list_model_aliases,
    resolve,
)


def test_compose_with_ctx_and_reasoning() -> None:
    spec = ModelSpec(model="qwen3.5:9b-q8_0", ctx=131072, reasoning="nothink")
    assert compose_model_string(spec, "think") == "openai/qwen3.5:9b-q8_0@131072::nothink"


def test_compose_inherits_default_reasoning() -> None:
    spec = ModelSpec(model="m")
    assert compose_model_string(spec, "nothink") == "openai/m::nothink"


def test_compose_ctx_override_wins() -> None:
    spec = ModelSpec(model="m", ctx=8192)
    assert compose_model_string(spec, "nothink", ctx_override=4096) == "openai/m@4096::nothink"


def test_resolve_known_alias() -> None:
    reg = get_registry()
    assert resolve("default", reg).model == "qwen3.5:9b-q8_0"


def test_resolve_raw_model_name() -> None:
    reg = get_registry()
    spec = resolve("llama3.2:3b", reg)
    assert spec.model == "llama3.2:3b" and spec.provider == "openai"


def test_overrides_merge() -> None:
    reg = get_registry({"fast": {"model": "x:1b", "reasoning": "nothink"}})
    assert "fast" in reg and reg["fast"].model == "x:1b"


def test_list_aliases_sorted() -> None:
    reg = get_registry({"zeta": {"model": "z"}})
    pairs = list_model_aliases(reg, "nothink")
    assert [a for a, _ in pairs] == sorted(a for a, _ in pairs)


def test_modelspec_is_frozen() -> None:
    spec = ModelSpec(model="m")
    try:
        spec.model = "other"  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - pydantic raises on frozen mutation
        return
    raise AssertionError("ModelSpec should be immutable")
