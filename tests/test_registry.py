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
    # @<ctx> must be the outermost (last) token, after ::<reasoning> (issue #3).
    spec = ModelSpec(model="qwen3.5:9b-q8_0", ctx=131072, reasoning="nothink")
    assert compose_model_string(spec, "think") == "openai/qwen3.5:9b-q8_0::nothink@131072"


def test_compose_inherits_default_reasoning() -> None:
    spec = ModelSpec(model="m")
    assert compose_model_string(spec, "nothink") == "openai/m::nothink"


def test_compose_ctx_override_wins() -> None:
    spec = ModelSpec(model="m", ctx=8192)
    assert compose_model_string(spec, "nothink", ctx_override=4096) == "openai/m::nothink@4096"


def test_compose_ctx_baked_into_model_name() -> None:
    # A trailing @<ctx> already on the model name must end up last, with the
    # reasoning suffix inserted *before* it (issue #6), matching the ordering
    # the AIAGENT_CONTEXT path produces (issue #3).
    spec = ModelSpec(model="SomeModel@131072")
    assert compose_model_string(spec, "nothink") == "openai/SomeModel::nothink@131072"


def test_compose_ctx_baked_with_reasoning_on_spec() -> None:
    spec = ModelSpec(model="SomeModel@131072", reasoning="think")
    assert compose_model_string(spec, "nothink") == "openai/SomeModel::think@131072"


def test_compose_ctx_override_beats_baked_model_ctx() -> None:
    # An explicit ctx (spec.ctx or ctx_override) wins over the baked @<ctx>,
    # and the baked one is not duplicated.
    spec = ModelSpec(model="SomeModel@131072", ctx=8192)
    assert compose_model_string(spec, "nothink") == "openai/SomeModel::nothink@8192"
    assert (
        compose_model_string(spec, "nothink", ctx_override=4096)
        == "openai/SomeModel::nothink@4096"
    )


def test_compose_non_numeric_at_suffix_is_left_untouched() -> None:
    # Only a trailing @<int> is treated as ctx; anything else stays in the name.
    spec = ModelSpec(model="org/model@latest")
    assert compose_model_string(spec, "nothink") == "openai/org/model@latest::nothink"


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


def test_default_alias_tracks_configured_model() -> None:
    # AIAGENT_MODEL (default_model) makes the `default` alias the single source
    # of truth instead of the baked placeholder (issue #4).
    reg = get_registry(default_model="qwen3.6:27b-q4_K_M")
    assert reg["default"].model == "qwen3.6:27b-q4_K_M"
    assert reg["default"].reasoning is None  # inherits default_reasoning at compose
    composed = compose_model_string(reg["default"], "nothink")
    assert composed == "openai/qwen3.6:27b-q4_K_M::nothink"


def test_default_alias_falls_back_to_placeholder() -> None:
    # No configured model -> the shipped placeholder alias is retained.
    assert get_registry(default_model="")["default"].model == "qwen3.5:9b-q8_0"
    assert get_registry()["default"].model == "qwen3.5:9b-q8_0"


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
