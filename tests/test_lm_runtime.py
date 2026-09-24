"""Tests for the LM layer and CLI runtime helpers (construction only, no network)."""

from __future__ import annotations

import dspy

from aiagent.cli._runtime import configure_lm, prediction_to_dict
from aiagent.config import Settings, load_settings
from aiagent.llm.lm import build_exact_lm, build_lm, configure_default, routing
from aiagent.llm.retry_lm import RetryAwareLM


def test_build_lm_composes_model_string() -> None:
    lm = build_lm("qwen3.5:9b-q8_0", settings=load_settings())
    assert lm.model == "openai/qwen3.5:9b-q8_0::nothink"


def test_build_lm_default_alias_when_none() -> None:
    lm = build_lm(settings=load_settings())
    assert lm.model.startswith("openai/qwen3.5:9b-q8_0")


def test_configure_default_sets_global_lm() -> None:
    lm = configure_default(load_settings())
    assert dspy.settings.lm is lm


def test_routing_scopes_lm() -> None:
    with routing("qwen3.5:9b-q8_0", settings=load_settings()) as lm:
        assert dspy.settings.lm is lm


def test_runtime_configure_lm_default() -> None:
    configure_lm(load_settings(), None)
    assert dspy.settings.lm is not None


def test_runtime_configure_lm_with_model() -> None:
    configure_lm(load_settings(), "llama3.2:3b")
    assert dspy.settings.lm.model == "openai/llama3.2:3b::nothink"


def test_prediction_to_dict_fallback() -> None:
    assert prediction_to_dict("plain") == {"result": "plain"}


def test_build_exact_lm_keeps_the_model_string_verbatim() -> None:
    exact = "openai/Qwen3.8-27B-MTP-devai-NVFP4::nothink@32768"
    lm = build_exact_lm(exact, settings=load_settings())
    assert isinstance(lm, RetryAwareLM)
    assert lm.model == exact


def test_build_exact_lm_skips_registry_resolution() -> None:
    # build_lm would resolve and compose this alias; the exact builder must not.
    lm = build_exact_lm("openai/qwen3.5:9b-q8_0", settings=load_settings())
    assert lm.model == "openai/qwen3.5:9b-q8_0"


def test_build_exact_lm_carries_the_settings() -> None:
    settings = Settings(
        api_base="http://router.test:1/v1",
        api_key="k",
        cache=False,
        request_timeout_s=12.5,
        num_retries=5,
    )
    lm = build_exact_lm("openai/m::think", settings=settings)
    assert lm.kwargs["api_base"] == "http://router.test:1/v1"
    assert lm.kwargs["api_key"] == "k"
    assert lm.kwargs["timeout"] == 12.5
    assert lm.cache is False
    assert lm.model_type == "chat"
    assert lm._max_retries == 5
    assert lm.num_retries == 0  # litellm's blind retry stays off
