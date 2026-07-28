"""Tests for the LM layer and CLI runtime helpers (construction only, no network)."""

from __future__ import annotations

import json

import dspy
import pytest

from aiagent.cli._runtime import configure_lm, prediction_to_dict
from aiagent.config import load_settings
from aiagent.llm.lm import build_lm, configure_default, routing


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


def test_build_lm_uses_per_alias_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # An alias carrying its own api_base/api_key is called at that endpoint,
    # so one session can reach several backends (issue #11).
    monkeypatch.setenv(
        "AIAGENT_REGISTRY_OVERRIDES",
        json.dumps(
            {
                "vllm": {
                    "model": "Qwen3.5-9B-NVFP4",
                    "api_base": "http://devai-router:11435/v1",
                    "api_key": "alias-key",
                }
            }
        ),
    )
    lm = build_lm("vllm", settings=load_settings())
    assert lm.model == "openai/Qwen3.5-9B-NVFP4::nothink"
    assert lm.kwargs["api_base"] == "http://devai-router:11435/v1"
    assert lm.kwargs["api_key"] == "alias-key"


def test_build_lm_falls_back_to_global_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_API_BASE", "http://devai-router:11434/v1")
    monkeypatch.setenv(
        "AIAGENT_REGISTRY_OVERRIDES", json.dumps({"plain": {"model": "x:1b"}})
    )
    lm = build_lm("plain", settings=load_settings())
    assert lm.kwargs["api_base"] == "http://devai-router:11434/v1"
    assert lm.kwargs["api_key"] == "local"


def test_prediction_to_dict_fallback() -> None:
    assert prediction_to_dict("plain") == {"result": "plain"}
