"""Tests for the LM layer and CLI runtime helpers (construction only, no network)."""

from __future__ import annotations

import dspy

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


def test_prediction_to_dict_fallback() -> None:
    assert prediction_to_dict("plain") == {"result": "plain"}
