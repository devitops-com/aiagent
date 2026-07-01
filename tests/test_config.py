"""Unit tests for settings resolution precedence."""

from __future__ import annotations

import pytest

from aiagent.config import DEFAULT_API_BASE, load_settings
from aiagent.exceptions import AiagentConfigError


def test_defaults_when_nothing_set() -> None:
    s = load_settings()
    assert s.api_base == DEFAULT_API_BASE
    assert s.api_key == "local"
    assert s.default_reasoning == "nothink"
    assert s.model == ""


def test_aiagent_env_sets_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_API_BASE", "http://x:9/v1")
    assert load_settings().api_base == "http://x:9/v1"


def test_ollama_host_fallback_appends_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://devai-router:11434")
    assert load_settings().api_base == "http://devai-router:11434/v1"


def test_openai_base_url_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw:8000/v1")
    assert load_settings().api_base == "http://gw:8000/v1"


def test_aiagent_env_beats_devai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_API_BASE", "http://win/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://lose/v1")
    assert load_settings().api_base == "http://win/v1"


def test_api_key_from_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert load_settings().api_key == "sk-test"


def test_model_from_devai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "qwen3.5:9b-q8_0")
    assert load_settings().model == "qwen3.5:9b-q8_0"


def test_openai_model_beats_ollama_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "a")
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "b")
    assert load_settings().model == "a"


def test_context_tokens_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT", "131072")
    assert load_settings().context_tokens == 131072


def test_empty_api_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_API_KEY", "")
    with pytest.raises(AiagentConfigError):
        load_settings()


def test_health_and_models_urls() -> None:
    s = load_settings()
    assert s.health_url() == "http://devai-router:11434/health"
    assert s.models_url() == "http://devai-router:11434/v1/models"


def test_redacted_masks_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    assert load_settings().redacted()["api_key"] == "***"
