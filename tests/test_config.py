"""Unit tests for settings resolution precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiagent.config import (
    DEFAULT_API_BASE,
    DEFAULT_DISTILL_DIR,
    DEFAULT_TRAINER_API_BASE,
    USER_ARTIFACTS_DIR,
    Settings,
    load_settings,
)
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


# --------------------------------------------------------------------------- System 1 settings


def test_system1_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIAGENT_DISTILL_DIR")
    monkeypatch.delenv("AIAGENT_ARTIFACTS_DIR")
    s = load_settings()
    assert s.distill_dir == DEFAULT_DISTILL_DIR == Path("/laya")
    assert s.trainer_api_base == DEFAULT_TRAINER_API_BASE == "http://devai-router:11438/v1"
    assert s.artifacts_dir == USER_ARTIFACTS_DIR
    assert USER_ARTIFACTS_DIR == Path.home() / ".local" / "share" / "aiagent" / "artifacts"
    assert s.system1_mode == {}
    assert s.system1_min_conf is None


def test_clean_env_points_the_system1_dirs_at_tmp_path(tmp_path: Path) -> None:
    s = load_settings()
    assert s.distill_dir == tmp_path / "distill"
    assert s.artifacts_dir == tmp_path / "artifacts"


def test_system1_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_TRAINER_API_BASE", "http://trainer:1/v1")
    monkeypatch.setenv("AIAGENT_SYSTEM1_MODE", '{"polarity": "shadow", "other-skill_2": "gate"}')
    monkeypatch.setenv("AIAGENT_SYSTEM1_MIN_CONF", "0.9")
    s = load_settings()
    assert s.trainer_api_base == "http://trainer:1/v1"
    assert s.system1_mode == {"polarity": "shadow", "other-skill_2": "gate"}
    assert s.system1_min_conf == 0.9


@pytest.mark.parametrize(
    "mode",
    [
        '{"polarity": "on"}',  # not off/shadow/gate
        '{"Polarity": "gate"}',  # upper case: not a skill name
        '{"2polarity": "gate"}',  # must start with a letter
        '{"a/b": "gate"}',
        '{"": "off"}',
        '["polarity"]',
    ],
)
def test_invalid_system1_mode_rejected(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("AIAGENT_SYSTEM1_MODE", mode)
    with pytest.raises(AiagentConfigError):
        load_settings()


@pytest.mark.parametrize("value", ["0", "1.5", "-0.1"])
def test_system1_min_conf_out_of_range_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AIAGENT_SYSTEM1_MIN_CONF", value)
    with pytest.raises(AiagentConfigError):
        load_settings()


def test_system1_min_conf_one_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_SYSTEM1_MIN_CONF", "1")
    assert load_settings().system1_min_conf == 1.0


def test_system1_mode_from_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    toml = Path(str(Settings.model_config["toml_file"]))
    toml.write_text('[system1_mode]\npolarity = "gate"\n', encoding="utf-8")
    assert load_settings().system1_mode == {"polarity": "gate"}
    monkeypatch.setenv("AIAGENT_SYSTEM1_MODE", '{"polarity": "shadow"}')
    assert load_settings().system1_mode == {"polarity": "shadow"}  # env beats TOML
