"""Shared test fixtures.

``clean_env`` (autouse) makes settings resolution hermetic: it clears every env
var aiagent reads and points the TOML source at a non-existent file, so tests
never pick up the developer's real shell env or ``~/.config/aiagent/config.toml``.
"""

from __future__ import annotations

import pytest

from aiagent.config import Settings

_VARS = [
    "AIAGENT_API_BASE",
    "AIAGENT_API_KEY",
    "AIAGENT_MODEL",
    "AIAGENT_CONTEXT",
    "AIAGENT_DEFAULT_ALIAS",
    "AIAGENT_NUM_THREADS",
    "AIAGENT_SKILLS_DIR",
    "AIAGENT_SESSIONS_DIR",
    "AIAGENT_PROXY_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_DEFAULT_MODEL",
    "CONTEXT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "toml_file", str(tmp_path / "absent.toml"))
    # Default to an empty user skills dir so discovery sees only built-ins unless
    # a test overrides AIAGENT_SKILLS_DIR.
    monkeypatch.setenv("AIAGENT_SKILLS_DIR", str(tmp_path / "no_user_skills"))
    # Isolate chat sessions to a temp dir so `aiagent chat` never touches ~/.config.
    monkeypatch.setenv("AIAGENT_SESSIONS_DIR", str(tmp_path / "chat_sessions"))
