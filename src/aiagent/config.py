"""Configuration for aiagent.

Settings resolve with this precedence (highest first):

1. ``AIAGENT_*`` environment variables
2. a TOML file at ``~/.config/aiagent/config.toml``
3. **devai-injected** environment (``OPENAI_BASE_URL`` / ``OLLAMA_HOST`` /
   ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` / ``OLLAMA_DEFAULT_MODEL`` / ``CONTEXT``)
4. code defaults

Layer 3 is what lets aiagent run as a devai agent that "knows nothing about
devai except the router URL it is handed via env" — the picker exports a handful
of env vars and aiagent adapts with no devai-specific code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from aiagent.exceptions import AiagentConfigError

DEFAULT_API_BASE = "http://devai-router:11434/v1"
DEFAULT_API_KEY = "local"  # devai single-mode has no auth; non-empty for LiteLLM
DEFAULT_MODEL = "qwen3.5:9b-q8_0"  # placeholder — confirm with `aiagent models list`
USER_CONFIG_PATH = Path.home() / ".config" / "aiagent" / "config.toml"
USER_SKILLS_DIR = Path.home() / ".config" / "aiagent" / "skills"

Reasoning = Literal["think", "nothink"]


def _ollama_host_to_base(host: str | None) -> str | None:
    """Turn a bare ``OLLAMA_HOST`` (no ``/v1``) into an OpenAI-compatible base."""
    if not host:
        return None
    return host.rstrip("/") + "/v1"


class _DevaiEnvSource(PydanticBaseSettingsSource):
    """Lowest-but-one settings source: devai's generic injected env vars.

    Sits below ``AIAGENT_*`` env and the TOML file but above code defaults, so a
    user override always wins yet a bare devai container still works.
    """

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:  # pragma: no cover - not used (we override __call__)
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        base = os.environ.get("OPENAI_BASE_URL") or _ollama_host_to_base(
            os.environ.get("OLLAMA_HOST")
        )
        if base:
            out["api_base"] = base
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            out["api_key"] = key
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_DEFAULT_MODEL")
        if model:
            out["model"] = model
        ctx = os.environ.get("AIAGENT_CONTEXT") or os.environ.get("CONTEXT")
        if ctx and ctx.isdigit():
            out["context_tokens"] = int(ctx)
        return out


class Settings(BaseSettings):
    """Resolved aiagent settings (immutable)."""

    model_config = SettingsConfigDict(
        env_prefix="AIAGENT_",
        toml_file=USER_CONFIG_PATH,
        extra="ignore",
        frozen=True,
    )

    # Connection
    api_base: str = DEFAULT_API_BASE
    api_key: str = DEFAULT_API_KEY
    request_timeout_s: float = 900.0  # generous: vLLM/SGLang cold start can be slow
    num_retries: int = 2
    cache: bool = True

    # Model selection
    model: str = ""  # effective model name; empty -> use default_alias from registry
    default_alias: str = "default"
    default_reasoning: Reasoning = "nothink"  # clean typed-field parsing in pipelines
    context_tokens: int | None = None  # maps to the @<ctx> model-string suffix
    registry_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Execution
    num_threads: int = 4

    # Optimizer defaults
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    max_rounds: int = 1

    # Skills
    skills_dir: Path = USER_SKILLS_DIR

    @field_validator("api_key")
    @classmethod
    def _api_key_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("api_key must be non-empty (LiteLLM rejects an empty key)")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            _DevaiEnvSource(settings_cls),
        )

    def health_url(self) -> str:
        """Router health endpoint (``/health`` lives at the root, not under /v1)."""
        base = self.api_base.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        return root.rstrip("/") + "/health"

    def models_url(self) -> str:
        """OpenAI-compatible models listing endpoint."""
        return self.api_base.rstrip("/") + "/models"

    def redacted(self) -> dict[str, Any]:
        """A dict of settings with the api_key masked, for display."""
        data = self.model_dump(mode="json")
        data["api_key"] = "***" if self.api_key else ""
        return data


def load_settings() -> Settings:
    """Construct :class:`Settings`, wrapping validation failures."""
    try:
        return Settings()
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed config error
        raise AiagentConfigError(f"invalid configuration: {exc}") from exc
