"""Configuration for aiagent.

Settings resolve with this precedence (highest first):

1. ``AIAGENT_*`` environment variables
2. a TOML file at ``~/.config/aiagent/config.toml``
3. **devai-injected** environment (``OPENAI_BASE_URL`` / ``OLLAMA_HOST`` /
   ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` / ``OLLAMA_DEFAULT_MODEL`` / ``CONTEXT`` /
   ``HTTPS_PROXY`` / ``HTTP_PROXY``)
4. code defaults

Layer 3 is what lets aiagent run as a devai agent that "knows nothing about
devai except the router URL it is handed via env" — the picker exports a handful
of env vars and aiagent adapts with no devai-specific code.
"""

from __future__ import annotations

import os
import re
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
# devai's fail-closed egress proxy (pipelock): the lab's ONLY route to the
# internet. Outbound URL fetches (e.g. the `sentiment` skill) go through it; TLS
# is MITM-inspected but transparently trusted because the lab image bakes the
# pipelock CA into the system store and sets SSL_CERT_FILE, which httpx honors.
DEFAULT_PROXY_URL = "http://devai-pipelock:8888"
DEFAULT_MODEL = "qwen3.5:9b-q8_0"  # placeholder — confirm with `aiagent models list`
USER_CONFIG_PATH = Path.home() / ".config" / "aiagent" / "config.toml"
USER_SKILLS_DIR = Path.home() / ".config" / "aiagent" / "skills"
USER_SESSIONS_DIR = Path.home() / ".config" / "aiagent" / "chat-sessions"
# Installed System 1 students (~1.3 GB each), eval reports, shadow logs: not ~/.config.
USER_ARTIFACTS_DIR = Path.home() / ".local" / "share" / "aiagent" / "artifacts"
# devai's laya volume (host /var/cache/devai/laya), mounted at /laya in the lab:
# base/ (read), inbox/ (write), datasets/ and runs/ (read).
DEFAULT_DISTILL_DIR = Path("/laya")
DEFAULT_TRAINER_API_BASE = "http://devai-router:11438/v1"  # OpenAI fine-tuning jobs API

Reasoning = Literal["think", "nothink"]
System1Mode = Literal["off", "shadow", "gate"]

_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")  # SkillManifest.name


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
        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )
        if proxy:
            out["proxy_url"] = proxy
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
    num_retries: int = 2  # retries for *transient* errors only (see RetryAwareLM)
    cache: bool = True
    # Forward proxy for outbound URL fetches (pipelock). Empty string disables
    # proxying (direct connection), e.g. when running outside devai-net.
    proxy_url: str = DEFAULT_PROXY_URL

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

    # Chat
    sessions_dir: Path = USER_SESSIONS_DIR

    # System 1 (distilled students; aiagent.system1 / aiagent.distill)
    distill_dir: Path = DEFAULT_DISTILL_DIR
    trainer_api_base: str = DEFAULT_TRAINER_API_BASE
    artifacts_dir: Path = USER_ARTIFACTS_DIR
    # Per-skill cascade mode; a skill not listed is off.
    system1_mode: dict[str, System1Mode] = Field(default_factory=dict)
    # Overrides every installed per-question threshold when set.
    system1_min_conf: float | None = Field(default=None, gt=0, le=1)

    @field_validator("api_key")
    @classmethod
    def _api_key_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("api_key must be non-empty (LiteLLM rejects an empty key)")
        return v

    @field_validator("system1_mode")
    @classmethod
    def _system1_mode_skill_names(
        cls, v: dict[str, System1Mode]
    ) -> dict[str, System1Mode]:
        bad = sorted(k for k in v if not _SKILL_NAME.match(k))
        if bad:
            raise ValueError(
                f"system1_mode keys must be skill names ({_SKILL_NAME.pattern}): {bad}"
            )
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
