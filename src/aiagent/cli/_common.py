"""Shared CLI helpers: context settings, output format, and printers.

Kept import-light on purpose — nothing here imports ``dspy`` or the heavy
runtime layers, so ``--help`` / ``config show`` / ``skills list`` stay fast.
"""

from __future__ import annotations

import enum
import json
from typing import Any

import click

from aiagent.config import Settings, load_settings

# ``-h`` as well as ``--help`` for every command/group.
CLI_CONTEXT_SETTINGS: dict[str, Any] = {"help_option_names": ["-h", "--help"]}


def get_settings() -> Settings:
    """Load resolved settings (env > TOML > devai-env > defaults)."""
    return load_settings()


class OutputFormat(enum.StrEnum):
    """How a command renders its result."""

    TABLE = "table"
    JSON = "json"


def print_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON on stdout."""
    click.echo(json.dumps(obj, indent=2, default=str, sort_keys=True))


def echo_err(message: str) -> None:
    """Print an error ``message`` to stderr."""
    click.echo(message, err=True)
