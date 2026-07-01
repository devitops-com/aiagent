"""Skill manifest schema, the resolved Skill record, and frontmatter parsing."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aiagent.exceptions import SkillManifestError

_FRONTMATTER_FENCE = "---"


class SkillSource(enum.StrEnum):
    """Where a skill was discovered."""

    BUILTIN = "builtin"
    USER = "user"


class SkillManifest(BaseModel):
    """Validated ``SKILL.md`` frontmatter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str
    entrypoint: str = "skill:build"  # "<module>:<callable>"
    model: str | None = None  # alias into the model registry
    trainset: str | None = None  # JSONL path relative to the skill dir
    devset: str | None = None  # JSONL path relative to the skill dir
    metric: str | None = None  # "skill:<attr>" or "<module>:<attr>"
    version: str | None = None


@dataclass(frozen=True)
class Skill:
    """A discovered, validated skill."""

    manifest: SkillManifest
    source: SkillSource
    directory: Path

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def entrypoint_module(self) -> str:
        return self.manifest.entrypoint.split(":", 1)[0]

    @property
    def entrypoint_attr(self) -> str:
        parts = self.manifest.entrypoint.split(":", 1)
        return parts[1] if len(parts) == 2 else "build"


def parse_skill_md(text: str) -> SkillManifest:
    """Parse YAML frontmatter from a ``SKILL.md`` body into a manifest."""
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_FENCE):
        raise SkillManifestError("SKILL.md is missing a '---' YAML frontmatter block")
    # ['', '<yaml>', '<body>']
    parts = stripped.split(_FRONTMATTER_FENCE, 2)
    if len(parts) < 3:
        raise SkillManifestError("SKILL.md frontmatter is not closed with '---'")
    try:
        raw = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise SkillManifestError(
            f"SKILL.md frontmatter is not valid YAML: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SkillManifestError("SKILL.md frontmatter must be a mapping")
    try:
        return SkillManifest(**raw)
    except ValidationError as exc:
        raise SkillManifestError(f"invalid SKILL.md manifest: {exc}") from exc
