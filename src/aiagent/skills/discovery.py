"""Discover skills from the built-in package data and the user skills dir.

Discovery is fail-soft: a malformed skill is collected as an error string rather
than aborting the whole scan, so one bad user skill can't blank the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

from aiagent.config import Settings
from aiagent.exceptions import SkillError
from aiagent.skills.base import Skill, SkillSource, parse_skill_md

_BUILTIN_PACKAGE = "aiagent.builtin_skills"
_MANIFEST = "SKILL.md"


@dataclass(frozen=True)
class Discovery:
    """Result of a scan: the skills that loaded, and per-skill error messages."""

    skills: list[Skill]
    errors: list[str]


def _load_skill_from_dir(directory: Path, source: SkillSource) -> Skill:
    manifest = parse_skill_md((directory / _MANIFEST).read_text(encoding="utf-8"))
    return Skill(manifest=manifest, source=source, directory=directory)


def _builtin_dirs() -> list[Path]:
    root = files(_BUILTIN_PACKAGE)
    dirs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.joinpath(_MANIFEST).is_file():
            continue
        # Real on-disk path (the bundle is unpacked, not a zipapp).
        with as_file(child) as real:
            dirs.append(Path(real))
    return dirs


def _user_dirs(skills_dir: Path) -> list[Path]:
    if not skills_dir.is_dir():
        return []
    return [
        child
        for child in sorted(skills_dir.iterdir())
        if child.is_dir() and (child / _MANIFEST).is_file()
    ]


def discover_all(settings: Settings) -> Discovery:
    """Scan built-in then user skills (user overrides built-in by name later)."""
    skills: list[Skill] = []
    errors: list[str] = []
    candidates: list[tuple[Path, SkillSource]] = [
        *((d, SkillSource.BUILTIN) for d in _builtin_dirs()),
        *((d, SkillSource.USER) for d in _user_dirs(settings.skills_dir)),
    ]
    for directory, source in candidates:
        try:
            skills.append(_load_skill_from_dir(directory, source))
        except (SkillError, OSError) as exc:
            errors.append(f"{directory.name} ({source}): {exc}")
    return Discovery(skills=skills, errors=errors)
