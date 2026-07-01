"""An immutable, name-indexed registry of discovered skills."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from aiagent.config import Settings
from aiagent.exceptions import SkillNotFoundError
from aiagent.skills.base import Skill
from aiagent.skills.discovery import discover_all


@dataclass(frozen=True)
class SkillRegistry:
    """Skills indexed by name (user skills shadow built-ins of the same name)."""

    _by_name: Mapping[str, Skill]

    @classmethod
    def build(cls, skills: Iterable[Skill]) -> SkillRegistry:
        index: dict[str, Skill] = {}
        for skill in skills:  # built-ins first, then user -> user wins
            index[skill.name] = skill
        return cls(MappingProxyType(index))

    def get(self, name: str) -> Skill:
        if name not in self._by_name:
            raise SkillNotFoundError(name, available=self.names())
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def all(self) -> list[Skill]:
        return [self._by_name[name] for name in self.names()]


def load_registry(settings: Settings) -> tuple[SkillRegistry, list[str]]:
    """Discover skills and build the registry; returns (registry, errors)."""
    discovery = discover_all(settings)
    return SkillRegistry.build(discovery.skills), discovery.errors
