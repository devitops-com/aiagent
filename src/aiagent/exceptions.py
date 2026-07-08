"""Exception hierarchy for aiagent.

All user-facing failures derive from :class:`AiagentError`; the CLI catches that
base class, prints the message to stderr, and exits with a clean code. Usage
errors (wrong command/flag) are handled separately by the Typer wrapper.
"""

from __future__ import annotations


class AiagentError(Exception):
    """Base class for all aiagent runtime errors."""


class AiagentConfigError(AiagentError):
    """Invalid or unloadable configuration."""


class DataLoadError(AiagentError, ValueError):
    """A dataset file was missing, empty, or contained a malformed row.

    Messages carry ``path:line`` context where applicable.
    """


class DevaiUnreachableError(AiagentError):
    """The devai router could not be reached or did not respond as expected."""


class SourceError(AiagentError):
    """A data source (local file or URL) could not be read or yielded no text."""


class SkillError(AiagentError):
    """Base class for skill discovery / loading failures."""


class SkillManifestError(SkillError):
    """A ``SKILL.md`` manifest was missing or failed validation."""


class SkillNotFoundError(SkillError):
    """A requested skill name is not in the registry."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        hint = f"; available: {', '.join(self.available)}" if self.available else ""
        super().__init__(f"unknown skill {name!r}{hint}")


class SkillLoadError(SkillError):
    """A skill's entrypoint module could not be imported or did not build a module."""


class AmbiguousSkillError(SkillError):
    """A free-text request matched more than one skill with no clear winner."""

    def __init__(self, request: str, candidates: list[str]) -> None:
        self.request = request
        self.candidates = candidates
        super().__init__(
            f"ambiguous request {request!r}; candidates: {', '.join(candidates)}"
        )
