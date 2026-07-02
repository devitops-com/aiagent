"""Persist and resume ``aiagent chat`` sessions as JSON turn logs.

Import-safe (no ``dspy``): a session is an ordered list of ``{"text", "answer"}``
turns saved as one JSON file per name under the sessions dir, so a conversation
survives across ``aiagent chat`` invocations and can be resumed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from aiagent.config import Settings
from aiagent.exceptions import AiagentError

# Session names become filenames, so keep them to a safe, path-traversal-free set.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def session_path(settings: Settings, name: str) -> Path:
    """Resolve a session name to its JSON file, rejecting unsafe names."""
    if not _SAFE_NAME.match(name):
        raise AiagentError(
            f"invalid session name {name!r} — use letters, digits, '.', '_', or '-'"
        )
    return settings.sessions_dir / f"{name}.json"


@dataclass
class ChatSession:
    """A named, on-disk chat session: an ordered list of Q&A turns."""

    name: str
    path: Path
    turns: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, settings: Settings, name: str) -> ChatSession:
        """Load a session by name (empty when it does not exist yet)."""
        path = session_path(settings, name)
        turns: list[dict[str, str]] = []
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AiagentError(
                    f"session {name!r} is unreadable ({exc}); start fresh with --new"
                ) from exc
            if isinstance(raw, list):
                turns = [
                    {"text": str(t["text"]), "answer": str(t["answer"])}
                    for t in raw
                    if isinstance(t, dict) and "text" in t and "answer" in t
                ]
        return cls(name=name, path=path, turns=turns)

    def append(self, text: str, answer: str) -> None:
        """Record a turn and persist the session."""
        self.turns.append({"text": text, "answer": answer})
        self._save()

    def clear(self) -> None:
        """Drop all turns and persist the (now empty) session."""
        self.turns = []
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.turns, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
