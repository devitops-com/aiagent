"""aiagent — a programmatic DSPy agent framework.

Focused on query/prompt optimization, goal-reaching loops, and autonomous data
processing over OpenAI-compatible local LLMs (the devai router). Not a chat UI.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    # Track the installed package metadata (from pyproject.toml, the single
    # source of truth) instead of a hand-maintained literal that can drift
    # from the packaged/released version (issue #5).
    __version__ = _version("aiagent")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
