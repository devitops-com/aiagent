"""The skills engine: discover, register, route to, and load skills.

A *skill* is a directory with a ``SKILL.md`` manifest (YAML frontmatter) and a
Python module exposing a ``build()`` factory that returns a ``dspy.Module``.
Built-in skills ship as package data under ``aiagent.builtin_skills``; user skills
live under ``~/.config/aiagent/skills``.
"""

from __future__ import annotations
