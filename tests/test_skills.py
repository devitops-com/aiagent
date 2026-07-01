"""Tests for the skills engine: manifest parsing, discovery, registry, router."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest

from aiagent.config import load_settings
from aiagent.exceptions import (
    AmbiguousSkillError,
    SkillManifestError,
    SkillNotFoundError,
)
from aiagent.skills.base import SkillSource, parse_skill_md
from aiagent.skills.loader import build_metric, build_module
from aiagent.skills.registry import load_registry
from aiagent.skills.router import route


def test_parse_manifest_defaults() -> None:
    m = parse_skill_md("---\nname: foo\ndescription: bar\n---\nbody")
    assert m.name == "foo"
    assert m.entrypoint == "skill:build"


def test_parse_missing_frontmatter_raises() -> None:
    with pytest.raises(SkillManifestError):
        parse_skill_md("just a body, no frontmatter")


def test_parse_bad_name_raises() -> None:
    with pytest.raises(SkillManifestError):
        parse_skill_md("---\nname: Bad Name\ndescription: x\n---\n")


def test_registry_discovers_builtins() -> None:
    registry, errors = load_registry(load_settings())
    assert {"extract", "chat"} <= set(registry.names())
    assert errors == []


def test_user_skill_shadows_builtin_and_errors_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd = tmp_path / "skills"
    (sd / "extract").mkdir(parents=True)
    (sd / "extract" / "SKILL.md").write_text(
        "---\nname: extract\ndescription: user override\n---\n"
    )
    (sd / "broken").mkdir()
    (sd / "broken" / "SKILL.md").write_text("not a valid manifest")
    monkeypatch.setenv("AIAGENT_SKILLS_DIR", str(sd))

    registry, errors = load_registry(load_settings())
    assert registry.get("extract").source == SkillSource.USER
    assert any("broken" in e for e in errors)


def test_skill_not_found() -> None:
    registry, _ = load_registry(load_settings())
    with pytest.raises(SkillNotFoundError):
        registry.get("does-not-exist")


def test_router_exact_match() -> None:
    registry, _ = load_registry(load_settings())
    assert route("extract", registry).skill.name == "extract"


def test_router_keyword_match() -> None:
    registry, _ = load_registry(load_settings())
    chosen = route("pull merchant date and amount from an expense", registry)
    assert chosen.skill.name == "extract"


def test_router_ambiguous_raises() -> None:
    registry, _ = load_registry(load_settings())
    with pytest.raises(AmbiguousSkillError):
        route("zzz qqq vvv", registry)


def test_build_module_returns_dspy_module() -> None:
    registry, _ = load_registry(load_settings())
    module = build_module(registry.get("extract"))
    assert isinstance(module, dspy.Module)


def test_build_metric_default_is_callable() -> None:
    registry, _ = load_registry(load_settings())
    assert callable(build_metric(registry.get("extract")))
