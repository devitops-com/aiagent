"""Load a skill's DSPy module, metric, and dataset paths.

``dspy`` is imported lazily inside the functions (never at module top), so the
import-light CLI commands stay fast and ``import aiagent.cli.app`` never pulls
DSPy. Skill entry modules are imported by file path, which works identically for
built-in (package-data) and user skills.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

from aiagent.exceptions import SkillLoadError
from aiagent.skills.base import Skill

if TYPE_CHECKING:
    import dspy


def _import_entry(skill: Skill) -> ModuleType:
    path = skill.directory / f"{skill.entrypoint_module}.py"
    if not path.is_file():
        raise SkillLoadError(f"{skill.name}: entry module {path.name} not found")
    mod_name = f"_aiagent_skill_{skill.source}_{skill.name}_{skill.entrypoint_module}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise SkillLoadError(f"{skill.name}: could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def build_module(skill: Skill) -> dspy.Module:
    """Instantiate the skill's DSPy module via its ``build()`` factory."""
    import dspy

    module = _import_entry(skill)
    factory = getattr(module, skill.entrypoint_attr, None)
    if not callable(factory):
        raise SkillLoadError(
            f"{skill.name}: entrypoint {skill.manifest.entrypoint!r} is not callable"
        )
    obj = factory()
    if not isinstance(obj, dspy.Module):
        raise SkillLoadError(
            f"{skill.name}: build() returned {type(obj).__name__}, not a dspy.Module"
        )
    return obj


def build_metric(skill: Skill) -> Callable[..., Any]:
    """Resolve the skill's metric, defaulting to the field-accuracy metric."""
    ref = skill.manifest.metric
    if ref is None:
        from aiagent.metrics.extraction import field_accuracy

        return field_accuracy
    location, _, attr = ref.partition(":")
    if not attr:
        raise SkillLoadError(f"{skill.name}: metric must be '<module>:<attr>'")
    source = (
        _import_entry(skill)
        if location == "skill"
        else importlib.import_module(location)
    )
    fn = getattr(source, attr, None)
    if not callable(fn):
        raise SkillLoadError(f"{skill.name}: metric {ref!r} is not callable")
    return cast("Callable[..., Any]", fn)


def dataset_path(skill: Skill, which: Literal["trainset", "devset"]) -> Path | None:
    """Resolve the skill's train/dev dataset path (relative to the skill dir)."""
    rel = getattr(skill.manifest, which)
    return (skill.directory / rel) if rel else None
