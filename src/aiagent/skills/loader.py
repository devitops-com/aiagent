"""Load a skill's DSPy module, metric, and dataset paths.

``dspy`` is imported lazily inside the functions (never at module top), so the
import-light CLI commands stay fast and ``import aiagent.cli.app`` never pulls
DSPy.

Built-in skill entry modules load by dotted import
(``aiagent.builtin_skills.<dir>.<module>``) so they resolve their compiled
``.pyc`` in the sourceless bundle, where the ``.py`` sources are stripped. User
skills load by file path (they ship their ``.py`` source, so no ``.pyc``
fallback is needed) since they live outside the ``aiagent`` package.
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
from aiagent.skills.base import Skill, SkillSource

if TYPE_CHECKING:
    import dspy

# Package that holds the shipped built-in skills (mirrors discovery.py).
_BUILTIN_PACKAGE = "aiagent.builtin_skills"


def _import_entry(skill: Skill) -> ModuleType:
    if skill.source is SkillSource.BUILTIN:
        return _import_builtin_entry(skill)
    return _import_user_entry(skill)


def _import_builtin_entry(skill: Skill) -> ModuleType:
    """Import a built-in skill's entry module by its dotted package name.

    The subpackage is the skill's on-disk directory name (not the manifest
    ``name``, which may differ), so this resolves the sourceless ``.pyc`` in the
    shipped bundle — where ``skill.py`` is stripped — exactly as it resolves the
    ``.py`` in a dev tree.
    """
    module_name = f"{_BUILTIN_PACKAGE}.{skill.directory.name}.{skill.entrypoint_module}"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SkillLoadError(
            f"{skill.name}: could not import entry module {module_name!r}: {exc}"
        ) from exc


def _import_user_entry(skill: Skill) -> ModuleType:
    """Import a user skill's entry module from its ``<module>.py`` file."""
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
