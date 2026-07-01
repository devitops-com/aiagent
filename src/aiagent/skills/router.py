"""Route a free-text request to a skill (Claude-Code-style invocation).

Deterministic first: exact name, then keyword overlap over name + description.
An optional DSPy ``Predict`` selector (``use_llm=True``) breaks ties using the LM.
``dspy`` is imported lazily only on the LLM path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aiagent.config import Settings
from aiagent.exceptions import AmbiguousSkillError
from aiagent.skills.base import Skill
from aiagent.skills.registry import SkillRegistry

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RouteResult:
    """The chosen skill plus a human-readable reason."""

    skill: Skill
    reason: str


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def _overlap(request_tokens: set[str], skill: Skill) -> int:
    target = _tokens(f"{skill.name} {skill.manifest.description}")
    return len(request_tokens & target)


def route(
    request: str,
    registry: SkillRegistry,
    *,
    use_llm: bool = False,
    settings: Settings | None = None,
) -> RouteResult:
    """Pick the skill best matching ``request``."""
    skills = registry.all()
    if not skills:
        raise AmbiguousSkillError(request, [])

    low = request.strip().casefold()
    for skill in skills:
        if skill.name.casefold() == low:
            return RouteResult(skill, "exact name match")

    req_tokens = _tokens(request)
    scored = sorted(
        ((_overlap(req_tokens, skill), skill) for skill in skills),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if best_score > 0 and best_score > runner_up:
        return RouteResult(best, f"keyword match (score {best_score})")

    if use_llm and settings is not None:
        return _route_with_llm(request, skills, settings)

    raise AmbiguousSkillError(request, [skill.name for skill in skills])


def _route_with_llm(
    request: str, skills: list[Skill], settings: Settings
) -> RouteResult:
    import dspy

    from aiagent.llm.lm import configure_default

    configure_default(settings)
    options = "\n".join(f"- {s.name}: {s.manifest.description}" for s in skills)
    selector = dspy.Predict("request, options -> skill_name")
    prediction = selector(request=request, options=options)
    chosen = (getattr(prediction, "skill_name", "") or "").strip()
    for skill in skills:
        if skill.name == chosen:
            return RouteResult(skill, "LLM selector")
    raise AmbiguousSkillError(request, [skill.name for skill in skills])
