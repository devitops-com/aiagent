"""System 1 first: a distilled student answers before the LLM predictor it wraps.

:func:`apply_system1` wraps each qualifying predictor of a skill's module that has an
installed student still bound to its signature, per ``settings.system1_mode[skill]``:

- ``gate``: the student answers when it is at least τ confident on every question (τ
  from the install, or ``system1_min_conf``); otherwise the LLM predictor answers.
- ``shadow``: the LLM predictor always answers, and the student's answer is logged
  next to it in ``shadow.jsonl`` (no input text) for the pilot.

A wrapper fails open: the student sees only states it can read whole, and any problem
loading or running it hands this and every later call to the LLM predictor, with the
kwargs unchanged. dspy is imported at the top, so this module is imported lazily, only
from ``aiagent run``'s handler: ``optimize``/``eval`` never see a wrapped module.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import dspy

from aiagent.config import Settings
from aiagent.distill.questions import (
    REASONING_FIELD,
    Derivation,
    derive,
    skill_source_sha256,
)
from aiagent.exceptions import ArtifactError, ArtifactNotInstalledError
from aiagent.skills.base import Skill
from aiagent.system1.artifacts import (
    SHADOW_LOG,
    InstalledArtifact,
    artifact_home,
    load_installed,
)
from aiagent.system1.contract import canonical_json
from aiagent.system1.runtime import Answer, System1Runtime

logger = logging.getLogger(__name__)

REASONING_MARKER: Final = "[system1]"

_DECIMALS: Final = 4
# `aiagent run --jsonl` calls one wrapper from several threads: load the student
# once, and never interleave two shadow lines.
_LOAD_LOCK: Final = threading.Lock()
_LOG_LOCK: Final = threading.Lock()

Mode = Literal["shadow", "gate"]


class _Student(NamedTuple):
    """The student's answers to one state, and whether each clears its τ."""

    answers: dict[str, Answer]
    accepted: bool


def student_state(
    derivation: Derivation, kwargs: Mapping[str, Any]
) -> dict[str, str] | None:
    """{input_field: text} exactly as build_row makes it for training, or None if the
    input is not a str."""
    field = derivation.input_field
    if field is None or not isinstance(kwargs.get(field), str):
        return None
    return {field: kwargs[field]}


class System1First(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """Student first, LLM predictor on doubt; fails open to the LLM."""

    def __init__(
        self,
        predictor: Any,
        *,
        derivation: Derivation,
        installed: InstalledArtifact,
        mode: Mode,
        min_conf: float | None,
        shadow_log: Path | None,
        runtime_factory: Callable[[Path], System1Runtime] | None = None,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.derivation = derivation
        self.installed = installed
        self.mode = mode
        self.min_conf = min_conf
        self.shadow_log = shadow_log
        self._runtime_factory = runtime_factory or System1Runtime
        self._questions = derivation.questions()
        self._runtime: System1Runtime | None = None
        self._broken = False

    def forward(self, **kwargs: Any) -> Any:
        """The student's prediction when the gate accepts it, else the LLM's."""
        state = None if self._broken else student_state(self.derivation, kwargs)
        if state is None:
            return self.predictor(**kwargs)
        started = time.perf_counter()
        student = self._student(state)
        student_ms = round((time.perf_counter() - started) * 1000)
        if self.mode == "gate":
            if student is not None and student.accepted:
                return self._prediction(student.answers)
            return self.predictor(**kwargs)
        prediction = self.predictor(**kwargs)
        if student is not None:
            self._log_shadow(student, prediction, student_ms)
        return prediction

    def _student(self, state: dict[str, str]) -> _Student | None:
        """The student's answers; None if the state does not fit or it failed."""
        try:
            if self._runtime is None:
                with _LOAD_LOCK:  # several `run --jsonl` threads may get here first
                    if self._broken:
                        return None  # another thread's load failed: warned once
                    if self._runtime is None:
                        try:
                            self._runtime = self._runtime_factory(self.installed.path)
                        except Exception:
                            self._broken = True  # before the next waiter gets the lock
                            raise
            if not self._runtime.fits(state, self._questions):
                return None  # it only ever trained on states it sees whole
            answers = self._runtime.predict(state, self._questions)
            accepted = all(
                answers[qid].answer_confidence >= self._threshold(qid)
                for qid in self._questions
            )
        except Exception as exc:  # fail open: the LLM answers this and every later call
            self._broken = True
            logger.warning(
                "System 1 student %s for %s/%s failed, using the LLM: %s",
                self.installed.artifact_id,
                self.installed.skill,
                self.installed.predictor,
                exc,
            )
            return None
        return _Student(answers, accepted)

    def _threshold(self, qid: str) -> float:
        if self.min_conf is not None:
            return self.min_conf
        return self.installed.thresholds[qid]

    def _prediction(self, answers: Mapping[str, Answer]) -> Any:
        """The student's answers as the wrapped predictor's dspy.Prediction."""
        mappings = self.derivation.mappings
        values = {m.field: m.decode(answers[m.field].key) for m in mappings}
        reasoning = (
            {REASONING_FIELD: REASONING_MARKER} if self.derivation.has_reasoning else {}
        )
        return dspy.Prediction(**values, **reasoning)

    def _log_shadow(self, student: _Student, llm: Any, student_ms: int) -> None:
        """Append one shadow.jsonl line; a failure only warns."""
        if self.shadow_log is None:
            return
        mappings, answers = self.derivation.mappings, student.answers
        keys = {m.field: answers[m.field].key for m in mappings}
        llm_keys = {m.field: m.encode(getattr(llm, m.field, None)) for m in mappings}
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifact_id": self.installed.artifact_id,
            "student": keys,
            "llm": llm_keys,
            "confidence": {
                m.field: round(answers[m.field].answer_confidence, _DECIMALS)
                for m in mappings
            },
            "would_accept": student.accepted,
            "agree": keys == llm_keys,
            "student_ms": student_ms,
        }
        try:
            line = canonical_json(record) + "\n"
            with _LOG_LOCK, self.shadow_log.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except (OSError, ValueError) as exc:
            logger.warning(
                "cannot write the System 1 shadow log %s: %s", self.shadow_log, exc
            )


def _replace(module: Any, name: str, wrapper: System1First) -> None:
    """Set the predictor at a dotted path ('gen.predict' -> module.gen.predict)."""
    *parents, attr = name.split(".")
    owner = module
    for part in parents:
        owner = getattr(owner, part)
    setattr(owner, attr, wrapper)


def _wrapper(
    name: str, predictor: Any, skill: Skill, settings: Settings, mode: Mode
) -> System1First | None:
    """The wrapper for one predictor, or None (with a warning unless not installed)."""
    where = f"{skill.name}/{name}"
    if "[" in name:
        logger.warning("System 1: cannot wrap %s (list or dict), using the LLM", where)
        return None
    try:
        installed = load_installed(settings.artifacts_dir, skill.name, name)
    except ArtifactNotInstalledError:
        return None
    except ArtifactError as exc:
        logger.warning("System 1: %s; using the LLM for %s", exc, where)
        return None
    derivation = derive(predictor.signature)
    binds = installed.binds
    if not derivation.qualifies or (
        derivation.signature_sha256,
        derivation.question_set_sha256,
    ) != (binds.signature_sha256, binds.question_set_sha256):
        logger.warning(
            "System 1: the student %s for %s is unbound (the signature or questions "
            "changed); re-run distill. Using the LLM",
            installed.artifact_id,
            where,
        )
        return None
    if mode == "gate" and skill_source_sha256(skill) != installed.skill_source_sha256:
        logger.warning(
            "System 1: skill %s changed since install, so %s only shadows; re-run "
            "`aiagent distill eval` and `install` to re-enable the gate",
            skill.name,
            where,
        )
        mode = "shadow"
    home = artifact_home(settings.artifacts_dir, skill.name, name)
    return System1First(
        predictor,
        derivation=derivation,
        installed=installed,
        mode=mode,
        min_conf=settings.system1_min_conf,
        shadow_log=home / SHADOW_LOG if mode == "shadow" else None,
    )


def apply_system1(module: Any, skill: Skill, settings: Settings) -> tuple[str, ...]:
    """Wrap each qualifying, installed predictor of `module` per
    settings.system1_mode[skill]; return the wrapped names."""
    mode = settings.system1_mode.get(skill.name, "off")
    if mode == "off":
        return ()  # artifacts_dir is not even looked at
    wrapped = []
    for name, predictor in list(module.named_predictors()):
        wrapper = _wrapper(name, predictor, skill, settings, mode)
        if wrapper is not None:
            _replace(module, name, wrapper)  # in place: it was built for this run
            wrapped.append(name)
    if not wrapped:
        logger.warning(
            "System 1 is %s for skill %s, but no predictor has an installed, bound "
            "student; the LLM answers",
            mode,
            skill.name,
        )
    return tuple(wrapped)
