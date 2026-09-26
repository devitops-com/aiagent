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
from the handlers of ``aiagent run`` and ``aiagent sentiment``: ``optimize``/``eval``
never see a wrapped module.

:class:`Student` is the installed student on its own (lazy load, τ, failing open), which
:class:`System1First` puts in front of the LLM predictor; its :meth:`Student.consult`
gives a :class:`Verdict` per state.

A module that declares ``system1_student = (skill, predictor)`` borrows another
skill's student instead (sentiment borrows ``polarity/classify``): :func:`apply_system1`
runs the same checks on it and hands the module a :class:`Student` through
``use_student``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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
from aiagent.skills.loader import build_module
from aiagent.skills.registry import SkillRegistry, load_registry
from aiagent.system1.artifacts import (
    SHADOW_LOG,
    InstalledArtifact,
    artifact_home,
    load_installed,
)
from aiagent.system1.contract import bytes_sha256, canonical_json
from aiagent.system1.runtime import Answer, System1Runtime

logger = logging.getLogger(__name__)

REASONING_MARKER: Final = "[system1]"

_DECIMALS: Final = 4
# `aiagent run --jsonl` calls one wrapper from several threads: load the student
# once, and never interleave two shadow lines.
_LOAD_LOCK: Final = threading.Lock()
_LOG_LOCK: Final = threading.Lock()
# A session.run already uses every core (onnxruntime's intra-op pool): concurrent runs
# only oversubscribe the CPU, so every student in the process takes its turn.
_PREDICT_LOCK: Final = threading.Lock()

Mode = Literal["shadow", "gate"]


class Verdict(NamedTuple):
    """The student's answers to one state (:meth:`Student.consult`)."""

    fits: bool  # it sees the whole state; if not, it is not run
    n_tokens: int  # the state's student tokens (what `fits` compares to the room)
    answers: dict[str, Answer]  # qid -> answer; empty when it does not fit
    label: str | None  # a single-question student's top key; None when it does not fit
    answer_confidence: float | None  # that answer's confidence; None likewise
    clears_tau: bool  # it fits and every answer is at least τ confident
    ms: int  # wall-clock of the consult: a first load or a wait for the lock included


def student_state(
    derivation: Derivation, kwargs: Mapping[str, Any]
) -> dict[str, str] | None:
    """{input_field: text} exactly as build_row makes it for training, or None if the
    input is not a str."""
    field = derivation.input_field
    if field is None or not isinstance(kwargs.get(field), str):
        return None
    return {field: kwargs[field]}


class Student:
    """An installed student: loaded once on first use, shared by threads; fails open.

    Its τ per question is the installed threshold, or `min_conf` when that is set. Any
    problem loading or running it warns once, and every later :meth:`consult` returns
    None.
    """

    def __init__(
        self,
        installed: InstalledArtifact,
        derivation: Derivation,
        *,
        min_conf: float | None,
        runtime_factory: Callable[[Path], System1Runtime] | None = None,
    ) -> None:
        if derivation.input_field is None:
            raise ValueError("a student answers a qualifying signature")
        self.installed = installed
        self.min_conf = min_conf
        self._runtime_factory = runtime_factory or System1Runtime
        self._input_field = derivation.input_field
        self._questions = derivation.questions()
        self._runtime: System1Runtime | None = None
        self._broken = False

    @property
    def broken(self) -> bool:
        """True once loading or running the student failed (warned once)."""
        return self._broken

    def threshold(self, qid: str) -> float:
        """τ for question `qid`: min_conf if set, else the installed threshold."""
        if self.min_conf is not None:
            return self.min_conf
        return self.installed.thresholds[qid]

    @property
    def tau(self) -> float:
        """τ of a one-question student; with several questions, the lowest."""
        return min(self.threshold(qid) for qid in self._questions)

    def state(self, text: str) -> dict[str, str]:
        """{input_field: text}: the state a training row makes of `text`."""
        return {self._input_field: text}

    def consult(self, state: dict[str, str]) -> Verdict | None:
        """The student's verdict on `state`; None only when the student is broken."""
        if self._broken:
            return None
        started = time.perf_counter()
        try:
            runtime = self._load()
            if runtime is None:
                return None  # another thread's load failed: warned once
            n_tokens = len(runtime.tokenizer.state_ids(state))
            fits = runtime.fits(state, self._questions, n_tokens=n_tokens)
            answers: dict[str, Answer] = {}
            if fits:  # it only ever trained on states it sees whole
                with _PREDICT_LOCK:
                    answers = runtime.predict(state, self._questions)
            clears_tau = fits and all(
                answers[qid].answer_confidence >= self.threshold(qid)
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
        ms = round((time.perf_counter() - started) * 1000)
        single = next(iter(answers.values())) if len(answers) == 1 else None
        return Verdict(
            fits=fits,
            n_tokens=n_tokens,
            answers=answers,
            label=None if single is None else single.key,
            answer_confidence=None if single is None else single.answer_confidence,
            clears_tau=clears_tau,
            ms=ms,
        )

    def _load(self) -> System1Runtime | None:
        """The runtime, loaded once; None if another thread's load failed."""
        if self._runtime is None:
            with _LOAD_LOCK:  # several `run --jsonl` threads may get here first
                if self._broken:
                    return None
                if self._runtime is None:
                    try:
                        self._runtime = self._runtime_factory(self.installed.path)
                    except Exception:
                        self._broken = True  # before the next waiter gets the lock
                        raise
        return self._runtime


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
        self.shadow_log = shadow_log
        self.student = Student(
            installed, derivation, min_conf=min_conf, runtime_factory=runtime_factory
        )

    def forward(self, **kwargs: Any) -> Any:
        """The student's prediction when the gate accepts it, else the LLM's."""
        state = None if self.student.broken else student_state(self.derivation, kwargs)
        verdict = None if state is None else self.student.consult(state)
        if state is None or verdict is None or not verdict.fits:
            return self.predictor(**kwargs)  # broken, or a state it cannot see whole
        if self.mode == "gate":
            if verdict.clears_tau:
                return self._prediction(verdict.answers)
            return self.predictor(**kwargs)
        prediction = self.predictor(**kwargs)
        self._log_shadow(verdict, prediction, state)
        return prediction

    def _prediction(self, answers: Mapping[str, Answer]) -> Any:
        """The student's answers as the wrapped predictor's dspy.Prediction."""
        mappings = self.derivation.mappings
        values = {m.field: m.decode(answers[m.field].key) for m in mappings}
        reasoning = (
            {REASONING_FIELD: REASONING_MARKER} if self.derivation.has_reasoning else {}
        )
        return dspy.Prediction(**values, **reasoning)

    def _log_shadow(self, verdict: Verdict, llm: Any, state: Mapping[str, str]) -> None:
        """Append one shadow.jsonl line; a failure only warns."""
        if self.shadow_log is None:
            return
        mappings, answers = self.derivation.mappings, verdict.answers
        keys = {m.field: answers[m.field].key for m in mappings}
        llm_keys = {m.field: m.encode(getattr(llm, m.field, None)) for m in mappings}
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifact_id": self.installed.artifact_id,
            # `run --jsonl` logs in completion order: this matches a line to its input
            # (sha256 of the input text) without logging the text.
            "input_sha256": bytes_sha256(next(iter(state.values())).encode("utf-8")),
            "student": keys,
            "llm": llm_keys,
            "confidence": {
                m.field: round(answers[m.field].answer_confidence, _DECIMALS)
                for m in mappings
            },
            "would_accept": verdict.clears_tau,
            "agree": keys == llm_keys,
            "student_ms": verdict.ms,
        }
        append_shadow(self.shadow_log, [record])


def append_shadow(
    path: Path, records: Sequence[Mapping[str, Any]], *, parents: bool = False
) -> None:
    """Append one line per record to the shadow log `path` at once, so that concurrent
    writers never interleave; `parents` creates its directory first. A failure only
    warns."""
    try:
        lines = "".join(canonical_json(record) + "\n" for record in records)
        if parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(lines)
    except (OSError, ValueError) as exc:
        logger.warning("cannot write the System 1 shadow log %s: %s", path, exc)


def _replace(module: Any, name: str, wrapper: System1First) -> None:
    """Set the predictor at a dotted path ('gen.predict' -> module.gen.predict)."""
    *parents, attr = name.split(".")
    owner = module
    for part in parents:
        owner = getattr(owner, part)
    setattr(owner, attr, wrapper)


class _Checked(NamedTuple):
    """An installed student that passed the checks, and the mode it may run in."""

    installed: InstalledArtifact
    derivation: Derivation
    mode: Mode


def _checked(
    name: str, predictor: Any, skill: Skill, settings: Settings, mode: Mode
) -> _Checked | None:
    """The student installed for skill/name, bound to `predictor`'s signature, and its
    effective mode; None (warned) if it cannot be used. ArtifactNotInstalledError when
    none is installed."""
    where = f"{skill.name}/{name}"
    if "[" in name:
        logger.warning("System 1: cannot wrap %s (list or dict), using the LLM", where)
        return None
    try:
        installed = load_installed(settings.artifacts_dir, skill.name, name)
    except ArtifactNotInstalledError:
        raise
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
    return _Checked(installed, derivation, mode)


def _wrapper(
    name: str, predictor: Any, skill: Skill, settings: Settings, mode: Mode
) -> System1First | None:
    """The wrapper for one predictor, or None (with a warning unless not installed)."""
    try:
        checked = _checked(name, predictor, skill, settings, mode)
    except ArtifactNotInstalledError:
        return None
    if checked is None:
        return None
    home = artifact_home(settings.artifacts_dir, skill.name, name)
    return System1First(
        predictor,
        derivation=checked.derivation,
        installed=checked.installed,
        mode=checked.mode,
        min_conf=settings.system1_min_conf,
        shadow_log=home / SHADOW_LOG if checked.mode == "shadow" else None,
    )


def _no_student(mode: Mode, skill: Skill, where: str) -> tuple[str, ...]:
    logger.warning(
        "System 1 is %s for skill %s, but %s has no installed student; the LLM answers",
        mode,
        skill.name,
        where,
    )
    return ()


def _borrow(
    module: Any,
    skill: Skill,
    settings: Settings,
    mode: Mode,
    registry: SkillRegistry | None,
) -> tuple[str, ...]:
    """Hand `module` the student it borrows (module.system1_student = (skill,
    predictor)), checked as a wrapper's is; return (module.system1_predictor,) or ()."""
    owner_name, name = module.system1_student
    where = f"{owner_name}/{name}"
    try:
        if registry is None:
            registry, _ = load_registry(settings)
        owner = registry.get(owner_name)
        predictor = dict(build_module(owner).named_predictors()).get(name)
    except Exception as exc:  # fail open: a user skill may shadow `owner_name` badly
        logger.warning(
            "System 1: cannot load %s for skill %s (%s); the LLM answers",
            where,
            skill.name,
            exc,
        )
        return ()
    if predictor is None:  # a user skill may shadow `owner_name` without it
        return _no_student(mode, skill, where)
    try:
        checked = _checked(name, predictor, owner, settings, mode)
    except ArtifactNotInstalledError:
        return _no_student(mode, skill, where)
    if checked is None:
        return ()
    student = Student(
        checked.installed, checked.derivation, min_conf=settings.system1_min_conf
    )
    target = module.system1_predictor
    home = artifact_home(settings.artifacts_dir, skill.name, target)
    module.use_student(student, checked.mode, home / SHADOW_LOG)
    return (target,)


def apply_system1(
    module: Any, skill: Skill, settings: Settings, registry: SkillRegistry | None = None
) -> tuple[str, ...]:
    """Wrap each qualifying, installed predictor of `module` per
    settings.system1_mode[skill]; return the wrapped names.

    A module that declares ``system1_student`` gets that skill's student instead (the
    `registry` resolves the skill, loaded if None), and the name returned is its
    ``system1_predictor``.
    """
    mode = settings.system1_mode.get(skill.name, "off")
    if mode == "off":
        return ()  # artifacts_dir is not even looked at
    if getattr(module, "system1_student", None) is not None:
        return _borrow(module, skill, settings, mode, registry)
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
