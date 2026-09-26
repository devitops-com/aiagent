"""System 1: the Student on its own, the part of the cascade that sentiment reuses.

Its Verdict per state (fits, n_tokens, answers, label, confidence, τ, time), its τ (the
installed threshold or ``system1_min_conf``), failing open with one warning, and the
process-wide lock that lets one student call run at a time.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from aiagent.builtin_skills.polarity.skill import Polarity
from aiagent.config import load_settings
from aiagent.distill.questions import derive
from aiagent.skills.registry import load_registry
from aiagent.system1 import cascade
from aiagent.system1.artifacts import InstalledArtifact
from aiagent.system1.cascade import Student, Verdict
from aiagent.system1.runtime import Answer, System1Runtime
from aiagent.system1.sequence import Encoded, SequenceTokenizer
from system1_helpers import FIXTURE, FIXTURE_MAX_LEN, install

TEXT = "Die Lieferung kam zu spät, aber der Support war hervorragend."
STATE = {"text": TEXT}
LONG_STATE = {"text": "word " * 200}  # far more tokens than the fixture's max_len (128)
CASCADE_LOGGER = "aiagent.system1.cascade"
LABELS = ("negative", "neutral", "mixed", "positive")


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


def installed_student(tau: float = 0.0) -> InstalledArtifact:
    """The fixture installed as polarity/classify with τ = `tau`."""
    settings = load_settings()
    registry, _ = load_registry(settings)
    return install(settings, registry.get("polarity"), tau=tau)


def student(
    installed: InstalledArtifact, *, min_conf: float | None = None, factory: Any = None
) -> Student:
    return Student(
        installed, derive(Polarity), min_conf=min_conf, runtime_factory=factory
    )


def cascade_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == CASCADE_LOGGER and r.levelno == logging.WARNING
    ]


class Fixed:
    """A runtime that sees every state whole as 3 tokens and answers `neutral` at a fixed
    confidence."""

    def __init__(self, confidence: float) -> None:
        rest = (1.0 - confidence) / 3
        self.answer = Answer(
            type="choice",
            keys=LABELS,
            probabilities=(rest, confidence, rest, rest),
            answer_confidence=confidence,
            confidence=0.0,
            encoded=Encoded((), ()),
        )
        self.tokenizer = SimpleNamespace(state_ids=lambda _state: [0, 0, 0])

    def fits(self, _state: Any, _questions: Any, *, n_tokens: int | None = None) -> bool:
        return True

    def predict(self, _state: Any, _questions: Any) -> dict[str, Answer]:
        return {"polarity": self.answer}


# --------------------------------------------------------------------------- Verdict


def test_a_state_it_sees_whole_gets_its_answers(runtime: System1Runtime) -> None:
    verdict = student(installed_student()).consult(STATE)
    expected = runtime.predict(STATE, derive(Polarity).questions())["polarity"]
    assert verdict is not None
    assert verdict.fits
    assert verdict.n_tokens == len(runtime.tokenizer.state_ids(STATE))
    assert 0 < verdict.n_tokens <= FIXTURE_MAX_LEN
    assert list(verdict.answers) == ["polarity"]
    assert verdict.answers["polarity"].probabilities == pytest.approx(expected.probabilities)
    assert verdict.label == expected.key
    assert verdict.answer_confidence == pytest.approx(expected.answer_confidence)
    assert verdict.clears_tau  # τ = 0
    assert isinstance(verdict.ms, int)
    assert verdict.ms >= 0


def test_a_state_too_long_is_counted_but_never_run(runtime: System1Runtime) -> None:
    predicted: list[Any] = []

    class Watched(System1Runtime):
        def predict(self, state: Any, questions: Any) -> dict[str, Answer]:
            predicted.append(state)
            return super().predict(state, questions)

    verdict = student(installed_student(), factory=Watched).consult(LONG_STATE)
    assert verdict is not None
    n_tokens = len(runtime.tokenizer.state_ids(LONG_STATE))
    assert n_tokens > FIXTURE_MAX_LEN
    assert verdict._replace(ms=0) == Verdict(
        fits=False,
        n_tokens=n_tokens,
        answers={},
        label=None,
        answer_confidence=None,
        clears_tau=False,
        ms=0,
    )
    assert predicted == []


@pytest.mark.parametrize(("state", "tokenized"), [(STATE, 2), (LONG_STATE, 1)])
def test_a_consult_counts_the_state_once_for_n_tokens_and_the_fit(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, str], tokenized: int
) -> None:
    # n_tokens and the fit check share one count; only predict's encode (a state that
    # fits) tokenizes it again
    calls: list[Any] = []
    real = SequenceTokenizer.state_ids

    def counting(self: SequenceTokenizer, counted: Any) -> list[int]:
        calls.append(counted)
        return real(self, counted)

    subject = student(installed_student())  # installing checks the golden answers
    monkeypatch.setattr(SequenceTokenizer, "state_ids", counting)
    verdict = subject.consult(state)
    assert verdict is not None
    assert verdict.fits is (tokenized == 2)
    assert calls == [state] * tokenized


@pytest.mark.parametrize(
    ("tau", "min_conf", "clears"),
    [(0.0, None, True), (1.0, None, False), (1.0, 0.0, True), (0.0, 1.0, False)],
)
def test_clears_tau_takes_min_conf_over_the_installed_threshold(
    tau: float, min_conf: float | None, clears: bool
) -> None:
    subject = student(installed_student(tau), min_conf=min_conf)
    assert subject.threshold("polarity") == (tau if min_conf is None else min_conf)
    assert subject.tau == subject.threshold("polarity")  # its only question
    verdict = subject.consult(STATE)
    assert verdict is not None
    assert verdict.fits
    assert verdict.clears_tau is clears


@pytest.mark.parametrize(("min_conf", "clears"), [(0.9, True), (math.nextafter(0.9, 1), False)])
def test_a_confidence_equal_to_tau_clears_it(min_conf: float, clears: bool) -> None:
    subject = student(
        installed_student(), min_conf=min_conf, factory=lambda _path: Fixed(0.9)
    )
    verdict = subject.consult(STATE)
    assert verdict is not None
    assert (verdict.fits, verdict.n_tokens, verdict.label) == (True, 3, "neutral")
    assert verdict.answer_confidence == 0.9
    assert verdict.clears_tau is clears


def test_its_state_is_what_a_training_row_reads() -> None:
    assert student(installed_student()).state(TEXT) == STATE


def test_a_student_needs_a_qualifying_signature() -> None:
    with pytest.raises(ValueError, match="qualifying"):
        Student(installed_student(), derive(dspy.Signature("a, b -> c")), min_conf=None)


# --------------------------------------------------------------------------- failing open


def test_a_failed_load_is_tried_and_warned_about_once_then_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    installed = installed_student()
    loads: list[Path] = []

    def failing(path: Path) -> System1Runtime:
        loads.append(path)
        raise RuntimeError("no onnxruntime here")

    subject = student(installed, factory=failing)
    assert not subject.broken
    assert subject.consult(STATE) is None
    assert subject.consult(STATE) is None
    assert subject.broken
    assert loads == [installed.path]
    [warning] = cascade_warnings(caplog)
    assert "no onnxruntime here" in warning
    assert installed.artifact_id in warning


def test_a_failed_run_is_warned_about_once_and_never_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    predicted: list[Any] = []

    class Failing(System1Runtime):
        def predict(self, state: Any, questions: Any) -> dict[str, Answer]:
            predicted.append(state)
            raise ValueError("bad logits")

    subject = student(installed_student(), factory=Failing)
    assert subject.consult(STATE) is None
    assert subject.consult(STATE) is None
    assert subject.broken
    assert predicted == [STATE]
    [warning] = cascade_warnings(caplog)
    assert "bad logits" in warning


# --------------------------------------------------------------------------- predict lock


class Gathered:
    """The fixture runtime, shared by students: callers meet at a barrier just before the
    predict lock, and each predict records whether that lock is held and how many run."""

    def __init__(self, real: System1Runtime, parties: int) -> None:
        self._real = real
        self._barrier = threading.Barrier(parties, timeout=30)  # fails, never hangs
        self._guard = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock_held: list[bool] = []

    @property
    def tokenizer(self) -> Any:
        return self._real.tokenizer

    def fits(self, state: Any, questions: Any, *, n_tokens: int | None = None) -> bool:
        self._barrier.wait()  # every caller is loaded and about to predict
        return self._real.fits(state, questions, n_tokens=n_tokens)

    def predict(self, state: Any, questions: Any) -> dict[str, Answer]:
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.lock_held.append(cascade._PREDICT_LOCK.locked())
        try:
            return self._real.predict(state, questions)
        finally:
            with self._guard:
                self.in_flight -= 1


def test_concurrent_consults_predict_one_at_a_time_across_students(
    runtime: System1Runtime,
) -> None:
    # `run --jsonl` threads share one student, and sentiment's student is another one in
    # the same process: the lock is process-wide.
    installed = installed_student()
    parties = 8
    gathered = Gathered(runtime, parties)
    students = [student(installed, factory=lambda _path: gathered) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=parties) as pool:
        futures = [pool.submit(students[i % 2].consult, STATE) for i in range(parties)]
        verdicts = [future.result() for future in futures]
    assert all(v is not None and v.fits for v in verdicts)
    assert len({v.label for v in verdicts if v is not None}) == 1
    assert gathered.lock_held == [True] * parties
    assert gathered.max_in_flight == 1
