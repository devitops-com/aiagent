"""System 1 in sentiment: the borrowed polarity student scores neutral segments (no network).

The fixture student (``tests/fixtures/system1``) is installed as ``polarity/classify``
the way ``distill install`` does. Its texts are picked by what it answers: it gives each
of NEUTRAL its top label ``neutral`` and each of OTHER another one (asserted below, so a
new fixture fails here first). ``install(tau=0.0)`` makes every answer clear τ, since
``system1_min_conf`` must be above 0.

Per-segment LLM answers use DummyLM's dict mode, as in test_sentiment.py: the explain
answer comes first, under a field header only the explain prompt carries, and no segment
text is a substring of another. Unless a test says otherwise, ``NEUTRAL`` is pinned to
the fixture's ``artifact_id`` and the current ``ScoreSegment`` hash.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner

from aiagent.builtin_skills.polarity.skill import Polarity
from aiagent.cli.app import app
from aiagent.config import Settings, load_settings
from aiagent.core import sentiment as sentiment_mod
from aiagent.core.sentiment import (
    SCORE_SIGNATURE_SHA256,
    NeutralCalibration,
    ScoreSegment,
    SentimentModule,
)
from aiagent.core.sentiment_stats import summarize
from aiagent.distill.questions import derive, signature_sha256
from aiagent.exceptions import SkillLoadError
from aiagent.skills.base import Skill
from aiagent.skills.loader import build_module
from aiagent.skills.registry import SkillRegistry, load_registry
from aiagent.system1 import cascade
from aiagent.system1.artifacts import SHADOW_LOG, InstalledArtifact, artifact_home
from aiagent.system1.cascade import Student, apply_system1
from aiagent.system1.runtime import Answer, System1Runtime
from aiagent.system1.sequence import Encoded
from system1_helpers import FIXTURE, FIXTURE_MAX_LEN, install

runner = CliRunner()

EXPLAIN_KEY = "[[ ## excerpts ## ]]"  # a field only the explain prompt carries
EXPLAIN = {"explanation": "Mostly neutral, with two sour notes."}
NEUTRAL = ("Alpha launched on time.", "Chapter two lies to the north.", "The report covers the area.")
OTHER = ("Beta crashed twice.", "The invoice shows the wrong address.")
LONG = " ".join(["lengthy"] * 150) + "."  # more student tokens than the fixture's 128
SCORES = {
    NEUTRAL[0]: 1,
    NEUTRAL[1]: 0,
    NEUTRAL[2]: -1,
    OTHER[0]: -6,
    OTHER[1]: -3,
    LONG: 2,
}
ORDER = [NEUTRAL[0], OTHER[0], NEUTRAL[1], OTHER[1], NEUTRAL[2]]  # student, LLM, ...
TEXT = "\n\n".join(ORDER)  # paragraphs: one segment each
CALIBRATION = {"level": 0.4, "se": 0.05, "sigma_between": 0.6, "sigma_within": 1.2}
LABELS = ("negative", "neutral", "mixed", "positive")
FIELDS = [
    "ts", "artifact_id", "run_id", "seg_index", "n_segments", "doc_sha256", "input_sha256",
    "n_tokens", "fits", "student", "confidence", "would_accept", "llm_samples", "student_ms",
]
GENERIC_WARNING = "no predictor has an installed"
STALE = "calibration is missing or stale"


# --------------------------------------------------------------------------- helpers


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def segment_of(content: str) -> str:
    return next(segment for segment in SCORES if segment in content)


def score_answer(segment: str, score: object | None = None) -> dict[str, str]:
    value = SCORES[segment] if score is None else score
    return {"reasoning": "weighed it", "score": str(value), "rationale": f"rated {value}"}


def answers() -> dict[str, dict[str, str]]:
    """DummyLM dict-mode answers: explain first, then one per segment."""
    return {EXPLAIN_KEY: EXPLAIN, **{segment: score_answer(segment) for segment in SCORES}}


def is_explain(entry: dict[str, Any]) -> bool:
    return EXPLAIN_KEY in entry["messages"][-1]["content"]


def scored_segments(lm: Any) -> Counter[str]:
    """How many score calls each segment got."""
    return Counter(
        segment_of(e["messages"][-1]["content"]) for e in lm.history if not is_explain(e)
    )


class RolloutLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
    """A DummyLM whose answer depends on the prompt and on the call's rollout id."""

    def __init__(self, pick: Callable[[str, int | None], dict[str, str]]) -> None:
        super().__init__({})
        self.pick = pick

    def forward(self, prompt: str | None = None, messages: Any = None, **kwargs: Any) -> Any:
        answer = self.pick(messages[-1]["content"], kwargs.get("rollout_id"))
        message = dotdict(content=self._format_answer_fields(answer), tool_calls=None)
        return dotdict(
            choices=[dotdict(message=message, finish_reason="stop")],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


def by_rollout(content: str, rollout: int | None) -> dict[str, str]:
    """Sample j of a segment scores SCORES + j: every segment has model spread."""
    if EXPLAIN_KEY in content:
        return EXPLAIN
    segment = segment_of(content)
    return score_answer(segment, SCORES[segment] + (rollout or 0))


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


@pytest.fixture
def registry() -> SkillRegistry:
    return load_registry(load_settings())[0]


@pytest.fixture
def polarity(registry: SkillRegistry) -> Skill:
    return registry.get("polarity")


def student_answer(runtime: System1Runtime, text: str) -> Answer:
    return runtime.predict({"text": text}, derive(Polarity).questions())["polarity"]


def test_the_fixture_texts_get_the_labels_the_tests_rely_on(runtime: System1Runtime) -> None:
    questions = derive(Polarity).questions()
    assert [student_answer(runtime, t).key for t in NEUTRAL] == ["neutral"] * 3
    assert all(student_answer(runtime, t).key != "neutral" for t in OTHER)
    assert not runtime.fits({"text": LONG}, questions)
    assert all(a not in b for a in SCORES for b in SCORES if a != b)


def settings_for(
    monkeypatch: pytest.MonkeyPatch, mode: str, *, min_conf: float | None = None
) -> Settings:
    monkeypatch.setenv("AIAGENT_SYSTEM1_MODE", json.dumps({"sentiment": mode}))
    if min_conf is not None:
        monkeypatch.setenv("AIAGENT_SYSTEM1_MIN_CONF", str(min_conf))
    return load_settings()


def pin(
    monkeypatch: pytest.MonkeyPatch, installed: InstalledArtifact, **changes: Any
) -> NeutralCalibration:
    """Pin NEUTRAL to `installed` and the current ScoreSegment hash, plus `changes`."""
    fields: dict[str, Any] = {
        "artifact_id": installed.artifact_id,
        "score_signature_sha256": signature_sha256(ScoreSegment),
        "n": 120,
        "measured": "2026-09-26, test corpus",
        **CALIBRATION,
        **changes,
    }
    calibration = NeutralCalibration(**fields)
    monkeypatch.setattr(sentiment_mod, "NEUTRAL", calibration)
    return calibration


def with_student(settings: Settings, registry: SkillRegistry) -> tuple[Any, tuple[str, ...]]:
    """The sentiment module after apply_system1, and what apply_system1 returned."""
    skill = registry.get("sentiment")
    module = build_module(skill)
    return module, apply_system1(module, skill, settings, registry)


def run(module: Any, lm: Any, text: str = TEXT, **kwargs: Any) -> Any:
    with dspy.context(lm=lm):
        return module(text=text, **kwargs)


def off(lm: Any, text: str = TEXT, **kwargs: Any) -> Any:
    """The same text in off mode, for comparison."""
    return run(SentimentModule(), lm, text, **kwargs)


def shadow_lines(settings: Settings) -> list[dict[str, Any]]:
    path = artifact_home(settings.artifacts_dir, "sentiment", "score") / SHADOW_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def without_student(prediction: Any) -> dict[str, Any]:
    """The prediction as a dict, less what only System 1 adds: `system1` and each
    segment's `student`."""
    data = prediction.toDict()
    data.pop("system1")
    data["segments"] = [
        {k: v for k, v in segment.items() if k != "student"} for segment in data["segments"]
    ]
    return data


# --------------------------------------------------------------------------- the calibration


def test_the_score_signature_hash_is_pinned() -> None:
    assert signature_sha256(ScoreSegment) == SCORE_SIGNATURE_SHA256, (
        "ScoreSegment changed, so a pinned NEUTRAL calibration no longer applies: "
        "recalibrate (docs/design/sentiment-system1.md §2.9), pin NEUTRAL again and "
        "update SCORE_SIGNATURE_SHA256"
    )


def test_the_student_variance_stands_in_for_an_r_sample_mean() -> None:
    calibration = NeutralCalibration(
        artifact_id="a", score_signature_sha256="h", n=10, measured="m", **CALIBRATION
    )
    assert calibration.variance(1) == pytest.approx(0.6**2 + 1.2**2)
    assert calibration.variance(3) == pytest.approx(0.6**2 + 1.2**2 / 3)


# --------------------------------------------------------------------------- gate


def test_gate_scores_the_neutral_segments_with_the_student(
    polarity: Skill,
    registry: SkillRegistry,
    runtime: System1Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0)
    calibration = pin(monkeypatch, installed)
    module, names = with_student(settings, registry)
    lm = DummyLM(answers())

    pred = run(module, lm, resample=2)

    assert names == ("score",)
    # Only the escalated segments reach the LLM, r times each, then explain.
    assert scored_segments(lm) == {OTHER[0]: 2, OTHER[1]: 2}
    assert len(lm.history) == 4 + 1
    assert [s["source"] for s in pred.segments] == [
        "student", "llm", "student", "llm", "student"
    ]
    for segment, row in zip(ORDER, pred.segments, strict=True):
        answer = student_answer(runtime, segment)
        assert row["student"] == {
            "label": answer.key, "confidence": round(answer.answer_confidence, 4)
        }
        if row["source"] == "student":
            assert (row["score"], row["rationale"]) == (0.4, None)
        else:
            assert (row["score"], row["rationale"]) == (SCORES[segment], f"rated {SCORES[segment]}")
    expected = summarize(
        [[SCORES[OTHER[0]]] * 2, [SCORES[OTHER[1]]] * 2],
        student_scores=[calibration.level] * 3,
        student_variance=calibration.variance(2),
    )
    assert pred.sentiment == round(expected.mean, 2)
    assert pred.volatility == round(expected.volatility, 3)
    assert pred.std_error == round(expected.std_error, 3)
    assert (pred.n_segments, pred.n_samples, pred.n_resampled) == (5, 4, 2)
    assert pred.model_uncertainty == 0.0  # over the two escalated segments only
    assert pred.system1 == {
        "mode": "gate",
        "student": "polarity/classify",
        "artifact_id": installed.artifact_id,
        "tau": 0.0,
        "accepted": 3,
        "too_long": 0,
        "coverage": 0.6,
    }
    # The explain prompt sees the LLM rationales and one System 1 line, never the text.
    [explain] = [e for e in lm.history if is_explain(e)]
    prompt = "\n".join(m["content"] for m in explain["messages"])
    assert "rated -6" in prompt and "rated -3" in prompt
    assert "3 of 5 segments neutral (System 1)" in prompt
    assert not any(segment in prompt for segment in SCORES)
    # Gate logs every segment too; a student segment has no LLM samples.
    lines = shadow_lines(settings)
    assert [line["llm_samples"] for line in lines] == [
        [], [-6.0, -6.0], [], [-3.0, -3.0], []
    ]
    assert [line["would_accept"] for line in lines] == [True, False, True, False, True]


def test_gate_at_min_conf_1_equals_off_mode_plus_the_system1_block(
    polarity: Skill, registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "gate", min_conf=1.0)
    installed = install(settings, polarity, tau=0.0)
    pin(monkeypatch, installed)
    module, _ = with_student(settings, registry)
    lm = DummyLM(answers())

    gate = run(module, lm, resample=2)

    assert scored_segments(lm) == dict.fromkeys(ORDER, 2)
    assert gate.system1 == {
        "mode": "gate",
        "student": "polarity/classify",
        "artifact_id": installed.artifact_id,
        "tau": 1.0,
        "accepted": 0,
        "too_long": 0,
        "coverage": 0.0,
    }
    assert all(s["source"] == "llm" and s["student"]["label"] in LABELS for s in gate.segments)
    baseline = off(DummyLM(answers()), resample=2)
    assert baseline.system1 is None
    assert all(s["student"] is None for s in baseline.segments)
    assert without_student(gate) == without_student(baseline)


def test_a_segment_the_student_cannot_see_whole_goes_to_the_llm(
    polarity: Skill,
    registry: SkillRegistry,
    runtime: System1Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0)
    pin(monkeypatch, installed)
    module, _ = with_student(settings, registry)
    lm = DummyLM(answers())

    pred = run(module, lm, text=f"{NEUTRAL[0]}\n\n{LONG}")

    assert scored_segments(lm) == {LONG: 1}
    assert [s["source"] for s in pred.segments] == ["student", "llm"]
    assert pred.segments[1]["student"] is None  # it never saw the segment
    assert pred.system1["too_long"] == 1
    assert (pred.system1["accepted"], pred.system1["coverage"]) == (1, 0.5)
    long_line = shadow_lines(settings)[1]
    assert long_line["fits"] is False
    assert long_line["n_tokens"] == len(runtime.tokenizer.state_ids({"text": LONG}))
    assert long_line["n_tokens"] > FIXTURE_MAX_LEN
    assert (long_line["student"], long_line["confidence"]) == (None, None)
    assert long_line["would_accept"] is False
    assert long_line["llm_samples"] == [2.0]


# --------------------------------------------------------------------------- shadow


def test_shadow_keeps_off_modes_statistics_and_logs_every_segment(
    polarity: Skill,
    registry: SkillRegistry,
    runtime: System1Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    installed = install(settings, polarity, tau=0.0)
    module, names = with_student(settings, registry)
    segments = [*ORDER, NEUTRAL[0]]  # the first segment again, at position 5
    text = "\n\n".join(segments)
    lm = RolloutLM(by_rollout)

    pred = run(module, lm, text=text, resample=3)

    assert names == ("score",)
    baseline = off(RolloutLM(by_rollout), text=text, resample=3)
    assert without_student(pred) == without_student(baseline)
    assert pred.model_uncertainty == pytest.approx(1.0)  # s, s+1, s+2 everywhere
    assert all(s["source"] == "llm" for s in pred.segments)
    assert pred.system1 == {
        "mode": "shadow",
        "student": "polarity/classify",
        "artifact_id": installed.artifact_id,
        "tau": 0.0,
        "accepted": 4,  # what the gate would have taken: the neutral ones
        "too_long": 0,
        "coverage": round(4 / 6, 3),
    }
    lines = shadow_lines(settings)
    assert len(lines) == len(segments)
    assert all(list(line) == FIELDS for line in lines)
    assert [line["seg_index"] for line in lines] == list(range(6))
    assert {line["n_segments"] for line in lines} == {6}
    assert len({line["run_id"] for line in lines}) == 1
    assert {line["doc_sha256"] for line in lines} == {sha256(text)}
    assert {line["artifact_id"] for line in lines} == {installed.artifact_id}
    for segment, line in zip(segments, lines, strict=True):
        answer = student_answer(runtime, segment)
        assert line["input_sha256"] == sha256(segment)
        assert line["n_tokens"] == len(runtime.tokenizer.state_ids({"text": segment}))
        assert line["fits"] is True
        assert (line["student"], line["confidence"]) == (
            answer.key, round(answer.answer_confidence, 4)
        )
        assert line["would_accept"] is (answer.key == "neutral")
        assert line["llm_samples"] == [SCORES[segment] + j for j in range(3)]
        assert isinstance(line["student_ms"], int)
        assert line["ts"].endswith("Z")
    assert lines[5]["student_ms"] == 0  # the repeat reused the first verdict
    assert not any(segment in json.dumps(lines) for segment in segments)  # no text

    run(module, RolloutLM(by_rollout), text=text, resample=3)

    run_ids = [line["run_id"] for line in shadow_lines(settings)]
    assert len(run_ids) == 12 and len(set(run_ids)) == 2  # one run_id per forward


def test_a_dropped_sample_is_logged_as_null(
    polarity: Skill, registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, polarity, tau=0.0)
    module, _ = with_student(settings, registry)

    def pick(content: str, rollout: int | None) -> dict[str, str]:
        if EXPLAIN_KEY not in content and segment_of(content) == OTHER[0] and rollout == 1:
            return score_answer(OTHER[0], "banana")
        return by_rollout(content, rollout)

    run(module, RolloutLM(pick), resample=3)

    assert shadow_lines(settings)[1]["llm_samples"] == [-6.0, None, -4.0]


def test_a_shadow_log_that_cannot_be_written_only_warns(
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, polarity, tau=0.0)
    blocker = settings.artifacts_dir / "system1" / "skills" / "sentiment"
    blocker.write_text("a file where the log directory would go\n", encoding="utf-8")
    module, _ = with_student(settings, registry)

    pred = run(module, DummyLM(answers()))

    assert pred.system1["mode"] == "shadow"
    [warning] = warnings_in(caplog)
    assert "cannot write the System 1 shadow log" in warning


# --------------------------------------------------------------------------- the guard


@pytest.mark.parametrize("stale", ["none", "artifact_id", "signature"])
def test_gate_without_a_matching_calibration_only_shadows(
    stale: str,
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0)
    if stale == "none":
        monkeypatch.setattr(sentiment_mod, "NEUTRAL", None)
    elif stale == "artifact_id":
        pin(monkeypatch, installed, artifact_id="a-another-student")
    else:
        pin(monkeypatch, installed, score_signature_sha256="0" * 64)
    module, names = with_student(settings, registry)
    lm = DummyLM(answers())

    pred = run(module, lm)

    assert names == ("score",)
    [warning] = warnings_in(caplog)
    assert STALE in warning
    assert scored_segments(lm) == dict.fromkeys(ORDER, 1)  # the LLM scores everything
    assert pred.system1["mode"] == "shadow"
    assert pred.system1["accepted"] == 3  # what a calibrated gate would have taken
    assert all(s["source"] == "llm" for s in pred.segments)
    assert without_student(pred) == without_student(off(DummyLM(answers())))


def test_the_shipped_calibration_is_none_so_gate_shadows(
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    install(settings, polarity, tau=0.0)
    module, _ = with_student(settings, registry)

    pred = run(module, DummyLM(answers()))

    assert pred.system1["mode"] == "shadow"
    assert any(STALE in w for w in warnings_in(caplog))


# --------------------------------------------------------------------------- failing open


def test_a_broken_student_leaves_every_segment_to_the_llm_with_one_warning(
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0)
    pin(monkeypatch, installed)
    loads: list[Path] = []

    def broken(path: Path) -> System1Runtime:
        loads.append(path)
        raise RuntimeError("no onnxruntime here")

    monkeypatch.setattr(cascade, "System1Runtime", broken)
    module, _ = with_student(settings, registry)
    lm = DummyLM(answers())

    first = run(module, lm, resample=2)
    second = run(module, lm, resample=2)

    assert loads == [installed.path]
    [warning] = warnings_in(caplog)
    assert "no onnxruntime here" in warning
    baseline = off(DummyLM(answers()), resample=2)
    assert first.toDict() == second.toDict() == baseline.toDict()  # system1 null too
    assert not (artifact_home(settings.artifacts_dir, "sentiment", "score") / SHADOW_LOG).exists()


class OtherPolarity(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Classify a passage."""  # other instructions: another signature hash

    text: str = dspy.InputField(desc="A passage of text to assess.")
    polarity: Literal["negative", "neutral", "mixed", "positive"] = dspy.OutputField(
        desc="Overall sentiment polarity of the passage."
    )


def test_a_student_bound_to_another_signature_is_not_used(
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, signature=OtherPolarity, tau=0.0)
    pin(monkeypatch, installed)
    module, names = with_student(settings, registry)

    pred = run(module, DummyLM(answers()))

    assert names == ()
    [warning] = warnings_in(caplog)
    assert "unbound" in warning and "polarity/classify" in warning
    assert pred.system1 is None
    assert all(s["source"] == "llm" and s["student"] is None for s in pred.segments)


def test_a_changed_polarity_skill_only_shadows(
    polarity: Skill,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0, source="0" * 64)  # SKILL.md since
    pin(monkeypatch, installed)
    module, names = with_student(settings, registry)
    lm = DummyLM(answers())

    pred = run(module, lm)

    assert names == ("score",)
    [warning] = warnings_in(caplog)
    assert "skill polarity changed since install" in warning
    assert pred.system1["mode"] == "shadow"
    assert scored_segments(lm) == dict.fromkeys(ORDER, 1)


def test_apply_system1_loads_the_registry_when_given_none(
    polarity: Skill, registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, polarity, tau=0.0)
    skill = registry.get("sentiment")
    module = build_module(skill)

    assert apply_system1(module, skill, settings) == ("score",)
    assert run(module, DummyLM(answers())).system1["mode"] == "shadow"


def test_a_polarity_skill_without_classify_leaves_sentiment_on_the_llm(
    registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A user skill may shadow the built-in polarity; this one has no `classify`.
    settings = settings_for(monkeypatch, "gate")
    shadowed = SkillRegistry(
        MappingProxyType({"polarity": registry.get("chat"), "sentiment": registry.get("sentiment")})
    )
    module, names = with_student(settings, shadowed)

    assert names == ()
    [warning] = warnings_in(caplog)
    assert "polarity/classify has no installed student" in warning
    assert run(module, DummyLM(answers())).system1 is None


@pytest.mark.parametrize("problem", ["missing", "broken"])
def test_a_polarity_skill_that_cannot_be_built_leaves_sentiment_on_the_llm(
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    problem: str,
) -> None:
    # No polarity skill at all, or a user skill shadowing it whose build fails: fail open.
    settings = settings_for(monkeypatch, "gate")
    skills = {"sentiment": registry.get("sentiment")}
    if problem == "broken":
        skills["polarity"] = registry.get("polarity")

        def broken_build(skill: Skill) -> Any:
            if skill.name == "polarity":
                raise SkillLoadError("polarity: cannot import skill.py")
            return build_module(skill)

        monkeypatch.setattr(cascade, "build_module", broken_build)
    module, names = with_student(settings, SkillRegistry(MappingProxyType(skills)))

    assert names == ()
    [warning] = warnings_in(caplog)
    assert "polarity/classify" in warning
    assert run(module, DummyLM(answers())).system1 is None


# --------------------------------------------------------------------------- concurrency


class WaitingRuntime:
    """A runtime that sees each state whole as 3 tokens and answers `negative`; the answer
    for any segment but the first waits until the first one's LLM call has arrived."""

    def __init__(self, first: str, llm_started: threading.Event) -> None:
        self.tokenizer = SimpleNamespace(state_ids=lambda _state: [0, 0, 0])
        self.first = first
        self.llm_started = llm_started
        self.consulted: list[tuple[str, int, bool]] = []  # (text, thread, LLM had started)

    def fits(self, _state: object, _questions: object, *, n_tokens: int | None = None) -> bool:
        return True

    def predict(self, state: dict[str, str], _questions: object) -> dict[str, Answer]:
        text = state["text"]
        started = text == self.first or self.llm_started.wait(timeout=30)
        self.consulted.append((text, threading.get_ident(), started))
        return {
            "polarity": Answer(
                type="choice",
                keys=LABELS,
                probabilities=(0.7, 0.1, 0.1, 0.1),
                answer_confidence=0.7,
                confidence=0.0,
                encoded=Encoded((), ()),
            )
        }


def test_each_segment_goes_to_the_llm_as_soon_as_the_student_has_decided_it(
    polarity: Skill, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    installed = install(settings, polarity, tau=0.0)
    llm_started = threading.Event()
    runtime = WaitingRuntime(ORDER[0], llm_started)

    class SignallingLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
        def forward(self, prompt: str | None = None, messages: Any = None, **kwargs: Any) -> Any:
            if ORDER[0] in messages[-1]["content"]:
                llm_started.set()
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    module = SentimentModule()
    student = Student(
        installed, derive(Polarity), min_conf=None, runtime_factory=lambda _path: runtime
    )
    module.use_student(student, "shadow", tmp_path / SHADOW_LOG)

    pred = run(module, SignallingLM(answers()))

    # In segment order, on the calling thread, and segment 0's LLM call was already out
    # while the student decided the others.
    assert runtime.consulted == [(s, threading.get_ident(), True) for s in ORDER]
    assert [s["student"]["label"] for s in pred.segments] == ["negative"] * 5


def test_run_jsonl_threads_share_one_student_load(
    polarity: Skill, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, polarity, tau=0.0)
    real = cascade.System1Runtime
    loads: list[Path] = []
    lock = threading.Lock()

    def counting(path: Path) -> System1Runtime:
        with lock:
            loads.append(path)
        return real(path)

    monkeypatch.setattr(cascade, "System1Runtime", counting)
    monkeypatch.setattr(
        "aiagent.cli.run.configure_lm", lambda _s, _m: dspy.configure(lm=DummyLM(answers()))
    )
    rows = tmp_path / "docs.jsonl"
    rows.write_text("".join(json.dumps({"text": TEXT}) + "\n" for _ in range(8)), encoding="utf-8")

    result = runner.invoke(
        app, ["run", "sentiment", "--jsonl", str(rows), "--concurrency", "4"]
    )

    assert result.exit_code == 0, result.output
    assert len(loads) == 1
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == 8
    assert all(o["system1"]["accepted"] == 3 for o in outputs)
    lines = shadow_lines(settings)
    assert len(lines) == 8 * 5
    assert len({line["run_id"] for line in lines}) == 8  # one per document


# --------------------------------------------------------------------------- entry points


def entry_point(command: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The argv of `command` on TEXT as JSON, with its configure_lm faked."""
    monkeypatch.setattr(
        f"aiagent.cli.{command}.configure_lm",
        lambda _s, _m: dspy.configure(lm=DummyLM(answers())),
    )
    if command == "run":
        return ["run", "sentiment", "--text", TEXT, "--json"]
    return ["sentiment", "--text", TEXT, "--json"]


@pytest.mark.parametrize("command", ["sentiment", "run"])
def test_both_entry_points_borrow_the_polarity_student(
    command: str,
    polarity: Skill,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, polarity, tau=0.0)
    pin(monkeypatch, installed)

    result = runner.invoke(app, entry_point(command, monkeypatch))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["system1"]["mode"] == "gate"
    assert payload["system1"]["accepted"] == 3
    assert [s["source"] for s in payload["segments"]] == [
        "student", "llm", "student", "llm", "student"
    ]
    assert warnings_in(caplog) == []


@pytest.mark.parametrize("command", ["sentiment", "run"])
def test_a_missing_student_is_named_polarity_classify(
    command: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings_for(monkeypatch, "shadow")

    result = runner.invoke(app, entry_point(command, monkeypatch))

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["system1"] is None
    [warning] = warnings_in(caplog)
    assert "polarity/classify" in warning
    assert GENERIC_WARNING not in warning


@pytest.mark.parametrize(
    ("mode", "segments", "line"),
    [
        (
            "gate",
            [NEUTRAL[0], OTHER[0], LONG],
            "system 1     : 1/3 segments by the student (gate, polarity {}), 1 too long",
        ),
        (
            "shadow",
            [NEUTRAL[0], OTHER[0], LONG],
            "system 1     : 1/3 segments would be by the student (shadow, polarity {}), "
            "1 too long",
        ),
        (
            "gate",
            [NEUTRAL[0], OTHER[0]],
            "system 1     : 1/2 segments by the student (gate, polarity {})",
        ),
    ],
)
def test_the_human_output_has_a_system_1_line(
    mode: str,
    segments: list[str],
    line: str,
    polarity: Skill,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(monkeypatch, mode)
    installed = install(settings, polarity, tau=0.0)
    pin(monkeypatch, installed)
    monkeypatch.setattr(
        "aiagent.cli.sentiment.configure_lm",
        lambda _s, _m: dspy.configure(lm=DummyLM(answers())),
    )

    result = runner.invoke(app, ["sentiment", "--text", "\n\n".join(segments)])

    assert result.exit_code == 0, result.output
    # The human line names the student by the first 8 characters of its id.
    assert line.format(installed.artifact_id[:8]) in result.stdout.splitlines()


def test_the_human_output_has_no_system_1_line_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aiagent.cli.sentiment.configure_lm",
        lambda _s, _m: dspy.configure(lm=DummyLM(answers())),
    )

    result = runner.invoke(app, ["sentiment", "--text", TEXT])

    assert result.exit_code == 0, result.output
    assert "system 1" not in result.stdout


# --------------------------------------------------------------------------- off mode


OFF_MODE = """
import json, sys
import dspy
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner
import aiagent.cli.sentiment as command
from aiagent.cli.app import app

answer = {"reasoning": "r", "score": "2", "rationale": "ok", "explanation": "Fine."}
command.configure_lm = lambda _s, _m: dspy.configure(lm=DummyLM([answer] * 50))
result = CliRunner().invoke(app, ["sentiment", "--text", sys.argv[1], "--json"])
loaded = sorted(m for m in sys.modules if m.startswith(("aiagent.system1", "onnxruntime")))
print(json.dumps({"exit": result.exit_code, "stdout": result.stdout, "loaded": loaded}))
"""


@pytest.mark.parametrize("mode", [None, "off"])
def test_off_mode_never_reads_artifacts_and_never_imports_system1(
    mode: str | None, tmp_path: Path
) -> None:
    not_a_dir = tmp_path / "artifacts-file"
    not_a_dir.write_text("any access would fail\n", encoding="utf-8")
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),  # no ~/.config/aiagent/config.toml
        "AIAGENT_ARTIFACTS_DIR": str(not_a_dir),
        "AIAGENT_SYSTEM1_MODE": json.dumps({"polarity": "gate"}),
    }
    if mode is not None:
        env["AIAGENT_SYSTEM1_MODE"] = json.dumps({"sentiment": mode, "polarity": "gate"})

    out = subprocess.run(
        [sys.executable, "-c", OFF_MODE, TEXT],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=300,
    )

    report = json.loads(out.stdout.splitlines()[-1])
    assert report["exit"] == 0, report
    assert report["loaded"] == []
    payload = json.loads(report["stdout"])
    assert payload["system1"] is None
    assert all(s["source"] == "llm" and s["student"] is None for s in payload["segments"])
    assert "System 1" not in out.stderr and "artifacts" not in out.stderr  # no warning


def test_off_mode_output_marks_every_segment_as_the_llms() -> None:
    pred = off(DummyLM(answers()))

    assert pred.system1 is None
    assert [s["source"] for s in pred.segments] == ["llm"] * 5
    assert all(s["student"] is None for s in pred.segments)


def test_parallel_forwards_keep_each_documents_lines_together(
    polarity: Skill, registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, polarity, tau=0.0)
    module, _ = with_student(settings, registry)
    lm = DummyLM(answers())

    with ThreadPoolExecutor(max_workers=4) as pool:  # as `run --jsonl` runs documents
        futures = [
            pool.submit(contextvars.copy_context().run, run, module, lm) for _ in range(4)
        ]
        for future in futures:
            future.result(timeout=120)

    run_ids = [line["run_id"] for line in shadow_lines(settings)]
    assert len(run_ids) == 4 * 5
    # A document's lines are written at once: its run_id never interleaves another's.
    blocks = [run_ids[i : i + 5] for i in range(0, 20, 5)]
    assert all(len(set(block)) == 1 for block in blocks)
    assert len({block[0] for block in blocks}) == 4
