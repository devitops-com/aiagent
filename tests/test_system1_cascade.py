"""System 1 first: the cascade wrapper and apply_system1, on the fixture student.

Each test installs the fixture for ``polarity/<predictor>`` the way ``distill install``
does (``make_run_dir`` -> ``verify_artifact`` -> ``install_artifact``), bound to the
predictor's derived signature. The LLM is a DummyLM: one without answers fails the call,
so ``lm.history == []`` proves the student answered.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner

from aiagent.builtin_skills.polarity.skill import Polarity
from aiagent.cli.app import app
from aiagent.config import Settings, load_settings
from aiagent.distill.dataset import build_row
from aiagent.distill.questions import derive
from aiagent.distill.splits import document_id
from aiagent.skills.base import Skill
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry
from aiagent.system1.artifacts import (
    SHADOW_LOG,
    InstalledArtifact,
    artifact_home,
    install_artifact,
    verify_artifact,
)
from aiagent.system1.cascade import (
    REASONING_MARKER,
    System1First,
    apply_system1,
    student_state,
)
from aiagent.system1.contract import content_sha256
from aiagent.system1.runtime import System1Runtime
from aiagent.system1.sequence import SequenceTokenizer
from system1_helpers import FIXTURE, FIXTURE_HEAD_MAX_LEN, FIXTURE_MAX_LEN, make_run_dir

runner = CliRunner()

DATASET = "d" * 64
TEXT = "Die Lieferung kam zu spät, aber der Support war hervorragend."
LONG_TEXT = "word " * 200  # far more tokens than the fixture's max_len (128)
CASCADE_LOGGER = "aiagent.system1.cascade"
LABELS = Literal["negative", "neutral", "mixed", "positive"]


class OtherPolarity(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Classify a passage."""  # other instructions: another signature hash

    text: str = dspy.InputField(desc="A passage of text to assess.")
    polarity: LABELS = dspy.OutputField(desc="Overall sentiment polarity of the passage.")


class Reasoned(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """The qualifying predictor sits at ``gen.predict`` (a ChainOfThought)."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = dspy.ChainOfThought(Polarity)

    def forward(self, text: str) -> Any:
        return self.gen(text=text)


class Listed(dspy.Module):  # type: ignore[misc]  # dspy ships no stubs
    """The predictor sits in a list (``steps[0]``), which cannot be replaced by name."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = [dspy.Predict(Polarity)]

    def forward(self, text: str) -> Any:
        return self.steps[0](text=text)


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


@pytest.fixture
def skill() -> Skill:
    registry, _ = load_registry(load_settings())
    return registry.get("polarity")


def settings_for(
    monkeypatch: pytest.MonkeyPatch, mode: str | None, *, min_conf: float | None = None
) -> Settings:
    if mode is not None:
        monkeypatch.setenv("AIAGENT_SYSTEM1_MODE", json.dumps({"polarity": mode}))
    if min_conf is not None:
        monkeypatch.setenv("AIAGENT_SYSTEM1_MIN_CONF", str(min_conf))
    return load_settings()


def install(
    settings: Settings,
    skill: Skill,
    *,
    predictor: str = "classify",
    signature: Any = Polarity,
    tau: float = 0.0,
    source: str | None = None,
) -> InstalledArtifact:
    """Install the fixture student for polarity/<predictor>, bound to `signature`."""
    binds = derive(signature).binds(skill)
    job = f"ftjob-{predictor}"
    run_dir = make_run_dir(
        settings.distill_dir, job, binds=binds.to_json(), dataset_manifest_sha256=DATASET
    )
    verified = verify_artifact(
        run_dir,
        expected=binds,
        dataset_manifest_sha256=DATASET,
        max_len=FIXTURE_MAX_LEN,
        head_max_len=FIXTURE_HEAD_MAX_LEN,
    )
    return install_artifact(
        run_dir,
        files=verified.files,
        artifacts_dir=settings.artifacts_dir,
        skill=skill.name,
        predictor=predictor,
        run=job,
        thresholds={"polarity": tau},
        target_precision=0.95,
        skill_source_sha256=source or binds.skill_source_sha256,
    )


def student_key(runtime: System1Runtime, text: str = TEXT) -> str:
    return runtime.predict({"text": text}, derive(Polarity).questions())["polarity"].key


def cascade_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == CASCADE_LOGGER and r.levelno == logging.WARNING
    ]


def shadow_lines(settings: Settings, predictor: str = "classify") -> list[dict[str, Any]]:
    path = artifact_home(settings.artifacts_dir, "polarity", predictor) / SHADOW_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- apply_system1


@pytest.mark.parametrize("mode", [None, "off"])
def test_off_wraps_nothing_and_never_reads_artifacts_dir(
    mode: str | None,
    skill: Skill,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    not_a_dir = tmp_path / "artifacts-file"
    not_a_dir.write_text("any access would fail\n", encoding="utf-8")
    monkeypatch.setenv("AIAGENT_ARTIFACTS_DIR", str(not_a_dir))
    settings = settings_for(monkeypatch, mode)
    module = build_module(skill)
    predictor = module.classify
    with caplog.at_level(logging.DEBUG):
        assert apply_system1(module, skill, settings) == ()
    assert module.classify is predictor
    assert caplog.records == []


def test_gate_answers_with_the_student_when_confident(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "gate")
    install(settings, skill, tau=0.0)
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ("classify",)
    assert isinstance(module.classify, System1First)
    lm = DummyLM([])
    with dspy.context(lm=lm):
        prediction = module(text=TEXT)
    assert lm.history == []
    assert dict(prediction.items()) == {"polarity": student_key(runtime)}


@pytest.mark.parametrize(("tau", "min_conf"), [(1.0, None), (0.0, 1.0)])
def test_gate_asks_the_llm_below_the_threshold(
    tau: float, min_conf: float | None, skill: Skill, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "gate", min_conf=min_conf)
    install(settings, skill, tau=tau)
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ("classify",)
    lm = DummyLM([{"polarity": "neutral"}])
    with dspy.context(lm=lm):
        prediction = module(text=TEXT)
    assert len(lm.history) == 1
    assert prediction.polarity == "neutral"


def test_gate_asks_the_llm_when_the_text_does_not_fit(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not runtime.fits({"text": LONG_TEXT}, derive(Polarity).questions())
    settings = settings_for(monkeypatch, "gate")
    install(settings, skill, tau=0.0)
    module = build_module(skill)
    apply_system1(module, skill, settings)
    lm = DummyLM([{"polarity": "positive"}])
    with dspy.context(lm=lm):
        prediction = module(text=LONG_TEXT)
    assert len(lm.history) == 1
    assert prediction.polarity == "positive"


def test_shadow_returns_the_llm_answer_and_logs_the_student(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    installed = install(settings, skill, tau=0.0)
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ("classify",)
    lm = DummyLM([{"polarity": "negative"}])
    with dspy.context(lm=lm):
        prediction = module(text=TEXT)
    assert prediction.polarity == "negative"
    [line] = shadow_lines(settings)
    assert list(line) == [
        "ts", "artifact_id", "input_sha256", "student", "llm", "confidence",
        "would_accept", "agree", "student_ms",
    ]
    # Rows of a `run --jsonl` batch log in completion order: the hash of the input
    # text lets them be matched back to their inputs without logging any text.
    assert line["input_sha256"] == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    key = student_key(runtime)
    assert line["artifact_id"] == installed.artifact_id
    assert line["student"] == {"polarity": key}
    assert line["llm"] == {"polarity": "negative"}
    assert 0.0 < line["confidence"]["polarity"] <= 1.0
    assert line["would_accept"] is True
    assert line["agree"] is (key == "negative")
    assert isinstance(line["student_ms"], int)
    assert line["ts"].endswith("Z")
    assert "Lieferung" not in json.dumps(line)  # no input text


def test_shadow_logs_nothing_when_the_text_does_not_fit(
    skill: Skill, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, skill, tau=0.0)
    module = build_module(skill)
    apply_system1(module, skill, settings)
    with dspy.context(lm=DummyLM([{"polarity": "mixed"}])):
        assert module(text=LONG_TEXT).polarity == "mixed"
    home = artifact_home(settings.artifacts_dir, "polarity", "classify")
    assert not (home / SHADOW_LOG).exists()


def test_the_student_loads_once(skill: Skill, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, skill, tau=0.0)
    loads: list[Path] = []

    def counting(path: Path) -> System1Runtime:
        loads.append(path)
        return System1Runtime(path)

    wrapper = System1First(
        dspy.Predict(Polarity),
        derivation=derive(Polarity),
        installed=installed,
        mode="gate",
        min_conf=None,
        shadow_log=None,
        runtime_factory=counting,
    )
    lm = DummyLM([])
    with dspy.context(lm=lm):
        first, second = wrapper(text=TEXT), wrapper(text="Great support.")
    assert loads == [installed.path]
    assert lm.history == []
    assert {first.polarity, second.polarity} <= {"negative", "neutral", "mixed", "positive"}


def test_shadow_log_write_failure_is_ignored(
    skill: Skill,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    installed = install(settings, skill, tau=0.0)
    wrapper = System1First(
        dspy.Predict(Polarity),
        derivation=derive(Polarity),
        installed=installed,
        mode="shadow",
        min_conf=None,
        shadow_log=tmp_path / "missing-dir" / SHADOW_LOG,
    )
    with dspy.context(lm=DummyLM([{"polarity": "mixed"}])):
        assert wrapper(text=TEXT).polarity == "mixed"
    [warning] = cascade_warnings(caplog)
    assert "shadow log" in warning


def test_a_failing_student_falls_back_to_the_llm_once_and_for_all(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, skill, tau=0.0)
    calls: list[Path] = []

    def failing(path: Path) -> System1Runtime:
        calls.append(path)
        raise RuntimeError("no onnxruntime here")

    wrapper = System1First(
        dspy.Predict(Polarity),
        derivation=derive(Polarity),
        installed=installed,
        mode="gate",
        min_conf=None,
        shadow_log=None,
        runtime_factory=failing,
    )
    lm = DummyLM([{"polarity": "neutral"}, {"polarity": "positive"}])
    with dspy.context(lm=lm):
        assert wrapper(text=TEXT).polarity == "neutral"
        assert wrapper(text=TEXT).polarity == "positive"
    assert calls == [installed.path]
    [warning] = cascade_warnings(caplog)
    assert "no onnxruntime here" in warning


@pytest.mark.parametrize("signature", [OtherPolarity, "text -> polarity"])
def test_hard_bind_mismatch_is_not_wrapped(
    signature: Any,
    skill: Skill,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = settings_for(monkeypatch, "gate")
    install(settings, skill, tau=0.0)
    module = build_module(skill)
    module.classify = dspy.Predict(signature)
    assert apply_system1(module, skill, settings) == ()
    assert not isinstance(module.classify, System1First)
    assert any("unbound" in w for w in cascade_warnings(caplog))


def test_soft_mismatch_in_gate_mode_only_shadows(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "gate")
    install(settings, skill, tau=0.0, source="0" * 64)
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ("classify",)
    assert module.classify.mode == "shadow"
    [warning] = cascade_warnings(caplog)
    assert "aiagent distill eval" in warning
    lm = DummyLM([{"polarity": "neutral"}])
    with dspy.context(lm=lm):
        assert module(text=TEXT).polarity == "neutral"
    assert len(shadow_lines(settings)) == 1


def test_broken_stamp_is_not_wrapped(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, skill, tau=0.0)
    os.utime(installed.path / "model.onnx", ns=(1, 1))
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ()
    assert any("changed since install" in w for w in cascade_warnings(caplog))


def test_no_installed_student_wraps_nothing_with_one_warning(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    module = build_module(skill)
    assert apply_system1(module, skill, settings) == ()
    [warning] = cascade_warnings(caplog)
    assert "polarity" in warning


def test_chain_of_thought_predictor_is_wrapped_at_its_dotted_path(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "gate")
    module = Reasoned()
    [(name, predictor)] = module.named_predictors()
    assert name == "gen.predict"
    install(settings, skill, predictor=name, signature=predictor.signature, tau=0.0)
    assert apply_system1(module, skill, settings) == ("gen.predict",)
    assert isinstance(module.gen.predict, System1First)
    lm = DummyLM([])
    with dspy.context(lm=lm):
        prediction = module(text=TEXT)
    assert lm.history == []
    assert prediction.reasoning == REASONING_MARKER
    assert prediction.polarity == student_key(runtime)


def test_predictor_in_a_list_is_not_wrapped(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "gate")
    module = Listed()
    assert apply_system1(module, skill, settings) == ()
    assert any("steps[0]" in w for w in cascade_warnings(caplog))


# --------------------------------------------------------------------------- student_state


def test_student_state_is_what_training_rows_tokenize(runtime: System1Runtime) -> None:
    derivation = derive(Polarity)
    questions = derivation.questions()
    state = student_state(derivation, {"text": TEXT, "config": {"temperature": 0.0}})
    assert state == {"text": TEXT}
    served = runtime.encode(state, questions)["polarity"].input_ids
    row = build_row(
        group_id=document_id(TEXT),
        index=0,
        text=TEXT,
        input_field="text",
        questions=questions,
        split="train",
        tokenizer=SequenceTokenizer.from_dir(FIXTURE / "tokenizer"),
        max_len=FIXTURE_MAX_LEN,
        head_max_len=FIXTURE_HEAD_MAX_LEN,
    )
    assert content_sha256(list(served)) == row.student_tokens["polarity"].ids_sha256


@pytest.mark.parametrize("kwargs", [{"text": 3}, {"text": None}, {}, {"passage": TEXT}])
def test_student_state_is_none_without_a_str_input(kwargs: dict[str, Any]) -> None:
    assert student_state(derive(Polarity), kwargs) is None


def test_student_state_is_none_for_an_unqualified_signature() -> None:
    assert student_state(derive(dspy.Signature("a, b -> c")), {"a": TEXT}) is None


# --------------------------------------------------------------------------- CLI


def test_run_in_gate_mode_answers_without_the_llm(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "gate")
    install(settings, skill, tau=0.0)
    lm = DummyLM([])

    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=lm)

    monkeypatch.setattr("aiagent.cli.run.configure_lm", fake)
    result = runner.invoke(app, ["run", "polarity", "--text", TEXT])
    assert result.exit_code == 0, result.output
    assert f"polarity: {student_key(runtime)}" in result.stdout
    assert lm.history == []


def test_the_student_loads_once_across_threads(
    skill: Skill, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `run --jsonl` calls one wrapper from several threads at once.
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, skill, tau=0.0)
    loads: list[Path] = []

    def slow_counting(path: Path) -> System1Runtime:
        loads.append(path)
        time.sleep(0.05)  # a real load takes ~0.85 s: every thread arrives before it ends
        return System1Runtime(path)

    wrapper = System1First(
        dspy.Predict(Polarity),
        derivation=derive(Polarity),
        installed=installed,
        mode="gate",
        min_conf=None,
        shadow_log=None,
        runtime_factory=slow_counting,
    )
    with dspy.context(lm=DummyLM([])), ThreadPoolExecutor(max_workers=8) as pool:
        # dspy keeps context overrides in contextvars: copy them per task, as run does.
        futures = [
            pool.submit(contextvars.copy_context().run, lambda: wrapper(text=TEXT).polarity)
            for _ in range(8)
        ]
        answers = [future.result() for future in futures]
    assert loads == [installed.path]
    assert len(set(answers)) == 1


def test_shadow_logs_one_whole_line_per_call_across_threads(
    skill: Skill, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, skill, tau=0.0)
    module = build_module(skill)
    apply_system1(module, skill, settings)
    lm = DummyLM([{"polarity": "negative"} for _ in range(32)])
    with dspy.context(lm=lm), ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, lambda: module(text=TEXT))
            for _ in range(32)
        ]
        for future in futures:
            future.result()
    lines = shadow_lines(settings)
    assert len(lines) == 32
    assert all(line["llm"] == {"polarity": "negative"} for line in lines)


def test_a_failing_student_load_is_tried_and_warned_about_once_across_threads(
    skill: Skill, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(monkeypatch, "gate")
    installed = install(settings, skill, tau=0.0)
    loads: list[Path] = []

    def slow_failing(path: Path) -> System1Runtime:
        loads.append(path)
        time.sleep(0.05)
        raise OSError("disk gone")

    wrapper = System1First(
        dspy.Predict(Polarity),
        derivation=derive(Polarity),
        installed=installed,
        mode="gate",
        min_conf=None,
        shadow_log=None,
        runtime_factory=slow_failing,
    )
    lm = DummyLM([{"polarity": "negative"} for _ in range(8)])
    with dspy.context(lm=lm), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, lambda: wrapper(text=TEXT).polarity)
            for _ in range(8)
        ]
        answers = [future.result() for future in futures]
    assert loads == [installed.path]
    assert len(cascade_warnings(caplog)) == 1
    assert answers == ["negative"] * 8


def test_batch_shadow_lines_match_their_inputs_by_hash(
    skill: Skill, runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = settings_for(monkeypatch, "shadow")
    install(settings, skill, tau=0.0)
    texts = [f"{TEXT} ({i})" for i in range(12)]
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")

    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=DummyLM([{"polarity": "mixed"} for _ in range(12)]))

    monkeypatch.setattr("aiagent.cli.run.configure_lm", fake)
    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(inputs), "--concurrency", "4"])

    assert result.exit_code == 0, result.output
    by_hash = {line["input_sha256"]: line for line in shadow_lines(settings)}
    assert set(by_hash) == {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts}
