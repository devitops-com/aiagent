"""Tests for the built-in ``polarity`` skill (the System 1 distillation pilot)."""

from __future__ import annotations

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner

from aiagent.cli.app import app
from aiagent.config import load_settings
from aiagent.distill.questions import derive
from aiagent.skills.base import SkillSource
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry
from aiagent.skills.router import route

runner = CliRunner()

POLARITY_QUESTIONS = {
    "polarity": {
        "type": "choice",
        "instructions": "Overall sentiment polarity of the passage.",
        "criteria": ["negative", "neutral", "mixed", "positive"],
    }
}
# Computed once (§3.2) and pasted: a change here needs a deliberate DERIVE_VERSION review,
# since it unbinds every student trained for this skill.
SIGNATURE_SHA256 = "19c40da896892a97474ad665f015799e4e70c5392c2c68bf10379b193a8d5113"
QUESTION_SET_SHA256 = "525e19edf9d35d9ae22503a146605e1a286a9691a3ffee91a8ca1399c3877a4e"


def test_registry_lists_polarity_as_builtin() -> None:
    registry, errors = load_registry(load_settings())
    assert registry.get("polarity").source is SkillSource.BUILTIN
    assert errors == []


def test_build_module_has_one_classify_predictor() -> None:
    registry, _ = load_registry(load_settings())
    module = build_module(registry.get("polarity"))
    assert type(module).__name__ == "PolarityModule"
    predictors = module.named_predictors()
    assert [name for name, _ in predictors] == ["classify"]
    assert type(predictors[0][1]) is dspy.Predict


def test_sentiment_request_still_routes_to_sentiment() -> None:
    registry, _ = load_registry(load_settings())
    assert route("sentiment of this article", registry).skill.name == "sentiment"


def test_run_polarity_prints_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=DummyLM([{"polarity": "mixed"}]))

    monkeypatch.setattr("aiagent.cli.run.configure_lm", fake)
    result = runner.invoke(
        app, ["run", "polarity", "--text", "Late delivery, but great support."]
    )
    assert result.exit_code == 0, result.output
    assert "polarity: mixed" in result.stdout


def test_classify_derives_the_pinned_question_and_hashes() -> None:
    registry, _ = load_registry(load_settings())
    module = build_module(registry.get("polarity"))
    derivation = derive(module.classify.signature)
    assert derivation.qualifies, derivation.reasons
    assert derivation.input_field == "text"
    assert not derivation.has_reasoning
    questions = derivation.questions()
    assert questions == POLARITY_QUESTIONS
    assert list(questions["polarity"]) == ["type", "instructions", "criteria"]
    assert derivation.signature_sha256 == SIGNATURE_SHA256
    assert derivation.question_set_sha256 == QUESTION_SET_SHA256
