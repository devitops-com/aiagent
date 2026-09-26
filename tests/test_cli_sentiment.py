"""CLI tests for ``aiagent sentiment`` using DSPy's DummyLM (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner

from aiagent.cli.app import app
from aiagent.ingest.sources import SourceDoc

runner = CliRunner()

_RESP = {
    "reasoning": "upbeat",
    "score": "6",
    "rationale": "clearly positive tone",
    "explanation": "The content is positive and fairly consistent throughout.",
}


def _install_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=DummyLM([dict(_RESP) for _ in range(800)]))

    monkeypatch.setattr("aiagent.cli.sentiment.configure_lm", fake)


def test_sentiment_text_human(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    result = runner.invoke(
        app, ["sentiment", "--text", "Great launch. Happy team. Smooth rollout."]
    )
    assert result.exit_code == 0, result.stdout
    assert "sentiment    : +6.00" in result.stdout
    assert "positive" in result.stdout
    assert "uncertainty  : n/a (no segment has two readable scores)" in result.stdout
    assert "consistent" in result.stdout  # explanation echoed


def test_sentiment_human_with_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    result = runner.invoke(
        app,
        ["sentiment", "--text", "Great launch. Happy team. Smooth rollout.", "--resample", "2"],
    )
    assert result.exit_code == 0, result.stdout
    assert (
        "uncertainty  : 0.00  (model spread over 3 resampled segments)"
        in result.stdout
    )


def test_sentiment_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    result = runner.invoke(
        app, ["sentiment", "--text", "A. B. C. D.", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["sentiment"] == 6.0
    assert payload["polarity"] == "very positive"
    assert payload["sources"] == ["text"]
    assert "volatility" in payload
    assert "significance_p" in payload
    assert payload["n_segments"] == 4
    # The default --resample 1: one LLM sample per segment, so no model uncertainty.
    assert payload["model_uncertainty"] is None
    assert payload["n_resampled"] == 0
    assert payload["n_samples"] == 4


def test_sentiment_json_with_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    result = runner.invoke(
        app, ["sentiment", "--text", "A. B. C. D.", "--resample", "3", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["model_uncertainty"] == 0.0  # the dummy answers alike every time
    assert (payload["n_resampled"], payload["n_samples"]) == (4, 12)


def test_cli_default_resample_mirrors_the_module() -> None:
    from aiagent.cli.sentiment import _DEFAULT_MAX_SEGMENTS, _DEFAULT_RESAMPLE
    from aiagent.core.sentiment import DEFAULT_MAX_SEGMENTS, DEFAULT_RESAMPLE

    assert _DEFAULT_RESAMPLE == DEFAULT_RESAMPLE == 1
    assert _DEFAULT_MAX_SEGMENTS == DEFAULT_MAX_SEGMENTS


def test_sentiment_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_dummy(monkeypatch)
    path = tmp_path / "review.txt"
    path.write_text("Lovely. Wonderful. Superb.", encoding="utf-8")
    result = runner.invoke(app, ["sentiment", "--file", str(path), "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert str(path) in payload["sources"]


def test_sentiment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    monkeypatch.setattr(
        "aiagent.cli.sentiment.fetch_source",
        lambda url, settings: SourceDoc(origin=url, text="Fantastic. Amazing. Great."),
    )
    result = runner.invoke(
        app, ["sentiment", "--url", "https://example.com/post", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["sources"] == ["https://example.com/post"]


def test_sentiment_requires_a_source() -> None:
    result = runner.invoke(app, ["sentiment"])
    assert result.exit_code == 1
    assert "at least one" in str(result.exception)


def test_run_sentiment_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    # The generic runner also drives the skill from a single text blob.
    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=DummyLM([dict(_RESP) for _ in range(800)]))

    monkeypatch.setattr("aiagent.cli.run.configure_lm", fake)
    result = runner.invoke(app, ["run", "sentiment", "--text", "Good. Fine. Nice."])
    assert result.exit_code == 0, result.stdout
    assert "sentiment:" in result.stdout

    result = runner.invoke(
        app, ["run", "sentiment", "--text", "Good. Fine. Nice.", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert (payload["model_uncertainty"], payload["n_resampled"]) == (None, 0)
    assert payload["n_samples"] == payload["n_segments"] == 3


def test_sentiment_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch)
    result = runner.invoke(app, ["sentiment", "--text", "Good. Bad. Great.", "-v"])
    assert result.exit_code == 0
    assert "[-v] skill=sentiment" in result.stderr
