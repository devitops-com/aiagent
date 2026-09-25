"""``aiagent run SKILL --jsonl FILE``: many inputs in one process (DummyLM, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM
from typer.testing import CliRunner

from aiagent.cli.app import app

runner = CliRunner()

# DummyLM in dict mode answers by a substring of the prompt, so the answers do not
# depend on which worker thread asks first.
LABELS = {
    "alpha-text": "positive",
    "beta-text": "negative",
    "gamma-text": "neutral",
    "delta-text": "mixed",
    "epsilon-text": "positive",
    "zeta-text": "negative",
}


def _install_dict_lm(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    configured: list[bool] = []

    def fake(_settings: object, _model: object) -> None:
        configured.append(True)
        answers = {text: {"polarity": label} for text, label in LABELS.items()}
        dspy.configure(lm=DummyLM(answers))

    monkeypatch.setattr("aiagent.cli.run.configure_lm", fake)
    return configured


def _jsonl(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "inputs.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _texts_file(tmp_path: Path, texts: list[str]) -> Path:
    return _jsonl(tmp_path, [json.dumps({"text": t}) for t in texts])


def test_run_jsonl_prints_one_prediction_per_line_in_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dict_lm(monkeypatch)
    texts = list(LABELS)
    path = _texts_file(tmp_path, texts)

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path), "--concurrency", "4"])

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["polarity"] for row in rows] == [LABELS[t] for t in texts]


def test_run_jsonl_reads_stdin_and_skips_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dict_lm(monkeypatch)
    stdin = '{"text": "beta-text"}\n\n{"text": "alpha-text"}\n'

    result = runner.invoke(app, ["run", "polarity", "--jsonl", "-"], input=stdin)

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["polarity"] for row in rows] == ["negative", "positive"]


@pytest.mark.parametrize(
    ("bad_line", "why"),
    [("not json", "not valid JSON"), ('["a list"]', "not a JSON object")],
)
def test_run_jsonl_rejects_a_bad_line_before_any_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_line: str, why: str
) -> None:
    configured = _install_dict_lm(monkeypatch)
    path = _jsonl(tmp_path, ['{"text": "alpha-text"}', bad_line])

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path)])

    assert result.exit_code == 1
    assert f"line 2: {why}" in str(result.exception)
    assert configured == []


def test_run_jsonl_refuses_an_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dict_lm(monkeypatch)
    path = _jsonl(tmp_path, [""])

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path)])

    assert result.exit_code == 1
    assert "no inputs" in str(result.exception)


def test_run_jsonl_cannot_be_combined_with_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dict_lm(monkeypatch)
    path = _texts_file(tmp_path, ["alpha-text"])

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path), "--text", "x"])

    assert result.exit_code == 1
    assert "--jsonl cannot be combined" in str(result.exception)


def test_a_failed_row_is_reported_in_place_and_the_rest_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dict_lm(monkeypatch)
    # DummyLM has no answer for "unknown-text", so the adapter cannot parse that row.
    path = _texts_file(tmp_path, ["alpha-text", "unknown-text", "beta-text"])

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path)])

    assert result.exit_code == 1
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["polarity"] == "positive"
    assert set(rows[1]) == {"error"}
    assert rows[2]["polarity"] == "negative"
    assert "1 of 3 inputs failed" in result.stderr


def test_a_line_separator_inside_a_json_string_is_not_a_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # json.dumps(..., ensure_ascii=False) writes U+2028/U+2029/U+0085 raw; they are
    # not JSONL line breaks, so run --jsonl must read its own output back.
    _install_dict_lm(monkeypatch)
    path = tmp_path / "inputs.jsonl"
    line = json.dumps({"text": "alpha-text  \u0085 end"}, ensure_ascii=False)
    path.write_text(line + "\r\n", encoding="utf-8")

    result = runner.invoke(app, ["run", "polarity", "--jsonl", str(path)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["polarity"] == "positive"


@pytest.mark.parametrize("source", ["file", "stdin"])
def test_input_that_is_not_utf8_is_refused_before_any_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    configured = _install_dict_lm(monkeypatch)
    raw = b'{"text": "caf\xe9"}\n'
    path = tmp_path / "latin1.jsonl"
    path.write_bytes(raw)
    args = ["run", "polarity", "--jsonl", str(path) if source == "file" else "-"]

    result = runner.invoke(app, args, input=raw if source == "stdin" else None)

    assert result.exit_code == 1
    assert "not UTF-8" in str(result.exception)
    assert configured == []


def test_ctrl_c_says_what_it_waits_for(capsys: pytest.CaptureFixture[str]) -> None:
    from aiagent.cli.run import _run_batch

    def interrupted(**_inputs: object) -> object:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_batch(interrupted, [{"text": "a"}, {"text": "b"}], 2)
    assert "interrupted" in capsys.readouterr().err


def test_no_input_names_all_three_ways() -> None:
    result = runner.invoke(app, ["run", "polarity"])

    assert result.exit_code == 1
    assert "--jsonl" in str(result.exception)
