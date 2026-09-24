"""The onnxruntime student: parity with laya's torch Agent, decoding, temperatures, loading."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest

from aiagent.exceptions import ArtifactError, QuestionError
from aiagent.system1.runtime import (
    INPUT_NAMES,
    AgentConfig,
    System1Runtime,
    clamp_temperature,
    decode,
    load_agent_config,
    temp_bucket,
)
from aiagent.system1.sequence import Encoded, to_internal
from system1_helpers import FIXTURE, FIXTURE_HEAD_MAX_LEN, FIXTURE_MAX_LEN, golden_rows

TOLERANCE = 1e-3
NUMERIC = ("confidence", "answer_confidence", "score", "noul")
POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": ["negative", "neutral", "mixed", "positive"],
}
FLAT = AgentConfig(max_len=128, head_max_len=96, temperature=(1.0, 1.0, 1.0),
                   temperature_by_options={})


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


def copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    shutil.copytree(FIXTURE, target)
    return target


def row_of(k: int) -> Encoded:
    return Encoded(input_ids=(), markers=tuple(range(k)))


# --------------------------------------------------------------------------- parity with laya


def test_predict_matches_laya_on_every_golden_row(runtime: System1Runtime) -> None:
    checked = 0
    for row in golden_rows():
        if "expected" not in row:
            continue
        answers = runtime.predict(row["state"], row["questions"])
        assert list(answers) == list(row["questions"])
        for qid, want in row["expected"].items():
            where = f"{row['id']}/{qid}"
            got = answers[qid]
            assert list(got.encoded.input_ids) == want["input_ids"], where
            assert list(got.encoded.markers) == want["markers"], where
            laya = {key: value for key, value in want["answer"].items() if key != "legend"}
            mine = got.to_laya()
            assert set(mine) == set(laya), where
            assert mine["type"] == laya["type"], where
            assert mine.get("choice") == laya.get("choice"), where
            for field in NUMERIC:
                if field in laya:
                    assert mine[field] == pytest.approx(laya[field], abs=TOLERANCE), where
            for key, p in laya.get("probabilities", {}).items():
                assert mine["probabilities"][key] == pytest.approx(p, abs=TOLERANCE), where
            checked += 1
    assert checked >= 30


def test_error_rows_raise_question_error(runtime: System1Runtime) -> None:
    errors = [row for row in golden_rows() if "error" in row]
    assert errors
    for row in errors:
        with pytest.raises(QuestionError, match=row["error"]):
            runtime.encode(row["state"], row["questions"])
        with pytest.raises(QuestionError, match=row["error"]):
            runtime.predict(row["state"], row["questions"])


def test_predict_without_questions_does_not_run_the_model(
    runtime: System1Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_session", None)  # any session.run would raise
    assert runtime.predict("good service", {}) == {}


def test_config_and_tokenizer_come_from_the_artifact(runtime: System1Runtime) -> None:
    assert (runtime.config.max_len, runtime.config.head_max_len) == (
        FIXTURE_MAX_LEN,
        FIXTURE_HEAD_MAX_LEN,
    )
    assert runtime.tokenizer.special.mask_token == "<mask>"
    encoded = runtime.encode({"text": "good service"}, {"polarity": POLARITY})
    assert list(encoded) == ["polarity"]
    assert len(encoded["polarity"].markers) == 4


def test_fits(runtime: System1Runtime) -> None:
    questions = {"polarity": POLARITY}
    assert runtime.fits("good service", questions)
    assert runtime.fits({"text": ""}, questions)
    assert not runtime.fits(" ".join(["good"] * FIXTURE_MAX_LEN), questions)


# --------------------------------------------------------------------------- temperatures


@pytest.mark.parametrize(
    ("value", "want"),
    [("2", 2.0), ("x", 1.0), (None, 1.0), (math.nan, 1.0), (math.inf, 1.0),
     (-math.inf, 1.0), (0.1, 0.5), (9, 5.0), (1.3, 1.3)],
)
def test_clamp_temperature(value: object, want: float) -> None:
    assert clamp_temperature(value) == want


@pytest.mark.parametrize(
    ("k", "bucket"),
    [(1, "2"), (2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-10"), (10, "6-10"), (11, "11+"),
     (40, "11+")],
)
def test_temp_bucket(k: int, bucket: str) -> None:
    assert temp_bucket("choice", k) == f"choice:{bucket}"
    assert temp_bucket("score", k) == f"score:{bucket}"


def test_load_agent_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "rl_agent_config.json"
    path.write_text("{}", encoding="utf-8")
    config = load_agent_config(path)
    assert config == AgentConfig(512, 192, (1.0, 1.0, 1.0), {})


def test_load_agent_config_clamps_the_fixture_temperatures() -> None:
    config = load_agent_config(FIXTURE / "rl_agent_config.json")
    assert (config.max_len, config.head_max_len) == (128, 96)
    assert config.temperature == (1.3, 0.8, 1.1)
    assert dict(config.temperature_by_options) == {
        "choice:2": 2.0,
        "choice:11+": 0.5,
        "score:6-10": 5.0,
    }
    assert config.temperature_for("choice", 2) == 2.0
    assert config.temperature_for("choice", 4) == 1.3
    assert config.temperature_for("score", 10) == 5.0
    assert config.temperature_for("noul", 2) == 1.1


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[1, 2]",
        json.dumps({"temperature": [1.0, 1.0]}),
        json.dumps({"temperature": "1.0"}),
        json.dumps({"temperature_by_options": [1.0]}),
        json.dumps({"max_len": "512"}),
        json.dumps({"head_max_len": 0}),
        json.dumps({"max_len": True}),
    ],
    ids=["bad-json", "not-object", "two-temps", "temp-str", "by-options-list", "max-len-str",
         "head-zero", "max-len-bool"],
)
def test_load_agent_config_rejects(tmp_path: Path, content: str) -> None:
    path = tmp_path / "rl_agent_config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactError, match="rl_agent_config.json"):
        load_agent_config(path)


def test_load_agent_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="rl_agent_config.json"):
        load_agent_config(tmp_path / "rl_agent_config.json")


# --------------------------------------------------------------------------- decode


def test_decode_uniform_choice() -> None:
    q = to_internal(POLARITY)
    answer = decode(np.zeros(4, dtype=np.float32), q, row_of(4), FLAT)
    assert answer.probabilities == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert answer.answer_confidence == pytest.approx(0.25)
    assert answer.confidence == pytest.approx(0.0, abs=1e-6)
    assert answer.key == "negative"  # the first argmax
    assert answer.keys == ("negative", "neutral", "mixed", "positive")
    assert answer.to_laya() == {
        "type": "choice",
        "choice": "negative",
        "probabilities": {"negative": 0.25, "neutral": 0.25, "mixed": 0.25, "positive": 0.25},
        "confidence": 0.0,
        "answer_confidence": 0.25,
    }


def test_decode_uses_only_the_first_k_logits_and_the_bucket_temperature() -> None:
    q = to_internal({"type": "choice", "instructions": "late?", "criteria": ["no", "yes"]})
    config = AgentConfig(128, 96, (1.0, 1.0, 1.0), {"choice:2": 2.0})
    logits = np.array([0.0, 2.0, 99.0], dtype=np.float32)  # padding beyond k is ignored
    answer = decode(logits, q, row_of(2), config)
    p_yes = math.e / (1 + math.e)
    assert answer.probabilities == pytest.approx((1 - p_yes, p_yes), abs=1e-6)
    assert answer.key == "yes"
    assert answer.answer_confidence == pytest.approx(p_yes, abs=1e-6)
    with pytest.raises(ValueError, match="score"):
        _ = answer.expected_level


def test_decode_noul_at_one_half_is_true() -> None:
    q = to_internal({"type": "noul", "instructions": "is it late ?"})
    answer = decode(np.zeros(2, dtype=np.float32), q, row_of(2), FLAT)
    assert answer.key == "true"
    assert answer.confidence == pytest.approx(0.5)
    assert answer.to_laya() == {
        "type": "noul", "noul": 0.5, "confidence": 0.5, "answer_confidence": 0.5,
    }
    below = decode(np.array([0.1, 0.0], dtype=np.float32), q, row_of(2), FLAT)
    assert below.key == "false"
    assert below.confidence == pytest.approx(below.answer_confidence)


def test_decode_score_expected_level() -> None:
    q = to_internal({"type": "score", "instructions": "how happy?", "criteria": ["a", "b", "c"]})
    logits = np.log(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    answer = decode(logits, q, row_of(3), FLAT)
    assert answer.probabilities == pytest.approx((1 / 6, 2 / 6, 3 / 6), abs=1e-6)
    assert answer.expected_level == pytest.approx(8 / 6, abs=1e-6)
    assert answer.key == "2"
    laya = answer.to_laya()
    assert laya["score"] == round(8 / 6, 4)
    assert laya["probabilities"] == {"0": 0.1667, "1": 0.3333, "2": 0.5}
    assert "legend" not in laya


def test_decode_single_option() -> None:
    q = to_internal({"type": "choice", "instructions": "pick", "criteria": ["only"]})
    answer = decode(np.array([-3.0], dtype=np.float32), q, row_of(1), FLAT)
    assert answer.probabilities == (1.0,)
    assert answer.answer_confidence == 1.0
    assert answer.confidence == 1.0
    assert answer.key == "only"


# --------------------------------------------------------------------------- loading


def test_missing_model_is_an_artifact_error(tmp_path: Path) -> None:
    model_dir = copy_fixture(tmp_path)
    (model_dir / "model.onnx").unlink()
    with pytest.raises(ArtifactError, match="model.onnx"):
        System1Runtime(model_dir)


def test_corrupt_model_is_an_artifact_error(tmp_path: Path) -> None:
    model_dir = copy_fixture(tmp_path)
    (model_dir / "model.onnx").write_bytes(b"not an onnx graph")
    with pytest.raises(ArtifactError, match="cannot load"):
        System1Runtime(model_dir)


def test_missing_tokenizer_is_an_artifact_error(tmp_path: Path) -> None:
    model_dir = copy_fixture(tmp_path)
    (model_dir / "tokenizer" / "tokenizer.json").unlink()
    with pytest.raises(ArtifactError, match="tokenizer.json"):
        System1Runtime(model_dir)


def fake_io(names: tuple[str, ...]) -> Any:
    return lambda self: [SimpleNamespace(name=name) for name in names]


def test_wrong_input_names_are_an_artifact_error(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (*INPUT_NAMES[:-1], "question_type")
    monkeypatch.setattr(ort.InferenceSession, "get_inputs", fake_io(names))
    with pytest.raises(ArtifactError, match="question_type"):
        System1Runtime(FIXTURE)


def test_input_names_in_another_order_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort.InferenceSession, "get_inputs", fake_io(tuple(reversed(INPUT_NAMES))))
    runtime = System1Runtime(FIXTURE)
    assert runtime.predict("good service", {"polarity": POLARITY})["polarity"].key


def test_no_logits_output_is_an_artifact_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort.InferenceSession, "get_outputs", fake_io(("act_logits",)))
    with pytest.raises(ArtifactError, match="logits"):
        System1Runtime(FIXTURE)


def test_importing_the_runtime_does_not_import_onnxruntime() -> None:
    code = "import aiagent.system1.runtime, sys; print('onnxruntime' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
