"""Unit tests for the System 1 contract primitives (hashing, ids, SHA256SUMS)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import pytest

from aiagent.system1.contract import (
    SUMS_FILE,
    Binds,
    bytes_sha256,
    canonical_json,
    content_sha256,
    dataset_id,
    file_sha256,
    format_sha256sums,
    parse_sha256sums,
    safe_id,
    safe_relpath,
)
from system1_helpers import ARTIFACT_FILES, make_base_checkpoint, make_run_dir, resum

HEX_A = "a" * 64
HEX_B = "0123456789abcdef" * 4


def test_canonical_json_keeps_key_order_and_writes_non_ascii_literally() -> None:
    assert canonical_json({"b": 1, "a": [1, "ž"]}) == '{"b":1,"a":[1,"ž"]}'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": value})


def test_content_sha256_hashes_the_compact_utf8_json() -> None:
    assert content_sha256([1, 2, 3]) == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert content_sha256({"k": "ž"}) == hashlib.sha256('{"k":"ž"}'.encode()).hexdigest()


def test_file_sha256_reads_in_chunks(tmp_path: Path) -> None:
    data = bytes(range(256)) * (3 * 4096 + 7)  # just over 3 MiB: several 1 MiB chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert file_sha256(path) == bytes_sha256(data) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("value", ["ftjob-abc_1.2", "a", "ds-0123456789ab", "A" * 128])
def test_safe_id_accepts(value: str) -> None:
    assert safe_id(value, "job id") == value


@pytest.mark.parametrize("value", ["../x", "a/b", "", ".x", "-x", "a b", "A" * 129, "a\n"])
def test_safe_id_rejects_naming_what(value: str) -> None:
    with pytest.raises(ValueError, match="job id"):
        safe_id(value, "job id")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("model.onnx", "model.onnx"), ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json")],
)
def test_safe_relpath_accepts(value: str, expected: str) -> None:
    assert safe_relpath(value) == PurePosixPath(expected)


@pytest.mark.parametrize(
    "value", ["/a", "a/../b", "a//b", "..", "", ".", "a/", "./a", "a/./b", "a\\b", "a\x00b"]
)
def test_safe_relpath_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        safe_relpath(value)


def test_sha256sums_round_trip_sorted_with_two_spaces() -> None:
    entries = {"tokenizer/tokenizer.json": HEX_B, "model.onnx": HEX_A}
    text = format_sha256sums(entries)
    assert text == f"{HEX_A}  model.onnx\n{HEX_B}  tokenizer/tokenizer.json\n"
    assert parse_sha256sums(text) == entries
    assert list(parse_sha256sums(text)) == ["model.onnx", "tokenizer/tokenizer.json"]


def test_sha256sums_accepts_the_binary_star_form_and_blank_lines() -> None:
    assert parse_sha256sums(f"{HEX_A} *model.onnx\n\n{HEX_B}  a.json") == {
        "model.onnx": HEX_A,
        "a.json": HEX_B,
    }


@pytest.mark.parametrize(
    "text",
    [
        f"{HEX_A.upper()}  model.onnx\n",  # upper-case hex
        f"{HEX_A[:-1]}  model.onnx\n",  # 63 chars
        f"{HEX_A} model.onnx\n",  # one space, no star
        f"{HEX_A}  ../model.onnx\n",  # unsafe path
        f"{HEX_A}  /etc/passwd\n",
        f"{HEX_A}  \n",  # no path
        f"{HEX_A}  model.onnx\n{HEX_B} *model.onnx\n",  # duplicate path
        "garbage\n",
    ],
)
def test_sha256sums_parse_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_sha256sums(text)


def test_format_sha256sums_rejects_bad_entries() -> None:
    with pytest.raises(ValueError):
        format_sha256sums({"../x": HEX_A})
    with pytest.raises(ValueError):
        format_sha256sums({"x": "nothex"})


def test_dataset_id() -> None:
    assert dataset_id(HEX_B) == "ds-0123456789ab"
    with pytest.raises(ValueError):
        dataset_id("0123")


def test_binds_to_json_key_order() -> None:
    binds = Binds(signature_sha256="s", question_set_sha256="q", skill_source_sha256="k")
    assert list(binds.to_json().items()) == [
        ("signature_sha256", "s"),
        ("question_set_sha256", "q"),
        ("skill_source_sha256", "k"),
    ]


def test_sums_file_name() -> None:
    assert SUMS_FILE == "SHA256SUMS"


# --------------------------------------------------------------------------- the test helpers


def test_make_run_dir_sums_exactly_the_artifact_files(tmp_path: Path) -> None:
    binds = Binds("s" * 64, "q" * 64, "k" * 64).to_json()
    run_dir = make_run_dir(tmp_path, "ftjob-1", binds=binds, dataset_manifest_sha256=HEX_B)
    sums = parse_sha256sums((run_dir / SUMS_FILE).read_text(encoding="utf-8"))
    assert sorted(sums) == sorted(["manifest.json", *ARTIFACT_FILES])
    assert all(file_sha256(run_dir / rel) == digest for rel, digest in sums.items())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["binds"]) == [*binds, "dataset_manifest_sha256"]
    assert manifest["files"]["model.onnx"]["size"] == (run_dir / "model.onnx").stat().st_size
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    assert (job["id"], job["status"], job["training_file"]) == ("ftjob-1", "succeeded", "ds-0123456789ab")
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2

    (run_dir / "golden.jsonl").write_text("{}\n", encoding="utf-8")
    resum(run_dir)
    sums = parse_sha256sums((run_dir / SUMS_FILE).read_text(encoding="utf-8"))
    assert sums["golden.jsonl"] == file_sha256(run_dir / "golden.jsonl")
    assert sums["manifest.json"] == file_sha256(run_dir / "manifest.json")


def test_resum_updates_row_counts_and_plain_sums(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    manifest = {"files": {"train.jsonl": {"sha256": "", "rows": 0}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    resum(tmp_path)
    entry = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["files"]
    assert entry == {"train.jsonl": {"sha256": file_sha256(tmp_path / "train.jsonl"), "rows": 2}}

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("new", encoding="utf-8")
    (plain / SUMS_FILE).write_text(f"{HEX_A}  a.txt\n", encoding="utf-8")
    resum(plain)
    assert parse_sha256sums((plain / SUMS_FILE).read_text(encoding="utf-8")) == {
        "a.txt": file_sha256(plain / "a.txt")
    }

    base = make_base_checkpoint(tmp_path / "distill")
    assert base == tmp_path / "distill" / "base" / "laya-multilingual@0000test"
    for rel in ("rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json"):
        assert (base / rel).is_file()
