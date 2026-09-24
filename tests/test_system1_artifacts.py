"""Trained-student artifacts: verification (hashes, binds, limits, golden) and installation."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aiagent.exceptions import ArtifactError, ArtifactNotInstalledError
from aiagent.system1.artifacts import (
    CURRENT_FILE,
    INSTALL_FILE,
    InstalledArtifact,
    ManifestBinds,
    VerifiedArtifact,
    artifact_home,
    check_golden,
    install_artifact,
    load_installed,
    read_manifest,
    verify_artifact,
)
from aiagent.system1.contract import SUMS_FILE, Binds, file_sha256, format_sha256sums
from aiagent.system1.runtime import System1Runtime
from system1_helpers import (
    ARTIFACT_FILES,
    FIXTURE,
    FIXTURE_HEAD_MAX_LEN,
    FIXTURE_MAX_LEN,
    MANIFEST,
    golden_rows,
    make_run_dir,
    resum,
)

BINDS = {
    "signature_sha256": "a" * 64,
    "question_set_sha256": "b" * 64,
    "skill_source_sha256": "c" * 64,
}
DATASET = "d" * 64
EXPECTED = Binds(**BINDS)
CURRENT_SOURCE = "e" * 64
ARTIFACT_ID = "a-0123456789ab"
OTHER_ID = "a-bbbbbbbbbbbb"
LIMITS = {"max_len": FIXTURE_MAX_LEN, "head_max_len": FIXTURE_HEAD_MAX_LEN}
POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": ["negative", "neutral", "mixed", "positive"],
}


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


def new_run(tmp_path: Path, job: str = "ftjob-1", **kwargs: Any) -> Path:
    return make_run_dir(
        tmp_path / "distill", job, binds=BINDS, dataset_manifest_sha256=DATASET, **kwargs
    )


def verify(run_dir: Path, *, expected: Binds = EXPECTED, dataset: str = DATASET,
           max_len: int = FIXTURE_MAX_LEN) -> VerifiedArtifact:
    return verify_artifact(run_dir, expected=expected, dataset_manifest_sha256=dataset,
                           max_len=max_len, head_max_len=FIXTURE_HEAD_MAX_LEN)


def install(run_dir: Path, files: Mapping[str, str], artifacts_dir: Path) -> InstalledArtifact:
    return install_artifact(
        run_dir,
        files=files,
        artifacts_dir=artifacts_dir,
        skill="polarity",
        predictor="classify",
        run=run_dir.name,
        thresholds={"polarity": 0.83},
        target_precision=0.95,
        skill_source_sha256=CURRENT_SOURCE,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def edit_manifest(run_dir: Path, **changes: Any) -> None:
    write_json(run_dir / MANIFEST, {**load_json(run_dir / MANIFEST), **changes})


def write_sums(run_dir: Path, paths: list[str], extra: str = "") -> None:
    """SHA256SUMS over `paths` with their real hashes, leaving manifest.files alone."""
    sums = {rel: file_sha256(run_dir / rel) for rel in paths}
    (run_dir / SUMS_FILE).write_text(format_sha256sums(sums) + extra, encoding="utf-8")


def all_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- verify: ok


def test_verify_ok(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    verified = verify(run_dir)
    assert verified.run_dir == run_dir
    assert verified.manifest.artifact_id == ARTIFACT_ID
    assert verified.manifest.binds == ManifestBinds(**BINDS, dataset_manifest_sha256=DATASET)
    assert verified.source_matches is True
    assert set(verified.files) == {MANIFEST, *ARTIFACT_FILES}
    for rel, digest in verified.files.items():
        assert digest == file_sha256(run_dir / rel)
    answer = verified.runtime.predict({"text": "good service"}, {"polarity": POLARITY})
    assert answer["polarity"].key in POLARITY["criteria"]


def test_soft_source_mismatch_only_clears_source_matches(tmp_path: Path) -> None:
    changed = Binds(BINDS["signature_sha256"], BINDS["question_set_sha256"], CURRENT_SOURCE)
    assert verify(new_run(tmp_path), expected=changed).source_matches is False


def test_job_bookkeeping_is_never_verified(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    job = load_json(run_dir / "job.json")
    write_json(run_dir / "job.json", {**job, "status": "cancelled"})
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"object":"fine_tuning.job.event","message":"late"}\n')
    (run_dir / "train.log").write_text("step 1\n", encoding="utf-8")
    assert verify(run_dir).manifest.artifact_id == ARTIFACT_ID


def test_read_manifest_ok(tmp_path: Path) -> None:
    manifest = read_manifest(new_run(tmp_path))
    assert manifest.format_version == 1
    assert manifest.laya.version == "0.3.20"
    assert manifest.golden.n == len(golden_rows())
    assert set(manifest.files) == set(ARTIFACT_FILES)


# --------------------------------------------------------------------------- verify: rejected


@pytest.mark.parametrize("field", ["signature_sha256", "question_set_sha256"])
def test_hard_bind_mismatch(tmp_path: Path, field: str) -> None:
    expected = Binds(**{**BINDS, field: "f" * 64})
    with pytest.raises(ArtifactError, match="unbound"):
        verify(new_run(tmp_path), expected=expected)


def test_dataset_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="dataset"):
        verify(new_run(tmp_path), dataset="f" * 64)


def test_max_len_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="max_len"):
        verify(new_run(tmp_path), max_len=256)


def test_flipped_byte_names_the_file(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    data = bytearray((run_dir / "model.onnx.data").read_bytes())
    data[1000] ^= 0xFF
    (run_dir / "model.onnx.data").write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match=r"model\.onnx\.data"):
        verify(run_dir)


def test_required_file_not_in_the_artifact(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    manifest = load_json(run_dir / MANIFEST)
    del manifest["files"]["golden.jsonl"]
    write_json(run_dir / MANIFEST, manifest)
    resum(run_dir)
    with pytest.raises(ArtifactError, match="golden.jsonl"):
        verify(run_dir)


def test_artifact_file_not_listed_in_sums(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    write_sums(run_dir, [MANIFEST, *(rel for rel in ARTIFACT_FILES if rel != "golden.jsonl")])
    with pytest.raises(ArtifactError, match=f"{SUMS_FILE} does not list 'golden.jsonl'"):
        verify(run_dir)


def test_missing_artifact_file(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    (run_dir / "rl_agent_config.json").unlink()
    with pytest.raises(ArtifactError, match="rl_agent_config.json"):
        verify(run_dir)


@pytest.mark.parametrize("field", ["sha256", "size"])
def test_manifest_files_disagree_with_sums(tmp_path: Path, field: str) -> None:
    run_dir = new_run(tmp_path)
    manifest = load_json(run_dir / MANIFEST)
    entry = manifest["files"]["rl_agent_config.json"]
    entry[field] = "0" * 64 if field == "sha256" else entry["size"] + 1
    write_json(run_dir / MANIFEST, manifest)
    write_sums(run_dir, [MANIFEST, *ARTIFACT_FILES])  # the sums themselves are right
    with pytest.raises(ArtifactError, match="rl_agent_config.json"):
        verify(run_dir)


def test_sums_mismatch_on_manifest(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    edit_manifest(run_dir, note="edited after packaging")
    with pytest.raises(ArtifactError, match="manifest.json"):
        verify(run_dir)


def test_parent_entry_in_sums(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    with (run_dir / SUMS_FILE).open("a", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  ../x\n")
    with pytest.raises(ArtifactError, match=r"\.\./x"):
        verify(run_dir)


@pytest.mark.parametrize("key", ["../x", "checkpoint/model.bin", SUMS_FILE, MANIFEST])
def test_bad_manifest_files_key(tmp_path: Path, key: str) -> None:
    run_dir = new_run(tmp_path)
    manifest = load_json(run_dir / MANIFEST)
    manifest["files"][key] = {"sha256": "0" * 64, "size": 1}
    write_json(run_dir / MANIFEST, manifest)
    with pytest.raises(ArtifactError, match=re.escape(key)):
        verify(run_dir)


def test_extra_sums_entry_outside_checkpoint(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    shutil.copyfile(FIXTURE / "provenance.json", run_dir / "provenance.json")
    write_sums(run_dir, [MANIFEST, *ARTIFACT_FILES, "provenance.json"])
    with pytest.raises(ArtifactError, match="provenance.json"):
        verify(run_dir)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"format_version": 2}, "format_version"),
        ({"laya_version": "0.3.21"}, "0.3.21"),
        ({"artifact_id": "../evil"}, "artifact_id"),
    ],
    ids=["format-2", "laya-0.3.21", "unsafe-id"],
)
def test_unsupported_manifest(tmp_path: Path, kwargs: dict[str, Any], match: str) -> None:
    run_dir = new_run(tmp_path, **kwargs)
    with pytest.raises(ArtifactError, match=match):
        read_manifest(run_dir)
    with pytest.raises(ArtifactError, match=match):
        verify(run_dir)


@pytest.mark.parametrize("content", ["{not json", '{"format_version": 1}'])
def test_invalid_manifest(tmp_path: Path, content: str) -> None:
    run_dir = new_run(tmp_path)
    (run_dir / MANIFEST).write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactError, match="manifest.json"):
        read_manifest(run_dir)


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="manifest.json"):
        read_manifest(tmp_path)


def test_missing_sums(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    (run_dir / SUMS_FILE).unlink()
    with pytest.raises(ArtifactError, match=SUMS_FILE):
        verify(run_dir)


def test_golden_probability_off_names_row_and_field(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    rows = golden_rows()
    probabilities = rows[0]["expected"]["polarity"]["answer"]["probabilities"]
    probabilities["neutral"] = round(probabilities["neutral"] + 0.01, 4)
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (run_dir / "golden.jsonl").write_text(lines, encoding="utf-8")
    resum(run_dir)
    with pytest.raises(ArtifactError, match=r"'g01'.*probabilities"):
        verify(run_dir)


def test_checkpoint_entries_are_ignored_and_never_copied(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    (run_dir / "checkpoint").mkdir()
    (run_dir / "checkpoint" / "big.bin").write_bytes(b"weights")
    with (run_dir / SUMS_FILE).open("a", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  checkpoint/big.bin\n")  # wrong on purpose: never hashed
    verified = verify(run_dir)
    assert not any(rel.startswith("checkpoint") for rel in verified.files)
    installed = install(run_dir, verified.files, tmp_path / "artifacts")
    assert not (installed.path / "checkpoint").exists()


# --------------------------------------------------------------------------- check_golden


def write_golden(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def first_row() -> dict[str, Any]:
    return golden_rows()[0]  # g01: one choice question, mixed 0.3008 / positive 0.2668


def test_check_golden_counts_every_row(runtime: System1Runtime) -> None:
    assert check_golden(runtime, FIXTURE / "golden.jsonl") == len(golden_rows())


def test_check_golden_ignores_legend_and_unknown_keys(
    runtime: System1Runtime, tmp_path: Path
) -> None:
    row = first_row()
    row["expected"]["polarity"]["answer"] |= {"legend": {"0": "x"}, "note": "ignored"}
    assert check_golden(runtime, write_golden(tmp_path, [row])) == 1


def test_check_golden_tolerates_a_choice_flip_inside_the_margin(
    runtime: System1Runtime, tmp_path: Path
) -> None:
    row = first_row()
    answer = row["expected"]["polarity"]["answer"]
    answer["choice"] = "positive"
    answer["probabilities"] |= {"mixed": 0.2850, "positive": 0.2860}  # margin 0.001
    path = write_golden(tmp_path, [row])
    assert check_golden(runtime, path, tolerance=0.02) == 1
    answer["probabilities"] |= {"mixed": 0.3008, "positive": 0.2668}  # margin 0.034
    with pytest.raises(ArtifactError, match="choice"):
        check_golden(runtime, write_golden(tmp_path, [row]), tolerance=0.01)


def mutate(row: dict[str, Any], change: str) -> None:
    want = row["expected"]["polarity"]
    if change == "input_ids":
        want["input_ids"][-2] += 1
    elif change == "markers":
        want["markers"][0] += 1
    elif change == "type":
        want["answer"]["type"] = "score"
    elif change == "confidence":
        del want["answer"]["confidence"]
    elif change == "answer_confidence":
        want["answer"]["answer_confidence"] = "high"
    elif change == "probabilities":
        del want["answer"]["probabilities"]["positive"]
    elif change == "choice":
        want["answer"]["choice"] = "negative"
    elif change == "unanswered":
        row["expected"]["other"] = want
    elif change == "neither":
        del row["expected"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("input_ids", "input_ids"),
        ("markers", "markers"),
        ("type", "type"),
        ("confidence", "confidence"),
        ("answer_confidence", "answer_confidence"),
        ("probabilities", "probabilities"),
        ("choice", "choice"),
        ("unanswered", "other"),
        ("neither", "expected"),
    ],
)
def test_check_golden_mismatch(
    runtime: System1Runtime, tmp_path: Path, change: str, match: str
) -> None:
    row = first_row()
    mutate(row, change)
    with pytest.raises(ArtifactError, match=rf"'g01'.*{match}"):
        check_golden(runtime, write_golden(tmp_path, [row]))


def test_check_golden_error_rows(runtime: System1Runtime, tmp_path: Path) -> None:
    error_row = next(row for row in golden_rows() if "error" in row)
    wrong_text = {**error_row, "error": "something else"}
    with pytest.raises(ArtifactError, match="something else"):
        check_golden(runtime, write_golden(tmp_path, [wrong_text]))
    no_error = {"id": "g99", "state": "ok", "questions": {"polarity": POLARITY},
                "error": "options exceed head_max_len"}
    with pytest.raises(ArtifactError, match="'g99'"):
        check_golden(runtime, write_golden(tmp_path, [no_error]))


def test_check_golden_invalid_question_in_expected_row(
    runtime: System1Runtime, tmp_path: Path
) -> None:
    row = first_row()
    row["questions"]["polarity"]["type"] = "multiple"
    with pytest.raises(ArtifactError, match="'g01'"):
        check_golden(runtime, write_golden(tmp_path, [row]))


@pytest.mark.parametrize("content", ["", "\n", "{not json\n", '{"id": "g1"}\n'])
def test_check_golden_bad_file(runtime: System1Runtime, tmp_path: Path, content: str) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactError, match="golden.jsonl"):
        check_golden(runtime, path)


def test_check_golden_missing_file(runtime: System1Runtime, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="golden.jsonl"):
        check_golden(runtime, tmp_path / "golden.jsonl")


# --------------------------------------------------------------------------- install


def test_artifact_home(tmp_path: Path) -> None:
    home = artifact_home(tmp_path, "polarity", "gen.predict")
    assert home == tmp_path / "system1" / "skills" / "polarity" / "gen.predict"
    for skill, predictor in (("../x", "classify"), ("polarity", "a/b"), ("polarity", "")):
        with pytest.raises(ArtifactError, match="unsafe"):
            artifact_home(tmp_path, skill, predictor)


def test_install_copies_the_verified_files(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    verified = verify(run_dir)
    artifacts = tmp_path / "artifacts"
    installed = install(run_dir, verified.files, artifacts)
    home = artifacts / "system1" / "skills" / "polarity" / "classify"
    assert installed.path == home / ARTIFACT_ID
    assert all_files(installed.path) == {*verified.files, INSTALL_FILE}
    assert load_json(home / CURRENT_FILE) == {"artifact_id": ARTIFACT_ID}
    record = load_json(installed.path / INSTALL_FILE)
    assert list(record) == [
        "format", "artifact_id", "run", "skill", "predictor", "binds", "skill_source_sha256",
        "thresholds", "target_precision", "installed_at", "stamp",
    ]
    assert record["binds"] == {**BINDS, "dataset_manifest_sha256": DATASET}
    assert record["format"] == 1 and record["run"] == "ftjob-1"
    assert set(record["stamp"]) == set(verified.files)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["installed_at"])
    assert installed == InstalledArtifact(
        skill="polarity",
        predictor="classify",
        artifact_id=ARTIFACT_ID,
        run="ftjob-1",
        path=home / ARTIFACT_ID,
        binds=ManifestBinds(**BINDS, dataset_manifest_sha256=DATASET),
        skill_source_sha256=CURRENT_SOURCE,
        thresholds={"polarity": 0.83},
        target_precision=0.95,
        installed_at=record["installed_at"],
    )
    assert not list(home.glob(".tmp-*"))
    # The installed copy is a complete student.
    student = System1Runtime(installed.path)
    assert student.predict("good service", {"polarity": POLARITY})["polarity"].key


def test_reinstalling_another_artifact_removes_the_old_one(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    first = new_run(tmp_path, "ftjob-1")
    install(first, verify(first).files, artifacts)
    second = new_run(tmp_path, "ftjob-2", artifact_id=OTHER_ID)
    installed = install(second, verify(second).files, artifacts)
    home = installed.path.parent
    assert sorted(p.name for p in home.iterdir()) == [OTHER_ID, CURRENT_FILE]
    assert load_installed(artifacts, "polarity", "classify").run == "ftjob-2"
    # The same id again replaces its directory.
    again = install(second, verify(second).files, artifacts)
    assert again.path == installed.path
    assert sorted(p.name for p in home.iterdir()) == [OTHER_ID, CURRENT_FILE]


def test_run_file_changed_after_verify(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = new_run(tmp_path)
    files = verify(run_dir).files
    install(run_dir, files, artifacts)
    current = artifact_home(artifacts, "polarity", "classify") / CURRENT_FILE
    before = current.read_bytes()
    with (run_dir / "golden.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ArtifactError, match="golden.jsonl"):
        install(run_dir, files, artifacts)
    assert current.read_bytes() == before
    assert not list(current.parent.glob(".tmp-*"))


def test_install_rejects_unsafe_file_keys(tmp_path: Path) -> None:
    run_dir = new_run(tmp_path)
    files = {**verify(run_dir).files, "../escape": "0" * 64}
    with pytest.raises(ArtifactError, match="escape"):
        install(run_dir, files, tmp_path / "artifacts")


def test_copy_failure_leaves_no_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = new_run(tmp_path)
    files = verify(run_dir).files
    install(run_dir, files, artifacts)
    home = artifact_home(artifacts, "polarity", "classify")
    before = (home / CURRENT_FILE).read_bytes()

    def boom(src: Any, dst: Any, **_: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", boom)
    with pytest.raises(ArtifactError, match="No space left"):
        install(run_dir, files, artifacts)
    assert (home / CURRENT_FILE).read_bytes() == before
    assert not list(home.glob(".tmp-*"))


# --------------------------------------------------------------------------- load_installed


@pytest.fixture
def installed(tmp_path: Path) -> InstalledArtifact:
    run_dir = new_run(tmp_path)
    return install(run_dir, verify(run_dir).files, tmp_path / "artifacts")


def test_load_installed_ok(tmp_path: Path, installed: InstalledArtifact) -> None:
    assert load_installed(tmp_path / "artifacts", "polarity", "classify") == installed


def test_load_installed_not_installed(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotInstalledError, match="polarity/classify"):
        load_installed(tmp_path / "artifacts", "polarity", "classify")


def test_load_installed_after_utime(tmp_path: Path, installed: InstalledArtifact) -> None:
    model = installed.path / "model.onnx"
    stat = model.stat()
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ArtifactError, match="changed since install"):
        load_installed(tmp_path / "artifacts", "polarity", "classify")


def test_load_installed_missing_file(tmp_path: Path, installed: InstalledArtifact) -> None:
    (installed.path / "golden.jsonl").unlink()
    with pytest.raises(ArtifactError, match="changed since install"):
        load_installed(tmp_path / "artifacts", "polarity", "classify")


@pytest.mark.parametrize(
    "current",
    ['{"artifact_id": "../evil"}', '{"artifact_id": 7}', "[]", "{not json"],
    ids=["unsafe", "not-str", "not-object", "bad-json"],
)
def test_load_installed_bad_current(
    tmp_path: Path, installed: InstalledArtifact, current: str
) -> None:
    (installed.path.parent / CURRENT_FILE).write_text(current, encoding="utf-8")
    with pytest.raises(ArtifactError) as info:
        load_installed(tmp_path / "artifacts", "polarity", "classify")
    assert not isinstance(info.value, ArtifactNotInstalledError)


def test_load_installed_unreadable_install_json(
    tmp_path: Path, installed: InstalledArtifact
) -> None:
    (installed.path / INSTALL_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactError, match="changed since install"):
        load_installed(tmp_path / "artifacts", "polarity", "classify")


def test_load_installed_ids_disagree(tmp_path: Path, installed: InstalledArtifact) -> None:
    home = installed.path.parent
    shutil.copytree(installed.path, home / OTHER_ID)
    (home / CURRENT_FILE).write_text(json.dumps({"artifact_id": OTHER_ID}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="changed since install"):
        load_installed(tmp_path / "artifacts", "polarity", "classify")
