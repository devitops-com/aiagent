"""Trained-student artifacts: verify a devai run, install it, load the installed one.

A run directory (``<distill_dir>/runs/<job>/``, written by devai) holds the artifact,
``manifest.json`` plus every ``manifest.files`` key, and ``SHA256SUMS`` over exactly
those (plus ``checkpoint/`` entries, which are ignored). ``job.json``, events and logs
are job bookkeeping and never read here.

:func:`verify_artifact` checks the files, the binds and the token limits, then
reproduces the golden answers on onnxruntime. :func:`install_artifact` copies the
verified files to ``<artifacts_dir>/system1/skills/<skill>/<predictor>/<artifact_id>/``,
stamps them and switches ``current.json``. :func:`load_installed` checks that stamp
(no hashing) whenever a skill runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from aiagent.exceptions import ArtifactError, ArtifactNotInstalledError, QuestionError
from aiagent.system1.contract import (
    SAFE_ID,
    SUMS_FILE,
    Binds,
    file_sha256,
    parse_sha256sums,
    safe_id,
    safe_relpath,
)
from aiagent.system1.runtime import Answer, System1Runtime, load_agent_config

logger = logging.getLogger(__name__)

FORMAT_VERSIONS: Final = frozenset({1})
LAYA_COMPAT: Final = frozenset({"0.3.20"})
GOLDEN_TOLERANCE: Final = 1e-3
MANIFEST_FILE: Final = "manifest.json"
REQUIRED_FILES: Final = (
    MANIFEST_FILE,
    "model.onnx",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "golden.jsonl",
)
CHECKPOINT_DIR: Final = "checkpoint"
INSTALL_FILE: Final = "install.json"
INSTALL_FORMAT: Final = 1
CURRENT_FILE: Final = "current.json"
SHADOW_LOG: Final = "shadow.jsonl"

_NUMERIC: Final = ("confidence", "answer_confidence", "score", "noul")
_TMP_PREFIX: Final = ".tmp-"


class ManifestBinds(BaseModel):
    """The artifact's binds: the dataset's three plus the dataset manifest hash."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    signature_sha256: str
    question_set_sha256: str
    skill_source_sha256: str
    dataset_manifest_sha256: str


class FileEntry(BaseModel):
    """One ``manifest.files`` entry."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    sha256: str
    size: int


class LayaInfo(BaseModel):
    """The laya release the student was trained and exported with."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    version: str
    commit: str | None = None


class GoldenInfo(BaseModel):
    """devai's note on golden.jsonl (its tolerance is not used; aiagent's is)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    n: int
    tolerance: float


class ArtifactManifest(BaseModel):
    """A run's ``manifest.json`` (devai writes it; unknown fields are ignored)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    format_version: int
    artifact_id: str
    binds: ManifestBinds
    laya: LayaInfo
    golden: GoldenInfo
    files: dict[str, FileEntry]


class _GoldenExpected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input_ids: list[int]
    markers: list[int]
    answer: dict[str, Any]


class _GoldenRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Any]
    expected: dict[str, _GoldenExpected] | None = None
    error: str | None = None


class _Stamp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    size: int
    mtime_ns: int
    inode: int


class _InstallRecord(BaseModel):
    """``install.json``; field order = JSON order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: int
    artifact_id: str
    run: str
    skill: str
    predictor: str
    binds: ManifestBinds
    skill_source_sha256: str
    thresholds: dict[str, float]
    target_precision: float
    installed_at: str
    stamp: dict[str, _Stamp]


def read_manifest(run_dir: Path) -> ArtifactManifest:
    """Parse and check format_version, laya version and a safe artifact_id."""
    path = run_dir / MANIFEST_FILE
    try:
        manifest = ArtifactManifest.model_validate_json(path.read_bytes())
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    except ValidationError as exc:
        raise ArtifactError(f"{path}: not an artifact manifest: {exc}") from exc
    if manifest.format_version not in FORMAT_VERSIONS:
        raise ArtifactError(
            f"{path}: unsupported format_version {manifest.format_version} "
            f"(supported: {sorted(FORMAT_VERSIONS)})"
        )
    if manifest.laya.version not in LAYA_COMPAT:
        raise ArtifactError(
            f"{path}: laya {manifest.laya.version} is not supported "
            f"(supported: {sorted(LAYA_COMPAT)})"
        )
    try:
        safe_id(manifest.artifact_id, "artifact_id")
    except ValueError as exc:
        raise ArtifactError(f"{path}: {exc}") from exc
    return manifest


@dataclass(frozen=True)
class VerifiedArtifact:
    """A run directory whose files, binds, limits and golden answers checked out."""

    run_dir: Path
    manifest: ArtifactManifest
    files: Mapping[str, str]  # relpath -> sha256: manifest.json + manifest.files keys
    source_matches: bool  # artifact skill_source_sha256 == expected (soft bind)
    runtime: System1Runtime  # the session golden ran on, reused: the model loads once


def _in_checkpoint(rel: str) -> bool:
    return PurePosixPath(rel).parts[0] == CHECKPOINT_DIR


def _artifact_file(rel: str) -> str:
    """rel if it can name an artifact file (safe, not SHA256SUMS or checkpoint/)."""
    safe_relpath(rel)
    if rel == SUMS_FILE or _in_checkpoint(rel):
        raise ValueError(f"{rel!r} cannot be an artifact file")
    return rel


def _read_sums(run_dir: Path) -> dict[str, str]:
    path = run_dir / SUMS_FILE
    try:
        return parse_sha256sums(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{path}: {exc}") from exc


def _verify_files(run_dir: Path, manifest: ArtifactManifest) -> dict[str, str]:
    """Hash every artifact file against SHA256SUMS and manifest.files."""
    for rel in manifest.files:
        try:
            if rel == MANIFEST_FILE:
                raise ValueError(f"{rel!r} cannot list itself")
            _artifact_file(rel)
        except ValueError as exc:
            raise ArtifactError(f"{run_dir / MANIFEST_FILE}: files: {exc}") from exc
    names = [MANIFEST_FILE, *manifest.files]
    for rel in REQUIRED_FILES:
        if rel not in names:
            raise ArtifactError(f"{run_dir}: required {rel!r} is not an artifact file")
    sums = _read_sums(run_dir)
    for rel in names:
        if rel not in sums:
            raise ArtifactError(f"{run_dir}: {SUMS_FILE} does not list {rel!r}")
    for rel in sums:
        if rel not in names and not _in_checkpoint(rel):
            raise ArtifactError(
                f"{run_dir}: {SUMS_FILE} lists {rel!r}, which is not an artifact file"
            )
    files = {}
    for rel in names:
        path = run_dir / rel
        if not path.is_file():
            raise ArtifactError(f"{run_dir}: artifact file {rel!r} is missing")
        digest = file_sha256(path)
        if digest != sums[rel]:
            raise ArtifactError(f"{run_dir}: {rel!r} does not match {SUMS_FILE}")
        entry = manifest.files.get(rel)
        size = path.stat().st_size
        if entry is not None and (entry.sha256 != digest or entry.size != size):
            raise ArtifactError(
                f"{run_dir}: {rel!r} does not match its {MANIFEST_FILE} files entry"
            )
        files[rel] = digest
    return files


def _check_binds(
    manifest: ArtifactManifest, expected: Binds, dataset_manifest_sha256: str
) -> bool:
    """Raise on a hard bind mismatch; return whether the soft (source) bind holds."""
    binds = manifest.binds
    if (
        binds.signature_sha256 != expected.signature_sha256
        or binds.question_set_sha256 != expected.question_set_sha256
    ):
        raise ArtifactError(
            f"artifact {manifest.artifact_id} is unbound: the skill's "
            "signature/questions changed since it was trained; label and train again"
        )
    if binds.dataset_manifest_sha256 != dataset_manifest_sha256:
        trained, wanted = binds.dataset_manifest_sha256, dataset_manifest_sha256
        raise ArtifactError(
            f"artifact {manifest.artifact_id} was trained on dataset "
            f"ds-{trained[:12]}, not ds-{wanted[:12]}"
        )
    return binds.skill_source_sha256 == expected.skill_source_sha256


def verify_artifact(
    run_dir: Path,
    *,
    expected: Binds,
    dataset_manifest_sha256: str,
    max_len: int,
    head_max_len: int,
) -> VerifiedArtifact:
    """Full verification (hashes, binds, limits, golden)."""
    manifest = read_manifest(run_dir)
    files = _verify_files(run_dir, manifest)
    source_matches = _check_binds(manifest, expected, dataset_manifest_sha256)
    config = load_agent_config(run_dir / "rl_agent_config.json")
    if (config.max_len, config.head_max_len) != (max_len, head_max_len):
        raise ArtifactError(
            f"artifact {manifest.artifact_id}: max_len/head_max_len "
            f"{config.max_len}/{config.head_max_len} differ from the dataset's "
            f"{max_len}/{head_max_len}"
        )
    runtime = System1Runtime(run_dir)
    check_golden(runtime, run_dir / "golden.jsonl")
    return VerifiedArtifact(run_dir, manifest, files, source_matches, runtime)


# --------------------------------------------------------------------------- golden


def _check_close(what: str, got: object, want: object, tolerance: float) -> None:
    ok = (
        isinstance(got, float)
        and isinstance(want, int | float)
        and not isinstance(want, bool)
        and abs(got - want) <= tolerance
    )
    if not ok:
        raise ArtifactError(f"{what}: {got!r} differs from laya's {want!r}")


def _check_answer(
    where: str, got: Answer, want: _GoldenExpected, tolerance: float
) -> None:
    if list(got.encoded.input_ids) != want.input_ids:
        raise ArtifactError(f"{where}: input_ids differ from laya's")
    markers = list(got.encoded.markers)
    if markers != want.markers:
        raise ArtifactError(
            f"{where}: markers {markers} differ from laya's {want.markers}"
        )
    mine, laya = got.to_laya(), want.answer
    if mine["type"] != laya.get("type"):
        raise ArtifactError(
            f"{where}: type {mine['type']!r} differs from laya's {laya.get('type')!r}"
        )
    for field in _NUMERIC:
        if field in mine or field in laya:
            what = f"{where}: {field}"
            _check_close(what, mine.get(field), laya.get(field), tolerance)
    if "probabilities" not in mine:
        return
    probabilities, laya_probabilities = mine["probabilities"], laya.get("probabilities")
    if not isinstance(laya_probabilities, dict) or set(laya_probabilities) != set(
        probabilities
    ):
        raise ArtifactError(
            f"{where}: probabilities keys {list(probabilities)} differ from laya's "
            f"{laya_probabilities!r}"
        )
    for key, p in probabilities.items():
        what = f"{where}: probabilities[{key!r}]"
        _check_close(what, p, laya_probabilities[key], tolerance)
    if "choice" in mine and mine["choice"] != laya.get("choice"):
        top = sorted(laya_probabilities.values(), reverse=True)
        margin = top[0] - top[1] if len(top) > 1 else math.inf
        if margin > 2 * tolerance:  # a near tie may flip; the probabilities still agree
            raise ArtifactError(
                f"{where}: choice {mine['choice']!r} differs from laya's "
                f"{laya.get('choice')!r}"
            )


def _check_golden_row(
    runtime: System1Runtime, row: _GoldenRow, tolerance: float
) -> None:
    where = f"golden row {row.id!r}"
    if row.error is not None:
        try:
            runtime.encode(row.state, row.questions)
        except QuestionError as exc:
            if row.error not in str(exc):
                raise ArtifactError(
                    f"{where}: expected an error containing {row.error!r}, got {exc}"
                ) from exc
            return
        raise ArtifactError(f"{where}: expected an error containing {row.error!r}")
    if row.expected is None:
        raise ArtifactError(f"{where}: has neither 'expected' nor 'error'")
    try:
        answers = runtime.predict(row.state, row.questions)
    except QuestionError as exc:
        raise ArtifactError(f"{where}: {exc}") from exc
    if set(row.expected) != set(answers):
        raise ArtifactError(
            f"{where}: expected answers for {sorted(row.expected)}, "
            f"the questions are {sorted(answers)}"
        )
    for qid, want in row.expected.items():
        _check_answer(f"{where}, question {qid!r}", answers[qid], want, tolerance)


def check_golden(
    runtime: System1Runtime, golden_path: Path, *, tolerance: float = GOLDEN_TOLERANCE
) -> int:
    """Reproduce golden.jsonl; return rows checked; ArtifactError at a mismatch."""
    try:
        lines = golden_path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {golden_path}: {exc}") from exc
    checked = 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = _GoldenRow.model_validate_json(line)
        except ValidationError as exc:
            raise ArtifactError(f"{golden_path} line {number}: {exc}") from exc
        _check_golden_row(runtime, row, tolerance)
        checked += 1
    if not checked:
        raise ArtifactError(f"{golden_path}: no golden rows")
    return checked


# --------------------------------------------------------------------------- install


@dataclass(frozen=True)
class InstalledArtifact:
    """A student installed for one (skill, predictor)."""

    skill: str
    predictor: str
    artifact_id: str
    run: str
    path: Path
    binds: ManifestBinds
    skill_source_sha256: str
    thresholds: Mapping[str, float]
    target_precision: float
    installed_at: str


def artifact_home(artifacts_dir: Path, skill: str, predictor: str) -> Path:
    """<artifacts_dir>/system1/skills/<skill>/<predictor> (both safe ids)."""
    try:
        safe_id(skill, "skill")
        safe_id(predictor, "predictor")
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    return artifacts_dir / "system1" / "skills" / skill / predictor


def _write_json(path: Path, data: object) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(f"{_TMP_PREFIX}{uuid.uuid4().hex[:8]}-{path.name}")
    try:
        _write_json(tmp, data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _stamp(path: Path) -> _Stamp:
    st = path.stat()
    return _Stamp(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _installed(record: _InstallRecord, path: Path) -> InstalledArtifact:
    return InstalledArtifact(
        skill=record.skill,
        predictor=record.predictor,
        artifact_id=record.artifact_id,
        run=record.run,
        path=path,
        binds=record.binds,
        skill_source_sha256=record.skill_source_sha256,
        thresholds=record.thresholds,
        target_precision=record.target_precision,
        installed_at=record.installed_at,
    )


def _warn_not_removed(_func: object, path: str, exc: BaseException) -> None:
    logger.warning("could not remove the old System 1 student %s: %s", path, exc)


def _remove_others(home: Path, keep: str) -> None:
    """Remove the other installed ids (best effort: the install already succeeded)."""
    for entry in home.iterdir():
        if entry.name != keep and SAFE_ID.fullmatch(entry.name) and entry.is_dir():
            shutil.rmtree(entry, onexc=_warn_not_removed)


def install_artifact(
    run_dir: Path,
    *,
    files: Mapping[str, str],
    artifacts_dir: Path,
    skill: str,
    predictor: str,
    run: str,
    thresholds: Mapping[str, float],
    target_precision: float,
    skill_source_sha256: str,
) -> InstalledArtifact:
    """Copy `files` from run_dir, re-hash the copies against `files`, stamp, switch
    current.json atomically."""
    home = artifact_home(artifacts_dir, skill, predictor)
    tmp = home / f"{_TMP_PREFIX}{uuid.uuid4().hex[:8]}"
    try:
        for rel in files:
            _artifact_file(rel)
        home.mkdir(parents=True, exist_ok=True)
        tmp.mkdir()
        for rel, digest in files.items():
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / rel, target)  # modes: the user's umask
            if file_sha256(target) != digest:
                raise ArtifactError(
                    f"{run_dir / rel} changed since it was verified; "
                    "run `aiagent distill eval` again"
                )
        manifest = read_manifest(tmp)
        record = _InstallRecord(
            format=INSTALL_FORMAT,
            artifact_id=manifest.artifact_id,
            run=run,
            skill=skill,
            predictor=predictor,
            binds=manifest.binds,
            skill_source_sha256=skill_source_sha256,
            thresholds=dict(thresholds),
            target_precision=target_precision,
            installed_at=_utc_now(),
            stamp={rel: _stamp(tmp / rel) for rel in files},
        )
        _write_json(tmp / INSTALL_FILE, record.model_dump(mode="json"))
        final = home / manifest.artifact_id
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        _write_json_atomic(home / CURRENT_FILE, {"artifact_id": manifest.artifact_id})
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot install {run_dir} into {home}: {exc}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # already gone after a success
    _remove_others(home, keep=record.artifact_id)
    return _installed(record, final)


def load_installed(
    artifacts_dir: Path, skill: str, predictor: str
) -> InstalledArtifact:
    """Read current.json and install.json and check the stamp (no hashing)."""
    home = artifact_home(artifacts_dir, skill, predictor)
    current = home / CURRENT_FILE
    try:
        data = json.loads(current.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactNotInstalledError(
            f"no System 1 student is installed for {skill}/{predictor}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {current}: {exc}") from exc
    artifact_id = data.get("artifact_id") if isinstance(data, dict) else None
    if not isinstance(artifact_id, str):
        raise ArtifactError(f"{current}: no 'artifact_id' string")
    try:
        safe_id(artifact_id, "artifact_id")
    except ValueError as exc:
        raise ArtifactError(f"{current}: {exc}") from exc
    path = home / artifact_id
    changed = f"{path} changed since install; run `aiagent distill install` again"
    try:
        record = _InstallRecord.model_validate_json((path / INSTALL_FILE).read_bytes())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{changed} ({INSTALL_FILE}: {exc})") from exc
    if record.artifact_id != artifact_id:
        raise ArtifactError(f"{changed} ({INSTALL_FILE} names {record.artifact_id})")
    for rel, stamped in record.stamp.items():
        try:
            now = _stamp(path / rel)
        except OSError as exc:
            raise ArtifactError(f"{changed} ({rel} is missing)") from exc
        if now != stamped:
            raise ArtifactError(f"{changed} ({rel})")
    if read_manifest(path).artifact_id != artifact_id:
        raise ArtifactError(f"{changed} ({MANIFEST_FILE} names another artifact)")
    return _installed(record, path)
