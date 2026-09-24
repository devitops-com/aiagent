"""The aiagent -> devai distillation dataset: ``<distill_dir>/inbox/ds-<sha12>/``.

A dataset is ``manifest.json``, one JSONL file per split and ``SHA256SUMS``. Rows are
compact canonical JSON with ``None`` fields dropped (a pool row has no ``gold`` and no
``teacher`` key); the manifest is indented JSON with nulls kept. The dataset id is
``ds-`` plus 12 hex chars of the manifest's sha256. :func:`validate_dataset` mirrors
devai's trainer check, so a bad dataset never costs a GPU window; :func:`write_dataset`
runs it before publishing a dataset atomically.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, ValidationError

from aiagent.distill.splits import LABELED_SPLITS, SPLIT_METHOD, SPLITS, Split
from aiagent.exceptions import DatasetContractError, DistillError, QuestionError
from aiagent.system1.contract import (
    SUMS_FILE,
    bytes_sha256,
    canonical_json,
    content_sha256,
    file_sha256,
    format_sha256sums,
    parse_sha256sums,
    safe_id,
)
from aiagent.system1.contract import dataset_id as dataset_id_of
from aiagent.system1.sequence import InternalQuestion, SequenceTokenizer, prepare

SCHEMA_VERSION: Final = 1
DEFAULT_BASE: Final = "laya-multilingual@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
DATASET_FILES: Final = (
    "manifest.json",
    "train.jsonl",
    "calib.jsonl",
    "heldout.jsonl",
    "pool.jsonl",
)
PROB_SUM_TOL: Final = 1e-6

_MANIFEST: Final = "manifest.json"
_AGENT_CONFIG: Final = "rl_agent_config.json"
_WEIGHTS: Final = "model.safetensors"
_TOKENIZER_JSON: Final = "tokenizer/tokenizer.json"
_REV_DIR_LEN: Final = 12  # devai's checkpoint_dirname: base/<name>@<revision[:12]>/
_LAYA_MAX_LEN: Final = 512  # laya's defaults when rl_agent_config.json omits them
_LAYA_HEAD_MAX_LEN: Final = 192
_ANSWER_KEYS: Final = ("gold", "teacher")  # in labeled rows, never in pool rows


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldAnswer(_Model):
    """The teacher's answer to one question: vote shares and their first argmax."""

    label: str
    probabilities: dict[str, float]


class TeacherInfo(_Model):
    """How a row was labeled: k samples, and the unparseable ones per question."""

    k: int
    parse_failures: dict[str, int]


class StudentTokens(_Model):
    """The student's row length and H(input_ids) for one question."""

    n: int
    ids_sha256: str


class Row(_Model):
    """One dataset row: a segment's state, its questions and (unless pool) its gold."""

    id: str
    group_id: str
    split: Split
    synthetic: bool = False
    state: dict[str, str]
    questions: dict[str, dict[str, Any]]
    gold: dict[str, GoldAnswer] | None = None
    teacher: TeacherInfo | None = None
    student_tokens: dict[str, StudentTokens]


class BaseCheckpointRef(_Model):
    """The base checkpoint a dataset was tokenized for."""

    name: str
    revision: str
    weights_sha256: str
    tokenizer_sha256: str


class SplitsInfo(_Model):
    """How rows were split, and the rows per split."""

    method: str
    counts: dict[Split, int]


class FileInfo(_Model):
    """One split file's sha256 and row count."""

    sha256: str
    rows: int


class TeacherSpec(_Model):
    """The teacher LM and sampling a dataset was labeled with."""

    model: str
    k: int
    temperature: float
    dspy: str
    adapter: str = "ChatAdapter"


class SourceInfo(_Model):
    """One source document and the segments it gave."""

    doc_id: str
    origin: str
    segments: int


class LabelStats(_Model):
    """Labeling totals: labeled rows, dropped rows, parse failures, label counts."""

    rows: int
    unlabeled: int
    parse_failures: dict[str, int]
    label_histogram: dict[str, dict[str, int]]


class ProducerBinds(_Model):
    """The hashes devai copies verbatim into the artifact's binds."""

    signature_sha256: str
    question_set_sha256: str
    skill_source_sha256: str


class Producer(_Model):
    """Who made the dataset, from what, and how (opaque to devai but for binds)."""

    name: Literal["aiagent"] = "aiagent"
    version: str
    created_at: str
    skill: str
    predictor: str
    input_field: str
    derive_version: int
    binds: ProducerBinds
    questions: dict[str, dict[str, Any]]
    teacher: TeacherSpec
    campaign: str
    round: int
    parent_dataset: str | None
    sources: list[SourceInfo]
    augmentation: None = None
    label_stats: LabelStats


class DatasetManifest(_Model):
    """manifest.json."""

    schema_version: Literal[1]
    base_checkpoint: BaseCheckpointRef
    max_len: int
    head_max_len: int
    splits: SplitsInfo
    files: dict[str, FileInfo]
    producer: Producer


@dataclass(frozen=True)
class BaseCheckpoint:
    """A staged base checkpoint under <distill_dir>/base/<name>@<revision[:12]>/."""

    name: str
    revision: str
    path: Path
    max_len: int
    head_max_len: int
    weights_sha256: str
    tokenizer_sha256: str

    def tokenizer(self) -> SequenceTokenizer:
        """SequenceTokenizer.from_dir(path/'tokenizer')."""
        return SequenceTokenizer.from_dir(self.path / "tokenizer")

    def ref(self) -> BaseCheckpointRef:
        """The manifest's base_checkpoint entry."""
        return BaseCheckpointRef(
            name=self.name,
            revision=self.revision,
            weights_sha256=self.weights_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
        )


@dataclass(frozen=True)
class DatasetRef:
    """A published dataset: its id, directory and manifest sha256."""

    id: str
    path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class Dataset:
    """A loaded, sum-checked dataset."""

    ref: DatasetRef
    manifest: DatasetManifest
    rows: Mapping[Split, tuple[Row, ...]]


def _length(config: Mapping[str, Any], key: str, default: int, where: Path) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DistillError(f"{where}: {key} must be a positive int, got {value!r}")
    return value


def load_base_checkpoint(distill_dir: Path, base: str) -> BaseCheckpoint:
    """Parse '<name>@<rev>', read rl_agent_config.json, hash weights and tokenizer.

    devai stages the base as ``base/<name>@<rev[:12]>/``; the full revision stays the
    identity recorded in the manifest (devai checks it against its catalog).
    """
    name, _, revision = base.partition("@")
    try:
        safe_id(name, "base checkpoint name")
        safe_id(revision, "base checkpoint revision")
    except ValueError as exc:
        raise DistillError(
            f"base {base!r}: expected '<name>@<revision>'; {exc}"
        ) from exc
    path = distill_dir / "base" / f"{name}@{revision[:_REV_DIR_LEN]}"
    if not path.is_dir():
        raise DistillError(f"base checkpoint not found: {path}")
    for rel in (_AGENT_CONFIG, _WEIGHTS, _TOKENIZER_JSON):
        if not (path / rel).is_file():
            raise DistillError(f"base checkpoint file missing: {path / rel}")
    config_path = path / _AGENT_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DistillError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DistillError(f"{config_path}: not a JSON object")
    return BaseCheckpoint(
        name=name,
        revision=revision,
        path=path,
        max_len=_length(config, "max_len", _LAYA_MAX_LEN, config_path),
        head_max_len=_length(config, "head_max_len", _LAYA_HEAD_MAX_LEN, config_path),
        weights_sha256=file_sha256(path / _WEIGHTS),
        tokenizer_sha256=file_sha256(path / _TOKENIZER_JSON),
    )


def gold_from_votes(keys: Sequence[str], votes: Mapping[str, int]) -> GoldAnswer:
    """Vote histogram -> probabilities (count/total, key order) and the first argmax."""
    unknown = sorted(set(votes) - set(keys))
    if unknown:
        raise ValueError(f"votes for unknown keys {unknown}; options are {list(keys)}")
    total = sum(votes.get(key, 0) for key in keys)
    if total <= 0:
        raise ValueError("no votes: a row needs at least one parsed teacher answer")
    probabilities = {key: votes.get(key, 0) / total for key in keys}
    label = max(keys, key=probabilities.__getitem__)  # max keeps the first of a tie
    return GoldAnswer(label=label, probabilities=probabilities)


def build_row(
    *,
    group_id: str,
    index: int,
    text: str,
    input_field: str,
    questions: Mapping[str, Mapping[str, Any]],
    split: Split,
    tokenizer: SequenceTokenizer,
    max_len: int,
    head_max_len: int,
    gold: Mapping[str, GoldAnswer] | None = None,
    teacher: TeacherInfo | None = None,
) -> Row:
    """A row with student_tokens computed (id, group_id and ids_sha256 per §3.1)."""
    state = {input_field: text}
    encoded = tokenizer.encode(
        state, prepare(questions), max_len=max_len, head_max_len=head_max_len
    )
    tokens = {
        qid: StudentTokens(
            n=len(row.input_ids), ids_sha256=content_sha256(list(row.input_ids))
        )
        for qid, row in encoded.items()
    }
    return Row(
        id=f"{group_id[7:19]}:{index:04d}",
        group_id=group_id,
        split=split,
        state=state,
        questions={qid: dict(qdef) for qid, qdef in questions.items()},
        gold=dict(gold) if gold is not None else None,
        teacher=teacher,
        student_tokens=tokens,
    )


def compute_label_stats(rows: Sequence[Row], *, unlabeled: int) -> LabelStats:
    """From the labeled rows' gold labels and teacher.parse_failures (pool skipped)."""
    labeled = [row for row in rows if row.gold is not None]
    failures: dict[str, int] = {}
    histogram: dict[str, dict[str, int]] = {}
    for row in labeled:
        for qid, answer in (row.gold or {}).items():
            counts = histogram.setdefault(qid, dict.fromkeys(answer.probabilities, 0))
            counts[answer.label] = counts.get(answer.label, 0) + 1
        for qid, n in (row.teacher.parse_failures if row.teacher else {}).items():
            failures[qid] = failures.get(qid, 0) + n
    return LabelStats(
        rows=len(labeled),
        unlabeled=unlabeled,
        parse_failures=failures,
        label_histogram=histogram,
    )


def _row_bytes(row: Row) -> bytes:
    data = row.model_dump(mode="json", exclude_none=True)
    return (canonical_json(data) + "\n").encode("utf-8")


def _manifest_bytes(manifest: DatasetManifest) -> bytes:
    data = manifest.model_dump(mode="json")
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return text.encode("utf-8")


def write_dataset(
    inbox: Path, *, base: BaseCheckpoint, producer: Producer, rows: Sequence[Row]
) -> DatasetRef:
    """Serialize, validate, publish atomically as inbox/ds-<sha12> (idempotent)."""
    distill_dir = inbox.parent
    if not distill_dir.is_dir():
        raise DistillError(
            f"distill_dir {distill_dir} does not exist: mount the devai laya volume "
            "or set AIAGENT_DISTILL_DIR"
        )
    by_split = {
        s: sorted((r for r in rows if r.split == s), key=lambda r: r.id) for s in SPLITS
    }
    jsonl = {s: b"".join(_row_bytes(r) for r in by_split[s]) for s in SPLITS}
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        base_checkpoint=base.ref(),
        max_len=base.max_len,
        head_max_len=base.head_max_len,
        splits=SplitsInfo(
            method=SPLIT_METHOD, counts={s: len(by_split[s]) for s in SPLITS}
        ),
        files={
            f"{s}.jsonl": FileInfo(sha256=bytes_sha256(jsonl[s]), rows=len(by_split[s]))
            for s in SPLITS
        },
        producer=producer,
    )
    payloads = {
        _MANIFEST: _manifest_bytes(manifest),
        **{f"{s}.jsonl": jsonl[s] for s in SPLITS},
    }
    manifest_sha256 = bytes_sha256(payloads[_MANIFEST])
    sums = format_sha256sums({name: bytes_sha256(b) for name, b in payloads.items()})
    ds = dataset_id_of(manifest_sha256)
    ref = DatasetRef(ds, inbox / ds, manifest_sha256)

    inbox.mkdir(exist_ok=True)
    tmp = inbox / f".tmp-{uuid.uuid4().hex[:8]}"
    try:
        tmp.mkdir()
        for name, data in payloads.items():
            (tmp / name).write_bytes(data)
        (tmp / SUMS_FILE).write_text(sums, encoding="utf-8")
        validate_dataset(tmp, tokenizer=base.tokenizer())
        existing = ref.path / SUMS_FILE
        if not ref.path.exists():
            os.replace(tmp, ref.path)
        elif not existing.is_file() or existing.read_text(encoding="utf-8") != sums:
            raise DistillError(f"{ref.path} already exists with different contents")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ref


# --------------------------------------------------------------------------- validation


def _fail(where: str, problem: str) -> NoReturn:
    raise DatasetContractError(f"{where}: {problem}")


def _errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '<row>'}: {err['msg']}"
        for err in exc.errors()
    )


def _check_sums(path: Path) -> dict[str, str]:
    try:
        entries = parse_sha256sums((path / SUMS_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        _fail(SUMS_FILE, f"cannot read: {exc}")
    except ValueError as exc:
        raise DatasetContractError(str(exc)) from exc
    missing = [name for name in DATASET_FILES if name not in entries]
    extra = sorted(set(entries) - set(DATASET_FILES))
    if missing or extra:
        _fail(
            SUMS_FILE,
            f"must list exactly {list(DATASET_FILES)}; "
            f"missing {missing}, extra {extra}",
        )
    for name in DATASET_FILES:
        if not (path / name).is_file():
            _fail(name, "missing")
        if file_sha256(path / name) != entries[name]:
            _fail(name, "sha256 does not match SHA256SUMS")
    return entries


def _read_manifest(path: Path) -> DatasetManifest:
    try:
        data = json.loads((path / _MANIFEST).read_bytes())
    except ValueError as exc:
        _fail(_MANIFEST, f"invalid JSON: {exc}")
    try:
        return DatasetManifest.model_validate(data)
    except ValidationError as exc:
        _fail(_MANIFEST, _errors(exc))


def _lines(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(n, line) for n, line in enumerate(lines, start=1) if line.strip()]


def _check_files(
    path: Path, manifest: DatasetManifest, sums: Mapping[str, str]
) -> dict[Split, list[tuple[int, str]]]:
    names = {f"{s}.jsonl" for s in SPLITS}
    if set(manifest.files) != names:
        _fail(_MANIFEST, f"files must list exactly {sorted(names)}")
    lines: dict[Split, list[tuple[int, str]]] = {}
    for split in SPLITS:
        name = f"{split}.jsonl"
        lines[split] = _lines(path / name)
        info = manifest.files[name]
        if info.sha256 != sums[name] or info.rows != len(lines[split]):
            _fail(_MANIFEST, f"files[{name!r}] does not match {name}")
        count, rows = manifest.splits.counts.get(split), len(lines[split])
        if count != rows:
            problem = f"splits.counts[{split!r}] is {count}, {name} has {rows} rows"
            _fail(_MANIFEST, problem)
    return lines


def _find_null(value: object, where: str) -> str | None:
    if value is None:
        return where
    items: list[tuple[str, object]] = []
    if isinstance(value, dict):
        items = [(f"{where}.{k}" if where else str(k), v) for k, v in value.items()]
    elif isinstance(value, list):
        items = [(f"{where}[{i}]", v) for i, v in enumerate(value)]
    for child_where, child in items:
        found = _find_null(child, child_where)
        if found is not None:
            return found
    return None


_Keys = Mapping[str, tuple[str, ...]]  # qid -> option keys, in order


def _check_gold(where: str, gold: Mapping[str, GoldAnswer], keys: _Keys) -> None:
    if set(gold) != set(keys):
        _fail(where, f"gold must answer exactly the questions {list(keys)}")
    for qid, answer in gold.items():
        what = f"gold[{qid!r}]"
        if tuple(answer.probabilities) != keys[qid]:
            _fail(
                where, f"{what}.probabilities keys must be {list(keys[qid])}, in order"
            )
        values = list(answer.probabilities.values())
        if not all(math.isfinite(v) and v >= 0 for v in values):
            _fail(where, f"{what}.probabilities must be finite and >= 0")
        total = sum(values)
        if abs(total - 1.0) > PROB_SUM_TOL:
            _fail(where, f"{what}.probabilities sum to {total}, not 1")
        first = keys[qid][values.index(max(values))]
        if answer.label != first:
            _fail(
                where,
                f"{what}.label is {answer.label!r}, not the first argmax {first!r}",
            )


def _check_row(
    where: str, line: str, split: Split, producer: Producer, keys: _Keys
) -> Row:
    """Rules 2-4 of the contract for one line (no tokenizer)."""
    try:
        data = json.loads(line)
    except ValueError as exc:
        _fail(where, f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        _fail(where, "not a JSON object")
    # questions are copied verbatim from the manifest (nulls and all); any other null
    # is a field the writer would have dropped
    null = _find_null({k: v for k, v in data.items() if k != "questions"}, "")
    if null is not None:
        _fail(where, f"null at {null}")
    for key in _ANSWER_KEYS:
        if split == "pool" and key in data:
            _fail(where, f"a pool row has {key!r}")
        if split != "pool" and key not in data:
            _fail(where, f"missing {key!r}")
    try:
        row = Row.model_validate(data)
    except ValidationError as exc:
        _fail(where, _errors(exc))
    if row.split != split:
        _fail(where, f"split {row.split!r} in {split}.jsonl")
    if set(row.state) != {producer.input_field}:
        _fail(where, f"state must be {{{producer.input_field!r}: str}}")
    if row.questions != producer.questions:
        _fail(where, "questions differ from producer.questions")
    if row.synthetic and split in ("calib", "heldout"):
        _fail(where, f"synthetic rows are not allowed in {split}")
    if row.gold is not None:
        _check_gold(where, row.gold, keys)
    if set(row.student_tokens) != set(keys):
        _fail(where, f"student_tokens must cover exactly the questions {list(keys)}")
    return row


_Rows = dict[Split, list[tuple[str, Row]]]


def _read(path: Path) -> tuple[DatasetManifest, dict[str, InternalQuestion], _Rows]:
    """Every contract rule but the re-tokenization (rule 5)."""
    sums = _check_sums(path)
    manifest = _read_manifest(path)
    lines = _check_files(path, manifest, sums)
    try:
        internal = prepare(manifest.producer.questions)
    except QuestionError as exc:
        _fail(_MANIFEST, f"producer.questions: {exc}")
    keys = {qid: q.keys for qid, q in internal.items()}
    rows: _Rows = {s: [] for s in SPLITS}
    ids: set[str] = set()
    group_home: dict[str, str] = {}  # calib, heldout and train/pool are disjoint
    for split in SPLITS:
        home = "train/pool" if split in ("train", "pool") else split
        for number, line in lines[split]:
            where = f"{split}.jsonl:{number}"
            row = _check_row(where, line, split, manifest.producer, keys)
            if row.id in ids:
                _fail(where, f"duplicate id {row.id!r}")
            ids.add(row.id)
            seen = group_home.setdefault(row.group_id, home)
            if seen != home:
                _fail(where, f"group_id {row.group_id} is also in {seen}")
            rows[split].append((where, row))
    for split in LABELED_SPLITS:
        if not rows[split]:
            _fail(f"{split}.jsonl", "no rows")
    return manifest, internal, rows


def _check_tokens(
    where: str,
    row: Row,
    internal: Mapping[str, InternalQuestion],
    tokenizer: SequenceTokenizer,
    manifest: DatasetManifest,
) -> None:
    """Rule 5: the base tokenizer rebuilds the student tokens; no state is cut."""
    limits = {"max_len": manifest.max_len, "head_max_len": manifest.head_max_len}
    state_ids = tokenizer.state_ids(row.state)
    for qid, q in internal.items():
        encoded = tokenizer.build(q, state_ids, truncate_left=False, **limits)
        if len(encoded.markers) != len(q.options):
            _fail(
                where,
                f"question {qid!r} options exceed head_max_len={manifest.head_max_len}",
            )
        room = tokenizer.room(q, **limits)
        if len(state_ids) > room:
            _fail(
                where,
                f"the state has {len(state_ids)} tokens; "
                f"question {qid!r} leaves room for {room}",
            )
        tokens = row.student_tokens[qid]
        ids = list(encoded.input_ids)
        if tokens.n != len(ids) or tokens.ids_sha256 != content_sha256(ids):
            _fail(where, f"student_tokens[{qid!r}] does not match the base tokenizer")


def validate_dataset(path: Path, *, tokenizer: SequenceTokenizer) -> DatasetManifest:
    """All contract rules; DatasetContractError('<file>:<line>: <problem>')."""
    manifest, internal, rows = _read(path)
    for split in SPLITS:
        for where, row in rows[split]:
            _check_tokens(where, row, internal, tokenizer, manifest)
    return manifest


def find_dataset(distill_dir: Path, dataset_id: str) -> Path:
    """datasets/<id>/ if present, else inbox/<id>/; DistillError if neither."""
    try:
        safe_id(dataset_id, "dataset id")
    except ValueError as exc:
        raise DistillError(str(exc)) from exc
    for area in ("datasets", "inbox"):
        path = distill_dir / area / dataset_id
        if path.is_dir():
            return path
    raise DistillError(
        f"dataset {dataset_id} not found under {distill_dir}/datasets "
        f"or {distill_dir}/inbox"
    )


def load_dataset(
    distill_dir: Path, dataset_id: str, *, expect_manifest_sha256: str | None = None
) -> Dataset:
    """Read and sum-check a dataset (every contract rule but the re-tokenization)."""
    path = find_dataset(distill_dir, dataset_id)
    manifest, _, rows = _read(path)
    manifest_sha256 = file_sha256(path / _MANIFEST)
    actual = dataset_id_of(manifest_sha256)
    if actual != dataset_id:
        raise DatasetContractError(
            f"{path}: manifest.json hashes to {actual}, not {dataset_id}"
        )
    if expect_manifest_sha256 is not None and manifest_sha256 != expect_manifest_sha256:
        raise DistillError(
            f"{dataset_id}: manifest sha256 {manifest_sha256} is not the expected "
            f"{expect_manifest_sha256}"
        )
    return Dataset(
        ref=DatasetRef(dataset_id, path, manifest_sha256),
        manifest=manifest,
        rows={s: tuple(row for _, row in rows[s]) for s in SPLITS},
    )


def derive_repair_dataset(
    parent: Dataset,
    *,
    selected: Collection[str],
    labeled: Sequence[Row],
    created_at: str,
    version: str,
    unlabeled_added: int,
) -> tuple[Producer, list[Row]]:
    """Child rows and producer for a repair round.

    Rows: parent train + `labeled` (as train), calib/heldout verbatim, pool minus every
    selected id (labeled or not). Producer: round+1, parent_dataset = the parent's id,
    label_stats recomputed (unlabeled = parent's + unlabeled_added), created_at and
    version replaced, everything else copied.
    """
    chosen = set(selected)
    moved = [
        r if r.split == "train" else r.model_copy(update={"split": "train"})
        for r in labeled
    ]
    rows = [
        *parent.rows["train"],
        *moved,
        *parent.rows["calib"],
        *parent.rows["heldout"],
        *(r for r in parent.rows["pool"] if r.id not in chosen),
    ]
    old = parent.manifest.producer
    unlabeled = old.label_stats.unlabeled + unlabeled_added
    producer = old.model_copy(
        update={
            "created_at": created_at,
            "version": version,
            "round": old.round + 1,
            "parent_dataset": parent.ref.id,
            "label_stats": compute_label_stats(rows, unlabeled=unlabeled),
        }
    )
    return producer, rows
