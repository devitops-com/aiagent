"""The aiagent -> devai dataset: base checkpoints, rows, writing, validation, repair."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from aiagent.distill.dataset import (
    DATASET_FILES,
    DEFAULT_BASE,
    PROB_SUM_TOL,
    SCHEMA_VERSION,
    BaseCheckpoint,
    BaseCheckpointRef,
    Dataset,
    DatasetManifest,
    GoldAnswer,
    Producer,
    ProducerBinds,
    Row,
    SourceInfo,
    TeacherInfo,
    TeacherSpec,
    build_row,
    compute_label_stats,
    derive_repair_dataset,
    find_dataset,
    gold_from_votes,
    load_base_checkpoint,
    load_dataset,
    validate_dataset,
    write_dataset,
)
from aiagent.distill.splits import SPLIT_METHOD, SPLITS, assign_split, document_id
from aiagent.exceptions import DatasetContractError, DistillError
from aiagent.system1.contract import (
    SUMS_FILE,
    bytes_sha256,
    canonical_json,
    content_sha256,
    dataset_id,
    file_sha256,
    format_sha256sums,
    parse_sha256sums,
)
from aiagent.system1.sequence import SequenceTokenizer, prepare
from system1_helpers import (
    FIXTURE,
    FIXTURE_HEAD_MAX_LEN,
    FIXTURE_MAX_LEN,
    make_base_checkpoint,
    resum,
)

BASE = "laya-multilingual@0000test"
KEYS = ("negative", "neutral", "mixed", "positive")
POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": list(KEYS),
}
QUESTIONS = {"polarity": POLARITY}
LIMITS = {"max_len": FIXTURE_MAX_LEN, "head_max_len": FIXTURE_HEAD_MAX_LEN}
HEX = "ab" * 32
VOTES = (
    {"negative": 3, "neutral": 1, "mixed": 0, "positive": 0},
    {"negative": 0, "neutral": 1, "mixed": 2, "positive": 1},
    {"negative": 0, "neutral": 0, "mixed": 1, "positive": 3},
)


@pytest.fixture(scope="module")
def tok() -> SequenceTokenizer:
    return SequenceTokenizer.from_dir(FIXTURE / "tokenizer")


@pytest.fixture
def distill(tmp_path: Path) -> Path:
    path = tmp_path / "distill"
    make_base_checkpoint(path, BASE)
    return path


@pytest.fixture
def base(distill: Path) -> BaseCheckpoint:
    return load_base_checkpoint(distill, BASE)


def texts_by_split(per_split: int = 2) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {s: [] for s in SPLITS}
    i = 0
    while any(len(texts) < per_split for texts in found.values()):
        text = f"Lieferung {i} spät, čudno: good service here."
        texts = found[assign_split(document_id(text))]
        if len(texts) < per_split:
            texts.append(text)
        i += 1
    return found


def make_row(
    tok: SequenceTokenizer, text: str, *, index: int = 0, split: str | None = None, votes: int = 0,
    questions: dict[str, Any] = QUESTIONS,
) -> Row:
    group_id = document_id(text)
    split = split or assign_split(group_id)
    labeled = split != "pool"
    return build_row(
        group_id=group_id,
        index=index,
        text=text,
        input_field="text",
        questions=questions,
        split=split,
        tokenizer=tok,
        **LIMITS,
        gold={"polarity": gold_from_votes(KEYS, VOTES[votes % 3])} if labeled else None,
        teacher=TeacherInfo(k=4, parse_failures={"polarity": votes % 2}) if labeled else None,
    )


def make_rows(tok: SequenceTokenizer, questions: dict[str, Any] = QUESTIONS) -> list[Row]:
    rows = []
    for n, (split, texts) in enumerate(texts_by_split().items()):
        for m, text in enumerate(texts):
            rows.append(make_row(tok, text, split=split, votes=n + m, questions=questions))
    first_train = texts_by_split()["train"][0]
    second = "Ein zweites Segment, čudno: good service."
    rows.append(make_row(tok, second, split="train", votes=1, questions=questions))
    rows[-1] = rows[-1].model_copy(
        update={"id": f"{document_id(first_train)[7:19]}:0001", "group_id": document_id(first_train)}
    )
    return rows


def make_producer(rows: list[Row], **changes: Any) -> Producer:
    fields: dict[str, Any] = {
        "version": "0.4.1",
        "created_at": "2026-09-24T19:15:00Z",
        "skill": "polarity",
        "predictor": "classify",
        "input_field": "text",
        "derive_version": 1,
        "binds": ProducerBinds(signature_sha256=HEX, question_set_sha256=HEX, skill_source_sha256=HEX),
        "questions": QUESTIONS,
        "teacher": TeacherSpec(model="openai/teacher::nothink@32768", k=4, temperature=0.7, dspy="3.2.1"),
        "campaign": "c-20260924T191500Z",
        "round": 0,
        "parent_dataset": None,
        "sources": [SourceInfo(doc_id=r.group_id, origin=f"reviews.jsonl:{i}", segments=1)
                    for i, r in enumerate(rows, start=1)],
        "label_stats": compute_label_stats(rows, unlabeled=2),
    }
    return Producer(**{**fields, **changes})


@pytest.fixture
def written(distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer) -> Path:
    rows = make_rows(tok)
    ref = write_dataset(distill / "inbox", base=base, producer=make_producer(rows), rows=rows)
    return ref.path


# --------------------------------------------------------------------------- base checkpoints


def test_constants() -> None:
    assert SCHEMA_VERSION == 1
    assert DEFAULT_BASE == "laya-multilingual@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    assert DATASET_FILES == ("manifest.json", "train.jsonl", "calib.jsonl", "heldout.jsonl", "pool.jsonl")
    assert PROB_SUM_TOL == 1e-6


def test_load_base_checkpoint(distill: Path) -> None:
    b = load_base_checkpoint(distill, BASE)
    path = distill / "base" / BASE
    assert (b.name, b.revision, b.path) == ("laya-multilingual", "0000test", path)
    assert (b.max_len, b.head_max_len) == (FIXTURE_MAX_LEN, FIXTURE_HEAD_MAX_LEN)
    assert b.weights_sha256 == file_sha256(path / "model.safetensors")
    assert b.tokenizer_sha256 == file_sha256(path / "tokenizer" / "tokenizer.json")
    assert b.ref() == BaseCheckpointRef(
        name="laya-multilingual", revision="0000test",
        weights_sha256=b.weights_sha256, tokenizer_sha256=b.tokenizer_sha256,
    )
    assert b.tokenizer().special == SequenceTokenizer.from_dir(FIXTURE / "tokenizer").special


def test_load_base_checkpoint_resolves_devais_rev12_directory(tmp_path: Path) -> None:
    # devai stages base/<name>@<revision[:12]>/ (laya_trainer catalog.checkpoint_dirname)
    # and its check_base compares the manifest's revision with the catalog's full one.
    distill = tmp_path / "distill"
    path = make_base_checkpoint(distill, "laya-multilingual@55cf4c4ebb4e")
    b = load_base_checkpoint(distill, DEFAULT_BASE)
    assert b.path == path
    assert b.ref().revision == "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def test_load_base_checkpoint_defaults_to_laya_lengths(distill: Path) -> None:
    (distill / "base" / BASE / "rl_agent_config.json").write_text('{"encoder": "x"}', encoding="utf-8")
    b = load_base_checkpoint(distill, BASE)
    assert (b.max_len, b.head_max_len) == (512, 192)


@pytest.mark.parametrize("missing", ["rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json"])
def test_load_base_checkpoint_missing_file(distill: Path, missing: str) -> None:
    (distill / "base" / BASE / missing).unlink()
    with pytest.raises(DistillError, match=re.escape(str(distill / "base" / BASE / missing))):
        load_base_checkpoint(distill, BASE)


def test_load_base_checkpoint_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(DistillError, match=re.escape(str(tmp_path / "base" / BASE))):
        load_base_checkpoint(tmp_path, BASE)


@pytest.mark.parametrize(
    ("config", "problem"),
    [("{not json", "cannot read"), ("[1]", "not a JSON object"),
     ('{"max_len": "1024"}', "max_len"), ('{"head_max_len": 0}', "head_max_len"),
     ('{"max_len": true}', "max_len")],
)
def test_load_base_checkpoint_bad_config(distill: Path, config: str, problem: str) -> None:
    (distill / "base" / BASE / "rl_agent_config.json").write_text(config, encoding="utf-8")
    with pytest.raises(DistillError, match=problem):
        load_base_checkpoint(distill, BASE)


@pytest.mark.parametrize(
    "base", ["laya-multilingual", "a@b@c", "@rev", "name@", "../x@rev", "name@../x", "na/me@rev", ".x@rev"]
)
def test_load_base_checkpoint_rejects_bad_names(distill: Path, base: str) -> None:
    with pytest.raises(DistillError):
        load_base_checkpoint(distill, base)


# --------------------------------------------------------------------------- rows


def test_gold_from_votes_follows_key_order() -> None:
    gold = gold_from_votes(KEYS, {"positive": 1, "mixed": 5, "negative": 2})
    assert list(gold.probabilities) == list(KEYS)
    assert gold.probabilities == {"negative": 0.25, "neutral": 0.0, "mixed": 0.625, "positive": 0.125}
    assert gold.label == "mixed"
    assert math.isclose(sum(gold.probabilities.values()), 1.0)


def test_gold_from_votes_breaks_ties_by_key_order() -> None:
    assert gold_from_votes(KEYS, {"positive": 2, "neutral": 2}).label == "neutral"
    assert gold_from_votes(("b", "a"), {"a": 1, "b": 1}).label == "b"


def test_gold_from_votes_needs_a_vote() -> None:
    with pytest.raises(ValueError, match="no votes"):
        gold_from_votes(KEYS, {"negative": 0})
    with pytest.raises(ValueError, match="unknown"):
        gold_from_votes(KEYS, {"bogus": 1})


def test_build_row(tok: SequenceTokenizer) -> None:
    text = "Die Lieferung kam zu spät, aber good service."
    group_id = document_id(text)
    gold = {"polarity": gold_from_votes(KEYS, VOTES[1])}
    teacher = TeacherInfo(k=4, parse_failures={"polarity": 0})
    row = build_row(group_id=group_id, index=4, text=text, input_field="text", questions=QUESTIONS,
                    split="train", tokenizer=tok, **LIMITS, gold=gold, teacher=teacher)
    assert row.id == f"{group_id[7:19]}:0004"
    assert (row.group_id, row.split, row.synthetic) == (group_id, "train", False)
    assert row.state == {"text": text}
    assert row.questions == QUESTIONS
    assert (row.gold, row.teacher) == (gold, teacher)
    encoded = tok.encode({"text": text}, prepare(QUESTIONS), **LIMITS)["polarity"]
    tokens = row.student_tokens["polarity"]
    assert tokens.n == len(encoded.input_ids)
    assert tokens.ids_sha256 == content_sha256(list(encoded.input_ids))
    pool = build_row(group_id=group_id, index=0, text=text, input_field="text", questions=QUESTIONS,
                     split="pool", tokenizer=tok, **LIMITS)
    assert pool.gold is None and pool.teacher is None


def test_row_forbids_extra_fields_and_is_frozen(tok: SequenceTokenizer) -> None:
    row = make_row(tok, "good service")
    with pytest.raises(ValueError):
        Row.model_validate({**row.model_dump(), "extra": 1})
    with pytest.raises(ValueError):
        row.id = "x"


def test_compute_label_stats(tok: SequenceTokenizer) -> None:
    rows = [
        make_row(tok, "one", split="train", votes=0),
        make_row(tok, "two", split="calib", votes=1),
        make_row(tok, "three", split="heldout", votes=1),
        make_row(tok, "four", split="pool"),
    ]
    stats = compute_label_stats(rows, unlabeled=5)
    assert stats.rows == 3
    assert stats.unlabeled == 5
    assert stats.parse_failures == {"polarity": 2}
    assert stats.label_histogram == {"polarity": {"negative": 1, "neutral": 0, "mixed": 2, "positive": 0}}
    assert list(stats.label_histogram["polarity"]) == list(KEYS)
    empty = compute_label_stats([], unlabeled=0)
    assert (empty.rows, empty.parse_failures, empty.label_histogram) == (0, {}, {})


# --------------------------------------------------------------------------- write_dataset


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def test_write_dataset_layout(distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer) -> None:
    rows = make_rows(tok)
    producer = make_producer(rows)
    ref = write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    manifest_bytes = (ref.path / "manifest.json").read_bytes()
    assert ref.manifest_sha256 == bytes_sha256(manifest_bytes)
    assert ref.id == dataset_id(ref.manifest_sha256)
    assert ref.id.startswith("ds-") and len(ref.id) == 15
    assert ref.path == distill / "inbox" / ref.id
    assert sorted(p.name for p in ref.path.iterdir()) == sorted([*DATASET_FILES, SUMS_FILE])
    assert (ref.path / SUMS_FILE).read_text(encoding="utf-8") == format_sha256sums(
        {name: file_sha256(ref.path / name) for name in DATASET_FILES}
    )
    assert not [p for p in (distill / "inbox").iterdir() if p.name.startswith(".tmp-")]

    text = manifest_bytes.decode("utf-8")
    raw = json.loads(text)
    assert text == json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    assert list(raw) == ["schema_version", "base_checkpoint", "max_len", "head_max_len", "splits",
                         "files", "producer"]
    assert '"parent_dataset": null' in text and '"augmentation": null' in text
    manifest = DatasetManifest.model_validate_json(manifest_bytes)
    assert manifest.producer == producer
    assert manifest.base_checkpoint == base.ref()
    assert (manifest.max_len, manifest.head_max_len) == (FIXTURE_MAX_LEN, FIXTURE_HEAD_MAX_LEN)
    assert manifest.splits.method == SPLIT_METHOD
    assert manifest.splits.counts == {"train": 3, "calib": 2, "heldout": 2, "pool": 2}
    for split in SPLITS:
        name = f"{split}.jsonl"
        assert manifest.files[name].sha256 == file_sha256(ref.path / name)
        assert manifest.files[name].rows == manifest.splits.counts[split]


def test_write_dataset_row_bytes(written: Path) -> None:
    for split in SPLITS:
        lines = read_lines(written / f"{split}.jsonl")
        ids = []
        for line in lines:
            data = json.loads(line)
            assert line == canonical_json(data) + "\n"  # compact, key order kept
            assert "č" in line and "\\u" not in line  # non-ASCII written literally
            assert list(data)[:6] == ["id", "group_id", "split", "synthetic", "state", "questions"]
            assert data["synthetic"] is False
            assert data["split"] == split
            if split == "pool":
                assert "gold" not in data and "teacher" not in data
                assert list(data) == ["id", "group_id", "split", "synthetic", "state", "questions",
                                      "student_tokens"]
            else:
                assert list(data)[6:] == ["gold", "teacher", "student_tokens"]
            ids.append(data["id"])
        assert ids == sorted(ids)


def test_questions_are_written_verbatim_with_their_nulls(
    distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer
) -> None:
    described = {"polarity": {**POLARITY, "criteria": {"negative": None, "neutral": "no sentiment",
                                                        "mixed": "", "positive": None}}}
    rows = make_rows(tok, described)
    ref = write_dataset(distill / "inbox", base=base, producer=make_producer(rows, questions=described),
                        rows=rows)
    line = read_lines(ref.path / "pool.jsonl")[0]
    assert '"criteria":{"negative":null,"neutral":"no sentiment","mixed":"","positive":null}' in line
    assert load_dataset(distill, ref.id).rows["pool"][0].questions == described


def test_write_dataset_is_idempotent(distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer) -> None:
    rows = make_rows(tok)
    producer = make_producer(rows)
    first = write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    second = write_dataset(distill / "inbox", base=base, producer=producer, rows=list(reversed(rows)))
    assert second == first
    assert [p.name for p in (distill / "inbox").iterdir()] == [first.id]


def test_write_dataset_refuses_a_different_existing_dir(
    distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer
) -> None:
    rows = make_rows(tok)
    producer = make_producer(rows)
    ref = write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    (ref.path / SUMS_FILE).write_text("", encoding="utf-8")
    with pytest.raises(DistillError, match="already exists"):
        write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    assert [p.name for p in (distill / "inbox").iterdir()] == [ref.id]


def test_write_dataset_validates_before_publishing(
    distill: Path, base: BaseCheckpoint, tok: SequenceTokenizer
) -> None:
    rows = make_rows(tok)
    other = {"polarity": {**POLARITY, "instructions": "Something else."}}
    with pytest.raises(DatasetContractError, match=r"\.jsonl:1: questions"):
        write_dataset(distill / "inbox", base=base, producer=make_producer(rows, questions=other), rows=rows)
    assert list((distill / "inbox").iterdir()) == []


def test_write_dataset_needs_the_distill_dir(
    tmp_path: Path, base: BaseCheckpoint, tok: SequenceTokenizer
) -> None:
    rows = make_rows(tok)
    with pytest.raises(DistillError, match="does not exist: mount the devai laya volume or set AIAGENT_DISTILL_DIR"):
        write_dataset(tmp_path / "nowhere" / "inbox", base=base, producer=make_producer(rows), rows=rows)
    assert not (tmp_path / "nowhere").exists()


# --------------------------------------------------------------------------- validate_dataset

Mutator = Callable[[dict[str, Any]], dict[str, Any] | None]


def edit_rows(directory: Path, split: str, mutate: Mutator, *, line: int | None = 1) -> None:
    """Apply `mutate` to one line (1-based) or every line (line=None); None drops the row."""
    path = directory / f"{split}.jsonl"
    out = []
    for number, text in enumerate(read_lines(path), start=1):
        data = json.loads(text)
        if line is None or number == line:
            data = mutate(data)
        if data is not None:
            # compact like the writer, but NaN stays writable so the validator can see it
            out.append(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")
    path.write_text("".join(out), encoding="utf-8")


def edit_manifest(directory: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = directory / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sums_only(directory: Path) -> None:
    """Rewrite SHA256SUMS only (manifest.files left as mutated)."""
    paths = parse_sha256sums((directory / SUMS_FILE).read_text(encoding="utf-8"))
    sums = {rel: file_sha256(directory / rel) for rel in paths}
    (directory / SUMS_FILE).write_text(format_sha256sums(sums), encoding="utf-8")


def with_(key: str, value: Any) -> Mutator:
    return lambda data: {**data, key: value}


def gold_with(**answer: Any) -> Mutator:
    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        gold = {**data["gold"]["polarity"], **answer}
        return {**data, "gold": {"polarity": gold}}
    return mutate


def without(key: str) -> Mutator:
    return lambda data: {k: v for k, v in data.items() if k != key}


def test_validate_dataset_accepts_what_write_wrote(written: Path, tok: SequenceTokenizer) -> None:
    manifest = validate_dataset(written, tokenizer=tok)
    assert manifest.producer.skill == "polarity"


def test_validate_accepts_train_and_pool_sharing_a_group(written: Path, tok: SequenceTokenizer) -> None:
    group = json.loads(read_lines(written / "train.jsonl")[0])["group_id"]
    edit_rows(written, "pool", with_("group_id", group))
    edit_rows(written, "train", with_("synthetic", True))  # synthetic is allowed in train and pool
    resum(written)
    validate_dataset(written, tokenizer=tok)


def _drop_pool_sum(d: Path) -> None:
    sums = parse_sha256sums((d / SUMS_FILE).read_text(encoding="utf-8"))
    del sums["pool.jsonl"]
    (d / SUMS_FILE).write_text(format_sha256sums(sums), encoding="utf-8")


def _extra_sum(d: Path) -> None:
    (d / "extra.txt").write_text("x", encoding="utf-8")
    sums = parse_sha256sums((d / SUMS_FILE).read_text(encoding="utf-8"))
    sums["extra.txt"] = file_sha256(d / "extra.txt")
    (d / SUMS_FILE).write_text(format_sha256sums(sums), encoding="utf-8")


FILE_LEVEL: list[tuple[str, Callable[[Path], None], str]] = [
    ("sums missing", lambda d: (d / SUMS_FILE).unlink(), r"^SHA256SUMS: "),
    ("sums unparseable", lambda d: (d / SUMS_FILE).write_text("nonsense\n", encoding="utf-8"),
     r"^SHA256SUMS line 1: "),
    ("sums lack a file", _drop_pool_sum, r"^SHA256SUMS: .*pool\.jsonl"),
    ("sums list another file", _extra_sum, r"^SHA256SUMS: .*extra\.txt"),
    ("file changed", lambda d: (d / "calib.jsonl").write_text("{}\n", encoding="utf-8"),
     r"^calib\.jsonl: sha256 does not match SHA256SUMS"),
    ("file missing", lambda d: (d / "heldout.jsonl").unlink(), r"^heldout\.jsonl: missing"),
    ("schema version", lambda d: (edit_manifest(d, lambda m: m.update(schema_version=2)), resum(d)),
     r"^manifest\.json: .*schema_version"),
    ("manifest not json", lambda d: ((d / "manifest.json").write_text("{", encoding="utf-8"), sums_only(d)),
     r"^manifest\.json: invalid JSON"),
    ("files sha", lambda d: (edit_manifest(d, lambda m: m["files"]["train.jsonl"].update(sha256="0" * 64)),
                             sums_only(d)), r"^manifest\.json: files\['train\.jsonl'\]"),
    ("files rows", lambda d: (edit_manifest(d, lambda m: m["files"]["pool.jsonl"].update(rows=7)),
                              sums_only(d)), r"^manifest\.json: files\['pool\.jsonl'\]"),
    ("files keys", lambda d: (edit_manifest(d, lambda m: m["files"].pop("pool.jsonl")), sums_only(d)),
     r"^manifest\.json: files must list"),
    ("counts", lambda d: (edit_manifest(d, lambda m: m["splits"]["counts"].update(train=9)), sums_only(d)),
     r"^manifest\.json: splits\.counts\['train'\] is 9"),
    ("counts keys", lambda d: (edit_manifest(d, lambda m: m["splits"]["counts"].pop("pool")), sums_only(d)),
     r"^manifest\.json: splits\.counts\['pool'\] is None"),
    ("bad questions", lambda d: (edit_manifest(d, lambda m: m["producer"].update(questions={"q": {"type": "x"}})),
                                 resum(d)), r"^manifest\.json: producer\.questions: question 'q'"),
    ("train empty", lambda d: (edit_rows(d, "train", lambda r: None, line=None),
                               edit_manifest(d, lambda m: m["splits"]["counts"].update(train=0)), resum(d)),
     r"^train\.jsonl: no rows"),
]


@pytest.mark.parametrize(("name", "mutate", "match"), FILE_LEVEL, ids=[c[0] for c in FILE_LEVEL])
def test_validate_rejects_file_level_problems(
    written: Path, tok: SequenceTokenizer, name: str, mutate: Callable[[Path], None], match: str
) -> None:
    mutate(written)
    with pytest.raises(DatasetContractError, match=match):
        validate_dataset(written, tokenizer=tok)


def _probabilities(**values: float) -> Mutator:
    return gold_with(probabilities=values)


def _long_state(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, "state": {"text": "good service " * 200}}


ROW_LEVEL: list[tuple[str, str, Mutator, str]] = [
    ("not an object", "train", lambda data: ["x"], r"^train\.jsonl:1: not a JSON object"),
    ("extra key", "train", with_("extra", 1), r"^train\.jsonl:1: .*extra"),
    ("wrong split", "train", with_("split", "calib"), r"^train\.jsonl:1: split 'calib'"),
    ("state key", "calib", with_("state", {"body": "x"}), r"^calib\.jsonl:1: state must be"),
    ("state not str", "calib", with_("state", {"text": 3}), r"^calib\.jsonl:1: .*state"),
    ("questions", "heldout", with_("questions", {"polarity": {**POLARITY, "criteria": list(reversed(KEYS))}}),
     r"^heldout\.jsonl:1: questions differ from producer\.questions"),
    ("gold null in pool", "pool", with_("gold", None), r"^pool\.jsonl:1: null at gold"),
    ("gold in pool", "pool", with_("gold", {"polarity": {"label": "mixed", "probabilities": {}}}),
     r"^pool\.jsonl:1: a pool row has 'gold'"),
    ("teacher in pool", "pool", with_("teacher", {"k": 4, "parse_failures": {}}),
     r"^pool\.jsonl:1: a pool row has 'teacher'"),
    ("gold null", "train", with_("gold", None), r"^train\.jsonl:1: null at gold"),
    ("nested null", "calib", gold_with(label=None), r"^calib\.jsonl:1: null at gold\.polarity\.label"),
    ("null in a list", "pool", with_("state", {"text": ["a", None]}), r"^pool\.jsonl:1: null at state\.text\[1\]"),
    ("no teacher", "heldout", without("teacher"), r"^heldout\.jsonl:1: missing 'teacher'"),
    ("no gold", "train", without("gold"), r"^train\.jsonl:1: missing 'gold'"),
    ("gold qids", "train", with_("gold", {}), r"^train\.jsonl:1: gold must answer exactly"),
    ("key order", "train", _probabilities(neutral=0.25, negative=0.25, mixed=0.25, positive=0.25),
     r"^train\.jsonl:1: gold\['polarity'\]\.probabilities keys must be"),
    ("missing key", "train", _probabilities(negative=0.5, neutral=0.5, mixed=0.0),
     r"^train\.jsonl:1: gold\['polarity'\]\.probabilities keys must be"),
    ("negative", "train", _probabilities(negative=-0.25, neutral=0.5, mixed=0.5, positive=0.25),
     r"^train\.jsonl:1: gold\['polarity'\]\.probabilities must be finite and >= 0"),
    ("nan", "train", _probabilities(negative=math.nan, neutral=0.5, mixed=0.5, positive=0.0),
     r"^train\.jsonl:1: gold\['polarity'\]\.probabilities must be finite and >= 0"),
    ("sum", "train", _probabilities(negative=0.5, neutral=0.5, mixed=0.5, positive=0.0),
     r"^train\.jsonl:1: gold\['polarity'\]\.probabilities sum to 1\.5"),
    ("label", "train", gold_with(label="neutral", probabilities={"negative": 0.5, "neutral": 0.0,
                                                                 "mixed": 0.5, "positive": 0.0}),
     r"^train\.jsonl:1: gold\['polarity'\]\.label is 'neutral', not the first argmax 'negative'"),
    ("synthetic calib", "calib", with_("synthetic", True), r"^calib\.jsonl:1: synthetic rows are not allowed"),
    ("synthetic heldout", "heldout", with_("synthetic", True), r"^heldout\.jsonl:1: synthetic"),
    ("tokens qids", "train", with_("student_tokens", {}), r"^train\.jsonl:1: student_tokens must cover"),
    ("tokens n", "train",
     lambda d: {**d, "student_tokens": {"polarity": {**d["student_tokens"]["polarity"], "n": 3}}},
     r"^train\.jsonl:1: student_tokens\['polarity'\] does not match the base tokenizer"),
    ("tokens hash", "pool",
     lambda d: {**d, "student_tokens": {"polarity": {**d["student_tokens"]["polarity"], "ids_sha256": HEX}}},
     r"^pool\.jsonl:1: student_tokens\['polarity'\] does not match the base tokenizer"),
    ("truncated state", "heldout", _long_state, r"^heldout\.jsonl:1: the state has \d+ tokens; question 'polarity' leaves room for \d+"),
]


@pytest.mark.parametrize(("name", "split", "mutate", "match"), ROW_LEVEL, ids=[c[0] for c in ROW_LEVEL])
def test_validate_rejects_row_level_problems(
    written: Path, tok: SequenceTokenizer, name: str, split: str, mutate: Mutator, match: str
) -> None:
    edit_rows(written, split, mutate)
    resum(written)
    with pytest.raises(DatasetContractError, match=match):
        validate_dataset(written, tokenizer=tok)


def test_validate_rejects_an_invalid_line(written: Path, tok: SequenceTokenizer) -> None:
    path = written / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{oops\n", encoding="utf-8")
    edit_manifest(written, lambda m: m["splits"]["counts"].update(train=4))
    resum(written)
    with pytest.raises(DatasetContractError, match=r"^train\.jsonl:4: invalid JSON"):
        validate_dataset(written, tokenizer=tok)


def test_validate_rejects_a_duplicate_id(written: Path, tok: SequenceTokenizer) -> None:
    first = json.loads(read_lines(written / "train.jsonl")[0])
    copy = {k: v for k, v in first.items() if k not in ("gold", "teacher")} | {"split": "pool"}
    path = written / "pool.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + canonical_json(copy) + "\n", encoding="utf-8")
    edit_manifest(written, lambda m: m["splits"]["counts"].update(pool=3))
    resum(written)
    with pytest.raises(DatasetContractError, match=r"^pool\.jsonl:3: duplicate id"):
        validate_dataset(written, tokenizer=tok)


@pytest.mark.parametrize(("split", "other"), [("heldout", "calib"), ("calib", "train"), ("heldout", "pool")])
def test_validate_rejects_shared_groups(written: Path, tok: SequenceTokenizer, split: str, other: str) -> None:
    group = json.loads(read_lines(written / f"{other}.jsonl")[0])["group_id"]
    edit_rows(written, split, with_("group_id", group))
    resum(written)
    with pytest.raises(DatasetContractError, match=f"group_id {group} is also in"):
        validate_dataset(written, tokenizer=tok)


def test_validate_checks_every_option_marker(written: Path, tok: SequenceTokenizer) -> None:
    """40 options with 3-word descriptions overflow head_max_len (the fixture's error row)."""
    crowded = {"type": "choice", "instructions": "Pick.",
               "criteria": {f"x{i}": "good service here" for i in range(40)}}
    one_hot = {key: float(key == "x0") for key in crowded["criteria"]}

    def crowd(data: dict[str, Any]) -> dict[str, Any]:
        gold = {"gold": {"polarity": {"label": "x0", "probabilities": one_hot}}} if "gold" in data else {}
        return {**data, "questions": {"polarity": crowded}, **gold}

    edit_manifest(written, lambda m: m["producer"].update(questions={"polarity": crowded}))
    for split in SPLITS:
        edit_rows(written, split, crowd, line=None)
    resum(written)
    with pytest.raises(DatasetContractError, match=r"^\w+\.jsonl:1: question 'polarity' options exceed head_max_len=96"):
        validate_dataset(written, tokenizer=tok)


# --------------------------------------------------------------------------- load / find


def test_load_dataset_from_the_inbox(written: Path, tok: SequenceTokenizer) -> None:
    distill = written.parent.parent
    ds = load_dataset(distill, written.name)
    assert isinstance(ds, Dataset)
    assert ds.ref.id == written.name and ds.ref.path == written
    assert ds.ref.manifest_sha256 == file_sha256(written / "manifest.json")
    assert ds.manifest == validate_dataset(written, tokenizer=tok)
    for split in SPLITS:
        lines = read_lines(written / f"{split}.jsonl")
        assert [r.id for r in ds.rows[split]] == [json.loads(line)["id"] for line in lines]
        assert all(isinstance(r, Row) for r in ds.rows[split])
    assert load_dataset(distill, written.name, expect_manifest_sha256=ds.ref.manifest_sha256) == ds


def test_load_dataset_prefers_datasets(written: Path) -> None:
    distill = written.parent.parent
    accepted = distill / "datasets" / written.name
    shutil.copytree(written, accepted)
    (written / "pool.jsonl").write_text("tampered\n", encoding="utf-8")  # the inbox copy is ignored
    assert find_dataset(distill, written.name) == accepted
    assert load_dataset(distill, written.name).ref.path == accepted


def test_load_dataset_checks_the_expected_hash(written: Path) -> None:
    with pytest.raises(DistillError, match="expected"):
        load_dataset(written.parent.parent, written.name, expect_manifest_sha256=HEX)


def test_load_dataset_checks_sums(written: Path) -> None:
    (written / "pool.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="pool.jsonl"):
        load_dataset(written.parent.parent, written.name)


def test_load_dataset_checks_the_id(written: Path) -> None:
    distill = written.parent.parent
    shutil.copytree(written, distill / "inbox" / "ds-000000000000")
    with pytest.raises(DatasetContractError, match=f"hashes to {written.name}"):
        load_dataset(distill, "ds-000000000000")


def test_find_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(DistillError, match="not found"):
        find_dataset(tmp_path, "ds-000000000000")
    with pytest.raises(DistillError, match="unsafe"):
        find_dataset(tmp_path, "../ds-x")


# --------------------------------------------------------------------------- repair


def test_derive_repair_dataset(written: Path, base: BaseCheckpoint, tok: SequenceTokenizer) -> None:
    distill = written.parent.parent
    parent = load_dataset(distill, written.name)
    pool = parent.rows["pool"]
    labeled_row = pool[0]
    labeled = [build_row(group_id=labeled_row.group_id, index=int(labeled_row.id[-4:]),
                         text=labeled_row.state["text"], input_field="text", questions=QUESTIONS,
                         split="pool", tokenizer=tok, **LIMITS,
                         gold={"polarity": gold_from_votes(KEYS, VOTES[2])},
                         teacher=TeacherInfo(k=4, parse_failures={"polarity": 1}))]
    producer, rows = derive_repair_dataset(
        parent, selected=[pool[0].id, pool[1].id], labeled=labeled,
        created_at="2026-09-25T08:00:00Z", version="0.5.0", unlabeled_added=1,
    )
    old = parent.manifest.producer
    assert producer.round == old.round + 1
    assert producer.parent_dataset == parent.ref.id
    assert (producer.created_at, producer.version) == ("2026-09-25T08:00:00Z", "0.5.0")
    for field in ("skill", "predictor", "input_field", "derive_version", "binds", "questions",
                  "teacher", "campaign", "sources", "augmentation"):
        assert getattr(producer, field) == getattr(old, field), field
    assert producer.label_stats.rows == old.label_stats.rows + 1
    assert producer.label_stats.unlabeled == old.label_stats.unlabeled + 1
    assert producer.label_stats.parse_failures["polarity"] == old.label_stats.parse_failures["polarity"] + 1
    assert producer.label_stats == compute_label_stats(rows, unlabeled=old.label_stats.unlabeled + 1)

    by_split = {s: [r for r in rows if r.split == s] for s in SPLITS}
    assert by_split["train"][: len(parent.rows["train"])] == list(parent.rows["train"])
    moved = by_split["train"][len(parent.rows["train"]):]
    assert [r.id for r in moved] == [labeled_row.id] and moved[0].split == "train"
    assert by_split["calib"] == list(parent.rows["calib"])
    assert by_split["heldout"] == list(parent.rows["heldout"])
    assert by_split["pool"] == [r for r in pool if r.id not in (pool[0].id, pool[1].id)] == []

    child = write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    assert child.id != parent.ref.id
    loaded = load_dataset(distill, child.id)
    assert loaded.manifest.producer.parent_dataset == parent.ref.id
    assert loaded.rows["calib"] == parent.rows["calib"]
    assert loaded.rows["heldout"] == parent.rows["heldout"]
    assert loaded.manifest.splits.counts == {"train": 4, "calib": 2, "heldout": 2, "pool": 0}


def test_gold_answer_model() -> None:
    answer = GoldAnswer(label="a", probabilities={"a": 1.0})
    assert answer.model_dump() == {"label": "a", "probabilities": {"a": 1.0}}
