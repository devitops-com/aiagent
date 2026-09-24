"""laya's input sequence builder: parity with laya on the fixture, validation, special tokens."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer

from aiagent.exceptions import QuestionError, System1Error
from aiagent.system1.sequence import (
    NOUL_DEFAULT_FALSE,
    NOUL_DEFAULT_TRUE,
    OPTION_TOKEN_CAP,
    Encoded,
    SequenceTokenizer,
    SpecialTokens,
    check_question,
    prepare,
    render_criterion,
    serialize_state,
    to_internal,
)
from system1_helpers import FIXTURE, FIXTURE_HEAD_MAX_LEN, FIXTURE_MAX_LEN, golden_rows

LIMITS = {"max_len": FIXTURE_MAX_LEN, "head_max_len": FIXTURE_HEAD_MAX_LEN}
POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": ["negative", "neutral", "mixed", "positive"],
}


@pytest.fixture(scope="module")
def tok() -> SequenceTokenizer:
    return SequenceTokenizer.from_dir(FIXTURE / "tokenizer")


def copy_tokenizer(tmp_path: Path) -> Path:
    target = tmp_path / "tokenizer"
    shutil.copytree(FIXTURE / "tokenizer", target)
    return target


def expected_rows() -> list[dict[str, Any]]:
    return [row for row in golden_rows() if "expected" in row]


def assert_golden_ids(tok: SequenceTokenizer) -> int:
    checked = 0
    for row in expected_rows():
        encoded = tok.encode(row["state"], prepare(row["questions"]), **LIMITS)
        assert list(encoded) == list(row["questions"])
        for qid, want in row["expected"].items():
            assert encoded[qid] == Encoded(tuple(want["input_ids"]), tuple(want["markers"])), (
                f"{row['id']}/{qid}"
            )
            checked += 1
    return checked


# --------------------------------------------------------------------------- parity with laya


def test_encode_matches_laya_on_every_golden_row(tok: SequenceTokenizer) -> None:
    assert assert_golden_ids(tok) >= 30


def test_error_rows_raise_like_laya(tok: SequenceTokenizer) -> None:
    errors = [row for row in golden_rows() if "error" in row]
    assert errors
    for row in errors:
        with pytest.raises(QuestionError, match=row["error"]):
            tok.encode(row["state"], prepare(row["questions"]), **LIMITS)


def test_saved_truncation_and_padding_state_is_reset_on_load(tmp_path: Path) -> None:
    target = copy_tokenizer(tmp_path)
    raw = Tokenizer.from_file(str(target / "tokenizer.json"))
    raw.enable_truncation(3)
    raw.enable_padding(length=256)
    raw.save(str(target / "tokenizer.json"))
    saved = json.loads((target / "tokenizer.json").read_text(encoding="utf-8"))
    assert saved["truncation"] is not None and saved["padding"] is not None

    assert assert_golden_ids(SequenceTokenizer.from_dir(target)) >= 30


def test_state_ids_replace_the_mask_token_and_keep_other_markers(tok: SequenceTokenizer) -> None:
    assert tok.state_ids("good <mask> service") == tok.ids("good   service")
    assert tok.special.mask_id not in tok.state_ids({"text": "a <mask> b"})
    assert tok.state_ids("[MASK]") == tok.ids("[MASK]") != []


# --------------------------------------------------------------------------- check_question

INVALID = [
    ("not-a-dict", "definition must be a dict"),
    ({"type": "bool", "instructions": "i"}, "unknown type"),
    ({"type": ["choice"], "instructions": "i", "criteria": ["a"]}, "unknown type"),
    ({"type": "noul"}, "no 'instructions'"),
    ({"type": "choice", "instructions": "i"}, "choice question takes 'criteria'"),
    ({"type": "choice", "instructions": "i", "criteria": "a"}, "choice question takes 'criteria'"),
    ({"type": "choice", "instructions": "i", "criteria": []}, "at least one criterion"),
    ({"type": "choice", "instructions": "i", "criteria": {}}, "at least one criterion"),
    ({"type": "choice", "instructions": "i", "criteria": [1, "b"]}, "choice keys must be strings"),
    ({"type": "choice", "instructions": "i", "criteria": [["a"]]}, "choice keys must be strings"),
    ({"type": "choice", "instructions": "i", "criteria": {1: "a"}}, "choice keys must be strings"),
    ({"type": "score", "instructions": "i", "criteria": {"a": 1}}, "score question takes"),
    ({"type": "score", "instructions": "i"}, "score question takes"),
    ({"type": "score", "instructions": "i", "criteria": []}, "at least one level"),
    ({"type": "noul", "instructions": "i", "criteria": ["true"]}, "noul question takes 'criteria'"),
    ({"type": "noul", "instructions": "i", "criteria": {"yes": "x"}}, "keyed only 'true'/'false'"),
    (
        {"type": "choice", "instructions": "i", "criteria": ["a"], "labels": {}},
        "'labels' is only supported for noul",
    ),
    ({"type": "noul", "instructions": "i", "labels": {"false": "A"}}, "noul labels must map"),
    ({"type": "noul", "instructions": "i", "labels": ["A", "B"]}, "noul labels must map"),
    ({"type": "noul", "instructions": "i", "labels": {"false": "A", "true": " A "}}, "noul labels"),
    ({"type": "noul", "instructions": "i", "labels": {"false": " ", "true": "B"}}, "noul labels"),
    ({"type": "noul", "instructions": "i", "labels": {"false": 1, "true": "B"}}, "noul labels"),
]


@pytest.mark.parametrize(("qdef", "problem"), INVALID)
def test_check_question_rejects(qdef: object, problem: str) -> None:
    with pytest.raises(QuestionError, match=problem) as info:
        check_question("q1", qdef)
    assert str(info.value).startswith("question 'q1': ")


@pytest.mark.parametrize(
    "qdef",
    [
        POLARITY,
        {"type": "choice", "instructions": {"k": "v"}, "criteria": {"a": None, "b": 0}},
        {"type": "score", "instructions": "i", "criteria": ["low", {"x": 1}]},
        {"type": "noul", "instructions": "i"},
        {"type": "noul", "instructions": "i", "criteria": {}},
        {"type": "noul", "instructions": "i", "criteria": {True: "t", "FALSE": "f"}},
        {"type": "noul", "instructions": "i", "labels": {"false": "B", "true": "A"}},
    ],
)
def test_check_question_accepts(qdef: object) -> None:
    check_question("q1", qdef)


def test_prepare_checks_and_keeps_question_order() -> None:
    internal = prepare({"z": POLARITY, "a": {"type": "noul", "instructions": "i"}})
    assert list(internal) == ["z", "a"]
    assert [q.qtype_index for q in internal.values()] == [0, 2]
    with pytest.raises(QuestionError, match="question 'bad'"):
        prepare({"ok": POLARITY, "bad": {"type": "choice", "instructions": "i", "criteria": []}})


# --------------------------------------------------------------------------- to_internal


def test_to_internal_turns_list_criteria_into_keys() -> None:
    q = to_internal(POLARITY)
    assert q.type == "choice"
    assert q.instructions == POLARITY["instructions"]
    assert q.keys == q.options == ("negative", "neutral", "mixed", "positive")


def test_to_internal_lower_cases_noul_keys_and_applies_labels() -> None:
    q = to_internal(
        {
            "type": "noul",
            "instructions": "late?",
            "criteria": {"True": "late", "FALSE": "on time"},
            "labels": {"false": " B ", "true": "A"},
        }
    )
    assert q.keys == ("false", "true")
    assert q.options == ("B: on time", "A: late")


def test_to_internal_noul_defaults() -> None:
    q = to_internal({"type": "noul", "instructions": "late?", "criteria": {"true": ""}})
    assert q.options == (f"false: {NOUL_DEFAULT_FALSE}", f"true: {NOUL_DEFAULT_TRUE}")


def test_to_internal_dict_instructions_keep_non_ascii() -> None:
    q = to_internal({"type": "noul", "instructions": {"task": "late?", "jezik": "hrvatski ž"}})
    assert q.instructions == '{"task": "late?", "jezik": "hrvatski ž"}'


def test_render_options_for_empty_zero_false_and_dict_criteria() -> None:
    choice = to_internal(
        {
            "type": "choice",
            "instructions": "i",
            "criteria": {"a": None, "b": "", "c": 0, "d": False, "e": {"x": 1, "y": "ž"}},
        }
    )
    assert choice.options == ("a", "b", "c: 0", "d: false", 'e: {"x": 1, "y": "ž"}')
    assert choice.keys == ("a", "b", "c", "d", "e")

    score = to_internal({"type": "score", "instructions": "i", "criteria": ["x", 0, ["a", 1]]})
    assert score.options == ("level 0: x", "level 1: 0", 'level 2: ["a", 1]')
    assert score.keys == ("0", "1", "2")

    noul = to_internal({"type": "noul", "instructions": "i", "criteria": {"false": 0, "true": False}})
    assert noul.options == ("false: 0", "true: false")


def test_render_criterion_and_serialize_state() -> None:
    assert render_criterion("as is") == "as is"
    assert render_criterion(None) == "null"
    assert render_criterion({"a": [1, 2]}) == '{"a": [1, 2]}'
    assert render_criterion(Path("/x")) == '"/x"'  # default=str
    assert serialize_state("as is") == "as is"
    assert serialize_state({"t": "ž", "n": [1, 2]}) == '{"t": "ž", "n": [1, 2]}'


# --------------------------------------------------------------------------- room / prefix / fits


def test_prefix_and_room(tok: SequenceTokenizer) -> None:
    q = to_internal(POLARITY)
    head = tok.prefix(q, head_max_len=FIXTURE_HEAD_MAX_LEN)
    special = tok.special
    assert head.input_ids[0] == special.cls_id and head.input_ids[-1] == special.sep_id
    assert [head.input_ids[m] for m in head.markers] == [special.mask_id] * 4
    room = tok.room(q, **LIMITS)
    assert room == FIXTURE_MAX_LEN - len(head.input_ids) - 1
    assert tok.room(q, max_len=len(head.input_ids), head_max_len=FIXTURE_HEAD_MAX_LEN) == 0


def test_a_state_of_exactly_room_tokens_is_kept_and_one_more_is_cut(tok: SequenceTokenizer) -> None:
    q = to_internal(POLARITY)
    room = tok.room(q, **LIMITS)
    head = tok.prefix(q, head_max_len=FIXTURE_HEAD_MAX_LEN)
    word = tok.ids("good")
    assert len(word) == 1

    exact = " ".join(["good"] * room)
    row = tok.build(q, tok.state_ids(exact), **LIMITS, truncate_left=False)
    assert row.input_ids == (*head.input_ids, *word * room, tok.special.sep_id)
    assert len(row.input_ids) == FIXTURE_MAX_LEN
    assert tok.fits(exact, {"p": q}, **LIMITS)

    longer = exact + " service"
    row = tok.build(q, tok.state_ids(longer), **LIMITS, truncate_left=False)
    assert row.input_ids == (*head.input_ids, *word * room, tok.special.sep_id)
    assert not tok.fits(longer, {"p": q}, **LIMITS)

    left = tok.build(q, tok.state_ids(longer), **LIMITS, truncate_left=True)
    assert left.input_ids[-2] == tok.ids("service")[0]


def test_encode_truncates_list_states_from_the_left(tok: SequenceTokenizer) -> None:
    q = prepare({"p": POLARITY})
    turns = ["late"] * 200 + ["excellent"]
    row = tok.encode(turns, q, **LIMITS)["p"]
    assert len(row.input_ids) == FIXTURE_MAX_LEN
    assert tok.ids('"]')[-1] == row.input_ids[-2]  # the newest turn survives
    assert not tok.fits(turns, q, **LIMITS)


def test_fits_needs_every_question_and_every_marker(tok: SequenceTokenizer) -> None:
    small = prepare({"p": POLARITY})
    assert tok.fits("good service", small, **LIMITS)
    assert tok.fits("good service", {}, **LIMITS)
    many = prepare({"m": {"type": "choice", "instructions": "i", "criteria": [f"x{i}" for i in range(80)]}})
    assert not tok.fits("", many, **LIMITS)
    with pytest.raises(QuestionError, match="question 'm' options exceed head_max_len=96"):
        tok.encode("", many, **LIMITS)


def test_option_cap_is_48_tokens(tok: SequenceTokenizer) -> None:
    long = " ".join(["good"] * 100)
    q = to_internal({"type": "choice", "instructions": "i", "criteria": {"a": long, "b": None}})
    head = tok.prefix(q, head_max_len=FIXTURE_HEAD_MAX_LEN)
    assert head.markers[1] - head.markers[0] == 1 + OPTION_TOKEN_CAP


# --------------------------------------------------------------------------- special tokens


def write_config(target: Path, **changes: object) -> None:
    path = target / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            config.pop(key)
        else:
            config[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")


def test_special_tokens_from_the_fixture(tok: SequenceTokenizer) -> None:
    assert tok.special == SpecialTokens(mask_token="<mask>", mask_id=4, cls_id=2, sep_id=3, pad_id=0)


def test_special_tokens_given_as_content_dicts(tmp_path: Path, tok: SequenceTokenizer) -> None:
    target = copy_tokenizer(tmp_path)
    write_config(
        target,
        mask_token={"content": "<mask>", "lstrip": True},
        cls_token={"content": "<bos>"},
        sep_token={"content": "<eos>"},
        pad_token={"content": "<pad>"},
    )
    assert SequenceTokenizer.from_dir(target).special == tok.special


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"sep_token": None}, "sep_token"),
        ({"pad_token": "<nope>"}, "<nope>"),
        ({"mask_token": {"id": 4}}, "mask_token"),
        ({"cls_token": 2}, "cls_token"),
    ],
)
def test_bad_special_tokens_raise(tmp_path: Path, changes: dict[str, object], problem: str) -> None:
    target = copy_tokenizer(tmp_path)
    write_config(target, **changes)
    with pytest.raises(System1Error, match=problem):
        SequenceTokenizer.from_dir(target)


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json"])
def test_missing_or_corrupt_tokenizer_files_raise(tmp_path: Path, name: str) -> None:
    target = copy_tokenizer(tmp_path)
    (target / name).unlink()
    with pytest.raises(System1Error, match=name):
        SequenceTokenizer.from_dir(target)
    (target / name).write_text("{not json", encoding="utf-8")
    with pytest.raises(System1Error, match=name):
        SequenceTokenizer.from_dir(target)


def test_a_tokenizer_config_that_is_not_an_object_raises(tmp_path: Path) -> None:
    target = copy_tokenizer(tmp_path)
    (target / "tokenizer_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(System1Error, match="not a JSON object"):
        SequenceTokenizer.from_dir(target)


# --------------------------------------------------------------------------- fixture sanity


def test_fixture_is_small_complete_and_host_path_free() -> None:
    files = [path for path in FIXTURE.rglob("*") if path.is_file()]
    assert sum(path.stat().st_size for path in files) < 1_000_000
    for rel in (
        "model.onnx",
        "model.onnx.data",
        "rl_agent_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "golden.jsonl",
        "provenance.json",
    ):
        assert (FIXTURE / rel).is_file(), rel
    for path in files:
        data = path.read_bytes()
        for needle in (b"/home/", b"/tmp/", b"scratchpad"):
            assert needle not in data, f"{path.name} contains {needle!r}"
    config = json.loads((FIXTURE / "rl_agent_config.json").read_text(encoding="utf-8"))
    assert (config["max_len"], config["head_max_len"]) == (FIXTURE_MAX_LEN, FIXTURE_HEAD_MAX_LEN)
    provenance = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["laya"] == {"version": "0.3.20", "commit": "23a1752"}


def all_questions() -> list[dict[str, Any]]:
    return [q for row in golden_rows() for q in row["questions"].values()]


def all_text() -> str:
    return json.dumps(golden_rows(), ensure_ascii=False)


def test_fixture_covers_the_states() -> None:
    rows = golden_rows()
    kinds = {type(row["state"]).__name__ for row in rows}
    assert kinds == {"str", "dict", "list"}
    assert any(row["state"] == "" for row in rows)
    text = all_text()
    for char in "čćšäü€":
        assert char in text, char
    states = json.dumps([row["state"] for row in rows], ensure_ascii=False)
    for marker in ("<mask>", "[MASK]", "zzunknown"):
        assert marker in states, marker
    instructions = json.dumps([q["instructions"] for q in all_questions()], ensure_ascii=False)
    assert "<mask>" in instructions and "[MASK]" in instructions
    assert any(
        isinstance(q["instructions"], dict)
        and not json.dumps(q["instructions"], ensure_ascii=False).isascii()
        for q in all_questions()
    )
    unk = 1
    assert any(unk in e["input_ids"] for row in expected_rows() for e in row["expected"].values())
    # truncation: a str/dict state cut on the right and a list state cut on the left
    full = [
        (type(row["state"]).__name__, e)
        for row in expected_rows()
        for e in row["expected"].values()
        if len(e["input_ids"]) == FIXTURE_MAX_LEN
    ]
    assert {"list"} <= {kind for kind, _ in full} and {"str", "dict"} & {kind for kind, _ in full}


def test_fixture_covers_the_questions() -> None:
    questions = all_questions()
    assert POLARITY in questions
    assert any(q.get("criteria") == ["no", "yes"] for q in questions)
    types = [q["type"] for q in questions]
    assert {"choice", "score", "noul"} <= set(types)
    score_levels = {len(q["criteria"]) for q in questions if q["type"] == "score"}
    assert {3, 10} <= score_levels
    choice_sizes = {len(q["criteria"]) for q in questions if q["type"] == "choice"}
    assert {2, 12} <= choice_sizes
    assert any(q["type"] == "noul" and "criteria" not in q for q in questions)
    assert any(
        q["type"] == "noul"
        and list(q["criteria"]) == ["True", "false"]
        and q["criteria"]["false"] == ""
        and q["labels"] == {"false": "B", "true": "A"}
        for q in questions
        if "labels" in q
    )
    mixed = next(
        q["criteria"]
        for q in questions
        if q["type"] == "choice" and isinstance(q["criteria"], dict) and None in q["criteria"].values()
    )
    values = list(mixed.values())
    assert "" in values and 0 in values and any(isinstance(v, dict) for v in values)
    assert any(isinstance(v, str) and v for v in values)
    squeeze = [
        q
        for q in questions
        if q["type"] == "choice"
        and isinstance(q["criteria"], dict)
        and len(q["criteria"]) == 6
        and all(isinstance(v, str) and len(v.split()) >= 15 for v in q["criteria"].values())
    ]
    assert squeeze
    counts = {len(row["questions"]) for row in golden_rows()}
    assert 1 in counts and max(counts) >= 3


def test_fixture_exercises_the_option_cap_and_the_error_row() -> None:
    gaps = [
        b - a
        for row in expected_rows()
        for e in row["expected"].values()
        for a, b in zip(e["markers"], e["markers"][1:], strict=False)
    ]
    assert 1 + OPTION_TOKEN_CAP in gaps
    errors = [row for row in golden_rows() if "error" in row]
    assert [row["error"] for row in errors] == ["options exceed head_max_len"]
    (question,) = errors[0]["questions"].values()
    assert len(question["criteria"]) == 40
