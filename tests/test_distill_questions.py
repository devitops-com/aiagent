"""Questions derived from a predictor's dspy signature: qualification, hashes, skill binds."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal

import dspy
import pytest
from pydantic import BaseModel

from aiagent.core.extract import ExtractExpense
from aiagent.core.sentiment import ScoreSegment
from aiagent.distill.questions import (
    BOOL_KEYS,
    DERIVE_VERSION,
    MAX_LEVELS,
    REASONING_FIELD,
    OutputMapping,
    derive,
    ordered_question,
    question_set_sha256,
    signature_sha256,
    skill_source_sha256,
)
from aiagent.skills.base import Skill, SkillManifest, SkillSource
from aiagent.system1.contract import Binds, content_sha256, file_sha256
from aiagent.system1.sequence import prepare

INSTRUCTIONS = "Answer the question about the passage."


def sig(*outputs: tuple[str, Any, dict[str, Any]], inputs: dict[str, Any] | None = None) -> Any:
    """A signature with `inputs` (default: one `text: str`) and the given outputs."""
    fields: dict[str, Any] = {
        name: (annotation, dspy.InputField(desc=f"The {name}."))
        for name, annotation in (inputs or {"text": str}).items()
    }
    for name, annotation, kwargs in outputs:
        fields[name] = (annotation, dspy.OutputField(**kwargs))
    return dspy.Signature(fields, INSTRUCTIONS)


def one(annotation: Any, **kwargs: Any) -> Any:
    return sig(("answer", annotation, {"desc": "The answer.", **kwargs}))


class TicketPriority(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Triage a support ticket."""

    text: str = dspy.InputField(desc="The ticket body.")
    priority: Literal["low", "normal", "high", "critical"] = dspy.OutputField(
        desc="How soon the ticket needs an answer."
    )


class TicketUrgent(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Triage a support ticket."""

    text: str = dspy.InputField(desc="The ticket body.")
    urgent: bool = dspy.OutputField(desc="Whether the ticket needs an answer today.")


class TicketSeverity(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Triage a support ticket."""

    text: str = dspy.InputField(desc="The ticket body.")
    severity: int = dspy.OutputField(desc="Severity from 1 (cosmetic) to 5 (outage).", ge=1, le=5)


def test_constants() -> None:
    assert DERIVE_VERSION == 1
    assert REASONING_FIELD == "reasoning"
    assert MAX_LEVELS == 10
    assert BOOL_KEYS == ("no", "yes")


# --------------------------------------------------------------------------- qualification

TEN = tuple(f"v{i}" for i in range(10))


@pytest.mark.parametrize(
    ("annotation", "kwargs", "kind", "keys", "values", "question"),
    [
        (Literal["a", "b"], {}, "literal", ("a", "b"), ("a", "b"),
         {"type": "choice", "criteria": ["a", "b"]}),
        (Literal[TEN], {}, "literal", TEN, TEN, {"type": "choice", "criteria": list(TEN)}),
        (bool, {}, "bool", ("no", "yes"), (False, True),
         {"type": "choice", "criteria": ["no", "yes"]}),
        (int, {"ge": 1, "le": 5}, "int", ("0", "1", "2", "3", "4"), (1, 2, 3, 4, 5),
         {"type": "score", "criteria": ["1", "2", "3", "4", "5"]}),
        (int, {"gt": 0, "lt": 6}, "int", ("0", "1", "2", "3", "4"), (1, 2, 3, 4, 5),
         {"type": "score", "criteria": ["1", "2", "3", "4", "5"]}),
        (int, {"ge": -2, "le": 7}, "int", tuple(str(i) for i in range(10)), tuple(range(-2, 8)),
         {"type": "score", "criteria": [str(v) for v in range(-2, 8)]}),
        (int, {"ge": 0, "le": 1}, "int", ("0", "1"), (0, 1),
         {"type": "score", "criteria": ["0", "1"]}),
    ],
)
def test_qualifying_outputs(
    annotation: Any,
    kwargs: dict[str, Any],
    kind: str,
    keys: tuple[str, ...],
    values: tuple[object, ...],
    question: dict[str, Any],
) -> None:
    d = derive(one(annotation, **kwargs))
    assert d.qualifies and d.reasons == ()
    assert d.input_field == "text"
    assert not d.has_reasoning
    (mapping,) = d.mappings
    assert (mapping.field, mapping.kind, mapping.keys, mapping.values) == ("answer", kind, keys, values)
    assert d.questions() == {
        "answer": {"type": question["type"], "instructions": "The answer.", "criteria": question["criteria"]}
    }
    assert list(d.questions()["answer"]) == ["type", "instructions", "criteria"]
    prepare(d.questions())  # a valid laya question
    assert d.question_set_sha256 == question_set_sha256(d.questions())
    assert d.signature_sha256 == signature_sha256(one(annotation, **kwargs))


@pytest.mark.parametrize(
    ("annotation", "kwargs", "reason"),
    [
        (Literal["a"], {}, "output 'answer': Literal needs 2-10 values (has 1)"),
        (Literal[tuple(f"v{i}" for i in range(11))], {},
         "output 'answer': Literal needs 2-10 values (has 11)"),
        (Literal[1, 2], {}, "output 'answer': Literal values must all be str"),
        (Literal["a", 2], {}, "output 'answer': Literal values must all be str"),
        (int, {"ge": 1}, "output 'answer': int needs both bounds (ge=/le=)"),
        (int, {"le": 5}, "output 'answer': int needs both bounds (ge=/le=)"),
        (int, {}, "output 'answer': int needs both bounds (ge=/le=)"),
        (int, {"ge": -10, "le": 10}, "output 'answer': 21 levels > 10"),
        (int, {"ge": 3, "le": 3}, "output 'answer': 1 levels < 2"),
        (str, {}, "output 'answer': str is not a closed answer set (use Literal[...] with 2-10 "
                  "values, bool, or int with ge/le spanning 2-10 levels)"),
        (float, {}, "output 'answer': float is not a closed answer set"),
        (list[str], {}, "output 'answer': list[str] is not a closed answer set"),
    ],
)
def test_unqualifying_outputs(annotation: Any, kwargs: dict[str, Any], reason: str) -> None:
    d = derive(one(annotation, **kwargs))
    assert not d.qualifies
    assert len(d.reasons) == 1 and d.reasons[0].startswith(reason)
    assert d.mappings == ()
    assert d.question_set_sha256 is None
    assert d.questions() == {}
    assert d.signature_sha256 == signature_sha256(one(annotation, **kwargs))


def test_reasoning_is_skipped_on_a_chain_of_thought_signature() -> None:
    (_, predictor), = dspy.ChainOfThought(TicketPriority).named_predictors()
    assert REASONING_FIELD in predictor.signature.output_fields
    d = derive(predictor.signature)
    assert d.qualifies and d.has_reasoning
    assert [m.field for m in d.mappings] == ["priority"]
    assert list(d.questions()) == ["priority"]
    # reasoning is part of the signature bind, not of the question set
    assert d.signature_sha256 != derive(TicketPriority).signature_sha256
    assert d.question_set_sha256 == derive(TicketPriority).question_set_sha256


def test_no_output_besides_reasoning() -> None:
    d = derive(sig(("reasoning", str, {})))
    assert not d.qualifies and d.has_reasoning
    assert d.reasons == ("needs one output besides reasoning (has none)",)


def test_two_outputs_are_not_supported_yet() -> None:
    d = derive(sig(("a", bool, {}), ("b", Literal["x", "y"], {})))
    assert not d.qualifies
    assert d.reasons == ("multi-output predictors not supported yet (has 2 outputs: a, b)",)
    assert d.mappings == () and d.questions() == {}


@pytest.mark.parametrize(
    ("inputs", "has"),
    [
        ({"text": str, "context": str}, "text: str, context: str"),
        ({"text": int}, "text: int"),
        ({}, "none"),
    ],
)
def test_inputs_must_be_exactly_one_str(inputs: dict[str, Any], has: str) -> None:
    d = derive(sig(("answer", bool, {}), inputs=inputs) if inputs else _no_input())
    assert not d.qualifies
    assert d.input_field is None
    assert f"needs exactly one str input field (has {has})" in d.reasons


def _no_input() -> Any:
    return dspy.Signature({"answer": (bool, dspy.OutputField())}, INSTRUCTIONS)


def test_every_reason_is_reported() -> None:
    d = derive(sig(("a", str, {}), ("b", float, {}), inputs={"x": int, "y": str}))
    assert not d.qualifies
    assert d.reasons[0].startswith("needs exactly one str input field")
    assert any(r.startswith("output 'a': str") for r in d.reasons)
    assert any(r.startswith("output 'b': float") for r in d.reasons)
    assert d.reasons[-1] == "multi-output predictors not supported yet (has 2 outputs: a, b)"


@pytest.mark.parametrize(
    ("desc", "instructions"),
    [
        ("Is it late?", "Is it late?"),
        ("${answer}", INSTRUCTIONS),
        ("${something_else}", INSTRUCTIONS),
        ("", INSTRUCTIONS),
        (None, INSTRUCTIONS),  # dspy fills in the ${answer} placeholder
    ],
)
def test_question_instructions(desc: str | None, instructions: str) -> None:
    kwargs = {} if desc is None else {"desc": desc}
    d = derive(sig(("answer", bool, kwargs)))
    assert d.questions()["answer"]["instructions"] == instructions


def test_questions_are_fresh_every_call() -> None:
    d = derive(TicketPriority)
    d.questions()["priority"]["criteria"].append("bogus")
    d.questions()["priority"]["instructions"] = "changed"
    assert d.questions()["priority"]["criteria"] == ["low", "normal", "high", "critical"]
    assert d.questions()["priority"]["instructions"] == "How soon the ticket needs an answer."


# --------------------------------------------------------------------------- mappings


def test_literal_mapping_encode_decode() -> None:
    (m,) = derive(TicketPriority).mappings
    assert m.encode("high") == "high"
    assert m.encode("urgent") is None
    assert m.encode(None) is None
    assert m.encode(1) is None
    assert m.decode("critical") == "critical"
    with pytest.raises(ValueError, match="bogus"):
        m.decode("bogus")


def test_bool_mapping_encode_decode() -> None:
    (m,) = derive(TicketUrgent).mappings
    assert m.encode(True) == "yes"
    assert m.encode(False) == "no"
    assert m.encode(1) is None  # 1 is not True
    assert m.encode(0) is None
    assert m.encode("yes") is None
    assert m.decode("yes") is True
    assert m.decode("no") is False
    with pytest.raises(ValueError):
        m.decode("true")


def test_int_mapping_encode_decode() -> None:
    (m,) = derive(TicketSeverity).mappings
    assert m.encode(1) == "0"
    assert m.encode(5) == "4"
    assert m.encode(6) is None
    assert m.encode(0) is None
    assert m.encode(True) is None  # a bool is not an int level
    assert m.encode(3.0) is None
    assert m.encode("3") is None
    assert m.decode("2") == 3
    with pytest.raises(ValueError):
        m.decode("5")


def test_output_mapping_is_a_plain_value() -> None:
    m = OutputMapping(field="q", kind="bool", keys=BOOL_KEYS, values=(False, True))
    assert m.encode(True) == "yes"
    assert m.decode("no") is False


# --------------------------------------------------------------------------- golden pins

# Computed once and pasted: a change here changes every bind, so it needs a deliberate
# DERIVE_VERSION review (and invalidates every trained student).
PINS = {
    "TicketPriority": (
        TicketPriority,
        {"priority": {"type": "choice", "instructions": "How soon the ticket needs an answer.",
                      "criteria": ["low", "normal", "high", "critical"]}},
        "49402da8b9cb63ae8c66fadce21962a32d1b3ae736ca0c40a46d23bbe6724a7c",
        "2ec7380fa810b5583dcc63a6224c54172f8dfb0eba834f55e87e7ca203bc5c05",
    ),
    "TicketUrgent": (
        TicketUrgent,
        {"urgent": {"type": "choice", "instructions": "Whether the ticket needs an answer today.",
                    "criteria": ["no", "yes"]}},
        "3adaec5003d7ac9947cce753a32c7a0b4e5dbceda482446d02b57a1d8340672d",
        "2d1a98e01a2b98489c3d730046ab0d2214af07c1342311fa0747895b53452258",
    ),
    "TicketSeverity": (
        TicketSeverity,
        {"severity": {"type": "score", "instructions": "Severity from 1 (cosmetic) to 5 (outage).",
                      "criteria": ["1", "2", "3", "4", "5"]}},
        "b6b3dc1fe9ee2e3e096079d2076080d28009cccdcc273fc01784984bd95da6b3",
        "7a85a6ec8e8dfffd175b145bbe2450baa583efe957acb5fcce73ef4165b66331",
    ),
}


@pytest.mark.parametrize("name", list(PINS))
def test_golden_pins(name: str) -> None:
    signature, questions, sig_sha, qs_sha = PINS[name]
    d = derive(signature)
    assert d.qualifies
    assert d.questions() == questions
    assert d.signature_sha256 == sig_sha
    assert d.question_set_sha256 == qs_sha


def test_signature_sha256_follows_the_contract() -> None:
    assert signature_sha256(TicketUrgent) == content_sha256([
        "aiagent.signature/1",
        "Triage a support ticket.",
        [["text", "str", "The ticket body.", None]],
        [["urgent", "bool", "Whether the ticket needs an answer today.", None]],
    ])
    assert signature_sha256(TicketSeverity) == content_sha256([
        "aiagent.signature/1",
        "Triage a support ticket.",
        [["text", "str", "The ticket body.", None]],
        [["severity", "int", "Severity from 1 (cosmetic) to 5 (outage).",
          "greater than or equal to: 1, less than or equal to: 5"]],
    ])


class Receipt(BaseModel):
    total: float


def test_signature_type_reprs() -> None:
    signature = sig(
        ("a", Literal["x", "y"], {"desc": "A."}),
        ("b", float, {"desc": "${b}"}),
        ("c", Receipt, {"desc": "C."}),
        ("d", list[str], {"desc": "D."}),
    )
    assert signature_sha256(signature) == content_sha256([
        "aiagent.signature/1",
        INSTRUCTIONS,
        [["text", "str", "The text.", None]],
        [
            ["a", ["Literal", ["x", "y"]], "A.", None],
            ["b", "float", "${b}", None],
            ["c", "Receipt", "C.", None],
            ["d", "list[str]", "D.", None],
        ],
    ])


class Color(enum.Enum):
    RED = 1


def test_a_literal_of_non_json_values_hashes_by_repr() -> None:
    signature = sig(("a", Literal[Color.RED, b"x"], {"desc": "A."}))
    assert signature_sha256(signature) == content_sha256([
        "aiagent.signature/1",
        INSTRUCTIONS,
        [["text", "str", "The text.", None]],
        [["a", ["Literal", ["<Color.RED: 1>", "b'x'"]], "A.", None]],
    ])
    assert derive(signature).reasons == ("output 'a': Literal values must all be str",)


def test_reordering_options_changes_only_the_question_hash() -> None:
    a = derive(one(Literal["x", "y", "z"]))
    b = derive(one(Literal["z", "y", "x"]))
    assert a.question_set_sha256 != b.question_set_sha256
    assert a.signature_sha256 != b.signature_sha256  # the Literal is part of the signature too


def test_an_input_desc_changes_only_the_signature_hash() -> None:
    a = derive(TicketUrgent)

    class Other(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
        """Triage a support ticket."""

        text: str = dspy.InputField(desc="The ticket text, verbatim.")
        urgent: bool = dspy.OutputField(desc="Whether the ticket needs an answer today.")

    b = derive(Other)
    assert a.signature_sha256 != b.signature_sha256
    assert a.question_set_sha256 == b.question_set_sha256


def test_ordered_question_forms() -> None:
    assert ordered_question({"type": "choice", "instructions": "I", "criteria": ["a", "b"]}) == [
        "choice", "I", [["a", None], ["b", None]]
    ]
    assert ordered_question(
        {"type": "choice", "instructions": "I", "criteria": {"a": "first", "b": None}}
    ) == ["choice", "I", [["a", "first"], ["b", None]]]
    assert ordered_question({"type": "score", "instructions": "I", "criteria": ["lo", "hi"]}) == [
        "score", "I", ["lo", "hi"]
    ]
    assert ordered_question({"type": "noul", "instructions": "I"}) == ["noul", "I", None, None]
    assert ordered_question(
        {"type": "noul", "instructions": "I", "criteria": {"True": "yes", "false": ""},
         "labels": {"false": "B", "true": "A"}}
    ) == ["noul", "I", [["false", ""], ["true", "yes"]], {"false": "B", "true": "A"}]


def test_question_set_sha256_follows_the_contract() -> None:
    questions = {
        "q1": {"type": "choice", "instructions": "I", "criteria": ["a", "b"]},
        "q2": {"type": "score", "instructions": "J", "criteria": ["lo", "hi"]},
    }
    assert question_set_sha256(questions) == content_sha256([
        ["q1", ["choice", "I", [["a", None], ["b", None]]]],
        ["q2", ["score", "J", ["lo", "hi"]]],
    ])
    swapped = {"q2": questions["q2"], "q1": questions["q1"]}
    assert question_set_sha256(swapped) != question_set_sha256(questions)


# --------------------------------------------------------------------------- skill binds


def make_skill(tmp_path: Path, source: SkillSource) -> Skill:
    directory = tmp_path / source.value / "demo"
    (directory / "__pycache__").mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\n", encoding="utf-8")
    (directory / "skill.py").write_text("def build(): ...\n", encoding="utf-8")
    (directory / "data").mkdir()
    (directory / "data" / "trainset.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    (directory / "__pycache__" / "skill.cpython-314.pyc").write_bytes(b"\x00compiled")
    manifest = SkillManifest(name="demo", description="d")
    return Skill(manifest=manifest, source=source, directory=directory)


def test_skill_source_sha256_of_a_user_skill(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, SkillSource.USER)
    d = skill.directory
    before = skill_source_sha256(skill)
    assert before == content_sha256([
        [rel, file_sha256(d / rel)] for rel in ("SKILL.md", "data/trainset.jsonl", "skill.py")
    ])
    (d / "stray.pyc").write_bytes(b"x")
    (d / "__pycache__" / "other.txt").write_text("cache", encoding="utf-8")
    assert skill_source_sha256(skill) == before
    (d / "skill.py").write_text("def build(): return 1\n", encoding="utf-8")
    assert skill_source_sha256(skill) != before


def test_skill_source_sha256_of_a_builtin_skill_ignores_python(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, SkillSource.BUILTIN)
    d = skill.directory
    before = skill_source_sha256(skill)
    assert before == content_sha256([
        [rel, file_sha256(d / rel)] for rel in ("SKILL.md", "data/trainset.jsonl")
    ])
    (d / "skill.py").write_text("def build(): return 1\n", encoding="utf-8")
    (d / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (d / "skill.pyc").write_bytes(b"x")
    assert skill_source_sha256(skill) == before  # the sourceless bundle has no .py files
    (d / "SKILL.md").write_text("---\nname: demo\ndescription: e\n---\n", encoding="utf-8")
    assert skill_source_sha256(skill) != before


def test_binds_of_a_qualifying_derivation(tmp_path: Path) -> None:
    skill = make_skill(tmp_path, SkillSource.USER)
    d = derive(TicketPriority)
    assert d.question_set_sha256 is not None
    assert d.binds(skill) == Binds(d.signature_sha256, d.question_set_sha256, skill_source_sha256(skill))
    with pytest.raises(ValueError, match="does not qualify"):
        derive(one(str)).binds(skill)


# --------------------------------------------------------------------------- shipped skills


def test_extract_expense_does_not_qualify() -> None:
    (_, predictor), = dspy.ChainOfThought(ExtractExpense).named_predictors()
    d = derive(predictor.signature)
    assert not d.qualifies and d.has_reasoning
    assert d.input_field == "text"
    assert [r.split(":")[0] for r in d.reasons[:3]] == [
        "output 'merchant'", "output 'date'", "output 'amount'"
    ]
    assert d.reasons[0].startswith("output 'merchant': str is not a closed answer set")
    assert d.reasons[1].startswith("output 'date': str is not a closed answer set")
    assert d.reasons[2].startswith("output 'amount': float is not a closed answer set")
    assert d.reasons[3] == (
        "multi-output predictors not supported yet (has 3 outputs: merchant, date, amount)"
    )


def test_score_segment_does_not_qualify() -> None:
    (_, predictor), = dspy.ChainOfThought(ScoreSegment).named_predictors()
    d = derive(predictor.signature)
    assert not d.qualifies
    assert d.reasons == (
        "output 'score': int needs both bounds (ge=/le=)",
        "output 'rationale': str is not a closed answer set (use Literal[...] with 2-10 values, "
        "bool, or int with ge/le spanning 2-10 levels)",
        "multi-output predictors not supported yet (has 2 outputs: score, rationale)",
    )
