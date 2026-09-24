"""laya questions derived from a predictor's dspy signature, and their bind hashes.

A predictor qualifies when it has exactly one ``str`` input and exactly one output
besides ``reasoning`` with a closed answer set: ``Literal[...]`` (2-10 ``str``
values) becomes a ``choice``, ``bool`` a two-option ``choice`` with neutral keys
(``no``/``yes``; not ``noul``, which has a label-word bias), and an ``int`` bounded
on both sides to 2-10 levels a ``score``. Pure: signatures are duck-typed
(``instructions``, ``input_fields``, ``output_fields`` of pydantic ``FieldInfo``),
so this module never imports dspy.

``signature_sha256`` binds a student to the whole signature, ``question_set_sha256`` to
the derived questions and ``skill_source_sha256`` (soft) to the skill's files.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args, get_origin

from aiagent.skills.base import Skill, SkillSource
from aiagent.system1.contract import Binds, content_sha256, file_sha256

DERIVE_VERSION: Final = 1
REASONING_FIELD: Final = "reasoning"
MAX_LEVELS: Final = 10
BOOL_KEYS: Final = ("no", "yes")  # decoded to (False, True)
Kind = Literal["literal", "bool", "int"]

_SIGNATURE_TAG: Final = "aiagent.signature/1"
_MIN_LEVELS: Final = 2
_PLACEHOLDER: Final = re.compile(r"\$\{[^}]*\}")
_CLOSED_SET_HINT: Final = (
    "is not a closed answer set (use Literal[...] with 2-10 values, bool, "
    "or int with ge/le spanning 2-10 levels)"
)
_PLAIN_TYPES: Final = (str, bool, int, float)


@dataclass(frozen=True)
class OutputMapping:
    """How one signature output maps to one laya question."""

    field: str  # == question id
    kind: Kind
    keys: tuple[str, ...]  # option keys, render order
    values: tuple[object, ...]  # typed values, same order
    instructions: str = ""  # the question's instructions

    def encode(self, value: object) -> str | None:
        """Teacher value -> key, or None (a bool must be bool; an int must not be)."""
        if self.kind == "bool":
            allowed = isinstance(value, bool)
        elif self.kind == "int":
            allowed = isinstance(value, int) and not isinstance(value, bool)
        else:
            allowed = isinstance(value, str)
        if not allowed or value not in self.values:
            return None
        return self.keys[self.values.index(value)]

    def decode(self, key: str) -> object:
        """Key -> typed value; ValueError for an unknown key."""
        if key not in self.keys:
            raise ValueError(f"output {self.field!r}: unknown answer key {key!r}")
        return self.values[self.keys.index(key)]

    def question(self) -> dict[str, Any]:
        """A fresh laya question: {"type", "instructions", "criteria"}."""
        if self.kind == "int":
            return {
                "type": "score",
                "instructions": self.instructions,
                "criteria": [str(v) for v in self.values],
            }
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": list(self.keys),
        }


@dataclass(frozen=True)
class Derivation:
    """Questions derived from one predictor signature."""

    qualifies: bool
    reasons: tuple[str, ...]  # empty iff qualifies
    input_field: str | None
    has_reasoning: bool
    mappings: tuple[OutputMapping, ...]  # one when it qualifies; never reasoning
    signature_sha256: str
    question_set_sha256: str | None  # None iff not qualifies

    def questions(self) -> dict[str, dict[str, Any]]:
        """Fresh laya question JSON, qid -> {"type","instructions","criteria"}."""
        if not self.qualifies:
            return {}
        return {m.field: m.question() for m in self.mappings}

    def binds(self, skill: Skill) -> Binds:
        """Binds for a qualifying derivation (ValueError otherwise)."""
        if self.question_set_sha256 is None:
            raise ValueError(
                "the predictor does not qualify: " + "; ".join(self.reasons)
            )
        return Binds(
            self.signature_sha256, self.question_set_sha256, skill_source_sha256(skill)
        )


def _extra(field: Any) -> Mapping[str, Any]:
    extra = getattr(field, "json_schema_extra", None)
    return extra if isinstance(extra, dict) else {}


def _type_repr(annotation: Any) -> Any:
    if annotation in _PLAIN_TYPES:
        return annotation.__name__
    if get_origin(annotation) is Literal:
        values = get_args(annotation)  # an enum member or a bytes value: its repr
        return ["Literal", [v if _is_json_scalar(v) else repr(v) for v in values]]
    if isinstance(annotation, type):
        return annotation.__qualname__
    return repr(annotation)


def _is_json_scalar(value: object) -> bool:
    return value is None or type(value) in _PLAIN_TYPES


def _type_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__qualname__
    return repr(annotation)


def _field_entries(fields: Mapping[str, Any]) -> list[list[Any]]:
    return [
        [
            name,
            _type_repr(field.annotation),
            _extra(field).get("desc"),
            _extra(field).get("constraints"),
        ]
        for name, field in fields.items()
    ]


def signature_sha256(signature: Any) -> str:
    """H(["aiagent.signature/1", instructions, inputs, outputs]), declaration order."""
    return content_sha256(
        [
            _SIGNATURE_TAG,
            signature.instructions,
            _field_entries(signature.input_fields),
            _field_entries(signature.output_fields),
        ]
    )


def ordered_question(qdef: Mapping[str, Any]) -> list[Any]:
    """The ordered form of one question, as question_set_sha256 hashes it."""
    qtype, ins = qdef["type"], qdef["instructions"]
    crit: Any = qdef.get("criteria")
    if qtype == "choice":
        pairs = crit.items() if isinstance(crit, dict) else ((c, None) for c in crit)
        return ["choice", ins, [[key, desc] for key, desc in pairs]]
    if qtype == "score":
        return ["score", ins, list(crit)]
    described = None
    if crit is not None:
        lowered = {str(k).lower(): v for k, v in crit.items()}
        described = [[key, lowered.get(key)] for key in ("false", "true")]
    return ["noul", ins, described, qdef.get("labels")]


def question_set_sha256(questions: Mapping[str, Mapping[str, Any]]) -> str:
    """H([[qid, ordered(qdef)], ...]) in question order."""
    return content_sha256(
        [[qid, ordered_question(qdef)] for qid, qdef in questions.items()]
    )


def skill_source_sha256(skill: Skill) -> str:
    """H([[relpath, file_sha256], ...]) over the skill's files (a soft bind).

    ``__pycache__/`` and ``*.pyc`` are skipped; so is ``*.py`` for a builtin skill,
    since the sourceless bundle has none (the value then matches dev tree and bundle).
    """
    skip = {".pyc", ".py"} if skill.source is SkillSource.BUILTIN else {".pyc"}
    entries = []
    for path in skill.directory.rglob("*"):
        rel = path.relative_to(skill.directory)
        if "__pycache__" in rel.parts or path.suffix in skip or not path.is_file():
            continue
        entries.append([rel.as_posix(), file_sha256(path)])
    return content_sha256(sorted(entries))


def _instructions(signature: Any, field: Any) -> str:
    desc = _extra(field).get("desc")
    if not isinstance(desc, str) or not desc or _PLACEHOLDER.fullmatch(desc):
        return str(signature.instructions)
    return desc


def _int_bounds(field: Any) -> tuple[int | None, int | None]:
    lows: list[int] = []
    highs: list[int] = []
    for item in getattr(field, "metadata", ()):
        for attr, offset, into in (
            ("ge", 0, lows),
            ("gt", 1, lows),
            ("le", 0, highs),
            ("lt", -1, highs),
        ):
            bound = getattr(item, attr, None)
            if isinstance(bound, int) and not isinstance(bound, bool):
                into.append(bound + offset)
    return (max(lows) if lows else None), (min(highs) if highs else None)


def _map_output(name: str, field: Any, instructions: str) -> OutputMapping | str:
    """The mapping for one output, or the reason it has none."""
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not all(isinstance(v, str) for v in values):
            return f"output {name!r}: Literal values must all be str"
        if not _MIN_LEVELS <= len(values) <= MAX_LEVELS:
            return f"output {name!r}: Literal needs 2-10 values (has {len(values)})"
        return OutputMapping(name, "literal", values, values, instructions)
    if annotation is bool:
        return OutputMapping(name, "bool", BOOL_KEYS, (False, True), instructions)
    if annotation is int:
        lo, hi = _int_bounds(field)
        if lo is None or hi is None:
            return f"output {name!r}: int needs both bounds (ge=/le=)"
        levels = hi - lo + 1
        if levels > MAX_LEVELS:
            return f"output {name!r}: {levels} levels > {MAX_LEVELS}"
        if levels < _MIN_LEVELS:
            return f"output {name!r}: {levels} levels < {_MIN_LEVELS}"
        keys = tuple(str(i) for i in range(levels))
        return OutputMapping(name, "int", keys, tuple(range(lo, hi + 1)), instructions)
    return f"output {name!r}: {_type_name(annotation)} {_CLOSED_SET_HINT}"


def derive(signature: Any) -> Derivation:
    """Apply the qualification rules to a dspy.Signature class."""
    reasons: list[str] = []
    inputs = dict(signature.input_fields)
    input_field = None
    if len(inputs) == 1 and next(iter(inputs.values())).annotation is str:
        input_field = next(iter(inputs))
    else:
        has = ", ".join(
            f"{name}: {_type_name(f.annotation)}" for name, f in inputs.items()
        )
        reasons.append(f"needs exactly one str input field (has {has or 'none'})")

    outputs = dict(signature.output_fields)
    has_reasoning = outputs.pop(REASONING_FIELD, None) is not None
    mappings = []
    for name, field in outputs.items():
        mapped = _map_output(name, field, _instructions(signature, field))
        if isinstance(mapped, str):
            reasons.append(mapped)
        else:
            mappings.append(mapped)
    if not outputs:
        reasons.append("needs one output besides reasoning (has none)")
    elif len(outputs) > 1:
        reasons.append(
            f"multi-output predictors not supported yet (has {len(outputs)} "
            f"outputs: {', '.join(outputs)})"
        )

    qualifies = not reasons
    kept = tuple(mappings) if qualifies else ()
    questions = {m.field: m.question() for m in kept}
    return Derivation(
        qualifies=qualifies,
        reasons=tuple(reasons),
        input_field=input_field,
        has_reasoning=has_reasoning,
        mappings=kept,
        signature_sha256=signature_sha256(signature),
        question_set_sha256=question_set_sha256(questions) if qualifies else None,
    )
