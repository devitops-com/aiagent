"""laya's input sequences: question validation, option rendering and token layout.

Port of laya v0.3.20 (Convai Innovations, Apache-2.0; see LICENSES/laya-Apache-2.0.txt):
``laya/common.py`` (serialize_state, render_criterion, render_options, build_sequence)
and ``laya/agent.py`` (_check_question, _to_internal, _encode_state). Changes: raw
``tokenizers`` instead of transformers; saved truncation/padding state is reset on
load; the 48-token option cap is a slice (what laya's ``truncation=True,
max_length=48`` keeps); questions become frozen :class:`InternalQuestion` values;
choice keys must be ``str``; errors are :class:`~aiagent.exceptions.QuestionError`.

A row is ``[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
state [SEP]``; the student scores the ``[MASK]`` markers. ``tokenizers`` is imported
only in :meth:`SequenceTokenizer.from_dir`, so importing this module stays cheap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from aiagent.exceptions import QuestionError, System1Error

if TYPE_CHECKING:
    from tokenizers import Tokenizer

QType = Literal["choice", "score", "noul"]
QTYPES: Final[Mapping[str, int]] = {"choice": 0, "score": 1, "noul": 2}
OPTION_TOKEN_CAP: Final = 48  # laya: tok(..., truncation=True, max_length=48)
HEAD_MIN_BUDGET: Final = 16
HEAD_MIN_TOKENS: Final = 8
OPTION_MIN_TOKENS: Final = 4
NOUL_DEFAULT_FALSE: Final = "no, the statement does not hold"
NOUL_DEFAULT_TRUE: Final = "yes, the statement holds"
NOUL_KEYS: Final = ("false", "true")
State = str | dict[str, Any] | list[Any]

_SPECIAL_KEYS: Final = ("mask_token", "cls_token", "sep_token", "pad_token")
_BAD_LABELS: Final = (
    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
)


@dataclass(frozen=True)
class InternalQuestion:
    """A validated, normalised laya question."""

    type: QType
    instructions: str  # non-str instructions -> json.dumps(ins, ensure_ascii=False)
    options: tuple[str, ...]  # render_options() output, render order
    keys: tuple[str, ...]  # choice keys | "0".."n-1" | ("false", "true")

    @property
    def qtype_index(self) -> int:
        """0 choice, 1 score, 2 noul."""
        return QTYPES[self.type]


@dataclass(frozen=True)
class Encoded:
    """One (state, question) row."""

    input_ids: tuple[int, ...]
    markers: tuple[int, ...]


@dataclass(frozen=True)
class SpecialTokens:
    """The four special tokens laya's layout needs."""

    mask_token: str
    mask_id: int
    cls_id: int
    sep_id: int
    pad_id: int


def serialize_state(state: State) -> str:
    """str as is; else json.dumps(state, ensure_ascii=False) (default separators)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(value: object) -> str:
    """str as is; else compact-ish JSON (", ", ": "), non-ASCII literal, default=str."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _noul_labels(labels: object) -> tuple[str, str]:
    if labels is None:
        return NOUL_KEYS
    if not isinstance(labels, dict) or set(labels) != set(NOUL_KEYS):
        raise QuestionError(_BAD_LABELS)
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise QuestionError(_BAD_LABELS)
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise QuestionError(_BAD_LABELS)
    return false_label, true_label


def _check_criteria(where: str, qtype: str, crit: object) -> None:
    if qtype == "choice":
        if not isinstance(crit, dict | list):
            raise QuestionError(
                f"{where}: a choice question takes 'criteria' as a dict of "
                "label -> description, or a list of labels"
            )
        if not crit:
            raise QuestionError(
                f"{where}: a choice question needs at least one criterion"
            )
        if not all(isinstance(key, str) for key in crit):
            raise QuestionError(f"{where}: choice keys must be strings")
    elif qtype == "score":
        if not isinstance(crit, list):
            raise QuestionError(
                f"{where}: a score question takes 'criteria' as a list of level "
                "descriptions, index 0 first"
            )
        if not crit:
            raise QuestionError(f"{where}: a score question needs at least one level")
    elif crit is not None:
        if not isinstance(crit, dict):
            raise QuestionError(
                f"{where}: a noul question takes 'criteria' as a dict with optional "
                "'true'/'false' descriptions, or omits it"
            )
        keys = {str(key).lower() for key in crit}
        if not keys <= set(NOUL_KEYS):
            raise QuestionError(
                f"{where}: a noul question takes 'criteria' keyed only 'true'/'false', "
                f"got {sorted(keys)}"
            )


def check_question(qid: str, qdef: object) -> None:
    """laya's _check_question rules, raising QuestionError."""
    where = f"question {qid!r}"
    if not isinstance(qdef, dict):
        raise QuestionError(
            f"{where}: definition must be a dict, got {type(qdef).__name__}"
        )
    qtype = qdef.get("type")
    if not isinstance(qtype, str) or qtype not in QTYPES:
        raise QuestionError(
            f"{where}: unknown type {qtype!r}; use one of {sorted(QTYPES)}"
        )
    if "instructions" not in qdef:
        raise QuestionError(
            f"{where}: no 'instructions'; add the text the model should answer"
        )
    _check_criteria(where, qtype, qdef.get("criteria"))
    if "labels" in qdef:
        if qtype != "noul":
            raise QuestionError(
                f"{where}: 'labels' is only supported for noul questions"
            )
        try:
            _noul_labels(qdef["labels"])
        except QuestionError as exc:
            raise QuestionError(f"{where}: {exc}") from exc


def _described(key: str, value: object) -> str:
    return key if value is None or value == "" else f"{key}: {render_criterion(value)}"


def _noul_option(label: str, value: object, default: str) -> str:
    return f"{label}: {render_criterion(value) if value not in (None, '') else default}"


def to_internal(qdef: Mapping[str, Any]) -> InternalQuestion:
    """laya's _to_internal + render_options."""
    qtype: QType = qdef["type"]
    crit: Any = qdef.get("criteria")
    ins = qdef["instructions"]
    instructions = ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)
    if qtype == "choice":
        choices = {c: None for c in crit} if isinstance(crit, list) else dict(crit)
        options = tuple(_described(key, value) for key, value in choices.items())
        keys = tuple(choices)
    elif qtype == "score":
        options = tuple(f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit))
        keys = tuple(str(i) for i in range(len(crit)))
    else:
        descriptions = {str(k).lower(): v for k, v in (crit or {}).items()}
        false_label, true_label = _noul_labels(qdef.get("labels"))
        options = (
            _noul_option(false_label, descriptions.get("false"), NOUL_DEFAULT_FALSE),
            _noul_option(true_label, descriptions.get("true"), NOUL_DEFAULT_TRUE),
        )
        keys = NOUL_KEYS
    return InternalQuestion(qtype, instructions, options, keys)


def prepare(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, InternalQuestion]:
    """check + to_internal, question order kept."""
    for qid, qdef in questions.items():
        check_question(qid, qdef)
    return {qid: to_internal(qdef) for qid, qdef in questions.items()}


def _special_token(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str) or not value:
        raise System1Error(f"tokenizer_config.json: no usable {key!r}")
    return value


class SequenceTokenizer:
    """laya's build_sequence over a tokenizers.Tokenizer."""

    def __init__(self, tokenizer: Tokenizer, special: SpecialTokens) -> None:
        self._tok = tokenizer
        self._special = special

    @classmethod
    def from_dir(cls, tokenizer_dir: Path) -> SequenceTokenizer:
        """Load tokenizer.json and the special tokens named in tokenizer_config.json."""
        from tokenizers import Tokenizer

        tokenizer_file = tokenizer_dir / "tokenizer.json"
        config_file = tokenizer_dir / "tokenizer_config.json"
        try:
            tok = Tokenizer.from_file(str(tokenizer_file))
        except Exception as exc:  # noqa: BLE001 - tokenizers raises a bare Exception
            raise System1Error(f"cannot load {tokenizer_file}: {exc}") from exc
        # A tokenizer.json saved after a truncating/padding call carries that state and
        # raw tokenizers applies it (transformers resets it per call): without the reset
        # every state could be cut to laya's last option length of 48 tokens.
        tok.no_truncation()
        tok.no_padding()
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise System1Error(f"cannot read {config_file}: {exc}") from exc
        if not isinstance(config, dict):
            raise System1Error(f"{config_file}: not a JSON object")
        tokens = [_special_token(config, key) for key in _SPECIAL_KEYS]
        ids = []
        for key, token in zip(_SPECIAL_KEYS, tokens, strict=True):
            token_id = tok.token_to_id(token)
            if token_id is None:
                raise System1Error(f"{key} {token!r} is not in {tokenizer_file}")
            ids.append(token_id)
        mask_id, cls_id, sep_id, pad_id = ids
        special = SpecialTokens(tokens[0], mask_id, cls_id, sep_id, pad_id)
        return cls(tok, special)

    @property
    def special(self) -> SpecialTokens:
        """The mask token and the special ids."""
        return self._special

    def ids(self, text: str) -> list[int]:
        """encode(text, add_special_tokens=False).ids"""
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def state_ids(self, state: State) -> list[int]:
        """ids(serialize_state(state).replace(mask_token, " "))"""
        return self.ids(serialize_state(state).replace(self._special.mask_token, " "))

    def prefix(self, q: InternalQuestion, *, head_max_len: int) -> Encoded:
        """[CLS] head [SEP] options [SEP], no state."""
        sp, mask = self._special, self._special.mask_token
        head = self.ids(f"{q.type} question: {q.instructions.replace(mask, ' ')}")
        opts = [
            [sp.mask_id, *self.ids(" " + o.replace(mask, " "))[:OPTION_TOKEN_CAP]]
            for o in q.options
        ]
        budget = head_max_len - sum(len(o) for o in opts)
        if budget < HEAD_MIN_BUDGET:
            per = max(
                OPTION_MIN_TOKENS, (head_max_len - HEAD_MIN_BUDGET) // max(1, len(opts))
            )
            opts = [o[:per] for o in opts]
            budget = head_max_len - sum(len(o) for o in opts)
        seq = [sp.cls_id, *head[: max(HEAD_MIN_TOKENS, budget)], sp.sep_id]
        markers = []
        for o in opts:
            markers.append(len(seq))
            seq.extend(o)
        seq.append(sp.sep_id)
        return Encoded(tuple(seq), tuple(markers))

    def room(self, q: InternalQuestion, *, max_len: int, head_max_len: int) -> int:
        """max(0, max_len - len(prefix) - 1): the state tokens a row keeps."""
        head = self.prefix(q, head_max_len=head_max_len)
        return max(0, max_len - len(head.input_ids) - 1)

    def build(
        self,
        q: InternalQuestion,
        state_ids: Sequence[int],
        *,
        max_len: int,
        head_max_len: int,
        truncate_left: bool,
    ) -> Encoded:
        """laya build_sequence."""
        head = self.prefix(q, head_max_len=head_max_len)
        room = max(0, max_len - len(head.input_ids) - 1)
        if truncate_left:
            state = state_ids[max(0, len(state_ids) - room) :]
        else:
            state = state_ids[:room]
        seq = (*head.input_ids, *state, self._special.sep_id)
        return Encoded(seq[:max_len], tuple(m for m in head.markers if m < max_len))

    def encode(
        self,
        state: State,
        questions: Mapping[str, InternalQuestion],
        *,
        max_len: int,
        head_max_len: int,
    ) -> dict[str, Encoded]:
        """Tokenize the state once, build every row, check that the markers fit."""
        state_ids = self.state_ids(state)
        truncate_left = isinstance(state, list)  # a conversation keeps its newest turns
        rows = {}
        for qid, q in questions.items():
            row = self.build(
                q,
                state_ids,
                max_len=max_len,
                head_max_len=head_max_len,
                truncate_left=truncate_left,
            )
            if len(row.markers) != len(q.options):
                raise QuestionError(
                    f"question {qid!r} options exceed head_max_len={head_max_len}"
                )
            rows[qid] = row
        return rows

    def fits(
        self,
        state: State,
        questions: Mapping[str, InternalQuestion],
        *,
        max_len: int,
        head_max_len: int,
    ) -> bool:
        """True iff no row truncates the state and every marker fits."""
        n = len(self.state_ids(state))
        for q in questions.values():
            head = self.prefix(q, head_max_len=head_max_len)
            room = max(0, max_len - len(head.input_ids) - 1)
            if n > room or any(m >= max_len for m in head.markers):
                return False
        return True
