"""A laya student on onnxruntime (CPU): encode, one session run per state, decode.

Port of laya v0.3.20 (Convai Innovations, Apache-2.0; see LICENSES/laya-Apache-2.0.txt):
``laya/agent.py`` (_encode_state, _decode_answers) and ``laya/common.py``
(clamp_temperature, temp_bucket, answer_confidence, confidence_from_probs). Changes:
onnxruntime over the exported graph instead of torch (no AMP, act head, hooks or
language temperatures); one state per ``session.run``, never several (measured 5x
slower with the traced export); answers are frozen :class:`Answer` values that keep
the unrounded probabilities and the encoded row, and :meth:`Answer.to_laya` gives
laya's answer dict minus ``action`` (and score's ``legend``); a bad artifact raises
:class:`~aiagent.exceptions.ArtifactError`.

numpy is imported here; ``onnxruntime`` only in :class:`System1Runtime`, so importing
this module does not load it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from aiagent.exceptions import ArtifactError, System1Error
from aiagent.system1.sequence import (
    QTYPES,
    Encoded,
    InternalQuestion,
    QType,
    SequenceTokenizer,
    State,
    prepare,
)

TEMP_MIN: Final = 0.5
TEMP_MAX: Final = 5.0
INPUT_NAMES: Final = (
    "input_ids",
    "attention_mask",
    "marker_pos",
    "marker_mask",
    "qtype",
)
OUTPUT_NAME: Final = "logits"

_DEFAULT_MAX_LEN: Final = 512
_DEFAULT_HEAD_MAX_LEN: Final = 192
_ENTROPY_FLOOR: Final = 1e-12
_DECIMALS: Final = 4


def clamp_temperature(value: object) -> float:
    """float(value) clamped to [0.5, 5.0]; 1.0 if not a finite number."""
    try:
        t = float(value)  # type: ignore[arg-type]  # anything float() accepts
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temp_bucket(qtype: QType, k: int) -> str:
    """'<type>:2' | ':3-5' | ':6-10' | ':11+'."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


@dataclass(frozen=True)
class AgentConfig:
    """The inference part of rl_agent_config.json (temperatures already clamped)."""

    max_len: int
    head_max_len: int
    temperature: tuple[float, float, float]
    temperature_by_options: Mapping[str, float]

    def temperature_for(self, qtype: QType, k: int) -> float:
        """temperature_by_options[bucket], else temperature[qtype_index]."""
        fallback = self.temperature[QTYPES[qtype]]
        return self.temperature_by_options.get(temp_bucket(qtype, k), fallback)


def _length(raw: Mapping[str, Any], key: str, default: int, path: Path) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArtifactError(f"{path}: {key!r} must be a positive int, got {value!r}")
    return value


def load_agent_config(path: Path) -> AgentConfig:
    """Defaults 512/192, [1,1,1], {}; ArtifactError on bad JSON or temperatures."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArtifactError(f"{path}: not a JSON object")
    temps = raw.get("temperature", [1.0, 1.0, 1.0])
    if not isinstance(temps, list) or len(temps) != len(QTYPES):
        raise ArtifactError(f"{path}: 'temperature' must be a list of 3 numbers")
    by_options = raw.get("temperature_by_options", {})
    if not isinstance(by_options, dict):
        raise ArtifactError(f"{path}: 'temperature_by_options' must be an object")
    choice, score, noul = (clamp_temperature(t) for t in temps)
    return AgentConfig(
        max_len=_length(raw, "max_len", _DEFAULT_MAX_LEN, path),
        head_max_len=_length(raw, "head_max_len", _DEFAULT_HEAD_MAX_LEN, path),
        temperature=(choice, score, noul),
        temperature_by_options={
            str(bucket): clamp_temperature(t) for bucket, t in by_options.items()
        },
    )


def _round(value: float) -> float:
    return round(value, _DECIMALS)


@dataclass(frozen=True)
class Answer:
    """One decoded answer (unrounded)."""

    type: QType
    keys: tuple[str, ...]
    probabilities: tuple[float, ...]
    answer_confidence: float  # max(p), the calibrated quantity; gate on this
    confidence: float  # laya's entropy confidence (noul: max(p1, 1-p1))
    encoded: Encoded

    @property
    def key(self) -> str:
        """choice/score: keys[first argmax]; noul: 'true' iff p[1] >= 0.5."""
        if self.type == "noul":
            return self.keys[1] if self.probabilities[1] >= 0.5 else self.keys[0]
        return self.keys[self.probabilities.index(max(self.probabilities))]

    @property
    def expected_level(self) -> float:
        """score only: sum(i * p_i); ValueError otherwise."""
        if self.type != "score":
            raise ValueError(f"expected_level needs a score answer, not {self.type}")
        return sum(i * p for i, p in enumerate(self.probabilities))

    def to_laya(self) -> dict[str, Any]:
        """laya's answer dict minus 'action', values rounded to 4 decimals."""
        answer: dict[str, Any] = {"type": self.type}
        if self.type == "noul":
            answer["noul"] = _round(self.probabilities[1])
        else:
            if self.type == "choice":
                answer["choice"] = self.key
            else:
                answer["score"] = _round(self.expected_level)
            pairs = zip(self.keys, self.probabilities, strict=True)
            answer["probabilities"] = {key: _round(p) for key, p in pairs}
        answer["confidence"] = _round(self.confidence)
        answer["answer_confidence"] = _round(self.answer_confidence)
        return answer


def _entropy_confidence(p: npt.NDArray[np.float32], k: int) -> float:
    if k < 2:
        return 1.0
    entropy = -(p * np.log(np.clip(p, _ENTROPY_FLOOR, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))


def decode(
    logits: npt.NDArray[np.float32],
    q: InternalQuestion,
    encoded: Encoded,
    config: AgentConfig,
) -> Answer:
    """laya _decode_answers for one row (logits[:k] are used)."""
    k = len(encoded.markers)  # >= 1: encode places a marker for every option
    z = np.asarray(logits[:k], dtype=np.float32) / config.temperature_for(q.type, k)
    p = np.exp(z - z.max())
    p /= p.sum()
    p1 = float(p[1]) if q.type == "noul" else 0.0
    return Answer(
        type=q.type,
        keys=q.keys,
        probabilities=tuple(float(x) for x in p),
        answer_confidence=float(np.clip(p.max(), 0.0, 1.0)),
        confidence=max(p1, 1.0 - p1) if q.type == "noul" else _entropy_confidence(p, k),
        encoded=encoded,
    )


class System1Runtime:
    """A laya student on onnxruntime (CPU), loaded from an artifact directory."""

    def __init__(self, model_dir: Path) -> None:
        self._config = load_agent_config(model_dir / "rl_agent_config.json")
        try:
            self._tokenizer = SequenceTokenizer.from_dir(model_dir / "tokenizer")
        except System1Error as exc:
            raise ArtifactError(f"{model_dir}: {exc}") from exc
        self._session = _load_session(model_dir / "model.onnx")

    @property
    def config(self) -> AgentConfig:
        """The artifact's rl_agent_config.json."""
        return self._config

    @property
    def tokenizer(self) -> SequenceTokenizer:
        """The artifact's tokenizer, with laya's sequence layout."""
        return self._tokenizer

    def encode(
        self, state: State, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Encoded]:
        """prepare + tokenizer.encode with the artifact's limits."""
        return self._encode(state, prepare(questions))

    def fits(self, state: State, questions: Mapping[str, Mapping[str, Any]]) -> bool:
        """True iff the student sees the whole state for every question."""
        return self._tokenizer.fits(
            state,
            prepare(questions),
            max_len=self._config.max_len,
            head_max_len=self._config.head_max_len,
        )

    def predict(
        self, state: State, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Answer]:
        """One session.run for all questions of this state."""
        internal = prepare(questions)
        if not internal:
            return {}
        rows = self._encode(state, internal)
        n = len(rows)
        width = max(len(row.input_ids) for row in rows.values())
        kmax = max(len(row.markers) for row in rows.values())
        input_ids = np.full((n, width), self._tokenizer.special.pad_id, dtype=np.int64)
        attention_mask = np.zeros((n, width), dtype=np.int64)
        marker_pos = np.zeros((n, kmax), dtype=np.int64)
        marker_mask = np.zeros((n, kmax), dtype=np.bool_)
        for i, row in enumerate(rows.values()):
            input_ids[i, : len(row.input_ids)] = row.input_ids
            attention_mask[i, : len(row.input_ids)] = 1
            marker_pos[i, : len(row.markers)] = row.markers
            marker_mask[i, : len(row.markers)] = True
        qtype = np.array([q.qtype_index for q in internal.values()], dtype=np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }
        logits = self._session.run([OUTPUT_NAME], feeds)[0]
        return {
            qid: decode(logits[i], internal[qid], rows[qid], self._config)
            for i, qid in enumerate(rows)
        }

    def _encode(
        self, state: State, internal: Mapping[str, InternalQuestion]
    ) -> dict[str, Encoded]:
        return self._tokenizer.encode(
            state,
            internal,
            max_len=self._config.max_len,
            head_max_len=self._config.head_max_len,
        )


def _load_session(model_file: Path) -> Any:
    """A CPU InferenceSession with laya's five inputs and a logits output."""
    import onnxruntime as ort  # lazy: a 30 MB library, loaded only when a student runs

    if not model_file.is_file():
        raise ArtifactError(f"missing {model_file}")
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.log_severity_level = 3  # errors only
    try:
        session = ort.InferenceSession(
            str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # onnxruntime raises its own (non-public) exception types
        raise ArtifactError(f"cannot load {model_file}: {exc}") from exc
    inputs = {i.name for i in session.get_inputs()}
    if inputs != set(INPUT_NAMES):  # fed by name, so the order does not matter
        want = sorted(INPUT_NAMES)
        raise ArtifactError(f"{model_file}: inputs {sorted(inputs)}, expected {want}")
    if OUTPUT_NAME not in {o.name for o in session.get_outputs()}:
        raise ArtifactError(f"{model_file}: no {OUTPUT_NAME!r} output")
    return session
