"""Calibrate sentiment's System 1 from a sentiment shadow log (design §2.9, PR 3).

docs/design/sentiment-system1.md §2.1, §2.4 and §2.9. Pure: the standard library plus
aiagent (clopper_pearson_lower of aiagent.distill.gates; NEUTRAL_LABEL,
SCORE_SIGNATURE_SHA256 and calibration_model of aiagent.core.sentiment; SCALE_MIN and
SCALE_MAX of aiagent.core.sentiment_stats; Reasoning of aiagent.llm.registry). From the
repository root:

    .venv/bin/python tools/system1/sentiment_calibration.py LOG.jsonl \\
        [--meta SIDECAR.jsonl] [--tau T ...] [--pin-tau T] [--level L] \\
        [--model MODEL] [--measured TEXT] [--json OUT.json]

LOG is <artifacts_dir>/system1/skills/sentiment/score/shadow.jsonl of one student, from
a shadow run at resample 3. --meta is a corpus sidecar, one JSON object per line keyed
by "doc_sha256" (hex) or "doc_id" ("sha256:<hex>" of the text), with "source" and
"lang". --tau replaces the default sweep (0.90 0.92 0.94 0.96 0.98). --pin-tau T
calibrates and tests at T instead of the installed τ (D9: a calibration carries its own
τ, which the gate applies on top of the student's), and only then is a block offered.
--level evaluates a given pin: (b) is computed at L, and nothing is offered to pin (D3:
the review run at the document run's NEUTRAL.level); with --pin-tau it is the
confirmation run (D9), whose verdict ends the report. --model names the LM the run
scored with, for lines without a model field (aiagent 0.7.0). --measured names the run
in NEUTRAL (default: the log's name, time span and size). --json writes every number.
Exit 0 when both pass tests pass at the τ tested (--pin-tau's, else the installed one),
1 if not, 2 on bad input.

**As implemented:**

- Lines that are not JSON (a line torn by a killed process) are left out and named by
  line number; any other malformed line is bad input.
- Runs: lines grouped by run_id. A run is complete when its lines share doc_sha256 and
  n_segments = n, with seg_index exactly 0..n−1. An incomplete run (a student that
  broke mid-run, a truncated log) is reported and left out of everything below. Of
  several complete runs of one doc_sha256 the first in the log is kept, the others
  reported and left out (a re-run is answered from the cache: the same samples twice).
- Per line: ȳ = the mean of the non-null llm_samples (extra rollouts included), k =
  their count. A line with k = 0 has no ȳ, and is reported.
- Accepted: at the installed τ, would_accept; at a sweep τ' or --pin-tau's, fits and
  student == "neutral" and confidence ≥ τ' (the logged confidence, rounded to 4
  decimals; the gate compares the unrounded one).
- The τ tested: --pin-tau's T when given, else the installed τ. The calibration, the
  pass test, the slices, the verdict and the exit code are at that τ; the installed τ's
  numbers stay in the JSON ("installed") and in the sweep table.
- Over the accepted lines with a ȳ: LEVEL = mean ȳ_i over n lines; SE = sd(ȳ)/√n, the
  sample sd (n − 1); σ_w² = mean s_i², the sample variance of line i's samples, over
  those with k_i ≥ 2; σ_b² = max(0, var(ȳ) − σ_w²·mean(1/k_i)), var the sample
  variance. This generalizes §2.1's var(ȳ) − σ_w²/3, which assumes k = 3 everywhere:
  a dropped sample or an extra rollout changes k, and for independent segments
  E[var(ȳ)] = σ_b² + σ_w²·mean(1/k_i).
- Coverage = accepted / lines, too_long = (not fits) / lines: overall, per n_tokens bin
  and per sidecar slice (source, lang; "(no sidecar)" for a document without a line).
  Lines without a ȳ count here.
- (a) Over the accepted lines with a ȳ: agree = those with −2 < ȳ < +2; bound = the
  one-sided Clopper-Pearson lower bound at α = 0.05, Beta.ppf(α, agree, n − agree + 1).
  PASS when bound ≥ 0.90. Also given per slice.
- (b) Per complete run (document) with an accepted line: off = mean ȳ over its
  positions; gate = the same with the pinned level at the accepted positions: --level
  L at every τ, or else round(LEVEL, 2) of the same τ; Δ = gate − off. A document
  with a position without a ȳ has no off-mode mean (off mode fails such a run): it is
  skipped, and counted. PASS with ≥ 100 documents and a 95th percentile of |Δ| ≤ 0.5,
  by nearest rank: the ⌈0.95·N⌉-th smallest |Δ|. The mean Δ is reported too, and
  documents, p95 |Δ| and mean Δ per slice at the τ tested.
- Model: NEUTRAL.model is the analysed lines' model without its "@<ctx>"
  (calibration_model: the context window does not change the scores, the rest of the
  string does); the lines left out above do not count. Analysed lines of several
  models are bad input. Analysed lines without a model field (aiagent 0.7.0) take
  --model's: when none names one, --model gives the model (with neither, nothing is
  pinned); when some do, --model must vouch for the others (bad input without it). A
  --model that differs from the model the lines name is bad input. --model must be a
  model string an LM reports, "<provider>/<model>::<think|nothink>[@<ctx>]", or the
  guard would never match it. The header, the JSON (lines.no_model) and the block say
  how many analysed lines took --model's.
- The block pins tau = T, round(LEVEL, 2), and se, σ_b and σ_w to 4 decimals (the JSON
  has them unrounded too). Without --pin-tau there is no block: the log only brackets
  the installed τ. With --pin-tau and --level (the confirmation run) there is none
  either: the last line is "CONFIRMATION at τ T, level L: PASS" (or FAIL).
- `make check` skips tools/: run ruff and mypy on it by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, get_args

from aiagent.core.sentiment import (
    NEUTRAL_LABEL,
    SCORE_SIGNATURE_SHA256,
    calibration_model,
)
from aiagent.core.sentiment_stats import SCALE_MAX, SCALE_MIN
from aiagent.distill.gates import clopper_pearson_lower
from aiagent.llm.registry import Reasoning

DEFAULT_SWEEP: Final = (0.90, 0.92, 0.94, 0.96, 0.98)
BAND: Final = 2.0  # (a): −BAND < ȳ < +BAND, the aggregate's own neutral band
ALPHA: Final = 0.05
MIN_BOUND: Final = 0.90  # (a): the owner's certification level
MIN_DOCUMENTS: Final = 100  # (b)
MAX_P95_DRIFT: Final = 0.5  # (b)
PERCENTILE: Final = 95  # (b), by nearest rank
LEVEL_DECIMALS: Final = 2  # NEUTRAL.level, like an LLM segment's mean
PIN_DECIMALS: Final = 4  # NEUTRAL.se, sigma_between, sigma_within
TOKEN_BIN_EDGES: Final = (32, 64, 128, 256, 512, 1024)
SLICE_KEYS: Final = ("source", "lang")
NO_SIDECAR: Final = "(no sidecar)"
NO_VALUE: Final = "(none)"
MAX_LINE: Final = 88  # ruff's line length: the pasted block goes into src/
NO_MODEL: Final = "no model (pass --model)"
NO_TAU: Final = (
    "no τ (pass --pin-tau T): a calibration pins the τ it was measured and tested at, "
    "and the log only brackets the installed one"
)
FROM_LOG, FROM_OPTION, FROM_BOTH = "log", "--model", "log and --model"
# compose_model_string's "<provider>/<model>::<reasoning>", once "@<ctx>" is dropped.
_REASONING: Final = "|".join(get_args(Reasoning))
MODEL_SHAPE: Final = re.compile(rf"[^/]+/.+::(?:{_REASONING})")
MODEL_FORM: Final = f"<provider>/<model>::<{_REASONING}>[@<ctx>]"
EXIT_PASS, EXIT_FAIL, EXIT_INPUT = 0, 1, 2

Stats = dict[str, Any]  # JSON-ready numbers


class InputError(Exception):
    """The log or the sidecar cannot be analysed; the message says where and why."""


@dataclass(frozen=True)
class Line:
    """One shadow-log line (§2.4), with its readable LLM samples in rollout order."""

    ts: str
    artifact_id: str
    model: str | None  # the LM's, "@<ctx>" included; None in a 0.7.0 log
    run_id: str
    seg_index: int
    n_segments: int
    doc_sha256: str
    n_tokens: int
    fits: bool
    student: str | None
    confidence: float | None
    would_accept: bool
    samples: tuple[float, ...]

    def accepted(self, tau: float | None) -> bool:
        """At the installed τ (None) would_accept; at τ' fits, neutral, conf ≥ τ'."""
        if tau is None:
            return self.would_accept
        return (
            self.fits
            and self.student == NEUTRAL_LABEL
            and self.confidence is not None
            and self.confidence >= tau
        )


@dataclass(frozen=True)
class Run:
    """One forward call: a document's lines by seg_index."""

    run_id: str
    doc_sha256: str
    lines: tuple[Line, ...]


@dataclass(frozen=True)
class Runs:
    """The complete runs kept for the analysis, and what was left out."""

    read: int
    kept: tuple[Run, ...]
    incomplete: tuple[Stats, ...]
    duplicates: tuple[Stats, ...]

    @property
    def lines(self) -> list[Line]:
        """The kept runs' lines, in order: the lines the analysis uses."""
        return [line for run in self.kept for line in run.lines]



def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _is_samples(value: object) -> bool:
    return isinstance(value, list) and all(s is None or _is_number(s) for s in value)


def _json_object(text: str, where: str) -> dict[str, Any]:
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{where}: not JSON ({exc.msg})") from None
    if not isinstance(record, dict):
        raise InputError(f"{where}: not a JSON object")
    return record


def parse_line(text: str, where: str) -> Line | None:
    """One log line; None when it is not JSON (a line torn by a killed process), and
    InputError naming `where` and the field unless it is §2.4's."""
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        raise InputError(f"{where}: not a JSON object")

    def get(name: str, ok: Callable[[object], bool], expected: str) -> Any:
        value = record.get(name)
        if not ok(value):
            raise InputError(f"{where}: {name} must be {expected}, got {value!r}")
        return value

    count, optional = "a non-negative integer", "a string or null"
    samples = get("llm_samples", _is_samples, "a list of numbers and nulls")
    return Line(
        ts=get("ts", _is_str, "a string"),
        artifact_id=get("artifact_id", _is_str, "a string"),
        model=get("model", lambda v: v is None or _is_str(v), optional),
        run_id=get("run_id", _is_str, "a string"),
        seg_index=get("seg_index", _is_count, count),
        n_segments=get("n_segments", lambda v: _is_count(v) and v != 0, "positive"),
        doc_sha256=get("doc_sha256", _is_str, "a string"),
        n_tokens=get("n_tokens", _is_count, count),
        fits=get("fits", lambda v: isinstance(v, bool), "a bool"),
        student=get("student", lambda v: v is None or _is_str(v), optional),
        confidence=get("confidence", lambda v: v is None or _is_number(v), "a number"),
        would_accept=get("would_accept", lambda v: isinstance(v, bool), "a bool"),
        samples=tuple(float(s) for s in samples if s is not None),
    )


def _read_lines(path: Path) -> Iterable[tuple[int, str]]:
    """(line number, text) per non-blank line; InputError if the file cannot be read."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: cannot read it ({exc})") from None
    for number, raw in enumerate(text.splitlines(), start=1):
        if raw.strip():
            yield number, raw


def read_log(path: Path) -> tuple[list[Line], list[int]]:
    """The log's lines, and the numbers of those that are not JSON; InputError if no
    line is readable or they mix students."""
    lines: list[Line] = []
    unreadable: list[int] = []
    for number, raw in _read_lines(path):
        parsed = parse_line(raw, f"{path}: line {number}")
        if parsed is None:
            unreadable.append(number)
        else:
            lines.append(parsed)
    if not lines:
        skipped = f" ({len(unreadable)} unreadable)" if unreadable else ""
        raise InputError(f"{path}: no shadow lines{skipped}")
    students = list(dict.fromkeys(line.artifact_id for line in lines))
    if len(students) > 1:
        raise InputError(
            f"{path}: the log mixes students ({', '.join(students)}); a calibration "
            "holds for one: split the log by artifact_id"
        )
    return lines, unreadable


def resolve_model(
    log: Path, lines: Sequence[Line], option: str | None
) -> tuple[str | None, str | None]:
    """(the model NEUTRAL pins, FROM_LOG, FROM_OPTION or FROM_BOTH), or (None, None)
    without one, over the analysed `lines`. InputError if they name several models, if
    --model (`option`, normalized) differs from theirs, or if only some name one and
    no --model vouches for the others."""
    models = list(dict.fromkeys(
        calibration_model(line.model) for line in lines if line.model is not None
    ))
    if len(models) > 1:
        raise InputError(
            f"{log}: its analysed lines mix several models ({', '.join(models)}); a "
            "calibration holds for one: split the log by model"
        )
    if not models:
        return (None, None) if option is None else (option, FROM_OPTION)
    named = models[0]
    if option is not None and option != named:
        raise InputError(
            f"{log}: its analysed lines were scored with {named}, but --model is "
            f"{option}"
        )
    unnamed = sum(1 for line in lines if line.model is None)
    if not unnamed:
        return named, FROM_LOG
    if option is None:
        raise InputError(
            f"{log}: {unnamed} of its {len(lines)} analysed lines name no model and "
            f"the others {named}: pass --model {named} if those (an aiagent 0.7.0 "
            "run) were scored with it too, or else split the log"
        )
    return named, FROM_BOTH


def _sidecar_key(record: Mapping[str, Any], where: str) -> str:
    if isinstance(record.get("doc_sha256"), str):
        return str(record["doc_sha256"])
    doc_id = record.get("doc_id")
    if isinstance(doc_id, str) and doc_id.startswith("sha256:"):
        return doc_id.removeprefix("sha256:")
    raise InputError(
        f'{where}: a sidecar line needs "doc_sha256": "<hex>" or '
        '"doc_id": "sha256:<hex>"'
    )


def read_sidecar(path: Path) -> dict[str, dict[str, str]]:
    """doc_sha256 -> {source, lang}; the first line per document wins."""
    sidecar: dict[str, dict[str, str]] = {}
    for number, raw in _read_lines(path):
        where = f"{path}: line {number}"
        record = _json_object(raw, where)
        values = {key: record.get(key) for key in SLICE_KEYS}
        strings = {k: v if isinstance(v, str) else NO_VALUE for k, v in values.items()}
        sidecar.setdefault(_sidecar_key(record, where), strings)
    return sidecar


def _run_problem(members: Sequence[Line]) -> str | None:
    """Why a run's lines are not one whole document; None when they are."""
    first = members[0]
    if any(
        m.n_segments != first.n_segments or m.doc_sha256 != first.doc_sha256
        for m in members
    ):
        return "its lines disagree on n_segments or doc_sha256"
    indexes = sorted(m.seg_index for m in members)
    if indexes == list(range(first.n_segments)):
        return None
    if len(set(indexes)) < len(indexes):
        return f"repeats a seg_index ({indexes})"
    return f"has {len(indexes)} of {first.n_segments} segments (seg_index {indexes})"


def group_runs(lines: Sequence[Line]) -> Runs:
    """Complete runs in log order, the first per document; the others reported."""
    by_run: dict[str, list[Line]] = {}
    for line in lines:
        by_run.setdefault(line.run_id, []).append(line)
    kept: list[Run] = []
    incomplete: list[Stats] = []
    duplicates: list[Stats] = []
    first_run: dict[str, str] = {}
    for run_id, members in by_run.items():
        doc, n_segments = members[0].doc_sha256, members[0].n_segments
        problem = _run_problem(members)
        if problem is not None:
            incomplete.append({
                "run_id": run_id,
                "doc_sha256": doc,
                "lines": len(members),
                "n_segments": n_segments,
                "problem": problem,
            })
        elif doc in first_run:
            duplicates.append(
                {"doc_sha256": doc, "run_id": run_id, "kept_run_id": first_run[doc]}
            )
        else:
            first_run[doc] = run_id
            ordered = tuple(sorted(members, key=lambda m: m.seg_index))
            kept.append(Run(run_id, doc, ordered))
    return Runs(len(by_run), tuple(kept), tuple(incomplete), tuple(duplicates))



def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    """The sample variance (n − 1); needs at least two values."""
    mu = _mean(values)
    return math.fsum((v - mu) ** 2 for v in values) / (len(values) - 1)


def _accepted(lines: Iterable[Line], tau: float | None) -> list[tuple[float, ...]]:
    """The samples of each accepted line that has a ȳ."""
    return [line.samples for line in lines if line.accepted(tau) and line.samples]


def neutral_stats(lines: Sequence[Line], tau: float | None) -> Stats:
    """LEVEL, n, SE, σ_w and σ_b over the accepted lines with a ȳ."""
    accepted = _accepted(lines, tau)
    means = [_mean(samples) for samples in accepted]
    n = len(means)
    within = [_variance(samples) for samples in accepted if len(samples) >= 2]
    var_means = _variance(means) if n >= 2 else None
    sigma_w2 = _mean(within) if within else None
    mean_inv_k = _mean([1 / len(samples) for samples in accepted]) if n else None
    sigma_b2 = None
    if var_means is not None and sigma_w2 is not None and mean_inv_k is not None:
        sigma_b2 = max(0.0, var_means - sigma_w2 * mean_inv_k)
    return {
        "n": n,
        "level": _mean(means) if n else None,
        "se": None if var_means is None else math.sqrt(var_means / n),
        "var_means": var_means,
        "n_within": len(within),
        "sigma_within": None if sigma_w2 is None else math.sqrt(sigma_w2),
        "mean_inv_k": mean_inv_k,
        "sigma_between": None if sigma_b2 is None else math.sqrt(sigma_b2),
    }


def agreement(lines: Iterable[Line], tau: float | None) -> Stats:
    """(a): the accepted lines with −2 < ȳ < +2, and the one-sided CP lower bound."""
    means = [_mean(samples) for samples in _accepted(lines, tau)]
    agree = sum(1 for mean in means if -BAND < mean < BAND)
    bound = clopper_pearson_lower(agree, len(means), ALPHA)
    passes = bound >= MIN_BOUND
    return {"n": len(means), "agree": agree, "bound": bound, "passes": passes}


def nearest_rank(values: Sequence[float], percentile: int) -> float | None:
    """The ⌈percentile/100 · N⌉-th smallest value; None for no values."""
    if not values:
        return None
    rank = -(-percentile * len(values) // 100)
    return sorted(values)[rank - 1]


def drift(runs: Sequence[Run], tau: float | None, level: float | None) -> Stats:
    """(b): Δ = gate − off per document with an accepted line; `level` is pinned."""
    deltas: list[Stats] = []
    skipped = 0
    for run in runs:
        accepted = [line.accepted(tau) for line in run.lines]
        if not any(accepted):
            continue
        if level is None or not all(line.samples for line in run.lines):
            skipped += 1
            continue
        off = [_mean(line.samples) for line in run.lines]
        gate = [level if a else mean for a, mean in zip(accepted, off, strict=True)]
        delta = _mean(gate) - _mean(off)
        deltas.append(
            {"run_id": run.run_id, "doc_sha256": run.doc_sha256, "delta": delta}
        )
    values = [d["delta"] for d in deltas]
    p95 = nearest_rank([abs(v) for v in values], PERCENTILE)
    enough = len(deltas) >= MIN_DOCUMENTS
    return {
        "level": level,
        "documents": len(deltas),
        "skipped": skipped,
        "mean_delta": _mean(values) if values else None,
        "p95_abs_delta": p95,
        "passes": enough and p95 is not None and p95 <= MAX_P95_DRIFT,
        "deltas": deltas,
    }


def _bin(n_tokens: int) -> str:
    low = 0
    for edge in TOKEN_BIN_EDGES:
        if n_tokens < edge:
            return f"{low}-{edge - 1}"
        low = edge
    return f"{low}+"


def _shares(lines: Sequence[Line], tau: float | None) -> Stats:
    n = len(lines)
    accepted = sum(1 for line in lines if line.accepted(tau))
    too_long = sum(1 for line in lines if not line.fits)
    return {
        "lines": n,
        "accepted": accepted,
        "coverage": accepted / n if n else None,
        "too_long": too_long,
        "too_long_share": too_long / n if n else None,
    }


def coverage(lines: Sequence[Line], tau: float | None) -> Stats:
    """Coverage and the too_long share, overall and per non-empty n_tokens bin."""
    labels = [_bin(edge - 1) for edge in TOKEN_BIN_EDGES] + [_bin(TOKEN_BIN_EDGES[-1])]
    bins = []
    for label in labels:
        members = [line for line in lines if _bin(line.n_tokens) == label]
        if members:
            bins.append({"bin": label, **_shares(members, tau)})
    return {**_shares(lines, tau), "by_n_tokens": bins}


def analyse(runs: Sequence[Run], tau: float | None, level: float | None) -> Stats:
    """Every number at one τ (None: the installed one); (b) at `level` if given, else
    at this τ's own round(LEVEL, 2)."""
    lines = [line for run in runs for line in run.lines]
    stats = neutral_stats(lines, tau)
    pinned = level
    if pinned is None and stats["level"] is not None:
        pinned = round(stats["level"], LEVEL_DECIMALS)
    agree = agreement(lines, tau)
    moved = drift(runs, tau, pinned)
    return {
        "tau": tau,
        **stats,
        "coverage": coverage(lines, tau),
        "agreement": agree,
        "drift": moved,
        "passes": agree["passes"] and moved["passes"],
    }


def _slice(runs: Sequence[Run], tau: float | None, level: float | None) -> Stats:
    """Coverage, (a) and (b) of one slice at `tau` (None: the installed τ), (b) at
    `level`."""
    lines = [line for run in runs for line in run.lines]
    moved = drift(runs, tau, level)
    keep = ("documents", "skipped", "mean_delta", "p95_abs_delta")
    return {
        **coverage(lines, tau),
        "agreement": agreement(lines, tau),
        "drift": {name: moved[name] for name in keep},
    }


def slices(
    runs: Sequence[Run],
    sidecar: Mapping[str, Mapping[str, str]],
    tau: float | None,
    level: float | None,
) -> Stats:
    """Coverage, (a) and (b) at the τ tested (`tau`, None: the installed one), per
    source and per lang; (b) at that analysis's pinned `level`."""
    result: Stats = {}
    for key in SLICE_KEYS:
        groups: dict[str, list[Run]] = {}
        for run in runs:
            value = sidecar.get(run.doc_sha256, {}).get(key, NO_SIDECAR)
            groups.setdefault(value, []).append(run)
        result[key] = {
            value: _slice(members, tau, level)
            for value, members in sorted(groups.items())
        }
    return result


def implied_tau(lines: Iterable[Line]) -> dict[str, float | None]:
    """The installed τ as would_accept shows it: above the highest neutral confidence
    it rejected, at most the lowest it accepted."""
    accepted, rejected = [], []
    for line in lines:
        if line.confidence is None:
            continue
        if line.would_accept:
            accepted.append(line.confidence)
        elif line.fits and line.student == NEUTRAL_LABEL:
            rejected.append(line.confidence)
    above, at_most = max(rejected, default=None), min(accepted, default=None)
    return {"above": above, "at_most": at_most}



def neutral_calibration(
    tested: Stats, artifact_id: str, model: str | None, measured: str
) -> tuple[Stats | None, str | None]:
    """(NeutralCalibration's fields as pinned, None), or (None, why it cannot be), from
    the analysis at the τ tested (`tested`, at the installed τ when its tau is None)."""
    fields = ("level", "se", "sigma_between", "sigma_within")
    missing = [name for name in fields if tested[name] is None]
    if missing:
        return None, (
            f"{', '.join(missing)} undefined over {tested['n']} accepted segments: "
            "SE needs n ≥ 2, and σ_w an accepted segment with ≥ 2 LLM samples "
            "(shadow-run at --resample 3)"
        )
    if tested["tau"] is None:
        return None, NO_TAU
    if model is None:
        return None, NO_MODEL
    return {
        "artifact_id": artifact_id,
        "score_signature_sha256": SCORE_SIGNATURE_SHA256,
        "model": model,
        "tau": tested["tau"],
        "level": round(tested["level"], LEVEL_DECIMALS),
        "se": round(tested["se"], PIN_DECIMALS),
        "sigma_between": round(tested["sigma_between"], PIN_DECIMALS),
        "sigma_within": round(tested["sigma_within"], PIN_DECIMALS),
        "n": tested["n"],
        "measured": measured,
    }, None


def build_summary(
    log: Path,
    lines: Sequence[Line],
    unreadable: Sequence[int],
    runs: Runs,
    sidecar: Mapping[str, Mapping[str, str]] | None,
    sweep: Sequence[float],
    measured: str | None,
    level: float | None,
    model: tuple[str | None, str | None],
    pin_tau: float | None,
) -> Stats:
    """Every number of the analysis, JSON-ready; `level` is a given pin for (b), `model`
    resolve_model's answer, and `pin_tau` the τ to calibrate and test at (None: the
    installed one)."""
    analysed = runs.lines
    installed = analyse(runs.kept, None, level)
    pinned = None if pin_tau is None else analyse(runs.kept, pin_tau, level)
    tested = installed if pinned is None else pinned
    measured_given = measured is not None
    if measured is None:
        stamps = sorted(line.ts for line in analysed or lines)
        measured = (
            f"{log.name}, {stamps[0]} to {stamps[-1]}: {len(runs.kept)} documents, "
            f"{len(analysed)} segments"
        )
    pinned_model, model_from = model
    pin: Stats | None
    cannot_pin: str | None
    if level is not None:
        given = f"--level {level:g} was"
        if pin_tau is not None:
            given = f"--pin-tau {pin_tau:g} and --level {level:g} were"
        pin, cannot_pin = None, (
            f"{given} given: (b) evaluates that pin on this log, so this log "
            "calibrates nothing"
        )
    else:
        artifact_id = lines[0].artifact_id
        pin, cannot_pin = neutral_calibration(
            tested, artifact_id, pinned_model, measured
        )
    no_sample = [line for line in analysed if not line.samples]
    unmatched = None
    if sidecar is not None:
        unmatched = sum(1 for run in runs.kept if run.doc_sha256 not in sidecar)
    return {
        "log": str(log),
        "artifact_id": lines[0].artifact_id,
        "score_signature_sha256": SCORE_SIGNATURE_SHA256,
        "model": pinned_model,
        "model_from": model_from,
        "runs": {
            "read": runs.read,
            "complete": len(runs.kept) + len(runs.duplicates),
            "analysed": len(runs.kept),
            "incomplete": list(runs.incomplete),
            "duplicates": list(runs.duplicates),
        },
        "lines": {
            "read": len(lines),
            "unreadable": list(unreadable),
            "analysed": len(analysed),
            "no_sample": len(no_sample),
            "no_sample_accepted": sum(1 for line in no_sample if line.would_accept),
            "no_model": sum(1 for line in analysed if line.model is None),
        },
        "installed_tau": implied_tau(analysed),
        "given_level": level,
        "pin_tau": pin_tau,
        "confirmation": pin_tau is not None and level is not None,
        "installed": installed,
        "pinned": pinned,
        "passes": tested["passes"],
        "sweep": [analyse(runs.kept, tau, level) for tau in sweep],
        "slices": (
            None
            if sidecar is None
            else slices(runs.kept, sidecar, pin_tau, tested["drift"]["level"])
        ),
        "documents_without_sidecar": unmatched,
        "neutral_calibration": pin,
        "measured_given": measured_given,
        "cannot_pin": cannot_pin,
    }



def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _verdict(passes: bool) -> str:
    return "PASS" if passes else "FAIL"


def _tau_label(tau: float | None) -> str:
    if tau is None:
        return "installed"
    text = f"{tau:.2f}"
    return text if float(text) == tau else repr(tau)


def _fraction(agreement: Stats) -> str:
    return f"{agreement['agree']}/{agreement['n']}"


def _model_row(summary: Stats) -> str:
    """The model, and where it is from: how many analysed lines took --model's."""
    model, source = summary["model"], summary["model_from"]
    unnamed, analysed = summary["lines"]["no_model"], summary["lines"]["analysed"]
    if model is None:
        return "model (no analysed line names one: pass --model)"
    if source == FROM_LOG:
        return f"model {model} (from the log)"
    if source == FROM_OPTION:
        return (
            f"model {model} (from --model: none of the {analysed} analysed lines names "
            "one)"
        )
    return (
        f"model {model} (from the log; from --model for the {unnamed} of {analysed} "
        "analysed lines that name none)"
    )


def _header(summary: Stats) -> list[str]:
    runs, lines, tau = summary["runs"], summary["lines"], summary["installed_tau"]
    rows = [
        f"Sentiment System 1 calibration from {summary['log']}",
        f"student {summary['artifact_id']}",
        _model_row(summary),
        f"runs: {runs['read']} read, {runs['complete']} complete, "
        f"{len(runs['incomplete'])} incomplete, {len(runs['duplicates'])} duplicate "
        f"documents; {runs['analysed']} analysed",
        f"lines: {lines['analysed']} analysed, {len(lines['unreadable'])} unreadable, "
        f"{lines['no_sample']} with no readable LLM sample "
        f"({lines['no_sample_accepted']} of them accepted)",
        f"τ implied by would_accept: above {_num(tau['above'])}, "
        f"at most {_num(tau['at_most'])}",
    ]
    if summary["pin_tau"] is not None:
        rows.append(
            f"τ tested: {_tau_label(summary['pin_tau'])} (--pin-tau: fits, neutral, "
            "confidence ≥ it); the sweep's first row is the installed τ"
        )
    if lines["unreadable"]:
        numbers = ", ".join(map(str, lines["unreadable"]))
        rows.append(
            f"  left out, not JSON (a line torn by a killed process?): line {numbers}"
        )
    rows += [
        f"  left out, incomplete: run {r['run_id']} ({r['problem']})"
        for r in runs["incomplete"]
    ]
    return rows + [
        f"  left out, duplicate document: run {d['run_id']} (doc "
        f"{d['doc_sha256'][:12]}, kept run {d['kept_run_id']})"
        for d in runs["duplicates"]
    ]


def _table(rows: Sequence[Sequence[object]], widths: Sequence[int]) -> list[str]:
    """Rows of cells: the first column left-aligned, the others right-aligned."""
    lines = []
    for row in rows:
        (first, width), *rest = zip(row, widths, strict=True)
        text = f"{first!s:<{width}}" + "".join(f"{c!s:>{w}}" for c, w in rest)
        lines.append(f"  {text}".rstrip())
    return lines


SHARE_HEAD: Final = ("lines", "accepted", "coverage", "too_long")
SHARE_WIDTHS: Final = (7, 10, 10, 10)


def _share_cells(shares: Stats) -> tuple[object, ...]:
    """lines, accepted, coverage and the too_long share, for a table row."""
    coverage, too_long = _pct(shares["coverage"]), _pct(shares["too_long_share"])
    return shares["lines"], shares["accepted"], coverage, too_long


def _bins_table(bins: Sequence[Stats]) -> list[str]:
    rows = [("n_tokens", *SHARE_HEAD), *((b["bin"], *_share_cells(b)) for b in bins)]
    return _table(rows, (10, *SHARE_WIDTHS))


def _given(level: float | None) -> str:
    return "" if level is None else f" at the given level {level:g}"


def _at(tau: float | None) -> str:
    """The τ an analysis is at, for a title: the installed one, or a given one."""
    return "the installed τ" if tau is None else f"τ {_tau_label(tau)}"


def _tested(summary: Stats) -> Stats:
    """The analysis the calibration, the pass test and the verdict rest on: at
    --pin-tau's τ when given, else at the installed one."""
    tested: Stats = summary["pinned"] or summary["installed"]
    return tested


def _tested_section(tested: Stats, level: float | None) -> list[str]:
    cov, a, b = tested["coverage"], tested["agreement"], tested["drift"]
    at = _at(tested["tau"])
    return [
        "",
        f"Accepted neutral segments at {at}",
        f"  LEVEL {_num(tested['level'])} over n = {tested['n']}, "
        f"SE {_num(tested['se'])}",
        f"  σ_w {_num(tested['sigma_within'])} (pooled over "
        f"{tested['n_within']} segments with ≥ 2 LLM samples), "
        f"σ_b {_num(tested['sigma_between'])} (var(ȳ) "
        f"{_num(tested['var_means'])}, mean 1/k {_num(tested['mean_inv_k'])})",
        f"  coverage {cov['accepted']}/{cov['lines']} = {_pct(cov['coverage'])}, "
        f"too_long {cov['too_long']}/{cov['lines']} = {_pct(cov['too_long_share'])}",
        *_bins_table(cov["by_n_tokens"]),
        "",
        f"Pass test at {at}",
        f"  (a) band agreement: {_fraction(a)} in (-{BAND:g}, +{BAND:g}), CP lower "
        f"bound {a['bound']:.4f} (α = {ALPHA}, need ≥ {MIN_BOUND:.2f}): "
        f"{_verdict(a['passes'])}",
        f"  (b) mean drift{_given(level)}: {b['documents']} documents (need ≥ "
        f"{MIN_DOCUMENTS}), "
        f"p95 |Δ| {_num(b['p95_abs_delta'])} (need ≤ {MAX_P95_DRIFT}), mean Δ "
        f"{_num(b['mean_delta'])}, {b['skipped']} skipped for a position without "
        f"a ȳ: {_verdict(b['passes'])}",
    ]


def _sweep(summary: Stats) -> list[str]:
    rows: list[Sequence[object]] = [
        ("τ", "n", "LEVEL", "coverage", "(a) agree", "bound", "", "(b) docs",
         "p95 |Δ|", "mean Δ", "")
    ]
    for row in [summary["installed"], *summary["sweep"]]:
        a, b = row["agreement"], row["drift"]
        rows.append((
            _tau_label(row["tau"]), row["n"], _num(row["level"]),
            _pct(row["coverage"]["coverage"]), _fraction(a), f"{a['bound']:.4f}",
            _verdict(a["passes"]), b["documents"], _num(b["p95_abs_delta"]),
            _num(b["mean_delta"]), _verdict(b["passes"]),
        ))
    title = (
        "τ sweep (accepted: fits, neutral, confidence ≥ τ; installed: would_accept"
        f"){_given(summary['given_level'])}"
    )
    table = _table(rows, (10, 6, 9, 10, 11, 8, 6, 10, 9, 9, 6))
    few = [row for row in [summary["installed"], *summary["sweep"]]
           if row["drift"]["documents"] < MIN_DOCUMENTS]
    if few:
        table.append(
            f"  (b) FAIL with fewer than {MIN_DOCUMENTS} documents is the count, not "
            "the drift: see p95 |Δ|."
        )
    return ["", title, *table]


def _slices(summary: Stats) -> list[str]:
    if summary["slices"] is None:
        return []
    rows: list[str] = []
    for key, groups in summary["slices"].items():
        table: list[Sequence[object]] = [(key, *SHARE_HEAD, "(a) agree", "bound")]
        table += [
            (value, *_share_cells(s), _fraction(s["agreement"]),
             f"{s['agreement']['bound']:.4f}")
            for value, s in groups.items()
        ]
        width = max(len(value) for value in [key, *groups]) + 2
        lines = _table(table, (width, *SHARE_WIDTHS, 11, 8))
        unmatched = summary["documents_without_sidecar"]
        tested = _tested(summary)
        level = tested["drift"]["level"]
        if summary["given_level"] is not None:
            at = _given(level)
        else:
            pinned = "n/a" if level is None else f"{level:g}"
            at = f" at this log's pinned level {pinned}"
        rows += [
            "",
            f"Slices by {key} at {_at(tested['tau'])} ({unmatched} documents without "
            f"a sidecar line); (b){at}",
            lines[0],
        ]
        for line, s in zip(lines[1:], groups.values(), strict=True):
            bins = ", ".join(
                f"{b['bin']} {b['accepted']}/{b['lines']}" for b in s["by_n_tokens"]
            )
            b = s["drift"]
            rows += [
                line,
                f"    n_tokens (accepted/lines): {bins}",
                f"    (b) drift: {b['documents']} documents, p95 |Δ| "
                f"{_num(b['p95_abs_delta'])}, mean Δ {_num(b['mean_delta'])}, "
                f"{b['skipped']} skipped",
            ]
    return rows


def _comment(text: str) -> list[str]:
    return textwrap.wrap(text, MAX_LINE, initial_indent="# ", subsequent_indent="# ")


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _chunks(value: str, width: int) -> list[str]:
    """`value` in pieces whose literal, escapes and quotes included, fits in `width`
    columns: cut after whitespace, or inside a word too long for one literal."""
    chunks = [""]
    for token in re.findall(r"\S+\s*|\s+", value):
        if len(_literal(chunks[-1] + token)) > width:
            chunks.append("")
            for char in token:
                if len(_literal(chunks[-1] + char)) > width:
                    chunks.append("")
                chunks[-1] += char
        else:
            chunks[-1] += token
    return [chunk for chunk in chunks if chunk]


def _keyword(name: str, value: str) -> list[str]:
    """`name="value",` in the call or, past MAX_LINE, the value as adjacent literals."""
    single = f"    {name}={_literal(value)},"
    if len(single) <= MAX_LINE:
        return [single]
    indent = " " * 8
    chunks = _chunks(value, MAX_LINE - len(indent))
    return [f"    {name}=(", *(indent + _literal(c) for c in chunks), "    ),"]


def paste_block(summary: Stats) -> list[str]:
    """NEUTRAL for src/aiagent/core/sentiment.py, with a comment that cites the run."""
    pin = summary["neutral_calibration"]
    if pin is None:
        return _comment(f"cannot pin: {summary['cannot_pin']}")
    tested = _tested(summary)
    a, b, at = tested["agreement"], tested["drift"], _at(pin["tau"])
    rows = [] if tested["passes"] else _comment(
        f"NOT READY: the pass test fails at {at}; do not pin this."
    )
    rows += _comment(
        f"sentiment's System 1 calibration (design §2.9), measured: {pin['measured']}"
    )
    rows += _comment(
        f"n = {pin['n']} accepted neutral segments at {at}. Pass test at {at}: "
        f"(a) {_fraction(a)} in (-2, +2), CP lower bound {a['bound']:.4f}; (b) p95 "
        f"|Δ| {_num(b['p95_abs_delta'])} over {b['documents']} documents, mean Δ "
        f"{_num(b['mean_delta'])}: {_verdict(tested['passes'])}."
    )
    if summary["model_from"] != FROM_LOG:
        lines = summary["lines"]
        rows += _comment(
            f"model is from --model for the {lines['no_model']} of {lines['analysed']} "
            "analysed lines that name none (aiagent 0.7.0): the log alone does not "
            "show it for them."
        )
    rows += _comment(
        "score_signature_sha256 is SCORE_SIGNATURE_SHA256 of the aiagent this script "
        "imported: pin it only if the lab's run scored with a ScoreSegment of that "
        "same hash."
    )
    if not summary["measured_given"]:
        rows += _comment(
            "measured is the default, which names neither the --resample nor the "
            "corpus: rerun with --measured naming them."
        )
    return [
        *rows,
        "NEUTRAL: NeutralCalibration | None = NeutralCalibration(",
        *_keyword("artifact_id", pin["artifact_id"]),
        *_keyword("score_signature_sha256", pin["score_signature_sha256"]),
        *_keyword("model", pin["model"]),
        f"    tau={pin['tau']!r},",
        f"    level={pin['level']!r},",
        f"    se={pin['se']!r},",
        f"    sigma_between={pin['sigma_between']!r},",
        f"    sigma_within={pin['sigma_within']!r},",
        f"    n={pin['n']},",
        *_keyword("measured", pin["measured"]),
        ")",
    ]


def _closing(summary: Stats) -> list[str]:
    """The confirmation run's verdict, or else the block to paste."""
    if summary["confirmation"]:
        return [
            f"CONFIRMATION at {_at(summary['pin_tau'])}, level "
            f"{summary['given_level']:g}: {_verdict(summary['passes'])}"
        ]
    return [
        "Paste into src/aiagent/core/sentiment.py, in place of "
        "`NEUTRAL: NeutralCalibration | None = None`:",
        *paste_block(summary),
    ]


def render(summary: Stats) -> str:
    """The report: counts, the τ tested and its pass test, the sweep, the slices, then
    the block to paste (or the confirmation run's verdict)."""
    return "\n".join([
        *_header(summary),
        *_tested_section(_tested(summary), summary["given_level"]),
        *_sweep(summary),
        *_slices(summary),
        "",
        *_closing(summary),
    ])



def _tau(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {text}")
    return value


def _level(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not SCALE_MIN <= value <= SCALE_MAX:  # also rejects nan
        raise argparse.ArgumentTypeError(
            f"must be a score in [{SCALE_MIN:g}, {SCALE_MAX:g}], got {text}"
        )
    return value


def _model(text: str) -> str:
    """An LM's model string, "@<ctx>" dropped (calibration_model); only the shape
    compose_model_string gives can ever match the guard."""
    model = calibration_model(text.strip())
    if not MODEL_SHAPE.fullmatch(model):
        raise argparse.ArgumentTypeError(
            f"must be the model string an LM reports, {MODEL_FORM} (the model= that "
            f"`aiagent run -v` prints), got {text!r}"
        )
    return model


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentiment_calibration.py",
        description="Calibrate sentiment's System 1 from a sentiment shadow log "
        "(docs/design/sentiment-system1.md §2.9).",
    )
    sweep = " ".join(map(str, DEFAULT_SWEEP))
    parser.add_argument("log", type=Path, help="the sentiment shadow log")
    parser.add_argument("--meta", type=Path, help="doc_sha256|doc_id, source, lang")
    parser.add_argument("--tau", type=_tau, action="append", help=f"default: {sweep}")
    parser.add_argument(
        "--pin-tau",
        type=_tau,
        help="calibrate and test at this τ (fits, neutral, confidence ≥ T) instead of "
        "the installed one, and pin it as NEUTRAL.tau; with --level, the confirmation "
        "run of that pin",
    )
    parser.add_argument(
        "--level",
        type=_level,
        help="compute (b) at this pinned NEUTRAL.level (e.g. the document run's, on "
        "the review run's log) instead of this log's own; offers nothing to pin",
    )
    parser.add_argument(
        "--model",
        type=_model,
        help="the LM model string the run scored with (the model= that `aiagent run "
        "-v` prints), for lines without a model field (aiagent 0.7.0); its @<ctx> is "
        "dropped",
    )
    parser.add_argument("--measured", help="NEUTRAL.measured: names the run")
    parser.add_argument("--json", type=Path, dest="json_out", help="every number")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """0 when the pass test passes at the τ tested, 1 when not, 2 on bad input."""
    args = parse_args(argv)
    try:
        lines, unreadable = read_log(args.log)
        runs = group_runs(lines)
        model = resolve_model(args.log, runs.lines, args.model)
        sidecar = None if args.meta is None else read_sidecar(args.meta)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    sweep = tuple(args.tau) if args.tau else DEFAULT_SWEEP
    summary = build_summary(
        args.log, lines, unreadable, runs, sidecar, sweep, args.measured, args.level,
        model, args.pin_tau,
    )
    print(render(summary))
    if args.json_out is not None:
        text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        try:
            args.json_out.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.json_out}: {exc}", file=sys.stderr)
            return EXIT_INPUT
    return EXIT_PASS if summary["passes"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
