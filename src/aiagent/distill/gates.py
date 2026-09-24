"""The ship gate: score a student on calib and held-out, fit τ, certify, decide.

Per question, τ is fitted on **calib** at ``fit_precision``, stricter than the target:
fitted at the target itself, held-out precision lands at about the target and its
Clopper-Pearson lower bound below it, so a calibrated student would almost never
ship. On **held-out**, the rows with ``answer_confidence >= τ`` are accepted, and the
question passes when the one-sided CP lower bound of their precision reaches
``precision`` at ``alpha`` and they cover at least ``min_coverage`` of held-out.
Fitting τ and certifying on the same rows would overstate the result.

:func:`decide` turns the question reports into ship / repair / stop, and the saved
:class:`EvalReport` drives ``repair`` and ``install``. numpy (for the ECE) comes in
with the runtime anyway; there is no scipy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from aiagent.exceptions import DatasetContractError, DistillError
from aiagent.system1.contract import content_sha256, safe_id
from aiagent.system1.runtime import Answer, System1Runtime

logger = logging.getLogger(__name__)

PRECISION_POINTS: Final = (0.90, 0.95, 0.98)
ECE_BINS: Final = 15
Verdict = Literal["ship", "repair", "stop"]
EXIT_CODES: Final[Mapping[Verdict, int]] = {"ship": 0, "repair": 3, "stop": 4}

_BISECTION_STEPS: Final = 100


@dataclass(frozen=True)
class GateTargets:
    """The ship-gate targets (recorded in each report)."""

    precision: float = 0.95
    fit_precision: float = 0.98
    alpha: float = 0.05
    min_coverage: float = 0.20
    epsilon: float = 0.01
    max_rounds: int = 3

    def __post_init__(self) -> None:
        """ValueError unless 0 < precision <= fit_precision < 1 and 0 < alpha < 1."""
        if not 0 < self.precision <= self.fit_precision < 1:
            raise ValueError(
                f"need 0 < precision ({self.precision}) <= fit_precision "
                f"({self.fit_precision}) < 1"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


def min_certifiable(precision: float, alpha: float) -> int:
    """ceil(ln alpha / ln precision): the fewest accepted held-out rows, all correct,
    whose CP lower bound reaches `precision` (59 at 0.95/0.05)."""
    return math.ceil(math.log(alpha) / math.log(precision))


@dataclass(frozen=True)
class EvalItem:
    """One dataset row as the gate sees it."""

    row_id: str
    state: Mapping[str, str]
    labels: Mapping[str, str]  # qid -> the teacher's label (empty for pool rows)
    ids_sha256: Mapping[str, str]  # qid -> the dataset's student_tokens ids_sha256


@dataclass(frozen=True)
class Scored:
    """The student's answer to one question of one row, and the teacher's label."""

    row_id: str
    label: str
    answer: Answer

    @property
    def correct(self) -> bool:
        """The student's answer key equals the teacher's label."""
        return self.answer.key == self.label


def score_items(
    runtime: System1Runtime,
    items: Sequence[EvalItem],
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Scored]]:
    """Predict every item, qid -> scored rows in item order.

    DatasetContractError if a row's ids_sha256 differs from H(encoded ids): the student
    would not see what it was trained on.
    """
    scored: dict[str, list[Scored]] = {qid: [] for qid in questions}
    for item in items:
        for qid, answer in runtime.predict(dict(item.state), questions).items():
            ids_sha256 = content_sha256(list(answer.encoded.input_ids))
            if ids_sha256 != item.ids_sha256.get(qid):
                raise DatasetContractError(
                    f"row {item.row_id}: the student's tokens for {qid!r} differ from "
                    "the dataset's ids_sha256 (tokenizer or sequence limits changed)"
                )
            scored[qid].append(Scored(item.row_id, item.labels[qid], answer))
    return scored


def _log_upper_tail(log_coefs: Sequence[float], k: int, n: int, p: float) -> float:
    """ln P(X >= k) for X ~ Binomial(n, p), 0 < p < 1, summed in log space."""
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [c + i * log_p + (n - i) * log_q for i, c in enumerate(log_coefs, start=k)]
    top = max(terms)
    return top + math.log(math.fsum(math.exp(t - top) for t in terms))


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided lower bound, Beta.ppf(alpha, k, n-k+1); 0.0 when k == 0 or n == 0.

    Bisection for the p with P(X >= k | n, p) = alpha.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return 0.0
    log_n = math.lgamma(n + 1)
    log_coefs = [
        log_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k, n + 1)
    ]
    target = math.log(alpha)
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        if _log_upper_tail(log_coefs, k, n, mid) < target:
            lo = mid
        else:
            hi = mid
    return lo


def expected_calibration_error(scored: Sequence[Scored], bins: int = ECE_BINS) -> float:
    """laya's ece_score on answer_confidence (same bin edges); 0.0 for no rows.

    laya returns NaN for no rows, which a JSON report cannot hold.
    """
    if not scored:
        return 0.0
    conf = np.array([s.answer.answer_confidence for s in scored], dtype=np.float64)
    correct = np.array([s.correct for s in scored], dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            gap = abs(float(conf[sel].mean()) - float(correct[sel].mean()))
            ece += float(sel.mean()) * gap
    return ece


def brier_score(scored: Sequence[Scored]) -> float:
    """Mean over rows of sum_k (p_k - onehot(label)_k)^2; 0.0 for no rows."""
    if not scored:
        return 0.0
    total = 0.0
    for s in scored:
        pairs = zip(s.answer.keys, s.answer.probabilities, strict=True)
        total += sum((p - (1.0 if key == s.label else 0.0)) ** 2 for key, p in pairs)
    return total / len(scored)


def accuracy(scored: Sequence[Scored]) -> float:
    """The share of rows the student got right; 0.0 for no rows."""
    return sum(s.correct for s in scored) / len(scored) if scored else 0.0


def _operating_points(scored: Sequence[Scored]) -> list[tuple[float, int, int]]:
    """(t, accepted, correct) of {conf >= t} for every distinct confidence t."""
    ordered = sorted(scored, key=lambda s: s.answer.answer_confidence, reverse=True)
    points = []
    accepted = correct = 0
    for i, s in enumerate(ordered):
        accepted += 1
        correct += s.correct
        t = s.answer.answer_confidence
        if i + 1 == len(ordered) or ordered[i + 1].answer.answer_confidence != t:
            points.append((t, accepted, correct))
    return points


def fit_threshold(scored: Sequence[Scored], *, target_precision: float) -> float | None:
    """The lowest confidence t whose accepted set {conf >= t} reaches the target
    precision (max coverage); None if none does."""
    passing = [
        t for t, n, k in _operating_points(scored) if k / n >= target_precision
    ]
    return min(passing) if passing else None


def coverage_at_precision(scored: Sequence[Scored], precision: float) -> float:
    """Descriptive: the best coverage with precision >= p on this set."""
    covered = [n for _, n, k in _operating_points(scored) if k / n >= precision]
    return max(covered) / len(scored) if covered else 0.0


@dataclass(frozen=True)
class QuestionReport:
    """Per-question metrics; the gate uses tau (from calib) on held-out."""

    qid: str
    n_calib: int
    n_heldout: int
    accuracy: float  # held-out, like ece, brier and coverage_at_precision
    ece: float
    brier: float
    coverage_at_precision: Mapping[str, float]  # "0.90" -> coverage
    tau: float | None
    accepted: int
    correct: int
    precision: float | None
    precision_lower: float
    coverage: float
    passes: bool


def question_report(
    qid: str,
    calib: Sequence[Scored],
    heldout: Sequence[Scored],
    targets: GateTargets,
) -> QuestionReport:
    """Fit τ on calib at fit_precision, then certify precision on held-out."""
    tau = fit_threshold(calib, target_precision=targets.fit_precision)
    accepted = [
        s for s in heldout if tau is not None and s.answer.answer_confidence >= tau
    ]
    n, k = len(accepted), sum(s.correct for s in accepted)
    lower = clopper_pearson_lower(k, n, targets.alpha)
    coverage = n / len(heldout) if heldout else 0.0
    return QuestionReport(
        qid=qid,
        n_calib=len(calib),
        n_heldout=len(heldout),
        accuracy=accuracy(heldout),
        ece=expected_calibration_error(heldout),
        brier=brier_score(heldout),
        coverage_at_precision={
            f"{p:.2f}": coverage_at_precision(heldout, p) for p in PRECISION_POINTS
        },
        tau=tau,
        accepted=n,
        correct=k,
        precision=k / n if n else None,
        precision_lower=lower,
        coverage=coverage,
        passes=(
            tau is not None
            and lower >= targets.precision
            and coverage >= targets.min_coverage
        ),
    )


def decide(
    questions: Sequence[QuestionReport],
    *,
    round: int,
    pool_remaining: int,
    previous_accuracy: float | None,
    targets: GateTargets,
) -> tuple[Verdict, str]:
    """ship / repair / stop and why; the first matching rule wins.

    ``accuracy`` is the mean held-out accuracy over the questions.
    """
    if not questions:
        raise ValueError("decide needs at least one question report")
    if all(q.passes for q in questions):
        return "ship", (
            f"every question certifies precision ≥ {targets.precision} at "
            f"α={targets.alpha} with coverage ≥ {targets.min_coverage}"
        )
    needed = min_certifiable(targets.precision, targets.alpha)
    smallest = min(q.n_heldout for q in questions)
    if smallest < needed:
        return "stop", (
            f"held-out has {smallest} rows; certifying {targets.precision} at "
            f"α={targets.alpha} needs ≥ {needed} accepted rows; label a larger corpus"
        )
    if round + 1 >= targets.max_rounds:
        return "stop", f"no rounds left (round {round + 1} of {targets.max_rounds})"
    if pool_remaining == 0:
        return "stop", "pool exhausted: no unlabeled rows left to repair with"
    mean = sum(q.accuracy for q in questions) / len(questions)
    if previous_accuracy is not None and mean - previous_accuracy < targets.epsilon:
        return "stop", (
            f"held-out accuracy {mean:.4f} did not improve by ≥ ε={targets.epsilon} "
            f"over the previous round's {previous_accuracy:.4f}"
        )
    failing = ", ".join(q.qid for q in questions if not q.passes)
    return "repair", (
        f"{failing}: below the gate; label the pool rows the student is least sure of"
    )


def _verdict(value: object) -> Verdict:
    if value == "ship" or value == "repair" or value == "stop":
        return value
    raise ValueError(f"unknown verdict {value!r}")


@dataclass(frozen=True)
class EvalReport:
    """A saved eval result; drives repair/install."""

    run: str
    artifact_id: str
    dataset_id: str
    parent_dataset: str | None
    campaign: str
    round: int
    skill: str
    predictor: str
    skill_source_sha256: str
    source_changed_since_labeling: bool
    targets: GateTargets
    questions: tuple[QuestionReport, ...]
    accuracy: float
    pool_remaining: int
    previous_run: str | None
    previous_accuracy: float | None
    files: Mapping[str, str]  # VerifiedArtifact.files; install re-hashes its copies
    verdict: Verdict
    reason: str
    created_at: str

    def to_json(self) -> dict[str, Any]:
        """Plain JSON data, in field order."""
        data = asdict(self)
        data["questions"] = list(data["questions"])
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EvalReport:
        """Parse to_json() output; KeyError/TypeError/ValueError if malformed."""
        fields = dict(data)
        questions = tuple(
            QuestionReport(
                **{**q, "coverage_at_precision": dict(q["coverage_at_precision"])}
            )
            for q in fields["questions"]
        )
        return cls(
            **{
                **fields,
                "targets": GateTargets(**fields["targets"]),
                "questions": questions,
                "files": dict(fields["files"]),
                "verdict": _verdict(fields["verdict"]),
            }
        )


def _report_path(artifacts_dir: Path, run: str) -> Path:
    try:
        safe_id(run, "run")
    except ValueError as exc:
        raise DistillError(str(exc)) from exc
    return artifacts_dir / "system1" / "evals" / f"{run}.json"


def save_report(artifacts_dir: Path, report: EvalReport) -> Path:
    """<artifacts_dir>/system1/evals/<run>.json, written atomically."""
    path = _report_path(artifacts_dir, report.run)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_json()
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_report(artifacts_dir: Path, run: str) -> EvalReport | None:
    """The saved report for `run`, None if there is none; DistillError if unreadable."""
    path = _report_path(artifacts_dir, run)
    if not path.exists():
        return None
    try:
        return EvalReport.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DistillError(f"unreadable eval report {path}: {exc}") from exc


def find_report_for_dataset(artifacts_dir: Path, dataset_id: str) -> EvalReport | None:
    """The newest (created_at) report whose dataset_id matches (unreadable: skipped)."""
    matches = []
    for path in sorted((artifacts_dir / "system1" / "evals").glob("*.json")):
        try:
            report = load_report(artifacts_dir, path.stem)
        except DistillError as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            continue
        if report is not None and report.dataset_id == dataset_id:
            matches.append(report)
    return max(matches, key=lambda r: (r.created_at, r.run), default=None)


def select_for_repair(
    runtime: System1Runtime,
    pool: Sequence[EvalItem],
    questions: Mapping[str, Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[str]:
    """Pool row ids by ascending min(answer_confidence) over questions, ties by row id,
    at most max_rows."""
    ranked = sorted(
        (_least_confidence(runtime, item, questions), item.row_id) for item in pool
    )
    return [row_id for _, row_id in ranked[: max(0, max_rows)]]


def _least_confidence(
    runtime: System1Runtime,
    item: EvalItem,
    questions: Mapping[str, Mapping[str, Any]],
) -> float:
    answers = runtime.predict(dict(item.state), questions)
    return min(a.answer_confidence for a in answers.values())
