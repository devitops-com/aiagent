"""The sentiment-analysis signatures and module.

Scores content on an integer −10 (very negative) … +10 (very positive) scale.
To produce genuine statistics rather than one opaque number, the corpus is split
into segments and each segment is scored ``resample`` times; the per-segment means
give a *content-volatility* distribution while the repeated samples give a
*model-uncertainty* estimate — the "both combined" strategy. A final
``dspy.Predict`` turns the aggregate into a plain-language explanation.

Sample *j* of a segment is one call with ``rollout_id=j`` at
:data:`SAMPLE_TEMPERATURE`: distinct rollout ids are distinct DSPy cache keys, so
every sample reaches the LLM, while re-running the same text is answered from the
cache with the same samples. Score calls run under DSPy's ChatAdapter **without**
its JSONAdapter fallback (server JSON mode, which devai backends strip, is never
requested); an unparseable sample is dropped, and any other error aborts the run.
A segment none of whose samples parses gets up to :data:`MAX_EXTRA_SAMPLES` more
rollout ids, one at a time, before the run fails: they are ordinary samples, cached
like the others.
Each distinct segment text is scored once per run. At most :data:`MAX_IN_FLIGHT`
LLM calls are in flight across the whole process, score and explain alike, however
many ``forward`` calls run at once (``run --jsonl``).

**System 1** (``system1_mode.sentiment``): the module borrows the installed polarity
student (:attr:`SentimentModule.system1_student`); ``apply_system1`` hands it over
through :meth:`SentimentModule.use_student`. The student decides each distinct
segment in order, and the segment's LLM samples go to the pool as soon as it is
decided. In ``gate`` mode a segment the student calls ``neutral`` at τ or more is
scored :data:`NEUTRAL`'s calibrated level, with no LLM call; every other segment is
scored as in off mode. ``shadow`` scores everything as off mode does. Both log one
line per segment (no text). Gate needs :data:`NEUTRAL` measured for this student and
this ``ScoreSegment``; otherwise it only shadows.

The heavy lifting (segmentation, statistics) lives in pure, dspy-free modules
(:mod:`aiagent.core.segment`, :mod:`aiagent.core.sentiment_stats`); this module
only orchestrates the LM calls and packs the result into a ``dspy.Prediction``.
Nothing here imports :mod:`aiagent.system1` unless a student is in use.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import math
import threading
import uuid
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Final

import dspy
from dspy.utils.exceptions import AdapterParseError

from aiagent.core.pipeline import Pipeline
from aiagent.core.segment import split_segments
from aiagent.core.sentiment_stats import SCALE_MAX, SCALE_MIN, summarize
from aiagent.exceptions import AiagentError, SourceError

if TYPE_CHECKING:
    from pathlib import Path

    from aiagent.system1.cascade import Mode, Student, Verdict

logger = logging.getLogger(__name__)

DEFAULT_RESAMPLE = 1
DEFAULT_MAX_SEGMENTS = 24
MAX_IN_FLIGHT: Final = 4  # the devai teacher runs --max-num-seqs 4
SAMPLE_TEMPERATURE: Final = 0.7  # > 0, or the rollout ids would give one answer r times
MAX_EXTRA_SAMPLES: Final = 2  # rollouts r, r+1 for a segment none of whose r parsed
NEUTRAL_LABEL: Final = "neutral"  # the one student label the gate accepts
_EXCERPT_BUDGET = 2000
_DECIMALS = 4  # a student confidence or τ in the output and the shadow log

# Process-wide: every LLM call of every SentimentModule holds a slot. A thread
# never holds one while it waits for its own pool, so nested pools cannot deadlock.
_IN_FLIGHT = threading.BoundedSemaphore(MAX_IN_FLIGHT)

_Sample = tuple[float, str]  # (score, rationale)


class ScoreSegment(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Rate the sentiment of a passage on an integer −10..+10 scale."""

    text: str = dspy.InputField(desc="A passage of text to assess.")
    score: int = dspy.OutputField(
        desc="Sentiment as an integer from -10 (very negative) to "
        "+10 (very positive); 0 is neutral."
    )
    rationale: str = dspy.OutputField(
        desc="One concise sentence justifying the score."
    )


@dataclass(frozen=True)
class NeutralCalibration:
    """What a segment the student calls neutral scores, from a sentiment shadow run.

    ``level`` is the mean LLM score of the segments the neutral-only gate would have
    taken, with its standard error ``se`` over ``n`` segments. ``sigma_between`` and
    ``sigma_within`` split the spread of those segments' LLM means around it (σ_b, and
    σ_w of one sample), so a student score stands in for an r-sample LLM mean with
    :meth:`variance`. It holds for the student ``artifact_id`` and the ``ScoreSegment``
    hashed in ``score_signature_sha256`` only; ``measured`` names the run.
    """

    artifact_id: str
    score_signature_sha256: str
    level: float
    se: float
    sigma_between: float
    sigma_within: float
    n: int
    measured: str

    def variance(self, resample: int) -> float:
        """σ²(r) = σ_b² + σ_w²/r: a student score against an r-sample LLM mean."""
        return self.sigma_between**2 + self.sigma_within**2 / resample


# signature_sha256(ScoreSegment), pinned: a test fails when ScoreSegment changes, since
# a calibration measured before no longer applies (recalibrate, then pin both again).
SCORE_SIGNATURE_SHA256: Final = (
    "b26dee349163d3219febe678ead8fd8feb1c8e13e3c0469a883f7e3cb7febc96"
)
# None until a lab run pins one (docs/design/sentiment-system1.md §2.9): gate shadows.
NEUTRAL: NeutralCalibration | None = None


class ExplainSentiment(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Explain an aggregate sentiment result in plain language."""

    overall: float = dspy.InputField(desc="Mean sentiment on the -10..+10 scale.")
    polarity: str = dspy.InputField(desc="Qualitative polarity label.")
    volatility: float = dspy.InputField(
        desc="Std-dev of sentiment across segments (higher = more mixed)."
    )
    confidence: str = dspy.InputField(
        desc="Statistical confidence that sentiment differs from neutral."
    )
    excerpts: str = dspy.InputField(desc="Representative per-segment rationales.")
    explanation: str = dspy.OutputField(
        desc="2-4 sentences explaining the sentiment, how consistent it is across "
        "the content, and how reliable the conclusion is."
    )


def _parse_score(value: Any) -> float | None:
    """A model score as a float clamped to the −10..+10 scale; None if unreadable."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(SCALE_MIN, min(SCALE_MAX, parsed))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _would_accept(verdict: Verdict | None) -> bool:
    """The neutral-only gate's call on one verdict: whole, neutral, at τ or more."""
    return (
        verdict is not None and verdict.clears_tau and verdict.label == NEUTRAL_LABEL
    )


@dataclass(frozen=True)
class _System1:
    """The borrowed student, its effective mode, and (gate only) the calibration."""

    student: Student
    mode: Mode
    shadow_log: Path
    neutral: NeutralCalibration | None  # set iff the gate may score segments

    def consult(self, segment: str) -> Verdict | None:
        return self.student.consult(self.student.state(segment))

    def takes(self, verdict: Verdict | None) -> bool:
        """True when the student scores this segment in place of the LLM."""
        return self.neutral is not None and _would_accept(verdict)


class SentimentModule(Pipeline):
    """Segment → resample-score → aggregate → explain sentiment pipeline."""

    default_alias = "default"
    # System 1: the installed polarity student scores, in place of `score`, the
    # segments it calls neutral; its shadow log lives under sentiment/score/.
    system1_student: ClassVar[tuple[str, str]] = ("polarity", "classify")
    system1_predictor: ClassVar[str] = "score"

    def __init__(self) -> None:
        super().__init__()
        self.score = dspy.ChainOfThought(ScoreSegment)
        self.explain = dspy.Predict(ExplainSentiment)
        self._system1: _System1 | None = None

    def use_student(self, student: Student, mode: Mode, shadow_log: Path) -> None:
        """Consult `student` on every segment from now on (apply_system1 calls this).

        Gate needs :data:`NEUTRAL` measured for this student and the current
        ``ScoreSegment``; otherwise it warns once and only shadows.
        """
        from aiagent.distill.questions import signature_sha256  # lazy: aiagent.system1

        calibration = NEUTRAL
        if mode == "gate" and (
            calibration is None
            or calibration.artifact_id != student.installed.artifact_id
            or calibration.score_signature_sha256 != signature_sha256(ScoreSegment)
        ):
            logger.warning(
                "sentiment's System 1 calibration is missing or stale (none is pinned "
                "for the student %s and the current ScoreSegment), so it only shadows; "
                "a sentiment shadow run on the target corpus measures one",
                student.installed.artifact_id,
            )
            mode = "shadow"
        neutral = calibration if mode == "gate" else None
        self._system1 = _System1(student, mode, shadow_log, neutral)

    def forward(
        self,
        text: str,
        resample: int = DEFAULT_RESAMPLE,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
    ) -> dspy.Prediction:
        segments = split_segments(text, max_segments=max_segments)
        if not segments:
            raise SourceError("no analyzable text in the provided sources")

        passes = max(1, resample)
        system1 = self._system1
        scored, verdicts = self._score_distinct(
            list(dict.fromkeys(segments)), passes, system1
        )
        seg_samples, seg_rationales = _per_position(segments, scored)
        if system1 is not None:
            _log_shadow(system1, text, segments, scored, verdicts)

        n_student = sum(1 for segment in segments if segment not in scored)
        level = variance = 0.0
        if n_student:
            # The student takes a segment only in a calibrated gate.
            assert system1 is not None and system1.neutral is not None
            level = system1.neutral.level
            variance = system1.neutral.variance(passes)
        stats = summarize(
            [samples for samples in seg_samples if samples],
            student_scores=[level] * n_student,
            student_variance=variance,
        )
        seg_scores = [
            round(sum(s) / len(s), 2) if s else round(level, 2) for s in seg_samples
        ]
        excerpts = "\n".join(
            f"- ({score:+.1f}) {rationale}"
            for score, rationale in zip(seg_scores, seg_rationales)
            if rationale is not None
        )[:_EXCERPT_BUDGET]
        if n_student:
            note = f"{n_student} of {len(segments)} segments neutral (System 1)"
            excerpts = f"{excerpts}\n{note}" if excerpts else note

        with _IN_FLIGHT:
            explanation = self.explain(
                overall=round(stats.mean, 2),
                polarity=stats.polarity,
                volatility=round(stats.volatility, 2),
                confidence=stats.confidence,
                excerpts=excerpts,
            ).explanation

        return dspy.Prediction(
            sentiment=round(stats.mean, 2),
            polarity=stats.polarity,
            volatility=round(stats.volatility, 3),
            model_uncertainty=_opt_round(stats.model_uncertainty, 3),
            std_error=round(stats.std_error, 3),
            t_statistic=_opt_round(stats.t_statistic, 3),
            significance_p=_opt_round(stats.p_value, 4),
            confidence=stats.confidence,
            ci95=(
                [round(stats.ci_low, 2), round(stats.ci_high, 2)]
                if stats.ci_low is not None and stats.ci_high is not None
                else None
            ),
            n_segments=stats.n_segments,
            n_samples=stats.n_samples,
            n_resampled=stats.n_resampled,
            segments=[
                {
                    "score": score,
                    "rationale": rationale,
                    "source": "llm" if segment in scored else "student",
                    "student": _student_answer(verdicts[segment]),
                }
                for segment, score, rationale in zip(
                    segments, seg_scores, seg_rationales
                )
            ],
            system1=_system1_block(system1, segments, verdicts),
            explanation=explanation,
        )

    def _score_distinct(
        self, segments: list[str], passes: int, system1: _System1 | None
    ) -> tuple[dict[str, list[_Sample | None]], dict[str, Verdict | None]]:
        """The LLM samples of each segment the student did not take, in rollout order
        (None: unparseable), and the student's verdict on every segment.

        The student decides the segments in order, on this thread; a segment it does
        not take has its ``passes`` samples submitted before the next is decided. A
        segment none of whose samples parsed then gets up to :data:`MAX_EXTRA_SAMPLES`
        more, one rollout at a time until one parses. Any exception other than a parse
        failure cancels the pending samples and propagates.
        """
        results: dict[str, list[_Sample | None]] = {}
        verdicts: dict[str, Verdict | None] = {}
        with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as pool:
            try:
                futures: dict[Future[_Sample | None], tuple[str, int]] = {}
                for segment in segments:
                    verdict = None if system1 is None else system1.consult(segment)
                    verdicts[segment] = verdict
                    if system1 is not None and system1.takes(verdict):
                        continue  # the student scores it: no LLM call at all
                    results[segment] = [None] * passes
                    futures |= self._submit(pool, segment, range(passes))
                _collect(futures, results)
                for rollout in range(passes, passes + MAX_EXTRA_SAMPLES):
                    unread = [
                        segment
                        for segment, samples in results.items()
                        if all(sample is None for sample in samples)
                    ]
                    futures = {}
                    for segment in unread:
                        results[segment].append(None)
                        futures |= self._submit(pool, segment, [rollout])
                    _collect(futures, results)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return results, verdicts

    def _submit(
        self, pool: ThreadPoolExecutor, segment: str, rollouts: Iterable[int]
    ) -> dict[Future[_Sample | None], tuple[str, int]]:
        """Submit one sample per rollout of `segment`."""
        # dspy keeps its context overrides in contextvars: a copy per task.
        return {
            pool.submit(
                contextvars.copy_context().run, self._sample, segment, rollout
            ): (segment, rollout)
            for rollout in rollouts
        }

    def _sample(self, segment: str, rollout: int) -> _Sample | None:
        """One score call: sample ``rollout`` of ``segment``; None if unparseable."""
        adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
        config = {"rollout_id": rollout, "temperature": SAMPLE_TEMPERATURE}
        with dspy.context(adapter=adapter), _IN_FLIGHT:
            try:
                pred = self.score(text=segment, config=config)
            except AdapterParseError:
                return None
        score = _parse_score(pred.score)
        if score is None:
            return None
        return score, str(pred.rationale or "")


def _collect(
    futures: dict[Future[_Sample | None], tuple[str, int]],
    results: dict[str, list[_Sample | None]],
) -> None:
    """Put each finished sample in its slot of ``results``; the first error raises."""
    for future in as_completed(futures):
        segment, rollout = futures[future]
        results[segment][rollout] = future.result()


def _per_position(
    segments: list[str], scored: dict[str, list[_Sample | None]]
) -> tuple[list[list[float]], list[str | None]]:
    """Each position's readable LLM scores and rationale; a repeated segment shares
    its samples, and a student segment has none (rationale None). AiagentError if an
    LLM segment has no readable sample at all."""
    seg_samples: list[list[float]] = []
    seg_rationales: list[str | None] = []
    for index, segment in enumerate(segments):
        if segment not in scored:  # the student's
            seg_samples.append([])
            seg_rationales.append(None)
            continue
        samples = [sample for sample in scored[segment] if sample is not None]
        if not samples:
            raise AiagentError(
                f"sentiment: the model gave no readable score for segment "
                f"{index + 1} of {len(segments)} ({len(scored[segment])} samples). "
                "With the cache on (the default), re-running the same text fails the "
                "same way: a higher --resample draws new samples, and "
                "AIAGENT_CACHE=false draws them all again"
            )
        seg_samples.append([score for score, _ in samples])
        # Samples are in rollout order: the first rationale is the same whatever
        # the --resample setting or the order the samples completed in.
        seg_rationales.append(next((r for _, r in samples if r), ""))
    return seg_samples, seg_rationales


def _student_answer(verdict: Verdict | None) -> dict[str, Any] | None:
    """{label, confidence} when the student answered the segment, else None."""
    if verdict is None or verdict.answer_confidence is None:
        return None
    return {
        "label": verdict.label,
        "confidence": round(verdict.answer_confidence, _DECIMALS),
    }


def _system1_block(
    system1: _System1 | None, segments: list[str], verdicts: dict[str, Verdict | None]
) -> dict[str, Any] | None:
    """The output's ``system1`` summary; None when no student gave a verdict."""
    if system1 is None or all(verdicts[s] is None for s in segments):
        return None
    installed = system1.student.installed
    accepted = sum(_would_accept(verdicts[s]) for s in segments)
    too_long = sum(
        1 for s in segments if (verdict := verdicts[s]) is not None and not verdict.fits
    )
    return {
        "mode": system1.mode,
        "student": f"{installed.skill}/{installed.predictor}",
        "artifact_id": installed.artifact_id,
        "tau": round(system1.student.tau, _DECIMALS),
        "accepted": accepted,
        "too_long": too_long,
        "coverage": round(accepted / len(segments), 3),
    }


def _log_shadow(
    system1: _System1,
    text: str,
    segments: list[str],
    scored: dict[str, list[_Sample | None]],
    verdicts: dict[str, Verdict | None],
) -> None:
    """One shadow-log line per segment position the student gave a verdict on (no
    text), all written at once; a failure only warns."""
    from aiagent.system1.cascade import append_shadow  # loaded by apply_system1

    run_id = uuid.uuid4().hex  # groups one forward's lines
    common = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_id": system1.student.installed.artifact_id,
        "run_id": run_id,
    }
    doc_sha256 = _sha256(text)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, segment in enumerate(segments):
        verdict = verdicts[segment]
        if verdict is None:
            continue
        confidence = verdict.answer_confidence
        records.append({
            **common,
            "seg_index": index,
            "n_segments": len(segments),
            "doc_sha256": doc_sha256,
            "input_sha256": _sha256(segment),
            "n_tokens": verdict.n_tokens,
            "fits": verdict.fits,
            "student": verdict.label,
            "confidence": None if confidence is None else round(confidence, _DECIMALS),
            "would_accept": _would_accept(verdict),
            "llm_samples": [
                None if sample is None else sample[0]
                for sample in scored.get(segment, [])
            ],
            "student_ms": 0 if segment in seen else verdict.ms,  # a repeat reuses it
        })
        seen.add(segment)
    if records:
        append_shadow(system1.shadow_log, records, parents=True)


def _opt_round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
