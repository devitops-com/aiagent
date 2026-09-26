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

The heavy lifting (segmentation, statistics) lives in pure, dspy-free modules
(:mod:`aiagent.core.segment`, :mod:`aiagent.core.sentiment_stats`); this module
only orchestrates the LM calls and packs the result into a ``dspy.Prediction``.
"""

from __future__ import annotations

import contextvars
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Final

import dspy
from dspy.utils.exceptions import AdapterParseError

from aiagent.core.pipeline import Pipeline
from aiagent.core.segment import split_segments
from aiagent.core.sentiment_stats import SCALE_MAX, SCALE_MIN, summarize
from aiagent.exceptions import AiagentError, SourceError

DEFAULT_RESAMPLE = 1
DEFAULT_MAX_SEGMENTS = 24
MAX_IN_FLIGHT: Final = 4  # the devai teacher runs --max-num-seqs 4
SAMPLE_TEMPERATURE: Final = 0.7  # > 0, or the rollout ids would give one answer r times
MAX_EXTRA_SAMPLES: Final = 2  # rollouts r, r+1 for a segment none of whose r parsed
_EXCERPT_BUDGET = 2000

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


class SentimentModule(Pipeline):
    """Segment → resample-score → aggregate → explain sentiment pipeline."""

    default_alias = "default"

    def __init__(self) -> None:
        super().__init__()
        self.score = dspy.ChainOfThought(ScoreSegment)
        self.explain = dspy.Predict(ExplainSentiment)

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
        scored = self._score_distinct(list(dict.fromkeys(segments)), passes)
        seg_samples, seg_rationales = _per_position(segments, scored)

        stats = summarize(seg_samples)
        seg_means = [round(sum(s) / len(s), 2) for s in seg_samples]
        excerpts = "\n".join(
            f"- ({mean:+.1f}) {rationale}"
            for mean, rationale in zip(seg_means, seg_rationales)
        )[:_EXCERPT_BUDGET]

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
                {"score": mean, "rationale": rationale}
                for mean, rationale in zip(seg_means, seg_rationales)
            ],
            explanation=explanation,
        )

    def _score_distinct(
        self, segments: list[str], passes: int
    ) -> dict[str, list[_Sample | None]]:
        """The samples of each segment, in rollout order (None: unparseable).

        ``passes`` samples of every segment, then, for a segment none of them parsed,
        up to :data:`MAX_EXTRA_SAMPLES` more, one rollout at a time until one parses.
        Any exception other than a parse failure cancels the pending samples and
        propagates.
        """
        results: dict[str, list[_Sample | None]] = {
            segment: [None] * passes for segment in segments
        }
        with ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT) as pool:
            self._draw(pool, results, [(s, r) for s in segments for r in range(passes)])
            for rollout in range(passes, passes + MAX_EXTRA_SAMPLES):
                unread = [
                    segment
                    for segment, samples in results.items()
                    if all(sample is None for sample in samples)
                ]
                for segment in unread:
                    results[segment].append(None)
                self._draw(pool, results, [(segment, rollout) for segment in unread])
        return results

    def _draw(
        self,
        pool: ThreadPoolExecutor,
        results: dict[str, list[_Sample | None]],
        jobs: list[tuple[str, int]],
    ) -> None:
        """Run one sample per (segment, rollout) job into its slot of ``results``."""
        # dspy keeps its context overrides in contextvars: a copy per task.
        futures: dict[Future[_Sample | None], tuple[str, int]] = {
            pool.submit(
                contextvars.copy_context().run, self._sample, segment, rollout
            ): (segment, rollout)
            for segment, rollout in jobs
        }
        try:
            for future in as_completed(futures):
                segment, rollout = futures[future]
                results[segment][rollout] = future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

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


def _per_position(
    segments: list[str], scored: dict[str, list[_Sample | None]]
) -> tuple[list[list[float]], list[str]]:
    """Each position's readable scores and rationale; a repeated segment shares its
    samples. AiagentError if a segment has no readable sample at all."""
    seg_samples: list[list[float]] = []
    seg_rationales: list[str] = []
    for index, segment in enumerate(segments):
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


def _opt_round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
