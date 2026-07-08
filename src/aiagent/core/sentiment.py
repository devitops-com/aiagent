"""The sentiment-analysis signatures and module.

Scores content on an integer −10 (very negative) … +10 (very positive) scale.
To produce genuine statistics rather than one opaque number, the corpus is split
into segments and each segment is scored several times (``resample``); the
per-segment means give a *content-volatility* distribution while the repeated
samples give a *model-uncertainty* estimate — the "both combined" strategy. A
final ``dspy.Predict`` turns the aggregate into a plain-language explanation.

The heavy lifting (segmentation, statistics) lives in pure, dspy-free modules
(:mod:`aiagent.core.segment`, :mod:`aiagent.core.sentiment_stats`); this module
only orchestrates the LM calls and packs the result into a ``dspy.Prediction``.
"""

from __future__ import annotations

from typing import Any

import dspy

from aiagent.core.pipeline import Pipeline
from aiagent.core.segment import split_segments
from aiagent.core.sentiment_stats import SCALE_MAX, SCALE_MIN, summarize
from aiagent.exceptions import SourceError

DEFAULT_RESAMPLE = 3
DEFAULT_MAX_SEGMENTS = 24
_EXCERPT_BUDGET = 2000


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


def _parse_score(value: Any) -> float:
    """Coerce a model score to a float clamped to the −10..+10 scale."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        parsed = 0.0
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
        seg_samples: list[list[float]] = []
        seg_rationales: list[str] = []
        for segment in segments:
            samples: list[float] = []
            rationale = ""
            for _ in range(passes):
                pred = self.score(text=segment)
                samples.append(_parse_score(pred.score))
                rationale = pred.rationale or rationale
            seg_samples.append(samples)
            seg_rationales.append(rationale)

        stats = summarize(seg_samples)
        seg_means = [round(sum(s) / len(s), 2) for s in seg_samples]
        excerpts = "\n".join(
            f"- ({mean:+.1f}) {rationale}"
            for mean, rationale in zip(seg_means, seg_rationales)
        )[:_EXCERPT_BUDGET]

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
            model_uncertainty=round(stats.model_uncertainty, 3),
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
            segments=[
                {"score": mean, "rationale": rationale}
                for mean, rationale in zip(seg_means, seg_rationales)
            ],
            explanation=explanation,
        )


def _opt_round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None
