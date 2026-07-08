"""Unit tests for the pure sentiment statistics and segmentation helpers."""

from __future__ import annotations

import pytest

from aiagent.core.segment import split_segments
from aiagent.core.sentiment_stats import summarize, two_sided_t_p
from aiagent.exceptions import SourceError

# --- statistics -----------------------------------------------------------


def test_two_sided_p_matches_known_critical_value() -> None:
    # t = 2.262157 at df = 9 is the two-sided 0.05 critical value.
    assert two_sided_t_p(2.262157, 9) == pytest.approx(0.05, abs=1e-4)


def test_two_sided_p_is_one_at_zero_t() -> None:
    assert two_sided_t_p(0.0, 5) == pytest.approx(1.0)


def test_varied_segments_have_volatility_and_significance() -> None:
    stats = summarize([[6, 7, 5], [3, 4, 2], [8, 9, 7], [1, 2, 0]])
    assert stats.n_segments == 4
    assert stats.n_samples == 12
    assert stats.mean == pytest.approx(4.5)
    assert stats.volatility > 0
    assert stats.model_uncertainty > 0
    assert 0.0 < stats.p_value < 1.0  # type: ignore[operator]
    assert stats.ci_low is not None and stats.ci_low < stats.mean < stats.ci_high  # type: ignore[operator]
    assert stats.polarity == "positive"


def test_single_segment_has_no_significance() -> None:
    stats = summarize([[5, 5, 5]])
    assert stats.n_segments == 1
    assert stats.volatility == 0.0
    assert stats.t_statistic is None
    assert stats.p_value is None
    assert stats.ci_low is None
    assert stats.confidence == "insufficient-data"


def test_unanimous_non_neutral_is_significant() -> None:
    stats = summarize([[8], [8], [8]])
    assert stats.std_error == 0.0
    assert stats.p_value == 0.0
    assert stats.t_statistic is None  # unanimous, non-neutral
    assert stats.confidence == "high"


def test_unanimous_neutral_is_not_significant() -> None:
    stats = summarize([[0], [0], [0]])
    assert stats.t_statistic == 0.0
    assert stats.p_value == pytest.approx(1.0)
    assert stats.polarity == "neutral / mixed"
    assert stats.confidence == "not-significant"


def test_scores_are_clamped_into_the_stats_via_polarity_bands() -> None:
    assert summarize([[10], [9], [8]]).polarity == "very positive"
    assert summarize([[-10], [-9], [-8]]).polarity == "very negative"


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(SourceError):
        summarize([])


# --- segmentation ---------------------------------------------------------


def test_split_prefers_paragraphs() -> None:
    segments = split_segments("First para.\n\nSecond para.\n\nThird.", max_segments=10)
    assert segments == ["First para.", "Second para.", "Third."]


def test_split_falls_back_to_sentences_for_single_block() -> None:
    segments = split_segments("Great job. Terrible idea. It works!", max_segments=10)
    assert segments == ["Great job.", "Terrible idea.", "It works!"]


def test_split_caps_by_merging_without_dropping_content() -> None:
    paragraphs = "\n\n".join(f"p{i}" for i in range(10))
    segments = split_segments(paragraphs, max_segments=3)
    assert len(segments) <= 3
    # Every original unit survives inside some merged segment.
    joined = " ".join(segments)
    for i in range(10):
        assert f"p{i}" in joined


def test_split_empty_text_yields_no_segments() -> None:
    assert split_segments("   \n\n  ", max_segments=5) == []
