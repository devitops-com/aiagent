"""Unit tests for the pure sentiment statistics and segmentation helpers."""

from __future__ import annotations

import math

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
    assert stats.model_uncertainty == pytest.approx(1.0)  # every segment's variance is 1
    assert stats.n_resampled == 4
    assert 0.0 < stats.p_value < 1.0  # type: ignore[operator]
    assert stats.ci_low is not None and stats.ci_low < stats.mean < stats.ci_high  # type: ignore[operator]
    assert stats.polarity == "positive"


def test_single_segment_has_no_significance() -> None:
    stats = summarize([[5, 5, 5]])
    assert stats.n_segments == 1
    assert stats.volatility == 0.0
    assert stats.model_uncertainty == 0.0
    assert stats.n_resampled == 1
    assert stats.t_statistic is None
    assert stats.p_value is None
    assert stats.ci_low is None
    assert stats.confidence == "insufficient-data"


def test_model_uncertainty_is_pooled_over_the_resampled_segments() -> None:
    # Variances 1, 0 and 8; the single-sample segment has no spread to pool.
    stats = summarize([[1, 2, 3], [5, 5, 5], [0, 4], [7]])
    assert stats.model_uncertainty == pytest.approx(3.0**0.5)  # sqrt of the mean variance
    assert stats.model_uncertainty != pytest.approx((1 + 0 + 8**0.5) / 3)  # not the mean std
    assert stats.n_resampled == 3
    assert stats.n_samples == 9  # every sample used, the single one too
    assert stats.n_segments == 4


def test_model_uncertainty_is_none_when_no_segment_was_resampled() -> None:
    stats = summarize([[1], [2], [6]])
    assert stats.model_uncertainty is None
    assert stats.n_resampled == 0
    assert stats.n_samples == 3
    assert stats.volatility > 0  # the rest of the statistics still stand


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


# --- the student term (System 1) --------------------------------------------


def test_student_scores_count_for_the_mean() -> None:
    stats = summarize([[4, 6], [-2]], student_scores=[0.5, 0.5], student_variance=1.0)
    assert stats.n_segments == 4
    assert stats.mean == pytest.approx((5 - 2 + 0.5 + 0.5) / 4)
    assert stats.polarity == "neutral / mixed"


def test_volatility_and_std_error_include_the_student_term() -> None:
    sigma2 = 1.5  # the calibrated σ²(r) of a student segment
    stats = summarize([[4, 6], [-2, -2]], student_scores=[0.5, 0.5], student_variance=sigma2)
    scores = [5.0, -2.0, 0.5, 0.5]  # the segment means, student and LLM alike
    mu = sum(scores) / 4
    s2 = sum((x - mu) ** 2 for x in scores) / 3
    volatility = math.sqrt(s2 + (2 / 4) * sigma2)  # + (n_student / n) σ²(r)
    assert stats.volatility == pytest.approx(volatility)
    assert stats.std_error == pytest.approx(volatility / math.sqrt(4))
    assert stats.t_statistic == pytest.approx(mu / stats.std_error)
    assert stats.ci_low == pytest.approx(mu - 1.959963984540054 * stats.std_error)
    # The same scores without the term would claim more certainty than they have.
    assert summarize([[5], [-2], [0.5], [0.5]]).std_error < stats.std_error


def test_at_full_coverage_the_std_error_is_not_below_the_student_spread() -> None:
    sigma2 = 0.81
    stats = summarize([], student_scores=[0.4] * 9, student_variance=sigma2)
    assert stats.mean == pytest.approx(0.4)
    assert stats.std_error >= math.sqrt(sigma2) / math.sqrt(9) - 1e-12
    assert stats.std_error == pytest.approx(0.9 / 3)  # not 0: every score is the same
    assert stats.p_value is not None and stats.p_value > 0.0
    assert (stats.n_samples, stats.n_resampled, stats.model_uncertainty) == (0, 0, None)


def test_one_student_segment_alone_has_its_spread_but_no_test() -> None:
    stats = summarize([], student_scores=[0.4], student_variance=0.25)
    assert stats.std_error == pytest.approx(0.5)
    assert stats.t_statistic is None and stats.p_value is None
    assert stats.confidence == "insufficient-data"


def test_model_uncertainty_ignores_the_student_segments() -> None:
    stats = summarize([[1, 2, 3], [5]], student_scores=[0.0, 0.0], student_variance=4.0)
    assert stats.model_uncertainty == pytest.approx(1.0)  # only [1, 2, 3] is resampled
    assert stats.n_resampled == 1
    assert stats.n_samples == 4  # LLM samples only
    assert stats.n_segments == 4


def test_model_uncertainty_is_none_at_one_sample_with_student_segments() -> None:
    stats = summarize([[3], [-1]], student_scores=[0.0], student_variance=1.0)
    assert (stats.model_uncertainty, stats.n_resampled, stats.n_samples) == (None, 0, 2)


def test_the_student_variance_needs_student_segments() -> None:
    assert summarize([[1], [3]], student_variance=5.0) == summarize([[1], [3]])


def test_summarize_rejects_no_segments_at_all() -> None:
    with pytest.raises(SourceError):
        summarize([], student_scores=[], student_variance=1.0)


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
