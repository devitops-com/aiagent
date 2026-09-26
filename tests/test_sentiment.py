"""SentimentModule: real resamples, parse failures, the in-flight bound, dedupe (no network).

Per-segment answers use DummyLM's dict mode (the first key found in the last message):
the explain answer goes first, under a field header only the explain prompt contains,
and no segment text is a substring of another. List mode is not used here, since with
several calls in flight it hands out answers in completion order.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dspy
import litellm
import pytest
from dspy.clients.cache import Cache
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM

from aiagent.core import sentiment as sentiment_mod
from aiagent.core.sentiment import (
    DEFAULT_RESAMPLE,
    MAX_IN_FLIGHT,
    SAMPLE_TEMPERATURE,
    SentimentModule,
    _parse_score,
)
from aiagent.exceptions import AiagentError

EXPLAIN_KEY = "[[ ## excerpts ## ]]"  # a field only the explain prompt carries
UNREADABLE = "-3 (mildly negative)"  # ChatAdapter cannot parse it as ScoreSegment's int
EXPLAIN = {"explanation": "Mostly upbeat, with one sour note."}
SCORES = {
    "Alpha launched on time.": 4,
    "Beta crashed twice.": -6,
    "Gamma was fine.": 0,
    "Delta delighted everyone.": 8,
}
SEGMENTS = list(SCORES)
TEXT = "\n\n".join(SEGMENTS)  # paragraphs: one segment each


def segment_of(content: str) -> str:
    return next(segment for segment in SEGMENTS if segment in content)


def score_answer(segment: str, score: object | None = None) -> dict[str, str]:
    value = SCORES[segment] if score is None else score
    return {"reasoning": "weighed it", "score": str(value), "rationale": f"rated {value}"}


def answers(**overrides: dict[str, str]) -> dict[str, dict[str, str]]:
    """DummyLM dict-mode answers: explain first, then one per segment."""
    table = {EXPLAIN_KEY: EXPLAIN}
    table.update({segment: score_answer(segment) for segment in SEGMENTS})
    table.update(overrides)
    return table


def score_calls(lm: Any) -> list[dict[str, Any]]:
    return [e for e in lm.history if EXPLAIN_KEY not in e["messages"][-1]["content"]]


def rollouts_by_segment(lm: Any) -> dict[str, list[int]]:
    """The rollout ids each segment was scored with, sorted."""
    rollouts: dict[str, list[int]] = {segment: [] for segment in SEGMENTS}
    for entry in score_calls(lm):
        segment = segment_of(entry["messages"][-1]["content"])
        rollouts[segment].append(entry["kwargs"]["rollout_id"])
    return {segment: sorted(ids) for segment, ids in rollouts.items()}


class RolloutLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
    """A DummyLM whose answer depends on the prompt and on the call's rollout id."""

    def __init__(self, pick: Callable[[str, int | None], dict[str, str]]) -> None:
        super().__init__({})
        self.pick = pick

    def forward(
        self, prompt: str | None = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        answer = self.pick(messages[-1]["content"], kwargs.get("rollout_id"))
        message = dotdict(content=self._format_answer_fields(answer), tool_calls=None)
        return dotdict(
            choices=[dotdict(message=message, finish_reason="stop")],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="dummy",
        )


def run(lm: Any, text: str = TEXT, **kwargs: Any) -> Any:
    with dspy.context(lm=lm):
        return SentimentModule()(text=text, **kwargs)


# --------------------------------------------------------------------------- defaults


def test_defaults() -> None:
    assert (DEFAULT_RESAMPLE, MAX_IN_FLIGHT, SAMPLE_TEMPERATURE) == (1, 4, 0.7)
    assert sentiment_mod.MAX_EXTRA_SAMPLES == 2


def test_parse_score_clamps_and_rejects_what_is_not_a_number() -> None:
    assert _parse_score(3) == 3.0
    assert _parse_score(" -4 ") == -4.0
    assert _parse_score(14) == 10.0
    assert _parse_score(-11) == -10.0
    for bad in (None, "banana", "", float("nan"), float("inf")):
        assert _parse_score(bad) is None


# --------------------------------------------------------------------------- rollout ids


@pytest.mark.parametrize("resample", [1, 3])
def test_each_segment_is_sampled_with_rollout_ids_0_to_r_at_temperature_0_7(
    resample: int,
) -> None:
    lm = DummyLM(answers())

    run(lm, resample=resample)

    rollouts: dict[str, list[int]] = {segment: [] for segment in SEGMENTS}
    for entry in score_calls(lm):
        rollouts[segment_of(entry["messages"][-1]["content"])].append(
            entry["kwargs"]["rollout_id"]
        )
        assert entry["kwargs"]["temperature"] == 0.7
    assert {segment: sorted(ids) for segment, ids in rollouts.items()} == {
        segment: list(range(resample)) for segment in SEGMENTS
    }
    [explain] = [e for e in lm.history if EXPLAIN_KEY in e["messages"][-1]["content"]]
    assert "rollout_id" not in explain["kwargs"]


def test_each_segment_score_lands_in_its_own_slot() -> None:
    lm = DummyLM(answers())

    pred = run(lm, resample=3)

    assert [s["score"] for s in pred.segments] == [float(SCORES[s]) for s in SEGMENTS]
    assert [s["rationale"] for s in pred.segments] == [
        f"rated {SCORES[s]}" for s in SEGMENTS
    ]
    assert pred.sentiment == pytest.approx(sum(SCORES.values()) / 4)
    assert (pred.n_segments, pred.n_samples, pred.n_resampled) == (4, 12, 4)
    assert pred.model_uncertainty == 0.0  # each segment answered alike every time
    assert pred.explanation == EXPLAIN["explanation"]


def test_one_sample_per_segment_has_no_model_uncertainty() -> None:
    pred = run(DummyLM(answers()))  # the default --resample 1

    assert pred.model_uncertainty is None
    assert (pred.n_resampled, pred.n_samples, pred.n_segments) == (0, 4, 4)
    assert pred.std_error > 0  # the other statistics still stand


def test_pooled_model_uncertainty_at_three_samples() -> None:
    offsets = {0: 0, 1: 2, 2: 4}  # every segment: s, s+2, s+4 -> variance 4

    def pick(content: str, rollout: int | None) -> dict[str, str]:
        if EXPLAIN_KEY in content:
            return EXPLAIN
        segment = segment_of(content)
        assert rollout is not None
        return score_answer(segment, SCORES[segment] - 2 + offsets[rollout])

    pred = run(RolloutLM(pick), resample=3)

    assert pred.model_uncertainty == pytest.approx(2.0)
    assert (pred.n_resampled, pred.n_samples) == (4, 12)
    assert [s["score"] for s in pred.segments] == [float(SCORES[s]) for s in SEGMENTS]
    # The rationale is sample 0's, whatever order the samples completed in.
    assert [s["rationale"] for s in pred.segments] == [
        f"rated {SCORES[s] - 2}" for s in SEGMENTS
    ]


# --------------------------------------------------------------------------- the DSPy cache


def stub_reply(score: object) -> str:
    return (
        f"[[ ## reasoning ## ]]\nweighed it\n\n[[ ## score ## ]]\n{score}\n\n"
        f"[[ ## rationale ## ]]\nrated it\n\n[[ ## completed ## ]]"
    )


class StubCompletion:
    """litellm.completion's stand-in: each real request for a segment scores one higher.

    It never sees the rollout id (DSPy strips it before litellm); the id only keys
    the cache, so distinct scores mean distinct requests got past the cache.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.seen: Counter[str] = Counter()
        self.unreadable_first: set[str] = set()  # their first request does not parse

    def __call__(self, **request: Any) -> Any:
        content = request["messages"][-1]["content"]
        with self.lock:
            self.requests += 1
            if EXPLAIN_KEY in content:
                reply = f"[[ ## explanation ## ]]\n{EXPLAIN['explanation']}\n\n[[ ## completed ## ]]"
            else:
                segment = segment_of(content)
                score: object = SCORES[segment] + self.seen[segment]
                if segment in self.unreadable_first and not self.seen[segment]:
                    score = UNREADABLE
                reply = stub_reply(score)
                self.seen[segment] += 1
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            model="stub",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


@pytest.fixture
def cached_lm(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, StubCompletion]:
    """A real dspy.LM with the DSPy cache on, in memory only, over a stubbed litellm."""
    memory_only = Cache(
        enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=None
    )
    monkeypatch.setattr(dspy, "cache", memory_only)  # the disk cache comes back after
    stub = StubCompletion()
    monkeypatch.setattr(litellm, "completion", stub)
    lm = dspy.LM(
        "openai/stub", api_base="http://127.0.0.1:9/v1", api_key="local", cache=True
    )
    return lm, stub


def test_resamples_are_real_calls_and_a_rerun_is_answered_from_the_cache(
    cached_lm: tuple[Any, StubCompletion],
) -> None:
    lm, stub = cached_lm

    first = run(lm, resample=3)

    assert stub.requests == 13  # 4 segments x 3 samples + explain (the bug made 5)
    assert first.model_uncertainty == pytest.approx(1.0)  # s, s+1, s+2 (the bug: 0.0)
    assert [s["score"] for s in first.segments] == [SCORES[s] + 1.0 for s in SEGMENTS]

    second = run(lm, resample=3)

    assert stub.requests == 13  # the same text again: every call from the cache
    assert second.toDict() == first.toDict()


def test_one_sample_is_sample_0_of_three(cached_lm: tuple[Any, StubCompletion]) -> None:
    lm, stub = cached_lm

    one = run(lm, resample=1)
    assert stub.requests == 5  # 4 samples + explain
    three = run(lm, resample=3)

    assert stub.requests == 5 + 8 + 1  # rollout 0 came from the cache; new stats, new explain
    assert one.model_uncertainty is None
    assert three.model_uncertainty == pytest.approx(1.0)


def test_a_redrawn_sample_is_cached_like_any_other(
    cached_lm: tuple[Any, StubCompletion],
) -> None:
    lm, stub = cached_lm
    beta = SEGMENTS[1]
    stub.unreadable_first = {beta}

    first = run(lm)  # the default --resample 1

    assert stub.requests == 4 + 1 + 1  # rollout 0 of each, rollout 1 of beta, explain
    assert first.segments[1]["score"] == SCORES[beta] + 1.0  # beta's second request

    second = run(lm)

    assert stub.requests == 6  # the same text again: every call from the cache
    assert second.toDict() == first.toDict()

    three = run(lm, resample=3)

    assert stub.requests == 6 + 3 * 2 + 1 + 1  # rollouts 1-2 of the others, 2 of beta
    assert three.n_samples == 3 * 3 + 2  # beta's rollout 0 still does not parse


# --------------------------------------------------------------------------- parse failures


def test_an_unparseable_sample_is_dropped_and_not_retried() -> None:
    alpha = SEGMENTS[0]

    def pick(content: str, rollout: int | None) -> dict[str, str]:
        if EXPLAIN_KEY in content:
            return EXPLAIN
        segment = segment_of(content)
        if segment == alpha and rollout == 1:
            return score_answer(segment, "banana")
        return score_answer(segment)

    lm = RolloutLM(pick)
    pred = run(lm, resample=3)

    assert pred.segments[0]["score"] == float(SCORES[alpha])  # the two that parsed
    assert pred.n_samples == 11
    assert (pred.n_resampled, pred.model_uncertainty) == (4, 0.0)
    # One call per sample: the failed one was not retried through JSONAdapter.
    assert len(lm.history) == 12 + 1
    assert all("response_format" not in entry["kwargs"] for entry in lm.history)


def test_a_segment_none_of_whose_samples_parses_draws_the_next_rollout() -> None:
    beta = SEGMENTS[1]

    def pick(content: str, rollout: int | None) -> dict[str, str]:
        if EXPLAIN_KEY in content:
            return EXPLAIN
        segment = segment_of(content)
        if segment == beta and rollout == 0:
            return score_answer(segment, UNREADABLE)
        return score_answer(segment)

    lm = RolloutLM(pick)
    pred = run(lm)  # the default --resample 1

    assert [s["score"] for s in pred.segments] == [float(SCORES[s]) for s in SEGMENTS]
    assert (pred.n_samples, pred.n_resampled, pred.model_uncertainty) == (4, 0, None)
    assert rollouts_by_segment(lm) == {s: [0, 1] if s == beta else [0] for s in SEGMENTS}
    assert all(entry["kwargs"]["temperature"] == 0.7 for entry in score_calls(lm))
    assert all("response_format" not in entry["kwargs"] for entry in lm.history)


@pytest.mark.parametrize("resample", [1, 3])
def test_a_segment_without_a_readable_sample_fails_the_run(resample: int) -> None:
    beta = SEGMENTS[1]
    lm = DummyLM(answers(**{beta: score_answer(beta, UNREADABLE)}))
    tried = resample + 2  # its r samples, then 2 more rollouts

    with pytest.raises(AiagentError, match=rf"segment 2 of 4 \({tried} samples\)") as err:
        run(lm, resample=resample)

    # The answers are cached: the error says how to get new ones.
    assert "--resample" in str(err.value) and "AIAGENT_CACHE=false" in str(err.value)
    assert rollouts_by_segment(lm)[beta] == list(range(tried))
    assert all("response_format" not in entry["kwargs"] for entry in lm.history)
    assert not any(EXPLAIN_KEY in e["messages"][-1]["content"] for e in lm.history)


def test_a_score_the_parser_cannot_read_is_dropped_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SentimentModule()

    def score(*, text: str, config: dict[str, Any]) -> Any:
        rollout = config["rollout_id"]
        value = None if rollout == 0 else SCORES[text]
        return dspy.Prediction(score=value, rationale=f"sample {rollout}")

    monkeypatch.setattr(module, "score", score)
    with dspy.context(lm=DummyLM({EXPLAIN_KEY: EXPLAIN})):
        pred = module(text="\n\n".join(SEGMENTS[:2]), resample=2)

    assert [s["score"] for s in pred.segments] == [float(SCORES[s]) for s in SEGMENTS[:2]]
    assert (pred.n_samples, pred.n_resampled, pred.model_uncertainty) == (2, 0, None)
    assert [s["rationale"] for s in pred.segments] == ["sample 1", "sample 1"]


def test_one_readable_sample_is_enough() -> None:
    def pick(content: str, rollout: int | None) -> dict[str, str]:
        if EXPLAIN_KEY in content:
            return EXPLAIN
        segment = segment_of(content)
        return score_answer(segment, SCORES[segment] if rollout == 2 else "n/a")

    pred = run(RolloutLM(pick), resample=3)

    assert [s["score"] for s in pred.segments] == [float(SCORES[s]) for s in SEGMENTS]
    assert (pred.n_samples, pred.n_resampled, pred.model_uncertainty) == (4, 0, None)
    assert [s["rationale"] for s in pred.segments] == [
        f"rated {SCORES[s]}" for s in SEGMENTS
    ]


# --------------------------------------------------------------------------- dedupe


def test_identical_segments_are_scored_once_and_counted_at_each_position() -> None:
    same, other = SEGMENTS[0], SEGMENTS[1]
    lm = DummyLM(answers())

    pred = run(lm, text=f"{same}\n\n{other}\n\n{same}", resample=2)

    calls = Counter(segment_of(e["messages"][-1]["content"]) for e in score_calls(lm))
    assert calls == {same: 2, other: 2}
    assert [s["score"] for s in pred.segments] == [
        float(SCORES[same]), float(SCORES[other]), float(SCORES[same])
    ]
    assert (pred.n_segments, pred.n_samples, pred.n_resampled) == (3, 6, 3)
    assert pred.sentiment == pytest.approx((2 * SCORES[same] + SCORES[other]) / 3, abs=0.01)


# --------------------------------------------------------------------------- concurrency


GRACE_S = 0.25  # how long the first full wave waits for a 5th call that must not come


class BarrierLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
    """Holds every score call at a Barrier(4) and records how many calls are inside.

    A wave of 4 passes only when 4 score calls are in flight at once. While they
    wait they hold all 4 slots, so no 5th call can arrive before the barrier
    releases: `arrived_at_release` is exactly 4, 8, 12, ... The first wave gives a
    5th call `GRACE_S` to show up; with a bound it never does, so a passing run does
    not depend on timing, while a missing bound shows up as a 5th arrival.
    """

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        super().__init__(table)
        self.cond = threading.Condition()
        self.inside = 0
        self.peak = 0
        self.arrived = 0
        self.arrived_at_release: list[int] = []
        self.barrier = threading.Barrier(MAX_IN_FLIGHT, action=self._release, timeout=30)

    def _release(self) -> None:
        with self.cond:
            if not self.arrived_at_release:
                self.cond.wait_for(lambda: self.arrived > MAX_IN_FLIGHT, timeout=GRACE_S)
            self.arrived_at_release.append(self.arrived)

    def forward(
        self, prompt: str | None = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        is_score = "rollout_id" in kwargs
        with self.cond:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
            self.arrived += is_score
            self.cond.notify_all()
        try:
            if is_score:
                self.barrier.wait()
            return super().forward(prompt=prompt, messages=messages, **kwargs)
        finally:
            with self.cond:
                self.inside -= 1


def test_at_most_four_llm_calls_in_flight_across_parallel_forwards() -> None:
    # Four documents at once, as `run sentiment --jsonl` runs them: 4 x 4 segments x 2.
    documents = [
        [f"Report {doc} item {item} is fine." for item in range(4)] for doc in range(4)
    ]
    table = {EXPLAIN_KEY: EXPLAIN}
    table.update(
        {s: {"reasoning": "r", "score": "1", "rationale": "ok"} for d in documents for s in d}
    )
    lm = BarrierLM(table)
    module = SentimentModule()

    def one(segments: list[str]) -> Any:
        with dspy.context(lm=lm):
            return module(text="\n\n".join(segments), resample=2)

    with ThreadPoolExecutor(max_workers=len(documents)) as outer:
        futures = [
            outer.submit(contextvars.copy_context().run, one, segments)
            for segments in documents
        ]
        preds = [future.result(timeout=120) for future in futures]

    assert lm.arrived_at_release == list(range(4, 33, 4))
    assert lm.peak == MAX_IN_FLIGHT
    assert [p.n_samples for p in preds] == [8, 8, 8, 8]
    assert len(lm.history) == 32 + 4


class RecordingSlots:
    """Stands in for the module's in-flight semaphore; records which threads hold a slot."""

    def __init__(self) -> None:
        self.inner = threading.BoundedSemaphore(MAX_IN_FLIGHT)
        self.lock = threading.Lock()
        self.held: Counter[int] = Counter()

    def holds(self, thread: int) -> bool:
        with self.lock:
            return self.held[thread] > 0

    def __enter__(self) -> None:
        self.inner.acquire()
        with self.lock:
            self.held[threading.get_ident()] += 1

    def __exit__(self, *exc: object) -> None:
        with self.lock:
            self.held[threading.get_ident()] -= 1
        self.inner.release()


def test_every_llm_call_holds_a_slot_and_forward_holds_none_while_it_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = RecordingSlots()
    monkeypatch.setattr(sentiment_mod, "_IN_FLIGHT", slots)
    forward_thread = threading.get_ident()
    seen: list[tuple[bool, bool, bool]] = []  # (explain?, caller holds, forward holds)

    class SlotCheckingLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
        def forward(
            self, prompt: str | None = None, messages: Any = None, **kwargs: Any
        ) -> Any:
            seen.append((
                EXPLAIN_KEY in messages[-1]["content"],
                slots.holds(threading.get_ident()),
                slots.holds(forward_thread),
            ))
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    run(SlotCheckingLM(answers()), resample=2)

    assert sorted(seen) == [(False, True, False)] * 8 + [(True, True, True)]


def test_an_llm_error_propagates_and_cancels_the_pending_calls() -> None:
    class FailingLM(DummyLM):  # type: ignore[misc]  # dspy ships no stubs
        def __init__(self) -> None:
            super().__init__({})
            self.lock = threading.Lock()
            self.calls = 0

        def forward(self, prompt: str | None = None, messages: Any = None, **kwargs: Any) -> Any:
            with self.lock:
                self.calls += 1
            time.sleep(0.01)
            raise RuntimeError("router exploded")

    lm = FailingLM()
    text = "\n\n".join(f"Paragraph number {i} says little." for i in range(24))

    with pytest.raises(RuntimeError, match="router exploded"):
        run(lm, text=text, resample=1)

    assert lm.calls < 24  # the pending samples never ran
