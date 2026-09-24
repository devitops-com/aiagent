"""Teacher labeling: k samples per state as votes, parse failures, concurrency, warm-up."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

import dspy
import litellm
import pytest
from dspy.utils import DummyLM
from dspy.utils.exceptions import AdapterParseError

from aiagent.config import Settings
from aiagent.distill import label as label_mod
from aiagent.distill.label import (
    DEFAULT_K,
    DEFAULT_TEMPERATURE,
    MAX_CONCURRENCY,
    WARM_UP_PROMPT,
    LabelConfig,
    Votes,
    label_states,
    warm_up_teacher,
)
from aiagent.distill.questions import OutputMapping, derive

KEYS = ("negative", "neutral", "mixed", "positive")
GREAT = "The food was great and the staff were friendly."
BROKEN = "It arrived broken and nobody answered my emails."
SO_SO = "Fast delivery, but the colour is not what the photo showed."


class Polarity(dspy.Signature):  # type: ignore[misc]  # dspy ships no stubs
    """Classify the overall sentiment polarity of a passage."""

    text: str = dspy.InputField(desc="A passage of text to assess.")
    polarity: Literal["negative", "neutral", "mixed", "positive"] = dspy.OutputField(
        desc="Overall sentiment polarity of the passage."
    )


MAPPINGS = derive(Polarity).mappings
TWO_QUESTIONS = (
    OutputMapping("polarity", "literal", KEYS, KEYS),
    OutputMapping("urgent", "bool", ("no", "yes"), (False, True)),
)


def parse_error() -> AdapterParseError:
    return AdapterParseError(adapter_name="ChatAdapter", signature=Polarity, lm_response="??")


class FakePredictor:
    """Records each call (thread-safely) and answers via `answer(call_index, kwargs)`."""

    def __init__(self, answer: Any, *, sleep: float = 0.0) -> None:
        self.answer = answer
        self.sleep = sleep
        self.lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[tuple[Any, Any]] = []
        self.active = 0
        self.peak = 0

    def __call__(self, **kwargs: Any) -> Any:
        with self.lock:
            self.calls.append(kwargs)
            self.contexts.append((dspy.settings.lm, dspy.settings.adapter))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.sleep)
            return self.answer(kwargs)
        finally:
            with self.lock:
                self.active -= 1


# --------------------------------------------------------------------------- LabelConfig


def test_label_config_defaults() -> None:
    config = LabelConfig()
    assert (config.k, config.temperature, config.concurrency) == (8, 0.7, 4)
    assert (DEFAULT_K, DEFAULT_TEMPERATURE, MAX_CONCURRENCY) == (8, 0.7, 4)
    assert LabelConfig(k=1, temperature=2.0, concurrency=1).k == 1
    assert LabelConfig(k=32, temperature=0.01, concurrency=4).k == 32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": 33},
        {"temperature": 0.0},
        {"temperature": -0.5},
        {"temperature": 2.01},
        {"temperature": math.nan},
        {"concurrency": 0},
        {"concurrency": 5},
    ],
)
def test_label_config_rejects_out_of_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LabelConfig(**kwargs)


# --------------------------------------------------------------------------- votes via DummyLM


def test_dummy_lm_votes_rollout_ids_and_temperature() -> None:
    lm = DummyLM({GREAT: {"polarity": "positive"}, BROKEN: {"polarity": "negative"}})
    config = LabelConfig(k=3, temperature=0.9, concurrency=2)

    votes = label_states(
        dspy.Predict(Polarity), [{"text": GREAT}, {"text": BROKEN}], MAPPINGS,
        lm=lm, config=config,
    )

    assert [v.counts for v in votes] == [
        {"polarity": {"negative": 0, "neutral": 0, "mixed": 0, "positive": 3}},
        {"polarity": {"negative": 3, "neutral": 0, "mixed": 0, "positive": 0}},
    ]
    assert list(votes[0].counts["polarity"]) == list(KEYS)  # every key, in key order
    assert all(v.parse_failures == {"polarity": 0} and v.k == 3 and v.labelable for v in votes)
    assert len(lm.history) == 6
    rollouts: dict[str, list[int]] = {GREAT: [], BROKEN: []}
    for entry in lm.history:
        text = GREAT if GREAT in entry["messages"][-1]["content"] else BROKEN
        rollouts[text].append(entry["kwargs"]["rollout_id"])
        assert entry["kwargs"]["temperature"] == 0.9
    assert {text: sorted(ids) for text, ids in rollouts.items()} == {
        GREAT: [0, 1, 2], BROKEN: [0, 1, 2]
    }


def test_no_json_fallback_one_call_per_sample() -> None:
    # Every answer is unparseable: with dspy's JSONAdapter fallback this would be 2k calls.
    lm = DummyLM({"a key no prompt contains": {"polarity": "mixed"}})

    [votes] = label_states(
        dspy.Predict(Polarity), [{"text": GREAT}], MAPPINGS,
        lm=lm, config=LabelConfig(k=3, concurrency=1),
    )

    assert len(lm.history) == 3
    assert votes.parse_failures == {"polarity": 3}
    assert not votes.labelable
    assert "response_format" not in lm.history[0]["kwargs"]


def test_progress_is_called_per_state_in_the_main_thread_and_results_keep_order() -> None:
    lm = DummyLM({
        GREAT: {"polarity": "positive"},
        BROKEN: {"polarity": "negative"},
        SO_SO: {"polarity": "mixed"},
    })
    seen: list[tuple[int, int, bool]] = []

    def progress(done: int, total: int) -> None:
        seen.append((done, total, threading.current_thread() is threading.main_thread()))

    votes = label_states(
        dspy.Predict(Polarity), [{"text": SO_SO}, {"text": GREAT}, {"text": BROKEN}], MAPPINGS,
        lm=lm, config=LabelConfig(k=2, concurrency=4), progress=progress,
    )

    assert seen == [(1, 3, True), (2, 3, True), (3, 3, True)]
    tops = [max(v.counts["polarity"], key=v.counts["polarity"].__getitem__) for v in votes]
    assert tops == ["mixed", "positive", "negative"]


def test_no_states_no_calls() -> None:
    predictor = FakePredictor(lambda kwargs: pytest.fail("called"))
    progress: list[tuple[int, int]] = []

    votes = label_states(
        predictor, [], MAPPINGS, lm=object(), config=LabelConfig(),
        progress=lambda done, total: progress.append((done, total)),
    )

    assert votes == [] and progress == [] and predictor.calls == []


# --------------------------------------------------------------------------- fake predictor


def test_at_most_concurrency_calls_in_flight_under_the_context() -> None:
    lm = object()
    predictor = FakePredictor(lambda kwargs: dspy.Prediction(polarity="mixed"), sleep=0.01)
    states = [{"text": f"text {i}"} for i in range(5)]

    votes = label_states(
        predictor, states, MAPPINGS, lm=lm, config=LabelConfig(k=4, temperature=0.5, concurrency=3)
    )

    assert 2 <= predictor.peak <= 3
    assert len(predictor.calls) == 20
    assert all(v.counts["polarity"]["mixed"] == 4 for v in votes)
    assert sorted(
        (call["text"], call["config"]["rollout_id"]) for call in predictor.calls
    ) == [(f"text {i}", j) for i in range(5) for j in range(4)]
    assert {call["config"]["temperature"] for call in predictor.calls} == {0.5}
    for active_lm, adapter in predictor.contexts:
        assert active_lm is lm
        assert isinstance(adapter, dspy.ChatAdapter)
        assert adapter.use_json_adapter_fallback is False


def test_parse_error_fails_every_question_and_bad_values_fail_one() -> None:
    def answer(kwargs: dict[str, Any]) -> Any:
        rollout = kwargs["config"]["rollout_id"]
        if rollout == 0:
            raise parse_error()
        if rollout == 1:
            return dspy.Prediction(polarity="banana", urgent=True)  # polarity unencodable
        if rollout == 2:
            return dspy.Prediction(polarity="mixed", urgent=1)  # 1 is not a bool
        return dspy.Prediction(polarity="mixed")  # urgent missing

    [votes] = label_states(
        FakePredictor(answer), [{"text": GREAT}], TWO_QUESTIONS,
        lm=object(), config=LabelConfig(k=4),
    )

    assert votes.parse_failures == {"polarity": 2, "urgent": 3}
    assert votes.counts == {
        "polarity": {"negative": 0, "neutral": 0, "mixed": 2, "positive": 0},
        "urgent": {"no": 0, "yes": 1},
    }
    assert votes.labelable


def test_only_failures_is_not_labelable() -> None:
    def answer(kwargs: dict[str, Any]) -> Any:
        raise parse_error()

    [votes] = label_states(
        FakePredictor(answer), [{"text": GREAT}], MAPPINGS, lm=object(), config=LabelConfig(k=2)
    )

    assert votes.counts == {"polarity": dict.fromkeys(KEYS, 0)}
    assert votes.parse_failures == {"polarity": 2}
    assert not votes.labelable


def test_labelable_needs_a_vote_for_every_question() -> None:
    counts = {"polarity": {"mixed": 1}, "urgent": {"no": 0, "yes": 0}}
    assert not Votes(counts=counts, parse_failures={"polarity": 0, "urgent": 1}, k=1).labelable
    counts = {"polarity": {"mixed": 1}, "urgent": {"no": 0, "yes": 1}}
    assert Votes(counts=counts, parse_failures={"polarity": 0, "urgent": 0}, k=1).labelable


def test_another_exception_propagates_and_cancels_the_rest() -> None:
    def answer(kwargs: dict[str, Any]) -> Any:
        raise RuntimeError("teacher exploded")

    predictor = FakePredictor(answer, sleep=0.005)
    states = [{"text": f"text {i}"} for i in range(50)]

    with pytest.raises(RuntimeError, match="teacher exploded"):
        label_states(predictor, states, MAPPINGS, lm=object(), config=LabelConfig(k=1, concurrency=1))

    assert len(predictor.calls) < len(states)


# --------------------------------------------------------------------------- warm-up


class RecordingLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[str]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return ["OK"]


def test_warm_up_teacher_sends_one_tiny_uncached_request(monkeypatch: pytest.MonkeyPatch) -> None:
    lm = RecordingLM()
    built: list[tuple[str, Settings]] = []

    def fake_build_exact_lm(model_string: str, *, settings: Settings) -> RecordingLM:
        built.append((model_string, settings))
        return lm

    monkeypatch.setattr(label_mod, "build_exact_lm", fake_build_exact_lm)
    settings = Settings()

    warm_up_teacher("openai/teacher-27B::nothink@32768", settings=settings)

    assert built == [("openai/teacher-27B::nothink@32768", settings)]
    assert lm.calls == [{
        "messages": [{"role": "user", "content": WARM_UP_PROMPT}],
        "max_tokens": 4,
        "cache": False,
    }]


def test_warm_up_teacher_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    lm = RecordingLM(error=ConnectionError("router down"))
    monkeypatch.setattr(label_mod, "build_exact_lm", lambda model_string, *, settings: lm)

    with pytest.raises(ConnectionError, match="router down"):
        warm_up_teacher("openai/teacher", settings=Settings())


Reply = tuple[int, dict[str, Any], dict[str, str]]  # status, JSON body, headers

OK: Reply = (
    200,
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "teacher",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    {},
)


def held(retry_after: str = "30", code: str = "gpu_held_by_job") -> Reply:
    """devai's router 503 while a training job holds the GPU (its gpu-arbiter's body)."""
    body = {"error": {"type": "server_error", "code": code, "message": "the GPU is held by laya-trainer"}}
    return 503, body, {"Retry-After": retry_after}


class Router:
    """A loopback OpenAI-compatible router: scripted replies, then OK."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests = 0
        router = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                router.requests += 1
                status, body, headers = router.replies.pop(0) if router.replies else OK
                data = json.dumps(body).encode()
                self.send_response(status)
                for name, value in {"Content-Type": "application/json", **headers}.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def settings(self) -> Settings:
        """The real RetryAwareLM against this router, without its own retries."""
        port = self.server.server_address[1]
        return Settings(api_base=f"http://127.0.0.1:{port}/v1", num_retries=0, request_timeout_s=10)


@pytest.fixture
def router() -> Iterator[Router]:
    router = Router()
    yield router
    router.server.shutdown()
    router.server.server_close()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_warm_up_teacher_waits_out_a_gpu_held_by_job_503(router: Router) -> None:
    router.replies = [held(), held(retry_after="12")]
    clock = FakeClock()

    warm_up_teacher("openai/teacher", settings=router.settings(), sleep=clock.sleep, clock=clock)

    assert router.requests == 3
    assert clock.sleeps == [30.0, 12.0]  # the router's Retry-After each time


def test_warm_up_teacher_gives_up_on_a_held_gpu_at_the_deadline(router: Router) -> None:
    router.replies = [held() for _ in range(100)]
    clock = FakeClock()

    with pytest.raises(litellm.ServiceUnavailableError):
        warm_up_teacher(
            "openai/teacher", settings=router.settings(), deadline_s=100, sleep=clock.sleep, clock=clock
        )

    assert clock.sleeps == [30.0, 30.0, 30.0]  # a 4th wait would pass the deadline


def test_warm_up_teacher_does_not_wait_on_another_503(router: Router) -> None:
    router.replies = [held(code="engine_down")]
    clock = FakeClock()

    with pytest.raises(litellm.ServiceUnavailableError):
        warm_up_teacher("openai/teacher", settings=router.settings(), sleep=clock.sleep, clock=clock)

    assert (router.requests, clock.sleeps) == (1, [])
