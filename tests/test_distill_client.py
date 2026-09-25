"""Tests for the fine-tuning jobs client and the volume readers (httpx MockTransport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from aiagent.distill import client as client_mod
from aiagent.distill.client import (
    TERMINAL,
    TIMEOUT_S,
    FineTuningJob,
    JobRequest,
    TrainerClient,
    read_events_from_volume,
    read_job_from_volume,
)
from aiagent.exceptions import DistillError, TrainerAPIError

BASE = "http://trainer.test:11438/v1"

JOB: dict[str, Any] = {
    "object": "fine_tuning.job",
    "id": "ftjob-abc123",
    "status": "queued",
    "model": "laya-multilingual",
    "training_file": "ds-9f2c41d07a1b",
    "fine_tuned_model": None,
    "error": None,
    "created_at": 1790000000,
    "finished_at": None,
    "hyperparameters": {"n_epochs": 4},
}

REQUEST = JobRequest(
    model="laya-multilingual",
    training_file="ds-9f2c41d07a1b",
    hyperparameters={"n_epochs": 4, "batch_size": 8, "learning_rate_multiplier": 1.0},
    suffix="polarity-classify",
    metadata={"campaign": "c-20260924T191500Z", "round": "0"},
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler, calls: list[httpx.Request] | None = None) -> TrainerClient:
    def record(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return handler(request)

    # A trailing slash on the base is dropped.
    return TrainerClient(BASE + "/", transport=httpx.MockTransport(record))


def _respond(status: int, body: object = None, **kwargs: Any) -> Handler:
    return lambda request: httpx.Response(status, json=body, **kwargs)


def _refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected request {request.method} {request.url}")


# --- models ------------------------------------------------------------------


def test_job_request_json_matches_the_contract() -> None:
    assert json.dumps(REQUEST.to_json()) == json.dumps(
        {
            "model": "laya-multilingual",
            "training_file": "ds-9f2c41d07a1b",
            "hyperparameters": {"n_epochs": 4, "batch_size": 8, "learning_rate_multiplier": 1.0},
            "suffix": "polarity-classify",
            "metadata": {"campaign": "c-20260924T191500Z", "round": "0"},
        }
    )


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        ("validating_files", False),
        ("queued", False),
        ("running", False),
        ("succeeded", True),
        ("failed", True),
        ("cancelled", True),
    ],
)
def test_job_terminal(status: str, terminal: bool) -> None:
    assert FineTuningJob(id="j", status=status).terminal is terminal
    assert (status in TERMINAL) is terminal


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (None, None),
        ({}, None),
        ({"message": None, "code": None}, None),
        ({"message": "CUDA out of memory", "code": "oom"}, "CUDA out of memory"),
    ],
)
def test_job_error_message(error: dict[str, Any] | None, message: str | None) -> None:
    assert FineTuningJob(id="j", status="failed", error=error).error_message == message


def test_job_ignores_unknown_fields() -> None:
    job = FineTuningJob.model_validate(JOB)
    assert (job.id, job.status, job.model, job.created_at) == (
        "ftjob-abc123",
        "queued",
        "laya-multilingual",
        1790000000,
    )


# --- TrainerClient: requests -------------------------------------------------


def test_create_job_posts_the_exact_body_and_path() -> None:
    calls: list[httpx.Request] = []
    job = _client(_respond(200, JOB), calls).create_job(REQUEST)

    assert [(r.method, str(r.url)) for r in calls] == [("POST", f"{BASE}/fine_tuning/jobs")]
    assert calls[0].headers["content-type"] == "application/json"
    assert list(json.loads(calls[0].content).items()) == list(REQUEST.to_json().items())
    assert job == FineTuningJob.model_validate(JOB)
    assert job.terminal is False


def test_get_job() -> None:
    calls: list[httpx.Request] = []
    done = {**JOB, "status": "succeeded", "fine_tuned_model": "ft:laya:polarity", "finished_at": 1790000900}
    job = _client(_respond(200, done), calls).get_job("ftjob-abc123")

    assert [(r.method, str(r.url)) for r in calls] == [("GET", f"{BASE}/fine_tuning/jobs/ftjob-abc123")]
    assert (job.status, job.terminal, job.fine_tuned_model) == ("succeeded", True, "ft:laya:polarity")
    assert job.error_message is None


def test_list_events() -> None:
    calls: list[httpx.Request] = []
    events = [
        {"object": "fine_tuning.job.event", "created_at": 1, "level": "info", "message": "epoch 1/4"},
        {"object": "fine_tuning.job.event", "created_at": 2, "level": "info", "message": "epoch 2/4"},
    ]
    got = _client(_respond(200, {"object": "list", "data": events}), calls).list_events("ftjob-abc123")

    assert [(r.method, str(r.url)) for r in calls] == [
        ("GET", f"{BASE}/fine_tuning/jobs/ftjob-abc123/events")
    ]
    assert got == events


def test_client_bypasses_the_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://devai-pipelock:8888")
    client = TrainerClient(BASE)
    assert client._client.trust_env is False
    assert client._client.timeout == httpx.Timeout(TIMEOUT_S)
    assert str(client._client.base_url) == BASE + "/"


# --- TrainerClient: errors ---------------------------------------------------


def _error_body(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": None}}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            httpx.Response(409, json=_error_body("job ftjob-1 is running")),
            "trainer API 409: job ftjob-1 is running (a training job is already running)",
        ),
        (
            httpx.Response(503, json=_error_body("gpu held by vllm"), headers={"Retry-After": "120"}),
            "trainer API 503: gpu held by vllm (GPU busy; retry after 120 s)",
        ),
        (
            httpx.Response(503, json=_error_body("gpu held by vllm")),
            "trainer API 503: gpu held by vllm",
        ),
        (httpx.Response(404, text="404 page not found\n"), "trainer API 404: 404 page not found"),
        (httpx.Response(400, json={"detail": "bad"}), 'trainer API 400: {"detail":"bad"}'),
        (httpx.Response(500), "trainer API 500: Internal Server Error"),
    ],
)
def test_error_status_messages(response: httpx.Response, expected: str) -> None:
    with pytest.raises(TrainerAPIError) as info:
        _client(lambda request: response).create_job(REQUEST)
    assert str(info.value) == expected


@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("[Errno 111] Connection refused"), httpx.ReadTimeout("timed out")]
)
def test_unreachable_names_the_base_url(exc: httpx.HTTPError) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(TrainerAPIError) as info:
        _client(fail).get_job("ftjob-abc123")
    message = str(info.value)
    assert message.startswith(f"cannot reach the trainer API at {BASE}: ")
    assert str(exc) in message
    assert message.endswith("(is devai's laya-trainer backend configured?)")


def test_invalid_json_is_an_error() -> None:
    with pytest.raises(TrainerAPIError, match="invalid JSON"):
        _client(lambda request: httpx.Response(200, text="<html>ok</html>")).get_job("j")


def test_invalid_job_object_is_an_error() -> None:
    with pytest.raises(TrainerAPIError, match="invalid job object"):
        _client(_respond(200, {"id": "j"})).get_job("j")
    with pytest.raises(TrainerAPIError, match="invalid job object"):
        _client(_respond(200, ["not", "a", "job"])).create_job(REQUEST)


@pytest.mark.parametrize("body", [{"object": "list"}, {"data": "x"}, {"data": [1]}, []])
def test_invalid_event_list_is_an_error(body: object) -> None:
    with pytest.raises(TrainerAPIError, match="invalid event list"):
        _client(_respond(200, body)).list_events("j")


@pytest.mark.parametrize("job_id", ["../x", "a/b", "", ".x"])
def test_unsafe_job_id_rejected_before_any_request(job_id: str) -> None:
    client = _client(_refuse)
    with pytest.raises(TrainerAPIError, match="unsafe job id"):
        client.get_job(job_id)
    with pytest.raises(TrainerAPIError, match="unsafe job id"):
        client.list_events(job_id)


# --- the volume ----------------------------------------------------------------


def _run_dir(distill_dir: Path, job_id: str = "ftjob-abc123") -> Path:
    run = distill_dir / "runs" / job_id
    run.mkdir(parents=True)
    return run


def test_read_job_from_volume_absent(tmp_path: Path) -> None:
    assert read_job_from_volume(tmp_path, "ftjob-abc123") is None
    _run_dir(tmp_path)  # the run dir alone, no job.json yet
    assert read_job_from_volume(tmp_path, "ftjob-abc123") is None


def test_read_job_from_volume_ok(tmp_path: Path) -> None:
    (_run_dir(tmp_path) / "job.json").write_text(json.dumps(JOB), encoding="utf-8")
    assert read_job_from_volume(tmp_path, "ftjob-abc123") == FineTuningJob.model_validate(JOB)


@pytest.mark.parametrize("text", ["{not json", '{"id": "ftjob-abc123"}', "[]", "\xff"])
def test_read_job_from_volume_malformed(tmp_path: Path, text: str) -> None:
    (_run_dir(tmp_path) / "job.json").write_bytes(text.encode("latin-1"))
    with pytest.raises(DistillError, match="job.json"):
        read_job_from_volume(tmp_path, "ftjob-abc123")


def test_volume_readers_wrap_read_errors(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    (run / "job.json").mkdir()
    (run / "events.jsonl").mkdir()
    with pytest.raises(DistillError, match="cannot read .*job.json"):
        read_job_from_volume(tmp_path, "ftjob-abc123")
    with pytest.raises(DistillError, match="cannot read .*events.jsonl"):
        read_events_from_volume(tmp_path, "ftjob-abc123", last=5)


def test_volume_readers_reject_an_unsafe_job_id(tmp_path: Path) -> None:
    with pytest.raises(DistillError, match="unsafe job id"):
        read_job_from_volume(tmp_path, "../x")
    with pytest.raises(DistillError, match="unsafe job id"):
        read_events_from_volume(tmp_path, "a/b", last=5)


def test_read_events_from_volume_last_n_skipping_bad_lines(tmp_path: Path) -> None:
    lines = [json.dumps({"created_at": i, "level": "info", "message": f"step {i}"}) for i in range(6)]
    lines[4] = '{"created_at": 4, "level": "info", "mess'  # a half-written line
    lines.insert(2, "[1, 2]")  # parseable, but not an event object
    lines.append("")
    (_run_dir(tmp_path) / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")

    got = read_events_from_volume(tmp_path, "ftjob-abc123", last=3)
    assert [e["message"] for e in got] == ["step 2", "step 3", "step 5"]
    all_events = read_events_from_volume(tmp_path, "ftjob-abc123", last=100)
    assert [e["created_at"] for e in all_events] == [0, 1, 2, 3, 5]
    assert read_events_from_volume(tmp_path, "ftjob-abc123", last=0) == []


def test_read_events_from_volume_absent(tmp_path: Path) -> None:
    assert read_events_from_volume(tmp_path, "ftjob-abc123", last=5) == []


def test_creating_a_job_waits_for_the_router_swap() -> None:
    # The router evicts the teacher and starts the trainer inside this POST.
    calls: list[httpx.Request] = []
    job_body = {"id": "ftjob-1", "status": "queued", "model": "laya-multilingual"}
    client = _client(_respond(200, job_body), calls)

    client.create_job(REQUEST)
    client.get_job("ftjob-1")

    assert calls[0].extensions["timeout"]["read"] == client_mod.CREATE_TIMEOUT_S
    assert client_mod.CREATE_TIMEOUT_S >= 180
    assert calls[1].extensions["timeout"]["read"] == client_mod.TIMEOUT_S
