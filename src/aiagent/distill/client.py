"""devai's laya trainer: the OpenAI fine-tuning jobs subset and its volume mirror.

``POST /fine_tuning/jobs`` starts a job, ``GET /fine_tuning/jobs/{id}`` and
``…/{id}/events`` follow it. devai also writes the job object to
``<distill_dir>/runs/<job>/job.json`` and its events to ``events.jsonl``, which the
campaign reads first. The client never uses the environment's proxy
(``trust_env=False``): the trainer lives on devai-net and is reached directly, never
through pipelock. ``httpx`` is imported at the top, so this module is only ever
imported lazily (from the campaign).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from aiagent.exceptions import DistillError, TrainerAPIError
from aiagent.system1.contract import safe_id

TIMEOUT_S: Final = 30.0
# Creating a job makes devai's router drain and evict the teacher and start the
# trainer inside the POST (drain alone may take 30 s), so it gets longer.
CREATE_TIMEOUT_S: Final = 180.0
TERMINAL: Final = frozenset({"succeeded", "failed", "cancelled"})

_JOBS: Final = "/fine_tuning/jobs"
_UNREACHABLE_HINT: Final = "(is devai's laya-trainer backend configured?)"


class FineTuningJob(BaseModel):
    """The fields aiagent reads from an OpenAI ``fine_tuning.job`` object."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    status: str
    model: str | None = None
    training_file: str | None = None
    fine_tuned_model: str | None = None
    error: dict[str, Any] | None = None
    created_at: int | None = None
    finished_at: int | None = None

    @property
    def terminal(self) -> bool:
        """True once the job has succeeded, failed or been cancelled."""
        return self.status in TERMINAL

    @property
    def error_message(self) -> str | None:
        """The job's ``error.message``, or None if there is none."""
        message = (self.error or {}).get("message")
        return message if isinstance(message, str) and message else None


@dataclass(frozen=True)
class JobRequest:
    """The POST /fine_tuning/jobs body."""

    model: str
    training_file: str
    hyperparameters: Mapping[str, float | int]
    suffix: str
    metadata: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        """The request body, in contract key order."""
        return {
            "model": self.model,
            "training_file": self.training_file,
            "hyperparameters": dict(self.hyperparameters),
            "suffix": self.suffix,
            "metadata": dict(self.metadata),
        }


class TrainerClient:
    """OpenAI fine-tuning jobs subset over httpx (trust_env=False, short timeout)."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base, timeout=timeout_s, trust_env=False, transport=transport
        )

    def create_job(self, request: JobRequest) -> FineTuningJob:
        """Start a fine-tuning job."""
        body = request.to_json()
        return _job(self._request("POST", _JOBS, body=body, timeout=CREATE_TIMEOUT_S))

    def get_job(self, job_id: str) -> FineTuningJob:
        """The job's current state."""
        return _job(self._request("GET", f"{_JOBS}/{_job_id(job_id)}"))

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        """The job's events, as the API lists them."""
        body = self._request("GET", f"{_JOBS}/{_job_id(job_id)}/events")
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
            raise TrainerAPIError(
                f"trainer API returned an invalid event list: {body!r}"
            )
        return data

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: object = None,
        timeout: float | None = None,
    ) -> Any:
        """Send one request; return the parsed JSON of a 2xx, else TrainerAPIError."""
        try:
            response = self._client.request(
                method,
                path,
                json=body,
                timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
            )
        except httpx.RequestError as exc:
            raise TrainerAPIError(
                f"cannot reach the trainer API at {self._base}: {exc} "
                + _UNREACHABLE_HINT
            ) from exc
        if not response.is_success:
            raise TrainerAPIError(_status_message(response))
        try:
            return response.json()
        except ValueError as exc:
            raise TrainerAPIError(
                f"trainer API returned invalid JSON for {method} {path}: {exc}"
            ) from exc


def _job_id(job_id: str) -> str:
    try:
        return safe_id(job_id, "job id")
    except ValueError as exc:
        raise TrainerAPIError(str(exc)) from exc


def _job(body: object) -> FineTuningJob:
    try:
        return FineTuningJob.model_validate(body)
    except ValidationError as exc:
        raise TrainerAPIError(
            f"trainer API returned an invalid job object: {exc}"
        ) from exc


def _status_message(response: httpx.Response) -> str:
    """'trainer API <status>: <error.message or body text>' plus a hint for 409/503."""
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    detail = error.get("message") if isinstance(error, dict) else None
    if not isinstance(detail, str) or not detail:
        detail = response.text.strip() or response.reason_phrase
    message = f"trainer API {response.status_code}: {detail}"
    retry_after = response.headers.get("Retry-After")
    if response.status_code == 409:
        message += " (a training job is already running)"
    elif response.status_code == 503 and retry_after:
        message += f" (GPU busy; retry after {retry_after} s)"
    return message


def _run_file(distill_dir: Path, job_id: str, name: str) -> Path:
    try:
        return distill_dir / "runs" / safe_id(job_id, "job id") / name
    except ValueError as exc:
        raise DistillError(str(exc)) from exc


def read_job_from_volume(distill_dir: Path, job_id: str) -> FineTuningJob | None:
    """runs/<job>/job.json, or None if absent; DistillError if unparseable."""
    path = _run_file(distill_dir, job_id, "job.json")
    try:
        return FineTuningJob.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise DistillError(f"cannot read {path}: {exc}") from exc


def read_events_from_volume(
    distill_dir: Path, job_id: str, *, last: int
) -> list[dict[str, Any]]:
    """The last `last` parseable lines of runs/<job>/events.jsonl."""
    path = _run_file(distill_dir, job_id, "events.jsonl")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise DistillError(f"cannot read {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a half-written or corrupt line
        if isinstance(event, dict):
            events.append(event)
    return events[-last:] if last > 0 else []
