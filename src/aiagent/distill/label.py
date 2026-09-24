"""Teacher labels for a distillation dataset: k sampled answers per state, as votes.

Each (state, sample j) is one call of the skill's own predictor at temperature T with
``rollout_id=j``: distinct rollout ids bypass the DSPy cache per sample, and each
sample is still cached, so an interrupted run resumes for free. At most
:data:`MAX_CONCURRENCY` calls are in flight (the devai teacher's batch size).

The adapter is DSPy's ChatAdapter **without** its JSONAdapter fallback, so one sample
is exactly one LM call in one decoding mode, and server JSON mode (which devai
backends strip or ignore) is never requested. An unparseable answer is a parse
failure; any other error aborts the run (``RetryAwareLM`` already retried the
transient ones). dspy is imported at the top, so this module is only ever imported
lazily.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Final

import dspy
from dspy.utils.exceptions import AdapterParseError

from aiagent.config import Settings
from aiagent.distill.questions import OutputMapping
from aiagent.llm.lm import build_exact_lm

MAX_CONCURRENCY: Final = 4  # the devai teacher runs --max-num-seqs 4
DEFAULT_K: Final = 8
DEFAULT_TEMPERATURE: Final = 0.7
WARM_UP_PROMPT: Final = "Reply with OK."

_MAX_K: Final = 32
_MAX_TEMPERATURE: Final = 2.0
_WARM_UP_MAX_TOKENS: Final = 4
# devai caches a "GPU held by a job" verdict for up to 30 s (its Retry-After), then
# the teacher's cold start takes about 2 min: 5 min covers both.
WARM_UP_DEADLINE_S: Final = 300.0
_HELD_CODE: Final = "gpu_held_by_job"
_HELD_RETRY_S: Final = 30.0
_HELD_MAX_WAIT_S: Final = 60.0

Sample = dict[str, str | None]  # qid -> answer key, None where it did not parse


@dataclass(frozen=True)
class LabelConfig:
    """k samples at temperature T with at most `concurrency` teacher calls in flight.

    T must be > 0: at T=0 the k samples would all be the same answer.
    """

    k: int = DEFAULT_K
    temperature: float = DEFAULT_TEMPERATURE
    concurrency: int = MAX_CONCURRENCY

    def __post_init__(self) -> None:
        """ValueError unless 1<=k<=32, 0<temperature<=2, 1<=concurrency<=4."""
        if not 1 <= self.k <= _MAX_K:
            raise ValueError(f"k must be 1-{_MAX_K}, got {self.k}")
        if not 0 < self.temperature <= _MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be in (0, {_MAX_TEMPERATURE}], "
                f"got {self.temperature}"
            )
        if not 1 <= self.concurrency <= MAX_CONCURRENCY:
            raise ValueError(
                f"concurrency must be 1-{MAX_CONCURRENCY}, got {self.concurrency}"
            )


@dataclass(frozen=True)
class Votes:
    """The teacher's votes for one state."""

    counts: Mapping[str, Mapping[str, int]]  # qid -> key -> n (every key, key order)
    parse_failures: Mapping[str, int]  # qid -> n
    k: int

    @property
    def labelable(self) -> bool:
        """Every question has at least one vote."""
        return all(sum(keys.values()) > 0 for keys in self.counts.values())


def _sample(
    predictor: Callable[..., Any],
    state: Mapping[str, str],
    rollout_id: int,
    *,
    lm: Any,
    temperature: float,
    mappings: Sequence[OutputMapping],
) -> Sample:
    """One teacher call: each question's answer key, None where it did not parse."""
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    config = {"rollout_id": rollout_id, "temperature": temperature}
    with dspy.context(lm=lm, adapter=adapter):
        try:
            pred = predictor(**state, config=config)
        except AdapterParseError:
            return dict.fromkeys((m.field for m in mappings), None)
    return {m.field: m.encode(getattr(pred, m.field, None)) for m in mappings}


def _tally(
    counts: dict[str, dict[str, int]], failures: dict[str, int], sample: Sample
) -> None:
    for qid, key in sample.items():
        if key is None:
            failures[qid] += 1
        else:
            counts[qid][key] += 1


def label_states(
    predictor: Callable[..., Any],
    states: Sequence[Mapping[str, str]],
    mappings: Sequence[OutputMapping],
    *,
    lm: Any,
    config: LabelConfig,
    progress: Callable[[int, int], None] | None = None,
) -> list[Votes]:
    """k samples per state from the teacher, same order as `states`.

    ``progress(done_states, total_states)`` runs in the calling thread as each state
    completes. Any exception other than a parse failure cancels the pending samples
    and propagates.
    """
    counts = [{m.field: dict.fromkeys(m.keys, 0) for m in mappings} for _ in states]
    failures = [dict.fromkeys((m.field for m in mappings), 0) for _ in states]
    pending = [config.k] * len(states)
    done = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures: dict[Future[Sample], int] = {
            pool.submit(
                _sample,
                predictor,
                state,
                j,
                lm=lm,
                temperature=config.temperature,
                mappings=mappings,
            ): i
            for i, state in enumerate(states)
            for j in range(config.k)
        }
        try:
            for future in as_completed(futures):
                i = futures[future]
                _tally(counts[i], failures[i], future.result())
                pending[i] -= 1
                if pending[i] == 0:
                    done += 1
                    if progress is not None:
                        progress(done, len(states))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [
        Votes(counts=c, parse_failures=f, k=config.k)
        for c, f in zip(counts, failures, strict=True)
    ]


def _held_retry_after(exc: BaseException) -> float | None:
    """Seconds to wait if ``exc`` is devai's 503 for a GPU a job holds, else None.

    litellm's ServiceUnavailableError keeps only the error message: the router's
    error code is in the exceptions it was raised from, and its headers are in
    ``litellm_response_headers`` (``response`` is a synthetic, empty one).
    """
    if getattr(exc, "status_code", None) != 503:
        return None
    chain: list[BaseException] = []
    link: BaseException | None = exc
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__ or link.__context__
    if not any(_HELD_CODE in str(part) for part in chain):
        return None
    headers = (
        getattr(exc, "litellm_response_headers", None)
        or getattr(getattr(exc, "response", None), "headers", None)
        or {}
    )
    try:
        seconds = float(headers.get("retry-after", _HELD_RETRY_S))
    except (TypeError, ValueError):
        seconds = _HELD_RETRY_S
    return min(max(seconds, 1.0), _HELD_MAX_WAIT_S)


def warm_up_teacher(
    model_string: str,
    *,
    settings: Settings,
    deadline_s: float = WARM_UP_DEADLINE_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """One tiny uncached request, so devai restores the teacher with exactly this model.

    Right after a job ends devai may still answer 503 ``gpu_held_by_job``; that is
    waited out (``Retry-After``) until ``deadline_s``. Other errors propagate; the
    campaign turns them into a warning.
    """
    lm = build_exact_lm(model_string, settings=settings)
    end = clock() + deadline_s
    while True:
        try:
            lm(
                messages=[{"role": "user", "content": WARM_UP_PROMPT}],
                max_tokens=_WARM_UP_MAX_TOKENS,
                cache=False,
            )
            return
        except Exception as exc:
            wait = _held_retry_after(exc)
            if wait is None or clock() + wait > end:
                raise
            sleep(wait)
