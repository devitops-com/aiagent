"""``-v``/``-vv``/``-vvv`` reporting shared by ``run``/``eval``/``optimize``.

Three levels, each strictly additive, all written to **stderr** (stdout stays
clean for ``--json``/scripting):

- ``-v``   routing: resolved skill + composed model string, elapsed time, call count.
- ``-vv``  DSPy level: the adapter's rendered prompt (system + user messages) and
  parsed completion for every LM call made inside the scope, via DSPy's own
  ``pretty_print_history``.
- ``-vvv`` LLM/wire level: per-call usage/cost/raw response model, plus LiteLLM's
  own verbose HTTP logging (real request/response bytes as LiteLLM emits them).

Import-safe at module top (no ``dspy``), matching the lazy-dspy CLI invariant.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import typer

VERBOSE_OPTION = typer.Option(
    0,
    "-v",
    "--verbose",
    count=True,
    help=(
        "Increase verbosity (repeatable): -v routing summary, "
        "-vv adds the DSPy-level rendered prompt/completion, "
        "-vvv adds raw LLM usage/cost and wire-level HTTP logging."
    ),
)


def _enable_wire_logging(verbose: int) -> None:
    if verbose < 3:
        return
    import dspy

    # DSPy disables LiteLLM's debug output at import time
    # (litellm.suppress_debug_info = True) for clean logs; only DSPy's own
    # re-enable path flips that flag back, so litellm._turn_on_debug() alone
    # (called after dspy is imported) has no visible effect.
    dspy.enable_litellm_logging()


@contextmanager
def verbosity_scope(*, verbose: int, skill: str) -> Iterator[None]:
    """Report routing/-DSPy/-wire detail for the LM calls made in this block."""
    if verbose < 1:
        yield
        return

    import dspy

    lm = dspy.settings.lm
    model_string = lm.model if lm is not None else "unknown"
    typer.echo(f"[-v] skill={skill} model={model_string}", file=sys.stderr)

    _enable_wire_logging(verbose)
    start = len(lm.history) if lm is not None else 0
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0

    added = lm.history[start:] if lm is not None else []
    if verbose >= 2 and added:
        from dspy.utils.inspect_history import pretty_print_history

        pretty_print_history(lm.history, n=len(added), file=sys.stderr)
        if verbose >= 3:
            for entry in added:
                typer.echo(
                    f"[-vvv] response_model={entry['response_model']} "
                    f"usage={entry['usage']} cost={entry['cost']}",
                    file=sys.stderr,
                )

    typer.echo(f"[-v] elapsed={elapsed:.2f}s calls={len(added)}", file=sys.stderr)
