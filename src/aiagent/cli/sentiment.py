"""``aiagent sentiment`` — score data sources on a -10..+10 sentiment scale.

Accepts any mix of raw ``--text``, local ``--file`` (.txt/.md/.html/.pdf), and
``--url`` sources; URLs are fetched through the configured proxy. Reports the
sentiment plus volatility, model uncertainty, statistical significance, and a
plain-language explanation, human-readable by default or as ``--json``.

Module-top imports stay ``dspy``-free (dspy is pulled in lazily by
``configure_lm``/``build_module``), preserving the fast-``--help`` invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from aiagent.cli._common import get_settings, print_json
from aiagent.cli._runtime import configure_lm
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.ingest.sources import SourceDoc, fetch_source, read_file
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry

# Mirrors aiagent.core.sentiment defaults; kept local so this module imports no
# dspy (importing core.sentiment would). Values are passed through to the module.
_DEFAULT_RESAMPLE = 3
_DEFAULT_MAX_SEGMENTS = 24


def sentiment(
    text: list[str] = typer.Option(
        [], "--text", "-t", help="Raw text to analyze (repeatable)."
    ),
    file: list[Path] = typer.Option(
        [], "--file", "-f", help="Local file: .txt/.md/.html/.pdf (repeatable)."
    ),
    url: list[str] = typer.Option(
        [], "--url", "-u", help="URL to fetch and analyze (repeatable)."
    ),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    resample: int = typer.Option(
        _DEFAULT_RESAMPLE, "--resample", help="LM samples per segment (uncertainty)."
    ),
    max_segments: int = typer.Option(
        _DEFAULT_MAX_SEGMENTS, "--max-segments", help="Cap on analyzed segments."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Analyze sentiment of the given sources. Requires a reachable router."""
    settings = get_settings()
    docs = _ingest(text, file, url, settings)

    registry, _ = load_registry(settings)
    target = registry.get("sentiment")
    configure_lm(settings, model)
    module = build_module(target)

    combined = "\n\n".join(doc.text for doc in docs)
    with verbosity_scope(verbose=verbose, skill="sentiment"):
        prediction = module(
            text=combined, resample=resample, max_segments=max_segments
        )

    _emit(prediction, docs, as_json)


def _ingest(
    texts: list[str], files: list[Path], urls: list[str], settings: Any
) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    for raw in texts:
        cleaned = raw.strip()
        if cleaned:
            docs.append(SourceDoc(origin="text", text=cleaned))
    docs.extend(read_file(path) for path in files)
    docs.extend(fetch_source(link, settings=settings) for link in urls)
    if not docs:
        raise AiagentError("provide at least one --text, --file, or --url")
    return docs


def _emit(prediction: Any, docs: list[SourceDoc], as_json: bool) -> None:
    origins = [doc.origin for doc in docs]
    if as_json:
        print_json(
            {
                "sentiment": prediction.sentiment,
                "polarity": prediction.polarity,
                "volatility": prediction.volatility,
                "model_uncertainty": prediction.model_uncertainty,
                "std_error": prediction.std_error,
                "t_statistic": prediction.t_statistic,
                "significance_p": prediction.significance_p,
                "confidence": prediction.confidence,
                "ci95": prediction.ci95,
                "n_segments": prediction.n_segments,
                "n_samples": prediction.n_samples,
                "segments": prediction.segments,
                "sources": origins,
                "explanation": prediction.explanation,
            }
        )
        return

    typer.echo(f"sentiment    : {prediction.sentiment:+.2f}  ({prediction.polarity})")
    typer.echo(
        f"volatility   : {prediction.volatility:.2f}  "
        f"(across {prediction.n_segments} segments)"
    )
    typer.echo(
        f"uncertainty  : {prediction.model_uncertainty:.2f}  "
        f"(model spread over {prediction.n_samples} samples)"
    )
    if prediction.significance_p is not None:
        t_value = prediction.t_statistic
        tstat = "n/a" if t_value is None else f"{t_value:+.2f}"
        typer.echo(
            f"significance : p={prediction.significance_p:.4f} "
            f"({prediction.confidence}); t={tstat}"
        )
    else:
        typer.echo(f"significance : {prediction.confidence}")
    if prediction.ci95 is not None:
        typer.echo(
            f"95% CI       : [{prediction.ci95[0]:+.2f}, {prediction.ci95[1]:+.2f}] "
            "(normal approx)"
        )
    typer.echo(f"sources      : {', '.join(origins)}")
    typer.echo("")
    typer.echo(prediction.explanation)
