"""``aiagent sentiment`` — score data sources on a -10..+10 sentiment scale.

Accepts any mix of raw ``--text``, local ``--file`` (.txt/.md/.html/.pdf), and
``--url`` sources; URLs are fetched through the configured proxy. Reports the
sentiment plus volatility, model uncertainty (with ``--resample`` 2 or more),
statistical significance, and a plain-language explanation, human-readable by
default or as ``--json``.

Module-top imports stay ``dspy``-free (dspy is pulled in lazily by
``configure_lm``/``build_module``), preserving the fast-``--help`` invariant. System 1
(``system1_mode.sentiment`` other than ``off``) is imported only when it is on, since
it loads numpy and onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from aiagent.cli._common import examples, get_settings, print_json
from aiagent.cli._runtime import configure_lm
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.ingest.sources import SourceDoc, fetch_source, read_file
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry

# Mirrors aiagent.core.sentiment defaults; kept local so this module imports no
# dspy (importing core.sentiment would). Values are passed through to the module.
_DEFAULT_RESAMPLE = 1
_DEFAULT_MAX_SEGMENTS = 24

SENTIMENT_EXAMPLES = examples(
    (
        "Score one text",
        'aiagent sentiment --text "The rollout was flawless and the team is thrilled."',
    ),
    (
        "A file and a web page together (fetched through the proxy), as JSON",
        "aiagent sentiment -f report.pdf -u https://example.com/article --json",
    ),
    (
        "Three samples per segment: also measures the model's uncertainty",
        "aiagent sentiment --url https://example.com/article --resample 3",
    ),
    (
        "A long document in at most 12 segments (merged, nothing dropped)",
        "aiagent sentiment --file book.txt --max-segments 12",
    ),
    (
        "System 1 shadow: polarity's student judges each segment, only logged",
        "AIAGENT_SYSTEM1_MODE='{\"sentiment\":\"shadow\"}' "
        "aiagent sentiment -f report.pdf",
    ),
)


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
        _DEFAULT_RESAMPLE,
        "--resample",
        help="LLM samples per segment; 2 or more measure model uncertainty.",
    ),
    max_segments: int = typer.Option(
        _DEFAULT_MAX_SEGMENTS,
        "--max-segments",
        help="Cap on analyzed segments; more are merged, never dropped.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Analyze sentiment of the given sources. Requires a reachable router.

    Joins the sources into one text, splits it into segments and scores each on
    -10 (very negative) to +10 (very positive), then reports the mean with its
    volatility, significance and a plain-language explanation. Give at least one
    --text, --file or --url; URLs are fetched through proxy_url.
    system1_mode.sentiment is off by default. In shadow, polarity's installed
    student judges each segment and its verdict is only logged: run it on the
    target corpus first. Use gate only for document corpora that passed the
    pass test on such a run (the pinned calibration passed on Wikipedia
    articles): a segment the student calls neutral at confidence 0.96 or more
    gets no LLM call and scores the pinned calibration's level. That
    calibration is for polarity student a866e0a4 on the teacher
    Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink; with another student, model or
    ScoreSegment, gate runs as shadow, with a warning.
    """
    settings = get_settings()
    docs = _ingest(text, file, url, settings)

    registry, _ = load_registry(settings)
    target = registry.get("sentiment")
    configure_lm(settings, model)
    module = build_module(target)
    if settings.system1_mode.get(target.name, "off") != "off":
        from aiagent.system1.cascade import apply_system1  # lazy: numpy, ORT

        apply_system1(module, target, settings, registry)

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
                "n_resampled": prediction.n_resampled,
                "segments": prediction.segments,
                "system1": prediction.system1,
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
    if prediction.model_uncertainty is None:
        typer.echo("uncertainty  : n/a (no segment has two readable scores)")
    else:
        typer.echo(
            f"uncertainty  : {prediction.model_uncertainty:.2f}  "
            f"(model spread over {prediction.n_resampled} resampled segments)"
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
    if prediction.system1 is not None:
        line = _system1_line(prediction.system1, prediction.n_segments)
        typer.echo(f"system 1     : {line}")
    typer.echo(f"sources      : {', '.join(origins)}")
    typer.echo("")
    typer.echo(prediction.explanation)


def _system1_line(block: dict[str, Any], n_segments: int) -> str:
    """E.g. '18/24 segments by the student (gate, polarity a866e0a4), 1 too long'."""
    taken = "by" if block["mode"] == "gate" else "would be by"
    skill = block["student"].split("/")[0]
    line = (
        f"{block['accepted']}/{n_segments} segments {taken} the student "
        f"({block['mode']}, {skill} {block['artifact_id'][:8]})"
    )
    if block["too_long"]:
        line += f", {block['too_long']} too long"
    return line
