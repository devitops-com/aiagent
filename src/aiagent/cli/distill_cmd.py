"""``aiagent distill`` — distill a skill's typed predictor into a laya System 1 student.

``plan`` -> ``label`` -> ``train`` (``status``) -> ``eval`` -> ``install``, or
``repair`` and train again. The campaign (``aiagent.distill.campaign``) does the
work; these handlers parse options, call it, print text or ``--json`` and map exit
codes. Every ``aiagent.distill`` import happens inside a handler, so
``import aiagent.cli.app`` stays free of dspy, numpy, tokenizers, onnxruntime and
httpx.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, echo_err, get_settings, print_json
from aiagent.config import Settings
from aiagent.exceptions import DistillError

if TYPE_CHECKING:
    from aiagent.distill.campaign import JobView, LabelOutcome, PredictorPlan
    from aiagent.distill.gates import EvalReport, QuestionReport
    from aiagent.system1.artifacts import InstalledArtifact

# Mirror aiagent.distill.label's defaults; kept local so this module stays import-light.
_DEFAULT_K = 8
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_CONCURRENCY = 4
# Mirrors aiagent.distill.dataset.DEFAULT_BASE; kept local so this module stays
# import-light.
_DEFAULT_BASE = "laya-multilingual@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
# Mirror aiagent.distill.gates.GateTargets(); kept local so this module stays
# import-light.
_DEFAULT_TARGET_PRECISION = 0.95
_DEFAULT_FIT_PRECISION = 0.98
_DEFAULT_ALPHA = 0.05
_DEFAULT_MIN_COVERAGE = 0.20
_DEFAULT_EPSILON = 0.01
_DEFAULT_MAX_ROUNDS = 3
# Mirror aiagent.distill.campaign's DEFAULT_HYPERPARAMETERS, DEFAULT_REPAIR_ROWS and
# the events status shows; kept local so this module stays import-light.
_DEFAULT_EPOCHS = 4
_DEFAULT_LR_MULTIPLIER = 1.0
_DEFAULT_REPAIR_ROWS = 256
_DEFAULT_EVENTS = 5

_PROGRESS_EVERY = 25  # labeled states between progress lines
_SHORT_HASH = 12
# Held-out rows a student needs accepted to certify the default 0.95 at α=0.05 when it
# is really 98% / 97% right (k = ⌊p·n⌋); the minimum itself comes from gates.
_ACCEPTED_AT_98 = 181
_ACCEPTED_AT_97 = 361
_CORPUS_SEGMENTS = "2,000-3,000"

distill_app = typer.Typer(
    name="distill",
    help=(
        "Distill a skill's typed predictor into a laya System 1 student "
        "(label -> train -> eval -> install)."
    ),
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)

_JSON_OPTION = typer.Option(False, "--json", help="Emit JSON.")
_CONCURRENCY_OPTION = typer.Option(
    _DEFAULT_CONCURRENCY, "--concurrency", min=1, max=4, help="Teacher calls in flight."
)


def _line(label: str, value: object) -> None:
    typer.echo(f"{label:<10}: {value}")


def _progress(done: int, total: int) -> None:
    if done % _PROGRESS_EVERY == 0 or done == total:
        echo_err(f"labeled {done}/{total}")


# --------------------------------------------------------------------------- plan


def _predictor_json(p: PredictorPlan) -> dict[str, Any]:
    from aiagent.distill.questions import DERIVE_VERSION

    return {
        "name": p.name,
        "qualifies": p.derivation.qualifies,
        "reasons": list(p.derivation.reasons),
        "derive_version": DERIVE_VERSION,
        "input_field": p.derivation.input_field,
        "has_reasoning": p.derivation.has_reasoning,
        "signature_sha256": p.derivation.signature_sha256,
        "question_set_sha256": p.derivation.question_set_sha256,
        "skill_source_sha256": p.binds.skill_source_sha256 if p.binds else None,
        "questions": p.derivation.questions(),
    }


def _emit_predictor(data: dict[str, Any]) -> None:
    _line("predictor", data["name"])
    typer.echo(f"  qualifies      : {'yes' if data['qualifies'] else 'no'}")
    for reason in data["reasons"]:
        typer.echo(f"    - {reason}")
    typer.echo(f"  derive_version : {data['derive_version']}")
    typer.echo(f"  input          : {data['input_field'] or '(none)'}")
    for label, key in (
        ("signature", "signature_sha256"),
        ("question set", "question_set_sha256"),
        ("skill source", "skill_source_sha256"),
    ):
        digest = data[key]
        typer.echo(f"  {label:<15}: {digest[:_SHORT_HASH] if digest else '-'}")
    for qid, question in data["questions"].items():
        keys = ", ".join(str(c) for c in question["criteria"])
        typer.echo(f"  question {qid} {question['type']} [{keys}]")
        typer.echo(f"    {question['instructions']}")


@distill_app.command("plan")
def plan_cmd(
    skill: str = typer.Argument(..., help="Skill name."),
    predictor: str | None = typer.Option(
        None, "--predictor", help="Only this predictor (its dotted name)."
    ),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Show which predictors of SKILL can be distilled, with questions and hashes."""
    from aiagent.distill import campaign
    from aiagent.distill.gates import GateTargets, min_certifiable

    plan = campaign.plan(skill, settings=get_settings())
    predictors = [p for p in plan.predictors if predictor in (None, p.name)]
    if not predictors:
        names = ", ".join(p.name for p in plan.predictors) or "none"
        raise DistillError(
            f"{skill} has no predictor {predictor!r} (predictors: {names})"
        )
    targets = GateTargets()
    needed = min_certifiable(targets.precision, targets.alpha)
    items = [_predictor_json(p) for p in predictors]
    data = {
        "skill": plan.skill.name,
        "source": str(plan.skill.source),
        "predictors": items,
        "heldout_sizing": {
            "precision": targets.precision,
            "alpha": targets.alpha,
            "min_accepted": needed,
            "accepted_at_98": _ACCEPTED_AT_98,
            "accepted_at_97": _ACCEPTED_AT_97,
            "segments": _CORPUS_SEGMENTS,
        },
    }
    if as_json:
        print_json(data)
    else:
        _line("skill", f"{plan.skill.name} ({plan.skill.source})")
        for item in items:
            _emit_predictor(item)
        _line(
            "held-out",
            f"certifying {targets.precision} at α={targets.alpha} needs ≥ {needed} "
            f"accepted held-out rows, all correct (≈{_ACCEPTED_AT_98} if the student "
            f"is 98% right, ≈{_ACCEPTED_AT_97} at 97%): label about "
            f"{_CORPUS_SEGMENTS} segments",
        )
    if not any(p.derivation.qualifies for p in predictors):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- label


def _emit_label(outcome: LabelOutcome, as_json: bool) -> None:
    stats = outcome.stats
    next_step = f"aiagent distill train {outcome.dataset.id}"
    if as_json:
        print_json(
            {
                "dataset": outcome.dataset.id,
                "path": str(outcome.dataset.path),
                "documents": outcome.documents,
                "segments": outcome.segments,
                "duplicates": outcome.duplicates,
                "counts": dict(outcome.counts),
                "label_histogram": stats.label_histogram,
                "parse_failures": stats.parse_failures,
                "unlabeled": stats.unlabeled,
                "warnings": list(outcome.warnings),
                "next": next_step,
            }
        )
        return
    _line("dataset", f"{outcome.dataset.id}  ({outcome.dataset.path})")
    _line(
        "documents",
        f"{outcome.documents} ({outcome.segments} segments, "
        f"{outcome.duplicates} duplicates)",
    )
    _line("rows", ", ".join(f"{split} {n}" for split, n in outcome.counts.items()))
    for qid, histogram in stats.label_histogram.items():
        counts = ", ".join(f"{key} {n}" for key, n in histogram.items())
        _line("labels", f"{qid}: {counts}")
    failures = ", ".join(f"{qid} {n}" for qid, n in stats.parse_failures.items())
    _line("failures", f"{failures or 'none'} (unparseable teacher samples)")
    _line("unlabeled", f"{stats.unlabeled} (no sample parsed; dropped)")
    _line("next", next_step)


@distill_app.command("label")
def label_cmd(
    skill: str = typer.Argument(..., help="Skill name."),
    predictor: str | None = typer.Option(
        None, "--predictor", help="Predictor to distill (default: the qualifying one)."
    ),
    file: list[Path] = typer.Option(
        [], "--file", "-f", help="Local file: .txt/.md/.html/.pdf (repeatable)."
    ),
    directory: list[Path] = typer.Option(
        [], "--dir", "-d", help="Directory of such files, recursive (repeatable)."
    ),
    url: list[str] = typer.Option([], "--url", "-u", help="URL to fetch (repeatable)."),
    jsonl: list[Path] = typer.Option(
        [],
        "--jsonl",
        help="JSONL file, one document per line in the input field (repeatable).",
    ),
    teacher: str | None = typer.Option(
        None, "--teacher", help="Teacher model alias or name (default: configured)."
    ),
    k: int = typer.Option(_DEFAULT_K, "--k", help="Teacher samples per segment."),
    temperature: float = typer.Option(
        _DEFAULT_TEMPERATURE, "--temperature", help="Teacher sampling temperature."
    ),
    concurrency: int = _CONCURRENCY_OPTION,
    base: str = typer.Option(
        _DEFAULT_BASE, "--base", help="Base checkpoint NAME@REVISION under base/."
    ),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Label documents with the teacher into a dataset in <distill_dir>/inbox/."""
    from aiagent.distill import campaign
    from aiagent.distill.label import LabelConfig
    from aiagent.distill.segment import Sources

    try:
        config = LabelConfig(k=k, temperature=temperature, concurrency=concurrency)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    sources = Sources(
        files=tuple(file), dirs=tuple(directory), urls=tuple(url), jsonl=tuple(jsonl)
    )
    outcome = campaign.label(
        skill,
        predictor=predictor,
        sources=sources,
        teacher=teacher,
        config=config,
        base=base,
        settings=get_settings(),
        progress=_progress,
    )
    for warning in outcome.warnings:
        echo_err(f"warning: {warning}")
    _emit_label(outcome, as_json)


# ----------------------------------------------------------------------- train, status


def _when(timestamp: object) -> str:
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        return "-"
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_job(view: JobView, as_json: bool, extra: dict[str, Any]) -> None:
    job = view.job
    if as_json:
        data = {**job.model_dump(mode="json"), "source": view.source}
        print_json({**data, "events": list(view.events), **extra})
        return
    _line("job", job.id)
    _line("status", job.status)
    _line("source", view.source)
    _line("model", job.model or "-")
    _line("dataset", job.training_file or "-")
    _line("student", job.fine_tuned_model or "-")
    _line("error", job.error_message or "-")
    if view.events:
        typer.echo(f"{'events':<10}:")
    for event in view.events:
        when = _when(event.get("created_at"))
        typer.echo(f"  {when} {event.get('level', '-')} {event.get('message', '')}")
    for label, value in extra.items():
        _line(label, value)


def _follow(
    job_id: str, *, warm_up: bool, events: int, as_json: bool, settings: Settings
) -> None:
    """--wait: poll until the job ends, warm the teacher up, print; fail unless it
    succeeded."""
    from aiagent.distill import campaign

    def on_poll(view: JobView) -> None:
        echo_err(f"{view.job.id}: {view.job.status} ({view.source})")

    view = campaign.wait(job_id, settings=settings, on_poll=on_poll, events=events)
    extra: dict[str, Any] = {}
    if warm_up:
        model = campaign.warm_up(view.job, settings=settings)
        extra["warm-up"] = model or "skipped (see the warning above)"
    succeeded = view.job.status == "succeeded"
    if succeeded:
        extra["next"] = f"aiagent distill eval {view.job.id}"
    _emit_job(view, as_json, extra)
    if not succeeded:
        reason = view.job.error_message or "no error message"
        raise DistillError(f"job {view.job.id} {view.job.status}: {reason}")


@distill_app.command("train")
def train_cmd(
    dataset: str = typer.Argument(..., help="Dataset id (ds-...) to train on."),
    epochs: int = typer.Option(_DEFAULT_EPOCHS, "--epochs", min=1),
    batch_size: int | None = typer.Option(
        None, "--batch-size", min=1, help="Default: the trainer's own choice."
    ),
    lr_multiplier: float = typer.Option(_DEFAULT_LR_MULTIPLIER, "--lr-multiplier"),
    wait: bool = typer.Option(False, "--wait", help="Follow the job until it ends."),
    no_warm_up: bool = typer.Option(
        False, "--no-warm-up", help="With --wait: do not warm the teacher up after."
    ),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Re-validate DATASET and start a devai fine-tuning job on it."""
    from aiagent.distill import campaign

    settings = get_settings()
    hyperparameters: dict[str, float | int] = {
        "n_epochs": epochs,
        "learning_rate_multiplier": lr_multiplier,
    }
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    job = campaign.train(dataset, hyperparameters=hyperparameters, settings=settings)
    if wait:
        _follow(
            job.id,
            warm_up=not no_warm_up,
            events=_DEFAULT_EVENTS,
            as_json=as_json,
            settings=settings,
        )
        return
    if as_json:
        print_json(job.model_dump(mode="json"))
        return
    _line("job", job.id)
    _line("status", job.status)
    _line("dataset", job.training_file or dataset)
    _line("next", f"aiagent distill status {job.id} --wait")


@distill_app.command("status")
def status_cmd(
    job: str = typer.Argument(..., help="Fine-tuning job id."),
    events: int = typer.Option(
        _DEFAULT_EVENTS, "--events", min=0, help="Last events to show."
    ),
    wait: bool = typer.Option(False, "--wait", help="Follow the job until it ends."),
    no_warm_up: bool = typer.Option(
        False, "--no-warm-up", help="With --wait: do not warm the teacher up after."
    ),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Show a fine-tuning job, from the distill volume if there, else the API."""
    from aiagent.distill import campaign

    settings = get_settings()
    if wait:
        _follow(
            job,
            warm_up=not no_warm_up,
            events=events,
            as_json=as_json,
            settings=settings,
        )
        return
    _emit_job(campaign.status(job, settings=settings, events=events), as_json, {})


# --------------------------------------------------------------------------- eval


def _emit_question(q: QuestionReport, fit_precision: float) -> None:
    typer.echo(f"question {q.qid}")
    _line("  n", f"calib {q.n_calib}, held-out {q.n_heldout}")
    _line("  accuracy", f"{q.accuracy:.4f}  ECE {q.ece:.4f}  Brier {q.brier:.4f}")
    coverage = ", ".join(f"@{p} {c:.2f}" for p, c in q.coverage_at_precision.items())
    _line("  coverage", coverage)
    tau = "none" if q.tau is None else f"{q.tau:.4f}"
    _line("  τ", f"{tau} (fitted on calib at precision {fit_precision})")
    precision = "-" if q.precision is None else f"{q.precision:.4f}"
    _line(
        "  accepted",
        f"{q.accepted} ({q.correct} correct): precision {precision}, "
        f"CP-lower {q.precision_lower:.4f}, coverage {q.coverage:.2f}",
    )
    _line("  pass", "yes" if q.passes else "no")


def _emit_report(report: EvalReport, path: Path, next_step: str | None) -> None:
    t = report.targets
    for question in report.questions:
        _emit_question(question, t.fit_precision)
    _line(
        "targets",
        f"precision {t.precision} (τ fitted at {t.fit_precision}), α={t.alpha}, "
        f"min coverage {t.min_coverage}, ε={t.epsilon}, max rounds {t.max_rounds}",
    )
    _line("round", report.round)
    if report.source_changed_since_labeling:
        _line("note", "the skill's files changed since labeling (soft bind)")
    _line("verdict", f"{report.verdict} — {report.reason}")
    _line("report", path)
    _line("next", next_step or "none: the campaign stops here")


@distill_app.command("eval")
def eval_cmd(
    run: str = typer.Argument(..., help="Run (fine-tuning job) id under runs/."),
    target_precision: float = typer.Option(
        _DEFAULT_TARGET_PRECISION,
        "--target-precision",
        help="Precision to certify on held-out (Clopper-Pearson lower bound).",
    ),
    fit_precision: float = typer.Option(
        _DEFAULT_FIT_PRECISION,
        "--fit-precision",
        help="Precision τ is fitted at on calib (>= the target).",
    ),
    alpha: float = typer.Option(_DEFAULT_ALPHA, "--alpha", help="CP significance."),
    min_coverage: float = typer.Option(
        _DEFAULT_MIN_COVERAGE,
        "--min-coverage",
        min=0.0,
        max=1.0,
        help="Least share of held-out the student must answer.",
    ),
    epsilon: float = typer.Option(
        _DEFAULT_EPSILON, "--epsilon", min=0.0, help="Least accuracy gain per round."
    ),
    max_rounds: int = typer.Option(
        _DEFAULT_MAX_ROUNDS, "--max-rounds", min=1, help="Rounds before stopping."
    ),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Verify RUN's student, score it, and decide: ship (0), repair (3) or stop (4)."""
    from aiagent.distill import campaign
    from aiagent.distill.gates import EXIT_CODES, GateTargets

    try:
        targets = GateTargets(
            precision=target_precision,
            fit_precision=fit_precision,
            alpha=alpha,
            min_coverage=min_coverage,
            epsilon=epsilon,
            max_rounds=max_rounds,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    outcome = campaign.evaluate(run, targets=targets, settings=get_settings())
    report = outcome.report
    next_step = {
        "ship": f"aiagent distill install {run}",
        "repair": f"aiagent distill repair {run}",
    }.get(report.verdict)
    if as_json:
        data = report.to_json()
        print_json({**data, "report_path": str(outcome.path), "next": next_step})
    else:
        _emit_report(report, outcome.path, next_step)
    raise typer.Exit(EXIT_CODES[report.verdict])


# --------------------------------------------------------------------- repair, install


@distill_app.command("repair")
def repair_cmd(
    run: str = typer.Argument(..., help="Run id whose eval verdict is repair."),
    max_rows: int = typer.Option(
        _DEFAULT_REPAIR_ROWS, "--max-rows", min=1, help="Pool rows to label."
    ),
    concurrency: int = _CONCURRENCY_OPTION,
    as_json: bool = _JSON_OPTION,
) -> None:
    """Label the pool rows RUN's student is least sure of into the next round."""
    from aiagent.distill import campaign

    outcome = campaign.repair(
        run,
        max_rows=max_rows,
        concurrency=concurrency,
        settings=get_settings(),
        progress=_progress,
    )
    next_step = f"aiagent distill train {outcome.dataset.id}"
    if as_json:
        print_json(
            {
                "dataset": outcome.dataset.id,
                "path": str(outcome.dataset.path),
                "labeled": outcome.labeled,
                "unlabeled": outcome.unlabeled,
                "pool_remaining": outcome.pool_remaining,
                "next": next_step,
            }
        )
        return
    _line("dataset", f"{outcome.dataset.id}  ({outcome.dataset.path})")
    _line("labeled", f"{outcome.labeled} ({outcome.unlabeled} unlabeled)")
    _line("pool left", outcome.pool_remaining)
    _line("next", next_step)


def _enable_hints(installed: InstalledArtifact) -> tuple[str, str]:
    mode = json.dumps({installed.skill: "shadow"}, separators=(",", ":"))
    env = f"AIAGENT_SYSTEM1_MODE='{mode}'"
    toml = f'[system1_mode]\n{installed.skill} = "shadow"'
    return env, toml


@distill_app.command("install")
def install_cmd(
    run: str = typer.Argument(..., help="Run id whose eval verdict is ship."),
    as_json: bool = _JSON_OPTION,
) -> None:
    """Install RUN's student for its skill (needs a ship verdict from eval)."""
    from aiagent.distill import campaign

    installed = campaign.install(run, settings=get_settings())
    env, toml = _enable_hints(installed)
    if as_json:
        print_json(
            {
                "path": str(installed.path),
                "artifact_id": installed.artifact_id,
                "skill": installed.skill,
                "predictor": installed.predictor,
                "run": installed.run,
                "thresholds": dict(installed.thresholds),
                "target_precision": installed.target_precision,
                "installed_at": installed.installed_at,
                "enable": {"env": env, "toml": toml},
            }
        )
        return
    _line("installed", installed.path)
    _line("artifact", installed.artifact_id)
    _line("skill", f"{installed.skill}/{installed.predictor}")
    taus = ", ".join(f"{q} τ={t:.4f}" for q, t in installed.thresholds.items())
    _line("thresholds", f"{taus} (certified precision {installed.target_precision})")
    _line("enable", env)
    typer.echo("            or in config.toml:")
    for line in toml.splitlines():
        typer.echo(f"              {line}")
    typer.echo('            then "gate" once the shadow log (shadow.jsonl) agrees')
