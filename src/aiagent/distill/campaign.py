"""A distillation campaign: plan, label, train, follow, evaluate, repair and install.

Orchestration only: every step composes the ``distill``/``system1`` modules and returns
data, and the CLI (``aiagent distill``) prints it. The flow per skill predictor:

- ``label`` segments documents by the student's token budget, splits them by document
  hash and has the teacher (the skill's own predictor on the LLM) label train, calib and
  held-out, then writes a dataset to ``<distill_dir>/inbox/``;
- ``train`` re-validates it and starts a devai fine-tuning job; ``status``/``wait``
  follow the job, ``warm_up`` restores the teacher afterwards;
- ``evaluate`` verifies the run's artifact, scores calib and held-out on onnxruntime and
  decides ship / repair / stop; ``repair`` labels the pool rows the student is least
  sure of into the next round's dataset; ``install`` copies a shipped student into
  ``artifacts_dir``.

:func:`teacher_lm` and :func:`trainer_client` are the only LLM and HTTP seams. dspy,
httpx and numpy (the gates and the runtime) are imported inside the functions.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from aiagent import __version__
from aiagent.config import Settings
from aiagent.distill.dataset import (
    BaseCheckpoint,
    Dataset,
    DatasetRef,
    GoldAnswer,
    LabelStats,
    Producer,
    ProducerBinds,
    Row,
    SourceInfo,
    TeacherInfo,
    TeacherSpec,
    build_row,
    compute_label_stats,
    derive_repair_dataset,
    gold_from_votes,
    load_base_checkpoint,
    load_dataset,
    validate_dataset,
    write_dataset,
)
from aiagent.distill.questions import DERIVE_VERSION, Derivation, derive
from aiagent.distill.segment import (
    Sources,
    collect_documents,
    question_room,
    segment_text,
    state_fits,
)
from aiagent.distill.splits import SPLITS, Split, assign_split, document_id
from aiagent.exceptions import DistillError
from aiagent.ingest.sources import SourceDoc
from aiagent.skills.base import Skill
from aiagent.system1.contract import Binds, dataset_id, safe_id
from aiagent.system1.sequence import SequenceTokenizer, prepare

if TYPE_CHECKING:
    from aiagent.distill.client import FineTuningJob, TrainerClient
    from aiagent.distill.gates import EvalItem, EvalReport, GateTargets
    from aiagent.distill.label import LabelConfig
    from aiagent.system1.artifacts import (
        ArtifactManifest,
        InstalledArtifact,
        VerifiedArtifact,
    )

logger = logging.getLogger(__name__)

POLL_S: Final = 30.0
# No batch_size: the trainer picks it, since it knows the GPU's memory.
DEFAULT_HYPERPARAMETERS: Final[Mapping[str, float | int]] = {
    "n_epochs": 4,
    "learning_rate_multiplier": 1.0,
}
DEFAULT_REPAIR_ROWS: Final = 256
HELDOUT_WARN_ROWS: Final = 300  # below this, a held-out gate at 0.95 rarely certifies

_SUFFIX_MAX: Final = 40  # devai's suffix rule: ^[a-z0-9][a-z0-9-]{0,39}$
_SUFFIX_FALLBACK: Final = "distill"
_DEFAULT_EVENTS: Final = 5


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class PredictorPlan:
    """One named predictor of a skill's module and what distilling it would bind."""

    name: str
    derivation: Derivation
    binds: Binds | None  # only when it qualifies
    predictor: Any  # the dspy predictor itself: the teacher labels through it


@dataclass(frozen=True)
class Plan:
    """A skill's predictors and whether each can be distilled."""

    skill: Skill
    predictors: tuple[PredictorPlan, ...]

    def select(self, name: str | None) -> PredictorPlan:
        """The named one, or the only qualifying one; DistillError otherwise."""
        skill = self.skill.name
        if name is not None:
            named = [p for p in self.predictors if p.name == name]
            if not named:
                names = ", ".join(p.name for p in self.predictors) or "none"
                raise DistillError(
                    f"{skill} has no predictor {name!r} (predictors: {names})"
                )
            chosen = named[0]
            if not chosen.derivation.qualifies:
                reasons = "; ".join(chosen.derivation.reasons)
                raise DistillError(f"{skill}/{name} cannot be distilled: {reasons}")
            return chosen
        qualifying = [p for p in self.predictors if p.derivation.qualifies]
        if len(qualifying) == 1:
            return qualifying[0]
        if qualifying:
            names = ", ".join(p.name for p in qualifying)
            raise DistillError(
                f"several predictors of {skill} qualify ({names}); "
                "pick one with --predictor"
            )
        reasons = "; ".join(
            f"{p.name}: {', '.join(p.derivation.reasons)}" for p in self.predictors
        )
        raise DistillError(
            f"no predictor of {skill} can be distilled ({reasons or 'it has none'})"
        )


def _binds(chosen: PredictorPlan) -> Binds:
    if chosen.binds is None:  # select() only returns qualifying predictors
        raise DistillError(f"{chosen.name} cannot be distilled")
    return chosen.binds


def plan(skill_name: str, *, settings: Settings) -> Plan:
    """build_module + derive for every named predictor (no LLM)."""
    from aiagent.skills.loader import build_module
    from aiagent.skills.registry import load_registry

    registry, _ = load_registry(settings)
    skill = registry.get(skill_name)
    module = build_module(skill)
    predictors = []
    for name, predictor in module.named_predictors():
        derivation = derive(predictor.signature)
        binds = derivation.binds(skill) if derivation.qualifies else None
        predictors.append(PredictorPlan(name, derivation, binds, predictor))
    return Plan(skill, tuple(predictors))


def teacher_lm(
    settings: Settings, *, alias: str | None = None, exact: str | None = None
) -> Any:
    """build_exact_lm(exact) if exact else build_lm(alias): the only LM seam."""
    from aiagent.llm.lm import build_exact_lm, build_lm

    if exact:
        return build_exact_lm(exact, settings=settings)
    return build_lm(alias, settings=settings)


def trainer_client(settings: Settings) -> TrainerClient:
    """TrainerClient(settings.trainer_api_base): the only HTTP seam."""
    from aiagent.distill.client import TrainerClient

    return TrainerClient(settings.trainer_api_base)


# --------------------------------------------------------------------------- label


@dataclass(frozen=True)
class LabelOutcome:
    """A written dataset and how it was made."""

    dataset: DatasetRef
    counts: Mapping[str, int]
    stats: LabelStats
    documents: int
    segments: int
    duplicates: int
    warnings: tuple[str, ...]  # e.g. a held-out too small to certify


@dataclass(frozen=True)
class _Segment:
    group_id: str
    index: int
    text: str
    split: Split


@dataclass(frozen=True)
class _RowBuilder:
    """Rows for one predictor's questions, tokenized for one base checkpoint."""

    field: str
    questions: dict[str, dict[str, Any]]
    tokenizer: SequenceTokenizer
    max_len: int
    head_max_len: int

    def fits(self) -> Callable[[str], bool]:
        """text -> whether the student sees it whole for every question."""
        room = question_room(
            self.tokenizer,
            prepare(self.questions),
            max_len=self.max_len,
            head_max_len=self.head_max_len,
        )
        return state_fits(self.tokenizer, input_field=self.field, room=room)

    def build(
        self,
        seg: _Segment,
        gold: Mapping[str, GoldAnswer] | None = None,
        teacher: TeacherInfo | None = None,
    ) -> Row:
        """The segment's row, labeled if gold is given."""
        return build_row(
            group_id=seg.group_id,
            index=seg.index,
            text=seg.text,
            input_field=self.field,
            questions=self.questions,
            split=seg.split,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            gold=gold,
            teacher=teacher,
        )


_Answers = tuple[dict[str, GoldAnswer], TeacherInfo]


def _teacher_answers(
    chosen: PredictorPlan,
    states: Sequence[Mapping[str, str]],
    *,
    lm: Any,
    config: LabelConfig,
    progress: Callable[[int, int], None] | None,
) -> list[_Answers | None]:
    """Per state, the gold answers and teacher info; None where no sample parsed."""
    from aiagent.distill.label import label_states

    mappings = chosen.derivation.mappings
    votes = label_states(
        chosen.predictor, states, mappings, lm=lm, config=config, progress=progress
    )
    answers: list[_Answers | None] = []
    for vote in votes:
        if not vote.labelable:
            answers.append(None)
            continue
        gold = {
            m.field: gold_from_votes(m.keys, vote.counts[m.field]) for m in mappings
        }
        info = TeacherInfo(k=vote.k, parse_failures=dict(vote.parse_failures))
        answers.append((gold, info))
    return answers


def _answered[T](
    items: Sequence[T], answers: Sequence[_Answers | None]
) -> list[tuple[T, _Answers]]:
    """The items the teacher answered, paired with their answers."""
    return [
        (item, answer)
        for item, answer in zip(items, answers, strict=True)
        if answer is not None
    ]


def _heldout_warning(rows: int) -> str:
    from aiagent.distill.gates import GateTargets, min_certifiable

    targets = GateTargets()
    needed = min_certifiable(targets.precision, targets.alpha)
    return (
        f"held-out has {rows} rows; certifying {targets.precision} at "
        f"α={targets.alpha} needs ≥ {needed} accepted rows, all correct (≈181 at a "
        "true precision of 98%); label a larger corpus (≈2,000-3,000 segments)"
    )


def _segments(
    docs: Sequence[SourceDoc], fits: Callable[[str], bool]
) -> tuple[list[_Segment], list[SourceInfo], int]:
    """Every document's segments, the first copy of each text only; and the sources."""
    segments: list[_Segment] = []
    sources: list[SourceInfo] = []
    seen: set[str] = set()
    duplicates = 0
    for doc in docs:
        group_id = document_id(doc.text)
        split = assign_split(group_id)
        kept = 0
        for index, text in enumerate(segment_text(doc.text, fits=fits)):
            if text in seen:
                duplicates += 1
                continue
            seen.add(text)
            segments.append(_Segment(group_id, index, text, split))
            kept += 1
        sources.append(SourceInfo(doc_id=group_id, origin=doc.origin, segments=kept))
    return segments, sources, duplicates


def label(
    skill_name: str,
    *,
    predictor: str | None,
    sources: Sources,
    teacher: str | None,
    config: LabelConfig,
    base: str,
    settings: Settings,
    progress: Callable[[int, int], None] | None = None,
) -> LabelOutcome:
    """Segment, split and teacher-label the sources into a new dataset in inbox/."""
    import dspy

    skill_plan = plan(skill_name, settings=settings)
    chosen = skill_plan.select(predictor)
    checkpoint = load_base_checkpoint(settings.distill_dir, base)
    builder = _RowBuilder(
        field=chosen.derivation.input_field or "",
        questions=chosen.derivation.questions(),
        tokenizer=checkpoint.tokenizer(),
        max_len=checkpoint.max_len,
        head_max_len=checkpoint.head_max_len,
    )
    docs = collect_documents(sources, input_field=builder.field, settings=settings)
    segments, source_infos, duplicates = _segments(docs, builder.fits())
    to_label = [s for s in segments if s.split != "pool"]  # pool rows stay unlabeled
    lm = teacher_lm(settings, alias=teacher)
    answers = _teacher_answers(
        chosen,
        [{builder.field: s.text} for s in to_label],
        lm=lm,
        config=config,
        progress=progress,
    )
    rows = [builder.build(seg, *answer) for seg, answer in _answered(to_label, answers)]
    unlabeled = len(to_label) - len(rows)
    rows.extend(builder.build(s) for s in segments if s.split == "pool")
    created_at = _utc_now()
    producer = Producer(
        version=__version__,
        created_at=created_at,
        skill=skill_plan.skill.name,
        predictor=chosen.name,
        input_field=builder.field,
        derive_version=DERIVE_VERSION,
        binds=ProducerBinds(**_binds(chosen).to_json()),
        questions=builder.questions,
        teacher=TeacherSpec(
            model=str(lm.model),
            k=config.k,
            temperature=config.temperature,
            dspy=dspy.__version__,
        ),
        campaign="c-" + created_at.replace("-", "").replace(":", ""),
        round=0,
        parent_dataset=None,
        sources=source_infos,
        label_stats=compute_label_stats(rows, unlabeled=unlabeled),
    )
    inbox = settings.distill_dir / "inbox"
    ref = write_dataset(inbox, base=checkpoint, producer=producer, rows=rows)
    counts: dict[str, int] = {s: sum(r.split == s for r in rows) for s in SPLITS}
    warnings: tuple[str, ...] = ()
    if counts["heldout"] < HELDOUT_WARN_ROWS:
        warnings = (_heldout_warning(counts["heldout"]),)
    return LabelOutcome(
        dataset=ref,
        counts=counts,
        stats=producer.label_stats,
        documents=len(docs),
        segments=len(segments),
        duplicates=duplicates,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- train


def _base_of(dataset: Dataset, settings: Settings) -> BaseCheckpoint:
    """The staged base the dataset was tokenized for, unchanged since."""
    ref = dataset.manifest.base_checkpoint
    base = f"{ref.name}@{ref.revision}"
    checkpoint = load_base_checkpoint(settings.distill_dir, base)
    staged = (checkpoint.weights_sha256, checkpoint.tokenizer_sha256)
    if staged != (ref.weights_sha256, ref.tokenizer_sha256):
        raise DistillError(
            f"base checkpoint changed on the volume: {checkpoint.path} no longer "
            f"matches the one {dataset.ref.id} was tokenized for"
        )
    return checkpoint


def job_suffix(skill: str, predictor: str) -> str:
    """'<skill>-<predictor>' in devai's suffix grammar: lowercase, each run of other
    characters one '-' (a ChainOfThought's predictor is '<attr>.predict'), <= 40."""
    suffix = re.sub(r"[^a-z0-9]+", "-", f"{skill}-{predictor}".lower()).strip("-")
    return suffix[:_SUFFIX_MAX] or _SUFFIX_FALLBACK


def train(
    dataset_id: str, *, hyperparameters: Mapping[str, float | int], settings: Settings
) -> FineTuningJob:
    """Re-validate the dataset against its base, then start a fine-tuning job."""
    from aiagent.distill.client import JobRequest

    dataset = load_dataset(settings.distill_dir, dataset_id)
    validate_dataset(
        dataset.ref.path, tokenizer=_base_of(dataset, settings).tokenizer()
    )
    manifest, producer = dataset.manifest, dataset.manifest.producer
    request = JobRequest(
        model=manifest.base_checkpoint.name,
        training_file=dataset.ref.id,
        hyperparameters=dict(hyperparameters),
        suffix=job_suffix(producer.skill, producer.predictor),
        metadata={"campaign": producer.campaign, "round": str(producer.round)},
    )
    return trainer_client(settings).create_job(request)


# --------------------------------------------------------------------------- status


@dataclass(frozen=True)
class JobView:
    """A job, its last events, and where they were read from."""

    job: FineTuningJob
    events: tuple[dict[str, Any], ...]
    source: Literal["volume", "api"]


def _event_time(event: Mapping[str, Any]) -> int:
    created = event.get("created_at")
    return created if isinstance(created, int) else 0


def status(
    job_id: str, *, settings: Settings, events: int = _DEFAULT_EVENTS
) -> JobView:
    """Volume first (runs/<job>/job.json), HTTP second."""
    from aiagent.distill.client import read_events_from_volume, read_job_from_volume

    job = read_job_from_volume(settings.distill_dir, job_id)
    if job is not None:
        last = read_events_from_volume(settings.distill_dir, job_id, last=events)
        return JobView(job, tuple(last), "volume")
    client = trainer_client(settings)
    job = client.get_job(job_id)
    listed = sorted(client.list_events(job_id), key=_event_time)  # oldest first
    return JobView(job, tuple(listed[-events:] if events > 0 else ()), "api")


def wait(
    job_id: str,
    *,
    settings: Settings,
    poll_s: float = POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    on_poll: Callable[[JobView], None] | None = None,
    events: int = _DEFAULT_EVENTS,
) -> JobView:
    """Poll status until the job is terminal."""
    while True:
        view = status(job_id, settings=settings, events=events)
        if on_poll is not None:
            on_poll(view)
        if view.job.terminal:
            return view
        sleep(poll_s)


def warm_up(job: FineTuningJob, *, settings: Settings) -> str | None:
    """Warm the teacher the job's dataset names; that model string, or None (warned)."""
    from aiagent.distill import label as label_mod

    try:
        if job.training_file is None:
            raise DistillError(f"job {job.id} names no training_file")
        dataset = load_dataset(settings.distill_dir, job.training_file)
        model = dataset.manifest.producer.teacher.model
        label_mod.warm_up_teacher(model, settings=settings)
    except Exception as exc:  # a failed warm-up only delays the next teacher call
        logger.warning("teacher warm-up skipped: %s", exc)
        return None
    return model


# --------------------------------------------------------------------------- evaluate


@dataclass(frozen=True)
class _Run:
    """A run whose artifact checked out against its dataset and the current skill."""

    manifest: ArtifactManifest
    dataset: Dataset
    predictor: PredictorPlan
    verified: VerifiedArtifact


def _run_dir(run: str, settings: Settings) -> Path:
    try:
        safe_id(run, "run")
    except ValueError as exc:
        raise DistillError(str(exc)) from exc
    run_dir = settings.distill_dir / "runs" / run
    if not run_dir.is_dir():
        raise DistillError(
            f"run {run} not found under {settings.distill_dir / 'runs'} "
            f"(see `aiagent distill status {run}`)"
        )
    return run_dir


def _prepare_run(run: str, settings: Settings) -> _Run:
    """Read the run's manifest, load its dataset and fully verify the artifact."""
    from aiagent.system1.artifacts import read_manifest, verify_artifact

    run_dir = _run_dir(run, settings)
    manifest = read_manifest(run_dir)
    dataset_sha256 = manifest.binds.dataset_manifest_sha256
    dataset = load_dataset(
        settings.distill_dir,
        dataset_id(dataset_sha256),
        expect_manifest_sha256=dataset_sha256,
    )
    producer = dataset.manifest.producer
    chosen = plan(producer.skill, settings=settings).select(producer.predictor)
    verified = verify_artifact(
        run_dir,
        expected=_binds(chosen),
        dataset_manifest_sha256=dataset_sha256,
        max_len=dataset.manifest.max_len,
        head_max_len=dataset.manifest.head_max_len,
    )
    return _Run(manifest, dataset, chosen, verified)


def _items(rows: Sequence[Row]) -> list[EvalItem]:
    from aiagent.distill.gates import EvalItem

    return [
        EvalItem(
            row_id=row.id,
            state=row.state,
            labels={qid: gold.label for qid, gold in (row.gold or {}).items()},
            ids_sha256={qid: t.ids_sha256 for qid, t in row.student_tokens.items()},
        )
        for row in rows
    ]


@dataclass(frozen=True)
class EvalOutcome:
    """A saved eval report and where it was saved."""

    report: EvalReport
    path: Path


def evaluate(run: str, *, targets: GateTargets, settings: Settings) -> EvalOutcome:
    """Verify the run's artifact, score calib and held-out, decide, save the report."""
    from aiagent.distill.gates import (
        EvalReport,
        decide,
        find_report_for_dataset,
        question_report,
        save_report,
        score_items,
    )

    prepared = _prepare_run(run, settings)
    dataset, producer = prepared.dataset, prepared.dataset.manifest.producer
    runtime = prepared.verified.runtime
    questions = producer.questions
    calib = score_items(runtime, _items(dataset.rows["calib"]), questions)
    heldout = score_items(runtime, _items(dataset.rows["heldout"]), questions)
    reports = tuple(
        question_report(qid, calib[qid], heldout[qid], targets) for qid in questions
    )
    parent = producer.parent_dataset
    previous = None
    if parent is not None:
        previous = find_report_for_dataset(settings.artifacts_dir, parent)
    previous_accuracy = previous.accuracy if previous is not None else None
    pool_remaining = len(dataset.rows["pool"])
    verdict, reason = decide(
        reports,
        round=producer.round,
        pool_remaining=pool_remaining,
        previous_accuracy=previous_accuracy,
        targets=targets,
    )
    report = EvalReport(
        run=run,
        artifact_id=prepared.manifest.artifact_id,
        dataset_id=dataset.ref.id,
        parent_dataset=producer.parent_dataset,
        campaign=producer.campaign,
        round=producer.round,
        skill=producer.skill,
        predictor=producer.predictor,
        skill_source_sha256=_binds(prepared.predictor).skill_source_sha256,
        source_changed_since_labeling=not prepared.verified.source_matches,
        targets=targets,
        questions=reports,
        accuracy=sum(q.accuracy for q in reports) / len(reports),
        pool_remaining=pool_remaining,
        previous_run=previous.run if previous is not None else None,
        previous_accuracy=previous_accuracy,
        files=dict(prepared.verified.files),
        verdict=verdict,
        reason=reason,
        created_at=_utc_now(),
    )
    return EvalOutcome(report, save_report(settings.artifacts_dir, report))


# --------------------------------------------------------------------------- install


def install(run: str, *, settings: Settings) -> InstalledArtifact:
    """Install a run with a ship report for this artifact and skill source."""
    from aiagent.distill.gates import load_report
    from aiagent.system1.artifacts import install_artifact, read_manifest

    again = f"run `aiagent distill eval {run}`"
    report = load_report(settings.artifacts_dir, run)
    if report is None or report.verdict != "ship":
        verdict = "no eval report" if report is None else f"verdict {report.verdict}"
        raise DistillError(f"run {run} has no ship verdict ({verdict}); {again} first")
    binds = _binds(plan(report.skill, settings=settings).select(report.predictor))
    if report.skill_source_sha256 != binds.skill_source_sha256:
        raise DistillError(f"skill {report.skill} changed since eval; {again} again")
    run_dir = _run_dir(run, settings)
    manifest = read_manifest(run_dir)
    if manifest.artifact_id != report.artifact_id:
        raise DistillError(
            f"run {run} holds artifact {manifest.artifact_id}, not "
            f"{report.artifact_id} as at eval; {again} again"
        )
    hard = (manifest.binds.signature_sha256, manifest.binds.question_set_sha256)
    if hard != (binds.signature_sha256, binds.question_set_sha256):
        raise DistillError(
            f"the skill's signature/questions changed since eval; {again} again"
        )
    thresholds = {}
    for question in report.questions:
        if question.tau is None:
            raise DistillError(
                f"corrupt eval report for {run}: a ship verdict without a τ for "
                f"{question.qid}; {again} again"
            )
        thresholds[question.qid] = question.tau
    return install_artifact(
        run_dir,
        files=report.files,
        artifacts_dir=settings.artifacts_dir,
        skill=report.skill,
        predictor=report.predictor,
        run=run,
        thresholds=thresholds,
        target_precision=report.targets.precision,
        skill_source_sha256=binds.skill_source_sha256,
    )


# --------------------------------------------------------------------------- repair


@dataclass(frozen=True)
class RepairOutcome:
    """The next round's dataset and what went into it."""

    dataset: DatasetRef
    labeled: int
    unlabeled: int
    pool_remaining: int


def repair(
    run: str,
    *,
    max_rows: int,
    concurrency: int,
    settings: Settings,
    progress: Callable[[int, int], None] | None = None,
) -> RepairOutcome:
    """Label the pool rows the student is least sure of into a round+1 dataset."""
    from aiagent.distill.gates import load_report, select_for_repair
    from aiagent.distill.label import LabelConfig

    report = load_report(settings.artifacts_dir, run)
    if report is None or report.verdict != "repair":
        verdict = "no eval report" if report is None else f"verdict {report.verdict}"
        raise DistillError(
            f"run {run} has no repair verdict ({verdict}); "
            f"run `aiagent distill eval {run}` first"
        )
    prepared = _prepare_run(run, settings)
    parent = prepared.dataset
    producer = parent.manifest.producer
    checkpoint = _base_of(parent, settings)
    pool = {row.id: row for row in parent.rows["pool"]}
    selected = select_for_repair(
        prepared.verified.runtime,
        _items(parent.rows["pool"]),
        producer.questions,
        max_rows=max_rows,
    )
    chosen = [pool[row_id] for row_id in selected]
    teacher = producer.teacher
    answers = _teacher_answers(
        prepared.predictor,
        [row.state for row in chosen],
        lm=teacher_lm(settings, exact=teacher.model),
        config=LabelConfig(
            k=teacher.k, temperature=teacher.temperature, concurrency=concurrency
        ),
        progress=progress,
    )
    labeled = [
        row.model_copy(update={"split": "train", "gold": gold, "teacher": info})
        for row, (gold, info) in _answered(chosen, answers)
    ]
    unlabeled = len(chosen) - len(labeled)
    child, rows = derive_repair_dataset(
        parent,
        selected=selected,
        labeled=labeled,
        created_at=_utc_now(),
        version=__version__,
        unlabeled_added=unlabeled,
    )
    ref = write_dataset(
        settings.distill_dir / "inbox", base=checkpoint, producer=child, rows=rows
    )
    return RepairOutcome(
        dataset=ref,
        labeled=len(labeled),
        unlabeled=unlabeled,
        pool_remaining=len(pool) - len(selected),
    )
