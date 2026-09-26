# System 1 in the sentiment skill: design

**Status:** design, 2026-09-26, revision 2. Nothing is implemented.
- A review raised 13 points on revision 1, all applied here (§5). The owner decided D1-D8 on
  2026-09-26: "go with recommendations" (§3).

**Scope:** aiagent only: `core/sentiment.py`, `core/sentiment_stats.py`, `system1/cascade.py`,
`cli/sentiment.py`, `cli/run.py`, a small lab analysis script, tests and docs. No devai change, no
new dependency, no new training campaign.

**Inputs:**
- The owner's decision of 2026-09-26: the `polarity` student (laya-multilingual; 4 labels
  negative/neutral/mixed/positive; certified at 0.90) may gate **document-style** use. Opinion-only
  use waits for a better student. Wall-clock time, not money, is the binding constraint.
- The fresh-500 shadow measurement, per whole text:
  - all texts: the student answers 24.6% alone, with 95.1% agreement (Clopper-Pearson lower bound
    0.906);
  - opinion text: 16.8% coverage at 92.0% (lower bound 0.848);
  - neutral or encyclopedic text: about 84-95% coverage, never wrong.
- The same shadow log, split by the student's label for this revision (§2.2).
- `split_segments` run over fresh-500: sentiment makes **1442 segments** of its 500 texts (§2.9).
- Student cost: 1-3 s to load, once per process (§2.8), then about 50-110 ms per decision.
- [laya-system1-distillation.md](laya-system1-distillation.md) §7, §13 (including "Revisit
  later") and Appendix B (the resample/cache bug).

---

## 1. Summary

- **Today (0.5.2):**
  - `sentiment` splits the text into up to 24 segments.
  - It scores each segment with `ChainOfThought(ScoreSegment)` (an integer from −10 to +10, plus a
    rationale), three times, one call at a time.
  - It aggregates the scores and makes one `Predict(ExplainSentiment)` call.
  - The three samples are identical requests, so DSPy's cache answers all but the first:
    `model_uncertainty` is always 0, and a run makes 25 real calls, not 73.
- **PR 1, released as 0.6.0 (D8), with no laya dependency:**
  - Every sample is real: rollout ids 0..r−1 at temperature 0.7, for every `--resample r`.
  - At most 4 LLM calls are in flight across the whole process.
  - A parse failure drops that one sample, with no retry through JSON mode. A segment none of
    whose samples parses gets up to 2 more rollouts before the run fails.
  - `model_uncertainty` is pooled over the resampled segments, and `null` when there are none.
  - Real uncertainty costs about 2.8× the wall-clock (57 s against 20 s for 24 segments), so the
    default becomes `--resample 1` and uncertainty is opt-in (D6).
- **PR 2: System 1, neutral only.**
  - The installed polarity student scores a segment only when its top label is **neutral** at
    confidence ≥ τ. The score is then one calibrated value. The LLM scores every other segment
    exactly as in off mode.
  - The calibration is pinned in code together with the student's `artifact_id` and
    `ScoreSegment`'s signature hash. PR 2 ships none, so gate runs as shadow until the lab pins one.
- **Lab, then PR 3:**
  - Two shadow runs through sentiment's own segmenter: one on the target document corpus (D7), and
    one on fresh-500 as the review corpus.
  - A written pass test (§2.9). Then pin the calibration.
- **Modelled wall-clock** for 24 segments at an assumed 2 s per CoT call and a speedup of 2.6 at 4
  in flight. Both numbers are measured before PR 1 merges (§2.8).

  | Run | `--resample 1` | `--resample 3` | Uncertainty |
  |---|---|---|---|
  | 0.5.2 today (sequential, cache bug) | 50 s | 50 s | fake (always 0) |
  | PR 1, off | 20 s | 57 s | `null` at r=1; real at r=3 |
  | PR 2 gate, fresh-500 mix (14% neutral coverage) | 21-23 s | 53-55 s | as off, escalated segments only |
  | PR 2 gate, documents at 70% coverage | 10-12 s | 22-24 s | as above |
  | PR 2 gate, documents at 90% coverage | 7-9 s | 10-12 s | as above |

## 2. Design

### 2.1 Scoring a student segment (redone: D2)

**Recommendation.**

- **One value.** The gate accepts only neutral segments (§2.2), so every student segment scores
  `NEUTRAL.level`: the **mean** LLM segment score over the accepted neutral segments of the
  calibration run (§2.9).
  - The statistic is the one the score stands in for: the expected LLM score of a segment that the
    gate accepts. So it is unbiased for those segments by construction.
  - It is rounded to 2 decimals, like an LLM segment's mean.
- **No provisional value.** PR 2 ships `NEUTRAL = None`, and gate then runs as shadow (§2.4).
  Measured and guessed values are never mixed.
- **The student's spread is measured too.** The same run gives two variance components over the
  accepted neutral segments:
  - σ_w²: the pooled within-segment variance of the r = 3 LLM samples;
  - σ_b² = max(0, var(ȳ_i) − σ_w²/3), where ȳ_i is segment i's LLM mean.
  - A student segment stands in for an r-sample LLM mean, whose residual variance is
    **σ²(r) = σ_b² + σ_w²/r**. With both parts pinned, one run serves every `--resample`.
- **The pinned record,** in `core/sentiment.py`, next to `ScoreSegment`, with a comment that cites
  the run:
  `NeutralCalibration(artifact_id, score_signature_sha256, level, se, sigma_between, sigma_within, n, measured)`.
  `n` and `se` are reported with it.
- **Statistics:**

  | Statistic | Computed over |
  |---|---|
  | mean, volatility, t-test, 95% CI | Every segment score, student and LLM alike. The variance gets the student term below. |
  | `model_uncertainty` | Pooled √(mean s²) over the segments with **≥ 2 LLM samples**; `null` when there are none. It stays "how much the LLM disagrees with itself". |
  | `n_resampled` | How many segments `model_uncertainty` covers. |
  | `n_samples` | LLM score samples used in the statistics. |

  - **The student term.** `summarize(llm_samples, student_scores=(), student_variance=0.0)`
    computes volatility² = s²(all segment scores) + (n_student/n)·σ²(r), and std_error =
    volatility/√n. It is pure and a few lines.
    - Without the term, student scores would enter the t-test as if they had no error.
      Volatility, the standard error, t, p, the CI and the "high/moderate" label would all come out
      overstated, more so as coverage rises.
    - At 100% coverage the standard error is σ(r)/√n, not 0.
  - **Why student segments count for the mean and spread:** without them, the statistics would
    describe only the hard segments, and there would be nothing to compute at full coverage.
  - **Why they do not count for `model_uncertainty`:** a student segment has no resamples.
    Counting it as zero spread would fake certainty, the same failure as the cache bug.
  - **Why pooled:** the mean of per-segment sample standard deviations is biased low by c4(r):
    0.80 at r = 2 and 0.89 at r = 3. So values were not comparable across `--resample` settings.
- **Every segment is marked:** `source: "student" | "llm"`, and `student: {label, confidence}`
  (§2.7).

**Honest limits.**
- The σ term restores the spread to first order. It assumes that the calibration corpus resembles
  the scored one, which is why the pass test runs on the target corpus.
- Two limits predate this design and are left unchanged; the manual will state them:
  - segments are not independent (neighbouring paragraphs, repeated boilerplate), so the standard
    error is optimistic in every mode;
  - the CI uses z = 1.96, while the p-value uses t(n−1).

**Alternatives (rejected):**

- **The expected value Σ p_k·LEVEL_k with median calibration** (revision 1). It is biased by
  construction: an accepted positive segment at p = 0.85 scores about 0.85·L_pos, 10-15% below the
  level it was calibrated to. Also, a median is not the mean of a skewed per-label distribution.
- **LEVEL by least squares on the four probabilities:** moot with one accepted label, and it would
  need the probabilities in the log.
- **Four levels (negative/neutral/mixed/positive):** they wait for a wider gate (§2.2).
- **The spread of the student's own distribution as `model_uncertainty`:** it mixes a calibrated
  label probability with an LLM's score spread, which are different quantities.
- **A score-type student:** `ScoreSegment` has 21 levels, more than the 10 that qualify, and it
  would need a new signature and a new campaign.

### 2.2 Gate policy per segment (redone: D3)

**Recommendation.**

- **The student answers a segment** only when all of these hold:
  - it can see the whole segment;
  - its top label is `neutral`;
  - its `answer_confidence` is at least τ: the installed τ (0.894 for `a866e0a4`), or
    `system1_min_conf` when that is set;
  - a calibration matching this student and `ScoreSegment` is pinned (§2.4).

  The segment then gets **no LLM call at all**.
- **Every other segment** goes to the LLM with `--resample` samples, exactly as in off mode.
- **No LLM resamples for a student segment.** Resampling measures the LLM's self-disagreement on a
  segment. For a segment the LLM did not score, that measures nothing, and it would spend exactly
  the calls the gate saves.
- **Fail open, as the cascade does:** if the student cannot load or run, every segment goes to the
  LLM, with one warning.

**What the polarity certificate does not cover.** Nothing from it carries over to this use. The
pass test in §2.9 is the certificate for sentiment.
- **It measures agreement with the polarity LLM's *label*,** not with `ScoreSegment`'s score band.
  A mildly positive passage is "positive" to polarity but +1 (the neutral band) to `ScoreSegment`.
- **It was measured on different units.**
  - The student was trained and certified on distill segments. Those pack paragraphs greedily up to
    the student's token limit (`distill/segment.py:179-190`), and fresh-500 was measured per whole
    text.
  - Sentiment splits a one-paragraph text into single sentences (`core/segment.py:31-37`), and a
    longer text into paragraphs, merged above 24.
  - Single sentences, headings and boilerplate are a different population, so τ's precision does
    not carry over.

**Why neutral only.** The fresh-500 shadow log, split by the student's label: 492 of its 500 lines
join to the corpus sidecar, and the other 8 lines add 2 acceptances, both right.

| Slice (per whole text, τ = 0.894) | Accepted | Agree | Precision | CP lower bound (α = 0.05) |
|---|---|---|---|---|
| All accepted | 121 (24.6%) | 115 | 0.950 | 0.904 |
| — neutral | 69 | 66 | 0.957 | 0.891 |
| — positive, negative or mixed (all opinion text) | 52 | 49 | 0.942 | 0.858 |
| Encyclopedic (Wikipedia, 52 texts), neutral | 47 (90.4%) | 47 | 1.000 | 0.938 |
| Opinion (440 texts), neutral | 22 (5.0%) | 19 | 0.864 | 0.684 |

- **On documents neutral-only costs nothing:** every Wikipedia acceptance was neutral.
- **It leaves one value to calibrate,** with plenty of data. The `mixed` question disappears, and
  little variance is lost.
- **On opinion text it does not raise precision.** The neutral acceptances are the weakest slice
  there: 3 wrong, where the LLM said positive (2) or negative (1), at confidence 0.91-0.93.
  - It does cut opinion coverage from 16.8% to 5.0% per text, which bounds how far wrong levels can
    move the mean.
  - Per sentence the picture may differ: review sentences such as "I ordered a size M" are often
    neutral. The review run (§2.9) measures that, and it is the number behind D3.
- **Widen the gate later,** with the opinion-weighted student and a calibration of its own.

### 2.3 Rationale and excerpts for student segments

**Recommendation.**

- **A student segment has no rationale** (`null` in the JSON) and costs no extra LLM call.
- **`ExplainSentiment` is unchanged:** the same signature and one call.
- **Its `excerpts` input** keeps the LLM segments' lines as today (`- (+4.3) <rationale>`), in
  segment order and within the same 2000-character budget. It adds one line such as
  `18 of 24 segments neutral (System 1)`.
  - The budget stays for LLM rationales.
  - Raw segment text, possibly from a fetched URL, stays out of the explain prompt. Today that
    prompt sees only model-written rationales.

**Alternatives (rejected):**

- **Quoting 160 characters of each student segment** (revision 1): it spends the excerpt budget
  and puts raw input text into the explain prompt.
- **A rationale LLM pass for student segments:** it costs one call per segment, or one batched call
  under a new signature. Either spends the decode time the gate saved.
- **Leaving student segments out entirely:** the explanation would not know that most segments were
  neutral.

### 2.4 How sentiment finds the student, and the calibration guard

**Recommendation: reuse the installed polarity student through a declared dependency (D1,
accepted).**

- **The declaration:** `SentimentModule.system1_student = ("polarity", "classify")`.
  - It serves one user. It stays a small branch in `apply_system1` and is not generalized further.
- **Loading.** `apply_system1(module, skill, settings, registry=None)` handles a module that
  declares a borrowed student:
  - it resolves `polarity` from the registry (loading it if `registry` is None) and builds it;
  - it runs `_wrapper`'s checks on `polarity/classify`:
    - installed;
    - hard bind (signature and questions): otherwise the student is not used, with a warning;
    - soft bind (polarity's skill files): if they changed, gate degrades to shadow;
  - it calls `module.use_student(student, mode, shadow_log)`, passing the shadow-log path in both
    modes.
- **The calibration guard,** in `SentimentModule.use_student`. It is the same pattern as the soft
  bind (`system1/cascade.py:243-250`).
  - Gate needs `NEUTRAL` to be pinned, with `NEUTRAL.artifact_id` equal to the student's and
    `NEUTRAL.score_signature_sha256 == signature_sha256(ScoreSegment)`.
  - Otherwise the module warns once ("sentiment's System 1 calibration is missing or stale …") and
    shadows.
  - So a new polarity student, or any change to `ScoreSegment`, can never gate with a stale value.
- **Lazy imports.**
  - `core/sentiment.py` names `Student` only under `TYPE_CHECKING`.
  - `cli/sentiment.py` imports `aiagent.system1.cascade` only when the mode is not `off`, as
    `run.py` does.
  - numpy and onnxruntime therefore stay out of off mode and out of `aiagent.cli.app`.
- **A small refactor in `cascade.py`: extract a `Student` class from `System1First`.**
  - It is a pure move, guarded by the existing cascade tests.
  - `Student` does the lazy load under `_LOAD_LOCK`, `fits`, `predict`, the τ threshold and
    failing open.
  - It adds a `_PREDICT_LOCK` around `session.run` (§2.5).
  - `Student.consult(text) -> Verdict | None` returns `None` only when the student is broken.
    - `Verdict` = `fits`, `n_tokens` (from `runtime.tokenizer.state_ids`), `label`,
      `answer_confidence`, `clears_tau` and `ms`.
    - For a segment that does not fit, `label` and `answer_confidence` are `None`.
  - `System1First` keeps the LLM fallback and the polarity shadow log, on top of `Student`.
- **Configuration:**
  - `system1_mode["sentiment"]`: the existing setting, default `off`.
  - It is independent of `system1_mode["polarity"]`.
  - There is no new setting.
- **Entry points:**
  - `aiagent sentiment` gains the lazy `apply_system1` call when the mode is not `off`.
  - `aiagent run sentiment` already calls `apply_system1`; `run.py` now passes its registry.
  - Neither emits the "no predictor has an installed student" warning any more. The warning for a
    missing student names `polarity/classify`.
- **Shadow mode:**
  - The LLM scores every segment, as in off mode. The statistics are exactly off mode's, plus the
    `system1` block.
  - The student is consulted on every segment.
  - One line per segment goes to `<artifacts_dir>/system1/skills/sentiment/score/shadow.jsonl`,
    with no input text:

    | Field | Meaning |
    |---|---|
    | `ts`, `artifact_id` | As in polarity's log. |
    | `run_id` | Random, one per `forward` call (one document in `run --jsonl`). It groups a document's lines. |
    | `seg_index`, `n_segments` | The segment's position in its run, and the run's segment count. |
    | `doc_sha256` | sha256 of the text given to `forward`. It joins a run to a corpus sidecar (source, language), as the fresh-500 analysis did. |
    | `input_sha256` | sha256 of the segment. |
    | `n_tokens`, `fits` | The segment's student tokens, and whether the student saw it whole. |
    | `student`, `confidence` | The student's top label and its `answer_confidence`; `null` when it did not fit. |
    | `would_accept` | What the neutral-only gate would have done. |
    | `llm_samples` | The segment's LLM scores, in rollout order; a dropped sample is `null`. |
    | `student_ms` | The student's time for this segment. |

  - Bands and agreement are derived offline, so the log does not fix the ±2 bands into the data.
- **Gate mode:** §2.2.

**Why reuse:**

- No new campaign: that would take days of teacher and GPU time, and wall-clock time is the binding
  constraint.
- One student to certify and maintain.
- The planned opinion-weighted student reaches sentiment's **shadow** mode through
  `distill install` alone. Its gate needs a new calibration run and pin, and the guard enforces
  that.

**The cost of reuse:** sentiment depends on another skill's signature. If polarity's signature
changes, the bind check switches sentiment's student off with a warning. It never answers wrongly
because of that.

**Alternatives (rejected):**

- **A sentiment-owned student:** its own `classify` predictor, campaign and install. It is worth that
  only if the shadow runs show the polarity student disagreeing with `ScoreSegment` much more than
  it did with the polarity LLM.
- **A separate `sentiment_student` setting:** a second knob for one choice.
- **Replacing `score` through `apply_system1`:** `ScoreSegment` does not qualify (21 levels, plus
  a rationale).

### 2.5 Batching and concurrency

**Recommendation.**

- **Load the student once per process.** The one `Student` is shared by the threads of
  `run sentiment --jsonl`.
- **Serialize student calls with a lock.**
  - The onnxruntime session sets inter-op threads to 1 but leaves intra-op threads at all cores.
    Four threads calling `session.run` at once oversubscribe the CPU.
  - 50-110 ms per call is negligible next to the LLM.
  - The lock also applies to polarity's `run --jsonl`, which has the same problem.
- **One student pass, in segment order,** with one `session.run` per segment. That is the
  runtime's rule: batched runs were 5x slower with the traced export. The dynamo export has not
  been measured batched.
- **Pipelined:**
  - The main thread decides segment *i*, then submits that segment's LLM samples to a
    `ThreadPoolExecutor(4)` before it decides segment *i+1*. Student CPU work therefore overlaps
    the LLM waits.
  - Each (segment, sample) pair is one task, run under `contextvars.copy_context()` as
    `run --jsonl` does.
  - Results land by index, so the statistics equal the sequential ones.
  - The first exception other than a parse failure cancels the pending tasks and propagates, as in
    `distill/label.py`.
  - The explain call comes after all scores are in.
- **At most 4 LLM calls in flight per process.** A module-level
  `BoundedSemaphore(MAX_IN_FLIGHT = 4)` wraps every LLM call the module makes, score and explain
  alike.
  - 4 matches the devai teacher's `--max-num-seqs 4` and the defaults of `run` and `label`.
  - It applies in off mode too.
  - Without it, `run sentiment --jsonl --concurrency 16` (`cli/run.py:23` allows 16) would put
    16 × 4 = 64 calls in flight. Under that load a 429 or 503 counts against `num_retries=2` and
    fails the row. With it, extra `--concurrency` only queues.
  - An outer thread never holds the semaphore while it waits for its pool, so the nested pools
    cannot deadlock.
- **Each distinct segment text is scored once per run.**
  - Otherwise identical segments (repeated boilerplate) would send the same (text, rollout id)
    request at the same time, and both would miss the cache.
  - A repeated segment reuses the samples and still counts at each of its positions in the
    statistics.

### 2.6 The resample/cache bug, and sample handling (PR 1)

**Recommendation.**

- **For every `--resample r`, including 1,** sample *j* = 0..r−1 calls
  `self.score(text=segment, config={"rollout_id": j, "temperature": SAMPLE_TEMPERATURE})`.
  - `SAMPLE_TEMPERATURE = 0.7`, `label.py`'s default.
  - r = 1 is then exactly sample 0 of r = 3, and a later r = 3 run takes it from the cache. It
    also samples the same score distribution as r = 3, not the server's default temperature.
  - The temperature must be above 0. At 0, DSPy still keys the cache by `rollout_id`, but it warns
    that the id has no effect on generation: the r samples would be one answer r times.
- **The same rollout ids are used for every segment and every run.**
  - Re-running the same text is answered from the cache with the same samples: deterministic and
    free.
  - A new text gets *r* real samples.
- **The adapter.** Score calls run under `ChatAdapter(use_json_adapter_fallback=False)`, as
  `distill/label.py:99-105` does.
  - Today any exception retries once through JSONAdapter, whose retry requests server JSON mode.
    devai strips that; the retry recovers only if the model follows the JSON prompt on its own.
  - The explain call keeps today's adapter (out of scope).
- **Parse failures become missing samples.**
  - An `AdapterParseError` drops that sample. `ScoreSegment.score` is an `int`, so a score that
    does not parse fails inside the adapter, before `_parse_score` sees it.
  - Today's 0.0 fallback in `_parse_score` goes, but it never ran: the adapter error came first,
    so today's parse failure goes to the JSONAdapter retry and, if that fails too, fails the run.
    No 0 was ever averaged in.
  - The segment keeps its other samples. Dropped samples do not count in `n_samples`.
  - **A segment none of whose samples parses** gets up to `MAX_EXTRA_SAMPLES = 2` more rollout
    ids (r, then r + 1), under the same adapter, until one parses. With the default r = 1, a
    single parse failure would otherwise fail the run, and since the cache keeps the failed
    answer, every re-run of that text too. The extra samples are ordinary cached calls: a re-run
    is deterministic and free, and a later r = 3 run reuses them.
  - If they all fail, the run errors, as today when the JSONAdapter retry fails too. The message
    says that a re-run fails the same way from the cache, and that a higher `--resample` or
    `AIAGENT_CACHE=false` draws new samples.
  - Any other error still aborts the run (`RetryAwareLM` has already retried the transient ones).
- **Cost:** 25 real calls become 73 at r = 3, about 2.8× the wall-clock at the same concurrency
  (§2.8). That is why the default becomes r = 1 (D6).

**Tests:**

1. **DummyLM history:** each segment is called with rollout ids {0..r−1} at temperature 0.7, for
   r = 3 and for r = 1 (id 0).
2. **Appendix B's reproduction, as a regression test:**
   - setup: a real `dspy.LM` with a memory-only cache (the disk cache is off and restored
     afterwards), and litellm's completion stubbed to return a score that depends on `rollout_id`;
   - 4 distinct segments × 3 samples, plus the explain call, make **13 completions** (the bug made
     5);
   - `model_uncertainty > 0` (the bug gave 0.0);
   - an identical second run makes 0 completions.
3. **Parse failures:**
   - one unparseable sample is dropped, and `n_samples` excludes it;
   - no sample of a segment parses: rollouts r and r + 1 are drawn for it alone, and the first
     that parses is its sample; a re-run takes it from the cache;
   - all r + 2 samples of a segment unparseable: the run errors, naming the way out;
   - no retry goes through JSONAdapter.

### 2.7 Output and JSON

| Field | 0.5.2 | New |
|---|---|---|
| `segments[i].source` | – | `"llm"` or `"student"`, always present |
| `segments[i].rationale` | str | str; `null` for a student segment |
| `segments[i].student` | – | `{label, confidence}` whenever the student answered the segment (shadow or gate); `null` otherwise (off, too long, student failed) |
| `model_uncertainty` | float (0.0 under the bug) | pooled √(mean s²) over segments with ≥ 2 LLM samples; `null` when there are none (D4) |
| `n_resampled` | – | the number of segments `model_uncertainty` covers |
| `n_samples` | LLM samples claimed (cache hits counted) | LLM score samples used in the statistics |
| `system1` | – | `null` when off or when no student is usable (a warning says why). Otherwise `{mode, student: "polarity/classify", artifact_id, tau, accepted, too_long, coverage}`, where `mode` is the effective mode |

- **`accepted`:** the segments the gate took (gate) or would have taken (shadow).
- **`coverage`:** `accepted / n_segments`.
- **`student`** is nested so that its float `confidence` cannot be confused with the top-level
  `confidence`, which is a string label.
- **In gate mode, `model_uncertainty` covers only the escalated segments,** which are the harder
  ones. It tends to rise with coverage and is not comparable with an off-mode run. `n_resampled`
  says how many segments it rests on.
- **Human output:**
  - a `system 1` line, e.g. `18/24 segments by the student (gate, polarity a866e0a4), 1 too long`;
  - the uncertainty line reads `n/a (no segment scored twice)` when the value is null.

**Compatibility: PR 1 is 0.6.0, not a patch release (D8).** It changes every user's scores (they
become means of samples at 0.7), it changes the default `--resample` (D6), and it changes a JSON
type. The CHANGELOG says that scores differ from 0.5.x.

### 2.8 Expected wall-clock (24 segments)

**Model:**

```
T ≈ [L + n·s]  (System 1 on)  +  r·m·t / S  +  t_e
```

| Symbol | Meaning | Value and source |
|---|---|---|
| L | Student load | **1-3 s.** The manual gives 0.85 s (`USER_MANUAL.md:389`) and 2.81 s (`:424`). Importing numpy, onnxruntime and tokenizers, and reading a 1.3 GB model from a cold page cache, add to it. |
| s | Student time per segment | 0.08 s (measured: 50-110 ms) |
| n | Segments | 24 |
| r | Resamples | 1 or 3 |
| m | Escalated segments | n × (1 − coverage) |
| t | One warm CoT call | **Assumed 2 s, not measured.** The anchor is a warm no-think polarity `Predict` call at about 0.6 s; a CoT call also writes a reasoning paragraph and a sentence. |
| t_e | The explain call | t |
| S | Speedup at 4 in flight | **2.6, not shown for this workload.** It was measured on the teacher with short `Predict` calls (4.4 calls/s at 4 in flight against 1/0.6 s sequentially). The teacher uses MTP speculative decoding, whose per-request gain shrinks as more requests run together. Sentiment runs on the `default` alias, which is not necessarily a 4-slot vLLM backend. On a 1-slot backend S ≈ 1. |

- The student pass is added in full, although pipelining hides most of its per-segment part.
- Process start, the dspy import and ingest are the same in every mode and are left out.

**At t = 2 s:**

| Run | LLM score calls (r = 3) | r = 1, S = 2.6 | r = 3, S = 2.6 | r = 3, S = 1 |
|---|---|---|---|---|
| 0.5.2 today (sequential, cache bug) | 24 real + 48 cache hits | 50 s | 50 s | 50 s |
| PR 1, off | 72 | 20 s | 57 s | 146 s |
| gate, 5% (neutral, opinion text) | 68 | 22-24 s | 58-60 s | 142-144 s |
| gate, 14% (neutral, fresh-500 mix) | 62 | 21-23 s | 53-55 s | 129-131 s |
| gate, 70% (documents, illustrative) | 22 | 10-12 s | 22-24 s | 48-50 s |
| gate, 90% (Wikipedia, per text) | 7 | 7-9 s | 10-12 s | 19-21 s |

- **What real uncertainty costs.** At the same concurrency, today's calls would take 20 s. Real
  uncertainty at r = 3 costs about 2.8× that, and r = 2 takes 39 s (98 s at S = 1). This is the
  basis for D6.
- **Break-even of the student.** Its fixed cost is about 2.9-4.9 s for 24 segments. It saves r·t/S
  per accepted segment: 0.77 s at r = 1, or 2.3 s at r = 3 (at S = 2.6). It therefore pays back
  after 4-7 accepted segments at r = 1, or 2-3 at r = 3.
- **For a one-sentence text in a one-shot run, the gate is always a net loss.**
  - The student costs 1.1-3.1 s.
  - The saving is at most r·t, and only when the sentence is accepted.
  - `run sentiment --jsonl` amortizes the load, so batch such texts.
- **Coverage is per segment here.** The fresh-500 figures are per text. The shadow runs give the
  per-segment ones (§2.9).
- **Before PR 1 merges:** measure t and S for `ScoreSegment` on the default backend, at
  `--resample 1` and `--resample 3`, each at 1 and at 4 in flight. A throwaway lab script that sets
  `MAX_IN_FLIGHT` does this. Put the numbers in the PR and redo this table from them.

### 2.9 Calibration and the pass test (lab, then PR 3)

**Recommendation.** Gate is enabled for a corpus only after a sentiment shadow run on that corpus
passes the test below. The run goes through sentiment's own module, so the real segmenter is used:

```bash
AIAGENT_SYSTEM1_MODE='{"sentiment":"shadow"}' aiagent run sentiment --jsonl docs.jsonl
# one document per line: {"text": "...", "resample": 3}
```

`aiagent sentiment` per document is equivalent, but it loads the student each time, and it joins
several sources into one text (`cli/sentiment.py:62`).

1. **Count first.** Run `split_segments` over the corpus. It is pure and costs nothing. The lab time
   is then (3·segments + documents)·t/S.
2. **The document run,** on the target document corpus (D7): the calibration, and the pass test
   for gate.
3. **The review run,** on fresh-500. Its 440 opinion texts put a number on the D3 risk.
   - Sentiment makes 1442 segments of fresh-500 (mean 2.9 per text; the English and Polish reviews
     5.3-5.9, because a one-paragraph review splits into sentences).
   - That is 4826 calls: about **62 min** at t = 2 s and S = 2.6, or 2.7 h at S = 1.
   - Revision 1's "25 min, one segment per text" was too low by about 2.4×, and by 5-6× for the
     reviews.
4. **Analysis.** A small pure script in PR 3 reads the shadow log and prints everything below, so
   the pinned numbers can be reproduced.
   - For each accepted neutral segment it takes ȳ_i, the mean of its LLM samples.
   - **LEVEL** = mean ȳ_i, with n and its standard error.
   - **σ_w and σ_b** as in §2.1.
   - Per-segment coverage, the `too_long` share, and coverage against `n_tokens`.
   - LEVEL is one number fitted on hundreds of segments. Its in-sample optimism is far below the
     tolerance, so the same run fits and tests; LEVEL's standard error shows it.

**The pass test** (thresholds: D2):

- **(a) Band agreement.** An accepted segment agrees when its ȳ lies in (−2, +2), the aggregate's
  own neutral band. The one-sided Clopper-Pearson lower bound (α = 0.05) of agreement over the
  accepted segments must be **≥ 0.90**, the owner's certification level.
  - That needs at least 29 accepted segments with no disagreement.
  - It takes about 90 at a true agreement of 96%, and about 130 at 95%.
- **(b) Mean drift.**
  - For each document with at least one accepted segment, Δ = (gate-mode mean) − (off-mode mean).
    The gate-mode mean is computed offline, with LEVEL in place of ȳ on the accepted segments.
  - Over **≥ 100** such documents, the 95th percentile of |Δ| must be **≤ 0.5**. The mean of Δ is
    reported too.
- **If the test fails at the installed τ,** the log shows whether a higher τ would pass. A per-skill
  τ is out of scope, so that becomes a follow-up decision.
- **Pin.** PR 3 adds `NEUTRAL` with the run's `artifact_id` and `ScoreSegment` hash, plus the
  script and its unit test. The owner then enables `system1_mode.sentiment = "gate"` for the
  corpora that passed.

### 2.10 Tests, docs, CHANGELOG

**Tests** are hermetic, use DummyLM, and keep the coverage gate at 85%.
- `install()` moves from `tests/test_system1_cascade.py:112` to `tests/system1_helpers.py`.
- The fixture student (`tests/fixtures/system1`) is installed as `polarity/classify`. Fixture texts
  are chosen by their golden label, neutral or not.

- **DummyLM rules:**
  - With 4 calls in flight, list mode hands out answers in completion order, so it is used only
    with identical answers. The existing tests do that and stay green.
  - Per-segment answers use dict mode, which returns the first key found in the last message:
    - the explain answer goes first, under a key only the explain prompt contains, since that
      prompt carries rationales;
    - no segment text is a substring of another.
- **`test_sentiment_stats.py`:**
  - `summarize` with `student_scores` and `student_variance`:
    - student scores count for the mean;
    - volatility and the standard error include (n_student/n)·σ²(r);
    - at 100% coverage the standard error is not smaller than σ(r)/√n;
  - `model_uncertainty`:
    - it is pooled over segments with ≥ 2 samples;
    - it is `None` at r = 1 and for an all-student run;
    - `n_resampled` is right;
  - `n_samples` counts the LLM samples used.
- **`test_sentiment.py` (PR 1):**
  - the §2.6 tests;
  - **the in-flight bound:**
    - a `threading.Barrier(4)` fake LM, not sleeps: the 5th call must not arrive before the barrier
      releases;
    - it runs with two `forward` calls in parallel threads, as `run --jsonl` makes, so the bound is
      process-wide;
  - each segment's score lands in its own slot (dict mode);
  - identical segments are scored once and counted at each position;
  - an LLM error propagates and cancels the pending calls.
- **`test_sentiment_system1.py` (PR 2)**, with `NEUTRAL` patched to the fixture's `artifact_id` and
  the current `ScoreSegment` hash, except where noted:
  - **gate at τ = 0** (`install(tau=0.0)`, since `system1_min_conf` must be greater than 0,
    `config.py:152`):
    - neutral segments go to the student and the others to the LLM;
    - `lm.history` holds only the escalated score calls and the explain call;
    - the excerpts carry the "k of n segments neutral" line and no segment text;
  - **gate at `min_conf` 1.0:** every segment goes to the LLM, and the output equals off mode's
    plus a `system1` block with `accepted` 0;
  - **a segment longer than the fixture's 128 tokens:** it goes to the LLM, `too_long` is 1, and its
    shadow line has `fits: false` and `n_tokens`;
  - **shadow:**
    - the statistics equal off mode's;
    - there is one line per segment with the §2.4 fields and no text;
    - one `run_id` per `forward` call, and `seg_index` 0..n−1;
  - **the stale-calibration guard:** `NEUTRAL` None, another `artifact_id`, or another
    `ScoreSegment` hash: gate runs as shadow, with one warning;
  - **a pinned-hash test:** `signature_sha256(ScoreSegment)` equals the literal pinned next to
    `NEUTRAL`, so editing the signature fails a test that says to recalibrate;
  - **failing open:**
    - a broken runtime: every segment goes to the LLM, with one warning;
    - a hard-bind mismatch: the student is not used, with a warning;
    - polarity's `SKILL.md` changed in gate mode: the run shadows instead;
  - **one student load** is shared by `run sentiment --jsonl` threads (the runtime factory is called
    once);
  - **both entry points:** `aiagent sentiment --json` and `aiagent run sentiment` take this path,
    without the "no installed student" warning;
  - **off mode:** `artifacts_dir` is never read, and `aiagent.system1` is not imported.
- **PR 3:** the analysis script on a hand-made log, with a known LEVEL, σ, band agreement and Δ.
- **Existing tests:** the cascade tests stay green after the `Student` extraction. The subprocess
  import test still guarantees that `aiagent.cli.app` loads neither numpy nor onnxruntime.

**Docs:**

- **USER_MANUAL, `sentiment` section:**
  - the default `--resample 1` (D6); use `--resample 3` for model uncertainty, at about 2.8× the
    time;
  - `model_uncertainty` is pooled; it is `null` without resamples; in gate mode it covers only the
    escalated segments (`n_resampled`);
  - scoring runs 4 LLM calls at a time per process;
  - existing limits: segments are not independent, and the CI uses z while p uses t(n−1);
  - System 1 is neutral-only, for document corpora that passed the pass test. Review and opinion
    corpora stay off or shadow. A call that mixes a review with an article counts as a review.
- **USER_MANUAL, `distill` → Serving:**
  - sentiment borrows polarity's student under `system1_mode.sentiment`;
  - the shadow log's location and fields;
  - gate needs a pinned calibration. Recalibrate after a new polarity student, a `ScoreSegment`
    change, or a change of the model behind the `default` alias;
  - "Not yet" now reads "other than `run` and `sentiment`".
- **USER_MANUAL settings table:** the `system1_mode` row names sentiment.
- **`builtin_skills/sentiment/SKILL.md`:** a short System 1 note, and the default `--resample`.
  This changes sentiment's skill-source hash, which is harmless because sentiment has no student of
  its own.
- **laya design doc, Appendix B:** mark the bug fixed and link here.
- **CLAUDE.md, Architecture:** one line saying that sentiment borrows polarity's student through
  `apply_system1`, neutral-only, behind a pinned calibration.

**CHANGELOG `[Unreleased]`, PR 1 (0.6.0):**

- **Changed:** sentiment scores differ from 0.5.x.
  - Every LLM score is a sample at temperature 0.7 (rollout ids 0..r−1).
  - The default `--resample` is 1 (D6); `model_uncertainty` is then `null`.
  - Segments are scored 4 LLM calls at a time per process.
  - `model_uncertainty` is pooled (√ of the mean within-segment variance), `null` when no segment
    was scored twice, and comes with `n_resampled`.
- **Fixed:** `sentiment` now actually measures model uncertainty.
  - Its resamples were identical requests, so the cache (on by default since the MVP) answered all
    but the first: `model_uncertainty` was always 0, and a run made 25 LLM calls, not 73.
  - Re-running the same text is still free.
- **Fixed:** a score that does not parse is no longer retried through JSONAdapter (server JSON
  mode). It is dropped, and the segment keeps its other samples; a segment with none gets up to 2
  more rollouts; if none parses, the run fails, as in 0.5.x when the JSON retry failed too.

**CHANGELOG, PR 2:**

- **Added:** System 1 for `sentiment`. `system1_mode.sentiment = "shadow" | "gate"` uses the
  installed polarity student, for neutral segments only. Gate needs a pinned calibration and
  shadows until one exists. The JSON gains `segments[i].source`, `segments[i].student` and a
  `system1` block.

### 2.11 Risks and scope

**Risks:**

1. **Opinion text.**
   - Sentiment is used on reviews: its own `SKILL.md` example is `review.txt`.
   - Per text, the student's opinion-text neutral acceptances were 19/22 right (lower bound 0.684).
     That use is outside the owner's approval.
   - Per sentence, coverage on reviews may be much higher than the 5% seen per text.
   - `aiagent sentiment` joins all sources into one text, so a single run cannot separate a review
     from an article.
   - Mitigation: D3, plus the review run's numbers.
2. **Different segments.**
   - The pass test runs on sentiment's own segments.
   - In long documents, merged paragraphs may exceed the student's 1002-token budget. They go to
     the LLM (`too_long`, visible through `n_tokens`), and coverage falls.
3. **Stale calibration.**
   - A new polarity student or a `ScoreSegment` change is guarded: the run shadows.
   - A change of the model behind the `default` alias is **not** guarded, because the served model
     is not known without a network call. The manual says to recalibrate.
4. **Two instruments.** The σ term restores the spread to first order only, and only for a corpus
   like the calibration corpus.
5. **Selection.**
   - The student takes the easy, neutral segments, and the LLM the hard ones.
   - The mean is fine. Comparing student and LLM segments, or gate-mode and off-mode
     `model_uncertainty`, is not like-for-like.
6. **After PR 1:**
   - Every user's numbers change, and by default uncertainty is `null` (D6).
   - One failed LLM call other than a parse failure still fails the whole run.
   - A segment whose r + 2 samples all fail to parse fails the run too. At the default r = 1 that
     is 3 samples; 0.5.x had one ChatAdapter answer plus the JSONAdapter retry. The cache keeps
     the failed answers, so a re-run of that text fails the same way until `--resample` goes up
     or the cache is off.
7. **The wall-clock model** rests on an assumed t and S until PR 1 measures them.

**Out of scope:**

- **Other students:** a sentiment-owned student, a score-type student, and the opinion-weighted
  student. The last is a separate campaign; see "Revisit later" (augmentation, a bigger corpus).
- **A wider gate:** positive, negative and mixed segments. That waits for the opinion-weighted
  student and a four-label calibration.
- **Batching and overlap:** batched ORT runs (measure the dynamo export first), and overlapping the
  student's load with LLM calls.
- **Segmentation:** capping sentiment's segments by student tokens (that changes the 24-segment
  cap).
- **Thresholds and gates:**
  - a τ per skill (`system1_min_conf` is global);
  - a merged-band gate that accepts when `p_neutral + p_mixed ≥ τ`: more coverage, but it is not
    what was certified.
- **Explanations:** student rationales, and any change to `ExplainSentiment`, including its adapter.
- **Other commands:** System 1 in `eval` and `optimize` (sentiment is run-only).
- **Statistics:** the independence assumption and the z-versus-t CI. They are documented, not
  changed.
- **Guarding the calibration against a model change.**

## 3. Owner decisions (2026-09-26)

The owner reviewed D1-D8 (§6) and answered "go with recommendations, build PR 1" (2026-09-26).
Each recommendation is the decision:

| # | Decision |
|---|---|
| D1 | Reuse the installed `polarity/classify` student through a declared dependency. |
| D2 | Neutral-only gate with one calibrated value (the mean LLM score of accepted neutral segments) and its spread terms, pinned to the student's `artifact_id` and `ScoreSegment`'s hash. Gate only after the pass test (§2.9). |
| D3 | Gate per corpus, documents only, after the pass test; review and opinion corpora stay off or shadow. |
| D4 | `model_uncertainty` is `null` when no segment was resampled. |
| D5 | PR 1 first: the cache fix and 4-in-flight scoring, in off mode too, with no flag and temperature 0.7, plus review items 7 and 8. |
| D6 | The default `--resample` is 1. |
| D7 | Calibrate on the corpus the owner means to gate; without one, about 110 full Wikipedia articles as a stand-in, plus the fresh-500 review run. |
| D8 | PR 1 is released as 0.6.0. |

## 4. Delivery

1. **PR 1 (no laya dependency), released as 0.6.0 (D8):**
   - the cache fix: rollout ids 0..r−1 at 0.7 for every r;
   - the adapter without JSON fallback, and parse failures as missing samples;
   - the process-wide bound of 4 in flight, and scoring each distinct segment once;
   - pooled `model_uncertainty`, `n_resampled`, `n_samples`, and the default `--resample` (D6);
   - their tests, and the CHANGELOG;
   - the measured t and S (§2.8) in the PR description.
2. **PR 2: System 1 for sentiment, shipped uncalibrated (gate runs as shadow):**
   - the `Student` extraction, with the predict lock;
   - the borrowed student and the calibration guard;
   - the neutral-only gate and the shadow log;
   - the student term in `summarize`;
   - the JSON changes, and docs.
3. **Lab, then PR 3:**
   - count the segments;
   - the document run and the review run, in shadow mode at r = 3;
   - the analysis and the pass test;
   - a PR that pins `NEUTRAL`, with the script.

   After that, the owner enables `system1_mode.sentiment = "gate"` for the corpora that passed.

## 5. Critique disposition

No point was rejected. Where a point offered a choice, the choice is named.

| # | Point | Applied in |
|---|---|---|
| 1 | The certificate does not cover sentiment's segments; the "at least the certified precision" argument; D2's lab time | §2.2 (argument deleted), §2.9 (shadow runs on the target corpus and on reviews, the written pass test, lab time from real segment counts: 1442 segments, about 62 min). The runs use `run sentiment --jsonl`: the same module and segmenter, with one student load. |
| 2 | The level mapping is biased by construction | §2.1: the top-label mean, with n and SE; the "< 30 lines" rule dropped. Least squares is moot with one label. |
| 3 | Gate only on `neutral` in PR 2 | §2.2. Checked on the fresh-500 log first. One premise is corrected: on opinion text the neutral acceptances are the *weakest* slice (19/22), not the strongest. The recommendation stands: it costs nothing on documents, leaves one value to calibrate, and cuts opinion coverage to 5%. |
| 4 | Student scores enter the t-test without error | §2.1: the (n_student/n)·σ² term, with σ²(r) = σ_b² + σ_w²/r, so one r = 3 calibration serves the r = 1 default; the §2.10 test; manual notes on independence and z versus t. |
| 5 | The calibration is not tied to what it measured | §2.1 (the pinned record), §2.4 (the guard degrades gate to shadow; PR 2 ships none), §2.10 (the hash test). |
| 6 | The shadow line cannot support calibration | §2.4: the proposed fields, plus `doc_sha256`, so that a run joins to a corpus sidecar (`run_id` is random and cannot join). Bands are derived offline. |
| 7 | The wall-clock table understates the fix and rests on unmeasured numbers | §2.8 (like-for-like 20 s against 57 s; S caveats and an S = 1 column; L = 1-3 s; measurement before PR 1 merges), D6. |
| 8 | PR 1 triples the calls that can abort a run | §2.6: ChatAdapter without fallback, parse failures as missing samples, the 0.0 fallback removed. |
| 9 | `model_uncertainty` changes meaning | §2.6 (rollout ids 0..r−1 at 0.7 for every r; the DSPy wording), §2.1 (pooled), §2.7 (`n_resampled`; gate mode not comparable). |
| 10 | Concurrency details | §2.5: a process-wide semaphore, which also covers the explain call; a lock rather than `intra_op_num_threads`; distinct segments scored once; lazy imports (§2.4). |
| 11 | The test plan | §2.10: `install()` moved, τ = 0 through `install`, DummyLM list and dict rules, the Barrier test, and the four missing tests. |
| 12 | Compatibility and naming | §2.7: 0.6.0 (D8), "LLM score samples used", `student: {label, confidence}` nested rather than renamed. |
| 13 | What to cut | Expected value, the 4-level table, the "< 30" rule, `llm_band`/`agree` and the 160-character quotes are gone. The borrowed-student branch stays small. The `Student` move, shadow before gate and the three-PR delivery are kept. |

## 6. Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| D2 (redone) | How a student segment is scored, and when it may gate | **Neutral only, one calibrated value.** The student scores a segment only when its top label is neutral at ≥ τ, as `NEUTRAL.level` (the mean LLM score of such segments). σ_b and σ_w enter the standard error. All of it is pinned to the student's `artifact_id` and `ScoreSegment`'s hash. Pass test before gate: a CP lower bound of band agreement ≥ 0.90, and a 95th percentile of the per-document mean drift ≤ 0.5 over ≥ 100 documents. |
| D3 (redone) | Where gate may be used | **Per corpus, documents only, after the pass test.** Review and opinion corpora stay off or shadow, as the owner's approval says. The fresh-500 review run records how much a review that goes through the gate by mistake would shift. A call that mixes sources counts as its most opinionated source. |
| D6 (new) | The default `--resample` | **1.** It is the fastest honest choice: 20 s against 57 s for 24 segments. Uncertainty is `null` by default, and `--resample 3` measures it on request. It has been fake since release without anyone noticing. |
| D7 (new) | The target document corpus, and lab time for the two shadow runs | **The corpus the owner means to gate.** If none is ready, use about 110 full Wikipedia articles (en/de/hr, the same pinned dump as fresh-500, disjoint from it) as a stand-in; that certifies encyclopedic documents only. Budget at t = 2 s: about 1.7 h for the documents (4.5 h at S = 1), plus about 1 h for fresh-500 (2.7 h). PR 1's measured t and S and the `split_segments` counts replace these estimates. |
| D8 (new) | The version of PR 1 | **0.6.0, not 0.5.3.** It changes every user's scores, the default `--resample` and a JSON type. The CHANGELOG says that scores differ from 0.5.x. |
