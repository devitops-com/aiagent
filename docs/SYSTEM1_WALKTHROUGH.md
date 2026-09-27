# System 1 walkthrough: teacher, student, shadow, gate

**Status:** 2026-09-27. Phases 1-9 are **DONE**. Phase 10, the confirmation run of owner decision
D9, is **IN PROGRESS**: its corpus is built, its analysis pre-registered, and the lab run started
at 11:32 UTC. Phase 11 is **NEXT** and depends on phase 10's verdict. Phases 9-11 use PR 3 of the
sentiment design (its §4): the calibration script, `--pin-tau`, the calibration's own τ and the
model guard. PR 3 is merged (#23, 2026-09-27) but not released, so no aiagent release has them
yet: use a checkout of `main`.

This is the record of the first System 1 campaign, end to end, as it ran in the devai lab: the
`polarity` student (distilled, shipped, installed, shadowed) and the calibration that lets
`sentiment` use it. It gives every command, what came out, how long it took and what was decided,
so that an operator can repeat it. The command reference is the [user manual](USER_MANUAL.md)
(`distill`, `sentiment`, settings). The reasons behind each step are in the design docs,
[laya-system1-distillation.md](design/laya-system1-distillation.md) and
[sentiment-system1.md](design/sentiment-system1.md).

**Conventions.**
- Times are UTC. Commands run in a devai lab container, in `bash`.
- `<pilot dir>` is the host directory holding the pilot corpora and lab runbooks. It is not in
  this repository: the corpora carry CC BY-SA text. The sentiment runs mounted it read-write at
  `/laya-pilot`; the polarity pilot copied its corpus into `~/phase4/`.
- `$DS` is a dataset id (`ds-<12 hex>`), `$JOB` a job id, which is also the run id
  (`ftjob-<24 hex>`).
- Excerpts are real output, shortened where marked `…`, with the lab home written as `~`.

## 1. The flow

```
 corpus (JSONL, one document per line)
      │  aiagent distill label           teacher: 27B LLM on :11437, k samples per segment
      ▼
 /laya/inbox/ds-<sha12>/                 the dataset: train / calib / held-out / pool
      │  aiagent distill train           the router swaps the teacher out for the trainer
      ▼
 /laya/runs/ftjob-<24 hex>/              devai's trainer (:11438, GPU): ONNX student
      │  aiagent distill eval            CPU, onnxruntime: τ fitted on calib, certified on held-out
      ├── repair ─▶ aiagent distill repair ─▶ train again (at most 3 rounds)
      ├── stop   ─▶ the campaign ends
      ▼  ship
 aiagent distill install                 artifacts_dir/system1/skills/<skill>/<predictor>/
      │  system1_mode: off ─▶ shadow     the LLM answers, the student is logged beside it
      ▼
 gate                                    the student answers when confidence ≥ τ, else the LLM
```

`sentiment` has no student of its own. It borrows the installed `polarity` student for neutral
segments only, and needs one more loop before its gate may run:

```
 installed polarity student
      │  aiagent run sentiment --jsonl, shadow mode, --resample 3
      ▼
 sentiment shadow log (one line per segment: student verdict + 3 LLM scores)
      │  tools/system1/sentiment_calibration.py
      ▼
 level, spread, pass test, τ sweep ─▶ confirmation run on fresh documents
      │  on PASS
      ▼
 NEUTRAL pinned in core/sentiment.py ─▶ gate for document corpora only
```

The picture of the same flow, with devai's side, is
[laya-system1-architecture.svg](design/laya-system1-architecture.svg).

## 2. Status by phase

| # | Phase | When | Status | Outcome |
|---|---|---|---|---|
| 1 | Plan and corpus | 2026-09-24/25 | DONE | `polarity/classify` qualifies; 2,400 documents |
| 2 | Lab checks, GPU smoke job | 2026-09-25 09:15-09:56 | DONE | 37 s hold for 80 rows |
| 3 | Label round 0 | 2026-09-25 09:56-10:20 | DONE | `ds-e9443d334cb2`, 23.6 min |
| 4 | Train, eval, repair: rounds 0-2 | 2026-09-25 10:20-10:46 | DONE | repair, repair, stop |
| 5 | Ship round 1 at 0.90, install | 2026-09-25 13:54-13:55 | DONE | student `a866e0a4…`, τ 0.8938 |
| 6 | Polarity shadow: 100 texts, batch, fresh-500 | 2026-09-25 13:55-15:18 | DONE | 0.951 agreement on accepted |
| 7 | Sentiment: resample fix, t and S, System 1 in shadow | 2026-09-26 | DONE | 0.6.0 and 0.7.0; t 1.88 s, S 3.18 |
| 8 | Calibration runs 1 and 2 | 2026-09-26 18:24-19:40 | DONE | 31 + 40 min, no failed row |
| 9 | Analysis, owner decision D9 | 2026-09-26/27 | DONE | fails at the installed τ; τ 0.96 |
| 10 | Confirmation run at τ 0.96 | 2026-09-27 | IN PROGRESS | pre-registered, not run |
| 11 | Pin, release, gate for documents | – | NEXT | only on PASS |

## 3. Prerequisites

**The devai lab.** A lab container on `devai-net`, which reaches the router's backends by port:

| Port | Backend | Role here |
|---|---|---|
| 11434 | Ollama | aiagent's default `api_base`. The lab's `default` alias points here (`qwen3.5:9b-q8_0`), **not** at the teacher. |
| 11437 | `vllm-devai` | The teacher: `Qwen3.8-27B-MTP-devai-NVFP4`, context 118784, MTP k=3, `--max-num-seqs 4` (4 calls at a time for everyone). |
| 11438 | `laya-trainer` | devai's fine-tuning jobs API. While a job runs it holds the GPU. |

- **One backend holds the GPU at a time.** A request for another backend swaps: the router's swap
  took 14 s in the GPU smoke test, and the teacher's cold start about 2 min (2 min 5 s there;
  design Appendix A: 119-124 s typical, 268 s the first time).
- **The trainer backend** and its file contract are devai's: see
  [devai's `docs/laya-trainer.md`](https://github.com/ksparavec/devai/blob/main/docs/laya-trainer.md).
- **The `/laya` mount** (`distill_dir`, host `/var/cache/devai/laya`): `base/` with
  `laya-multilingual@55cf4c4ebb4e/` staged read-only, `inbox/` writable by the lab, `datasets/`
  and `runs/` read-only.
- **Free space:** at least 5 GB in `~`. An installed student is about 1.3 GB, and the two
  calibration runs added about 8,300 entries to the DSPy cache (`~/.dspy_cache`). Keep the cache:
  it makes a rerun free and identical.

**aiagent.** Each phase needs the version that added its feature:

| Version (2026) | Adds |
|---|---|
| 0.5.0 (09-25) | `aiagent distill`, the `polarity` skill, `system1_mode` for `run` |
| 0.5.1 (09-25) | `run --jsonl` (one student load per batch); `train` waits 3 min for devai to create the job |
| 0.5.2 (09-25) | `input_sha256` in polarity's shadow lines |
| 0.6.0 (09-26) | real `sentiment` resamples (the cache fix), 4 calls in flight |
| 0.7.0 (09-26) | `system1_mode.sentiment`, sentiment's shadow log |

The sentiment runs used the lab image's own aiagent 0.7.0 (`/usr/local/bin/aiagent`). For
another version, install a release by its tag, `v` included (`0.7.0` without it is a 404):

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/devitops-com/aiagent/releases/download/v0.7.0/install.sh \
  | AIAGENT_VERSION=v0.7.0 sh
type -a aiagent     # the one you mean must come first:
                    # an older ~/.local/bin/aiagent wins over the image's
aiagent version
```

**Settings: pin the teacher.** Keep them in one file and `source` it in every shell. The pilot
kept it in `~/phase4/env.sh`, the sentiment runs in their output directory. `pinned` is the guard
every sentiment run called first:

```bash
mkdir -p ~/phase4
cat > ~/phase4/env.sh <<'EOF'
export AIAGENT_API_BASE=http://devai-router:11437/v1           # the teacher, not the 11434 default
export AIAGENT_MODEL='Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784'  # -> openai/…::mtp::nothink@118784
export AIAGENT_DISTILL_DIR=/laya                               # the default, for the record
export AIAGENT_TRAINER_API_BASE=http://devai-router:11438/v1   # the default, for the record
export LITELLM_LOCAL_MODEL_COST_MAP=True                       # 0.5.0 only; later versions set it
# CONTEXT (every lab launcher injects one) overrides the @118784 above; the others change results.
unset AIAGENT_CONTEXT_TOKENS AIAGENT_CONTEXT CONTEXT AIAGENT_SYSTEM1_MODE AIAGENT_SYSTEM1_MIN_CONF \
      AIAGENT_CACHE AIAGENT_DEFAULT_REASONING
pinned() {   # refuse to run unless aiagent is pinned to the teacher, with no context override
  aiagent config show --json | python3 -c 'import json, sys
c = json.load(sys.stdin)
want = ("http://devai-router:11437/v1", "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784", None)
sys.exit((c["api_base"], c["model"], c["context_tokens"]) != want)' \
  && aiagent models list \
     | grep -Eq '^  default +-> openai/Qwen3\.8-27B-MTP-devai-NVFP4::mtp::nothink@118784$' \
  || { echo "NOT PINNED to the teacher: re-source env.sh" >&2; return 1; }
}
EOF
source ~/phase4/env.sh
aiagent config show          # api_base …:11437/v1, model as above, context_tokens = None
aiagent doctor               # status : ok; models lists Qwen3.8-27B-MTP-devai-NVFP4
aiagent models list          # default -> openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784
curl -s -m 10 --noproxy '*' http://devai-router:11437/health; echo
curl -s -m 10 --noproxy '*' http://devai-router:11438/health; echo
```

- **Why pin.** The `default ->` line of `models list` is the exact string `label` records in the
  dataset, `train --wait` warms back up after training, `repair` relabels with, and the sentiment
  calibration is valid for. It must not change during a campaign.
- **The teacher's `/health`** must read `"backend":"vllm-devai"`,
  `"current_model":"Qwen3.8-27B-MTP-devai-NVFP4"`, `"current_context":118784` and
  `"current_spec":"mtp/k=3"`. Record it: after each training job it must read the same.
- **The trainer's `/health`** must show no job (`"running":false`, or `job_runner` not `busy`).
- **`models list` does not apply `context_tokens`.** Its `default ->` line matches the recorded
  string only while `config show` prints `context_tokens = None`. `pinned` checks both.

## 4. Phase 1: plan and corpus (DONE, 2026-09-24/25)

**Plan.** No LLM, no GPU:

```bash
aiagent distill plan polarity
```

```
skill     : polarity (builtin)
predictor : classify
  qualifies      : yes
  derive_version : 1
  input          : text
  signature      : 19c40da89689
  question set   : 525e19edf9d3
  skill source   : 251d08756fb0
  question polarity choice [negative, neutral, mixed, positive]
    Overall sentiment polarity of the passage.
held-out  : certifying 0.95 at α=0.05 needs ≥ 59 accepted held-out rows, all correct (≈181 if the student is 98% right, ≈361 at 97%): label about 2,000-3,000 segments
```

**The corpus,** `<pilot dir>/corpus.jsonl`, one `{"text": "…"}` per line:
- **2,400 documents in 7 languages:** en 820, hr 540, de 490, pl 300, sl 150, sr 89, bs 11.
- **Sources:** clothing reviews (en), Allegro reviews (pl), GoEmotions (en), SB10k tweets (de),
  ParlaSent parliament sentences (hr, sr/bs, sl) and Wikipedia leads (en/de/hr), all openly
  licensed and stratified by their own labels, so that `mixed` is not starved.
- **Short-form:** 20-988 characters, median 145. Every document fits the student whole (the
  longest uses 262 of 1,002 state tokens), so 2,400 documents are 2,400 segments.
- **Splits** by `sha256(group_id) mod 100`: train 1,448, calib 277, held-out 334, pool 341.
  Held-out clears `label`'s 300-row warning and the 59-row floor.
- The sidecar `corpus.meta.jsonl` (source, language, source label) is for spot checks only. It has
  no `text`, and `label` refuses it.

**Size the corpus before labelling.** Calib and held-out are frozen across rounds. To certify 0.95
at α = 0.05 a student that is 98% right needs about 181 accepted held-out rows, so it must accept
more than half of this corpus's 334. That turned out to be the binding limit (phase 4).

## 5. Phase 2: lab checks and the GPU smoke job (DONE, 2026-09-25)

**The GPU smoke job** (devai's side, 09:15:43-09:16:20): `ftjob-f91b6d747c913af35c209ad9`, 80 rows
(50 train, 10 calib, 12 held-out, 8 pool), 4 epochs in `lean` memory mode. Its hold was 37 s
(validating 5.5 s, training 4.1 s, exporting 12.2 s, calibrating 1.8 s, parity 4.9 s, packaging
2.1 s) and its manifest's peak VRAM 3.17 GiB (devai observed about 3.6 GiB). It proved the path:
router swap, trainer, volume, ONNX export, parity.

**Pre-flight** (09:56:11-09:56:14): `version`, `config show`, `doctor`, `models list`, both
`/health` calls, `ls -la /laya/{base,inbox,datasets,runs}`, `test -w /laya/inbox`, `df -h`,
`distill plan polarity`, then one `aiagent run polarity --text "…"`. That last call proves the
teacher answers, and warms it.

## 6. Phase 3: label (DONE, 2026-09-25 09:56-10:20)

The teacher must stay resident for the whole run: a request for another model swaps it out.

```bash
source ~/phase4/env.sh
CORPUS=~/phase4/corpus.jsonl
sha256sum "$CORPUS"    # f38f91e68a12b735aba583a9609df7f4905d884474a284217d699511d022788c
aiagent distill label polarity --jsonl "$CORPUS" --teacher "$AIAGENT_MODEL" \
  --k 3 --concurrency 4 --json > ~/phase4/label-r0.json 2> ~/phase4/label-r0.log
DS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"])' ~/phase4/label-r0.json)
```

- **k = 3,** not the default 8. laya's own recipe used 3, and a smaller k buys more documents per
  teacher hour.
- **Progress** goes to stderr every 25 segments: `labeled 25/2059`, … The total is train + calib +
  held-out; the pool is not labelled.

**Result** (`label-r0.json`):

```json
{"dataset": "ds-e9443d334cb2", "path": "/laya/inbox/ds-e9443d334cb2",
 "documents": 2400, "segments": 2400, "duplicates": 0,
 "counts": {"train": 1448, "calib": 277, "heldout": 334, "pool": 341},
 "label_histogram": {"polarity": {"negative": 608, "neutral": 644, "mixed": 425, "positive": 382}},
 "parse_failures": {"polarity": 0}, "unlabeled": 0, "warnings": [],
 "next": "aiagent distill train ds-e9443d334cb2"}
```

- **Time:** 23.6 min (09:56:32-10:20:07) for 2,059 segments × 3 = 6,177 teacher calls, 4.4 calls
  per second at 4 in flight.
- **Pace:** 25 segments every 10-11 s at first, every 17-18 s after 1,200; the last 9 segments took
  5 min. Estimate the ETA from recent progress lines, never from the first one.
- **What to look at:** parse failures (above about 1% of calls, check the `::nothink`), `unlabeled`
  rows, and a starved label (`mixed` under about 5%). All were clean.
- **Spot-check the teacher** before spending a GPU window: print 20 held-out rows of
  `/laya/inbox/$DS/heldout.jsonl` (`gold.polarity.label`, `probabilities`, the text). The gate
  certifies agreement with the teacher, not correctness.

## 7. Phase 4: train, eval and repair, three rounds (DONE, 2026-09-25 10:20-10:46)

Each round is one GPU window: the teacher is swapped out for the trainer and back.

**Train and follow it** (terminal A), with a hold monitor beside it (terminal B):

```bash
aiagent distill train "$DS" --wait 2>&1 | tee ~/phase4/train-r0.log
JOB=$(grep -o -m1 'ftjob-[0-9a-f]*' ~/phase4/train-r0.log)
aiagent distill status "$JOB" --events 100 --json > ~/phase4/status-r0.json
```

```bash
while :; do
  date -u +%FT%TZ; curl -s -m 10 --noproxy '*' http://devai-router:11438/health; echo; sleep 30
done
```

`train --wait` polls the volume every 30 s, and after the job warms the teacher up with the
dataset's exact teacher string (`Reply with OK.`, waiting out the router's 503s):

```
ftjob-6af5edfd74fc76e4556837b9: validating_files (volume)
ftjob-6af5edfd74fc76e4556837b9: running (volume)
…
ftjob-6af5edfd74fc76e4556837b9: succeeded (volume)
job       : ftjob-6af5edfd74fc76e4556837b9
status    : succeeded
source    : volume
model     : laya-multilingual
dataset   : ds-e9443d334cb2
student   : ft:laya-multilingual:devai:polarity-classify:6af5edfd74fc
error     : -
events    :
  2026-09-25T10:21:33Z info Phase: exporting
  …
  2026-09-25T10:22:50Z info The job has successfully completed
warm-up   : openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784
next      : aiagent distill eval ftjob-6af5edfd74fc76e4556837b9
```

- **The hold.** While the job runs, the trainer's `/health` has `"job_runner": {"status": "busy",
  "phase": "training", "hold_until": …}`, and every other GPU request gets
  `HTTP/1.1 503`, `Retry-After: 30`, `{"error":{"code":"gpu_held_by_job", …}}`.
- **What a job costs** (from `/laya/runs/$JOB/job.json` and `manifest.json`, `lean`, batch 32,
  micro-batch 8, 4 epochs):

  | Round | Train rows | Hold (created → finished) | Training | Parity check | Peak VRAM | Job end → teacher back |
  |---|---|---|---|---|---|---|
  | r0 | 1,448 | 123 s | 38.9 s | 45.2 s | 3.21 GiB | 2 min 35 s |
  | r1 | 1,704 | 136 s | 45.2 s | 49.6 s | 3.21 GiB | 2 min 24 s |
  | r2 | 1,789 | 140 s | 47.3 s | 47.3 s | 3.20 GiB | 2 min 20 s |

  Parity was 334/334 argmax agreement in every round (max probability difference ≤ 1.4e-5), and
  the fitted temperature about 2.1-2.3.
- **`LAYA_MAX_HOLD_S`** (devai's cap on a hold) was 7200 s during the pilot. After these
  measurements the owner set it to 900 s on devai's side.
- **Fast mode** (`memory_mode` `fast`, a direct POST, since `train` cannot request it): 126 s, peak
  6.7 GiB, training 39.4 s against 38.9 s in `lean`. No gain; `lean` stays. Its run was not
  evaluated (see Troubleshooting).

**Evaluate.** CPU only; neither the teacher nor the GPU is needed:

```bash
aiagent distill eval "$JOB" | tee ~/phase4/eval-r0.txt    # no `set -e`: it exits 3 or 4 on purpose
```

```
question polarity
  n       : calib 277, held-out 334
  accuracy: 0.7305  ECE 0.0556  Brier 0.3539
  coverage: @0.90 0.48, @0.95 0.26, @0.98 0.06
  τ       : 0.9290 (fitted on calib at precision 0.98)
  accepted: 56 (54 correct): precision 0.9643, CP-lower 0.8918, coverage 0.17
  pass    : no
targets   : precision 0.95 (τ fitted at 0.98), α=0.05, min coverage 0.2, ε=0.01, max rounds 3
round     : 0
verdict   : repair — polarity: below the gate; label the pool rows the student is least sure of
report    : ~/.local/share/aiagent/artifacts/system1/evals/ftjob-6af5edfd74fc76e4556837b9.json
next      : aiagent distill repair ftjob-6af5edfd74fc76e4556837b9
```

**Repair** labels the pool rows the student is least sure of into the next round's dataset (same
calib and held-out), with the parent's teacher string, k and temperature; then train and eval
again:

```bash
aiagent distill repair "$JOB" --json > ~/phase4/repair-r1.json   # 256 rows in 2 min; pool left: 85
```

**The three rounds,** with the defaults in every round (target 0.95, fit 0.98, α 0.05, min
coverage 0.2):

| Round | Dataset | Job | Accuracy | ECE | τ | Accepted (correct) | CP lower | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| r0 | `ds-e9443d334cb2` | `ftjob-6af5edfd74fc76e4556837b9` | 0.7305 | 0.056 | 0.9290 | 56 (54) | 0.8918 | 0.17 | repair |
| r1 | `ds-57c9c09510f6` (+256) | `ftjob-d0e411c9b85d1168ff7ea1fa` | 0.7425 | 0.035 | 0.8938 | 84 (81) | 0.9103 | 0.25 | repair |
| r2 | `ds-4f0ddf475621` (+85) | `ftjob-a40f596f637ba6c331490aea` | 0.7545 | 0.076 | 0.9568 | 44 (44) | 0.9342 | 0.13 | stop |

- r2 stopped because no rounds were left (`stop — no rounds left (round 3 of 3)`), with the pool
  exhausted too.
- **Why no ship at 0.95.** The student agrees with the teacher on 73-75% of held-out and is well
  calibrated (ECE 0.03-0.08), but 334 held-out rows are too few: r2 was right on all 44 rows it
  accepted and still bounds at 0.934.
- **Wall-clock:** from the first label call to r2's verdict, 44 min; three GPU windows, six swaps.

## 8. Phase 5: ship round 1 at 0.90, install (DONE, 2026-09-25 13:54-13:55)

**Decision (owner, 2026-09-25):** ship r1 at a certified precision of 0.90 instead of 0.95. A
bigger-corpus campaign at 0.95 may follow later.

```bash
aiagent distill eval ftjob-d0e411c9b85d1168ff7ea1fa --target-precision 0.90
aiagent distill install ftjob-d0e411c9b85d1168ff7ea1fa | tee ~/phase4/install.txt
```

- **eval:** the same τ 0.8938 and 84 accepted (81 correct), CP lower bound 0.9103 ≥ 0.90,
  coverage 0.25 ≥ 0.20: `ship`, "every question certifies precision ≥ 0.9 at α=0.05 with coverage
  ≥ 0.2". A ship is checked before the round limit, so a re-evaluated earlier round can still ship.
- **install:**

  ```
  installed : ~/.local/share/aiagent/artifacts/system1/skills/polarity/classify/a866e0a4734f66ddb975ac1d2e41780c8961913cf6fa738ef53b6bc7b843e273
  artifact  : a866e0a4734f66ddb975ac1d2e41780c8961913cf6fa738ef53b6bc7b843e273
  skill     : polarity/classify
  thresholds: polarity τ=0.8938 (certified precision 0.9)
  enable    : AIAGENT_SYSTEM1_MODE='{"polarity":"shadow"}'
  …
  ```

Installing switches nothing on: `system1_mode` does, per skill.

## 9. Phase 6: polarity in shadow (DONE, 2026-09-25 13:55-15:18)

Keep the teacher pinned, so that the LLM in shadow mode is the teacher the student learned from.

**Three shadow runs, in order:**

1. **100 one-shot runs** (0.5.0, 13:55-14:01) on the leftover pool texts of the last dataset:

   ```bash
   AIAGENT_SYSTEM1_MODE='{"polarity":"shadow"}' aiagent run polarity --text "…"   # once per text
   ```

   Agreement 0.94 on all rows and 69/69 on the rows the gate would have accepted, but a
   `would_accept` share of 0.69 against eval's coverage of 0.25: leftover pool texts are the
   ones the student is sure of, so the sample was biased. Each call took 3.04 s in shadow against
   2.05 s off, the student's load (about 0.85 s) dominating its `student_ms`.
2. **The same 100 texts in one process** (0.5.1, `run --jsonl`, 14:45-14:46): 28.2 s in shadow
   against 18.1 s off, for all 100.
3. **fresh-500** (0.5.2, 15:17-15:18): 500 new texts from the pilot's sources, disjoint from the
   corpus, with the pilot's source and language mix but each source's natural label mix
   (`<pilot dir>/fresh500.jsonl`, described in `FRESH.md` there):

   ```bash
   AIAGENT_SYSTEM1_MODE='{"polarity":"shadow"}' aiagent run polarity --jsonl fresh500.jsonl \
     > fresh-shadow-out.jsonl
   ```

**fresh-500 results.** The log's `input_sha256` joins each line to the sidecar's source and
language:

| Slice | Texts | `would_accept` | Agreement on accepted | CP lower (α 0.05) |
|---|---|---|---|---|
| All | 500 | 123 (24.6%) | 117/123 = 0.951 | 0.906 |
| Opinion text (no Wikipedia) | 447 | 75 (16.8%) | 69/75 = 0.920 | 0.848 |
| Wikipedia (en/de/hr) | 53 | 48 (84-95% per language) | 48/48 | – |

- 97.9 s for all 500 texts, 196 ms each including the teacher. The first line's `student_ms` was
  1,333 ms (the load), then p50 80 ms and p95 278 ms.
- The student's load is about 0.54 s parsing its 34 MB `tokenizer.json` plus about 0.40 s for the
  onnxruntime session, once per process. One-shot runs pay it every time; batches pay it once.
- Polish was the weakest language (overall agreement 0.58).

**Decision (owner, 2026-09-26):** polarity's gate is approved for **document-style** use.
Opinion-only use waits for a better (opinion-weighted, larger) student. Next: System 1 for
`sentiment`. Wall-clock time, not money, is the binding constraint.

**fp16 was rejected in any form** (owner, 2026-09-25). A test changed 0 of 611 answers, but fp16
compute was 11% slower on the lab CPU, and fp16 storage loaded slower with 580-890 MiB more RAM,
because onnxruntime memory-maps fp32 weights and must copy fp16 ones. Only disk use halved. (int8
had already failed parity in the design phase.)

## 10. Phase 7: sentiment's resample fix, t and S, System 1 in shadow (DONE, 2026-09-26)

**The resample/cache fix (0.6.0).** Up to 0.5.x, sentiment's resamples were identical requests,
so DSPy's cache answered all but the first: `model_uncertainty` was always 0, and a 24-segment
text made 25 real calls, not 73. Since 0.6.0 sample *j* is a real call with rollout id *j* at
temperature 0.7, the default is `--resample 1`, and at most 4 LLM calls are in flight per
process.

**t and S, measured** before the fix merged, on the resident teacher, from a throwaway lab
container: `<pilot dir>/measure_ts.py` times 24 warm `ScoreSegment` calls one at a time with the
cache off, then 24 more at 4 in flight.

```bash
source ~/phase4/env.sh
# the bundle's own Python, from the first line of the aiagent launcher
PY=$(head -1 "$(readlink -f "$(command -v aiagent)")" | sed 's/^#!//; s/ .*//')
"$PY" /laya-pilot/measure_ts.py /laya-pilot/fresh500.jsonl 24
```

- **t = 1.88 s** per call (mean; median 1.67 s, p90 2.50 s), on short fresh-500 texts.
- **S = 3.18:** the 24 calls took 14.2 s at 4 in flight against 45.2 s one at a time.
- **So a 24-segment text** takes about 16 s at `--resample 1` and 45 s at `--resample 3`
  (0.5.x: 47 s). The 4 slots are shared by every lab user.

**System 1 for sentiment (0.7.0).** `system1_mode.sentiment` borrows the installed `polarity`
student. In gate mode a segment the student calls `neutral` at confidence ≥ τ gets no LLM call and
scores one calibrated level. 0.7.0 ships no calibration (`NEUTRAL = None`), so gate runs as shadow,
with a warning. Shadow logs one line per segment, without text, to
`<artifacts_dir>/system1/skills/sentiment/score/shadow.jsonl`: the student's label, confidence and
`would_accept` beside the LLM's samples.

## 11. Phase 8: the calibration runs (DONE, 2026-09-26 18:24-19:40)

**Two corpora,** in `<pilot dir>/sentiment/`, one `forward` call per line, `{"text": …,
"resample": 3}`:
- **`docs.jsonl`, run 1:** 120 full Wikipedia articles, 40 each en/de/hr, as the owner's stand-in
  for a document corpus (design D7). The calibration and the pass test.
- **`fresh500-r3.jsonl`, run 2:** the fresh-500 texts with `"resample": 3`. The review run: it
  measures what the gate would do to opinion text.

**Count first.** Segmenting is pure and costs nothing. The teacher calls are 3 per distinct segment
plus one explanation per document:

```bash
"$PY" -I - /laya-pilot/sentiment/docs.jsonl <<'EOF'
import json, sys
from aiagent.core.segment import split_segments
docs = [json.loads(l)["text"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
per_doc = [split_segments(t, max_segments=24) for t in docs]
scored = sum(len(set(s)) for s in per_doc)
positions = sum(map(len, per_doc))
calls = 3 * scored + len(docs)
print(f"documents {len(docs)}  positions {positions}  scored {scored}  calls {calls}")
EOF
```

| Corpus | Documents | Positions | Scored | Teacher calls | Estimate at 0.592 s per call |
|---|---|---|---|---|---|
| `docs.jsonl` | 120 | 1,110 | 1,109 | 3,447 | 34 min |
| `fresh500-r3.jsonl` | 500 | 1,444 | 1,440 | 4,820 | 48 min |

The lab window was booked for 2 h.

**Pre-flight** (18:23:55-18:24:16): the checks of section 3 with `pinned`, then a one-document
shadow run that proves sentiment finds the student, logs, and warms the teacher:

```bash
AIAGENT_SYSTEM1_MODE='{"sentiment":"shadow"}' aiagent run sentiment --json \
  --text "The Danube is the second-longest river in Europe. It flows through ten countries into the Black Sea."
```

The command prints its JSON result (indented; shortened and reflowed here):

```
{…, "n_segments": 2, "polarity": "neutral / mixed",
 "segments": [{…, "score": 0.0, "source": "llm",
               "student": {"confidence": 0.9507, "label": "neutral"}}, …],
 "sentiment": 0.0, …,
 "system1": {"accepted": 2, "artifact_id": "a866e0a4…e273", "coverage": 1.0, "mode": "shadow",
             "student": "polarity/classify", "tau": 0.8938, "too_long": 0}, …}
```

The runbook's checker then compares it with the shadow log: `mode` is `shadow`, `artifact_id` is
the installed student's, and the log has one line per segment (2 and 2).

**One run.** Detached, since a run takes 30-50 min and must survive a lost shell. A small driver
moves any old shadow log aside first and collects this run's log after, so that each run has its
own. `AIAGENT_SYSTEM1_MODE` is set on the command only, so nothing stays in shadow mode:

```bash
cat > ~/phase4/run_shadow.sh <<'EOF'
#!/bin/bash
# run_shadow.sh NAME INPUT: one sentiment shadow run, start to finish
source ~/phase4/env.sh
pinned || exit 1
name=$1 input=$2 out=/laya-pilot/sentiment/out
# the shadow log under the default artifacts_dir
sl=~/.local/share/aiagent/artifacts/system1/skills/sentiment/score/shadow.jsonl
[ -e "$sl" ] && mv "$sl" "$out/aside-$(date -u +%Y%m%dT%H%M%SZ)-shadow.jsonl"
date -u +%FT%TZ > "$out/$name.start"
AIAGENT_SYSTEM1_MODE='{"sentiment":"shadow"}' aiagent run sentiment -v --jsonl "$input" \
  --concurrency 4 > "$out/$name-out.jsonl" 2> "$out/$name.err"
rc=$?
date -u +%FT%TZ > "$out/$name.end"
mv "$sl" "$out/$name-shadow.jsonl"
echo "$rc" > "$out/$name.exit"   # written last: the run is over
EOF
setsid nohup bash ~/phase4/run_shadow.sh docs /laya-pilot/sentiment/docs.jsonl \
  > /laya-pilot/sentiment/out/docs.driver.log 2>&1 < /dev/null &
```

Run 2 is `run_shadow.sh fresh500 /laya-pilot/sentiment/fresh500-r3.jsonl`, started only after
`docs.exit` exists.

- **`-v` prints the composed model string** on stderr, `[-v] skill=sentiment
  model=openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784`, and at the end `[-v]
  elapsed=… calls=…`. It is the record of what the run really scored with.
- **Follow it** by the lines in the live shadow log and the rows in `docs-out.jsonl`; the lab's
  `<pilot dir>/sentiment/check.py progress` prints both with a rate and an ETA.
- **Do not run anything else** that uses sentiment in shadow or gate mode in the same lab home
  meanwhile: it would share the teacher's 4 slots and the one shadow log.

**What came back** (the runs' `.check` summaries):

| | Run 1: `docs` | Run 2: `fresh500` |
|---|---|---|
| Time | 18:24:24-18:55:51, **31 min** | 19:00:26-19:40:48, **40 min** |
| Rows ok / failed | 120 / 0 | 500 / 0 |
| Shadow lines (complete runs) | 1,110 (120) | 1,444 (500) |
| `would_accept` at the installed τ | 699 | 186 |
| Too long for the student | 19 (all Croatian) | 0 |
| LLM samples that did not parse | 2 | 3 |
| `[-v] elapsed`, calls | 1885.26 s, 3,448 | 2419.99 s, 4,834 |
| Pace | about 35 positions per minute | about 36 positions per minute |
| `student_ms` p50 / p95 | 158 / 969 ms | 39 / 138 ms |

Run 1 came to 0.547 s per call at 4 in flight, faster than the 0.592 s estimate. The teacher's
`/health` was the same before and after both runs: no swap.

## 12. Phase 9: the analysis and decision D9 (DONE, 2026-09-26/27)

**The script** is `tools/system1/sentiment_calibration.py`, from PR 3 (merged, not yet released; see
Status). It reads a sentiment shadow log and prints the level, the spread, per-segment coverage, the
pass test, a τ sweep, slices by source and language, and the `NEUTRAL = NeutralCalibration(…)` block
to pin. It runs on the host, from the repository root, with the dev venv (`make dev-install`).
0.7.0's shadow lines carry no model, so `--model` names it:

```bash
S="<pilot dir>/sentiment"
.venv/bin/python tools/system1/sentiment_calibration.py "$S/out/docs-shadow.jsonl" \
  --meta "$S/docs.meta.jsonl" --model openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784
```

**The pass test** (design §2.9, owner decision D2), over the accepted segments:
- **(a) band agreement:** a segment agrees when the mean ȳ of its LLM samples lies in (−2, +2);
  the one-sided Clopper-Pearson lower bound (α = 0.05) of agreement must be ≥ 0.90;
- **(b) mean drift:** per document with an accepted segment, Δ = the document's mean with LEVEL in
  place of ȳ on accepted segments, minus its off-mode mean; over ≥ 100 documents the 95th
  percentile of |Δ| must be ≤ 0.5.

**Run 1 at the installed τ 0.8938:**

```
Accepted neutral segments at the installed τ
  LEVEL -0.0057 over n = 699, SE 0.0340
  σ_w 0.5218 (pooled over 699 segments with ≥ 2 LLM samples), σ_b 0.8478 (…)
  coverage 699/1110 = 63.0%, too_long 19/1110 = 1.7%

Pass test at the installed τ
  (a) band agreement: 672/699 in (-2, +2), CP lower bound 0.9471 (α = 0.05, need ≥ 0.90): PASS
  (b) mean drift: 119 documents (need ≥ 100), p95 |Δ| 0.5908 (need ≤ 0.5), mean Δ -0.0066, …: FAIL
```

(b) fails because of real out-of-band disagreements: segments the student calls neutral whose LLM
mean lies outside (−2, +2). The sweep shows where it would pass:

```
  τ              n    LEVEL  coverage  (a) agree   bound        (b) docs  p95 |Δ|   mean Δ
  installed    699  -0.0057     63.0%    672/699  0.9471  PASS       119   0.5908  -0.0066  FAIL
  0.90         689   0.0097     62.1%    665/689  0.9514  PASS       119   0.5100  -0.0056  FAIL
  0.92         647   0.0026     58.3%    625/647  0.9518  PASS       119   0.5000  -0.0081  PASS
  0.94         581  -0.0264     52.3%    564/581  0.9564  PASS       116   0.4241  -0.0121  PASS
  0.96         472  -0.0205     42.5%    462/472  0.9643  PASS       114   0.3400  -0.0086  PASS
  0.98         252   0.0000     22.7%    250/252  0.9752  PASS        96   0.1000   0.0008  FAIL
```

At 0.98, (b) fails on its count (96 documents), not on drift.

**Run 2, the review run,** is analysed at run 1's level, not its own: −0.01, run 1's LEVEL at the
installed τ, rounded. It fails, as expected for opinion text:

```bash
.venv/bin/python tools/system1/sentiment_calibration.py "$S/out/fresh500-shadow.jsonl" \
  --meta "$S/fresh500-r3.meta.jsonl" --model openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784 \
  --level -0.01
```

```
  coverage 186/1444 = 12.9%, too_long 0/1444 = 0.0%
  (a) band agreement: 173/186 in (-2, +2), CP lower bound 0.8912 (…): FAIL
  (b) mean drift at the given level -0.01: 115 documents (…), p95 |Δ| 1.1144 (need ≤ 0.5), …: FAIL
```

By source, the Wikipedia texts in fresh-500 held (112 of 113 accepted segments in band), while
the tweets and parliament sentences drove the drift (p95 |Δ| 4.34 for `sb10k-de`, 4.32 for
`parlasent-hr`). This is the measurement behind decision D3: gate for document corpora only;
review and opinion corpora stay off or shadow.

**Decision D9 (owner, 2026-09-27): "τ 0.96 + confirm run".**
- The calibration pins its **own** τ, 0.96 (`NeutralCalibration.tau`). In gate mode the student
  takes a neutral segment only at confidence ≥ max(the student's τ, 0.96).
- τ 0.96 was chosen from run 1's own sweep, so before it is pinned a **confirmation run** on 120
  fresh Wikipedia articles, disjoint from run 1, tests the pass test at τ 0.96 with run 1's level
  −0.02, both fixed in advance. (The design doc's record of D9, §3, adds a bootstrap P(pass) of
  0.92 for run 1 at τ 0.96, and 0.97 for a fixed-τ held-out check.)
- D9 is recorded in the design doc (§3) and fixed in the pre-registration,
  `<pilot dir>/sentiment/confirm/PREREG.md`.
- Pin only on PASS. The pin is run 1's, measured at τ 0.96:

  ```bash
  .venv/bin/python tools/system1/sentiment_calibration.py "$S/out/docs-shadow.jsonl" \
    --meta "$S/docs.meta.jsonl" --model openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784 \
    --pin-tau 0.96
  ```

  ```
  Pass test at τ 0.96
    (a) band agreement: 462/472 in (-2, +2), CP lower bound 0.9643 (α = 0.05, need ≥ 0.90): PASS
    (b) mean drift: 114 documents (need ≥ 100), p95 |Δ| 0.3400 (need ≤ 0.5), mean Δ -0.0086, …: PASS
  …
  NEUTRAL: NeutralCalibration | None = NeutralCalibration(
      artifact_id="a866e0a4734f66ddb975ac1d2e41780c8961913cf6fa738ef53b6bc7b843e273",
      score_signature_sha256=(
          "b26dee349163d3219febe678ead8fd8feb1c8e13e3c0469a883f7e3cb7febc96"
      ),
      model="openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink",
      tau=0.96,
      level=-0.02,
      se=0.0332,
      sigma_between=0.6832,
      sigma_within=0.4039,
      n=472,
      measured=(…),
  )
  ```

  At τ 0.96 the student would take 42.5% of the document segments, against 63.0% at its own τ.

**The model guard** (owner, 2026-09-26, in PR 3). `NEUTRAL.model` is the model string the
calibration was measured with, its `@<ctx>` aside. A gate-mode run whose score calls use another
model string (`::think` against `::nothink` included), or no LM, only shadows, with one warning.
PR 3's shadow lines carry the `model` too. A router that serves another model under the same name
is still not detected: recalibrate after changing what the `default` alias serves.

## 13. Phase 10: the confirmation run (IN PROGRESS, 2026-09-27)

**Prepared** (2026-09-27):
- **Corpus** `<pilot dir>/sentiment/confirm/confirm.jsonl`: 120 fresh Wikipedia articles, 40 each
  en/de/hr, built by run 1's builder with a new seed. It excludes run 1's articles, the pilot's
  and fresh-500's by id, URL and title, and every row group they came from; its disjointness
  checks report no failure. 1,051 segment positions, all distinct and all short enough for the
  student: 3,273 teacher calls, about 30 min at run 1's 0.547 s per call. Book 1 h.
- **Pre-registration** `confirm/PREREG.md`, whose sha256 the lab checks before the run starts. It
  fixes the hypothesis, the inputs' sha256, the run's settings (as run 1's: aiagent 0.7.0, the
  student at its installed τ, the pinned teacher, shadow, `--concurrency 4`), the sha256 of the
  analysis script and of the four modules it imports, the command, and the decision rule.
- **Runbook** `confirm/RUNBOOK-CONFIRM.md`: run 1's procedure with its own output directory
  `out-confirm/`, since run 1's `out/` is sealed (`SHA256SUMS`, re-checked before and after).

**The run** is run 1's procedure with the new input: pre-flight with `pinned`, a one-document
shadow run on a new text (run 1's Danube sentence would come from the DSPy cache and warm
nothing), then the detached run, then `.check` and reruns of failed rows only.

**The analysis, once, on the complete log:** the run's log and its reruns' logs joined in order
(the script keeps each document's first complete run), then the pre-registered command:

```bash
S="<pilot dir>/sentiment"; O=$S/out-confirm; A=$S/confirm/analysis; mkdir -p "$A"
for f in "$O/confirm-shadow.jsonl" "$O"/confirm-rerun*-shadow.jsonl; do   # the run, then reruns
  [ -e "$f" ] || continue
  cat "$f"; [ -z "$(tail -c1 "$f")" ] || echo   # after a torn last line, start a new one
done > "$A/confirm-all-shadow.jsonl"
.venv/bin/python tools/system1/sentiment_calibration.py "$A/confirm-all-shadow.jsonl" \
  --meta "$S/confirm/confirm.meta.jsonl" \
  --model openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink \
  --pin-tau 0.96 --level -0.02 --json "$A/confirm-analysis.json"
```

Its last line is `CONFIRMATION at τ 0.96, level -0.02: PASS` (exit 0) or `… : FAIL` (exit 1). The
verdict counts only if all 120 documents were analysed (a teacher that cannot score a segment is
the one allowed exception), the only model on the `[-v]` lines is the pinned one, and the
teacher's `/health` did not change. Fewer than 100 documents in (b) with all 120 analysed is a
FAIL, not a failed run.

## 14. Phase 11: after the verdict (NEXT)

- **On PASS:** paste run 1's `NEUTRAL` block (section 12, generated again with `--measured`
  naming run 1 and the confirmation) into `src/aiagent/core/sentiment.py`, record the figures in
  the design doc (D9) and the CHANGELOG, in a follow-up PR to PR 3, then release. After that the owner enables
  gate, per corpus, for document corpora only:

  ```toml
  [system1_mode]
  sentiment = "gate"   # documents only; reviews and opinion text stay "off" or "shadow"
  ```

- **On FAIL:** nothing is pinned, `NEUTRAL` stays `None`, gate keeps running as shadow, and the
  owner decides with the full analysis output.
- **Later, separately:** an opinion-weighted, larger polarity student (possibly with the deferred
  synthetic augmentation), a campaign at 0.95, and a wider sentiment gate with its own
  calibration. Any new student or `ScoreSegment` change needs a new calibration: the guard shadows
  until one is pinned.

## 15. Troubleshooting: what went wrong for real

Each row was met in these runs or while preparing them. Where a release fixed it, the row says
which.

| Symptom | Cause | What to do |
|---|---|---|
| The first call of a phase takes 2-4.5 min | The teacher's vLLM cold start after a swap (119-124 s typical, 268 s the first time; 2 min 20-35 s from the end of each training job) | Warm it with one small call before a run; book the time in the GPU window. |
| `503`, `Retry-After: 30`, `"code":"gpu_held_by_job"` | A training job holds the GPU; every other backend is refused until the job's record is final | Wait. The warm-up of `train --wait` and `status --wait` waits it out itself (5-minute deadline). A label run that meets it aborts once aiagent's 2 retries are used up, and a sentiment `--jsonl` run fails those rows: re-run the same command (or only the failed rows), and the DSPy cache replays every finished call. |
| The teacher string or context changes; a second cold start after training | `CONTEXT` (injected by every lab launcher) or `AIAGENT_CONTEXT[_TOKENS]` overrides the `@118784` in the model | `unset` all three (section 3). Check `config show` prints `context_tokens = None`; `models list` does not show the override. |
| A sentiment run on the wrong model; the teacher evicted | The lab's `default` alias is `qwen3.5:9b-q8_0` on Ollama, not the teacher | Pin `AIAGENT_API_BASE` and `AIAGENT_MODEL`; start runs through `pinned`; read the `[-v] skill=… model=` line. PR 3's model guard (merged, not yet released) makes gate shadow on another model. |
| `pgrep: command not found` | The lab image has no `pgrep` or `pkill` | List runs from `/proc`: `for p in /proc/[0-9]*; do tr '\0' ' ' 2>/dev/null <"$p/cmdline" \| grep -q 'aiagent run sentiment' && echo "${p#/proc/}"; done`, then `kill <pid>`. |
| An `unreadable` count in the check, a document missing | A killed or interrupted process tore the shadow log's last line | The analysis leaves the line out and names it; rerun that document. When joining a log with its reruns, end each file with a newline first so the next one starts whole. |
| A sentiment run 3× slower than planned | Another client shares the teacher's 4 slots, or a swap | Check `active_reqs` in the teacher's `/health`. One run at a time per lab. |
| `would_accept` far above eval's coverage in shadow | The shadow texts were leftover pool rows, which the student is sure of (0.69 against 0.25) | Shadow on a fresh, representative set (fresh-500). |
| Shadow or gate makes a one-shot `run` slower | Each process loads the student: about 0.54 s tokenizer + 0.40 s onnxruntime | Use `run --jsonl` for batches: 100 texts took 28.2 s in one process. |
| Every command that reaches the LLM starts about 5 s late in the lab (fixed in 0.5.1) | litellm fetched its cost map through pipelock and timed out | `LITELLM_LOCAL_MODEL_COST_MAP=True`; 0.5.1 and later set it. |
| `train` exits 1, `cannot reach the trainer API … timed out` (fixed in 0.5.1) | The first POST includes the teacher's eviction; 0.5.0 waited 30 s | 0.5.1 waits 3 min. If a job appeared (`ls -t /laya/runs \| head -1`), follow it with `distill status <JOB> --wait`; a `409` means the first POST started one. |
| Would fp16 or int8 halve the student? | fp16: 11% slower compute, more RAM (fp32 is memory-mapped); int8 dynamic failed parity (97/133) | Neither is supported; students stay fp32 (about 1.3 GB). |
| An extra training job confuses the next round | `repair`'s ε rule finds "the previous round" through eval reports on the parent dataset | Never `eval` a side job such as the fast-mode measurement; keep every real round's report. |
| A script stops at `distill eval` | `eval` exits 3 (repair) and 4 (stop) on purpose | Never wrap it in `set -e`; branch on the exit code. |

## 16. Timings at a glance

| Step | Measured |
|---|---|
| Teacher cold start | about 2 min (119-124 s typical, 268 s first time) |
| Router swap (smoke test) | 14 s |
| Label, 2,400 short documents, k = 3 | 23.6 min, 6,177 calls, 4.4 calls/s |
| Repair, 256 rows | 2 min |
| Training job (hold), about 1,450-1,790 train rows × 4 epochs | 123-140 s; peak VRAM 3.2 GiB (`lean`) |
| Job end to teacher answering again | 2 min 20-35 s |
| Eval (CPU) | under 1 min |
| Student load, once per process | about 0.85 s (one-shot `student_ms` p50 848 ms; by hand: 0.54 s tokenizer, 0.40 s onnxruntime) |
| Student decision, warm (`student_ms`) | p50 39-158 ms, p95 138-969 ms: fresh-500 80 / 278 ms, sentiment run 1 158 / 969 ms, run 2 39 / 138 ms |
| Polarity shadow, 500 texts in one process | 97.9 s |
| One warm `ScoreSegment` call (t), speedup at 4 in flight (S) | 1.88 s, 3.18 |
| Sentiment shadow, 120 articles at `--resample 3` | 31 min (3,448 calls, 0.547 s each at 4 in flight) |
| Sentiment shadow, fresh-500 at `--resample 3` | 40 min (4,834 calls) |

## 17. Where the numbers come from

- **Polarity pilot, in the lab home** (`~/phase4/`): `label-r0.json` and `.log`,
  `train-r{0,1,2}.log`, `status-r*.json`, `eval-r{0,1,2}.json` and `.txt`, `eval-r1-p090.json`,
  `install.txt`, `hold-probe-r0.txt`, `trainer-health-r*.log`, `shadow-summary.txt`,
  `batch-{off,shadow}.summary`, `fresh-shadow-report.txt`.
- **Training jobs:** `/laya/runs/<job>/job.json` and `manifest.json` (hold, timings, peak VRAM,
  parity, calibration).
- **Corpora and runbooks,** in `<pilot dir>`: `CORPUS.md`, `FRESH.md`, `RUNBOOK.md` (polarity);
  `sentiment/CORPUS.md` and `RUNBOOK.md` (runs 1 and 2), `sentiment/out/*.check` and `*.err`;
  `sentiment/confirm/CORPUS.md`, `PREREG.md` and `RUNBOOK-CONFIRM.md` (the confirmation run).
- **Analysis:** `tools/system1/sentiment_calibration.py` (PR 3, merged, not yet released) on
  `sentiment/out/*-shadow.jsonl`.
- **Not in a lab file:** devai's own observations of the GPU smoke test (the 14 s router swap, the
  2 min 5 s cold start, about 3.6 GiB of VRAM), the owner's `LAYA_MAX_HOLD_S` of 900 s (a devai
  setting), and two measurements made by hand during the pilot: the fp16 test (0 of 611 answers
  changed, 11% slower compute, 580-890 MiB more RAM) and the split of the student's load (0.54 s
  tokenizer, 0.40 s onnxruntime).
- **Design and decisions:** [laya-system1-distillation.md](design/laya-system1-distillation.md)
  (§11, §13, Appendix A) and [sentiment-system1.md](design/sentiment-system1.md) (§2.4, §2.8,
  §2.9, and D1-D9 in §3). The confirmation run's rules are in
  `<pilot dir>/sentiment/confirm/PREREG.md`.
