# aiagent User Manual

A programmatic [DSPy](https://dspy.ai) agent framework focused on **query/prompt
optimization, goal-reaching loops, and autonomous data processing** over local
LLMs. It talks to any OpenAI-compatible router (the
[devai](https://github.com/ksparavec/devai) `devai-router`) and ships as a single
self-extracting, precompiled bundle that includes its own Python.

aiagent is **not a chat UI** — basic chat is a minor convenience feature. The MVP
demo is a self-optimizing expense extractor that lifts `{merchant, date, amount}`
accuracy by compiling few-shot demonstrations from a small labeled dataset.

> Status: early MVP under active construction. See `CHANGELOG.md`.

For installation and a first-run walkthrough, see the
[Quick Start in the README](../README.md#quick-start). This manual documents every
implemented feature in detail.

---

## Table of contents

- [Concepts](#concepts)
- [Command reference](#command-reference)
  - [Global behavior](#global-behavior)
  - [`doctor` — check connectivity](#doctor--check-connectivity)
  - [`config show` — inspect resolved settings](#config-show--inspect-resolved-settings)
  - [`models list` — aliases and advertised models](#models-list--aliases-and-advertised-models)
  - [`skills list` — discovered skills](#skills-list--discovered-skills)
  - [`run` — run a skill once](#run--run-a-skill-once)
  - [`eval` — score a skill over a dev set](#eval--score-a-skill-over-a-dev-set)
  - [`optimize` — compile a skill](#optimize--compile-a-skill)
  - [`chat` — resumable multi-turn Q&A](#chat--resumable-multi-turn-qa)
  - [`shell` — devai agent entrypoint](#shell--devai-agent-entrypoint)
  - [`version`](#version)
- [Configuration](#configuration)
- [Model strings, aliases, and reasoning](#model-strings-aliases-and-reasoning)
- [Verbosity (`-v` / `-vv` / `-vvv`)](#verbosity--v---vv---vvv)
- [The expense-extraction demo](#the-expense-extraction-demo)
- [Datasets (JSONL)](#datasets-jsonl)
- [Metrics](#metrics)
- [Optimizers](#optimizers)
- [Writing your own skill](#writing-your-own-skill)
- [Chat sessions on disk](#chat-sessions-on-disk)
- [Development](#development)
- [Packaging](#packaging)
- [Releasing](#releasing)
- [Deploying as a devai agent](#deploying-as-a-devai-agent)

---

## Concepts

**Skill.** A unit of capability, in the Claude-Code style: a directory containing
a `SKILL.md` manifest (YAML frontmatter + docs) and a Python module exposing a
`build()` factory that returns a DSPy module. Skills are auto-discovered from two
sources — **built-in** skills shipped with aiagent, and **user** skills under
`~/.config/aiagent/skills`. The two ship separately and never merge.

**Pipeline.** A skill's runnable program is a `dspy.Module` (aiagent's base class
is `Pipeline`). The built-in `extract` skill uses a `ChainOfThought` program over
a typed signature.

**Metric.** A callable following DSPy's contract
`metric(example, prediction, trace=None) -> float | bool`. It returns a **float**
fraction when scoring (`trace is None`) and a strict **bool** gate when an
optimizer is bootstrapping demonstrations (`trace is not None`).

**Optimization.** Feeding a skill's training set through a DSPy prompt optimizer
(`BootstrapFewShot` or `MIPROv2`) produces a *compiled program* — the same module
with learned few-shot demonstrations and/or instructions baked in. Compiled
programs are saved as **state-only JSON** and reloaded onto a fresh instance for
reuse.

**Model registry.** A small table mapping friendly aliases (e.g. `default`) to
devai model strings. The single source of truth for composing the router's
control-surface string `openai/<model>::<reasoning>[@<ctx>]`.

**Router.** aiagent never assumes native tool-calling or server JSON mode. It uses
DSPy's default **ChatAdapter**, which is portable across devai's ollama / vLLM /
SGLang backends.

---

## Command reference

Run `aiagent --help` for the top-level command list, or `aiagent <command>
--help` for any single command. Commands that only inspect configuration or
connectivity (`doctor`, `config`, `models`, `skills`, `version`, `--help`) never
import DSPy, so they start instantly. The commands that run a model (`run`,
`eval`, `optimize`, `chat`) import DSPy lazily on first use.

### Global behavior

- **Usage errors** (unknown command, missing option, bad flag) print the relevant
  help text and exit with code `2`.
- **Runtime errors** print `error: <message>` to stderr and exit `1`.
- **`--json`** on any command that supports it emits a machine-readable object on
  stdout; all human progress/verbosity goes to stderr, so `--json` output stays
  clean for piping.

### `doctor` — check connectivity

Pre-flight check. Online, it probes the router's `GET /health` and
`GET /v1/models`; offline, it validates configuration only (no network).

```bash
aiagent doctor                 # full check against the router
aiagent doctor --offline       # config sanity only (build / CI, no router)
aiagent doctor --timeout 30    # override per-probe timeout (seconds)
aiagent doctor --json          # machine-readable report
```

| Option | Description |
|--------|-------------|
| `--offline`, `-O` | Skip all network; validate config only. |
| `--timeout <s>` | Per-probe timeout in seconds (default: configured `request_timeout_s`). |
| `--json` | Emit a JSON report. |

**Exit codes:** `0` healthy · `1` unreachable or degraded · `2` config error.

The report includes `api_base`, the effective `model` (or default alias), health
status, the models endpoint status, and the list of advertised model IDs. If the
router is unreachable it prints a cold-start hint: devai's vLLM/SGLang backends
are recreated on demand and the first request to a cold backend can take minutes —
raise `AIAGENT_REQUEST_TIMEOUT` (seconds) if a call appears to hang.

### `config show` — inspect resolved settings

Prints the fully resolved settings after applying the precedence chain (env → TOML
→ devai-env → defaults). The `api_key` is always masked.

```bash
aiagent config show
aiagent config show --json
```

### `models list` — aliases and advertised models

Shows the registry aliases (each expanded to its composed model string) and, when
the router is reachable, the models it actually advertises. The registry default
is a **placeholder** — use this command to confirm the real served tag.

```bash
aiagent models list
aiagent models list --json
```

### `skills list` — discovered skills

Lists discovered skills with their source, description, and preferred model alias.
Manifests that fail to parse are skipped with a warning rather than aborting.

```bash
aiagent skills list                    # built-in + user
aiagent skills list --source builtin   # built-in only
aiagent skills list --source user      # user skills only
aiagent skills list --json
```

| Option | Description |
|--------|-------------|
| `--source <all\|builtin\|user>` | Filter by skill source (default `all`). |
| `--json` | Emit JSON, including any skipped-manifest errors. |

### `run` — run a skill once

Runs a skill a single time on one input and prints its prediction. Requires a
reachable router.

```bash
aiagent run extract --text "Lunch at Chipotle $12.50 on 3/4/2025"
aiagent run extract --input inputs.json          # JSON object of named inputs
aiagent run extract --text "..." --json          # JSON prediction
aiagent run "extract expense fields" --route     # treat SKILL as free text
aiagent run extract --text "..." --model default -v
```

| Option | Description |
|--------|-------------|
| `--text`, `-t <str>` | Input text (bound to the skill's `text` input). |
| `--input`, `-i <path>` | A JSON **object** of named inputs (alternative to `--text`). |
| `--model <alias\|name>` | Override the model for this run. |
| `--route` | Treat the positional argument as free text and route it to a skill. |
| `--json` | Emit the prediction as JSON. |
| `-v` / `-vv` / `-vvv` | Increase verbosity (see [Verbosity](#verbosity--v---vv---vvv)). |

Provide exactly one of `--text` or `--input`. With `--route`, the positional
argument is matched against skills by exact name first, then by keyword overlap
over each skill's name + description (see [routing](#routing)).

### `sentiment` — analyze sentiment of data sources

Scores one or more data sources on a **−10** (very negative) … **+10** (very
positive) scale and reports statistical properties plus a plain-language
explanation. Works out of the box — just supply the sources. Requires a
reachable router.

```bash
aiagent sentiment --text "The rollout was flawless and the team is thrilled."
aiagent sentiment --file review.txt --file notes.md         # local files
aiagent sentiment --url https://example.com/post --json     # fetched via proxy
aiagent sentiment --file report.pdf --url https://... -t "extra note"  # mix + repeat
```

| Option | Description |
|--------|-------------|
| `--text`, `-t <str>` | Raw text to analyze (repeatable). |
| `--file`, `-f <path>` | Local file: `.txt`/`.md`/`.html`/`.pdf` (repeatable). |
| `--url`, `-u <url>` | URL to fetch and analyze via the proxy (repeatable). |
| `--model <alias\|name>` | Model override. |
| `--resample <n>` | LM samples per segment for the uncertainty estimate (default 3). |
| `--max-segments <n>` | Cap on analyzed segments; content is merged, never dropped (default 24). |
| `--json` | Emit the full result (scores, stats, per-segment breakdown, sources) as JSON. |
| `-v` / `-vv` / `-vvv` | Increase verbosity. |

Supply at least one `--text`/`--file`/`--url` (repeatable and mixable). The
corpus is split into segments and each is scored several times; the reported
fields are:

- **sentiment** — mean score on the −10..+10 scale, with a qualitative polarity.
- **volatility** — standard deviation of sentiment across segments (how mixed).
- **model uncertainty** — mean self-disagreement across the resamples.
- **significance** — a one-sample t-test of the segment scores against neutral
  (0): a t-statistic, two-sided p-value, and a confidence label; plus an
  approximate 95% confidence interval.

URLs egress through the configured [`proxy_url`](#settings) (devai's pipelock);
the skill is run-only, so `aiagent run sentiment --text "..."` also works for a
single text blob.

### `eval` — score a skill over a dev set

Evaluates a skill across its dev set and reports the aggregate metric score.
Optionally loads a compiled program first, so you can measure the lift from
optimization.

```bash
aiagent eval extract                                    # baseline
aiagent eval extract --compiled compiled/extract.json   # score a compiled program
aiagent eval extract --devset my_dev.jsonl              # custom dev set
aiagent eval extract --num-threads 8 --json
```

| Option | Description |
|--------|-------------|
| `--devset <path>` | Dev JSONL override (defaults to the skill's declared `devset`). |
| `--compiled <path>` | Load a compiled program's state before evaluating. |
| `--model <alias\|name>` | Model override. |
| `--num-threads <n>` | Parallel evaluation threads (default: configured `num_threads`). |
| `--json` | Emit `{skill, score, n}` as JSON. |
| `-v` / `-vv` / `-vvv` | Increase verbosity. |

Output reports the skill name, the number of examples `n`, and the `score`
(fraction of fields correct, `0.00`–`1.00`). If the skill declares no dev set and
none is passed, it errors.

### `optimize` — compile a skill

Compiles a skill against its metric using a DSPy prompt optimizer, optionally
measuring before/after scores on a dev set and saving the compiled program.

```bash
aiagent optimize extract --out compiled/extract.json                 # default: bootstrap
aiagent optimize extract --optimizer mipro --out compiled/extract.json
aiagent optimize extract --trainset my_train.jsonl --devset my_dev.jsonl
```

| Option | Description |
|--------|-------------|
| `--trainset <path>` | Train JSONL (defaults to the skill's declared `trainset`; required if absent). |
| `--devset <path>` | Dev JSONL for baseline/after measurement (defaults to the skill's `devset`). |
| `--optimizer <bootstrap\|mipro>` | Which optimizer to use (default `bootstrap`). |
| `--out <path>` | Save the compiled program as state-only JSON (parent dirs created). |
| `--model <alias\|name>` | Model override. |
| `--num-threads <n>` | Parallel threads (default: configured `num_threads`). |
| `-v` / `-vv` / `-vvv` | Increase verbosity. |

When a dev set is available it prints the `baseline`, `after`, and `lift` scores;
with `--out` it also prints the save path. Optimizer sizing (bootstrapped demos,
labeled demos, rounds) is drawn from configuration — see
[Configuration](#configuration) and [Optimizers](#optimizers).

### `chat` — resumable multi-turn Q&A

A basic conversational loop. Each answer is produced with prior turns as context,
and the conversation is persisted to a named session so it can be resumed later.
This is a minor convenience feature; aiagent is not a chat framework.

```bash
aiagent chat                        # session "default"
aiagent chat --session research     # a named, separately-persisted session
aiagent chat --session research --new   # start fresh, discard that session's history
```

| Option | Description |
|--------|-------------|
| `--session`, `-s <name>` | Session to resume or start (default `default`). |
| `--new` | Start fresh, discarding the session's history. |
| `--skill <name>` | Chat skill to use (default `chat`). |
| `--model <alias\|name>` | Model override. |

In-loop commands: type `:quit` / `:q` to exit, `:reset` to clear the current
session. `Ctrl-D` / `Ctrl-C` also exits. Sessions are stored under
`~/.config/aiagent/chat-sessions/<name>.json` (see
[Chat sessions on disk](#chat-sessions-on-disk)). For a single-shot question with
no history, `aiagent run chat --text "..."` also works.

### `shell` — devai agent entrypoint

Prints a short banner (router URL + effective model + example commands) and execs
an interactive shell with the environment already wired. This is the entrypoint
the devai model-picker uses when you select **AI Agent**.

```bash
aiagent shell
aiagent shell --model <name>    # pin the model for the session
```

The shell is taken from `$SHELL` (falling back to `bash`, then `/bin/sh`).

### `version`

```bash
aiagent version     # prints the aiagent version string
```

---

## Configuration

Settings resolve with this precedence (highest first):

1. **`AIAGENT_*` environment variables** — e.g. `AIAGENT_MODEL`, `AIAGENT_API_BASE`.
2. **TOML file** at `~/.config/aiagent/config.toml`.
3. **devai-injected environment** — the generic vars a devai container exports.
4. **Code defaults**.

Layer 3 is what lets aiagent run as a devai agent that knows nothing about devai
except the router URL it is handed. The devai vars it understands:

| devai env var | Maps to |
|---------------|---------|
| `OPENAI_BASE_URL` | `api_base` |
| `OLLAMA_HOST` | `api_base` (with `/v1` appended) |
| `OPENAI_API_KEY` | `api_key` |
| `OPENAI_MODEL` / `OLLAMA_DEFAULT_MODEL` | `model` |
| `CONTEXT` / `AIAGENT_CONTEXT` | `context_tokens` |
| `HTTPS_PROXY` / `HTTP_PROXY` | `proxy_url` |

### Settings

Every field below is settable via `AIAGENT_<FIELD>` (uppercased) or a TOML key of
the same name. Inspect the resolved result with `aiagent config show`.

| Field | Default | Meaning |
|-------|---------|---------|
| `api_base` | `http://devai-router:11434/v1` | OpenAI-compatible router base URL. |
| `api_key` | `local` | API key; must be **non-empty** (LiteLLM rejects empty). devai single-mode has no auth. |
| `request_timeout_s` | `900` | Per-request timeout; generous, to absorb cold starts. |
| `num_retries` | `2` | LM retries. Worst-case wait is `(num_retries + 1) * timeout`. |
| `cache` | `true` | Enable DSPy/LiteLLM response caching. |
| `proxy_url` | `http://devai-pipelock:8888` | Forward proxy for outbound URL fetches (the `sentiment` skill). Empty string = direct connection. An injected `HTTPS_PROXY`/`HTTP_PROXY` overrides the default. |
| `model` | `""` | Effective model name; empty falls back to `default_alias`. |
| `default_alias` | `default` | Registry alias used when no model is configured. |
| `default_reasoning` | `nothink` | `think` or `nothink`; the `::<reasoning>` suffix. |
| `context_tokens` | `null` | Maps to the `@<ctx>` model-string suffix. |
| `registry_overrides` | `{}` | Per-alias `ModelSpec` overrides (see below). |
| `num_threads` | `4` | Default parallelism for eval/optimize. |
| `max_bootstrapped_demos` | `4` | BootstrapFewShot: max self-generated demos. |
| `max_labeled_demos` | `8` | BootstrapFewShot: max labeled demos. |
| `max_rounds` | `1` | BootstrapFewShot: bootstrapping rounds. |
| `skills_dir` | `~/.config/aiagent/skills` | Where user skills are discovered. |
| `sessions_dir` | `~/.config/aiagent/chat-sessions` | Where chat sessions are stored. |

### TOML example

```toml
# ~/.config/aiagent/config.toml
api_base = "http://devai-router:11434/v1"
model = "qwen3.5:9b-q8_0"
default_reasoning = "nothink"
num_threads = 8

[registry_overrides.fast]
model = "qwen3.5:9b-q8_0"
reasoning = "nothink"
ctx = 8192
```

An alias defined under `registry_overrides` becomes selectable as
`--model fast` on any command.

---

## Model strings, aliases, and reasoning

aiagent composes the devai router's control-surface model string:

```
openai/<model>::<reasoning>[@<ctx>]
```

- The `openai/` provider prefix routes DSPy/LiteLLM through the OpenAI-compatible
  path against `api_base`.
- `::<reasoning>` is `think` or `nothink` (from the alias's `ModelSpec` or, if
  unset, `default_reasoning`).
- `@<ctx>` is the optional context-window suffix. It must be the **outermost /
  last** token, because the router parses right-to-left; a `@<ctx>` baked into a
  model name is peeled off and re-emitted last so it never lands mid-name.

**Aliases vs. raw names.** `--model` (and the `model` setting) accepts either a
registry alias or a raw model name. Anything not found in the registry is treated
as a literal model name — which is exactly how devai's picker passes names like
`qwen3.5:9b-q8_0`. When `AIAGENT_MODEL` is set, the `default` alias resolves to it,
keeping the configured model the single source of truth.

The registry's baked default is a **placeholder** — always confirm the real served
tag with `aiagent models list`.

### Routing

`aiagent run <text> --route` picks a skill deterministically:

1. **Exact name match** (case-insensitive).
2. **Keyword overlap** over each skill's name + description; the highest unique
   score wins.

If nothing matches uniquely the command reports an ambiguous-skill error listing
the candidates. (An optional LLM tie-breaker exists in the routing engine but is
not enabled from the CLI.)

---

## Verbosity (`-v` / `-vv` / `-vvv`)

`run`, `eval`, and `optimize` accept a repeatable `-v` flag. All verbosity output
goes to **stderr**, so `--json`/stdout stays clean for scripting. Each level is
additive:

| Level | Adds |
|-------|------|
| `-v` | Routing summary: resolved skill + composed model string, elapsed time, LM call count. |
| `-vv` | DSPy level: the adapter's rendered prompt (system + user) and parsed completion for every LM call, via DSPy's `pretty_print_history`. |
| `-vvv` | LLM/wire level: per-call usage / cost / response model, plus LiteLLM's own verbose HTTP logging (real request/response bytes). |

---

## The expense-extraction demo

The shipped `extract` skill is the MVP demo. It extracts three fields from a
free-text expense note or receipt line:

- **merchant** — vendor name only,
- **date** — ISO 8601 `YYYY-MM-DD`,
- **amount** — a plain number (no currency symbol or separators).

It's implemented as a `ChainOfThought` program over a typed `dspy.Signature`; the
typed `amount: float` output is coerced by the ChatAdapter, and the field
descriptions steer the model toward the metric's normalization.

Full loop:

```bash
aiagent doctor                                            # verify the router
aiagent eval extract                                      # baseline field-accuracy
aiagent optimize extract --out compiled/extract.json      # BootstrapFewShot + save
aiagent eval extract --compiled compiled/extract.json     # measurable lift
```

The bundled `trainset.jsonl` / `devset.jsonl` deliberately mix currency symbols
(`$`, `USD`, thousands separators) and date formats (`M/D/YYYY`, `Mon D YYYY`,
`YYYY/MM/DD`) so optimization has real normalization patterns to learn.

---

## Datasets (JSONL)

Datasets are JSONL — one JSON object per line. For the expense demo each row has
four keys; `text` is the only input, the rest are gold labels:

```json
{"text": "Lunch at Chipotle $12.50 on 3/4/2025", "merchant": "Chipotle", "date": "2025-03-04", "amount": 12.50}
```

Rows are validated against a pydantic model before use. Blank lines are skipped; a
malformed row fails fast with `path:line` context, and an all-empty or missing
file is rejected (so an empty train/dev set surfaces clearly rather than as a
confusing downstream error).

---

## Metrics

Metrics follow DSPy's contract:

```
metric(example, prediction, trace=None) -> float | bool
```

The demo's `field_accuracy` is **dual-use**:

- **Scoring** (`trace is None`): returns the **float** fraction of
  `{merchant, date, amount}` correct, so the dev-set score is continuous.
- **Gating** (`trace is not None`, during optimizer bootstrapping): returns a
  strict **bool** — only fully-correct traces (fraction `== 1.0`) seed few-shot
  demonstrations, so a partial match never pollutes the demos.

Comparison is tolerant: strings are whitespace-collapsed and case-folded; amounts
are parsed tolerant of `$`, `USD`, and thousands separators and compared within a
`0.01` tolerance. A second metric, `exact_match`, is true only when every field is
correct in both modes.

A skill selects its metric via the `metric:` key in `SKILL.md`
(`skill:<attr>` or `<module>:<attr>`); with no `metric` declared, `field_accuracy`
is the default.

---

## Optimizers

`optimize` wraps two DSPy prompt optimizers behind one uniform call:

- **`bootstrap`** (default) — `BootstrapFewShot`. Works from ~10 examples;
  bootstraps few-shot demonstrations from training traces that pass the metric
  gate. Sized by `max_bootstrapped_demos`, `max_labeled_demos`, `max_rounds`.
- **`mipro`** — `MIPROv2`. Joint instruction + demonstration search (heavier). Uses
  the dev set as its validation set (or the train set if no dev set is given), and
  runs non-interactively (no permission prompt).

The optimization sequence is: validate the train set → baseline eval (if a dev set
is present) → compile → after eval (if a dev set is present) → save (if `--out`).

Compiled programs are persisted as **state-only JSON** (`save_program=False`) and
reloaded onto a fresh instance of the same module — so you save learned state, not
code.

---

## Writing your own skill

A user skill is a directory under `~/.config/aiagent/skills/<name>/` containing a
`SKILL.md` manifest and a Python module. Discovered automatically; list yours with
`aiagent skills list --source user`.

### `SKILL.md` manifest

YAML frontmatter followed by free-form Markdown docs. Frontmatter fields:

| Key | Required | Meaning |
|-----|----------|---------|
| `name` | yes | Skill name; `^[a-z][a-z0-9_-]*$`. |
| `description` | yes | One-line description (also used for keyword routing). |
| `entrypoint` | no | `"<module>:<callable>"` factory (default `skill:build`). |
| `model` | no | Model alias into the registry. |
| `trainset` | no | Train JSONL path, relative to the skill dir. |
| `devset` | no | Dev JSONL path, relative to the skill dir. |
| `metric` | no | `skill:<attr>` or `<module>:<attr>` (default: `field_accuracy`). |
| `version` | no | Free-form version string. |

Unknown frontmatter keys are rejected.

```markdown
---
name: extract
description: Self-optimizing extraction of {merchant, date, amount} from a free-text expense note.
entrypoint: skill:build
model: default
trainset: trainset.jsonl
devset: devset.jsonl
metric: skill:metric
version: 0.1.0
---

# Extract expense fields

...docs...
```

### The Python module

The entrypoint module must expose the factory named in `entrypoint` (default
`build`). It takes no arguments and must return a `dspy.Module`:

```python
import dspy

def build() -> dspy.Module:
    return MyPipeline()
```

If the manifest declares a `metric` of the form `skill:<attr>`, that attribute is
resolved from this same module.

User skills load by **file path** (they ship their `.py` source), whereas built-in
skills load by dotted import so they resolve their compiled `.pyc` in the
sourceless bundle. Keep skill engine (the discovery/loader machinery) and skill
content (your directory) separate.

---

## Chat sessions on disk

Each `aiagent chat` session is a JSON file under
`~/.config/aiagent/chat-sessions/<name>.json` — an ordered list of
`{"text", "answer"}` turns. Sessions persist across invocations so a conversation
can be resumed with `--session <name>`. Session names must match
`^[A-Za-z0-9][A-Za-z0-9._-]*$` (they become filenames; unsafe names are rejected
to prevent path traversal). An unreadable session file errors with a hint to start
fresh via `--new`.

---

## Development

```bash
make dev-install     # .venv with aiagent + dev deps (uv, editable)
make check           # ruff + mypy (strict)
make test            # pytest (hermetic; live devai tests are opt-in: -m live)
make test-cov        # pytest with coverage gate (85%)
make lock            # regenerate requirements*.txt (needed before `make package`)
make package         # build dist/aiagent-install.sh
make release         # tag + publish GitHub release with the installer asset
```

Requires `uv` and Python 3.13. Tests are hermetic by default: LLM-driven CLI
commands use `dspy.utils.DummyLM`, `doctor`/`models` use an httpx mock, and an
autouse fixture neutralizes env/TOML/skills-dir so nothing leaks in. Live devai
tests (`pytest -m live`) must run inside the `devai-net` network.

---

## Packaging

`make package` produces `dist/aiagent-install.sh` — a single **makeself**
self-extracting, run-once installer (**linux-x86_64**, ~63 MB). It carries a
relocatable CPython 3.13 with aiagent and every dependency,
**sourceless-precompiled** (`.pyc` only; nothing compiles at runtime). The heavy
tree is **zstd -19** compressed and decompressed at install time by a **bundled
static zstd**, so the target host needs neither Python nor zstd. makeself adds a
**SHA256** integrity check, and the `-s` launcher keeps it hermetic (ignores the
host user site). ML libraries (numpy, tokenizers, tiktoken) are kept for future
RAG features; the bundle stays torch-free.

```bash
./aiagent-install.sh                             # -> ~/.local
AIAGENT_PREFIX=/usr/local ./aiagent-install.sh   # system / image install
sh ./aiagent-install.sh --check                  # verify integrity only
aiagent --help
```

Build deps: `uv`, `makeself`, `curl`, and a C toolchain (to build the static zstd
once; it's cached under `.cache/`). Run `make lock` before `make package`.

---

## Releasing

Maintainers cut a release with `make release`. It reads the version from
`pyproject.toml`, promotes the `CHANGELOG.md` `[Unreleased]` section to that
version, rebuilds the installer, tags `vX.Y.Z`, pushes, and publishes a GitHub
release with two assets attached: the `aiagent-install.sh` bundle and the
`install.sh` bootstrap that powers the one-liner in the Quick Start.

Before releasing: bump `version` in `pyproject.toml`, add entries under
`## [Unreleased]` (an empty section is refused), and run `make lock` if
dependencies changed. Pre-flight guards require a clean tree on `main`, in sync
with `origin`, with the tag and release not yet present. For CI or non-interactive
runs, set `AIAGENT_RELEASE_ASSUME_YES=1` to skip the prompt.

---

## Deploying as a devai agent

aiagent runs as a first-class agent inside devai's `shell-gpu` / `shell-cpu`
containers. Each agent receives only the router URL via env and talks to it over
the OpenAI-compatible API — aiagent has no devai-specific code; it adapts through
config env-fallbacks (`AIAGENT_API_BASE` → `OPENAI_BASE_URL` → `OLLAMA_HOST`).

Apply this change set in the **devai** repo:

**1. Register the agent** — `scripts/model-picker.py`, add to `_AGENTS`:

```python
("aiagent", "AI Agent", "Programmatic DSPy agent: optimize, evaluate, autonomous data processing"),
```

**2. Launch handler** — in `model-picker.py`'s `_build()`, add a branch (a
configured subshell where `aiagent run|optimize|eval|chat` are ready):

```python
if agent_id == "aiagent":
    os.environ["AIAGENT_API_BASE"] = f"{base}/v1"   # base = http://devai-router:<port>
    os.environ["AIAGENT_API_KEY"]  = "local"
    os.environ["AIAGENT_MODEL"]    = name
    os.environ["AIAGENT_CONTEXT"]  = str(context_tokens)
    return ["aiagent", "shell", "--model", name]
```

**3. Fetch the bundle** — `Makefile` `fetch-cli`, mirror the `claude` recipe to
ETag-download `aiagent-install.sh` (a GitHub release asset) into
`/var/cache/devai/pip/bin/`. *(Local dev: copy a `make package` output there.)*

**4. Install at image build** — `deploy/Dockerfile.lab` (build-time install, like
aider's `uv tool install`):

```dockerfile
RUN AIAGENT_PREFIX=/usr/local sh /var/cache/bin/aiagent-install.sh
```

This lands `/usr/local/bin/aiagent` (no Python or zstd needed in the image — the
bundle is self-contained). Both shell images are glibc x86_64, so the one x86_64
bundle serves `shell-cpu` and `shell-gpu`. Then `make shell-gpu` → pick **AI
Agent** → a shell wired to the router, where the demo runs live.
