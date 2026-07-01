# aiagent

A programmatic [DSPy](https://dspy.ai) agent framework focused on **query/prompt
optimization, goal-reaching loops, and autonomous data processing** over local
LLMs — not a chat UI (basic chat is a minor feature). It runs against an
OpenAI-compatible router (the [devai](https://github.com/ksparavec/devai)
`devai-router`) and ships as a single self-extracting, precompiled bundle that
includes its own Python.

> Status: early MVP under active construction. See `CHANGELOG.md`.

## What it does (MVP)

- **Self-optimizing pipelines** — define a typed DSPy program + metric, evaluate
  over data, then `optimize` (BootstrapFewShot) to measurably lift quality and
  save the compiled program for reuse.
- **Skills** — auto-discovered, Claude-Code-style skills (a `SKILL.md` manifest +
  a Python module exposing a DSPy module), invoked from the CLI.
- **Local-first** — talks to any OpenAI-compatible backend; designed to run as a
  first-class agent inside devai's `shell-gpu` / `shell-cpu` containers.

## The demo (expense extraction)

```bash
aiagent doctor                                            # verify the router
aiagent eval extract                                      # baseline field-accuracy
aiagent optimize extract --out compiled/extract.json      # compile + save
aiagent eval extract --compiled compiled/extract.json     # measurable lift
```

## Development

```bash
make dev-install     # .venv with aiagent + dev deps (uv, editable)
make check           # ruff + mypy (strict)
make test            # pytest (hermetic; live devai tests are opt-in: -m live)
make lock            # regenerate requirements*.txt (needed before `make package`)
make package         # build dist/aiagent-install.sh
```

Requires `uv` and Python 3.13.

## Packaging

`make package` produces `dist/aiagent-install.sh` — a single **makeself**
self-extracting, run-once installer (**linux-x86_64**). It carries a relocatable
CPython 3.13 with aiagent + every dependency, **sourceless-precompiled** (`.pyc`
only; nothing compiles at runtime). The heavy tree is **zstd -19** compressed and
decompressed at install time by a **bundled static zstd**, so the target host
needs neither Python nor zstd. makeself adds a **SHA256** integrity check, and the
`-s` launcher keeps it hermetic (ignores the host user site). ML libraries
(numpy, tokenizers, tiktoken) are kept for future RAG features. ~63 MB.

```bash
./aiagent-install.sh                             # -> ~/.local
AIAGENT_PREFIX=/usr/local ./aiagent-install.sh   # system / image install
sh ./aiagent-install.sh --check                  # verify integrity only
aiagent --help
```

Build deps: `uv`, `makeself`, `curl`, and a C toolchain (to build the static
zstd once; it's cached under `.cache/`). Run `make lock` before `make package`.

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

**2. Launch handler** — in `model-picker.py`'s `_build()`, add a branch (chosen
behavior: a configured subshell where `aiagent run|optimize|eval|chat` are ready):

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
bundle is self-contained). Both shell images are glibc x86_64, so the one
x86_64 bundle serves `shell-cpu` and `shell-gpu`. Then `make shell-gpu` → pick
**AI Agent** → a shell wired to the router, where the demo runs live.
