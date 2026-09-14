# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-09-14

### Fixed
- **The installer put its bundled Python on your PATH, shadowing the system
  interpreter.** Every release since the first one linked the bundled
  interpreter into `$PREFIX/bin` as `python<X.Y>` alongside `aiagent`. With the
  default prefix that is `~/.local/bin`, which sits ahead of `/usr/bin` for most
  users, so `python3.14` in any shell resolved to aiagent's private interpreter
  — sourceless (`.pyc` only), carrying aiagent's own site-packages, and with
  pip/setuptools stripped by the build. On a host with its own Python 3.14 it
  silently hijacked it.

  Nothing needed the link: console-script shebangs are rewritten at install time
  to the bundled interpreter's *absolute* path, so `aiagent` works without it.
  Only `aiagent` is placed on PATH now; the interpreter stays at
  `$PREFIX/lib/aiagent/bin/python<X.Y>`.

  Because earlier installers created it, upgrading also **removes** the stale
  link — but only when it still points into aiagent's own `lib/aiagent`, so a
  `python<X.Y>` you installed yourself is never touched. The installer says so
  when it removes one.

  `make package` now asserts that `$PREFIX/bin` contains nothing but `aiagent`,
  so this cannot regress silently. Anyone on <= 0.3.0 who does not upgrade can
  clear it by hand: `rm ~/.local/bin/python3.14` (check it points into
  `~/.local/lib/aiagent/` first).

## [0.3.0] - 2026-09-14

### Changed
- **Python 3.14.** The pinned dev / CI / bundled-interpreter version moves from
  3.13 to 3.14 (`.python-version`, with `requires-python` and the trove
  classifier kept in sync). All 85 locked pins were audited against 3.14;
  litellm was the only incompatible one.
- **litellm floor raised to `>=1.93`** (was `>=1.64.0`; lock moves 1.90.1 ->
  1.100.1). 1.93.0 is the first release whose `requires-python` admits 3.14 —
  everything earlier caps at `<3.14` and cannot be installed at all, which is
  what broke `make package`. No upper bound: litellm tracks provider APIs and is
  worth following.
- **The packaging build now strips the AWS/Bedrock subtree** (boto3, botocore,
  s3transfer, jmespath, python-dateutil, six) the way it already strips hf_xet.
  litellm 1.98 promoted boto3 from an extra to a core dependency; aiagent talks
  to a local devai router and never takes the Bedrock path, and litellm imports
  boto3 lazily inside those handlers rather than at module scope. That is ~21 MB
  uncompressed (botocore's per-service JSON is 20 MB of it) for code that never
  runs. Tracking latest litellm *with* the strip costs +0.9 MB of installer over
  pinning at 1.97 without it; the installer lands at ~71 MB (from ~63 MB, the
  rest being the interpreter and litellm bumps).

  The set is exactly what is reachable only through boto3 — urllib3 is
  deliberately not in it, since requests needs it independently. The strip set
  is declared once in `build-binary.sh` and feeds both the removal and
  `verify-versions.py`'s allow-list, so a strip the audit doesn't know about
  fails the build. The bundle's smoke probe now imports litellm and
  `RetryAwareLM` with the subtree gone, so a future litellm that imports boto3
  at module scope fails the build instead of shipping a broken bundle. A
  pip/uv install of aiagent is unaffected and still gets boto3.
- **`make lock` now resolves against `.python-version`** (`uv pip compile
  --python-version`) rather than whichever interpreter happens to be active. uv
  treats an existing `requirements*.txt` as pins to preserve and does not
  re-validate them against a changed `requires-python`, so locking from a stale
  venv silently kept a pin the target Python rejects — surfacing only later as a
  `ResolutionImpossible` during `make package`.

### Security
- **pypdf floor raised to `>=6.16.1`** (lock moves 6.14.2 -> 6.18.1). Every
  earlier release carries resource-exhaustion advisories reachable from ordinary
  text extraction — PYSEC-2026-3655/3656 (crafted `/ToUnicode` entries and font
  widths), 3910/3911 (deeply nested outlines, re-used XForm objects), 3912
  (`read_until_whitespace`) and 3913 (an infinite loop in
  `TreeObject.insert_child`). The `sentiment` skill extracts text from PDFs
  fetched over URLs, so that input is attacker-controlled and these were
  genuinely reachable. The floor, not just the pin, is raised so a plain
  `pip install aiagent` cannot resolve back onto a vulnerable pypdf.

### Fixed
- **The dependency-audit workflow could never pass** — 68 runs, 68 failures. It
  installed the project with `pip install -e`, told pip-audit to
  `--skip-editable`, then passed `--strict`, which fails the audit if dependency
  collection is skipped for *any* dependency. The skip it was configured to
  perform was the thing `--strict` treated as fatal, so it reported
  `aiagent: distribution marked as editable` every time and never once scanned
  for a CVE. It now audits `requirements.txt` and `requirements-dev.txt`
  directly — the runtime lock is what the installer actually bundles — which
  removes the editable case entirely. The workflow had also been auto-disabled
  by GitHub for repository inactivity and has been re-enabled.

  `diskcache` 5.6.3 (reached via dspy) is suppressed via `--ignore-vuln
  PYSEC-2026-2447`: pickle deserialization allows code execution by an attacker
  who can already write to the cache directory, 5.6.3 is the latest release and
  upstream has published no fix, so there is nothing to upgrade to. Remove the
  flag when a fixed diskcache ships.
- **Usage errors printed a traceback instead of help** (Typer 0.27.2): Typer
  moved `Abort` out of `typer._click.exceptions` into `typer.exceptions`
  (fastapi/typer#1942). `cli/app.py` imported `Abort` and `UsageError` from the
  vendored module in a single `try` block, so the relocation raised
  `ImportError` and dropped the vendored **`UsageError`** from `_USAGE_ERRORS`
  as collateral — every usage error (unknown command, missing required
  argument) then escaped `main()` as an unhandled traceback with exit code 0.
  `Abort` now comes from the public `typer.Abort`, `UsageError` keeps its own
  guarded import, and a regression test drives the module entry point to assert
  help + exit 2. This was latent for any fresh install, independent of 3.14.

### Notes
- Typer 0.27.0 changed metavar rendering (breaking, fastapi/typer#1863):
  argument metavars are no longer upper-cased and types print as the Python
  type, so `aiagent run [OPTIONS] SKILL` now reads `aiagent run [OPTIONS]
  {skill}` and the type column shows `<str>` rather than `TEXT`. Help text
  only — no CLI surface change. The metavar test was rewritten to check the
  usage line case-insensitively (it had been passing for `run` on an unrelated
  `--route` description rather than on the metavar).

## [0.2.1] - 2026-07-09

### Fixed
- **Non-retryable 4xx errors were retried, hammering the router** (issue #10):
  DSPy hands `num_retries` to `litellm.completion(retry_strategy=
  "exponential_backoff_retry")`, whose retry loop is not status-aware, so a
  permanent client error (`400`/`404`/`422`) was re-issued several times — a
  ~150s retry storm that, against a model router, repeatedly reloaded a backend
  that would always fail. A new `RetryAwareLM` (`llm/retry_lm.py`) wraps
  `dspy.LM`: it disables litellm's blind retry and retries **only** transient
  failures (connection errors, timeouts, `429`, and `5xx`), surfacing `4xx`
  client errors immediately — matching Codex/Claude Code/OpenCode. Cold-start
  resilience (generous `request_timeout_s` + retries on transient errors) is
  preserved; `num_retries` now bounds transient retries only.

## [0.2.0] - 2026-07-08

### Added
- **`sentiment` built-in skill + `aiagent sentiment` command.** Scores one or
  more data sources on a −10 (very negative) … +10 (very positive) scale and
  reports statistical properties — **volatility** (sentiment spread across the
  content), **model uncertainty** (self-disagreement across resamples), and
  **significance** (a one-sample t-test vs. neutral, with t-statistic, two-sided
  p-value, and a confidence label) — plus a plain-language explanation. Human
  output by default, `--json` for structured output. Sources are supplied with
  repeatable `--text`, `--file` (`.txt`/`.md`/`.html`/`.pdf`), and `--url`;
  local files are read directly and URLs are fetched through the devai egress
  proxy (pipelock). Works out of the box with no dataset or configuration — the
  skill is run-only, so `aiagent run sentiment --text "..."` works too. New
  `proxy_url` setting (`AIAGENT_PROXY_URL`, default `http://devai-pipelock:8888`;
  an injected `HTTPS_PROXY`/`HTTP_PROXY` is honored). Adds a pure-Python `pypdf`
  dependency (stays torch-free).

## [0.1.4] - 2026-07-07

### Added
- **`aiagent chat` REPL line editing, history, and completion** (issues #7, #8):
  the `you>` prompt now uses `readline` for inline emacs-style editing, Up/Down
  history, and `Ctrl-R` reverse search, with history persisted across sessions
  under `sessions_dir/history`. Tab completes the `:` directives, and when
  `fzf` is on `PATH`, `:history` opens a vertical fuzzy picker over past prompts
  and `:help` a command palette; both pre-fill the chosen line for editing
  before submit. `fzf` and `readline` are optional — the REPL degrades cleanly
  when either is absent.

### Fixed
- **Reasoning suffix appended after a pre-baked `@<ctx>`** (issue #6):
  `compose_model_string` glued `::<reasoning>` **after** a `@<ctx>` already
  baked into the model name (`<model>@<ctx>::<reasoning>`), which strict
  gateways reject. A trailing `@<int>` context suffix on the model name is now
  peeled off and re-emitted last, matching the `AIAGENT_CONTEXT` path
  (`<model>::<reasoning>@<ctx>`); an explicit ctx still wins and is never
  duplicated.

## [0.1.3] - 2026-07-07

Adds repeatable -v flags shared by aiagent run/eval/optimize, each level
strictly additive and written to stderr (stdout stays clean for --json):
    
- -v  routing: resolved skill, composed model string, elapsed time, call count.
- -vv DSPy level: the ChatAdapter-rendered system/user messages and parsed
      completion for every LM call made in the invocation, via DSPy's own
      pretty_print_history.
- -vvv LLM/wire level: per-call usage/cost, plus real over-the-wire HTTP
      (the actual request LiteLLM sent and the raw response body).


## [0.1.2] - 2026-07-03

### Fixed
- **Control-surface model string composed in the wrong order** (issue #3):
  `compose_model_string` emitted `@<ctx>` **before** `::<reasoning>`. The devai
  router only strips `@<ctx>` when it is the outermost (last) token, so with a
  context set (`AIAGENT_CONTEXT`) the `@<ctx>` survived into the model name and
  Ollama rejected the request with `invalid model name`. The string is now
  `openai/<model>::<reasoning>[@<ctx>]`, with `@<ctx>` last.
- **`aiagent version` out of sync with the packaged version** (issue #5): the
  command printed a hand-maintained `__version__` literal that drifted from
  `pyproject.toml` (printed `0.1.0` on the `v0.1.1` bundle). `__version__` now
  derives from the installed package metadata, so it always tracks the release.

### Changed
- **`AIAGENT_MODEL` is the single source of truth for the `default` alias**
  (issue #4): when a model is configured, the registry's `default` alias (and
  `aiagent models list`) resolves to it instead of the baked placeholder, so
  skills and callers that route through the `default` alias no longer silently
  use a different model.

## [0.1.1] - 2026-07-02

### Fixed
- **CLI crash on argument-bearing commands** (issue #1): the shipped bundle
  paired a pre-Click-8.2 Typer with Click 8.2+, whose `Parameter.make_metavar`
  gained a required `ctx` argument — mutually incompatible, so rendering usage or
  help for any command with a positional argument (`run`, `eval`, online
  `doctor`) crashed. Dependency floors are pinned to a compatible era
  (`typer>=0.16`, `click>=8.2`).
- **Built-in skills failing to load from the sourceless bundle** (issue #2):
  skill entry modules were imported by `skill.py` file path, but the bundle
  strips `.py` sources (`.pyc` only), so every built-in raised `entry module
  skill.py not found`. Built-ins now load by dotted import
  (`aiagent.builtin_skills.<dir>.<module>`), which resolves the compiled `.pyc`.
- `aiagent run chat --text "..."` crashing with
  `ChatSkill.forward() got an unexpected keyword argument 'text'`.

### Added
- **Resumable multi-turn chat**: `aiagent chat` persists its conversation history
  to a named session (`--session/-s`, `--new`, `:reset`) under
  `<config>/chat-sessions/` (`AIAGENT_SESSIONS_DIR`), so a conversation can be
  resumed across invocations. History is optional, so `run chat --text` also
  works single-shot.

### Changed
- **Bundle build hardening** against the non-reproducible artifact behind the
  v0.1.0 defect: dependencies install with `--no-cache-dir` (always fetched fresh
  from the configured index), and the build now audits every bundled module
  against `requirements.txt` at both the dist-info **and** imported-`__version__`
  level, failing on any stale/mismatched module.

## [0.1.0] - 2026-07-01

### Added
- Initial project scaffold: uv + Python 3.13, hatchling src-layout, Typer CLI,
  ruff/mypy/bandit/pytest tooling, CI workflows.
- **Config + LLM layer**: pydantic-settings with env-fallback resolution
  (`AIAGENT_*` → `OPENAI_*` → `OLLAMA_HOST` → defaults); a model registry that
  composes devai control-surface model strings (`<model>@<ctx>::<reasoning>`),
  `dspy.LM` construction and per-step routing. DSPy's default ChatAdapter.
- **Core**: typed `ExtractExpense` signature + `ChainOfThought` module; JSONL
  loader with pydantic-validated rows; a dual-use field-accuracy metric (float
  for scoring, strict gate for bootstrapping); a structured `Evaluate` wrapper.
- **Optimization**: `optimize()` over `BootstrapFewShot` (default) / `MIPROv2`,
  with state-only JSON save/load.
- **Skills engine**: Claude-Code-style auto-discovery of `SKILL.md` skills
  (built-in package data + user dir), an immutable registry, a free-text router,
  and dynamic module loading (dspy imported lazily).
- **CLI**: `doctor`, `models list`, `config show`, `skills list`, `run`,
  `optimize`, `eval`, `chat`, and `shell` (the devai picker entrypoint).
- **Packaging**: `make package` → a single **makeself** self-extracting
  `aiagent-install.sh` (linux-x86_64) that includes its own CPython 3.13,
  sourceless-precompiled (`.pyc` only), **zstd -19** compressed with a **bundled
  static zstd** (no Python or zstd needed on the target), SHA256 integrity, and a
  hermetic `-s` launcher. ML libs (numpy/tokenizers/tiktoken) kept for future RAG.
  Trims the Tcl/Tk runtime and hf_xet. ~63 MB.
- Built-in `extract` (self-optimizing expense extraction demo) and `chat` skills.
