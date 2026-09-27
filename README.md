# aiagent

A programmatic [DSPy](https://dspy.ai) agent framework focused on **query/prompt
optimization and autonomous data processing** over local LLMs: run skills, analyze
sentiment, evaluate and optimize prompts, and distill System 1 students. Not a chat
UI (basic chat is a minor feature). It runs against an OpenAI-compatible router (the
[devai](https://github.com/ksparavec/devai) `devai-router`) and ships as a single
self-extracting, precompiled bundle that includes its own Python.

> Status: early MVP under active construction. See `CHANGELOG.md`.

## Quick Start

Linux x86_64, no Python or dependencies required — the bundle ships its own:

```bash
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | sh
```

Installs to `~/.local` by default. Override the prefix, or pin a version, via env:

```bash
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | sudo AIAGENT_PREFIX=/usr/local sh
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | AIAGENT_VERSION=v0.8.0 sh
```

The installed files belong to whoever runs the installer (root for a system
install), and every user can read and run them. The installer unpacks under
`$TMPDIR`, by default `/var/tmp` (never `/tmp`), and also works where that
directory is mounted `noexec`.

The installer from the GitHub releases is the only way to install aiagent: it is
not published to PyPI (its package metadata carries the `Private :: Do Not Upload`
classifier, which PyPI refuses).

Then, with a reachable router, try these two:

```bash
# 1. Verify the router is reachable and see the models it advertises.
aiagent doctor

# 2. Extract structured fields from a free-text expense note.
aiagent run extract --text 'Lunch at Chipotle $12.50 on 3/4/2025'
```

The second prints `merchant`, `date` (ISO), and `amount` parsed from the note. (Single
quotes keep the shell from expanding `$1` in `$12.50`: in double quotes it would become `2.50`.)
Run `aiagent --help` for the full command list.

### Verify the download

Every release is built by GitHub Actions, which also signs a
[build provenance attestation](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations)
for both release files. With the [GitHub CLI](https://cli.github.com) (logged in),
set `AIAGENT_VERIFY=1` and `install.sh` checks the installer's attestation before it
runs it. If the check fails, or `gh` is missing, it stops without running anything:

```bash
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | AIAGENT_VERIFY=1 sh
```

That does not check `install.sh` itself. To check both files by hand:

```bash
curl -fsSLO https://github.com/devitops-com/aiagent/releases/latest/download/install.sh
curl -fsSLO https://github.com/devitops-com/aiagent/releases/latest/download/aiagent-install.sh
gh attestation verify install.sh --repo devitops-com/aiagent
gh attestation verify aiagent-install.sh --repo devitops-com/aiagent
sh aiagent-install.sh
```

Every published release (v0.4.0 and later) was built and attested by GitHub
Actions; earlier versions are no longer published.

## Quick examples

```bash
# Install to ~/.local (Linux x86_64).
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | sh
# Is the router reachable, and which models does it serve?
aiagent doctor
# Run one skill on one input.
aiagent run polarity --text 'Arrived late, but support was great.'
# Score documents and web pages on a -10..+10 scale.
aiagent sentiment --file report.pdf --url https://example.com/article
# Can a skill's predictor be distilled into a System 1 student?
aiagent distill plan polarity
```

Every command's options: `aiagent <command> --help`. The
[User Manual](docs/USER_MANUAL.md#command-reference) has examples and sample output for each,
and the [System 1 walkthrough](docs/SYSTEM1_WALKTHROUGH.md) records the first teacher/student
campaign step by step: `distill` label, train, eval, repair and install for `polarity`, its
shadow run, and the runs that calibrate `sentiment` to use the student.

## Documentation

The **[User Manual](docs/USER_MANUAL.md)** documents every implemented feature in
detail: all CLI commands (including `sentiment` and System 1 distillation, shadow
and gate), configuration and model strings, the self-optimizing expense demo,
datasets and metrics, writing your own skills, plus development, packaging,
releasing, and deploying as a devai agent.

The **[System 1 walkthrough](docs/SYSTEM1_WALKTHROUGH.md)** is the record of the first System 1
campaign, end to end: a teacher LLM labels a corpus, devai trains a small student on it,
aiagent certifies, installs and shadows it, and lab runs calibrate `sentiment` to use it. It
gives every command and what came out.
