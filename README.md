# aiagent

A programmatic [DSPy](https://dspy.ai) agent framework focused on **query/prompt
optimization, goal-reaching loops, and autonomous data processing** over local
LLMs — not a chat UI (basic chat is a minor feature). It runs against an
OpenAI-compatible router (the [devai](https://github.com/ksparavec/devai)
`devai-router`) and ships as a single self-extracting, precompiled bundle that
includes its own Python.

> Status: early MVP under active construction. See `CHANGELOG.md`.

## Quick Start

Linux x86_64, no Python or dependencies required — the bundle ships its own:

```bash
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | sh
```

Installs to `~/.local` by default. Override the prefix, or pin a version, via env:

```bash
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | AIAGENT_PREFIX=/usr/local sh
curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | AIAGENT_VERSION=v0.1.0 sh
```

Then, with a reachable router, try these two:

```bash
# 1. Verify the router is reachable and see the models it advertises.
aiagent doctor

# 2. Extract structured fields from a free-text expense note.
aiagent run extract --text "Lunch at Chipotle $12.50 on 3/4/2025"
```

The second prints `merchant`, `date` (ISO), and `amount` parsed from the note.
Run `aiagent --help` for the full command list.

## Documentation

The **[User Manual](docs/USER_MANUAL.md)** documents every implemented feature in
detail: all CLI commands, configuration and model strings, the self-optimizing
expense demo, datasets and metrics, writing your own skills, plus development,
packaging, releasing, and deploying as a devai agent.
