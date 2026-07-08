---
name: sentiment
description: Sentiment analysis of documents, articles, and web pages on a -10 to +10 scale, with volatility, statistical significance, and a plain-language explanation.
entrypoint: skill:build
model: default
version: 0.1.0
---

# sentiment

Measure the sentiment of one or more data sources on a scale from **−10**
(very negative) to **+10** (very positive). Works out of the box — just point it
at your data; no dataset, tuning, or configuration required.

Beyond the headline score it reports **volatility** (how much sentiment swings
across the content), **model uncertainty** (how much the model disagrees with
itself), and **statistical significance** (a t-test that the sentiment differs
from neutral), plus a short human-readable explanation.

## Usage

```bash
# Local files (.txt, .md, .html, .pdf), URLs, or raw text — repeatable, mixable.
aiagent sentiment --file review.txt --url https://example.com/post --json
aiagent sentiment --text "The rollout was flawless and the team is thrilled."

# The generic runner also works for a single blob of text:
aiagent run sentiment --text "..."
```

URLs are fetched through the devai egress proxy (pipelock); local files are read
directly. Output is human-readable by default, or structured JSON with `--json`.
