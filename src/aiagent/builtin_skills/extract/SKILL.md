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

Extracts structured expense fields — **merchant**, **date** (ISO `YYYY-MM-DD`),
and **amount** (a plain number) — from a free-text expense note or receipt line.

This is the MVP demo skill. It exists to show the core aiagent loop end-to-end:

```bash
aiagent eval extract                                    # baseline field-accuracy
aiagent optimize extract --out compiled/extract.json    # BootstrapFewShot + save
aiagent eval extract --compiled compiled/extract.json   # measurable lift
```

The bundled `trainset.jsonl` / `devset.jsonl` mix currency symbols (`$`, `USD`,
thousands separators) and date formats (`M/D/YYYY`, `Mon D YYYY`, `YYYY/MM/DD`)
so optimization has real normalization patterns to learn.
