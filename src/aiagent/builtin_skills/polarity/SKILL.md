---
name: polarity
description: Label a passage's polarity as negative, neutral, mixed or positive (System 1 distillation pilot).
entrypoint: skill:build
model: default
version: 0.1.0
---

# polarity

Labels the overall polarity of a passage as exactly one of **negative**,
**neutral** (no sentiment), **mixed** (both negative and positive) or
**positive**. It is a single `dspy.Predict` (the predictor `classify`) with one
text input and one `Literal` output, so it qualifies for System 1 distillation:
a small local student model can answer it first and hand over to the LLM only
when it is unsure.

```bash
aiagent run polarity --text "The delivery was two weeks late, but support was excellent."
```

The skill is run-only: no trainset, devset or metric. To train, evaluate and
install a student for it, see the `distill` section of `docs/USER_MANUAL.md`.
For a scored sentiment analysis of whole documents, use the `sentiment` skill.
