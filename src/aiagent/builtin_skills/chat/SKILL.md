---
name: chat
description: Basic multi-turn chat against the configured model (a minor convenience feature).
entrypoint: skill:build
model: default
version: 0.1.0
---

# Chat

A minimal conversational skill built on `dspy.Predict` + `dspy.History`. aiagent
is not a chat framework — this exists so `aiagent chat` works for quick Q&A. For
anything substantive, use `run` / `optimize` / `eval`.
