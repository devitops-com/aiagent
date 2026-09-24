"""System 1: distilled laya students run in-process (import-light; no numpy, no dspy).

The low layer of distillation: the aiagent <-> devai contract primitives, laya's input
sequence builder, the onnxruntime student, trained-artifact verification and
installation, and the ``System1First`` cascade around a skill's predictor.
"""

from __future__ import annotations
