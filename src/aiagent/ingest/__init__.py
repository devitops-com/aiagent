"""Data-source ingestion for aiagent skills (import-light; no ``dspy``).

Turns a user-supplied source — a local file or an ``http(s)`` URL — into plain
text a skill can analyze. URL fetches route through the configured forward proxy
(devai's pipelock). Text, HTML, and PDF inputs are supported out of the box.
"""
