"""Guardrail implementations.

Each guardrail exposes an async ``check`` returning a ``Verdict`` so the
orchestrator can run them in parallel under a latency budget and aggregate
the results into a single routing decision.
"""
