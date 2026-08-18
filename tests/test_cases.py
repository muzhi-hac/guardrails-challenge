"""Test scenarios.

Four buckets:

- benign        — realistic requests that must NOT be blocked (false-positive rate)
- adversarial   — injection, jailbreak, multi-turn escalation
- grounding     — questions whose answers are absent from the knowledge base
- persona       — attempts to push the assistant out of its brand voice
"""
