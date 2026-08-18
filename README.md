# Guardrails Challenge

> **Status: scaffold.** Structure is in place; implementation and documentation are in progress.

A guardrails system for a multi-turn customer-service assistant in the telecom
domain, covering four failure modes:

1. **Persona / brand-voice drift** — the assistant stops sounding like the brand
   (formality slips, TTS-unsafe output).
2. **Ungrounded policy and pricing claims** — invented tariffs, dates, or
   entitlements not supported by the retrieved knowledge base.
3. **Prompt injection and jailbreaks** — both from user input and from retrieved
   documents treated as untrusted data.
4. **PII exposure** — inbound redaction before logging, outbound leak prevention.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` to run against the live API. Without it, the demo and
evaluation fall back to recorded responses.

## Usage

_To be filled in once the entry points exist._

## Repository layout

| Path | Contents |
|---|---|
| `DESIGN.md` | Design thinking, architecture, trade-offs |
| `src/guardrails/` | Guardrail implementations |
| `src/chatbot.py` | Chatbot loop the guardrails wrap |
| `src/utils.py` | Shared helpers |
| `tests/` | Test scenarios and results |
| `examples/` | Demo notebook and example interactions |
| `docs/evaluation.md` | Reflection and evaluation |
