# Guardrails Challenge

> **Status: chatbot with measured retrieval, no guard pipeline wired in yet.**
> The knowledge base, a German-aware BM25 retriever, a multi-turn chatbot, a
> real Anthropic-backed completion layer, a run-scoped JSONL trace writer, and
> a CLI entry point all exist and are tested (`298 passed, 6 skipped`).
> Retrieval recall is measured, not assumed — see the "已知局限" (known
> limitations) section of [DESIGN.md](DESIGN.md). What is **not** here yet: the four
> guards themselves and the orchestrator that would route their verdicts to an
> action. `python -m chatbot` talks straight to the model over the retrieved
> context with no guard in the loop.

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

```bash
export ANTHROPIC_API_KEY=...          # required
export ANTHROPIC_BASE_URL=...         # optional, for a third-party relay endpoint
PYTHONPATH=src ./.venv/bin/python -m chatbot --profile telco_de --question "Was kostet Tarif M?"
```

Omit `--question` to enter a REPL. Each run leaves one trace at
`runs/<run_id>.jsonl`.

The guard layer is not wired into this entry point yet — see the roadmap in
DESIGN.md.

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
