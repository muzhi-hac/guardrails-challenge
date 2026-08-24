# Test Results and Analysis

Measured on 2026-08-25. The default suite is reproducible without credentials:

```text
PYTHONPATH=src:tests ./.venv/bin/python -m pytest -q
492 passed, 11 skipped
```

Pytest collects 503 cases in total. The 11 default skips are five deliberately
recorded retrieval-limitation queries and six live-endpoint tests.

## Test distribution

| Area | Collected | Contents |
|---|---:|---|
| Scenario corpus | 33 | Four end-to-end buckets plus their report |
| Chatbot and CLI | 53 | Turn wiring, actions, traces, CLI, 4 live chatbot cases |
| Guard implementations | 125 | Document, grounding, injection, persona, PII and tone |
| Pipeline, profiles and shared types | 70 | Deadlines, routing, tier gates, validation, records |
| Provider | 4 | Text/structured semantics and 2 live provider cases |
| Retrieval, KB, chunking and recall | 60 | Indexing, chunk stability, search and recall report |
| Locale rules and tokenisation | 147 | German and English parsing boundaries |
| Utilities | 11 | Profile loading and trace persistence |
| **Total** | **503** | **492 pass and 11 skip in the default run** |

## Scenario corpus

Run with:

```text
PYTHONPATH=src:tests ./.venv/bin/python -m pytest tests/test_cases.py -q -s -k scenario_report
```

| Bucket | Passed / total | Criterion |
|---|---:|---|
| Benign | **21 / 21** | Real customer questions were not blocked and the fixed valid reply reached the customer |
| Adversarial | **4 / 4** | Override, role reassignment, encoded payload and cross-turn assembly stopped before generation |
| Grounding | **3 / 3** | Unsupported price, date and duration claims did not reach the customer |
| Persona | **4 / 4** | Address form, emoji, forbidden phrase and residual tone drift produced guard findings |

The benign bucket directly reuses all 21 entries from
`tests/fixtures/recall_queries.py`; it does not substitute easier synthetic
questions. On this set the observed guard-layer false-positive block rate is
**0 / 21**. The adversarial stop rate is **4 / 4**. These are scenario-set
results, not estimates of production precision or recall.

## Retrieval recall

The fixed 21-query set is judged by `doc_id`: a query succeeds when any
document in its expected set is present in the first *k* results. The criterion
does not require a particular chunk, rank within the top *k*, or full ordering.

```text
exact-term            (n=16)  @1=0.88  @3=0.94  @5=1.00
limitation-derivation (n=1)   @1=0.00  @3=0.00  @5=0.00
limitation-paraphrase (n=4)   @1=0.50  @3=0.50  @5=0.75
overall               (n=21)  @1=0.76  @3=0.81  @5=0.90
```

## Latency

| Path | Measured latency | Scope |
|---|---:|---|
| Tier-0 INPUT guards | 0.01 ms | Fixed local evaluation run |
| Tier-0 RETRIEVAL guards | 0.31 ms | Fixed local evaluation run |
| Tier-0 OUTPUT guards | 1.96 ms | Fixed local evaluation run |
| **All tier-0 stages** | **about 2.3 ms** | Sum of the three measurements above |
| Tier-1 judge, `claude-sonnet-5` | **3583 ms median** | Three forced-tool probes, `max_tokens=100` |
| Tier-1 judge, `claude-opus-5` | 6303 ms median | Same probe and route |

The two judge models produced the same decision in all three constrained
probes; Sonnet was about 1.8 times faster. These measurements drove the shipped
`models.judge` choice and the increase of the chat budget from 1500 ms to 5000
ms. Voice remains tier 0: the measured Sonnet median is more than 24 times its
150 ms budget.

The live benign before/after transcript took 4549.50 ms at OUTPUT with the
judge enabled. That single observation is evidence that the new 5000 ms budget
can be consumed substantially; it is not a latency percentile.

## Not yet measured

The following cells stay deliberately blank until the required data exists:

- **Guard precision and recall.** This needs a labelled adversarial corpus with
  independent ground truth; four hand-picked attack scenarios are not enough.
- **Cost per session.** Providers report token counts, but this project does
  not yet own a dated model-price table or a session traffic distribution.
- **Tier-1 reach rate.** The judge is wired and traces identify tier 1, but no
  representative traffic sample has been run through it.
- **Latency percentiles under concurrency.** The reported medians and one live
  turn are serial probes, not p50/p95/p99 measurements at production load.

Publishing those as zeros would confuse “not measured” with “measured and none
observed,” the same class of error the verdict vocabulary avoids by keeping
`ERROR`, `TIMEOUT` and `SKIPPED` distinct from `PASS`.
