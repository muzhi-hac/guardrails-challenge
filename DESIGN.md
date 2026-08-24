# Design

> **Status.** The end-to-end guard layer is implemented. Six registered guards
> run across INPUT, RETRIEVAL and OUTPUT; the chatbot executes their routed
> actions, supports a `--no-guards` baseline, and records stage summaries.
> Retrieval, scenario, latency and live before/after evidence are recorded in
> `tests/test_results.md` and `examples/example_runs.md`.

## 1. Domain and why

The system is a guardrail layer for a **German-language telecom
customer-service assistant**. Both halves of that phrase were chosen for what
they make checkable.

**Telecom, because the claims are concrete and the errors are costly.** A
customer-service turn in this domain is mostly factual assertions with a price
tag attached: what Tarif M costs per month, how long the `Kündigungsfrist` is,
when a `Widerrufsfrist` starts running, whether a fault has to be fixed within a
stated `Entstörfrist`. Each of those either appears in the operator's own
documents or it does not. That is what makes grounding a *measurable* property
rather than a matter of taste: an invented tariff price is not "off-tone", it is
wrong, and the customer acts on it. It is also the failure a model's own safety
training has no opinion about — an invented price is not unsafe, so nothing in
the model's alignment stack objects to it. That gap is the honest scope of what
a guard layer adds.

**German, because the language decides part of the answer.** German
grammaticalises the T-V distinction: `Sie`/`Ihnen`/`Ihr` versus `du`/`dir`/`dein`.
"This brand uses the formal register" is therefore a closed set of pronouns and
costs a regular expression, not a judge call. That turns one whole class of
brand-voice violation from a subjective read into a deterministic check. There
is a useful asymmetry as well: every ambiguity in German register detection sits
on the *formal* side — lowercase `sie` is "she"/"they", a sentence-initial `Sie`
is indistinguishable from it, `Ihr` is both the polite possessive and
"her"/"their" — but detecting a *violation* of a formal brand voice only
requires finding the informal pronouns, and those are unambiguous. The common
case takes the high-precision path.

Choosing German also forced `locale` to be a dimension of the client profile
from the first module rather than a patch applied later. English is present as a
mirrored corpus and a second rule implementation precisely so that the
language-dependence is visible instead of assumed away; English has no
equivalent of `Sie`/`du` and degrades to weaker register cues, and the design
says so rather than pretending the check ports.

**The corpus separates statutory rules from company policy, deliberately.**
Every German knowledge-base document that touches law carries two sections: one
headed with the statutory position (`Gesetzliche Vorgaben …`, stating that it
holds independently of the operator's terms) and one headed with the operator's
own terms (`Unsere Vertragsbedingungen`). Telling a customer that the law
requires something it does not is a different kind of harm from being merely
off-brand — it is a claim about their rights against a third party, and it
survives the conversation. The distinction is load-bearing enough that authoring
the corpus produced a correction of exactly this kind: the requirement that a
`Widerruf` be declared in `Textform` was demoted from statutory to company
policy, because § 355 BGB imposes no form requirement on the declaration at all.
A layer that flattens "the law says" and "our contract says" into one bucket of
"policy facts" cannot catch that class of error, so the corpus does not flatten
it.

## 2. Failure modes addressed

The registry contains six guards. Its stage and tier shape is part of the
configuration contract and is asserted by tests.

| Guard | Stage | Tier | Failure mode |
|---|---|---:|---|
| `injection` | INPUT | 0 | User instruction override, role reassignment, encoded payloads and cross-turn assembly |
| `document` | RETRIEVAL | 0 | Indirect instructions inside retrieved documents |
| `persona` | OUTPUT | 0 | Address form, emoji, forbidden language, sentence length and TTS hazards |
| `grounding` | OUTPUT | 0 | Unsupported numbers, prices, dates, durations and commitments |
| `pii` | OUTPUT | 0 | Outbound personal data not present in the customer turn |
| `tone` | OUTPUT | 1 | Formal, empathetic and precise brand tone that deterministic rules cannot establish |

The injection split is deliberate. User text is adversarial input and must be
checked before retrieval. A retrieved instruction instead means the knowledge
base was compromised or mis-authored; it is checked after search but before any
model call. The per-turn nonce delimiter remains an independent prompt-side
mitigation. It prevents document text from closing its own untrusted region,
while `DocumentGuard` prevents the poisoned chunk from reaching generation at
all.

Grounding is deterministic for the domain facts that cause concrete customer
harm: entities in the reply must be supported by same-locale retrieved chunks,
and commitments must be on the client's allow-list. Persona and PII checks use
the same evidence-first contract. Tone remains a separate tier-1 guard because
formality as a broad style, empathy and precision are not reducible to the
closed grammatical and lexical sets used by `PersonaGuard`.

## 3. Architecture

### 3.1 Verdicts, not booleans

A guard does not return a boolean and it does not return a decision. It returns
a `Verdict`: an outcome, a severity, and the `Evidence` — the finding kind and
the character span in the original text — that justifies it. The orchestrator
aggregates the verdicts of one stage and maps the aggregate onto exactly one
`Action` using the client profile's routing table.

**This was originally the other way round, and getting it wrong first is what
makes the argument concrete.** In the first design each guard returned
`ALLOW / REWRITE / HANDOVER / BLOCK` and decided for itself. The consequence
only becomes visible on the second client: serving a client whose escalation
rules differ would mean editing guard code, which means routing policy is
scattered across the detection layer, which means detection and policy cannot be
audited separately — and the multi-tenant claim that the whole configuration
layer rests on is simply false. Guards now report what they saw; profiles say
what it is worth.

The same principle recurs one layer down, which is the sign it is the right one:
the completion provider surfaces `stop_reason` as a fact on the result rather
than raising on a refusal. A component that observes reports; a component that
owns policy decides.

**`outcome` and `severity` are independent, and that independence is a
guarantee.** `outcome` records what the guard observed; `severity` records what
this client wants done about it. A client may override an emoji finding to
`none`, and the routing table will send that turn straight through with
`continue` — but the verdict in the trace still says an emoji was found, at the
span where it was found. Configuration can change the consequence of a finding;
it cannot change the fact of it. If configuration could suppress the finding
itself, the trace would report that nothing happened, when what actually
happened is that something was found and deliberately allowed — and an audit
needs both facts, not the second one silently collapsed into the first.

Fail-open and fail-closed ride on the same mechanism rather than a second one.
A timeout does not set a flag; it produces a severity. `on_timeout: none` routes
through the ordinary table to `continue`, which *is* fail-open. `on_timeout:
high` routes wherever the table sends HIGH, which *is* fail-closed. Per-guard
overrides then compose for free: the PII guard stays fail-closed even in voice
mode, where everything else fails open. Guards only ever return `PASS` or
`FAIL`; `ERROR`, `TIMEOUT` and `SKIPPED` are constructed by the orchestrator
from the resolved profile, so no guard ever holds a piece of the policy.

That last point had a bug in it that a test caught rather than a review. The
severity aggregation originally excluded non-conclusive verdicts, on the
reasoning that a check which did not finish has no opinion about the content.
The reasoning was sound while a severity could only originate in a guard. The
orchestrator changed the premise — it *stamps* timeouts and errors with the
client's configured severity — so excluding them meant the policy was computed
and then discarded one step later, and fail-closed was a no-op. A pipeline test
asserting that the PII guard's `critical` timeout override reaches the routing
table is what surfaced it.

### 3.2 Tiered cascade

Every guard declares a `tier`. Tier 0 is deterministic: regular expressions,
lexicon lookups, entity extraction and set comparison. Tier 1 calls a model for
qualities those mechanisms do not establish. The shipped tier-1 guard evaluates
the dimensions already declared in `PersonaSpec.tone`; it does not create a
second brand specification.

The cascade is capped by mode, not by guard-specific channel lists.
`max_tier: 0` in voice and `max_tier: 1` in chat lets the orchestrator gate a
guard before invocation, so voice never pays for a disallowed judge call.
`judge_min_budget_ms` rejects a profile that enables tier 1 inside an impossible
budget. A missing structured judge client raises; the pipeline converts that
exception to an `ERROR` verdict at the profile's `on_error` severity instead of
letting an unperformed check look like PASS.

The measured total for all tier-0 stages is about 2.3 ms. The selected tier-1
judge has a 3583 ms median, which is the empirical reason determinism remains
the primary mechanism and voice excludes the judge.

### 3.3 Latency budgets

There is **one absolute deadline per stage, shared by every guard in it**, not
a separate allowance for each guard. Tasks run concurrently, are cancelled
against that shared deadline, and their verdicts return in registry order so a
trace is stable across runs.

| Mode | Budget | Cascade | On timeout / error |
|---|---:|---|---|
| `voice` | 150 ms | tier 0 only | `none` — fail open (PII overrides to `critical`) |
| `chat` | 5000 ms | tiers 0 and 1 | `high` — fail closed |

The original chat budget was 1500 ms, chosen before any judge existed. Forced-
tool measurements on 2026-08-25 put `claude-sonnet-5` at 3583 ms median and
`claude-opus-5` at 6303 ms, with the same decisions in three constrained
probes. Leaving the budget unchanged would make every normal OUTPUT stage time
out and route HIGH to handover. The configuration therefore selects Sonnet and
reserves 5000 ms.

Voice remains at 150 ms and `max_tier: 0`. The selected judge's measured median
is more than 24 times that budget, so the exclusion is a measurement result,
not an intuition about conversational latency. The voice mode is consequently
the weaker policy; streaming deterministic checks are the credible path to
improving it without inserting several seconds before speech.

### 3.4 Configuration-driven client profiles

A profile is the whole of what varies between deployments: persona spec, locale,
per-guard thresholds and entity lists, the two latency budgets, and the
severity-to-action routing table. Nothing in a profile describes *how* a check is
performed. `Profile` maps the YAML directly; `ResolvedProfile` is that profile
flattened for one mode, so override resolution happens exactly once instead of
being re-implemented inside every guard.

Three profiles exist: `telco_de`, `telco_en` (the same client's second language
channel), and `gesundheit_de` — a statutory health insurer, a different client
entirely. **The second client differs in configuration only, and a test pins
that claim** (`tests/test_config.py::test_second_client_differs_only_in_configuration`).
The differences are substantive rather than cosmetic: `gesundheit_de` routes
MEDIUM to `handover` instead of `rewrite`, because a half-corrected statement
about coverage is worse than a handover; it marks ungrounded numbers and dates
`critical` rather than `high`; its `allowed_commitments` list is empty, so the
assistant may commit to nothing; and it defines no voice mode at all, so asking
it to resolve for voice raises rather than silently falling back to a default
that nobody chose.

Routing is a full five-level map, not a threshold, because severity is not one
dimension and actions do not escalate monotonically with it. In `telco_de`,
HIGH — an ungrounded pricing or policy claim — routes to `handover`, because a
person should answer questions about price and policy. CRITICAL routes to
`safe_fallback`, which *looks* less severe as an action and is the right one:
the critical findings are injection and outbound PII, and in those cases the
reply must not reach the customer **or** be dropped verbatim into a human
agent's queue. A threshold cannot express that; a map can.

`extra="forbid"` on the profile schema is deliberate: a mistyped configuration
key is the worst kind of silent failure, because the setting the author believes
they applied simply is not applied and nothing says so. A test loads a profile
containing `emojis_allowed` and asserts it is rejected.

**`brand_name` versus `name`, and why it exists.** `name` is the profile
identifier — `telco_de`. `brand_name` is what the customer is told —
`TeleNova`. They were the same field until a real call was made and the
assistant, asked "Wer sind Sie?", answered that it was the customer service
assistant *von telco_de*. The identifier had leaked into customer-facing text.
The lesson generalises past the one-line fix: a display name is part of the
client's configuration, not a reuse of the internal key, and the bug was found by
making an actual call rather than by reading the code. `telco_de` and `telco_en`
now share `TeleNova` — two channels of one client — which is itself the evidence
that the field is modelling the right thing.

### 3.5 Retrieval as a bounded dependency

**Retrieval quality upper-bounds the grounding guard.** If retrieval misses the
chunk that supports a correct answer, the grounding guard flags that correct
answer as unsupported. The turn is blocked or escalated, the customer is
inconvenienced, and the trace records a guard finding — but the defect is in
retrieval, not in the guard. This is why retrieval recall is measured
*separately* (§5) rather than folded into a single end-to-end score: a combined
number cannot tell you which of the two components to fix, and the temptation is
always to tune the guard.

The same reasoning determines the chunking strategy. Chunks follow Markdown `##`
sections rather than a fixed token count, because splitting `Tarif M` and
`29,99 EUR` into two chunks produces exactly the same false positive from a
third source. `chunk_id` is a readable slug rather than a content hash, because
a trace is read by people, and because a hash's apparent stability is illusory
here: fixing a typo changes a content hash completely, whereas a slug changes
only when the section heading genuinely changes — which is precisely when the id
*should* change.

**Lexical, not vector.** The queries in this domain are exact words: tariff
codes, clause numbers, `Kündigungsfrist`. Dense embeddings blur exactly those;
`Tarif S` and `Tarif M` can sit closer to each other than either does to the
right document. The real retrieval difficulty in German is not semantic
similarity, it is **compounding** — `Kündigungsfrist` has to match a document
that says `Kündigung` and `Frist` — and that is a morphology problem, solved by
a lexicon rather than by a 400 MB model. In production this layer would be
Elasticsearch, where the analyzer, field weights and synonym tables get tuned
against the corpus; `rank_bm25` is used here so that evaluation runs offline and
reproducibly, and a vector store over eighteen documents would be operational
overhead with no benefit.

The scorer is standard `BM25Okapi`, and it stayed standard on evidence. A
two-document test fixture produced all-zero scores, because classic BM25's idf,
`ln((N-df+0.5)/(df+0.5))`, is exactly zero when a term appears in half the
corpus — at N=2, df=1, everything scores zero. The tempting fix is to switch to
BM25+. Measuring the real corpus first showed 12 of 747 German terms with
idf ≤ 0, none of them a domain term, and identical top-1 results from both
variants on every probe query. The degenerate idf was a property of the fixture,
not of the deployment; the fixture was enlarged to five chunks and the standard
scorer kept.

**Tokenization lives in the language-rules module, not in the retriever.**
Compound decomposition, umlaut alias closure and lexicon-checked stemming are
*facts about German*, and `LocaleRules` is the protocol for facts about a
language; a retriever that owned them would have to own them again for the next
language. Putting them there also proved the locale abstraction was carrying
real weight rather than existing to serve the persona guard alone. Two
consequences fell out of that placement. First, alias folding has to be a
closure over the decomposition, not a surface-level rewrite: mapping `ä→ae` only
on the surface word means a query for `Kuendigung` misses a document that
writes `Kündigungsfrist`, because the document-side constituent `kündigung`
never generated `kuendigung`. Second, the decomposition cache is bounded at 2048
rather than unbounded, because the tokenizer does not only run over a fixed
corpus at index time — it also runs over **user queries** in a long-lived
process, where an unbounded cache grows monotonically with user input. That one
was caught by a re-review overturning an already-approved `maxsize=None`.

Expansion is always additive: the original surface word is always kept alongside
its constituents, because exact word matching is this retriever's primary
capability and any normalisation that *replaced* the original would weaken the
thing that works best.

## 4. Trade-offs considered

**Speed versus safety.** Chat reserves 5000 ms because the selected judge was
measured at 3583 ms median; the former 1500 ms budget was disproved. Voice keeps
150 ms, deterministic checks only, and fails open. The PII guard is the one
carve-out: its profile override remains fail-closed because an unchecked leak is
worse than an interrupted call. A uniform budget would make the design look
symmetrical on paper and be unusable in speech. The measured asymmetry is kept
explicit instead.

**Simplicity versus coverage.** Lexical retrieval over a hand-built German
lexicon buys transparency, offline reproducibility, near-zero latency, and the
ability to say exactly *why* a query missed. It gives up paraphrase robustness.
An embedding model would retrieve "Wie komme ich aus meinem Vertrag heraus?"
that the tokenizer cannot; the measured cost of not having one is in §5, and it
is neither zero nor catastrophic. The same trade recurs in the language rules: a
dictionary-driven decomposer rather than a morphological model means unknown
compounds degrade silently to whole-word matching — the failure direction is
conservative (recall is lost, wrong recall is not produced), but extending the
corpus means extending the lexicon, and that is human work that does not
disappear. The honest summary is that the simple choice is right at eighteen
documents and would be re-examined at eighteen hundred.

**Cost versus reliability.** Tier 0 keeps the primary factual and privacy
defences deterministic and near-zero cost. Chat additionally runs the tone
judge on generated replies, paying several seconds for a non-deterministic check
that can itself be attacked. That is acceptable for this evaluated
implementation because it makes tier gating, forced structured output, budgets
and error routing observable. It is not the intended production topology:
shadow or asynchronous judgement should measure reach and flag conversations
without blocking the current turn. The rejected alternative remains one LLM
judge doing everything, which would put even prices and PII at the mercy of the
model's worst sample.

Two smaller trades, recorded because they were deliberate rather than
overlooked. A topic-drift guard and a decaying per-session risk score were both
built and then cut: the drift detector was an embedding-distance proxy for a
problem it did not actually solve, and the decay constant was a number that
could not be justified. What replaced the risk score is concrete — concatenate
the last N user turns and re-run injection detection, which catches a payload
assembled across turns and involves no magic constants. And `Action.REWRITE`
does not distinguish a deterministic local repair (stripping markup:
microseconds, no model call) from regenerating through the model (which is what
makes rewriting unaffordable mid-call), so TTS findings escalate to handover in
voice where a local fix would have done. Widening the action vocabulary for a
problem not yet measured was rejected; the upgrade trigger is written down
instead.

## 5. How we know it works

The full, reproducible record is in `tests/test_results.md`; this section keeps
the design-level numbers in sync with it.

### Scenario corpus

The four-bucket end-to-end corpus uses local completion stubs, not network
calls. Benign traffic directly reuses the 21 real customer questions from the
recall set. Results are **21/21 benign**, **4/4 adversarial**, **3/3 grounding**
and **4/4 persona**. The observed benign block false-positive rate is 0/21 on
that set; it is not a claim about production precision.

### Retrieval recall

A hit is judged by `doc_id`: any document in the expected set within the first
*k* results succeeds, without binding to a chunk or exact ranking.

```text
exact-term            (n=16)  @1=0.88  @3=0.94  @5=1.00
limitation-derivation (n=1)   @1=0.00  @3=0.00  @5=0.00
limitation-paraphrase (n=4)   @1=0.50  @3=0.50  @5=0.75
overall               (n=21)  @1=0.76  @3=0.81  @5=0.90
```

The set intentionally retains both natural phrasings that retrieve and known
boundaries that do not. Removing the miss would fit the evaluation to the
implementation rather than measure it.

### Latency

Tier-0 measurements are INPUT 0.01 ms, RETRIEVAL 0.31 ms and OUTPUT 1.96 ms,
about **2.3 ms total**. Forced-tool judge medians are **3583 ms for
`claude-sonnet-5`** and **6303 ms for `claude-opus-5`**. Those figures drove
the model choice, 5000 ms chat budget and tier-0-only voice policy.

### Still unmeasured

Guard precision/recall on an independently labelled adversarial corpus, cost
per session, tier-1 reach on representative traffic, and latency percentiles
under concurrency remain unmeasured. A blank value means no evidence was
collected, not zero observed events.

## 6. Known limitations

### Retrieval and language

- **Compound decomposition is lexicon-driven.** A compound not covered by the
  lexicon degrades to whole-word matching. The lexicon covers the current
  corpus; extending the corpus requires extending the lexicon. The failure
  direction is conservative — degrading to exact matching costs recall, it does
  not manufacture wrong recall.
- **No derivational morphology.** The tokenizer reduces inflection against a
  lexicon, but does not bridge a verb to its corresponding noun:
  `gedrosselt` (throttled, participle) never reaches `Drosselung` (throttling,
  noun). Both phrasings are deliberately kept in the evaluation set, in order to
  record where the boundary is rather than hide it.
- **Purely lexical retrieval is weaker on paraphrase queries** — measured, not
  assumed: of the five known-limitation cases, three still hit at k=5 and two of
  those at rank 1, because a real customer question usually still carries a
  domain noun that survives tokenization.
- **`der 1. Platz` is still split incorrectly** as a sentence boundary; this is
  unfixed in the language-rules module. A wider ordinal rule would swallow real
  sentence boundaries, which is the more damaging error.
- **`24/7` decomposes into two `NUMBER` entities** in both locales. It is the
  same class of extractor noise as the ordinal-splitting item above: a surface
  pattern that the number extractor is right to notice and wrong to split.
- **Currency recognition is currently coupled into the language rules.**
  Currency is region-shaped knowledge, not language-shaped; its correct home is
  the `region` field the profile does not have yet.
- **The compound-decomposition cache is bounded**, because the tokenizer does
  not only run over a fixed corpus at index time — it also runs over user
  queries in a long-lived chat process, where an unbounded cache would grow
  without limit with user input.

### Sourced from vendor documentation, not measured here

- `thinking` is left at the model's default rather than disabled. Per Anthropic's
  published documentation, disabling it on the current generation of models has
  two failure modes — a tool call can land in visible text instead of being
  emitted, and internal tags can leak into the output — so lowering effort is the
  supported latency control. **This is a judgement drawn from documentation, not
  a behaviour this project measured**, and the two must not be conflated.

### Relay behaviour observations

**Verified 2026-08-22. These apply only to the third-party relay endpoint,
account routing, calling convention, SDK version and model identifier configured
at that time. They do not represent the official endpoint, other channels, or
future behaviour.**

- A request carrying an invalid `effort` value received no parameter error, and
  no verifiable constraining effect was observed with valid values either. This
  project therefore does not rely on that field to enforce judge effort.
- The json-schema output format did not, within the tested scope, force a
  schema-conforming result — it returned plain text instead. The tier-1 judge
  therefore uses forced tool choice, verified on the same day within the same
  scope.
- A request carrying `max_tokens=32` returned roughly 700 characters of text with
  a `stop_reason` of `"end_turn"` — that is, this relay endpoint did not truncate
  at 32 tokens. The path where "thinking exhausts the budget and the body comes
  back empty" therefore cannot be reproduced against this endpoint with a live
  call; its correctness rests on construction (the protocol requires
  `stop_reason` to be present) and on unit tests, not on evidence observed from
  an actual call.
- One candidate model identifier returned "no available channel" under the
  routing in effect at the time; the planned model comparison was therefore
  adjusted to two models actually reachable on that route. This is an
  availability-driven change to an experiment, and is not a claim that the model
  is unavailable elsewhere.

## 7. Next production work

1. **Shadow mode.** Run new guards in report-only beside enforcement, measure
   reach and false positives, and make rollback a configuration decision.
2. **Adversarial testing of the judge.** Forced tool choice constrains output
   shape, not the integrity of the judgement; tier 1 remains the untested model
   link in the defence chain.
3. **Streaming per-sentence checks.** Verify complete sentences before emitting
   them so voice can gain deterministic protection without a multi-second pause.

Further out — automated red-team generation, a trace visualiser, more locales —
are worth doing and are deliberately not in the top three.
