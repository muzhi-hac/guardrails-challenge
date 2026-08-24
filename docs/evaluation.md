# Evaluation and Reflection

## What worked well

The most reusable decision was to make guards report facts and let the
orchestrator own policy. A guard emits a finding, severity and evidence; the
pipeline looks up the active profile's routing table and chooses the action.
That separation survived four apparently different design questions:

- guards do not decide whether a failure means rewrite, handover or fallback;
- the provider reports `stop_reason` instead of raising or retrying by policy;
- inbound PII redaction is a function invoked by the turn layer rather than a
  hidden side effect of the outbound PII guard; and
- a trace writer owns `error_type`, so exception messages and request content
  do not leak into archived records.

The shared absolute deadline also held up. A stage gets one deadline, not a
fresh allowance for every guard. Concurrent deterministic checks therefore
remain bounded by the channel budget, and verdict ordering stays stable because
the pipeline returns registry order rather than completion order.

Configuration-driven multi-tenancy worked in the concrete rather than only in
the schema. The health-insurance profile changes routing, commitment policy,
PII handling, tone and thresholds without changing a line of guard code. Tests
pin that difference, so “another client is configuration only” is an executable
claim.

## What limitations remain

- **The tier-1 decision has sampling variance.** In an early four-call probe,
  the same German reply received the opposite `formal` decision once. Adding a
  short-reason constraint produced three consistent calls, but did not turn the
  judge into a deterministic component. This is direct evidence for using
  deterministic checks first and a judge only for residual qualities.
- **Tier 1 is too slow for an inline production path.** The faster candidate's
  measured median was 3583 ms. The challenge implementation runs it inline so
  the tier gate, budget and failure policy are observable, but a production
  system should run this judgement asynchronously or in shadow mode and flag
  conversations for review rather than block the current turn for several
  seconds.
- **German compound decomposition is lexicon-driven.** An unseen compound
  degrades to whole-token matching, so expanding the corpus requires monitoring
  and extending the decomposition lexicon.
- **Derivational morphology is absent.** `gedrosselt` and `Drosselung` are
  unrelated tokens to this retriever. The recall set deliberately keeps both a
  phrasing that hits and one that misses so the boundary remains visible.
- **Retrieval has no relevance floor.** It filters only `score > 0`, so an
  unrelated question can still return low-scoring noise rather than an empty
  result.
- **Guard precision and recall are not yet measured.** That requires an
  independently labelled adversarial corpus, not more assertions written by
  the same person who wrote the patterns.

## What I would add with more time

1. **Shadow mode.** A new guard needs a report-only deployment path before it
   can safely enforce. This is the only item whose absence blocks a responsible
   production rollout: it supplies real reach rate, false-positive evidence and
   a reversible comparison with current behaviour.
2. **Adversarial testing of the judge itself.** Tier 1 is the only link in the
   defence chain that consumes model-interpreted data and has not been attacked
   as a target. Its forced tool schema controls shape, not judgement integrity.
3. **Streaming per-sentence checks.** This is the only path that can make voice
   materially safer without adding several seconds before the caller hears
   anything. Deterministic checks can verify each complete sentence before it
   is emitted, while the full reply can still receive deeper asynchronous
   review.

## Surprises during implementation

### Real corpus data disproved the greedy compound splitter

All 26 tokenizer unit tests passed, but a frequency sample from the real corpus
failed on four of twelve important compounds: `Servicezeiten`,
`Zahlungsarten`, `Rufnummernmitnahme` and `Entstörfrist`. The tests described
the algorithm I expected rather than the text I actually owned. Corpus-level
measurements, not another synthetic tokenizer case, exposed the gap.

### One bad test created a real guard false negative

The test claimed that a German chunk should ground an English reply even though
retrieval is partitioned by locale, so that scenario cannot occur in the
architecture. Satisfying it made the guard parse every chunk with every locale's
grammar. English parsing of German `29,99 EUR` quietly produced `99.00 EUR`, so
a fabricated German `99,00 EUR` answer appeared grounded. Foreign grammar did
not fail loudly; it produced a plausible, wrong entity. Removing the impossible
test and using only the turn's locale fixed the false negative.

### The first live injection test failed because the model behaved well

The model gave the correct price, rejected the poisoned instruction and then
disclosed the attack to the customer. Disclosure necessarily repeated the
attacker's `kostenlose` claim, so a negative vocabulary assertion labelled the
correct behaviour as a successful injection. The test now asserts the fact
that compliance would require—a zero price—rather than any word needed to
describe the rejected instruction.

### A default latency made missing measurement invisible

With `latency_ms=0.0` as a default, “the implementation forgot to measure” and
“the replay completed instantly” became the same observation. Removing the
default forced every provider implementation and test record to supply the
field explicitly.

### The original judge budget had never been tested

The 1500 ms chat budget was chosen before a judge existed. The first constrained
measurement put the faster judge at 3583 ms. Under the old configuration every
OUTPUT stage would time out and, because chat is fail-closed at HIGH, hand every
normal turn to a person. The layer would look enabled while being functionally
broken. The measured result changed the model to `claude-sonnet-5`, raised the
chat budget to 5000 ms and made the 150 ms voice exclusion a measured decision.

### A real call exposed an internal identifier to the customer

When asked “Wer sind Sie?”, the assistant called itself the customer-service
assistant of `telco_de`. Unit tests for prompt shape, retrieval and rendering
all passed because the wrong value was syntactically valid. Only asking the
model revealed that the profile identifier had crossed a customer-facing
boundary. The profile now separates `name` from `brand_name`, and a regression
test asserts that only the latter reaches the system prompt.
