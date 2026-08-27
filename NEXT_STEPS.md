# Track 4 Shopping Copilot: Current Next Steps

This roadmap starts from the best combination measured on the 200-session public set.

## Current best system

```text
Typed conversation state
    → always-other clarification
    → weighted BM25 Top 100
    → exact and token-overlap constraint reranking
    → Top 10 recommendations
```

| Metric | Official baseline | Step 1 | Current best |
|---|---:|---:|---:|
| Hit Rate@10 | 0.125000 | 0.860000 | **0.970000** |
| MRR | 0.068034 | 0.554147 | **0.664567** |
| MTTC | 9.810 | 3.575 | **2.535** |
| TechnicalScore | 0.106710 | 0.744744 | **0.853670** |

The system remains fully offline, uses no model or third-party dependency, and reports zero tokens.

## Completed priorities

- Controlled clarification-policy ablation
- Typed constraint state and explicit Intent Override handling
- Constraint-aware BM25 reranking

The experiments showed:

- Feature-first narrowly beat always `other` in isolation by `0.000526` TechnicalScore.
- Always `other` was clearly better in the full reranking combination: `0.853670` versus `0.835842`.
- Reranking 100 BM25 candidates was the best tested pool width.
- Pools larger than 100 did not recover more sessions and slightly reduced MRR.
- Default reranker weights beat the tested alternatives.

Full experiment tables and failure traces are recorded in `HISTORY.md`.

## Updated priority ranking

| Priority | Next step | Expected value | Effort | Reason |
|---:|---|---|---|---|
| 1 | Candidate-aware clarification and exhaustion recovery | High | Medium | Prevent repeated useless questions while preserving the broad coverage of `other` |
| 2 | Category-aware candidate generation and low-confidence fallback | Very high | Medium | Most remaining misses are outside the Top-100 pool or buried among generic matches |
| 3 | Robust constraint-boundary and paraphrase parsing | High | Medium | Semicolons, negation, and rewording can corrupt exact phrase evidence |
| 4 | Private-set confidence and held-out evaluation | Very high | Medium | The public set contains only 200 of 1,000 sessions |
| 5 | Better tie-breaking for generic constraints | Medium–high | Medium | Common values such as cotton and Imported leave many indistinguishable products |
| 6 | Cautious profile-aware cold start | Medium | Low | May help vague first turns but should never override explicit intent |
| 7 | Semantic retrieval or lightweight reranking | Uncertain | Medium–high | Useful only if it fixes measured vocabulary mismatch without blurring exact matches |
| 8 | Grounded explanations | Low score / high demo | Low–medium | Improves trust and presentation |
| 9 | Interactive demonstration | High judging value | Medium | Best after scoring and robustness behavior stabilize |

## Priority 1: Candidate-aware clarification and exhaustion recovery

### Problem

Always asking `other` gives excellent coverage, but after all constraints are disclosed the customer repeatedly says there is no additional preference. Failed sessions then make no progress through turn 10.

Feature-first is not a safe global replacement: it slightly improved isolated MRR but made the full system slower and reduced the full TechnicalScore.

### Proposed policy

Keep `other` as the default, but switch only when evidence supports it:

```text
If other has not been exhausted:
    ask other
Else if a specific unasked attribute can split the candidate pool:
    ask that attribute
Else:
    stop asking and switch retrieval strategy
```

Estimate question value using:

```text
question_value(attribute) =
    probability of receiving an answer
    × expected candidate reduction
    × remaining-turn value
```

### Required ablation

Compare against the current best:

- always `other`;
- stop after `other` is exhausted;
- switch to the highest-value specific attribute after exhaustion;
- switch retrieval strategy without asking another question.

Track Hit Rate, MRR, MTTC, repeated-question count, and per-scenario results.

## Priority 2: Category-aware candidate generation and fallback retrieval

### Problem

The reranker cannot recover a target absent from its BM25 candidate pool. Increasing the pool globally from 100 to 500 did not improve Hit Rate and reduced MRR.

Example: `public_0020` remained at BM25 rank 284 after all constraints were disclosed. It never reached the Top-100 reranker.

### Proposed approach

Use category and confidence to create better candidates rather than simply requesting more:

1. Build catalog category buckets.
2. Retrieve within the likely category when category confidence is high.
3. Combine a category-scoped result list with the global BM25 list.
4. Trigger a wider fallback only when the Top 10 remains low-confidence.
5. Fuse candidate sources using reciprocal rank or calibrated scores.

### Success criteria

- Recover targets currently outside the global Top 100.
- Preserve or improve MRR on normal sessions.
- Avoid increasing every turn's latency and memory cost.

## Priority 3: Constraint-boundary and paraphrase robustness

### Problem

The simulator uses semicolons to separate disclosed constraints, but catalog feature values can contain semicolons internally. Blind splitting destroys the original exact phrase.

Example: the fabric statement in `public_0020` contains several semicolons and becomes multiple apparent constraints.

### Proposed improvements

- Match complete reply payloads against candidate catalog feature strings before splitting.
- Use the previously requested attribute as a parsing hint.
- Preserve both the original payload and conservative typed fragments.
- Add tests for punctuation, casing, synonyms, negation, and replacements.
- Treat unfamiliar messages as uncertain evidence instead of hard filters.

### Important private-set risk

The current templates receive the strongest parsing. Organizer paraphrasing may change conversational framing even though the structured `ask_attribute` protocol remains stable.

## Priority 4: Held-out and robustness evaluation

### Required test sets

- punctuation and case perturbations;
- light and heavy paraphrases;
- explicit negation and correction cases;
- targets sampled from catalog products not used by public sessions;
- category-balanced and popularity-balanced target samples;
- no-preference and Boundary cases.

Every major change must update `HISTORY.md` with overall, per-scenario, latency, memory, cost, and failure results.

## Priority 5: Generic-constraint tie-breaking

### Problem

Some targets expose only generic metadata:

```text
polyester
100% Polyester
Imported
Button closure
```

Many products satisfy every constraint. For `public_0083`, the target entered the candidate pool at BM25 rank 66 but only reached reranked position 17.

### Options to test

- stronger category hierarchy matching;
- store or brand evidence when legitimately available;
- rating-count or popularity as a small final tie-breaker;
- diversification only when several candidates have equivalent evidence;
- uncertainty-aware questions that seek a more discriminative attribute.

Do not treat popularity as relevance without an ablation; it may exploit public sampling rather than user intent.

## Priority 6: Cautious profile-aware cold start

Use `preference_tags` only when the first customer message provides little information:

```text
explicit customer constraints
    > accumulated conversation state
    > anonymized profile tags
```

Disable or sharply reduce the profile contribution once explicit preferences arrive. Generic profile tags such as comfort and fit can otherwise overpower the actual request.

## Priority 7: Optional semantic layer

Evaluate semantic retrieval only after the lexical failure cases are categorized.

Potential options:

- catalog-derived synonym expansion;
- local dense retrieval fused with BM25;
- a small local cross-encoder over the Top 50–100;
- an optional LLM parser or reranker with a complete offline fallback.

Keep the semantic layer only if it improves both the unchanged evaluator and paraphrase robustness. The official environment may disable network access.

## Priorities 8–9: Explanations and demo

Generate explanations strictly from catalog fields and active constraints. Show:

- the structured shopping state;
- which question was selected and why;
- how each answer changes candidate count and target rank;
- Intent Override behavior;
- grounded reasons for the final recommendations.

The interface should visualize the official `Agent` implementation rather than introduce a separate scoring path.

## Recommended immediate action

Build a confidence-aware fallback that activates only after `other` is exhausted. It should combine category-scoped retrieval with the global Top 100. This directly addresses both remaining problems—repetitive clarification and missing candidates—without globally expanding the pool or adding a deep-learning dependency.
