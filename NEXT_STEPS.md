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
- OpenAI semantic-query and weighted sparse/dense retrieval ablation
- Static BM25 field-weight search with a stratified holdout
- Strict category-scoped BM25 and global/category RRF ablation
- Counterfactual clarification data, information-gain, and lightweight neural-policy ablation
- Observable four-class intent classifier, intent-conditioned hybrid routing,
  semantic-query prompt, and intent-specific Top-10 ablations

The experiments showed:

- Feature-first narrowly beat always `other` in isolation by `0.000526` TechnicalScore.
- Always `other` was clearly better in the full reranking combination: `0.853670` versus `0.835842`.
- Reranking 100 BM25 candidates was the best tested pool width.
- Pools larger than 100 did not recover more sessions and slightly reduced MRR.
- Default reranker weights beat the tested alternatives.
- Global hybrid fusion did not beat sparse retrieval. The best nonzero dense
  configuration was 85% sparse / 15% dense at `0.852034`, versus `0.853670`
  for sparse-only. Semantic retrieval should be reserved for a measured
  low-confidence fallback rather than applied globally.
- Information-aware decay produced a tentative public-set gain: 5% dense at
  exactly one known constraint and 0% afterward scored `0.853798`. It changed
  only two Browsing sessions, so held-out confirmation is required before
  selecting it as the default.
- Static BM25 tuning found a full-public gain that regressed on holdout, so the
  existing field allocation remains the default.
- Strict category-only retrieval scored `0.835434`, versus `0.853670` globally,
  and left nine targets outside its scoped Top 100 across every available turn.
- Equal global/category RRF scored `0.853846` and improved holdout MRR, but it
  recovered none of the six global recommendation failures. Keep it optional
  as a ranking feature rather than treating it as a recall solution.
- A true unscored category filter fixed eight of nine diagnostic scoped misses,
  but did not add recall beyond global BM25. Clean OR scored `0.847249`,
  hard/soft `0.837877`, per-constraint RRF `0.838220`, and global/clean RRF
  `0.844605`, all below the `0.853670` global control. Only `public_0020`
  remains a global candidate-generation miss; the other five global failures
  are reranking problems.
- Candidate-aware information gain scored `0.845883` at best, below the
  `0.853670` always-`other` control. A 1,825-parameter MLP trained on 6,000
  catalog-only counterfactual rows also failed its product-disjoint test:
  accuracy `0.818182` and regret `0.064616`, versus `0.827273` and `0.061964`
  for always `other`. Keep learned questioning experimental.
- The 34.6 KB observable intent classifier achieved 100% on simulator-template
  labels, but only 67.125% against hidden session scenarios because future
  Boundary and Override events are unknowable. Treat the former as a template
  integration result, not a natural-language generalization claim.
- Conservative intent routing with the trained classifier and v2 prompt scored
  `0.848786`: MTTC improved from `2.535` to `2.515`, but MRR fell from
  `0.664567` to `0.646954`. The matching intent-specific Top-10 layer also
  regressed to `0.847242`. Both stay demonstration-only.
- In a balanced 120-state screen, the guarded `natural_description` rewrite
  improved dense Recall@100 from `0.541667` to `0.583333`; `catalog_phrase`
  produced the strongest dense MRR at `0.204654`. Route prompt style by intent,
  retain exact anchors, and keep the deterministic offline fallback.

Full experiment tables and failure traces are recorded in `HISTORY.md`.

## Updated priority ranking

| Priority | Next step | Expected value | Effort | Reason |
|---:|---|---|---|---|
| 1 | Conditional deep fallback for the sole candidate miss | Focused | Medium | `public_0020` is now the only target never entering global Top 100 |
| 2 | Better reranking for five in-pool misses | High | Medium | The other five failures are available to the reranker but remain outside final Top 10 |
| 3 | Robust constraint-boundary and paraphrase parsing | High | Medium | Semicolons, negation, and rewording can corrupt exact phrase evidence |
| 4 | Private-set confidence and held-out evaluation | Very high | Medium | The public set contains only 200 of 1,000 sessions |
| 5 | Multi-step question-value labels | Uncertain | Medium–high | One-step labels are tie-heavy; only revisit if the simulator/action semantics change |
| 6 | Better tie-breaking for generic constraints | Medium–high | Medium | Common values such as cotton and Imported leave many indistinguishable products |
| 7 | Cautious profile-aware cold start | Medium | Low | May help vague first turns but should never override explicit intent |
| 8 | Conditional semantic fallback | Uncertain | Medium–high | Global dense fusion blurred exact matches; retry only on measured lexical failure |
| 9 | Grounded explanations | Low score / high demo | Low–medium | Improves trust and presentation |
| 10 | Interactive demonstration | High judging value | Medium | Best after scoring and robustness behavior stabilize |

## Completed clarification-policy finding

### Problem

Always asking `other` gives excellent coverage, but after all constraints are disclosed the customer repeatedly says there is no additional preference. Failed sessions then make no progress through turn 10.

Feature-first is not a safe global replacement: it slightly improved isolated MRR but made the full system slower and reduced the full TechnicalScore.

### Tested policy

The experiment kept `other` as a fallback and switched only when observable
candidate statistics supported a named attribute:

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

### Result

Information gain and the learned MLP both lost to always `other`. Retain the
fixed policy. Revisit only with full multi-step labels or different simulator
semantics; adding more one-step catalog samples did not remove the imbalance.

## Priority 1: Conditional deep fallback

### Problem

The reranker cannot recover a target absent from its BM25 candidate pool. After
correcting category filtering, `public_0020` is the only public target that
never enters the selected global Top 100 during its session.

The fallback should run only when global retrieval is uncertain or `other` is
exhausted, search deeper than 100, and preserve full fabric phrases so the
novelty-shirt boilerplate does not become a bag of generic numbers and tokens.

### Proposed approach

Test the following sources separately before fusing them:

1. A deeper global pool only for low-confidence sessions.
2. Exact full-feature phrase retrieval before semicolon splitting.
3. Description/features routes that must contribute a unique candidate.
4. A separately calibrated rerank prior for candidates retrieved below rank 100.

### Success criteria

- Recover targets currently outside the global Top 100.
- Preserve or improve MRR on normal sessions.
- Avoid increasing every turn's latency and memory cost.
- Require each added route to recover at least one existing global failure.

## Priority 2: Better reranking for in-pool misses

Focus on `public_0083`, `public_0087`, `public_0144`, `public_0145`, and
`public_0174`. Their targets enter the global Top 100, so additional retrieval
routes are unnecessary. Test phrase preservation, field-normalized evidence,
generic-constraint down-weighting, and a calibrated tie-breaker while retaining
the global retrieval order as a strong prior.

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

## Priority 5: Multi-step question-value labels

The current counterfactual builder scores the next retrieval turn along an
always-`other` reference trajectory. If clarification learning is revisited,
enumerate complete reachable action sequences or use dynamic programming so an
action receives credit for later recovery. Keep the same product-disjoint split
and require a gain over always `other` on untouched catalog targets before any
public-set evaluation.

## Priority 6: Generic-constraint tie-breaking

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

## Priority 7: Cautious profile-aware cold start

Use `preference_tags` only when the first customer message provides little information:

```text
explicit customer constraints
    > accumulated conversation state
    > anonymized profile tags
```

Disable or sharply reduce the profile contribution once explicit preferences arrive. Generic profile tags such as comfort and fit can otherwise overpower the actual request.

## Priority 8: Optional semantic layer

Evaluate semantic retrieval only after the lexical failure cases are categorized.

Potential options:

- catalog-derived synonym expansion;
- local dense retrieval fused with BM25;
- a small local cross-encoder over the Top 50–100;
- an optional LLM parser or reranker with a complete offline fallback.

Keep the semantic layer only if it improves both the unchanged evaluator and paraphrase robustness. The official environment may disable network access.

## Priorities 9–10: Explanations and demo

Generate explanations strictly from catalog fields and active constraints. Show:

- the structured shopping state;
- which question was selected and why;
- how each answer changes candidate count and target rank;
- Intent Override behavior;
- grounded reasons for the final recommendations.

The interface should visualize the official `Agent` implementation rather than introduce a separate scoring path.

## Recommended immediate action

Stop asking `other` after it is exhausted. Trigger a deeper retrieval fallback
only for sessions whose target-like evidence remains low-confidence; on the
public failures this should isolate `public_0020`. Separately improve reranking
for the five targets already inside the global Top 100. Do not add category or
constraint RRF globally, because both full-public and holdout tests regressed.
