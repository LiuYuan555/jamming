# Agent Development History

This file is the durable record of major agent changes and their measured effect on the official public evaluation set. Update it whenever a change alters retrieval, conversation policy, state handling, personalization, model use, or another scored behavior.

## Update protocol

For each major change:

1. Run `python3.11 -m unittest discover -s tests -v`.
2. Run the unchanged evaluator with `python3.11 -m evaluator.local_evaluator --output <change>_results.json`.
3. Record the overall and per-scenario metrics below.
4. Describe exactly what changed, including dependencies, token usage, and known risks.
5. Keep generated result JSON files local; `.gitignore` excludes them.
6. Do not modify the evaluator or `data/public_set.jsonl` when reporting scores.

Metrics use the official formula:

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

## Score progression

| Version | Major change | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Score change |
|---|---|---:|---:|---:|---:|---:|---:|
| Official baseline | Stateless weak BM25 | 0.125000 | 0.068034 | 9.810 | 0.1190 | 0.106710 | — |
| Step 1 | Conversation state and open clarification | **0.860000** | **0.554147** | **3.575** | **0.7425** | **0.744744** | **+0.638034** |
| Step 2 | Typed state, policy ablation, and constraint reranking | **0.970000** | **0.664567** | **2.535** | **0.8465** | **0.853670** | **+0.108926** |

## Official baseline — Stateless weak BM25

**Recorded:** 2026-08-27

**Organizer source revision:** `34078351e1c3615e5505a2e829600b56a542e462`

### Behavior

- Indexed product title, categories, features, details, store, and description using SQLite FTS5.
- Searched only terms from the latest customer message.
- Joined terms using an OR query.
- Stored no conversation information beyond whether `reset` had been called.
- Ignored the anonymized user profile.
- Returned `ask_attribute: None` on every turn.
- Used no model, external API, or tokens.

### Overall metrics

| Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore | Tokens |
|---:|---:|---:|---:|---:|---:|
| 0.125000 | 0.068034 | 9.810 | 0.1190 | 0.106710 | 0 |

### Per-scenario metrics

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.237500 | 0.126508 | 8.625 |
| Browsing | 80 | 0.025000 | 0.004514 | 10.750 |
| Intent Override | 30 | 0.133333 | 0.104167 | 10.066667 |
| Boundary | 10 | 0.000000 | 0.000000 | 11.000 |

### Main limitation

After a miss, the baseline did not request an attribute. The customer therefore supplied no new product constraint, and the agent repeatedly searched unhelpful follow-up text.

## Step 1 — Conversation state and open clarification

**Recorded:** 2026-08-27

**Status:** Implemented and locally validated

### Changes

- Replaced the session-ID set with a separate `SessionState` for each conversation.
- Stored customer messages, accumulated query terms, seen terms, asked attributes, and the anonymized profile.
- Extracted the useful payload from deterministic clarification and override replies.
- Ignored known “no preference” and unsuccessful-recommendation boilerplate.
- Searched BM25 using accumulated terms from the whole useful conversation rather than only the latest message.
- Returned a natural clarification question and recommendations in the same turn.
- Used `ask_attribute: "other"` as an open clarification channel that can reveal up to two remaining constraints.
- Added focused tests for clarification output, cross-turn accumulation, and boilerplate handling.
- Kept the implementation on the Python standard library with zero model tokens and no network requirement.

### Overall metrics

| Metric | Baseline | Step 1 | Absolute change |
|---|---:|---:|---:|
| Hit Rate@10 | 0.125000 | **0.860000** | **+0.735000** |
| MRR | 0.068034 | **0.554147** | **+0.486113** |
| MTTC | 9.810 | **3.575** | **−6.235 turns** |
| Efficiency | 0.1190 | **0.7425** | **+0.6235** |
| TechnicalScore | 0.106710 | **0.744744** | **+0.638034** |

The TechnicalScore is approximately **6.98 times** the official baseline score.

### Per-scenario metrics

| Scenario | Sessions | Baseline Hit → Step 1 Hit | Baseline MRR → Step 1 MRR | Baseline MTTC → Step 1 MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.237500 → **0.850000** | 0.126508 → **0.509663** | 8.625 → **3.225** |
| Browsing | 80 | 0.025000 → **0.862500** | 0.004514 → **0.517996** | 10.750 → **3.475** |
| Intent Override | 30 | 0.133333 → **0.866667** | 0.104167 → **0.683056** | 10.066667 → **4.600** |
| Boundary | 10 | 0.000000 → **0.900000** | 0.000000 → **0.812500** | 11.000 → **4.100** |

### Validation

```text
Tests: 6 passed
Public sessions: 200
Prompt tokens: 0
Completion tokens: 0
Evaluator modified: no
Public labels modified: no
```

### Interpretation

Most of the improvement comes from opening the clarification channel and retaining the resulting constraints. Browsing improved particularly strongly because its first message is intentionally vague.

### Known limitations and next experiment

- The question policy always selects `other`; it is effective for the released protocol but is not yet adaptive or category-aware.
- The parser recognizes current simulator response markers, with a full-message fallback for unknown wording.
- Constraints are accumulated as search terms rather than tracked as typed, active, rejected, or replaced values.
- The next major experiment should add structured constraint state and explicit Intent Override handling, followed by a clean ablation against Step 1.

## Step 2 — Typed state, policy ablation, and constraint reranking

**Recorded:** 2026-08-27

**Status:** Implemented, selected as the default, and locally validated

### Hypothesis

Step 1 proved that conversation state and clarification matter. The next experiment tested whether performance could improve further by:

1. selecting a better fixed clarification policy;
2. representing constraints as typed, auditable state;
3. retrieving a wider BM25 pool and reranking it using exact constraint evidence.

### Changes

- Added configurable clarification policies: none, always `other`, always `feature`, always `material`, feature-first sequence, and other-first sequence.
- Added typed constraint records with attribute, source, turn, status, and retraction metadata.
- Added explicit no-preference tracking.
- Added conservative parsing for Buying, Browsing, clarification, retry, Boundary, free-text, and Intent Override messages.
- Added explicit override handling that retracts only the original superseded preference.
- Widened BM25 retrieval from 10 to 100 candidates.
- Added an offline constraint reranker using:
  - attribute-specific catalog fields;
  - normalized exact-phrase matching;
  - token-overlap scoring;
  - hard-miss and negation penalties;
  - numeric budget handling;
  - a BM25 retrieval-rank prior.
- Added configurable component and weight switches for controlled ablations.
- Added session tracing that reports target BM25 and reranked positions for failure analysis.
- Kept the system on the Python standard library with no model, API, network, or token requirement.

### Clarification-policy ablation

The first experiment held Step 1 retrieval and state logic fixed and changed only the question policy.

| Question policy | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| None | 0.195 | 0.136575 | 9.285 | 0.172772 |
| Always material | 0.515 | 0.354492 | 6.495 | 0.453948 |
| Always feature | 0.565 | 0.363935 | 6.045 | 0.490781 |
| Other-first sequence | 0.860 | 0.547063 | 3.670 | 0.740719 |
| Always other | 0.860 | 0.554147 | **3.575** | 0.744744 |
| Feature-first sequence | 0.860 | **0.576899** | 3.890 | **0.745270** |

Feature-first won this isolated comparison by only `0.000526`. It improved MRR but required more turns. The difference was considered too small to trust without checking it in the full combination.

### Component and combination ablation

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Step 1: accumulated terms + always other | 0.860 | 0.554147 | 3.575 | 0.744744 |
| Typed state + always other | 0.865 | 0.554438 | 3.530 | 0.748231 |
| Typed state + feature-first | 0.865 | 0.561331 | 3.830 | 0.744299 |
| Typed state + reranking + feature-first | 0.960 | 0.642472 | 2.845 | 0.835842 |
| **Typed state + reranking + always other** | **0.970** | **0.664567** | **2.535** | **0.853670** |

The best full combination was not the policy that narrowly won the isolated test. Always `other` combined better with the reranker because it supplied broad constraint coverage sooner.

### Candidate-pool ablation

| BM25 candidates reranked | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---:|---:|---:|---:|---:|
| 50 | 0.955 | 0.661873 | 2.670 | 0.842662 |
| 75 | 0.965 | 0.665123 | 2.585 | 0.850337 |
| **100** | **0.970** | **0.664567** | 2.535 | **0.853670** |
| 125 | 0.970 | 0.662609 | 2.530 | 0.853183 |
| 150 | 0.970 | 0.658409 | 2.525 | 0.852023 |
| 200 | 0.970 | 0.657117 | **2.520** | 0.851735 |
| 500 | 0.970 | 0.655734 | **2.520** | 0.851320 |

Increasing the pool beyond 100 did not recover more sessions and slightly reduced MRR. Candidate pool 100 was selected from a reasonably flat region rather than from a sharp single-point gain.

Additional reranker-weight tests varied exact-phrase, token-overlap, hard-miss, retrieval-rank, and rank-decay weights. None beat the default configuration.

### Overall metrics

| Metric | Step 1 | Step 2 best | Absolute change |
|---|---:|---:|---:|
| Hit Rate@10 | 0.860000 | **0.970000** | **+0.110000** |
| MRR | 0.554147 | **0.664567** | **+0.110420** |
| MTTC | 3.575 | **2.535** | **−1.040 turns** |
| Efficiency | 0.7425 | **0.8465** | **+0.1040** |
| TechnicalScore | 0.744744 | **0.853670** | **+0.108926** |

Compared with the official baseline, the TechnicalScore is approximately **8.00 times** higher.

### Per-scenario metrics

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.950000 | 0.617262 | 2.200 |
| Browsing | 80 | 0.987500 | 0.631121 | 2.225 |
| Intent Override | 30 | 0.966667 | 0.816984 | 4.033333 |
| Boundary | 10 | 1.000000 | 0.853333 | 3.200 |

### Validation and feasibility

```text
Tests: 32 passed
Public sessions: 200
Full evaluator wall time: 10.81 seconds
Prompt tokens: 0
Completion tokens: 0
External dependencies: none
Network requirement: none
Evaluator modified: no
Public labels modified: no
```

The normalized catalog used by the reranker adds approximately 110 MB of process memory in isolation and duplicates some metadata already stored by SQLite FTS5.

### Remaining problems and concrete examples

The best combination still misses 6 of 200 public sessions: 4 Buying, 1 Browsing, and 1 Intent Override.

#### Problem 1: the target can fall outside the 100-candidate pool

`public_0020` targets `B08P4SSFX4`, a novelty “Grandma” long-sleeve T-shirt. After all constraints were revealed, its BM25 rank was 284. The reranker only sees the first 100 candidates, so it could never recover the target. Its constraints—cotton, grey, a common fabric-composition statement, and Imported—also occur in many listings.

This is a candidate-generation failure, not a Top-100 reranking failure. Increasing the global pool to 500 did not improve aggregate Hit Rate and reduced MRR, so a future solution needs better candidate generation rather than only a wider pool.

#### Problem 2: generic constraints leave too many indistinguishable products

`public_0083` targets `B0BPMCJ1RD`, a CHICZONE polyester button-down jacket. The final BM25 rank was 66, so the target reached the reranker, but it finished at reranked position 17. Constraints such as polyester, 100% Polyester, Imported, and Button closure match many products and do not identify the exact listing.

Exact lexical reranking cannot reliably break ties when every disclosed constraint is generic. Stronger category modeling, product priors, or evidence beyond the four intent-card constraints may be required.

#### Problem 3: semicolons inside a constraint are ambiguous

The customer protocol separates multiple disclosed constraints using semicolons, but individual feature strings can also contain semicolons. In `public_0020`, one fabric-composition feature is split into several apparent constraints. The search terms survive, but the original exact phrase is lost, weakening phrase scoring.

A robust parser needs a way to preserve catalog-derived feature boundaries without assuming every semicolon is a separator.

#### Problem 4: semantically correct override handling can remove useful benchmark evidence

`public_0144` targets `B08LMMDYV7`, an Urban Republic winter parka. Before the override, the target BM25 rank was 97. Correctly retracting the earlier `Zipper closure` preference moved it to rank 101, just outside the selected pool. The evaluator's target product does not change when intent is overridden, so the retracted preference can remain statistically diagnostic even though using it would be semantically questionable.

This creates a benchmark-versus-product-behavior tension. Retaining a rejected preference may improve benchmark recall but would violate the customer's stated intent in a real shopping assistant.

#### Problem 5: repeated `other` questions become useless after exhaustion

After all hidden constraints have been disclosed, the current policy continues asking `other`. Failed sessions then receive the same no-preference response through turn 10. This does not add information and makes the conversation look repetitive.

A candidate-aware policy should stop or switch strategy when `other` is marked as exhausted. However, the isolated fixed policies showed that switching too early can also reduce score, so this requires a controlled adaptive-policy experiment.

### Example of the combination working as intended

`public_0016` targets `B07PH3X7QK`, an Amazon Essentials combat boot. Initially the target was BM25 rank 206. After `other` revealed leather and Imported, it moved to rank 130. A second answer revealed Rubber sole and the distinctive phrase “Shaft measures approximately ankle-high from arch”; the target moved to BM25 rank 27 and reranked position 1 on turn 3.

This is the ideal case for the selected architecture: clarification supplies distinctive metadata, BM25 retrieves a broad candidate set, and exact constraint reranking promotes the correct product.

### Next experiment

The next major change should focus on candidate-aware clarification and failure recovery rather than increasing the pool globally. Promising directions include:

- switch questions only when the open channel is exhausted or low-value;
- preserve semicolon-containing catalog feature boundaries;
- add category-aware candidate generation;
- detect low-information generic constraint sets and use a different tie-breaker;
- evaluate whether a low-confidence fallback can search beyond the first 100 candidates without harming normal-session MRR.

## Template for future entries

Copy this section when a new major change is evaluated:

```markdown
## Step N — Change name

**Recorded:** YYYY-MM-DD
**Status:** Implemented and locally validated

### Hypothesis

What weakness should this change address?

### Changes

- Exact behavioral and architectural changes.

### Overall metrics

| Metric | Previous | New | Absolute change |
|---|---:|---:|---:|
| Hit Rate@10 | | | |
| MRR | | | |
| MTTC | | | |
| Efficiency | | | |
| TechnicalScore | | | |

### Per-scenario metrics

Include Buying, Browsing, Intent Override, and Boundary results.

### Validation and feasibility

- Tests:
- Tokens:
- Latency:
- Dependencies:
- Network requirement:

### Interpretation

State what improved, what regressed, and whether the hypothesis was supported.

### Known limitations and next experiment

- Remaining risks and the next controlled change.
```
