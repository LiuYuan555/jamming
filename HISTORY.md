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
| Step 3 experiment | OpenAI semantic queries and weighted hybrid retrieval (best nonzero dense weight) | 0.970000 | 0.657446 | 2.510 | 0.8490 | 0.852034 | −0.001636 |
| Step 3 follow-up | Information-aware dense decay (5% at one constraint, then 0%) | 0.970000 | 0.664659 | 2.530 | 0.8470 | 0.853798 | +0.000128 |
| Step 4 experiment | Equal-weight global/category BM25 RRF | 0.970000 | 0.665153 | 2.535 | 0.8465 | 0.853846 | +0.000176 |
| Step 5 experiment | Unscored category filter and constraint-query variants (best clean route) | 0.970000 | 0.638498 | 2.465 | 0.8535 | 0.847249 | −0.006421 |

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

## Step 3 experiment — OpenAI semantic queries and weighted hybrid retrieval

**Recorded:** 2026-08-28

**Status:** Implemented and locally validated as an optional experiment; not selected as the default

### Hypothesis

Dense semantic retrieval could recover products whose catalog wording differs
from the disclosed customer constraints, while BM25 would preserve precision on
exact metadata. Weighted reciprocal-rank fusion (RRF) avoids directly combining
incompatible raw BM25 and cosine score scales.

### Changes

- Precomputed normalized 256-dimensional `text-embedding-3-small` vectors for all 50,000 catalog products.
- Used `gpt-5.6-luna` to rewrite active typed constraints into compact semantic queries.
- Cached semantic query text and embeddings by constraint state and model configuration.
- Added exact NumPy cosine search over the 49 MB memory-mapped catalog matrix.
- Fused BM25 and dense ranks with configurable weighted RRF before the existing constraint reranker.
- Added resumable/concurrent catalog building, public query-cache warming, and fusion-weight ablation tools.
- Preserved the original offline pipeline as the default and as a no-network fallback.

The one-time catalog build consumed 10,819,987 embedding input tokens. Warming
all 600 unique public conversation states consumed 96,891 combined prompt/input
tokens and 14,969 query-model completion tokens with no errors.

### Fusion-weight ablation

| Sparse weight | Dense weight | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---:|---:|---:|---:|---:|---:|
| **1.00** | **0.00** | **0.970** | **0.664567** | 2.535 | **0.853670** |
| 0.95 | 0.05 | 0.970 | 0.647129 | 2.515 | 0.848839 |
| 0.90 | 0.10 | 0.970 | 0.655165 | 2.515 | 0.851249 |
| 0.85 | 0.15 | 0.970 | 0.657446 | 2.510 | 0.852034 |
| 0.80 | 0.20 | 0.970 | 0.647401 | 2.495 | 0.849320 |
| 0.60 | 0.40 | 0.970 | 0.638288 | 2.475 | 0.846986 |
| 0.50 | 0.50 | 0.965 | 0.629399 | 2.505 | 0.841220 |
| 0.40 | 0.60 | 0.950 | 0.605010 | 2.570 | 0.825103 |
| 0.20 | 0.80 | 0.695 | 0.458272 | 4.755 | 0.609882 |
| 0.00 | 1.00 | 0.710 | 0.456786 | 4.620 | 0.619636 |

### Interpretation

The hypothesis was not supported by global fusion on the public set. A 15%
dense contribution slightly improved MTTC (`2.535` to `2.510`) while preserving
Hit Rate, but reduced MRR enough to lower the total score by `0.001636`. Larger
dense weights increasingly displaced exact catalog matches. Dense-only
retrieval was substantially weaker than the selected lexical pipeline.

The result does not show that semantic retrieval has no value. It shows that a
global dense weight is too blunt for this catalog and simulator. A future test
should trigger dense candidate generation only when lexical candidate recall or
confidence is low, rather than perturbing every well-performing BM25 session.

### Information-aware dense-decay follow-up

A follow-up tested whether dense weight should decline as the conversation
reveals more information. Four initial schedules and two gentler schedules were
evaluated using both total active-constraint count and distinct-attribute count.
Each four-value schedule applies at 1, 2, 3, and 4+ known items.

| Schedule basis | Dense schedule | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---:|---:|---:|---:|
| Fixed sparse-only | `0, 0, 0, 0` | 0.970 | 0.664567 | 2.535 | 0.853670 |
| Constraints | `0.40, 0.25, 0.10, 0` | 0.970 | 0.627835 | **2.475** | 0.843850 |
| Constraints | `0.20, 0.15, 0.10, 0` | 0.970 | 0.641637 | 2.495 | 0.847591 |
| Constraints | `0.15, 0.10, 0.05, 0` | 0.970 | 0.643821 | 2.505 | 0.848046 |
| Constraints | `0.10, 0.05, 0, 0` | 0.970 | 0.654998 | 2.520 | 0.851099 |
| **Constraints** | **`0.05, 0, 0, 0`** | **0.970** | **0.664659** | **2.530** | **0.853798** |
| Attributes | `0.05, 0, 0, 0` | 0.970 | 0.664659 | 2.530 | 0.853798 |

The gentlest schedule improved the public TechnicalScore by only `0.000128`.
It changed two Browsing sessions: `public_0006` moved from turn 2/rank 7 to
turn 1/rank 9, and `public_0014` improved from rank 5 to rank 4 on turn 1.
Buying, Intent Override, and Boundary metrics were unchanged. Constraint and
attribute bases tie because the schedule activates only when exactly one item
is known.

This is evidence for a narrowly targeted semantic cold-start route, but the
gain is too small and concentrated in two public examples to justify replacing
the selected sparse default without held-out validation. More aggressive decay
schedules found targets earlier but suffered substantial MRR losses because an
early low-ranked hit ends the session before later sparse-only turns can recover
the rank.

### Validation and feasibility

```text
Tests: 39 passed with hybrid dependencies
Public sessions per weight: 200
Catalog embeddings: 50,000 × 256 float32, 49 MB
Catalog embedding norms: all nonzero and within float32 tolerance of 1.0
Dependencies: numpy 2.4.6, openai 3.5.0
Live network required for unseen semantic queries: yes
Offline/cached BM25 fallback: yes
Evaluator modified: no
Public labels modified: no
```

## Step 4 experiment — Category-scoped BM25 and global/category rank fusion

**Recorded:** 2026-08-28

**Status:** Implemented and locally validated as an optional route; global BM25 remains the default

### Hypothesis and routes

Restricting retrieval to the active catalog category could promote targets
buried below the global BM25 Top 100. The experiment compared the unchanged
global route, strict category-only BM25, and equal-weight global/category RRF
with `rrf_k=60`. Every normalized category token had to occur in the catalog
`categories` field, and all routes passed their Top 100 to the unchanged
constraint reranker.

### Results

| Route | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Global BM25 | **0.970000** | 0.664567 | **2.535** | 0.853670 |
| Category only | 0.945000 | 0.658780 | 2.735 | 0.835434 |
| Global/category RRF | **0.970000** | **0.665153** | **2.535** | **0.853846** |

On the fixed 60-session holdout, global/category RRF scored `0.828815` versus
`0.822042` globally, driven by MRR (`0.634940` versus `0.611250`) rather than
additional hits. Category-only scored `0.805058`.

### Category candidate recall

The target entered the category-scoped Top 100 at least once in 191 of 200
sessions (`0.955` candidate recall). It never entered that pool in nine:

| Session | Target | Scenario |
|---|---|---|
| `public_0002` | `B071X54486` | Intent Override |
| `public_0017` | `B089RXP8K2` | Buying |
| `public_0020` | `B08P4SSFX4` | Buying |
| `public_0028` | `B0B9ZYDDZ1` | Buying |
| `public_0083` | `B0BPMCJ1RD` | Buying |
| `public_0120` | `B08GPGX2QG` | Browsing |
| `public_0145` | `B00IJZZWGA` | Buying |
| `public_0174` | `B0794VPVBH` | Buying |
| `public_0198` | `B08K1ZJZ4N` | Intent Override |

Category-only missed 11 final recommendations. The other two targets,
`public_0087` and `public_0144`, entered the scoped pool but the reranker did
not promote them into the final Top 10. Category-only recovered none of the six
global recommendation failures and lost five sessions found by global BM25.
Equal-weight fusion retained exactly the same six hits and misses as global.

### Interpretation

Strict category scope is not a candidate-recall solution: large categories can
still contain more than 100 stronger lexical matches. Fusion is safer and has a
small MRR benefit, but it recovered no missing target. It remains optional
rather than replacing the default. The next fallback should activate only on
low-confidence sessions and test broader category ancestors, deeper conditional
pools, or complementary exact-field routes.

### Validation

```text
Tests: 45 passed
Prompt/completion tokens: 0
Network requirement: none
Evaluator or public labels modified: no
```

Full results and per-turn traces are in
`artifacts/category_retrieval_ablation.json`.

## Step 5 experiment — Clean category filtering and constraint-query fusion

**Recorded:** 2026-08-28

**Status:** Implemented and evaluated; not selected as the default

### Design

Category membership was moved into an unscored SQL filter. Five routes used the
same Top-100 cutoff, constraint reranker, question policy, and evaluator:

- global BM25 control;
- clean category filter with all non-category constraint tokens joined by OR;
- hard tokens joined by AND plus soft tokens joined by OR;
- one scoped BM25 ranking per constraint with equal-weight RRF;
- equal-weight RRF between global BM25 and clean category OR.

### Full-public results

| Route | Hit Rate@10 | MRR | MTTC | Candidate recall@100 | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| **Global control** | **0.970** | **0.664567** | 2.535 | **0.995** | **0.853670** |
| Clean category OR | **0.970** | 0.638498 | **2.465** | **0.995** | 0.847249 |
| Hard AND + soft OR | 0.960 | 0.630589 | 2.565 | 0.985 | 0.837877 |
| Per-constraint RRF | 0.955 | 0.641401 | 2.585 | 0.980 | 0.838220 |
| Global/clean-category RRF | **0.970** | 0.630351 | 2.475 | **0.995** | 0.844605 |

The fixed 60-session holdout confirmed the result: global scored `0.822042`,
versus `0.808661` for clean OR, `0.808502` for hard/soft, `0.789054` for
per-constraint RRF, and `0.813758` for global/clean fusion.

### Interpretation

The clean filter corrected the previous scored-category query and brought eight
of its nine diagnostic candidate misses back inside the scoped Top 100. However,
those eight targets were already available to global BM25. Across all sessions,
both global and clean OR missed only `public_0020` at the candidate stage and
ended with exactly the same six recommendation misses.

Clean OR found some targets earlier, reducing MTTC, but changed 54 final ranks
and lowered MRR substantially. Hard AND was brittle because catalog wording did
not always contain every extracted hard token. Per-constraint RRF rewarded
generic products that appeared across several routes and introduced three new
recommendation misses. Global/clean fusion could not recover `public_0020`
because neither source retrieved it, while RRF perturbed otherwise strong global
ordering.

The remaining six global failures divide cleanly: `public_0020` is the only
candidate-generation miss; `public_0083`, `public_0087`, `public_0144`,
`public_0145`, and `public_0174` enter the global Top 100 but fail final
reranking. Future work should therefore use a conditional deeper fallback for
the first case and better generic-evidence tie-breaking for the other five.

### Validation

```text
Tests: 47 passed
Prompt/completion tokens: 0
Network requirement: none
Evaluator or public labels modified: no
```

Complete results are stored locally in `artifacts/clean_category_ablation.json`.

## Step 6 experiment — Learned clarification policy

**Recorded:** 2026-08-29

**Status:** Implemented and evaluated; not selected as the default

### Design

Three isolated tracks tested whether the next `ask_attribute` can be selected
from observable conversation state and Top-100 retrieval diagnostics:

- a catalog information-gain heuristic;
- a counterfactual data builder that branches every state over all ten legal
  actions and reruns retrieval/reranking on the next customer turn;
- a 1,825-parameter NumPy MLP trained to estimate action utility, with a
  confidence fallback to `other`.

Target identifiers, hidden intent cards, scenario labels, and target catalog
fields are used only to generate offline labels. They are excluded from model
features. Train, validation, and test groups are split by a deterministic hash
of `parent_asin`.

### Public information-gain ablation

| Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| **Always `other`** | **0.970** | **0.664567** | **2.535** | **0.853670** |
| Information gain | 0.960 | 0.659528 | 2.775 | 0.842358 |
| Information gain, repeats allowed | 0.960 | 0.663944 | 2.665 | 0.845883 |

The best information-gain variant improved 18 sessions and regressed 21.

### Counterfactual and neural results

The 200 public sessions produced 2,650 action rows over 265 decision states.
Only 54 states had a unique best action, and `other` was among the best tied
actions in 83.4% of states.

A separate catalog-only augmentation run sampled 500 non-public products and
produced 6,000 action rows over 600 states. It had almost the same structure:
`other` was among the best actions in 83.5% of states, and only 117 states had
a unique winner.

The validation-selected 112-feature, hidden-16 MLP used 1,825 parameters. On
the untouched catalog-product test split:

| Policy | Best-action accuracy | Mean one-step regret |
|---|---:|---:|
| **Always `other`** | **0.827273** | **0.061964** |
| Learned MLP | 0.818182 | 0.064616 |

Confidence fallback can recover the fixed baseline by reverting uncertain
decisions to `other`, but no tested setting improves it.

### Interpretation

Catalog augmentation increases sample count but does not correct the underlying
label imbalance. In the released simulator, `other` reveals up to two arbitrary
constraints while named actions reveal only matching constraints. The learned
model therefore converges toward the same broad action and generalizes slightly
worse when it departs from it.

The experimental modules remain available for future simulator variants, but
the production agent keeps `always_other`. The next useful extension would be
multi-step dynamic-programming labels or an action protocol where `other` is no
longer a wildcard.

### Validation

```text
Tests: 69 passed
Model parameters: 1,825
Runtime dependency: NumPy only for the optional learned policy
Network requirement: none
Evaluator or public labels modified: no
```

## Step 7 experiment — Observable intent-aware hybrid retrieval

**Recorded:** 2026-08-29

**Status:** Implemented as a presentation prototype; not selected as the default

### Architecture

- Trained a 1,640-parameter, 34.6 KB NumPy softmax model over hashed conversation
  features and asked-attribute history.
- Classified the intent observable through the current turn as Buying, Browsing,
  Intent Override, or Boundary. Buying/Browsing remain provisional until a
  correction or delegation event is actually observed.
- Routed calibrated probabilities into conservative sparse/dense RRF schedules.
- Added a Shannon-entropy question gate, guarded semantic rewrite prompts, and
  an opt-in intent-specific Top-10 layer.
- Preserved sparse-only retrieval, `always_other`, and deterministic query text
  as the network-free production fallback.

### Classifier result

The product-disjoint synthetic run used 16,000 turns from 1,000 non-public
catalog products. On 800 public simulator turns, observable-label accuracy and
macro F1 were both `1.0000`; hidden-scenario accuracy was `0.67125` with macro
F1 `0.628994`. The difference is causal, not merely model error: pre-event
Boundary messages are Browsing, and pre-event Override messages are Buying.
The perfect observable score is template-matched and requires a manually
paraphrased/OOD set before any natural-language accuracy claim.

### Retrieval and question result

| Policy | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| **Sparse-only** | **0.970** | **0.664567** | 2.535 | **0.853670** |
| Fixed 85/15 | 0.970 | 0.657446 | 2.510 | 0.852034 |
| Conservative router, trained classifier + v2 prompt | 0.970 | 0.646954 | **2.515** | 0.848786 |
| Conservative router + entropy questions | 0.960 | 0.604883 | 2.700 | 0.827465 |
| Trained classifier + v2 prompt + intent Top-10 | 0.970 | 0.641808 | 2.515 | 0.847242 |

Intent routing slightly improved time to first hit, but dense candidates
perturbed exact ranking and reduced MRR. Entropy questions lost because the
simulator's `other` action can reveal two remaining constraints of any type.
Browsing diversity was score-neutral or slightly harmful, while Boundary's
broad category-only ordering caused the largest Top-10 regression.

### Dense rewrite prompt result

A balanced 120-state cached screen used `gpt-4o-mini` for the narrow rewrite and
the existing 256-dimensional `text-embedding-3-small` catalog vectors.

| Rewrite | Dense R@10 | Dense R@100 | Dense MRR |
|---|---:|---:|---:|
| `natural_description` | 0.283333 | **0.583333** | 0.182052 |
| `catalog_phrase` | **0.308333** | 0.550000 | **0.204654** |
| Deterministic | **0.308333** | 0.541667 | 0.190693 |

Natural description is the recall-oriented prompt; catalog phrase is the
early-rank prompt. A first-pass polarity error showed why the model input must
exclude internal flags such as `hard: false`. The final sanitizer rejects
invented negation and restores the deterministic query when explicit value
anchors are lost. The 120-state screen cost no more than approximately
`$0.01005`.

The tested prompt and sanitizer were then wired into the runtime retriever.
Warming 600 natural-description states reported 116,733 input/embedding tokens
and 12,629 completion tokens with zero errors; the final intent route added
3,885 tokens for Boundary-specific catalog prompts. Cached reruns use zero
tokens.

### Decision

Use the architecture for the presentation because it makes intent, retrieval,
question value, and fallback behavior explicit. Keep sparse-only plus
`always_other` as score mode until intent routing wins on a locked held-out set.

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
