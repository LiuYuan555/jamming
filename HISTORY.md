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
