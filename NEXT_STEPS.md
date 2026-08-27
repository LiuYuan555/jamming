# Track 4 Shopping Copilot: Ranked Next Steps

This roadmap prioritizes improvements to the official weak BM25 baseline. Priorities are based on expected impact on Hit Rate@10, MRR, and MTTC; implementation effort; robustness to the private evaluation set; offline compatibility; and demo value.

## Baseline weaknesses to address

The starter agent currently:

- searches only the latest customer message;
- does not remember constraints from previous turns;
- never asks a clarification question (`ask_attribute` is always `None`);
- ignores the anonymized user profile;
- uses a broad OR-based BM25 query without explicit constraint handling;
- returns no explanation for its recommendations.

These weaknesses are most visible in Browsing sessions, where the initial request is intentionally vague.

## Priority ranking

| Priority | Next step | Expected impact | Effort | Main reason |
|---:|---|---|---|---|
| 1 | Add conversation state and clarification | Very high | Low–medium | Unlocks new information and prevents every turn from repeating the same search |
| 2 | Track structured constraints and intent overrides | Very high | Medium | Converts conversation history into reliable retrieval signals and handles a scored scenario directly |
| 3 | Add constraint-aware BM25 reranking | High | Medium | Improves ranking while remaining fast, explainable, and fully offline |
| 4 | Build an adaptive question-selection policy | High | Medium | Reduces wasted turns by asking questions that are likely to receive useful answers |
| 5 | Harden parsing against paraphrases and corrections | High | Medium | Protects performance when private customer wording differs from public templates |
| 6 | Add a rigorous evaluation and ablation workflow | High | Low–medium | Prevents changes that improve a few examples while damaging the overall or private-set score |
| 7 | Use the anonymized profile as a cautious cold-start prior | Medium | Low | Can improve vague first turns without overpowering explicit customer preferences |
| 8 | Add semantic retrieval or a lightweight reranker | Uncertain–medium | Medium–high | May help vocabulary mismatch, but must prove value over strong lexical matching |
| 9 | Add grounded recommendation explanations | Low technical / high demo | Low–medium | Improves trust and presentation, although it does not directly affect the benchmark score |
| 10 | Build a polished interactive demo | High judging value | Medium | Makes the system easy for judges to understand after the scoring path is stable |

## 1. Add conversation state and clarification

### What to build

Maintain state for each `session_id`, including:

- all customer messages;
- extracted constraints;
- attributes already requested;
- attributes for which the customer had no preference;
- previously returned products, if useful;
- the anonymized user profile.

On uncertain turns, return recommendations and ask a question simultaneously:

```python
{
    "message": "Which features matter most to you?",
    "ask_attribute": "feature",
    "recommendations": [...],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

Search using the accumulated conversation state instead of only the latest message.

### Why this is first

The current baseline never requests more information. When it misses, the simulated customer asks the agent to request a specific attribute, but the baseline returns `None` again. This wastes nearly all later turns.

Clarification can improve all three core metrics:

- Hit Rate@10: the agent receives more target-specific information;
- MRR: additional constraints can move the target higher;
- MTTC: useful questions can locate the target in fewer turns.

### First experiment

Compare:

1. official stateless baseline;
2. stateful retrieval with no questions;
3. stateful retrieval that always asks `feature`;
4. stateful retrieval that asks `other` while useful constraints remain.

This separates the value of memory from the value of clarification.

## 2. Track structured constraints and intent overrides

### What to build

Represent the current shopping intent explicitly:

```python
{
    "category": [],
    "material": [],
    "color": [],
    "size": [],
    "style": [],
    "brand": [],
    "budget": [],
    "feature": [],
    "use_case": [],
    "negative_constraints": [],
}
```

For every stored constraint, also retain:

- the turn where it appeared;
- whether it came from the customer or profile;
- whether it is currently active;
- a confidence score or hard/soft designation.

Detect corrections such as:

- “Actually…”
- “Ignore my earlier preference…”
- “Instead…”
- “I changed my mind…”

Update the affected constraint rather than simply appending contradictory values.

### Why this is second

Conversation memory is only useful if the agent can distinguish active preferences from old or rejected ones. Intent Override is also an explicit evaluation scenario, representing 15% of both public and private sessions.

### Success criteria

- No unrelated constraints are lost after an override.
- The corrected preference affects the next ranking.
- Intent Override Hit Rate, MRR, and MTTC improve without reducing Buying or Browsing performance.

## 3. Add constraint-aware BM25 reranking

### What to build

Keep BM25 as a fast candidate generator, then rerank candidates using structured evidence:

```text
final_score =
    normalized_bm25
    + exact_constraint_bonus
    + category_bonus
    + soft_preference_bonus
    + quality_tie_breaker
    - hard_constraint_violation_penalty
```

Potential improvements include:

- category-scoped candidate generation;
- exact phrase matching for disclosed feature strings;
- separate title, feature, detail, category, store, and description weights;
- deterministic material, color, size, and price checks;
- strong penalties for violating explicit requirements;
- small rating or popularity tie-breakers only after relevance.

### Why this comes before embeddings

The evaluator derives customer constraints from catalog metadata. Many disclosed phrases may therefore match the target lexically. A strong exact and field-aware retriever can exploit this signal without model downloads, API cost, or network risk.

### Success criteria

- Higher MRR, not only Hit Rate@10.
- Fewer hard-constraint violations in the Top 10.
- Low CPU latency and no external dependency.

## 4. Build an adaptive question-selection policy

### What to build

Choose `ask_attribute` according to the current uncertainty rather than following one fixed sequence.

A practical policy can consider:

- whether the attribute is already known;
- whether the customer previously said they had no preference;
- how often the simulator is likely to provide that attribute;
- how strongly the attribute would divide the current candidate set;
- whether several leading candidates are otherwise tied.

Example heuristic:

```text
question_value(attribute) =
    probability_of_answer(attribute)
    × expected_candidate_reduction(attribute)
    × remaining_turn_value
```

### Recommended starting point

Prioritize `feature`, `material`, and `color`. Use `other` as a broad fallback. Avoid repeatedly asking for attributes that have already produced “no preference.”

### Why this is not Priority 1

A sophisticated question policy cannot help if the agent does not remember and use the answer. State and constraint handling must be reliable first.

## 5. Harden parsing against paraphrases and corrections

### What to build

Test and support:

- lowercase and punctuation changes;
- filler words;
- synonyms and mild paraphrases;
- negation;
- corrections and retractions;
- multiple constraints in one customer message;
- “no preference” responses;
- unexpected but valid wording.

Use normalization, token matching, field vocabularies, and conservative fallback extraction. An optional LLM parser can be evaluated later, but the primary path should remain offline.

### Why it matters

The organizer may paraphrase customer language in final evaluation. A parser that relies only on exact public message templates can score well locally and fail privately.

## 6. Add an evaluation and ablation workflow

### What to build

Record every experiment with:

- overall Hit Rate@10, MRR, MTTC, efficiency, and TechnicalScore;
- per-scenario metrics;
- median and p95 latency;
- model token use and estimated cost, if applicable;
- the exact configuration and code revision.

Run one-change-at-a-time ablations:

1. official baseline;
2. + conversation state;
3. + clarification;
4. + structured constraints;
5. + reranking;
6. + adaptive questions;
7. + profile prior;
8. + semantic retrieval.

### Why this is high priority

Public evaluation contains only 200 sessions, while 800 sessions remain private. A complicated method can overfit the public set. Ablations reveal which components genuinely help and which only add complexity.

Do not modify the evaluator or public labels when reporting results.

## 7. Use the user profile as a cautious cold-start prior

### What to build

When the opening request is vague, use `preference_tags` as weak query terms or tie-breakers. Reduce or remove their influence after the customer gives explicit constraints.

Suggested precedence:

```text
explicit current-turn constraints
    > accumulated conversation constraints
    > anonymized profile preferences
```

### Why it is medium priority

Profile tags are generic and may match many products. They can help a vague opening turn but should never override direct customer intent.

## 8. Add semantic retrieval or a lightweight reranker

### Options

- combine BM25 and dense retrieval using Reciprocal Rank Fusion;
- rerank the BM25 Top 50 using a small cross-encoder;
- use an LLM to score candidate–constraint compatibility;
- add catalog-derived synonym expansion as a cheaper alternative.

### Why it is lower priority

Semantic retrieval is valuable when customer and catalog vocabulary differ. In this benchmark, constraints are generated from catalog metadata, so exact matching may already be unusually effective. Dense retrieval should be adopted only if an ablation improves overall and per-scenario metrics.

### Operational requirement

The official environment may disable network access. Any model-dependent feature should either run locally or fall back safely to the offline lexical path.

## 9. Add grounded recommendation explanations

### What to build

Generate concise explanations strictly from catalog fields and active constraints:

> Recommended because it is cotton, matches the requested color, and includes the feature you prioritized.

If information is missing, say that it is not listed rather than inventing it.

### Why it is lower for technical scoring

The evaluator does not score the natural-language `message` for correctness. Explanations nevertheless improve trust, make debugging easier, and strengthen the live demonstration.

## 10. Build a polished interactive demo

### Suggested demo flow

Show three short scenarios:

1. **Browsing:** vague request → useful clarification → target found.
2. **Constraint refinement:** customer adds material, color, or feature requirements.
3. **Intent Override:** customer changes a preference and the agent updates its state.

Display:

- the current structured shopping state;
- the selected `ask_attribute` and why it was chosen;
- the ranked products;
- grounded match explanations;
- turn count and target rank for the demonstration.

Build this only after the official `Agent` path is stable and reproducible.

## Recommended implementation roadmap

### Phase 1: Highest-value functional upgrade

1. Create a development branch.
2. Add per-session state.
3. Accumulate customer constraints across turns.
4. Return recommendations and a clarification question together.
5. Re-run the official evaluator.

### Phase 2: Retrieval quality

1. Add structured constraint parsing.
2. Add Intent Override handling.
3. Add category filtering and constraint-aware reranking.
4. Tune only with documented ablations.

### Phase 3: Private-set robustness

1. Add paraphrase and negation tests.
2. Add adaptive question selection.
3. Test profile use at cold start.
4. Measure latency and memory use.

### Phase 4: Optional innovation and presentation

1. Evaluate semantic retrieval or a lightweight reranker.
2. Keep an offline fallback.
3. Add grounded explanations.
4. Build the demonstration interface and scripted walkthrough.

## Recommended immediate action

Implement a minimal stateful agent that asks `feature` or `other`, accumulates the resulting customer constraints, and searches the complete accumulated query on every turn. Benchmark this before attempting embeddings or an LLM.

This experiment is small, directly addresses the starter’s largest weakness, and establishes a stronger baseline for every later idea.
