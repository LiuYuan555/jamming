# Demo console

A local web UI for the shopping copilot. Not part of the scored agent — the
evaluator never imports anything in this folder, and nothing here can change a
score.

```bash
python -m ui.server                      # auto-discovers catalog.jsonl
python -m ui.server --catalog PATH --port 8787
```

Startup takes ~18s (FTS5 index build + reranker catalog + display index), then
opens `http://127.0.0.1:8787/`. Standard library only — no pip install, no API
key, no network. Press Enter to send, Shift+Enter for a newline.

## Local response generation (optional)

```bash
C:\Users\user\.venvs\techjam-ui\Scripts\python.exe -m ui.server --llm
```

Generates the assistant's confirmation sentence with a local
`Qwen/Qwen2.5-1.5B-Instruct` on the GPU. No API key, no network after the first
model download, ~3.1GB VRAM at fp16.

**This cannot change TechnicalScore.** The evaluator reads `ask_attribute` and
discards `message`, so generation is a demo asset only — which is why it is
opt-in and off by default. Leaving it on during a scoring run would add ~1-2s
per turn for no benefit.

Responsibility is split three ways, deliberately:

| Part of the reply | Produced by | Why |
|---|---|---|
| "You're looking for X with Y." | local LLM | phrasing is cosmetic, so a small model is safe here |
| "What other preferences should I consider?" | agent template | drives `ask_attribute`; must match that field exactly |
| ✓ evidence chips per product | exact catalog matching | instant, and cannot hallucinate an attribute |

The model is never shown the product list. Given titles it invents attributes
("features a black color") and recites raw ASINs at the customer; given only the
customer's own stated requirements it restates them accurately.

Degradation ladder, in `ResponseGenerator._validate`: valid JSON → a clean bare
sentence → the deterministic template. A reply containing a product code is
rejected outright. Generation is skipped entirely when no constraints have been
stated, because a small model asked to confirm an empty list invents
requirements wholesale.

Measured on an RTX 5060 Laptop (8GB): **12.9 tok/s**, ~1-2s per turn at 10-25
completion tokens. That throughput is only ~9% of the card's memory bandwidth —
decode of a 1.5B model on Windows is kernel-launch-bound, not compute-bound.
Generating less is far more effective than a bigger GPU here.

## Two views

The right rail has a **Results / Details** toggle.

**Results** is the customer-facing view and the default: a plain ranked list of
the ten products, with price, store, rating, rank movement (`↑6`, `NEW`), and
green chips naming which of the customer's stated constraints each product
actually satisfies. No accordions, nothing to expand.

**Details** is the engineering view: typed state, retrieval query, retrieval
routes, declined attributes, components, configuration, latency and tokens.

Match chips are computed by normalized phrase containment against the catalog's
own text for that product. That check deliberately does not import the reranker,
so it cannot silently drift when scoring changes — it either matches the
catalog text or shows nothing.

## Why it survives changes to the agent

The UI binds to the frozen organizer contract and nothing else:

```python
Agent(catalog_path)
Agent.reset(session_id, user_profile) -> None
Agent.respond(session_id, user_message, turn, top_k) -> dict
```

Everything richer is produced by `probe.py`, which fails soft. Three rules keep
it decoupled:

1. **Every internal access is wrapped.** `probe.safe()` swallows exceptions, so a
   renamed or deleted attribute yields `None` instead of a traceback.
2. **Panels are data, not layout.** The server emits a list of sections; the
   frontend renders whatever arrives. A section that cannot be built is simply
   absent, and its panel disappears.
3. **Unknown section kinds degrade, never break.** `index.html` dispatches on
   `section.kind` and falls back to formatted JSON for kinds it has never seen.

Two panels are already fully generic: **Retrieval routes** walks the
`last_retrieval` trace dictionary, so a new route appears the moment the agent
records one; **Active configuration** walks `agent.config` as a dataclass, so new
config fields show up untouched.

Verified by running the inspector against mutated configs:

| Agent config | Sections produced |
|---|---|
| default | constraints, query, routes, components, config |
| `use_typed_state=False` | query, routes, components, config |
| `use_reranker=False`, `question_policy="feature_first"` | constraints, query, routes, components, config |

Typed state off drops the constraints panel and falls back to the untyped
`query_terms` path. No edit to the UI was required for either case.

## What to add when you change the agent

To surface something new, add one builder to `probe.py` and append it to
`SESSION_SECTIONS` or `AGENT_SECTIONS`. If its `kind` is one of `constraints`,
`chips`, `routes`, `toggles`, `text`, `tree`, or `products`, it renders with no
frontend change at all. Any other `kind` renders as readable JSON until you add a
matching entry to the `R` renderer table in `index.html`.

## Files

```text
ui/server.py    stdlib HTTP server; owns sessions and rank-delta history
ui/probe.py     defensive introspection -> normalized sections (never raises)
ui/index.html   single-file frontend, no build step, light + dark
```

## Notes

- Single-threaded by design. The agent holds a SQLite connection bound to its
  creating thread, so requests are serialized rather than threaded. Fine for a
  demo at ~200ms/turn.
- Rank deltas (`↑6`, `NEW`) are computed from the contract's output alone, so
  they keep working regardless of how retrieval changes underneath.
- The catalog has no image URLs, so product thumbnails are deterministic colour
  blocks hashed from each `parent_asin`.
- Presets are ordered for a live walkthrough: open with **Stated preference**,
  not **Buying**, if you want to demo retraction — `_apply_retraction()` only
  targets `INITIAL_PREFERENCE` records, and the Buying template produces
  `INITIAL_REQUIREMENT`, which an override deliberately leaves standing.
