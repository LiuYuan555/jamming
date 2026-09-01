# TechJam 2026 Track 4 submission manifest

This repository deliberately separates the system used for official scoring
from optional deployment-oriented demonstrations. That distinction prevents
experimental components from being credited with benchmark results they did
not produce.

## Official scored agent

- Entry class: `starter.agent.Agent`
- Python: 3.10 or newer
- Runtime dependencies: Python standard library only
- Network/API key: not required
- Run command: `python3 -m evaluator.local_evaluator`
- Measured public-set result: Hit Rate@10 `0.970`, MRR `0.664567`, MTTC
  `2.535`, TechnicalScore `0.853670`
- Runtime path: template/rule parsing, typed conversation state, weighted
  SQLite FTS5 BM25, constraint-aware reranking, deterministic question policy
- Model token usage: zero

Required scored-agent source:

```text
starter/__init__.py
starter/agent.py
starter/constraint_reranker.py
starter/conversation_state.py
starter/hybrid_retrieval.py
starter/question_policy.py
starter/semantic_query_prompt.py
```

The organizer supplies the frozen catalog. `data/catalog.jsonl` is intentionally
ignored and should not be copied into the submission archive unless the portal
explicitly requests it.

## Optional deployment/demo layer

The following modules demonstrate how the score-mode control plane extends to
ordinary customer language without becoming dependent on an external API:

```text
starter/llm_agent.py       optional orchestration adapter
starter/local_llm.py       Qwen2.5-1.5B extraction, rewrite and reply
starter/local_dense.py     local MiniLM dense search
starter/hf_hybrid_retrieval.py
ui/                       local engineering/customer demo
```

These components are experimental and are not the source of the headline
TechnicalScore. Model weights and MiniLM catalog-vector assets are not bundled.
They require `torch`, `transformers`, and `sentence-transformers`, plus a
separate asset build. If those dependencies or assets are unavailable, the
agent fails soft to the deterministic scored path.

## Presentation assets

- `docs/techjam_submission_slides.html`: self-contained 16:9 slide deck; use
  arrow keys to navigate or print to PDF with one slide per page.
- `ui/index.html`: frontend for the local live demo. Use `python3 -m ui.server`
  for the deterministic path or `.venv/bin/python -m ui.server --llm` to show
  Qwen fall-through extraction and the grounded reply.

## Include in the project submission

- `README.md`, `HISTORY.md`, `DATA_ATTRIBUTION.md`
- `starter/`, `requirements-hybrid.txt`, `requirements-demo.txt`
- `docs/agent_api_contract.json`, `docs/evaluation_config.json`,
  `docs/intent_aware_hybrid_architecture.md`, and the slide deck
- `ui/` when submitting the demonstration experience
- `tests/`, `evaluator/`, and `tools/` when the portal permits reproducibility
  material

## Exclude from the project submission

- `.git/`, `.venv/`, `.pytest_cache/`, `__pycache__/`, `.DS_Store`
- `.env`, API keys, model caches, logs
- generated `results.json`, `*_results.json`, and `artifacts/`
- `catalog.jsonl.gz`, `data/catalog.jsonl`, and other downloaded release files
- organizer/private evaluation material

The durable experiment record is `HISTORY.md`; generated per-run result files
are intentionally excluded.
