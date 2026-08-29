"""Precompute public-set semantic query rewrites and query embeddings.

This does not use hidden labels in a retrieval decision. It simply enumerates
the deterministic always-``other`` conversation states that the public
simulator can disclose, then caches the same runtime OpenAI calls concurrently
so multiple fusion-weight ablations do not repay their latency.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.conversation_state import ConstraintSource, ConversationState
from starter.hybrid_retrieval import HybridConfig, HybridRetriever
from starter.semantic_query_prompt import DEFAULT_QUERY_MODEL


HARD_SOURCES = {
    ConstraintSource.INITIAL_CATEGORY,
    ConstraintSource.INITIAL_REQUIREMENT,
    ConstraintSource.OVERRIDE,
}


def public_constraint_states(samples: list[dict], categories: dict, products: dict) -> list[list[dict]]:
    unique: dict[str, list[dict]] = {}
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        state = ConversationState(sample["user_profile"])
        last_asked: str | None = None

        for turn in range(1, MAX_TURNS + 1):
            state.observe(message, turn, asked_attribute=last_asked)
            constraints = [
                {
                    "attribute": record.attribute.value,
                    "value": record.value,
                    "hard": record.source in HARD_SOURCES,
                }
                for record in state.active_constraints()
            ]
            key = json.dumps(constraints, sort_keys=True, separators=(",", ":"))
            unique.setdefault(key, constraints)
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                message, boundary_used = customer_reply(
                    effective,
                    "other",
                    disclosed,
                    boundary_used,
                )
            last_asked = "other"
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm the public semantic-query cache")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--query-model", default=DEFAULT_QUERY_MODEL)
    parser.add_argument("--prompt-variant", default="natural_description")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--embedding-path", default="artifacts/catalog_embeddings.npy")
    parser.add_argument("--ids-path", default="artifacts/catalog_embedding_ids.json")
    parser.add_argument("--cache-path", default="artifacts/hybrid_query_cache.sqlite")
    parser.add_argument("--output", default="artifacts/hybrid_cache_warm_summary.json")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    states = public_constraint_states(samples, categories, products)
    print(f"Discovered {len(states)} unique semantic query states", flush=True)

    retriever = HybridRetriever(HybridConfig(
        enabled=True,
        embedding_path=args.embedding_path,
        ids_path=args.ids_path,
        cache_path=args.cache_path,
        query_model=args.query_model,
        query_prompt_variant=args.prompt_variant,
        embedding_model=args.embedding_model,
        dimensions=args.dimensions,
        strict=True,
    ))
    prompt_tokens = 0
    completion_tokens = 0
    errors: list[str] = []
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(retriever.search, state, top_k=1) for state in states]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                prompt_tokens += result.prompt_tokens
                completion_tokens += result.completion_tokens
                if result.error:
                    errors.append(result.error)
                if completed % 25 == 0 or completed == len(states):
                    print(
                        f"Cached {completed}/{len(states)} states "
                        f"({prompt_tokens:,} prompt, {completion_tokens:,} completion tokens)",
                        flush=True,
                    )
    finally:
        retriever.close()

    summary = {
        "unique_states": len(states),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "errors": errors,
        "query_model": args.query_model,
        "prompt_variant": args.prompt_variant,
        "embedding_model": args.embedding_model,
        "dimensions": args.dimensions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
