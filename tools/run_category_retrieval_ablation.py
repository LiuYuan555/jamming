"""Compare global BM25, category-only BM25, and their weighted RRF fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent, AgentConfig, CategoryRetrievalConfig


EXPERIMENTS = {
    "global": CategoryRetrievalConfig(mode="global"),
    "category_only": CategoryRetrievalConfig(mode="category_only"),
    "global_category_rrf": CategoryRetrievalConfig(
        mode="fused", global_weight=0.5, category_weight=0.5
    ),
}


def category_candidate_diagnostics(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    """Replay category-only sessions and retain pre-reranker candidate ranks."""

    sessions: list[dict] = []
    for sample in samples:
        session_id = f"category_trace_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        turns: list[dict] = []
        hit = False
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            trace = agent._sessions[session_id].last_retrieval
            category_ids = list(trace.get("category", ()))
            global_ids = list(trace.get("global", ()))
            selected_ids = list(trace.get("selected_sparse", ()))
            turns.append({
                "turn": turn,
                "category_rank": category_ids.index(target) + 1 if target in category_ids else None,
                "global_rank": global_ids.index(target) + 1 if target in global_ids else None,
                "selected_sparse_rank": selected_ids.index(target) + 1 if target in selected_ids else None,
                "category_pool_size": len(category_ids),
            })
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        category_ranks = [item["category_rank"] for item in turns if item["category_rank"]]
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "target": target,
            "recommendation_hit": hit,
            "ever_in_category_top_100": bool(category_ranks),
            "best_category_rank": min(category_ranks) if category_ranks else None,
            "turns": turns,
        })

    complete_misses = [item for item in sessions if not item["ever_in_category_top_100"]]
    return {
        "sample_count": len(sessions),
        "ever_in_category_top_100_count": len(sessions) - len(complete_misses),
        "category_candidate_recall": round(
            (len(sessions) - len(complete_misses)) / len(sessions), 6
        ),
        "complete_miss_count": len(complete_misses),
        "complete_misses": complete_misses,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run category retrieval ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--output", default="artifacts/category_retrieval_ablation.json"
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    results: dict[str, dict] = {}
    for name, category_config in EXPERIMENTS.items():
        print(f"Evaluating {name} ...", flush=True)
        agent = Agent(
            args.catalog,
            config=AgentConfig(category_retrieval=category_config),
        )
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
        finally:
            agent.close()
        results[name] = result
        print(json.dumps({
            "configuration": name,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "technical_score": result["recommended_technical_score"],
        }), flush=True)

    print("Tracing category-only candidate recall ...", flush=True)
    trace_agent = Agent(
        args.catalog,
        config=AgentConfig(
            category_retrieval=CategoryRetrievalConfig(
                mode="category_only", trace_all_routes=True
            )
        ),
    )
    try:
        diagnostics = category_candidate_diagnostics(
            trace_agent, samples, catalog_ids, categories, products
        )
    finally:
        trace_agent.close()

    summary = [
        {
            "configuration": name,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "efficiency": result["efficiency"],
            "technical_score": result["recommended_technical_score"],
            "scenario_metrics": result["scenario_metrics"],
        }
        for name, result in results.items()
    ]
    summary.sort(key=lambda item: item["technical_score"], reverse=True)
    payload = {
        "scope_definition": (
            "All normalized tokens from the active category constraint must occur in "
            "the catalog categories field; BM25 ranks with all accumulated constraints."
        ),
        "fusion": {"method": "weighted_rrf", "global_weight": 0.5, "category_weight": 0.5, "rrf_k": 60.0},
        "summary": summary,
        "category_candidate_diagnostics": diagnostics,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Best configuration:")
    print(json.dumps(summary[0], indent=2))
    print(json.dumps({
        "category_candidate_recall": diagnostics["category_candidate_recall"],
        "complete_miss_count": diagnostics["complete_miss_count"],
        "complete_miss_sample_ids": [
            item["sample_id"] for item in diagnostics["complete_misses"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
