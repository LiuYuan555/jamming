"""Evaluate unscored category filtering and constraint-query variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig, CategoryRetrievalConfig


EXPERIMENTS = {
    "a_global_control": CategoryRetrievalConfig(mode="global"),
    "b_clean_category_all_or": CategoryRetrievalConfig(mode="clean_or"),
    "c_clean_category_hard_and_soft_or": CategoryRetrievalConfig(
        mode="clean_hard_soft"
    ),
    "d_clean_category_per_constraint_rrf": CategoryRetrievalConfig(
        mode="clean_constraint_rrf"
    ),
    "e_global_clean_category_rrf": CategoryRetrievalConfig(
        mode="clean_global_fused",
        global_weight=0.5,
        category_weight=0.5,
    ),
}


def candidate_diagnostics(agent: Agent, samples: list[dict]) -> dict:
    """Summarize whether targets ever entered the selected sparse Top 100."""

    states = list(agent._sessions.values())
    if len(states) != len(samples):
        raise RuntimeError("session/sample count mismatch during candidate tracing")
    sessions: list[dict] = []
    for sample, state in zip(samples, states):
        target = str(sample["ground_truth"]["parent_asin"])
        ranks: list[int] = []
        for trace in state.retrieval_history:
            selected = list(trace.get("selected_sparse", ()))
            if target in selected:
                ranks.append(selected.index(target) + 1)
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "target": target,
            "ever_in_selected_top_100": bool(ranks),
            "best_selected_rank": min(ranks) if ranks else None,
        })
    misses = [item for item in sessions if not item["ever_in_selected_top_100"]]
    return {
        "candidate_recall_at_100": round((len(sessions) - len(misses)) / len(sessions), 6),
        "candidate_miss_count": len(misses),
        "candidate_misses": misses,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean category retrieval ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--output", default="artifacts/clean_category_ablation.json"
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    results: dict[str, dict] = {}
    summaries: list[dict] = []
    for name, route in EXPERIMENTS.items():
        print(f"Evaluating {name} ...", flush=True)
        agent = Agent(args.catalog, config=AgentConfig(category_retrieval=route))
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
            diagnostics = candidate_diagnostics(agent, samples)
        finally:
            agent.close()
        recommendation_misses = [
            item for item in result["sessions"] if not item["hit"]
        ]
        summary = {
            "configuration": name,
            "mode": route.mode,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "efficiency": result["efficiency"],
            "technical_score": result["recommended_technical_score"],
            "candidate_recall_at_100": diagnostics["candidate_recall_at_100"],
            "candidate_miss_count": diagnostics["candidate_miss_count"],
            "recommendation_miss_count": len(recommendation_misses),
            "scenario_metrics": result["scenario_metrics"],
        }
        results[name] = {
            "route": {
                "mode": route.mode,
                "global_weight": route.global_weight,
                "category_weight": route.category_weight,
                "rrf_k": route.rrf_k,
            },
            "summary": summary,
            "candidate_diagnostics": diagnostics,
            "recommendation_misses": recommendation_misses,
            "evaluation": result,
        }
        summaries.append(summary)
        print(json.dumps({
            key: value
            for key, value in summary.items()
            if key != "scenario_metrics"
        }), flush=True)

    summaries.sort(key=lambda item: item["technical_score"], reverse=True)
    payload = {
        "definitions": {
            "clean_scope": "category token membership is an unscored SQL filter",
            "b": "all non-category constraint tokens joined with OR",
            "c": "hard constraint tokens joined with AND; soft tokens joined with OR; groups joined with AND",
            "d": "one scoped BM25 ranking per non-category constraint, fused with equal-weight RRF",
            "e": "equal-weight RRF of global BM25 and clean scoped all-OR BM25",
        },
        "summary": summaries,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Best configuration:")
    print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
