"""Compare fixed hybrid weights with information-aware dense-weight schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig
from starter.hybrid_retrieval import HybridConfig


SCHEDULES: dict[str, tuple[float, ...]] = {
    # Values apply at 1, 2, 3, and 4+ known constraints/attributes.
    "adaptive_aggressive": (0.40, 0.25, 0.10, 0.00),
    "adaptive_moderate": (0.30, 0.20, 0.10, 0.00),
    "adaptive_conservative": (0.20, 0.15, 0.10, 0.00),
    "adaptive_very_conservative": (0.15, 0.10, 0.05, 0.00),
    "adaptive_micro": (0.10, 0.05, 0.00, 0.00),
    "adaptive_tiny": (0.05, 0.00, 0.00, 0.00),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run information-aware hybrid ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="artifacts/adaptive_hybrid_ablation")
    parser.add_argument(
        "--bases",
        default="constraints,attributes",
        help="comma-separated schedule bases: constraints and/or attributes",
    )
    parser.add_argument(
        "--schedules",
        default=",".join(SCHEDULES),
        help="comma-separated schedule names",
    )
    parser.add_argument("--query-model", default="gpt-5.6-luna")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--dense-pool-size", type=int, default=100)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--embedding-path", default="artifacts/catalog_embeddings.npy")
    parser.add_argument("--ids-path", default="artifacts/catalog_embedding_ids.json")
    parser.add_argument("--cache-path", default="artifacts/hybrid_query_cache.sqlite")
    args = parser.parse_args()

    bases = [value.strip() for value in args.bases.split(",") if value.strip()]
    if not bases or any(value not in {"constraints", "attributes"} for value in bases):
        raise SystemExit("--bases must contain constraints and/or attributes")
    schedule_names = [value.strip() for value in args.schedules.split(",") if value.strip()]
    if not schedule_names or any(value not in SCHEDULES for value in schedule_names):
        raise SystemExit(f"--schedules must use: {', '.join(SCHEDULES)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    experiments: list[tuple[str, tuple[float, ...], str, float, float]] = [
        ("fixed_sparse_only", (), "constraints", 1.0, 0.0),
        ("fixed_best_hybrid", (), "constraints", 0.85, 0.15),
    ]
    for basis in bases:
        for name in schedule_names:
            schedule = SCHEDULES[name]
            experiments.append((f"{name}_{basis}", schedule, basis, 0.85, 0.15))

    summaries: list[dict] = []
    for name, schedule, basis, sparse_weight, dense_weight in experiments:
        config = AgentConfig(
            question_policy="always_other",
            use_typed_state=True,
            use_reranker=True,
            candidate_pool_size=args.candidate_pool_size,
            hybrid_config=HybridConfig(
                enabled=True,
                embedding_path=args.embedding_path,
                ids_path=args.ids_path,
                cache_path=args.cache_path,
                query_model=args.query_model,
                embedding_model=args.embedding_model,
                dimensions=args.dimensions,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                rrf_k=args.rrf_k,
                dense_pool_size=args.dense_pool_size,
                dense_weight_schedule=schedule,
                schedule_basis=basis,
                strict=dense_weight > 0 or bool(schedule),
            ),
        )
        print(f"Evaluating {name} ...", flush=True)
        agent = Agent(args.catalog, config=config)
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
        finally:
            agent.close()
        result["adaptive_hybrid_config"] = {
            "name": name,
            "schedule_basis": basis,
            "dense_weight_schedule": schedule,
            "fixed_sparse_weight": sparse_weight,
            "fixed_dense_weight": dense_weight,
            "rrf_k": args.rrf_k,
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "configuration": name,
            "schedule_basis": basis,
            "dense_weight_schedule": schedule,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "efficiency": result["efficiency"],
            "technical_score": result["recommended_technical_score"],
            "scenario_metrics": result["scenario_metrics"],
        }
        summaries.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "scenario_metrics"}, indent=2))

    summaries.sort(key=lambda item: item["technical_score"], reverse=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Best configuration:")
    print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
