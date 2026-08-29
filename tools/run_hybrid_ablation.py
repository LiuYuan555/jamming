"""Evaluate weighted sparse/dense fusion configurations on the public set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig
from starter.hybrid_retrieval import HybridConfig


def parse_weights(raw: str) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for item in raw.split(","):
        try:
            sparse_text, dense_text = item.strip().split(":", 1)
            sparse, dense = float(sparse_text), float(dense_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "weights must look like '1:0,0.7:0.3,0:1'"
            ) from exc
        if sparse < 0 or dense < 0 or sparse + dense == 0:
            raise argparse.ArgumentTypeError("weights must be non-negative with a positive sum")
        values.append((sparse, dense))
    return values


def label(sparse: float, dense: float) -> str:
    return f"sparse_{sparse:g}_dense_{dense:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weighted hybrid retrieval ablations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="artifacts/hybrid_ablation")
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=parse_weights("1:0,0.8:0.2,0.6:0.4,0.5:0.5,0.4:0.6,0.2:0.8,0:1"),
        help="comma-separated sparse:dense RRF weights",
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
    parser.add_argument(
        "--deterministic-query",
        action="store_true",
        help="embed typed constraints directly instead of using the lightweight LLM rewrite",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    summaries: list[dict] = []

    for sparse_weight, dense_weight in args.weights:
        name = label(sparse_weight, dense_weight)
        hybrid_config = HybridConfig(
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
            use_llm_query=not args.deterministic_query,
            strict=dense_weight > 0,
        )
        config = AgentConfig(
            question_policy="always_other",
            use_typed_state=True,
            use_reranker=True,
            candidate_pool_size=args.candidate_pool_size,
            hybrid_config=hybrid_config,
        )
        print(f"Evaluating {name} ...", flush=True)
        agent = Agent(args.catalog, config=config)
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
        finally:
            agent.close()
        result["hybrid_config"] = {
            "query_model": args.query_model,
            "embedding_model": args.embedding_model,
            "dimensions": args.dimensions,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "rrf_k": args.rrf_k,
            "candidate_pool_size": args.candidate_pool_size,
            "dense_pool_size": args.dense_pool_size,
            "use_llm_query": not args.deterministic_query,
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "configuration": name,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "efficiency": result["efficiency"],
            "technical_score": result["recommended_technical_score"],
            "reported_token_usage": result["reported_token_usage"],
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    summaries.sort(key=lambda item: item["technical_score"], reverse=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Best configuration:")
    print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
