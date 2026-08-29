"""Compare fixed retrieval with observable intent routing and entropy gating."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig
from starter.hybrid_retrieval import HybridConfig
from starter.intent_hybrid_router import (
    IntentConditionedHybridAgent,
    IntentRoutingConfig,
)


EXPERIMENTS = (
    "sparse_only",
    "fixed_85_15",
    "intent_router",
    "intent_router_entropy",
    "intent_router_conservative",
    "intent_router_conservative_entropy",
    "intent_router_topk",
    "intent_router_conservative_topk",
)


def _summary(name: str, result: dict, diagnostics: dict | None = None) -> dict:
    return {
        "configuration": name,
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
        "reported_token_usage": result["reported_token_usage"],
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/intent_hybrid_ablation.json")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-path", default="artifacts/catalog_embeddings.npy")
    parser.add_argument("--ids-path", default="artifacts/catalog_embedding_ids.json")
    parser.add_argument("--cache-path", default="artifacts/hybrid_query_cache.sqlite")
    parser.add_argument(
        "--classifier-model",
        help="optional JSON artifact from tools.train_intent_classifier",
    )
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = set(requested) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"unknown experiments: {', '.join(sorted(unknown))}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.allow_network:
        # Force cache-only dense retrieval even if the shell has a key. Missing
        # queries fail open to sparse retrieval in HybridRetriever.
        os.environ.pop("OPENAI_API_KEY", None)

    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    results: list[dict] = []
    classifier = None
    if args.classifier_model:
        from starter.intent_classifier import IntentClassifier

        classifier = IntentClassifier.load(args.classifier_model)

    common_hybrid = dict(
        enabled=True,
        embedding_path=args.embedding_path,
        ids_path=args.ids_path,
        cache_path=args.cache_path,
        dimensions=args.dimensions,
        sparse_weight=0.85,
        dense_weight=0.15,
        strict=False,
    )
    for name in requested:
        if name == "sparse_only":
            agent: Agent = Agent(
                args.catalog,
                AgentConfig(hybrid_config=HybridConfig(enabled=False)),
            )
        elif name == "fixed_85_15":
            agent = Agent(
                args.catalog,
                AgentConfig(hybrid_config=HybridConfig(**common_hybrid)),
            )
        else:
            routing_config = (
                IntentRoutingConfig.conservative()
                if "conservative" in name
                else None
            )
            agent = IntentConditionedHybridAgent(
                args.catalog,
                AgentConfig(hybrid_config=HybridConfig(**common_hybrid)),
                classifier=classifier,
                routing_config=routing_config,
                use_entropy_gate=name.endswith("entropy"),
                use_intent_topk="topk" in name,
            )
        try:
            result = evaluate(agent, samples, catalog_ids, categories, products)
            diagnostics = (
                agent.diagnostics()
                if isinstance(agent, IntentConditionedHybridAgent)
                else None
            )
        finally:
            agent.close()
        results.append(_summary(name, result, diagnostics))
        print(json.dumps(results[-1], indent=2), flush=True)

    results.sort(key=lambda item: item["technical_score"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
