from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig
from starter.constraint_reranker import RerankConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one controlled agent ablation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-policy", default="always_other")
    parser.add_argument("--typed-state", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-decay", type=float, default=0.20)
    parser.add_argument("--exact-phrase-bonus", type=float, default=7.0)
    parser.add_argument("--overlap-weight", type=float, default=3.0)
    parser.add_argument("--hard-miss-penalty", type=float, default=2.0)
    args = parser.parse_args()

    rerank_config = RerankConfig(
        retrieval_rank_weight=args.rank_weight,
        retrieval_rank_decay=args.rank_decay,
        exact_phrase_bonus=args.exact_phrase_bonus,
        token_overlap_weight=args.overlap_weight,
        hard_miss_penalty=args.hard_miss_penalty,
    )
    config = AgentConfig(
        question_policy=args.question_policy,
        use_typed_state=args.typed_state,
        use_reranker=args.reranker,
        candidate_pool_size=args.candidate_pool_size,
        rerank_config=rerank_config,
    )
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(
        Agent(args.catalog, config=config),
        samples,
        catalog_ids,
        categories,
        products,
    )
    result["ablation_config"] = {
        "question_policy": config.question_policy,
        "use_typed_state": config.use_typed_state,
        "use_reranker": config.use_reranker,
        "candidate_pool_size": config.candidate_pool_size,
        "rerank_config": {
            "retrieval_rank_weight": rerank_config.retrieval_rank_weight,
            "retrieval_rank_decay": rerank_config.retrieval_rank_decay,
            "exact_phrase_bonus": rerank_config.exact_phrase_bonus,
            "token_overlap_weight": rerank_config.token_overlap_weight,
            "hard_miss_penalty": rerank_config.hard_miss_penalty,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
