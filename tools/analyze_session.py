from __future__ import annotations

import argparse
import json

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent, AgentConfig, _terms
from starter.constraint_reranker import Constraint
from starter.conversation_state import ConstraintSource


def trace_sample(sample: dict, catalog_path: str, categories: dict, products: dict) -> dict:
    config = AgentConfig(
        question_policy="always_other",
        use_typed_state=True,
        use_reranker=True,
        candidate_pool_size=100,
    )
    agent = Agent(catalog_path, config=config)
    session_id = f"trace_{sample['sample_id']}"
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

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        state = agent._sessions[session_id].structured
        assert state is not None
        terms = list(dict.fromkeys(_terms(state.retrieval_text())))[:80]
        expression = " OR ".join(f'"{term}"' for term in terms)
        bm25_ids: list[str] = []
        if expression:
            bm25_ids = [
                str(row[0])
                for row in agent.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 5000",
                    (expression,),
                ).fetchall()
            ]
        constraints = [
            Constraint(
                record.attribute.value,
                record.value,
                hard=record.source
                in {
                    ConstraintSource.INITIAL_CATEGORY,
                    ConstraintSource.INITIAL_REQUIREMENT,
                    ConstraintSource.OVERRIDE,
                },
            )
            for record in state.active_constraints()
        ]
        reranked = agent.reranker.rerank(bm25_ids[:100], constraints, top_k=100)
        reranked_ids = [candidate.parent_asin for candidate in reranked]
        returned_ids = [item["parent_asin"] for item in response["recommendations"]]
        turns.append({
            "turn": turn,
            "user_message": user_message,
            "active_constraints": [
                f"{record.attribute.value}:{record.value}"
                for record in state.active_constraints()
            ],
            "ask_attribute": response["ask_attribute"],
            "target_bm25_rank": bm25_ids.index(target) + 1 if target in bm25_ids else None,
            "target_rerank_rank_within_top_100": (
                reranked_ids.index(target) + 1 if target in reranked_ids else None
            ),
            "target_returned": target in returned_ids,
        })
        if override_applied and target in returned_ids:
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

    product = products[target]
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "title": product.get("title"),
        "intent_card": card,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace public sessions through the best combination")
    parser.add_argument("sample_ids", nargs="+")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = {sample["sample_id"]: sample for sample in load_jsonl(args.dataset)}
    _, categories, products = catalog_index(args.catalog)
    for sample_id in args.sample_ids:
        if sample_id not in samples:
            raise SystemExit(f"unknown sample_id: {sample_id}")
        print(json.dumps(
            trace_sample(samples[sample_id], args.catalog, categories, products),
            indent=2,
        ))


if __name__ == "__main__":
    main()
