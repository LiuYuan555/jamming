from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.constraint_reranker import (
    CatalogReranker,
    Constraint,
    ProductDocument,
    RerankConfig,
    rerank_documents,
)


def document(
    parent_asin: str,
    *,
    title: str = "",
    categories: list[str] | None = None,
    features: list[str] | None = None,
    details: dict[str, str] | None = None,
    store: str = "",
    description: list[str] | None = None,
    price: float | None = None,
) -> ProductDocument:
    return ProductDocument.from_product({
        "parent_asin": parent_asin,
        "title": title,
        "categories": categories or [],
        "features": features or [],
        "details": details or {},
        "store": store,
        "description": description or [],
        "price": price,
    })


class ConstraintRerankerTest(unittest.TestCase):
    def test_exact_disclosed_feature_overrides_bm25_rank(self) -> None:
        candidates = [
            document("GENERIC", title="Trail shirt", features=["Comfortable fabric"]),
            document(
                "TARGET",
                title="Trail shirt",
                features=["Machine washable with a waterproof zippered pocket"],
            ),
        ]

        ranked = rerank_documents(
            candidates,
            [Constraint("feature", "Machine washable with a waterproof zippered pocket")],
        )

        self.assertEqual(ranked[0].parent_asin, "TARGET")
        self.assertTrue(ranked[0].matched_constraints)
        self.assertTrue(ranked[1].missed_hard_constraints)

    def test_attribute_limits_matches_to_semantically_relevant_fields(self) -> None:
        candidates = [
            document("STORE_ONLY", title="Shirt", store="Cotton Company"),
            document("MATERIAL", title="Shirt", features=["100% cotton"]),
        ]

        ranked = rerank_documents(candidates, [Constraint("material", "cotton")])

        self.assertEqual(ranked[0].parent_asin, "MATERIAL")
        self.assertFalse(ranked[1].matched_constraints)

    def test_exact_phrase_respects_token_boundaries(self) -> None:
        candidates = [
            document("SHREDDED", title="Shredded fabric shirt"),
            document("RED", title="Red fabric shirt"),
        ]

        ranked = rerank_documents(candidates, [Constraint("color", "red")])

        self.assertEqual(ranked[0].parent_asin, "RED")
        self.assertFalse(ranked[1].matched_constraints)

    def test_negated_constraint_penalizes_matching_product(self) -> None:
        candidates = [
            document("LEATHER", title="Leather walking shoe"),
            document("CANVAS", title="Canvas walking shoe"),
        ]

        ranked = rerank_documents(
            candidates,
            [Constraint("material", "leather", negated=True)],
        )

        self.assertEqual(ranked[0].parent_asin, "CANVAS")
        self.assertEqual(ranked[1].violated_constraints, ("material:leather",))

    def test_budget_constraint_understands_around_and_upper_bound(self) -> None:
        candidates = [
            document("EXPENSIVE", title="Boot", price=95.0),
            document("MATCH", title="Boot", price=50.0),
        ]

        around = rerank_documents(candidates, [Constraint("budget", "budget around $50")])
        under = rerank_documents(candidates, [Constraint("budget", "under $60")])

        self.assertEqual(around[0].parent_asin, "MATCH")
        self.assertEqual(under[0].parent_asin, "MATCH")

    def test_no_constraints_preserves_bm25_order(self) -> None:
        candidates = [document("A"), document("B"), document("C")]

        ranked = rerank_documents(candidates, [], top_k=3)

        self.assertEqual([item.parent_asin for item in ranked], ["A", "B", "C"])

    def test_config_can_disable_constraint_scoring_for_ablation(self) -> None:
        candidates = [
            document("FIRST", title="Generic shirt"),
            document("SECOND", features=["rare exact requirement"]),
        ]
        config = RerankConfig(
            exact_phrase_bonus=0.0,
            token_overlap_weight=0.0,
            hard_miss_penalty=0.0,
            budget_match_bonus=0.0,
        )

        ranked = rerank_documents(
            candidates,
            [Constraint("feature", "rare exact requirement")],
            config=config,
        )

        self.assertEqual(ranked[0].parent_asin, "FIRST")

    def test_catalog_api_deduplicates_and_ignores_unknown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            products = [
                {"parent_asin": "A", "title": "Generic shoe"},
                {"parent_asin": "B", "title": "Blue running shoe"},
            ]
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            reranker = CatalogReranker(catalog_path)

            ranked = reranker.rerank(
                ["A", "UNKNOWN", "B", "B"],
                [Constraint("color", "blue")],
            )

        self.assertEqual([item.parent_asin for item in ranked], ["B", "A"])
        self.assertEqual(ranked[0].recommendation(), {"parent_asin": "B"})


if __name__ == "__main__":
    unittest.main()
