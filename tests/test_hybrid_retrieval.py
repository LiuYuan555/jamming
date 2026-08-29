from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.hybrid_retrieval import (
    DenseCatalogIndex,
    HybridConfig,
    catalog_embedding_text,
    deterministic_semantic_query,
    weighted_rrf,
)


class HybridConfigTests(unittest.TestCase):
    def test_defaults_use_tested_lightweight_rewrite_stack(self) -> None:
        config = HybridConfig()
        self.assertEqual(config.query_model, "gpt-4o-mini")
        self.assertEqual(config.embedding_model, "text-embedding-3-small")
        self.assertEqual(config.semantic_prompt_variant(), "natural_description")

    def test_unknown_prompt_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HybridConfig(query_prompt_variant="invented")

    def test_requires_at_least_one_positive_route_weight(self) -> None:
        with self.assertRaises(ValueError):
            HybridConfig(sparse_weight=0.0, dense_weight=0.0)

    def test_constraint_schedule_reduces_dense_weight_as_information_grows(self) -> None:
        config = HybridConfig(dense_weight_schedule=(0.4, 0.2, 0.1, 0.0))
        constraints = [
            {"attribute": "category", "value": "shoes"},
            {"attribute": "material", "value": "leather"},
            {"attribute": "feature", "value": "waterproof"},
        ]
        self.assertEqual(config.fusion_weights(constraints[:1]), (0.6, 0.4))
        self.assertEqual(config.fusion_weights(constraints[:2]), (0.8, 0.2))
        self.assertEqual(config.fusion_weights(constraints[:3]), (0.9, 0.1))
        self.assertEqual(config.fusion_weights(constraints * 2), (1.0, 0.0))

    def test_attribute_schedule_counts_distinct_attributes(self) -> None:
        config = HybridConfig(
            dense_weight_schedule=(0.3, 0.1, 0.0),
            schedule_basis="attributes",
        )
        constraints = [
            {"attribute": "category", "value": "shoes"},
            {"attribute": "feature", "value": "waterproof"},
            {"attribute": "feature", "value": "high traction"},
        ]
        self.assertEqual(config.fusion_weights(constraints), (0.9, 0.1))


class RankFusionTests(unittest.TestCase):
    def test_single_sparse_route_preserves_sparse_order(self) -> None:
        self.assertEqual(
            weighted_rrf(
                ["A", "B", "C"],
                ["C", "B", "A"],
                sparse_weight=1.0,
                dense_weight=0.0,
                limit=3,
            ),
            ["A", "B", "C"],
        )

    def test_overlap_is_rewarded_when_routes_have_equal_weight(self) -> None:
        fused = weighted_rrf(
            ["A", "B", "C"],
            ["D", "B", "E"],
            sparse_weight=0.5,
            dense_weight=0.5,
            rrf_k=60.0,
            limit=5,
        )
        self.assertEqual(fused[0], "B")

    def test_dense_weight_can_change_the_first_candidate(self) -> None:
        fused = weighted_rrf(
            ["A", "B"],
            ["B", "A"],
            sparse_weight=0.2,
            dense_weight=0.8,
            rrf_k=0.0,
            limit=2,
        )
        self.assertEqual(fused, ["B", "A"])


class SemanticTextTests(unittest.TestCase):
    def test_deterministic_query_keeps_attribute_labels_and_values(self) -> None:
        query = deterministic_semantic_query([
            {"attribute": "category", "value": "running shoes"},
            {"attribute": "feature", "value": "high-traction outsole"},
        ])
        self.assertEqual(query, "category: running shoes; feature: high-traction outsole")

    def test_catalog_text_is_labeled_and_bounded(self) -> None:
        text = catalog_embedding_text({
            "title": "Trail Shoe",
            "categories": ["Shoes", "Athletic"],
            "features": ["High traction", "Breathable"],
        }, max_chars=80)
        self.assertIn("Title: Trail Shoe", text)
        self.assertIn("Features: High traction Breathable", text)
        self.assertLessEqual(len(text), 80)


class DenseCatalogIndexTests(unittest.TestCase):
    def test_exact_dot_product_search(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is an optional hybrid dependency")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "embeddings.npy"
            ids_path = root / "ids.json"
            np.save(matrix_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            ids_path.write_text(json.dumps(["A", "B"]), encoding="utf-8")
            index = DenseCatalogIndex(matrix_path, ids_path, dimensions=2)
            self.assertEqual(index.search([0.9, 0.1], top_k=2), ("A", "B"))


if __name__ == "__main__":
    unittest.main()
