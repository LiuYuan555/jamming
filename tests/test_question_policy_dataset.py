from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index
from starter.agent import Agent
from tools.build_question_policy_dataset import (
    ACTION_ORDER,
    build_counterfactual_rows,
    split_for_parent_asin,
    synthesize_catalog_samples,
)


class QuestionPolicyDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = []
        for index in range(12):
            products.append(
                {
                    "parent_asin": f"DISTRACTOR_{index:02d}",
                    "title": f"Shoes shoes comfort {index}",
                    "categories": ["Clothing", "Shoes"],
                    "features": [f"ordinary comfort feature {index}"],
                    "details": {},
                    "description": [],
                    "store": "Example",
                }
            )
        products.append(
            {
                "parent_asin": "SECRET_TARGET_ASIN",
                "title": "Everyday footwear target",
                "categories": ["Clothing", "Shoes"],
                "features": ["rare cobalt zipper"],
                "details": {},
                "description": [],
                "store": "Example",
            }
        )
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.catalog_ids, self.categories, self.products = catalog_index(
            self.catalog_path
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _sample(self, sample_id: str = "example") -> dict:
        return {
            "sample_id": sample_id,
            "scenario_type": "browsing",
            "user_profile": {
                "average_prior_rating": 4.0,
                "preference_tags": ["comfort"],
                "summary": "Prior purchases emphasize comfort.",
            },
            "ground_truth": {"parent_asin": "SECRET_TARGET_ASIN"},
        }

    def test_split_is_deterministic_and_product_grouped(self) -> None:
        first = split_for_parent_asin("A", seed="fixed")
        self.assertEqual(first, split_for_parent_asin("A", seed="fixed"))
        self.assertIn(first, {"train", "validation", "test"})
        with self.assertRaises(ValueError):
            split_for_parent_asin("A", train_ratio=.9, validation_ratio=.2)

    def test_enumerates_all_actions_and_keeps_hidden_data_out_of_input(self) -> None:
        agent = Agent(self.catalog_path)
        try:
            rows = build_counterfactual_rows(
                agent,
                [self._sample()],
                self.catalog_ids,
                self.categories,
                self.products,
                max_states_per_sample=1,
                candidate_depth=20,
            )
        finally:
            agent.close()

        self.assertEqual({row["model_input"]["action"] for row in rows}, set(ACTION_ORDER))
        self.assertEqual(len(rows), len(ACTION_ORDER))
        group_ids = {row["metadata"]["product_group_id"] for row in rows}
        splits = {row["metadata"]["split"] for row in rows}
        self.assertEqual(len(group_ids), 1)
        self.assertEqual(len(splits), 1)

        forbidden_keys = {
            "target",
            "target_parent_asin",
            "parent_asin",
            "ground_truth",
            "intent_card",
            "scenario_type",
            "features",
            "details",
            "title",
            "price",
        }
        for row in rows:
            state = row["model_input"]["state"]
            self.assertTrue(forbidden_keys.isdisjoint(state))
            self.assertNotIn("SECRET_TARGET_ASIN", json.dumps(row["model_input"]))

    def test_labels_come_from_action_specific_downstream_ranking(self) -> None:
        agent = Agent(self.catalog_path)
        try:
            rows = build_counterfactual_rows(
                agent,
                [self._sample()],
                self.catalog_ids,
                self.categories,
                self.products,
                max_states_per_sample=1,
                candidate_depth=20,
            )
        finally:
            agent.close()
        by_action = {row["model_input"]["action"]: row["label"] for row in rows}
        self.assertTrue(by_action["feature"]["hit_at_10"])
        self.assertTrue(by_action["other"]["hit_at_10"])
        self.assertFalse(by_action["material"]["hit_at_10"])
        self.assertGreater(
            by_action["feature"]["training_reward"],
            by_action["material"]["training_reward"],
        )
        self.assertTrue(by_action["feature"]["is_best_action"])

    def test_duplicate_target_sessions_cannot_cross_splits(self) -> None:
        agent = Agent(self.catalog_path)
        try:
            rows = build_counterfactual_rows(
                agent,
                [self._sample("one"), self._sample("two")],
                self.catalog_ids,
                self.categories,
                self.products,
                max_states_per_sample=1,
                candidate_depth=20,
            )
        finally:
            agent.close()
        groups_to_splits: dict[str, set[str]] = {}
        for row in rows:
            metadata = row["metadata"]
            groups_to_splits.setdefault(metadata["product_group_id"], set()).add(
                metadata["split"]
            )
        self.assertTrue(groups_to_splits)
        self.assertTrue(all(len(splits) == 1 for splits in groups_to_splits.values()))

    def test_synthetic_samples_exclude_public_targets_and_follow_mix(self) -> None:
        synthetic = synthesize_catalog_samples(
            self.products,
            count=5,
            seed=7,
            profiles=[{"preference_tags": ["fit"]}],
            excluded_parent_asins={"SECRET_TARGET_ASIN"},
        )
        self.assertEqual(len(synthetic), 5)
        self.assertNotIn(
            "SECRET_TARGET_ASIN",
            {sample["ground_truth"]["parent_asin"] for sample in synthetic},
        )
        self.assertTrue(all("intent_card" in sample for sample in synthetic))


if __name__ == "__main__":
    unittest.main()
