from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import AgentConfig
from starter.conversation_state import ConversationState
from starter.information_gain_policy import (
    InformationGainAgent,
    InformationGainConfig,
    InformationGainQuestionPolicy,
)


class InformationGainPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "COTTON",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["cotton", "quick dry"],
                "details": {},
                "store": "Same Brand",
                "description": [],
                "price": 20,
            },
            {
                "parent_asin": "WOOL",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["wool", "quick dry"],
                "details": {},
                "store": "Same Brand",
                "description": [],
                "price": 20,
            },
            {
                "parent_asin": "NYLON",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["nylon", "quick dry"],
                "details": {},
                "store": "Same Brand",
                "description": [],
                "price": 20,
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.state = ConversationState({})
        self.state.observe(
            "I'm looking for Shirts, but I'm still exploring.", turn=1
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_selects_named_attribute_that_best_partitions_candidates(self) -> None:
        policy = InformationGainQuestionPolicy(self.catalog_path)

        result = policy.decide(self.state, ["COTTON", "WOOL", "NYLON"])

        self.assertEqual(result.decision.ask_attribute, "material")
        material = next(item for item in result.scores if item.attribute == "material")
        self.assertEqual(material.answer_rate, 1.0)
        self.assertGreater(material.information_gain, 0.0)

    def test_no_preference_masks_the_attribute(self) -> None:
        policy = InformationGainQuestionPolicy(self.catalog_path)
        self.state.observe(
            "I don't have an additional preference for material.",
            turn=2,
            asked_attribute="material",
        )

        scores = policy.score_attributes(
            self.state, ["COTTON", "WOOL", "NYLON"]
        )
        material = next(item for item in scores if item.attribute == "material")

        self.assertTrue(material.masked)
        self.assertEqual(material.mask_reason, "no_preference")

    def test_previously_asked_attribute_is_masked(self) -> None:
        policy = InformationGainQuestionPolicy(self.catalog_path)

        scores = policy.score_attributes(
            self.state,
            ["COTTON", "WOOL", "NYLON"],
            asked_attributes=["material"],
        )
        material = next(item for item in scores if item.attribute == "material")

        self.assertTrue(material.masked)
        self.assertEqual(material.mask_reason, "already_asked")

    def test_zero_catalog_coverage_is_masked(self) -> None:
        policy = InformationGainQuestionPolicy(self.catalog_path)

        scores = policy.score_attributes(
            self.state, ["COTTON", "WOOL", "NYLON"]
        )
        brand = next(item for item in scores if item.attribute == "brand")

        self.assertTrue(brand.masked)
        self.assertEqual(brand.mask_reason, "zero_coverage")

    def test_policy_api_has_no_target_or_hidden_intent_argument(self) -> None:
        policy = InformationGainQuestionPolicy(self.catalog_path)

        with self.assertRaises(TypeError):
            policy.decide(  # type: ignore[call-arg]
                self.state,
                ["COTTON", "WOOL"],
                target_parent_asin="COTTON",
            )

    def test_adapter_preserves_retrieval_and_returns_valid_question(self) -> None:
        agent = InformationGainAgent(
            self.catalog_path,
            config=AgentConfig(
                use_typed_state=True,
                use_reranker=False,
                candidate_pool_size=10,
            ),
            policy_config=InformationGainConfig(candidate_limit=10),
        )
        try:
            agent.reset("session", {})
            response = agent.respond(
                "session",
                "I'm looking for Shirts, but I'm still exploring.",
                turn=1,
                top_k=10,
            )

            self.assertEqual(response["ask_attribute"], "material")
            self.assertTrue(response["message"])
            self.assertEqual(len(response["recommendations"]), 3)
            self.assertEqual(agent._sessions["session"].asked_attributes, ["material"])
            self.assertEqual(agent._sessions["session"].last_asked_attribute, "material")
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
