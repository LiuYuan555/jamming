from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent, AgentConfig


class StatefulAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "RUNNING_SHOE",
                "title": "Road running shoe",
                "categories": ["Shoes"],
                "features": ["breathable cotton lining"],
                "details": {},
                "store": "Example",
                "description": [],
            },
            {
                "parent_asin": "COTTON_SHIRT",
                "title": "Cotton shirt",
                "categories": ["Shirts"],
                "features": ["breathable cotton fabric"],
                "details": {},
                "store": "Example",
                "description": [],
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        # Preserve the Step 1 path for focused state-accumulation tests.
        self.agent = Agent(
            self.catalog_path,
            config=AgentConfig(use_typed_state=False, use_reranker=False),
        )
        self.agent.reset("session", {"preference_tags": ["comfort"]})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_asks_for_clarification_while_returning_recommendations(self) -> None:
        response = self.agent.respond(
            "session", "I'm looking for road running shoes.", turn=1, top_k=10
        )

        self.assertEqual(response["ask_attribute"], "other")
        self.assertTrue(response["message"])
        self.assertTrue(response["recommendations"])

    def test_accumulates_search_terms_across_turns(self) -> None:
        self.agent.respond("session", "I'm looking for road running shoes.", turn=1, top_k=10)
        response = self.agent.respond(
            "session", "For that, what matters is: breathable cotton.", turn=2, top_k=10
        )

        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING_SHOE")
        state = self.agent._sessions["session"]
        self.assertIn("running", state.query_terms)
        self.assertIn("cotton", state.query_terms)

    def test_ignores_no_preference_boilerplate(self) -> None:
        self.agent.respond("session", "I'm looking for road running shoes.", turn=1, top_k=10)
        before = list(self.agent._sessions["session"].query_terms)
        self.agent.respond(
            "session",
            "I don't have an additional preference for material.",
            turn=2,
            top_k=10,
        )

        self.assertEqual(self.agent._sessions["session"].query_terms, before)

    def test_default_configuration_uses_best_combination(self) -> None:
        agent = Agent(self.catalog_path)
        self.assertTrue(agent.config.use_typed_state)
        self.assertTrue(agent.config.use_reranker)
        self.assertEqual(agent.config.question_policy, "always_other")
        self.assertEqual(agent.config.candidate_pool_size, 100)

        agent.reset("best", {"preference_tags": ["comfort"]})
        response = agent.respond(
            "best", "I'm looking for Shoes, but I'm still exploring.", turn=1, top_k=10
        )
        self.assertIsNotNone(agent._sessions["best"].structured)
        self.assertEqual(response["ask_attribute"], "other")


if __name__ == "__main__":
    unittest.main()
