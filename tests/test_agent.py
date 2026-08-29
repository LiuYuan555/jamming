from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import (
    Agent,
    AgentConfig,
    BM25FieldWeights,
    CategoryRetrievalConfig,
)


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
                "features": ["soft jersey fabric"],
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

    def test_configurable_bm25_field_weights_change_ranking(self) -> None:
        title_heavy = Agent(
            self.catalog_path,
            config=AgentConfig(
                use_typed_state=False,
                use_reranker=False,
                bm25_weights=BM25FieldWeights(
                    title=10, categories=0, features=0.1, details=0,
                    store=0, description=0,
                ),
            ),
        )
        feature_heavy = Agent(
            self.catalog_path,
            config=AgentConfig(
                use_typed_state=False,
                use_reranker=False,
                bm25_weights=BM25FieldWeights(
                    title=0.1, categories=0, features=10, details=0,
                    store=0, description=0,
                ),
            ),
        )
        try:
            title_heavy.reset("title", {})
            feature_heavy.reset("feature", {})
            title_result = title_heavy.respond("title", "cotton", turn=1, top_k=10)
            feature_result = feature_heavy.respond("feature", "cotton", turn=1, top_k=10)
            self.assertEqual(title_result["recommendations"][0]["parent_asin"], "COTTON_SHIRT")
            self.assertEqual(feature_result["recommendations"][0]["parent_asin"], "RUNNING_SHOE")
        finally:
            title_heavy.close()
            feature_heavy.close()

    def test_bm25_weights_reject_all_zero(self) -> None:
        with self.assertRaises(ValueError):
            BM25FieldWeights(0, 0, 0, 0, 0, 0)

    def test_category_only_scope_excludes_other_categories(self) -> None:
        agent = Agent(
            self.catalog_path,
            config=AgentConfig(
                use_reranker=False,
                bm25_weights=BM25FieldWeights(
                    title=10, categories=1, features=0.1, details=0,
                    store=0, description=0,
                ),
                category_retrieval=CategoryRetrievalConfig(
                    mode="category_only", trace_all_routes=True
                ),
            ),
        )
        try:
            agent.reset("category", {})
            result = agent.respond(
                "category",
                "I'm looking for Shoes. A key requirement is: cotton.",
                turn=1,
                top_k=10,
            )
            self.assertEqual(result["recommendations"][0]["parent_asin"], "RUNNING_SHOE")
            trace = agent._sessions["category"].last_retrieval
            self.assertIn("COTTON_SHIRT", trace["global"])
            self.assertNotIn("COTTON_SHIRT", trace["category"])
        finally:
            agent.close()

    def test_category_retrieval_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            CategoryRetrievalConfig(mode="unknown")

    def test_clean_category_filter_does_not_score_other_categories(self) -> None:
        agent = Agent(
            self.catalog_path,
            config=AgentConfig(
                use_reranker=False,
                bm25_weights=BM25FieldWeights(
                    title=10, categories=1, features=0.1, details=0,
                    store=0, description=0,
                ),
                category_retrieval=CategoryRetrievalConfig(mode="clean_or"),
            ),
        )
        try:
            agent.reset("clean", {})
            result = agent.respond(
                "clean",
                "I'm looking for Shoes. A key requirement is: cotton.",
                turn=1,
                top_k=10,
            )
            self.assertEqual(result["recommendations"][0]["parent_asin"], "RUNNING_SHOE")
            trace = agent._sessions["clean"].last_retrieval
            self.assertEqual(trace["clean_category"], ("RUNNING_SHOE",))
            self.assertEqual(trace["selected_sparse"], ("RUNNING_SHOE",))
        finally:
            agent.close()

    def test_multi_route_rrf_rewards_overlap(self) -> None:
        self.assertEqual(
            Agent._multi_rrf(
                [["A", "B", "C"], ["B", "D", "A"]],
                rrf_k=60,
                limit=4,
            )[0],
            "B",
        )


if __name__ == "__main__":
    unittest.main()
