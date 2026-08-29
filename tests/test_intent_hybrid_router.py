from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.conversation_state import ConversationState
from starter.intent_hybrid_router import (
    EntropyGateConfig,
    IntentHybridRouter,
    IntentPrediction,
    IntentRoutingConfig,
    IntentTopKConfig,
    IntentTopKPolicy,
    ObservableIntentBaseline,
    ShannonEntropyQuestionGate,
    ShoppingIntent,
    RoutedHybridConfig,
    _RoutingContext,
)
from starter.hybrid_retrieval import HybridConfig


def prediction(intent: ShoppingIntent, confidence: float = 1.0) -> IntentPrediction:
    remainder = (1.0 - confidence) / 3.0
    return IntentPrediction(
        {item: confidence if item is intent else remainder for item in ShoppingIntent}
    )


class IntentHybridRouterTest(unittest.TestCase):
    def test_dense_prompt_style_routes_by_observable_intent(self) -> None:
        routed = RoutedHybridConfig(HybridConfig(), IntentHybridRouter())
        boundary_token = routed.activate(
            _RoutingContext(prediction(ShoppingIntent.BOUNDARY), None)
        )
        try:
            self.assertEqual(routed.semantic_prompt_variant(), "catalog_phrase")
        finally:
            routed.deactivate(boundary_token)
        buying_token = routed.activate(
            _RoutingContext(prediction(ShoppingIntent.BUYING), None)
        )
        try:
            self.assertEqual(routed.semantic_prompt_variant(), "natural_description")
        finally:
            routed.deactivate(buying_token)

    def test_buying_becomes_more_sparse_as_constraints_accumulate(self) -> None:
        router = IntentHybridRouter()
        weights = [
            router.route(prediction(ShoppingIntent.BUYING), count).dense_weight
            for count in range(1, 6)
        ]
        self.assertEqual(weights, [0.30, 0.20, 0.10, 0.05, 0.05])

    def test_browsing_is_denser_than_buying(self) -> None:
        router = IntentHybridRouter()
        browsing = router.route(prediction(ShoppingIntent.BROWSING), 1)
        buying = router.route(prediction(ShoppingIntent.BUYING), 1)
        self.assertGreater(browsing.dense_weight, buying.dense_weight)
        self.assertAlmostEqual(browsing.sparse_weight + browsing.dense_weight, 1.0)

    def test_override_uses_dense_recovery_then_decays(self) -> None:
        router = IntentHybridRouter()
        values = [
            router.route(
                prediction(ShoppingIntent.INTENT_OVERRIDE),
                information_count=3,
                turns_since_override=age,
            ).dense_weight
            for age in range(4)
        ]
        self.assertEqual(values, [0.50, 0.35, 0.20, 0.20])

    def test_uncertain_prediction_blends_to_safe_baseline(self) -> None:
        router = IntentHybridRouter()
        uncertain = IntentPrediction({item: 0.25 for item in ShoppingIntent})
        decision = router.route(uncertain, 1)
        self.assertAlmostEqual(decision.dense_weight, 0.15)

    def test_conservative_profile_has_requested_schedules(self) -> None:
        config = IntentRoutingConfig.conservative()
        self.assertEqual(config.browsing_dense, (0.20, 0.15, 0.10, 0.05))
        self.assertEqual(config.buying_dense, (0.10, 0.05, 0.0, 0.0))
        self.assertEqual(config.override_dense, (0.20, 0.10, 0.0))
        self.assertEqual(config.boundary_dense, (0.10, 0.05, 0.05))

    def test_conservative_profile_keeps_browsing_relatively_denser(self) -> None:
        router = IntentHybridRouter(IntentRoutingConfig.conservative())
        for count in range(1, 6):
            browsing = router.route(prediction(ShoppingIntent.BROWSING), count)
            buying = router.route(prediction(ShoppingIntent.BUYING), count)
            self.assertGreaterEqual(browsing.dense_weight, buying.dense_weight)

    def test_observable_classifier_keeps_boundary_mass_on_ambiguous_turn_one(self) -> None:
        classifier = ObservableIntentBaseline()
        result = classifier.predict_proba(
            ["I'm looking for Shirts, but I'm still exploring."]
        )
        self.assertEqual(result.intent, ShoppingIntent.BROWSING)
        self.assertGreater(result.probabilities[ShoppingIntent.BOUNDARY], 0)

    def test_classifier_detects_observed_boundary_and_override(self) -> None:
        classifier = ObservableIntentBaseline()
        boundary = classifier.predict_proba(
            [
                "I'm looking for Shirts, but I'm still exploring.",
                "I don't have a preference for material; please use your judgment.",
            ]
        )
        override = classifier.predict_proba(
            [
                "I'm looking for Shirts. I like blue.",
                "Actually, ignore blue. What I need is: cotton.",
            ]
        )
        self.assertEqual(boundary.intent, ShoppingIntent.BOUNDARY)
        self.assertEqual(override.intent, ShoppingIntent.INTENT_OVERRIDE)

    def test_generic_exhausted_attribute_reply_is_not_boundary(self) -> None:
        classifier = ObservableIntentBaseline()
        result = classifier.predict_proba(
            [
                "I'm looking for Shirts. A key requirement is: cotton.",
                "I don't have an additional preference for material.",
            ]
        )
        self.assertEqual(result.intent, ShoppingIntent.BUYING)

    def test_browsing_becomes_observable_buying_after_concrete_reply(self) -> None:
        classifier = ObservableIntentBaseline()
        result = classifier.predict_proba(
            [
                "I'm looking for Shirts, but I'm still exploring.",
                "For that, what matters is: cotton.",
            ]
        )
        self.assertEqual(result.intent, ShoppingIntent.BUYING)

    def test_trained_classifier_api_is_adapted_with_asked_history(self) -> None:
        from starter.intent_hybrid_router import _classify_observable_history

        class PortablePrediction:
            probabilities = {
                "buying": 0.05,
                "browsing": 0.10,
                "intent_override": 0.80,
                "boundary": 0.05,
            }

        class PortableClassifier:
            def __init__(self) -> None:
                self.received: tuple[list[str], list[str]] | None = None

            def predict(self, messages, asked_attributes):
                self.received = (list(messages), list(asked_attributes))
                return PortablePrediction()

            def predict_proba(self, messages, asked_attributes):
                raise AssertionError("adapter should prefer the safeguarded predict API")

        classifier = PortableClassifier()
        result = _classify_observable_history(
            classifier,
            ["Actually, what I need is cotton."],
            ["material"],
        )
        self.assertEqual(result.intent, ShoppingIntent.INTENT_OVERRIDE)
        self.assertEqual(
            classifier.received,
            (["Actually, what I need is cotton."], ["material"]),
        )


class IntentTopKPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "A",
                "title": "Basic shirt",
                "categories": ["Shirts"],
                "features": ["blue"],
                "details": {},
                "store": "Same Store",
                "description": [],
            },
            {
                "parent_asin": "B",
                "title": "Basic shirt",
                "categories": ["Shirts"],
                "features": ["blue"],
                "details": {},
                "store": "Same Store",
                "description": [],
            },
            {
                "parent_asin": "C",
                "title": "Artistic cotton top",
                "categories": ["Shirts"],
                "features": ["cotton"],
                "details": {},
                "store": "Different Store",
                "description": [],
            },
            {
                "parent_asin": "SHOE",
                "title": "Basic shoe",
                "categories": ["Shoes"],
                "features": ["blue"],
                "details": {},
                "store": "Same Store",
                "description": [],
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(item) + "\n" for item in products), encoding="utf-8"
        )
        self.policy = IntentTopKPolicy(
            self.catalog,
            IntentTopKConfig(pool_size=10, diversity_weight=0.12),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def payload(*ids: str) -> list[dict[str, str]]:
        return [{"parent_asin": value} for value in ids]

    def test_buying_preserves_constraint_ranked_order(self) -> None:
        state = ConversationState({})
        state.observe("I'm looking for Shirts. A key requirement is: cotton.", 1)
        result = self.policy.apply(
            ShoppingIntent.BUYING, self.payload("C", "A", "B"), state, top_k=3
        )
        self.assertEqual(
            [item["parent_asin"] for item in result.recommendations],
            ["C", "A", "B"],
        )
        self.assertFalse(result.changed)

    def test_browsing_pins_first_and_diversifies_metadata(self) -> None:
        state = ConversationState({})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        result = self.policy.apply(
            ShoppingIntent.BROWSING, self.payload("A", "B", "C"), state, top_k=3
        )
        ids = [item["parent_asin"] for item in result.recommendations]
        self.assertEqual(ids[0], "A")
        self.assertEqual(ids[1], "C")
        self.assertTrue(result.changed)

    def test_override_uses_only_active_post_retraction_constraint(self) -> None:
        state = ConversationState({})
        state.observe("I'm looking for Shirts. blue.", 1)
        state.observe(
            "Actually, blue. What I need is: cotton.",
            2,
            asked_attribute="other",
        )
        self.assertEqual(
            [record.value for record in state.active_constraints() if record.value == "blue"],
            [],
        )
        result = self.policy.apply(
            ShoppingIntent.INTENT_OVERRIDE,
            self.payload("A", "C"),
            state,
            top_k=2,
        )
        self.assertEqual(result.recommendations[0]["parent_asin"], "C")

    def test_boundary_refusal_is_missing_information_not_filter(self) -> None:
        state = ConversationState({})
        state.observe("I'm looking for Shirts, but I'm still exploring.", 1)
        before = len(state.active_constraints())
        state.observe(
            "I don't have a preference for color; please use your judgment.",
            2,
            asked_attribute="color",
        )
        self.assertEqual(len(state.active_constraints()), before)
        result = self.policy.apply(
            ShoppingIntent.BOUNDARY,
            self.payload("SHOE", "A"),
            state,
            top_k=2,
        )
        ids = [item["parent_asin"] for item in result.recommendations]
        self.assertEqual(ids[0], "A")
        self.assertCountEqual(ids, ["A", "SHOE"])


class ShannonEntropyQuestionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "COTTON",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["cotton", "quick dry"],
                "details": {},
                "store": "Store",
                "description": [],
                "price": 20,
            },
            {
                "parent_asin": "WOOL",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["wool", "warm"],
                "details": {},
                "store": "Store",
                "description": [],
                "price": 20,
            },
            {
                "parent_asin": "NYLON",
                "title": "Trail shirt",
                "categories": ["Shirts"],
                "features": ["nylon", "water resistant"],
                "details": {},
                "store": "Store",
                "description": [],
                "price": 20,
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(item) + "\n" for item in products), encoding="utf-8"
        )
        self.state = ConversationState({})
        self.state.observe("I'm looking for Shirts, but I'm still exploring.", 1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_entropy_is_measured_in_bits_and_wildcard_is_not_discounted(self) -> None:
        gate = ShannonEntropyQuestionGate(
            self.catalog, EntropyGateConfig(candidate_limit=3, minimum_value=0)
        )
        scores = gate.score_attributes(self.state, ["COTTON", "WOOL", "NYLON"])
        material = next(item for item in scores if item.attribute == "material")
        other = next(item for item in scores if item.attribute == "other")
        self.assertGreater(material.prior_entropy_bits, 1.0)
        self.assertGreater(material.information_gain_bits, 0)
        self.assertEqual(material.answer_coverage, 1.0)
        self.assertGreaterEqual(other.value, material.value)

    def test_no_preference_masks_attribute_without_adding_constraint(self) -> None:
        gate = ShannonEntropyQuestionGate(self.catalog)
        before = len(self.state.active_constraints())
        self.state.observe(
            "I don't have an additional preference for material.",
            2,
            asked_attribute="material",
        )
        scores = gate.score_attributes(self.state, ["COTTON", "WOOL", "NYLON"])
        material = next(item for item in scores if item.attribute == "material")
        self.assertEqual(len(self.state.active_constraints()), before)
        self.assertTrue(material.masked)
        self.assertEqual(material.mask_reason, "no_preference")

    def test_low_information_falls_back_to_other(self) -> None:
        gate = ShannonEntropyQuestionGate(
            self.catalog, EntropyGateConfig(candidate_limit=3, minimum_value=2.0)
        )
        result = gate.decide(self.state, ["COTTON", "WOOL", "NYLON"])
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.decision.ask_attribute, "other")


if __name__ == "__main__":
    unittest.main()
