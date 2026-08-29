from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter.conversation_state import ConversationState
from starter.neural_question_policy import (
    ACTIONS,
    ActionValueNetwork,
    NeuralPolicyConfig,
    NeuralQuestionPolicy,
    StateVectorizer,
    TrainingConfig,
    observable_state_from_conversation,
)
from tools.train_neural_question_policy import load_examples, split_by_group


def example_state(active: str = "feature") -> dict[str, object]:
    return {
        "turn": 2,
        "message_count": 2,
        "active_constraint_count": 2,
        "asked_attributes": ["other"],
        "last_asked_attribute": "other",
        "active_constraint_counts": {"category": 1, active: 1},
        "no_preference_attributes": [],
        "candidate_attribute_coverage": {"material": 0.3, "feature": 0.8},
        "retrieval": {"candidate_count": 100, "score_gap": 0.2},
    }


class StateVectorizerTest(unittest.TestCase):
    def test_action_conditioned_vectors_have_a_stable_shape(self) -> None:
        vectorizer = StateVectorizer()
        material = vectorizer.vectorize(example_state(), "material")
        feature = vectorizer.vectorize(example_state(), "feature")

        self.assertEqual(material.shape, (vectorizer.dimension,))
        self.assertFalse(np.array_equal(material, feature))
        self.assertTrue(np.isfinite(material).all())

    def test_hidden_target_fields_are_not_model_features(self) -> None:
        vectorizer = StateVectorizer()
        ordinary = example_state()
        leaked = dict(ordinary, target_asin="secret", target_rank=1, scenario_type="buying")

        np.testing.assert_array_equal(
            vectorizer.vectorize(ordinary, "other"),
            vectorizer.vectorize(leaked, "other"),
        )

    def test_counterfactual_builder_feature_aliases_are_consumed(self) -> None:
        vectorizer = StateVectorizer()
        canonical = example_state()
        builder_state = {
            "turn": canonical["turn"],
            "message_count": canonical["message_count"],
            "active_constraint_count": canonical["active_constraint_count"],
            "active_constraint_type_counts": canonical["active_constraint_counts"],
            "prior_ask_attribute_counts": {"other": 1},
            "no_preference_type_counts": {},
            "retrieval_text_token_count": 4,
            "retrieval_route_candidate_counts": {"selected_sparse": 100},
        }

        names = vectorizer.feature_names
        vector = vectorizer.vectorize(builder_state, "feature")

        self.assertEqual(vector[names.index("active_count:feature")], 1)
        self.assertEqual(vector[names.index("asked_count:other")], 1)
        self.assertEqual(vector[names.index("asked_count")], 1)
        self.assertEqual(vector[names.index("query_term_count")], 4)
        self.assertEqual(vector[names.index("candidate_count")], 100)


class ActionValueNetworkTest(unittest.TestCase):
    def test_small_network_learns_and_round_trips_without_pickle(self) -> None:
        vectorizer = StateVectorizer()
        states: list[dict[str, object]] = []
        actions: list[str] = []
        values: list[float] = []
        for index in range(30):
            state = example_state("material" if index % 2 else "feature")
            for action in ("material", "feature", "other"):
                states.append(state)
                actions.append(action)
                values.append(1.0 if action == "other" else 0.0)
        features = vectorizer.matrix(states, actions)
        model = ActionValueNetwork(vectorizer.dimension, 8, seed=3)
        result = model.fit(
            features,
            np.asarray(values),
            config=TrainingConfig(
                hidden_size=8,
                learning_rate=0.01,
                batch_size=30,
                epochs=150,
                patience=30,
                seed=3,
            ),
        )

        self.assertLess(result.training_mse, 0.02)
        self.assertLess(model.parameter_count, 2_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            model.save(path)
            loaded = ActionValueNetwork.load(path)
            np.testing.assert_allclose(loaded.predict(features), model.predict(features))


class NeuralQuestionPolicyTest(unittest.TestCase):
    def _linear_action_model(self, scores: dict[str, float]) -> ActionValueNetwork:
        vectorizer = StateVectorizer()
        model = ActionValueNetwork(vectorizer.dimension, len(ACTIONS), seed=1)
        model.w1.fill(0.0)
        model.b1.fill(0.0)
        model.w2.fill(0.0)
        model.b2.fill(0.0)
        action_offset = vectorizer.feature_names.index("action:category")
        for index, action in enumerate(ACTIONS):
            model.w1[action_offset + index, index] = 1.0
            model.w2[index, 0] = scores.get(action, 0.0)
        return model

    def test_confident_model_action_is_used(self) -> None:
        model = self._linear_action_model({"material": 0.8, "other": 0.1})
        policy = NeuralQuestionPolicy(
            model, NeuralPolicyConfig(confidence_margin=0.05)
        )

        trace = policy.decide_with_trace(example_state(), ["material", "other"])

        self.assertEqual(trace.decision.ask_attribute, "material")
        self.assertFalse(trace.used_fallback)

    def test_low_confidence_falls_back_to_other(self) -> None:
        model = self._linear_action_model({"material": 0.51, "other": 0.5})
        policy = NeuralQuestionPolicy(
            model, NeuralPolicyConfig(confidence_margin=0.05)
        )

        trace = policy.decide_with_trace(example_state(), ["material", "other"])

        self.assertEqual(trace.model_action, "material")
        self.assertEqual(trace.decision.ask_attribute, "other")
        self.assertTrue(trace.used_fallback)

    def test_no_preference_action_is_masked(self) -> None:
        model = self._linear_action_model({"material": 1.0, "feature": 0.4})
        policy = NeuralQuestionPolicy(model)
        state = dict(example_state(), no_preference_attributes=["material"])

        decision = policy.decide(state, ["material", "feature"])

        self.assertEqual(decision.ask_attribute, "feature")

    def test_other_fallback_survives_transient_boundary_refusal(self) -> None:
        model = self._linear_action_model({"other": 1.0, "feature": 0.4})
        policy = NeuralQuestionPolicy(model)
        state = dict(example_state(), no_preference_attributes=["other"])

        decision = policy.decide(state, ["other", "feature"])

        self.assertEqual(decision.ask_attribute, "other")

    def test_existing_conversation_state_can_be_adapted(self) -> None:
        conversation = ConversationState({})
        conversation.observe(
            "I'm looking for Shoes. A key requirement is: leather.", turn=1
        )
        state = observable_state_from_conversation(
            conversation,
            ["other"],
            query_term_count=3,
            retrieval={"candidate_count": 100},
        )

        self.assertEqual(state["active_constraint_counts"]["material"], 1)
        self.assertEqual(state["active_constraint_counts"]["category"], 1)
        self.assertEqual(state["query_term_count"], 3)


class DatasetLoaderTest(unittest.TestCase):
    def test_expands_action_values_and_splits_product_groups(self) -> None:
        rows = [
            {
                "parent_asin": product,
                "state_id": f"{product}:1",
                "state": example_state(),
                "action_values": {"material": 0.2, "other": 0.8},
            }
            for product in ("A", "B", "C", "D")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            examples = load_examples(path)
        training, holdout = split_by_group(examples, 0.25, seed=5)

        self.assertEqual(len(examples), 8)
        self.assertTrue({row.group_id for row in training}.isdisjoint(
            {row.group_id for row in holdout}
        ))

    def test_loads_namespaced_counterfactual_builder_schema(self) -> None:
        row = {
            "model_input": {"state": example_state(), "action": "feature"},
            "label": {"training_reward": 0.73, "target_rank_at_depth": 3},
            "metadata": {
                "product_group_id": "hashed-product",
                "split": "validation",
                "sample_id": "sample-1",
                "state_turn": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual-v1.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            examples = load_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].value, 0.73)
        self.assertEqual(examples[0].group_id, "hashed-product")
        self.assertEqual(examples[0].state_id, "sample-1:2")
        self.assertEqual(examples[0].split, "validation")


if __name__ == "__main__":
    unittest.main()
