from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter.intent_classifier import (
    INTENTS,
    IntentClassifier,
    IntentVectorizer,
    observable_intent,
)
from tools.evaluate_intent_classifier import classification_metrics
from tools.train_intent_classifier import generate_catalog_examples


class IntentClassifierTest(unittest.TestCase):
    def test_observable_labels_do_not_claim_future_hidden_events(self) -> None:
        vague = ["I'm looking for Shoes, but I'm still exploring."]
        self.assertEqual(observable_intent(vague), "browsing")
        self.assertEqual(
            observable_intent(vague + [
                "I don't have a preference for color; please use your judgment."
            ]),
            "boundary",
        )
        self.assertEqual(
            observable_intent([
                "I'm looking for Shoes. I like a flexible sole.",
                "Actually, ignore my earlier preference. What I need is: leather.",
            ]),
            "intent_override",
        )
        self.assertEqual(
            observable_intent(vague + [
                "For that, what matters is: leather; color: black."
            ]),
            "buying",
        )

    def test_hash_vector_is_deterministic_and_history_sensitive(self) -> None:
        vectorizer = IntentVectorizer(hash_dimension=64)
        initial = ["I'm looking for Shoes, but I'm still exploring."]
        first = vectorizer.vectorize(initial, [])
        second = vectorizer.vectorize(initial, [])
        later = vectorizer.vectorize(
            initial + ["For that, what matters is: leather."], ["material"]
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (vectorizer.dimension,))
        self.assertFalse(np.array_equal(first, later))

    def test_observability_mask_and_model_round_trip(self) -> None:
        vectorizer = IntentVectorizer(hash_dimension=64)
        weights = np.zeros((len(INTENTS), vectorizer.dimension), dtype=np.float32)
        bias = np.asarray([0.0, 1.0, 20.0, 19.0], dtype=np.float32)
        model = IntentClassifier(weights, bias, hash_dimension=64)
        vague = ["I'm looking for Shoes, but I'm still exploring."]
        prediction = model.predict(vague)
        self.assertEqual(prediction.label, "browsing")
        self.assertTrue(prediction.provisional)
        self.assertEqual(prediction.probabilities["intent_override"], 0.0)
        self.assertEqual(prediction.probabilities["boundary"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.json"
            model.save(path)
            restored = IntentClassifier.load(path)
            np.testing.assert_allclose(
                model.predict_proba(vague), restored.predict_proba(vague)
            )

    def test_catalog_generation_is_product_grouped_and_target_safe(self) -> None:
        products = {
            f"P{index}": {
                "parent_asin": f"P{index}",
                "title": f"Shoe {index}",
                "categories": ["Clothing", "Shoes"],
                "features": ["leather", f"comfort feature {index}"],
                "details": {},
            }
            for index in range(8)
        }
        categories = {key: value["categories"] for key, value in products.items()}
        rows = generate_catalog_examples(
            products,
            categories,
            product_count=6,
            seed=4,
            excluded_parent_asins={"P0"},
            max_turns=3,
        )
        self.assertEqual(len(rows), 6 * 4 * 3)
        group_splits: dict[str, set[str]] = {}
        for row in rows:
            group_splits.setdefault(row.group_id, set()).add(row.split)
            self.assertNotIn("P0", " ".join(row.messages))
            self.assertIn(row.observable_label, INTENTS)
        self.assertTrue(all(len(values) == 1 for values in group_splits.values()))

    def test_confusion_matrix_uses_truth_rows(self) -> None:
        result = classification_metrics(
            ["buying", "buying", "boundary"],
            ["buying", "browsing", "boundary"],
        )
        self.assertEqual(result["count"], 3)
        matrix = result["confusion_matrix_truth_rows_prediction_columns"]
        self.assertEqual(matrix[INTENTS.index("buying")][INTENTS.index("browsing")], 1)
        self.assertEqual(matrix[INTENTS.index("boundary")][INTENTS.index("boundary")], 1)


if __name__ == "__main__":
    unittest.main()
