from __future__ import annotations

import unittest

from starter.question_policy import QuestionPolicy, QuestionPolicyConfig


class QuestionPolicyTest(unittest.TestCase):
    def test_none_policy_never_asks_a_structured_question(self) -> None:
        policy = QuestionPolicy("none")

        self.assertIsNone(policy.decide([]).ask_attribute)
        self.assertIsNone(policy.decide(["feature", "material"]).ask_attribute)

    def test_fixed_policies_repeat_the_configured_attribute(self) -> None:
        cases = {
            "always_other": "other",
            "always_feature": "feature",
            "always_material": "material",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                policy = QuestionPolicy(QuestionPolicyConfig(name=name))
                self.assertEqual(policy.decide([]).ask_attribute, expected)
                self.assertEqual(policy.decide([expected, expected]).ask_attribute, expected)

    def test_feature_first_sequence_and_fallback(self) -> None:
        policy = QuestionPolicy("feature_first")

        actual = [policy.decide(["asked"] * turn).ask_attribute for turn in range(6)]

        self.assertEqual(
            actual,
            ["feature", "material", "color", "other", "other", "other"],
        )

    def test_other_first_sequence_and_fallback(self) -> None:
        policy = QuestionPolicy("other_first")

        actual = [policy.decide(["asked"] * turn).ask_attribute for turn in range(6)]

        self.assertEqual(
            actual,
            ["other", "feature", "material", "color", "other", "other"],
        )

    def test_message_agrees_with_selected_attribute(self) -> None:
        policy = QuestionPolicy("feature_first")

        feature = policy.decide([])
        material = policy.decide(["feature"])

        self.assertIn("features", feature.message.lower())
        self.assertIn("material", material.message.lower())

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown question policy"):
            QuestionPolicy("not_a_policy")

    def test_supported_policy_names_are_discoverable(self) -> None:
        self.assertEqual(
            set(QuestionPolicy.names()),
            {
                "none",
                "always_other",
                "always_feature",
                "always_material",
                "feature_first",
                "other_first",
            },
        )


if __name__ == "__main__":
    unittest.main()
