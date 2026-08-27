from __future__ import annotations

import unittest

from starter.conversation_state import (
    Attribute,
    ConstraintSource,
    ConstraintStatus,
    ConversationState,
    MessageKind,
    classify_constraint,
)


class ConstraintClassificationTest(unittest.TestCase):
    def test_public_evaluator_taxonomy(self) -> None:
        examples = {
            "budget around $30": Attribute.BUDGET,
            "100% cotton": Attribute.MATERIAL,
            "color: purple": Attribute.COLOR,
            "wide width": Attribute.SIZE,
            "slim fit": Attribute.STYLE,
            "for outdoor hiking": Attribute.USE_CASE,
            "machine washable": Attribute.FEATURE,
        }

        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertIs(classify_constraint(text), expected)


class ConversationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "preference_tags": ["comfort", "fit"],
            "summary": "Prefers comfort and fit.",
        }
        self.state = ConversationState(self.profile)

    def test_retains_defensive_profile_copy_and_messages(self) -> None:
        self.profile["preference_tags"].append("mutated")
        update = self.state.observe(
            "I'm looking for women's shoes, but I'm still exploring.", turn=1
        )

        self.assertEqual(self.state.user_profile["preference_tags"], ["comfort", "fit"])
        self.assertEqual(self.state.messages[0].text, update.message.text)
        self.assertIs(update.message.kind, MessageKind.INITIAL)

    def test_parses_buying_and_browsing_initial_messages(self) -> None:
        buying = self.state.observe(
            "I'm looking for women's shoes. A key requirement is: waterproof.",
            turn=1,
        )

        self.assertEqual(
            [(item.value, item.attribute, item.source) for item in buying.added],
            [
                ("women's shoes", Attribute.CATEGORY, ConstraintSource.INITIAL_CATEGORY),
                ("waterproof", Attribute.FEATURE, ConstraintSource.INITIAL_REQUIREMENT),
            ],
        )

        second = ConversationState({})
        browsing = second.observe(
            "I'm looking for women's dresses, but I'm still exploring.", turn=1
        )
        self.assertEqual(len(browsing.added), 1)
        self.assertIs(browsing.added[0].attribute, Attribute.CATEGORY)

    def test_splits_and_classifies_clarification_constraints(self) -> None:
        update = self.state.observe(
            "For that, what matters is: cotton; color: blue.",
            turn=2,
            asked_attribute="other",
        )

        self.assertEqual(
            [(item.value, item.attribute) for item in update.added],
            [("cotton", Attribute.MATERIAL), ("color: blue", Attribute.COLOR)],
        )
        self.assertIs(update.message.asked_attribute, Attribute.OTHER)

    def test_tracks_both_no_preference_templates_without_search_constraints(self) -> None:
        first = self.state.observe(
            "I don't have a preference for material; please use your judgment.",
            turn=2,
            asked_attribute="material",
        )
        second = self.state.observe(
            "I don't have an additional preference for color.",
            turn=3,
            asked_attribute="color",
        )

        self.assertFalse(first.added)
        self.assertFalse(second.added)
        self.assertEqual(
            self.state.no_preference_attributes,
            {Attribute.MATERIAL, Attribute.COLOR},
        )

    def test_retry_boilerplate_does_not_become_constraint(self) -> None:
        update = self.state.observe(
            "Those options are not quite right yet. Ask me about one specific attribute.",
            turn=2,
        )

        self.assertIs(update.message.kind, MessageKind.RETRY_PROMPT)
        self.assertFalse(update.added)
        self.assertFalse(self.state.constraints)

    def test_override_retracts_only_initial_preference(self) -> None:
        self.state.observe(
            "I'm looking for men's coats. lightweight construction.", turn=1
        )
        self.state.observe(
            "For that, what matters is: waterproof; wool.", turn=2,
            asked_attribute="other",
        )
        update = self.state.observe(
            "Actually, ignore my earlier preference. What I need is: insulated lining.",
            turn=3,
        )

        self.assertEqual(len(update.retracted), 1)
        old = update.retracted[0]
        self.assertEqual(old.value, "lightweight construction")
        self.assertIs(old.status, ConstraintStatus.RETRACTED)
        self.assertEqual(old.retracted_turn, 3)
        self.assertIs(update.added[0].source, ConstraintSource.OVERRIDE)
        self.assertEqual(
            [item.value for item in self.state.active_constraints()],
            ["men's coats", "waterproof", "wool", "insulated lining"],
        )
        self.assertNotIn("lightweight construction", self.state.retrieval_text())

    def test_explicit_retraction_matches_only_named_constraint(self) -> None:
        self.state.observe(
            "For that, what matters is: cotton; waterproof.", turn=1
        )
        update = self.state.observe(
            "Actually, ignore cotton. What I need is: wool.", turn=2
        )

        self.assertEqual([item.value for item in update.retracted], ["cotton"])
        self.assertEqual(
            [item.value for item in self.state.active_constraints()],
            ["waterproof", "wool"],
        )

    def test_unknown_message_uses_single_conservative_fallback(self) -> None:
        update = self.state.observe(
            "Something lightweight with reinforced pockets would be ideal.", turn=4
        )

        self.assertEqual(len(update.added), 1)
        self.assertIs(update.added[0].source, ConstraintSource.FREE_TEXT)
        self.assertEqual(
            update.added[0].value,
            "Something lightweight with reinforced pockets would be ideal",
        )

    def test_later_positive_answer_supersedes_no_preference_marker(self) -> None:
        self.state.observe(
            "I don't have an additional preference for material.", turn=2
        )
        self.state.observe(
            "For that, what matters is: leather.", turn=3
        )

        self.assertNotIn(Attribute.MATERIAL, self.state.no_preference_attributes)
        self.assertIs(
            self.state.no_preference_history[0].status,
            ConstraintStatus.RETRACTED,
        )


if __name__ == "__main__":
    unittest.main()
