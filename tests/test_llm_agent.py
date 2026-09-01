import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.llm_agent import LLMAgent
from starter.local_llm import LocalLLM, explicit_finish_intent
from ui.server import App


class FakeLocalLLM:
    """Fast deterministic stand-in that exercises orchestration, not weights."""

    def __init__(self, model_name="fake-qwen"):
        self.model_name = model_name
        self.device = "cpu"
        self.error = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.rejects = 0
        self.generate_ms = 0.0

    def load(self):
        return True

    def _record(self, prompt, completion, milliseconds):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.generate_ms += milliseconds

    def extract(self, message):
        self._record(30, 8, 11.0)
        if "waterproof" not in message.lower():
            return []
        return [
            ("category", "jacket"),
            ("feature", "lightweight"),
            ("feature", "waterproof"),
            ("color", "blue"),
            ("use_case", "rainy hikes"),
        ]

    def rewrite(self, constraints, profile_tags=()):
        self._record(24, 6, 7.0)
        return " ".join(value for _, value in constraints)

    def reply(
        self,
        constraints,
        result_count,
        conversation=(),
        proposed_question=None,
        required_action=None,
    ):
        self._record(36, 10, 13.0)
        latest = conversation[-1]["content"].lower() if conversation else ""
        if required_action == "finish" or "satisfied" in latest or "that is all" in latest:
            return {
                "action": "finish",
                "reply": "Glad I could help—enjoy your selection!",
            }
        if not constraints:
            return {
                "action": "continue",
                "reply": "I'm ready to help you find the right item.",
            }
        return {
            "action": "continue",
            "reply": "You're looking for a lightweight waterproof blue jacket for rainy hikes.",
        }

    def usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "rejects": self.rejects,
            "generate_ms": self.generate_ms,
        }


class LLMFallthroughTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "B000000001",
                "title": "Lightweight Waterproof Blue Hiking Jacket",
                "categories": ["Clothing", "Jackets"],
                "features": ["lightweight", "waterproof", "blue"],
                "details": {"use": "rainy hikes"},
                "store": "Demo",
                "description": "A jacket for rainy hikes.",
            },
            {
                "parent_asin": "B000000002",
                "title": "Black Wool Coat",
                "categories": ["Clothing", "Coats"],
                "features": ["warm"],
                "details": {},
                "store": "Demo",
                "description": "A winter coat.",
            },
        ]
        self.catalog.write_text(
            "".join(json.dumps(item) + "\n" for item in products),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_random_language_falls_through_and_usage_is_per_turn(self):
        fake = FakeLocalLLM()
        with patch("starter.llm_agent.LocalLLM", return_value=fake):
            agent = LLMAgent(self.catalog, use_llm=True, use_dense=False)
        try:
            agent.reset("demo", {"preference_tags": ["packable"]})
            response = agent.respond(
                "demo",
                "I need a lightweight waterproof blue jacket for rainy hikes.",
                1,
                10,
            )
            self.assertEqual(agent.last_diagnostics["extraction_route"], "llm_fallthrough")
            self.assertTrue(agent.last_diagnostics["generated_reply"])
            self.assertIn("You're looking for", response["message"])
            self.assertEqual(response["usage"], {"prompt_tokens": 66, "completion_tokens": 18})

            response2 = agent.respond(
                "demo", "For that, what matters is: leather", 2, 10
            )
            self.assertEqual(agent.last_diagnostics["extraction_route"], "deterministic_parser")
            self.assertEqual(response2["usage"], {"prompt_tokens": 36, "completion_tokens": 10})
        finally:
            agent.close()

    def test_capability_mislabeled_as_material_is_coerced_to_feature(self):
        llm = LocalLLM()
        raw = '{"constraints":[{"attribute":"material","value":"waterproof"}]}'
        with patch.object(llm, "_generate", return_value=raw):
            self.assertEqual(
                llm.extract("I need something waterproof."),
                [("feature", "waterproof")],
            )

    def test_valid_empty_extraction_is_not_a_model_failure(self):
        llm = LocalLLM()
        with patch.object(llm, "_generate", return_value='{"constraints":[]}'):
            self.assertEqual(llm.extract("Hello there!"), [])
            self.assertEqual(llm.rejects, 0)

    def test_nonempty_but_ungrounded_extraction_activates_fallback(self):
        llm = LocalLLM()
        raw = '{"constraints":[{"attribute":"material","value":"wool"}]}'
        with patch.object(llm, "_generate", return_value=raw):
            self.assertIsNone(llm.extract("Hello there!"))
            self.assertGreater(llm.rejects, 0)

    def test_dialogue_reply_validates_finish_action(self):
        llm = LocalLLM()
        raw = '{"action":"finish","reply":"Glad I could help—enjoy your selection!"}'
        with patch.object(llm, "_generate", return_value=raw):
            result = llm.reply(
                [("category", "jacket")],
                10,
                conversation=[
                    {"role": "user", "content": "Find me a jacket."},
                    {"role": "assistant", "content": "Here are ten matches."},
                    {
                        "role": "user",
                        "content": "Thank you, I am satisfied with the recommendations.",
                    },
                ],
                proposed_question="What other preferences should I consider?",
            )
        self.assertEqual(result["action"], "finish")
        self.assertNotIn("?", result["reply"])

    def test_dialogue_reply_rejects_unapproved_action(self):
        llm = LocalLLM()
        raw = '{"action":"buy","reply":"I bought it for you."}'
        with patch.object(llm, "_generate", return_value=raw):
            self.assertIsNone(llm.reply([], 0))
        self.assertEqual(llm.rejects, 1)

    def test_explicit_finish_guard_is_high_precision(self):
        self.assertTrue(
            explicit_finish_intent(
                "Thank you, I am satisfied with the recommendations."
            )
        )
        self.assertTrue(explicit_finish_intent("I'm all set, thanks."))
        self.assertFalse(explicit_finish_intent("I am not satisfied yet."))
        self.assertFalse(explicit_finish_intent("Thanks, I also need it waterproof."))
        self.assertFalse(
            explicit_finish_intent("I am satisfied, but I also want to see red ones.")
        )

    def test_ui_exposes_fallthrough_diagnostics(self):
        fake = FakeLocalLLM()
        app = App.__new__(App)
        with patch("starter.llm_agent.LocalLLM", return_value=fake):
            app._build(self.catalog, True, "fake-qwen", False, False)
        try:
            app._start("ui-demo", {})
            payload = app._message(
                "ui-demo",
                "I need a lightweight waterproof blue jacket for rainy hikes.",
            )
            self.assertTrue(payload["generated"])
            self.assertEqual(
                payload["llm_diagnostics"]["extraction_route"], "llm_fallthrough"
            )
            self.assertEqual(payload["sections"][0]["id"], "llm_fallthrough")
            self.assertEqual(payload["sections"][0]["badge"], "llm fallthrough")
            self.assertGreater(payload["generate_ms"], 0)
            self.assertGreater(payload["usage"]["prompt_tokens"], 0)
        finally:
            app.agent.close()

    def test_message_without_constraints_still_gets_generated_acknowledgement(self):
        fake = FakeLocalLLM()
        app = App.__new__(App)
        with patch("starter.llm_agent.LocalLLM", return_value=fake):
            app._build(self.catalog, True, "fake-qwen", False, False)
        try:
            app._start("ui-greeting", {})
            payload = app._message("ui-greeting", "Hello there!")
            self.assertTrue(payload["generated"])
            self.assertIn("I'm ready to help", payload["message"])
            self.assertIn("preferences", payload["message"])
            self.assertEqual(
                payload["llm_diagnostics"]["extraction_route"],
                "llm_fallthrough",
            )
            self.assertEqual(payload["llm_diagnostics"]["constraints"], [])
            self.assertEqual(payload["llm_diagnostics"]["llm_calls"], 2)
        finally:
            app.agent.close()

    def test_satisfaction_ends_conversation_and_suppresses_entropy_question(self):
        fake = FakeLocalLLM()
        app = App.__new__(App)
        with patch("starter.llm_agent.LocalLLM", return_value=fake):
            app._build(self.catalog, True, "fake-qwen", True, False)
        try:
            app._start("ui-satisfied", {})
            first = app._message(
                "ui-satisfied",
                "I need a lightweight waterproof blue jacket for rainy hikes.",
            )
            self.assertIsNotNone(first["ask_attribute"])

            finished = app._message(
                "ui-satisfied",
                "Thank you, I am satisfied with the recommendations.",
            )
            self.assertEqual(
                finished["message"],
                "Glad I could help—enjoy your selection!",
            )
            self.assertIsNone(finished["ask_attribute"])
            self.assertEqual(finished["entropy_scores"], [])
            self.assertTrue(finished["llm_diagnostics"]["conversation_complete"])
            self.assertEqual(
                finished["llm_diagnostics"]["dialogue_action"], "finish"
            )
            self.assertEqual(len(finished["recommendations"]), 1)
            self.assertEqual(
                app.agent._dialogue_history["ui-satisfied"][-2]["content"],
                "Thank you, I am satisfied with the recommendations.",
            )
        finally:
            app.agent.close()


if __name__ == "__main__":
    unittest.main()
