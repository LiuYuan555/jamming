from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

QUESTION_MESSAGES = (
    "What features or qualities matter most to you?",
    "Is there anything else the product must have?",
    "Are there any other preferences I should consider?",
)


@dataclass
class SessionState:
    """Information retained for one evaluator conversation."""

    user_profile: dict
    messages: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    seen_terms: set[str] = field(default_factory=set)
    asked_attributes: list[str] = field(default_factory=list)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _information_text(message: str) -> str:
    """Remove deterministic simulator boilerplate before adding search terms."""

    lowered = message.lower()
    if lowered.startswith("those options are not quite right"):
        return ""
    if lowered.startswith("i don't have a preference"):
        return ""
    if lowered.startswith("i don't have an additional preference"):
        return ""

    # Clarification and override replies place the useful payload after these
    # markers. Unknown wording falls back to the complete message so that the
    # agent does not depend entirely on the public simulator templates.
    for marker in ("for that, what matters is:", "what i need is:"):
        position = lowered.rfind(marker)
        if position >= 0:
            return message[position + len(marker):]
    return message


class Agent:
    """Stateful BM25 agent with deterministic clarification and no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Retain the profile for later personalization work, but do not use it
        # as a retrieval signal in this first state-and-clarification step.
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        state.messages.append(user_message)
        for term in _terms(_information_text(user_message)):
            if term not in state.seen_terms:
                state.seen_terms.add(term)
                state.query_terms.append(term)

        # Search the accumulated conversation instead of forgetting earlier
        # requirements whenever a clarification response arrives.
        expression = " OR ".join(f'"{term}"' for term in state.query_terms[:80])
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]

        question_index = min(len(state.asked_attributes), len(QUESTION_MESSAGES) - 1)
        state.asked_attributes.append("other")
        return {
            "message": QUESTION_MESSAGES[question_index],
            # The official simulator treats "other" as an open clarification
            # channel and can disclose up to two remaining constraints.
            "ask_attribute": "other",
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
