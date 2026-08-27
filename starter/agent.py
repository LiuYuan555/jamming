from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from starter.constraint_reranker import CatalogReranker, Constraint, RerankConfig
from starter.conversation_state import (
    ConstraintSource,
    ConversationState,
)
from starter.question_policy import QuestionPolicy


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


@dataclass(frozen=True)
class AgentConfig:
    """Switches used for controlled component and combination ablations."""

    question_policy: str = "always_other"
    use_typed_state: bool = True
    use_reranker: bool = True
    candidate_pool_size: int = 100
    rerank_config: RerankConfig = field(default_factory=RerankConfig)

    def __post_init__(self) -> None:
        if self.candidate_pool_size < 10:
            raise ValueError("candidate_pool_size must be at least 10")


@dataclass
class SessionState:
    """Information retained for one evaluator conversation."""

    user_profile: dict
    messages: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    seen_terms: set[str] = field(default_factory=set)
    asked_attributes: list[str] = field(default_factory=list)
    last_asked_attribute: str | None = None
    structured: ConversationState | None = None


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
    """Configurable offline shopping agent used for controlled ablations."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or AgentConfig()
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self.question_policy = QuestionPolicy(self.config.question_policy)
        self._build_index()
        self.reranker = (
            CatalogReranker(self.catalog_path, config=self.config.rerank_config)
            if self.config.use_reranker
            else None
        )

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
        self._sessions[session_id] = SessionState(
            user_profile=dict(user_profile),
            structured=(
                ConversationState(user_profile)
                if self.config.use_typed_state
                else None
            ),
        )

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
        if state.structured is not None:
            state.structured.observe(
                user_message,
                turn,
                asked_attribute=state.last_asked_attribute,
            )
            query_terms = list(dict.fromkeys(_terms(state.structured.retrieval_text())))
        else:
            for term in _terms(_information_text(user_message)):
                if term not in state.seen_terms:
                    state.seen_terms.add(term)
                    state.query_terms.append(term)
            query_terms = state.query_terms

        # Search the accumulated conversation instead of forgetting earlier
        # requirements whenever a clarification response arrives.
        expression = " OR ".join(f'"{term}"' for term in query_terms[:80])
        if not expression:
            recommendations: list[dict] = []
        else:
            candidate_limit = (
                self.config.candidate_pool_size
                if self.reranker is not None
                else top_k
            )
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, candidate_limit),
            ).fetchall()
            candidate_ids = [str(row[0]) for row in rows]
            if self.reranker is not None and state.structured is not None:
                constraints = [
                    Constraint(
                        record.attribute.value,
                        record.value,
                        hard=record.source
                        in {
                            ConstraintSource.INITIAL_CATEGORY,
                            ConstraintSource.INITIAL_REQUIREMENT,
                            ConstraintSource.OVERRIDE,
                        },
                    )
                    for record in state.structured.active_constraints()
                ]
                recommendations = [
                    candidate.recommendation()
                    for candidate in self.reranker.rerank(
                        candidate_ids,
                        constraints,
                        top_k=top_k,
                    )
                ]
            else:
                recommendations = [
                    {"parent_asin": parent_asin}
                    for parent_asin in candidate_ids[:top_k]
                ]

        decision = self.question_policy.decide(state.asked_attributes)
        if decision.ask_attribute is not None:
            state.asked_attributes.append(decision.ask_attribute)
        state.last_asked_attribute = decision.ask_attribute
        return {
            "message": decision.message,
            "ask_attribute": decision.ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
