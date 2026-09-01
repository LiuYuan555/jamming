from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from starter.constraint_reranker import CatalogReranker, Constraint, RerankConfig
from starter.conversation_state import (
    ConstraintSource,
    ConversationState,
)
from starter.hybrid_retrieval import HybridConfig, HybridRetriever, weighted_rrf
from starter.question_policy import QuestionPolicy


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


@dataclass(frozen=True)
class BM25FieldWeights:
    """SQLite FTS5 BM25 weights in catalog-column order."""

    title: float = 6.0
    categories: float = 4.0
    features: float = 2.5
    details: float = 2.5
    store: float = 1.5
    description: float = 1.0

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("BM25 field weights must be finite non-negative numbers")
        if not any(values):
            raise ValueError("at least one BM25 field weight must be positive")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.title,
            self.categories,
            self.features,
            self.details,
            self.store,
            self.description,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "title": self.title,
            "categories": self.categories,
            "features": self.features,
            "details": self.details,
            "store": self.store,
            "description": self.description,
        }


@dataclass(frozen=True)
class CategoryRetrievalConfig:
    """Select the global BM25 route, category scope, or rank fusion."""

    mode: str = "global"
    global_weight: float = 0.5
    category_weight: float = 0.5
    rrf_k: float = 60.0
    trace_all_routes: bool = False

    def __post_init__(self) -> None:
        allowed_modes = {
            "global",
            "category_only",
            "fused",
            "clean_or",
            "clean_hard_soft",
            "clean_constraint_rrf",
            "clean_global_fused",
        }
        if self.mode not in allowed_modes:
            raise ValueError(
                "category retrieval mode must be one of: "
                + ", ".join(sorted(allowed_modes))
            )
        values = (self.global_weight, self.category_weight, self.rrf_k)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("category retrieval weights must be finite and non-negative")
        if self.rrf_k <= 0:
            raise ValueError("category retrieval rrf_k must be positive")
        if self.mode in {"fused", "clean_global_fused"} and (
            self.global_weight + self.category_weight <= 0
        ):
            raise ValueError("fused category retrieval needs a positive route weight")


@dataclass(frozen=True)
class AgentConfig:
    """Switches used for controlled component and combination ablations."""

    question_policy: str = "always_other"
    use_typed_state: bool = True
    use_reranker: bool = True
    candidate_pool_size: int = 100
    bm25_weights: BM25FieldWeights = field(default_factory=BM25FieldWeights)
    category_retrieval: CategoryRetrievalConfig = field(
        default_factory=CategoryRetrievalConfig
    )
    rerank_config: RerankConfig = field(default_factory=RerankConfig)
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)

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
    last_retrieval: dict[str, tuple[str, ...]] = field(default_factory=dict)
    retrieval_history: list[dict[str, tuple[str, ...]]] = field(default_factory=list)


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
        self.hybrid_retriever = (
            HybridRetriever(self.config.hybrid_config)
            if self.config.hybrid_config.enabled
            else None
        )

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        cursor.execute(
            "CREATE TABLE product_category_scope("
            "rowid INTEGER PRIMARY KEY, terms TEXT NOT NULL)"
        )
        batch: list[tuple[int, str, str, str, str, str, str, str]] = []
        scope_batch: list[tuple[int, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for rowid, line in enumerate(handle, start=1):
                product = json.loads(line)
                categories_text = _text(product.get("categories"))
                batch.append(
                    (
                        rowid,
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        categories_text,
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                scope_batch.append(
                    (rowid, " " + " ".join(dict.fromkeys(_terms(categories_text))) + " ")
                )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products("
                        "rowid, parent_asin, title, categories, features, details, store, description"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    cursor.executemany(
                        "INSERT INTO product_category_scope(rowid, terms) VALUES (?, ?)",
                        scope_batch,
                    )
                    batch.clear()
                    scope_batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products("
                "rowid, parent_asin, title, categories, features, details, store, description"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            cursor.executemany(
                "INSERT INTO product_category_scope(rowid, terms) VALUES (?, ?)",
                scope_batch,
            )
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

    def close(self) -> None:
        """Release local database/cache resources held by the agent."""

        if self.hybrid_retriever is not None:
            self.hybrid_retriever.close()
        self.connection.close()

    def _bm25_search(self, expression: str, limit: int) -> list[str]:
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, ?, ?, ?, ?, ?, ?) LIMIT ?",
            (expression, *self.config.bm25_weights.as_tuple(), limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _bm25_search_scoped(
        self,
        expression: str,
        category_terms: list[str],
        limit: int,
    ) -> list[str]:
        """Apply category membership as an unscored SQL filter."""

        if not expression or not category_terms:
            return []
        scope_checks = " AND ".join(
            "instr(product_category_scope.terms, ?) > 0" for _ in category_terms
        )
        rows = self.connection.execute(
            "SELECT products.parent_asin FROM products "
            "JOIN product_category_scope ON product_category_scope.rowid = products.rowid "
            "WHERE products MATCH ? AND "
            + scope_checks
            + " ORDER BY bm25(products, 0.0, ?, ?, ?, ?, ?, ?) LIMIT ?",
            (
                expression,
                *(f" {term} " for term in category_terms),
                *self.config.bm25_weights.as_tuple(),
                limit,
            ),
        ).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _category_scoped_expression(
        expression: str,
        active_records: list,
    ) -> str | None:
        category_values = [
            record.value
            for record in active_records
            if record.attribute.value == "category"
        ]
        category_terms = list(dict.fromkeys(_terms(" ".join(category_values))))
        if not expression or not category_terms:
            return None
        category_clause = " AND ".join(f'"{term}"' for term in category_terms)
        return f"categories : ({category_clause}) AND ({expression})"

    @staticmethod
    def _category_scope_terms(active_records: list) -> list[str]:
        values = [
            record.value
            for record in active_records
            if record.attribute.value == "category"
        ]
        return list(dict.fromkeys(_terms(" ".join(values))))

    @staticmethod
    def _constraint_expression_or(constraints: list[Constraint]) -> str:
        terms = list(dict.fromkeys(
            term
            for constraint in constraints
            if constraint.attribute != "category"
            for term in _terms(constraint.value)
        ))
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _hard_soft_expression(constraints: list[Constraint]) -> str:
        hard_terms = list(dict.fromkeys(
            term
            for constraint in constraints
            if constraint.attribute != "category" and constraint.hard
            for term in _terms(constraint.value)
        ))
        soft_terms = list(dict.fromkeys(
            term
            for constraint in constraints
            if constraint.attribute != "category" and not constraint.hard
            for term in _terms(constraint.value)
        ))
        hard = " AND ".join(f'"{term}"' for term in hard_terms)
        soft = " OR ".join(f'"{term}"' for term in soft_terms)
        if hard and soft:
            return f"({hard}) AND ({soft})"
        return hard or soft

    @staticmethod
    def _multi_rrf(
        rankings: list[list[str]],
        *,
        rrf_k: float,
        limit: int,
    ) -> list[str]:
        scores: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        order = 0
        for ranking in rankings:
            for rank, parent_asin in enumerate(ranking, start=1):
                if parent_asin not in first_seen:
                    first_seen[parent_asin] = order
                    order += 1
                scores[parent_asin] = scores.get(parent_asin, 0.0) + 1.0 / (
                    rrf_k + rank
                )
        return sorted(
            scores,
            key=lambda parent_asin: (-scores[parent_asin], first_seen[parent_asin]),
        )[:limit]

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

        active_records = (
            state.structured.active_constraints()
            if state.structured is not None
            else []
        )
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
            for record in active_records
        ]
        semantic_constraints = [
            {
                "attribute": constraint.attribute,
                "value": constraint.value,
                "hard": constraint.hard,
            }
            for constraint in constraints
        ]
        prompt_tokens = 0
        completion_tokens = 0

        # Search the accumulated conversation instead of forgetting earlier
        # requirements whenever a clarification response arrives.
        expression = " OR ".join(f'"{term}"' for term in query_terms[:80])
        hybrid_enabled = self.hybrid_retriever is not None
        if not expression and not hybrid_enabled:
            recommendations: list[dict] = []
        else:
            candidate_limit = (
                self.config.candidate_pool_size
                if self.reranker is not None
                else top_k
            )
            category_expression = self._category_scoped_expression(expression, active_records)
            category_config = self.config.category_retrieval
            clean_modes = {
                "clean_or",
                "clean_hard_soft",
                "clean_constraint_rrf",
                "clean_global_fused",
            }
            category_terms = self._category_scope_terms(active_records)
            need_global = (
                category_config.mode in {"global", "fused", "clean_global_fused"}
                or category_expression is None
                or category_config.trace_all_routes
            )
            need_category = (
                category_expression is not None
                and (
                    category_config.mode in {"category_only", "fused"}
                    or category_config.trace_all_routes
                )
            )
            global_ids = (
                self._bm25_search(expression, candidate_limit)
                if expression and need_global
                else []
            )
            category_ids = (
                self._bm25_search(category_expression, candidate_limit)
                if need_category and category_expression is not None
                else []
            )
            clean_ids: list[str] = []
            constraint_routes: list[list[str]] = []
            if category_config.mode in clean_modes and category_terms:
                if category_config.mode == "clean_hard_soft":
                    clean_expression = self._hard_soft_expression(constraints)
                else:
                    clean_expression = self._constraint_expression_or(constraints)
                # Browsing starts with only a category. Use the normal query
                # for deterministic cold-start ordering until a requirement
                # becomes available.
                clean_expression = clean_expression or expression
                if category_config.mode == "clean_constraint_rrf":
                    for constraint in constraints:
                        if constraint.attribute == "category":
                            continue
                        route_terms = list(dict.fromkeys(_terms(constraint.value)))
                        route_expression = " OR ".join(
                            f'"{term}"' for term in route_terms
                        )
                        if route_expression:
                            constraint_routes.append(
                                self._bm25_search_scoped(
                                    route_expression,
                                    category_terms,
                                    candidate_limit,
                                )
                            )
                    if constraint_routes:
                        clean_ids = self._multi_rrf(
                            constraint_routes,
                            rrf_k=category_config.rrf_k,
                            limit=candidate_limit,
                        )
                    else:
                        clean_ids = self._bm25_search_scoped(
                            clean_expression,
                            category_terms,
                            candidate_limit,
                        )
                else:
                    clean_ids = self._bm25_search_scoped(
                        clean_expression,
                        category_terms,
                        candidate_limit,
                    )
            if category_config.mode == "category_only" and category_expression is not None:
                sparse_ids = category_ids
            elif category_config.mode == "fused" and category_expression is not None:
                sparse_ids = weighted_rrf(
                    global_ids,
                    category_ids,
                    sparse_weight=category_config.global_weight,
                    dense_weight=category_config.category_weight,
                    rrf_k=category_config.rrf_k,
                    limit=candidate_limit,
                )
            elif category_config.mode == "clean_global_fused" and category_terms:
                sparse_ids = weighted_rrf(
                    global_ids,
                    clean_ids,
                    sparse_weight=category_config.global_weight,
                    dense_weight=category_config.category_weight,
                    rrf_k=category_config.rrf_k,
                    limit=candidate_limit,
                )
            elif category_config.mode in clean_modes and category_terms:
                sparse_ids = clean_ids
            else:
                sparse_ids = global_ids
            retrieval_trace = {
                "global": tuple(global_ids),
                "category": tuple(category_ids),
                "clean_category": tuple(clean_ids),
                "selected_sparse": tuple(sparse_ids),
            }
            state.last_retrieval = retrieval_trace
            state.retrieval_history.append(retrieval_trace)
            candidate_ids = sparse_ids
            if self.hybrid_retriever is not None:
                sparse_weight, dense_weight = self.config.hybrid_config.fusion_weights(
                    semantic_constraints
                )
                dense_result = (
                    self.hybrid_retriever.search(
                        semantic_constraints,
                        top_k=self.config.hybrid_config.dense_pool_size,
                    )
                    if dense_weight > 0
                    else None
                )
                dense_ids = () if dense_result is None else dense_result.ranked_ids
                if dense_result is not None:
                    prompt_tokens += dense_result.prompt_tokens
                    completion_tokens += dense_result.completion_tokens
                candidate_ids = weighted_rrf(
                    sparse_ids,
                    dense_ids,
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    rrf_k=self.config.hybrid_config.rrf_k,
                    limit=candidate_limit,
                )
            if self.reranker is not None and state.structured is not None:
                rerank_constraints = list(constraints)
                for tag in state.user_profile.get("preference_tags", []):
                    rerank_constraints.append(Constraint(attribute="other", value=tag, hard=False, importance=0.1))
                    
                recommendations = [
                    candidate.recommendation()
                    for candidate in self.reranker.rerank(
                        candidate_ids,
                        rerank_constraints,
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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

from starter.intent_hybrid_router import IntentConditionedHybridAgent

class Agent(IntentConditionedHybridAgent):
    def __init__(
        self,
        catalog_path="data/catalog.jsonl",
        config=None,
    ) -> None:
        from starter.intent_classifier import IntentClassifier
        from starter.agent import AgentConfig
        from starter.hybrid_retrieval import HybridConfig
        
        try:
            classifier = IntentClassifier.load("artifacts/techjam_intent_classifier.json")
        except FileNotFoundError:
            classifier = None
            
        super().__init__(
            catalog_path=catalog_path,
            config=AgentConfig(
                question_policy="shannon_entropy", 
                hybrid_config=HybridConfig(enabled=False)
            ),
            classifier=classifier,
            use_entropy_gate=True,
            use_intent_topk=True
        )
