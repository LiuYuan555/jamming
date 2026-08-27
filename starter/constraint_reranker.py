"""Offline, constraint-aware reranking for BM25 product candidates.

The existing agent can request a wider BM25 candidate pool and pass its ordered
``parent_asin`` values to :class:`CatalogReranker`.  The reranker preserves a
small retrieval-rank prior, then rewards typed constraints that match the
catalog fields in which that constraint is expected to occur.

This module intentionally does not extract constraints from conversation text.
Keeping extraction and scoring separate makes both stages easier to ablate and
prevents an uncertain parser from silently becoming a hard product filter.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:\$|usd\s*)?(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
ALLOWED_ATTRIBUTES = frozenset({
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
})
TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")

# Token overlap ignores conversational glue while exact-phrase matching still
# uses the complete normalized value.  This makes a verbatim feature disclosure
# decisive without rewarding every product for words such as "the" and "with".
OVERLAP_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
})


DEFAULT_FIELD_WEIGHTS: Mapping[str, float] = {
    "title": 1.35,
    "categories": 1.15,
    "features": 1.30,
    "details": 1.10,
    "store": 1.25,
    "description": 0.90,
}

ATTRIBUTE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "category": ("categories", "title"),
    "material": ("features", "details", "title", "description"),
    "color": ("title", "features", "details", "description"),
    "size": ("details", "features", "title", "description"),
    "style": ("title", "categories", "features", "details", "description"),
    "brand": ("store", "title"),
    "feature": ("features", "details", "title", "description"),
    "use_case": ("features", "description", "title", "categories"),
    "other": TEXT_FIELDS,
}


def normalize_text(value: object) -> str:
    """Return lowercase alphanumeric text with whitespace collapsed."""

    return " ".join(token.lower() for token in TOKEN_RE.findall(str(value or "")))


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)


@dataclass(frozen=True)
class Constraint:
    """A product requirement extracted from the conversation.

    ``hard`` controls the missing-match penalty; it is not a hard filter unless
    ``RerankConfig.drop_hard_misses`` is enabled.  ``negated`` represents an
    exclusion such as "not leather".  ``importance`` can down-weight uncertain
    extraction or weak profile preferences.
    """

    attribute: str
    value: str
    hard: bool = True
    negated: bool = False
    importance: float = 1.0

    def __post_init__(self) -> None:
        if self.attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"unsupported constraint attribute: {self.attribute!r}")
        if not normalize_text(self.value):
            raise ValueError("constraint value must contain searchable text")
        if not math.isfinite(self.importance) or self.importance < 0:
            raise ValueError("constraint importance must be a finite non-negative number")


@dataclass(frozen=True)
class RerankConfig:
    """Scoring weights exposed for controlled ablation experiments."""

    retrieval_rank_weight: float = 1.0
    retrieval_rank_decay: float = 0.20
    exact_phrase_bonus: float = 7.0
    token_overlap_weight: float = 3.0
    hard_miss_penalty: float = 2.0
    negated_match_penalty: float = 10.0
    budget_match_bonus: float = 7.0
    budget_around_tolerance: float = 0.15
    drop_hard_misses: bool = False
    field_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS)
    )

    def __post_init__(self) -> None:
        numeric_values = (
            self.retrieval_rank_weight,
            self.retrieval_rank_decay,
            self.exact_phrase_bonus,
            self.token_overlap_weight,
            self.hard_miss_penalty,
            self.negated_match_penalty,
            self.budget_match_bonus,
            self.budget_around_tolerance,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric_values):
            raise ValueError("reranking weights must be finite non-negative numbers")
        unknown_fields = set(self.field_weights) - set(TEXT_FIELDS)
        if unknown_fields:
            raise ValueError(f"unknown field weights: {sorted(unknown_fields)}")
        if any(not math.isfinite(value) or value < 0 for value in self.field_weights.values()):
            raise ValueError("field weights must be finite non-negative numbers")


@dataclass(frozen=True)
class ProductDocument:
    parent_asin: str
    fields: Mapping[str, str]
    price: float | None

    @classmethod
    def from_product(cls, product: Mapping[str, object]) -> "ProductDocument":
        raw_price = product.get("price")
        try:
            price = float(raw_price) if raw_price not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        return cls(
            parent_asin=str(product["parent_asin"]),
            fields={field_name: normalize_text(_flatten(product.get(field_name))) for field_name in TEXT_FIELDS},
            price=price,
        )


@dataclass(frozen=True)
class ScoredCandidate:
    parent_asin: str
    score: float
    retrieval_rank: int
    matched_constraints: tuple[str, ...]
    missed_hard_constraints: tuple[str, ...]
    violated_constraints: tuple[str, ...]

    def recommendation(self) -> dict[str, str]:
        """Return the evaluator-compatible recommendation payload."""

        return {"parent_asin": self.parent_asin}


def _overlap_tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in normalize_text(value).split() if token not in OVERLAP_STOPWORDS)


def _text_match_quality(
    document: ProductDocument,
    constraint: Constraint,
    config: RerankConfig,
) -> float:
    query = normalize_text(constraint.value)
    query_tokens = _overlap_tokens(query)
    best = 0.0
    for field_name in ATTRIBUTE_FIELDS[constraint.attribute]:
        field_text = document.fields.get(field_name, "")
        if not field_text:
            continue
        overlap = 0.0
        if query_tokens:
            overlap = len(query_tokens.intersection(field_text.split())) / len(query_tokens)
        # Padding enforces token boundaries (for example, color ``red`` must
        # not count as an exact phrase inside ``shredded``).
        phrase = config.exact_phrase_bonus if f" {query} " in f" {field_text} " else 0.0
        field_weight = config.field_weights.get(field_name, 1.0)
        best = max(best, field_weight * (phrase + config.token_overlap_weight * overlap))
    return best


def _budget_match_quality(
    document: ProductDocument,
    constraint: Constraint,
    config: RerankConfig,
) -> float:
    numbers = MONEY_RE.findall(constraint.value.replace(",", ""))
    if not numbers or document.price is None:
        return 0.0
    budget = float(numbers[-1])
    lowered = constraint.value.lower()
    if any(marker in lowered for marker in ("under", "below", "up to", "at most", "<=")):
        matches = document.price <= budget
    elif any(marker in lowered for marker in ("over", "above", "at least", ">=")):
        matches = document.price >= budget
    else:
        tolerance = max(1.0, budget * config.budget_around_tolerance)
        matches = abs(document.price - budget) <= tolerance
    return config.budget_match_bonus if matches else 0.0


def score_document(
    document: ProductDocument,
    constraints: Sequence[Constraint],
    retrieval_rank: int,
    config: RerankConfig,
) -> ScoredCandidate:
    """Score one normalized product while retaining diagnostics."""

    if retrieval_rank < 1:
        raise ValueError("retrieval_rank must be at least one")
    score = config.retrieval_rank_weight / (
        1.0 + config.retrieval_rank_decay * (retrieval_rank - 1)
    )
    matched: list[str] = []
    missed: list[str] = []
    violated: list[str] = []

    for constraint in constraints:
        quality = (
            _budget_match_quality(document, constraint, config)
            if constraint.attribute == "budget"
            else _text_match_quality(document, constraint, config)
        )
        label = f"{constraint.attribute}:{constraint.value}"
        if constraint.negated:
            if quality > 0:
                score -= config.negated_match_penalty * constraint.importance
                violated.append(label)
            continue
        if quality > 0:
            score += quality * constraint.importance
            matched.append(label)
        elif constraint.hard:
            score -= config.hard_miss_penalty * constraint.importance
            missed.append(label)

    return ScoredCandidate(
        parent_asin=document.parent_asin,
        score=score,
        retrieval_rank=retrieval_rank,
        matched_constraints=tuple(matched),
        missed_hard_constraints=tuple(missed),
        violated_constraints=tuple(violated),
    )


def rerank_documents(
    documents: Sequence[ProductDocument],
    constraints: Sequence[Constraint],
    *,
    top_k: int = 10,
    config: RerankConfig | None = None,
) -> list[ScoredCandidate]:
    """Rerank documents supplied in BM25 order.

    Ties are stable because retrieval rank is the secondary ordering key.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    active_config = config or RerankConfig()
    scored = [
        score_document(document, constraints, rank, active_config)
        for rank, document in enumerate(documents, start=1)
    ]
    if active_config.drop_hard_misses:
        scored = [candidate for candidate in scored if not candidate.missed_hard_constraints]
    scored.sort(key=lambda candidate: (-candidate.score, candidate.retrieval_rank))
    return scored[:top_k]


class CatalogReranker:
    """Compact in-memory catalog lookup used after SQLite FTS5 retrieval."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        config: RerankConfig | None = None,
    ) -> None:
        self.config = config or RerankConfig()
        self.documents: dict[str, ProductDocument] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = ProductDocument.from_product(json.loads(line))
                self.documents[document.parent_asin] = document

    def rerank(
        self,
        candidate_ids: Iterable[str],
        constraints: Sequence[Constraint],
        *,
        top_k: int = 10,
    ) -> list[ScoredCandidate]:
        """Rerank known candidate IDs while preserving BM25 order.

        Duplicate and unknown IDs are ignored, matching the evaluator's
        recommendation-normalization behavior.
        """

        documents: list[ProductDocument] = []
        seen: set[str] = set()
        for value in candidate_ids:
            parent_asin = str(value)
            if parent_asin in seen or parent_asin not in self.documents:
                continue
            seen.add(parent_asin)
            documents.append(self.documents[parent_asin])
        return rerank_documents(documents, constraints, top_k=top_k, config=self.config)
