"""Dense semantic retrieval and weighted sparse/dense rank fusion.

The catalog embeddings are generated once by ``tools/build_catalog_embeddings.py``.
At runtime, a small OpenAI model rewrites the active typed constraints into a
compact catalog-style query, ``text-embedding-3-small`` embeds that query, and
the resulting nearest neighbours are fused with BM25 using weighted reciprocal
rank fusion (RRF).

Imports of optional third-party dependencies are deliberately lazy.  The
existing offline BM25 agent therefore remains usable when ``openai`` or
``numpy`` is not installed, when no API key is present, or when network access
is disabled by the evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from starter.semantic_query_prompt import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_MODEL,
    PROMPT_SUITE_VERSION,
    build_rewrite_input,
    canonical_constraints,
    deterministic_query,
    get_prompt_variant,
    sanitize_query,
)

QUERY_PROMPT_VERSION = PROMPT_SUITE_VERSION


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for optional OpenAI-backed dense retrieval."""

    enabled: bool = False
    embedding_path: str = "artifacts/catalog_embeddings.npy"
    ids_path: str = "artifacts/catalog_embedding_ids.json"
    cache_path: str = "artifacts/hybrid_query_cache.sqlite"
    query_model: str = DEFAULT_QUERY_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    query_prompt_variant: str = "natural_description"
    dimensions: int = 256
    # Best nonzero-dense public-set configuration. Hybrid remains opt-in
    # because sparse-only still achieved the higher overall TechnicalScore.
    sparse_weight: float = 0.85
    dense_weight: float = 0.15
    rrf_k: float = 60.0
    dense_pool_size: int = 100
    use_llm_query: bool = True
    dense_weight_schedule: tuple[float, ...] = ()
    schedule_basis: str = "constraints"
    strict: bool = False

    def __post_init__(self) -> None:
        numeric = (self.sparse_weight, self.dense_weight, self.rrf_k)
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("hybrid weights and rrf_k must be finite and non-negative")
        if self.sparse_weight == 0 and self.dense_weight == 0:
            raise ValueError("at least one hybrid retrieval weight must be positive")
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        if self.dense_pool_size < 1:
            raise ValueError("dense_pool_size must be positive")
        if self.schedule_basis not in {"constraints", "attributes"}:
            raise ValueError("schedule_basis must be 'constraints' or 'attributes'")
        get_prompt_variant(self.query_prompt_variant)
        if any(
            not math.isfinite(weight) or not 0 <= weight <= 1
            for weight in self.dense_weight_schedule
        ):
            raise ValueError("scheduled dense weights must be finite values from 0 to 1")

    def semantic_prompt_variant(self) -> str:
        """Return the tested rewrite style for this retrieval context."""

        return self.query_prompt_variant

    def fusion_weights(
        self,
        constraints: Sequence[Mapping[str, object]],
    ) -> tuple[float, float]:
        """Resolve sparse/dense weights for the current information state.

        Schedule item zero applies when one constraint/attribute is known,
        item one when two are known, and the final item is reused after the
        schedule is exhausted. With no schedule, the fixed configured weights
        are returned unchanged.
        """

        if not self.dense_weight_schedule:
            return self.sparse_weight, self.dense_weight
        if self.schedule_basis == "attributes":
            information_count = len({
                str(item.get("attribute", ""))
                for item in constraints
                if str(item.get("attribute", ""))
            })
        else:
            information_count = len(constraints)
        index = min(max(information_count, 1), len(self.dense_weight_schedule)) - 1
        dense_weight = self.dense_weight_schedule[index]
        return 1.0 - dense_weight, dense_weight


@dataclass(frozen=True)
class SemanticSearchResult:
    query_text: str
    ranked_ids: tuple[str, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def _compact(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        text = " ".join(f"{key} {item}" for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def catalog_embedding_text(product: Mapping[str, object], *, max_chars: int = 2400) -> str:
    """Create a stable catalog-style document for product precomputation."""

    fields = (
        ("Title", "title", 360),
        ("Category", "categories", 420),
        ("Features", "features", 900),
        ("Details", "details", 700),
        ("Store", "store", 180),
        ("Description", "description", 900),
    )
    parts: list[str] = []
    for label, key, field_limit in fields:
        value = _compact(product.get(key))
        if value:
            parts.append(f"{label}: {value[:field_limit]}")
    return ". ".join(parts)[:max_chars]


def deterministic_semantic_query(constraints: Sequence[Mapping[str, object]]) -> str:
    """Return a no-network semantic query used as an LLM/API fallback."""

    return deterministic_query(constraints)


def weighted_rrf(
    sparse_ids: Sequence[str],
    dense_ids: Sequence[str],
    *,
    sparse_weight: float,
    dense_weight: float,
    rrf_k: float = 60.0,
    limit: int = 100,
) -> list[str]:
    """Fuse ranked lists without assuming BM25 and cosine share a score scale."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if rrf_k < 0 or not math.isfinite(rrf_k):
        raise ValueError("rrf_k must be finite and non-negative")
    if any(weight < 0 or not math.isfinite(weight) for weight in (sparse_weight, dense_weight)):
        raise ValueError("fusion weights must be finite and non-negative")
    if sparse_weight == 0 and dense_weight == 0:
        return []

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for ranked, weight in ((sparse_ids, sparse_weight), (dense_ids, dense_weight)):
        if weight == 0:
            continue
        seen_in_route: set[str] = set()
        for rank, raw_id in enumerate(ranked, start=1):
            parent_asin = str(raw_id)
            if parent_asin in seen_in_route:
                continue
            seen_in_route.add(parent_asin)
            if parent_asin not in first_seen:
                first_seen[parent_asin] = counter
                counter += 1
            scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (rrf_k + rank)

    ordered = sorted(scores, key=lambda value: (-scores[value], first_seen[value]))
    return ordered[:limit]


class DenseCatalogIndex:
    """Memory-mapped exact cosine search over the frozen 50k catalog."""

    def __init__(self, embedding_path: str | Path, ids_path: str | Path, dimensions: int) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on host environment.
            raise RuntimeError("hybrid retrieval requires numpy") from exc

        embedding_path = Path(embedding_path)
        ids_path = Path(ids_path)
        if not embedding_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                "catalog embedding artifacts are missing; run tools/build_catalog_embeddings.py"
            )
        self._np = np
        self.matrix = np.load(embedding_path, mmap_mode="r")
        self.ids = tuple(str(value) for value in json.loads(ids_path.read_text(encoding="utf-8")))
        if self.matrix.ndim != 2 or self.matrix.shape[1] != dimensions:
            raise ValueError(
                f"embedding matrix has shape {self.matrix.shape}; expected (*, {dimensions})"
            )
        if self.matrix.shape[0] != len(self.ids):
            raise ValueError("embedding row count does not match catalog ID count")

    def search(self, query_embedding: object, top_k: int) -> tuple[str, ...]:
        np = self._np
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self.matrix.shape[1],):
            raise ValueError(
                f"query embedding has shape {query.shape}; expected {(self.matrix.shape[1],)}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return ()
        query = query / norm
        scores = self.matrix @ query
        count = min(max(0, top_k), len(self.ids))
        if count == 0:
            return ()
        if count == len(self.ids):
            indices = np.argsort(-scores)
        else:
            partition = np.argpartition(scores, -count)[-count:]
            indices = partition[np.argsort(-scores[partition])]
        return tuple(self.ids[int(index)] for index in indices[:count])


class HybridRetriever:
    """Generate/cache a semantic query, embed it, and search the dense index."""

    def __init__(self, config: HybridConfig) -> None:
        self.config = config
        self.index: DenseCatalogIndex | None = None
        self.unavailable_reason: str | None = None
        self._client: object | None = None
        self._cache: sqlite3.Connection | None = None
        self._lock = threading.RLock()

        try:
            self.index = DenseCatalogIndex(
                config.embedding_path,
                config.ids_path,
                config.dimensions,
            )
            cache_path = Path(config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS semantic_cache ("
                "cache_key TEXT PRIMARY KEY, query_text TEXT NOT NULL, embedding BLOB NOT NULL)"
            )
            self._cache.commit()
        except Exception as exc:
            self.unavailable_reason = str(exc)
            if config.strict:
                raise

    @property
    def available(self) -> bool:
        return self.index is not None and self._cache is not None

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def search(
        self,
        constraints: Sequence[Mapping[str, object]],
        *,
        top_k: int | None = None,
    ) -> SemanticSearchResult:
        fallback = deterministic_semantic_query(constraints)
        if not fallback:
            return SemanticSearchResult("", ())
        if not self.available:
            return SemanticSearchResult("", (), error=self.unavailable_reason)

        canonical = canonical_constraints(constraints)
        variant_name = self.config.semantic_prompt_variant()
        variant = get_prompt_variant(variant_name)
        cache_material = "\n".join((
            QUERY_PROMPT_VERSION,
            variant_name,
            self.config.query_model if self.config.use_llm_query else "deterministic",
            self.config.embedding_model,
            str(self.config.dimensions),
            canonical,
        ))
        cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
        assert self._cache is not None
        with self._lock:
            cached = self._cache.execute(
                "SELECT query_text, embedding FROM semantic_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if cached is not None:
            query_text = str(cached[0])
            embedding = self._embedding_from_blob(cached[1])
            ranked = self.index.search(embedding, top_k or self.config.dense_pool_size)
            return SemanticSearchResult(query_text, ranked)

        if not os.environ.get("OPENAI_API_KEY"):
            error = "OPENAI_API_KEY is not configured and this semantic query is not cached"
            if self.config.strict:
                raise RuntimeError(error)
            return SemanticSearchResult("", (), error=error)

        prompt_tokens = 0
        completion_tokens = 0
        query_text = fallback
        errors: list[str] = []
        client = self._openai_client()
        if client is None:
            return SemanticSearchResult("", (), error=self.unavailable_reason)

        if self.config.use_llm_query:
            try:
                response = client.responses.create(
                    model=self.config.query_model,
                    instructions=variant.instructions,
                    input=build_rewrite_input(constraints),
                    max_output_tokens=100,
                )
                query_text = sanitize_query(response.output_text, constraints)
                usage = getattr(response, "usage", None)
                prompt_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                completion_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            except Exception as exc:  # Fall back to the auditable typed text.
                errors.append(f"query rewrite failed: {exc}")

        try:
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=[query_text],
                dimensions=self.config.dimensions,
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            usage = getattr(response, "usage", None)
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        except Exception as exc:
            errors.append(f"query embedding failed: {exc}")
            if self.config.strict:
                raise
            return SemanticSearchResult(
                query_text,
                (),
                prompt_tokens,
                completion_tokens,
                "; ".join(errors),
            )

        blob = self._embedding_to_blob(embedding)
        with self._lock:
            self._cache.execute(
                "INSERT OR REPLACE INTO semantic_cache(cache_key, query_text, embedding) VALUES (?, ?, ?)",
                (cache_key, query_text, blob),
            )
            self._cache.commit()
        ranked = self.index.search(embedding, top_k or self.config.dense_pool_size)
        return SemanticSearchResult(
            query_text,
            ranked,
            prompt_tokens,
            completion_tokens,
            "; ".join(errors) or None,
        )

    def _openai_client(self) -> object | None:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                from openai import OpenAI

                self._client = OpenAI(timeout=20.0, max_retries=2)
                return self._client
            except Exception as exc:
                self.unavailable_reason = f"OpenAI client unavailable: {exc}"
                if self.config.strict:
                    raise
                return None

    def _embedding_to_blob(self, embedding: object) -> bytes:
        assert self.index is not None
        np = self.index._np
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.tobytes()

    def _embedding_from_blob(self, blob: bytes) -> object:
        assert self.index is not None
        vector = self.index._np.frombuffer(blob, dtype=self.index._np.float32)
        if vector.shape != (self.config.dimensions,):
            raise ValueError("cached query embedding has an unexpected dimension")
        return vector


__all__ = [
    "DenseCatalogIndex",
    "HybridConfig",
    "HybridRetriever",
    "SemanticSearchResult",
    "catalog_embedding_text",
    "deterministic_semantic_query",
    "weighted_rrf",
]
