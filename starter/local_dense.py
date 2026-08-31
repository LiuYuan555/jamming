"""Offline dense retrieval over the frozen catalog.

Replaces the OpenAI ``text-embedding-3-small`` route in
:mod:`starter.hybrid_retrieval` with a local sentence-transformer, so the agent
has no API key, no network dependency and no per-query cost. The catalog
vectors are precomputed once by ``tools/build_hf_embeddings.py``.

Model: ``all-MiniLM-L6-v2`` -- 22M parameters, 384 dimensions, ~80MB on disk,
Apache-2.0. Chosen because it embeds the whole 50k catalog in ~60s on a laptop
GPU and a single query in ~5ms, which keeps dense retrieval affordable on every
turn. Everything here degrades to "no dense results" rather than raising, so a
missing model or missing vectors leaves the sparse pipeline untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from starter.hybrid_retrieval import SemanticSearchResult
from starter.semantic_query_prompt import deterministic_query

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MATRIX = "artifacts/hf_catalog_embeddings.npy"
DEFAULT_IDS = "artifacts/hf_catalog_embedding_ids.json"


class LocalDenseRetriever:
    """Exact cosine search over precomputed MiniLM catalog vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        matrix_path: str = DEFAULT_MATRIX,
        ids_path: str = DEFAULT_IDS,
        top_k: int = 100,
        device: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name
        self.model = None
        self.matrix = None
        self.ids: tuple[str, ...] = ()
        self.available = False
        self.device = device
        self.unavailable_reason = "not initialised"
        self._np = None

        try:
            import numpy as np

            matrix_file = Path(matrix_path)
            ids_file = Path(ids_path)
            if not matrix_file.is_file() or not ids_file.is_file():
                self.unavailable_reason = (
                    "catalog vectors missing; run tools/build_hf_embeddings.py"
                )
                return

            self._np = np
            # Memory-mapped: the 50k x 384 float32 matrix is ~77MB and is only
            # ever read, so there is no reason to hold a private copy.
            self.matrix = np.load(matrix_file, mmap_mode="r")
            self.ids = tuple(json.loads(ids_file.read_text(encoding="utf-8")))
            if self.matrix.shape[0] != len(self.ids):
                self.unavailable_reason = (
                    f"vector/id mismatch: {self.matrix.shape[0]} vs {len(self.ids)}"
                )
                return

            import torch
            from sentence_transformers import SentenceTransformer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=self.device)
            self.available = True
            self.unavailable_reason = ""
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    def encode(self, text: str):
        """Return one normalized query vector, or None."""

        if not self.available or not text.strip():
            return None
        try:
            vector = self.model.encode(text, normalize_embeddings=True)
            return self._np.asarray(vector, dtype=self._np.float32)
        except Exception:
            return None

    def search(
        self,
        constraints: Sequence[Mapping[str, object]],
        *,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> SemanticSearchResult:
        """Rank catalog products against the query.

        ``query_text`` lets the caller supply an LLM-rewritten query; without
        it the deterministic constraint rendering is used, which is what keeps
        the route working when no language model is available.
        """

        text = query_text or deterministic_query(constraints)
        if not text:
            return SemanticSearchResult("", ())
        if not self.available:
            return SemanticSearchResult("", (), error=self.unavailable_reason)

        embedding = self.encode(text)
        if embedding is None:
            return SemanticSearchResult(text, (), error="query embedding failed")

        # Vectors are pre-normalized, so a dot product is cosine similarity.
        scores = self.matrix @ embedding
        count = min(max(0, top_k or self.top_k), len(self.ids))
        if count == 0:
            return SemanticSearchResult(text, ())

        np = self._np
        partition = np.argpartition(scores, -count)[-count:]
        ordered = partition[np.argsort(-scores[partition])]
        return SemanticSearchResult(
            query_text=text,
            ranked_ids=tuple(self.ids[int(index)] for index in ordered[:count]),
            prompt_tokens=0,
            completion_tokens=0,
        )

    def close(self) -> None:
        self.model = None
        self.matrix = None
