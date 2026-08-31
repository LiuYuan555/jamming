"""100% Offline HuggingFace Semantic Retriever."""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Sequence, Mapping
from sentence_transformers import SentenceTransformer

from starter.semantic_query_prompt import deterministic_query
from starter.hybrid_retrieval import SemanticSearchResult

class HFHybridRetriever:
    def __init__(self, top_k: int = 100) -> None:
        self.top_k = top_k
        self.model = None
        self.matrix = None
        self.ids = tuple()
        self.available = False
        
        try:
            # Load the precomputed local embeddings
            matrix_path = Path("artifacts/hf_catalog_embeddings.npy")
            ids_path = Path("artifacts/hf_catalog_embedding_ids.json")
            
            if matrix_path.exists() and ids_path.exists():
                self.matrix = np.load(matrix_path).astype(np.float32)
                with open(ids_path, "r", encoding="utf-8") as f:
                    self.ids = tuple(json.load(f))
                
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Local AI] Loading HuggingFace model on {device.upper()}...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self.available = True
            else:
                self.unavailable_reason = "HF embeddings not built yet. Run tools/build_hf_embeddings.py"
        except Exception as e:
            self.unavailable_reason = str(e)

    def search(
        self,
        constraints: Sequence[Mapping[str, object]],
        *,
        top_k: int | None = None,
    ) -> SemanticSearchResult:
        # Step 1: Use the deterministic math function to format the constraints perfectly
        query_text = deterministic_query(constraints)
        if not query_text:
            return SemanticSearchResult("", ())
            
        if not self.available:
            return SemanticSearchResult("", (), error=self.unavailable_reason)

        # Step 2: Embed the string offline
        try:
            embedding = self.model.encode(query_text, normalize_embeddings=True)
            embedding = np.asarray(embedding, dtype=np.float32)
        except Exception as exc:
            return SemanticSearchResult(query_text, (), error=f"HF embedding failed: {exc}")

        # Step 3: Fast local vector math (Cosine Similarity via Dot Product)
        scores = self.matrix @ embedding
        count = min(max(0, top_k or self.top_k), len(self.ids))
        
        if count == 0:
            return SemanticSearchResult(query_text, ())
            
        partition = np.argpartition(scores, -count)[-count:]
        indices = partition[np.argsort(-scores[partition])]
        ranked_ids = tuple(self.ids[int(index)] for index in indices[:count])
        
        return SemanticSearchResult(
            query_text=query_text,
            ranked_ids=ranked_ids,
            prompt_tokens=0,
            completion_tokens=0,
            error=None
        )

