"""Build local HuggingFace embeddings for the catalog."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from starter.hybrid_retrieval import catalog_embedding_text

import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading HuggingFace model (all-MiniLM-L6-v2) on {device.upper()}...")
    # This model is tiny, fast, and generates 384-dimensional vectors
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    catalog_path = Path("data/catalog.jsonl")
    ids = []
    texts = []
    
    print("Reading catalog...")
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = product["parent_asin"]
            # Re-use the exact same formatting as before!
            text = catalog_embedding_text(product)
            
            ids.append(parent_asin)
            texts.append(text)
            
    print(f"Embedding {len(texts)} products... (this will run on CPU/GPU locally)")
    # Encode all texts into a numpy array of shape (N, 384)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Save the files
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    np.save(artifacts_dir / "hf_catalog_embeddings.npy", embeddings)
    with open(artifacts_dir / "hf_catalog_embedding_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
        
    print("Done! Saved fully offline embeddings to artifacts/.")

if __name__ == "__main__":
    main()

