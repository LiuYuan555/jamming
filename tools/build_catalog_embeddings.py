"""Precompute normalized OpenAI embeddings for the frozen product catalog.

The builder is resumable.  It writes a NumPy ``.npy`` matrix through a memory
map and records the number of completed rows after every successful API batch.
Generated artifacts live under ``artifacts/`` and are ignored by git.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from starter.hybrid_retrieval import catalog_embedding_text


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_catalog(path: Path, max_chars: int) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            ids.append(str(product["parent_asin"]))
            texts.append(catalog_embedding_text(product, max_chars=max_chars))
    return ids, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen catalog's dense embedding index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2400)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured")
    if args.dimensions < 1 or args.batch_size < 1 or args.max_chars < 1 or args.workers < 1:
        raise SystemExit("dimensions, batch-size, max-chars, and workers must be positive")

    try:
        import numpy as np
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install requirements-hybrid.txt before building embeddings") from exc

    catalog_path = Path(args.catalog)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "catalog_embeddings.npy"
    ids_path = output_dir / "catalog_embedding_ids.json"
    metadata_path = output_dir / "catalog_embeddings.meta.json"

    print(f"Loading catalog text from {catalog_path} ...", flush=True)
    ids, texts = load_catalog(catalog_path, args.max_chars)
    catalog_signature = {
        "catalog_path": str(catalog_path.resolve()),
        "catalog_bytes": catalog_path.stat().st_size,
        "rows": len(ids),
        "model": args.model,
        "dimensions": args.dimensions,
        "max_chars": args.max_chars,
    }

    completed = 0
    total_prompt_tokens = 0
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_signature = {key: metadata.get(key) for key in catalog_signature}
        if previous_signature != catalog_signature:
            raise SystemExit(
                "Existing embedding artifacts use a different catalog/model configuration. "
                "Move the existing artifacts aside before rebuilding."
            )
        completed = int(metadata.get("completed_rows", 0))
        total_prompt_tokens = int(metadata.get("prompt_tokens", 0))
        if not matrix_path.is_file() or not ids_path.is_file():
            raise SystemExit("Embedding metadata exists but its matrix/ID artifacts are missing")
        stored_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if stored_ids != ids:
            raise SystemExit("Catalog IDs changed since this embedding build started")
        matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
        if matrix.shape != (len(ids), args.dimensions):
            raise SystemExit(f"Existing matrix has unexpected shape {matrix.shape}")
    else:
        matrix = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(ids), args.dimensions),
        )
        _write_json(ids_path, ids)
        _write_json(metadata_path, {
            **catalog_signature,
            "completed_rows": 0,
            "prompt_tokens": 0,
            "complete": False,
        })

    if completed >= len(ids):
        print(f"Embedding index is already complete: {matrix.shape} at {matrix_path}")
        return

    client = OpenAI(timeout=120.0, max_retries=5)

    def embed_range(start: int, end: int) -> tuple[int, int, object, int]:
        response = client.embeddings.create(
            model=args.model,
            input=texts[start:end],
            dimensions=args.dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        batch = np.asarray([item.embedding for item in ordered], dtype=np.float32)
        if batch.shape != (end - start, args.dimensions):
            raise RuntimeError(f"API returned unexpected embedding shape {batch.shape}")
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch = np.divide(batch, norms, out=np.zeros_like(batch), where=norms != 0)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return start, end, batch, tokens

    cursor = completed
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while cursor < len(ids):
            ranges: list[tuple[int, int]] = []
            for _ in range(args.workers):
                if cursor >= len(ids):
                    break
                end = min(cursor + args.batch_size, len(ids))
                ranges.append((cursor, end))
                cursor = end
            futures = [executor.submit(embed_range, start, end) for start, end in ranges]
            # A checkpoint advances only after the complete wave succeeds. An
            # interrupted wave may repeat a few API calls but can never leave a
            # falsely marked partial matrix.
            results = [future.result() for future in futures]
            for start, end, batch, tokens in results:
                matrix[start:end] = batch
                total_prompt_tokens += tokens
            matrix.flush()
            completed = results[-1][1]
            _write_json(metadata_path, {
                **catalog_signature,
                "completed_rows": completed,
                "prompt_tokens": total_prompt_tokens,
                "complete": completed == len(ids),
            })
            print(
                f"Embedded {completed:,}/{len(ids):,} products "
                f"({100.0 * completed / len(ids):.1f}%, {total_prompt_tokens:,} input tokens)",
                flush=True,
            )

    print(f"Wrote {matrix.shape} normalized embeddings to {matrix_path}")


if __name__ == "__main__":
    main()
