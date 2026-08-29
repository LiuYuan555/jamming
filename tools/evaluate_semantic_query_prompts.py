"""Ablate semantic-query rewrite prompts against dense target retrieval.

The evaluation deliberately isolates the rewrite: every prompt sees the same
parsed constraints, uses the same ``text-embedding-3-small`` catalog index, and
is scored against the same public session states.  It does not tune sparse/dense
fusion weights, so a prompt cannot win merely because another retrieval setting
changed at the same time.

Examples:

    # Cheap smoke test (deterministic baseline plus four prompts, 24 states)
    python tools/evaluate_semantic_query_prompts.py --limit-states 24

    # Full public-state ablation, reusing the SQLite cache from the smoke test
    python tools/evaluate_semantic_query_prompts.py

    # Network-free report from cached rows
    python tools/evaluate_semantic_query_prompts.py --offline
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import threading
from typing import Mapping, Sequence

# Allow ``python tools/evaluate_semantic_query_prompts.py`` from the repository
# root without requiring callers to set PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.conversation_state import ConstraintSource, ConversationState
from starter.hybrid_retrieval import DenseCatalogIndex
from starter.semantic_query_prompt import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_MODEL,
    PROMPT_SUITE_VERSION,
    PROMPT_VARIANTS,
    build_rewrite_input,
    canonical_constraints,
    deterministic_query,
    get_prompt_variant,
    sanitize_query,
)


DETERMINISTIC_VARIANT = "deterministic"
HARD_SOURCES = {
    ConstraintSource.INITIAL_CATEGORY,
    ConstraintSource.INITIAL_REQUIREMENT,
    ConstraintSource.OVERRIDE,
}


@dataclass(frozen=True)
class StateRecord:
    sample_id: str
    scenario_type: str
    turn: int
    target: str
    constraints: tuple[dict[str, object], ...]

    @property
    def state_id(self) -> str:
        material = "\0".join((
            self.sample_id,
            str(self.turn),
            self.target,
            canonical_constraints(self.constraints),
        ))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class EvaluatedQuery:
    state_id: str
    variant: str
    query_text: str
    target_rank: int | None
    prompt_tokens: int
    completion_tokens: int
    used_fallback: bool
    error: str | None = None


class PromptCache:
    """Small thread-safe cache for expensive rewrite and embedding calls."""

    def __init__(self, path: str | Path, dimensions: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS prompt_embedding_cache ("
            "cache_key TEXT PRIMARY KEY, query_text TEXT NOT NULL, embedding BLOB NOT NULL, "
            "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, "
            "used_fallback INTEGER NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> tuple[str, object, int, int, bool] | None:
        import numpy as np

        with self.lock:
            row = self.connection.execute(
                "SELECT query_text, embedding, prompt_tokens, completion_tokens, used_fallback "
                "FROM prompt_embedding_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[1], dtype=np.float32)
        if vector.shape != (self.dimensions,):
            return None
        return str(row[0]), vector, int(row[2]), int(row[3]), bool(row[4])

    def put(
        self,
        key: str,
        query_text: str,
        embedding: object,
        prompt_tokens: int,
        completion_tokens: int,
        used_fallback: bool,
    ) -> object:
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"embedding has shape {vector.shape}; expected {(self.dimensions,)}"
            )
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO prompt_embedding_cache "
                "(cache_key, query_text, embedding, prompt_tokens, completion_tokens, used_fallback) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    query_text,
                    vector.tobytes(),
                    prompt_tokens,
                    completion_tokens,
                    int(used_fallback),
                ),
            )
            self.connection.commit()
        return vector

    def close(self) -> None:
        self.connection.close()


def public_states(
    samples: Sequence[dict],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict],
) -> list[StateRecord]:
    """Reproduce always-``other`` states without exposing hidden fields to prompts."""

    records: list[StateRecord] = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, dict(products))
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        state = ConversationState(sample["user_profile"])
        last_asked: str | None = None

        for turn in range(1, MAX_TURNS + 1):
            state.observe(message, turn, asked_attribute=last_asked)
            constraints = tuple(
                {
                    "attribute": record.attribute.value,
                    "value": record.value,
                    "hard": record.source in HARD_SOURCES,
                }
                for record in state.active_constraints()
            )
            canonical = canonical_constraints(constraints)
            dedup_key = (target, canonical)
            if constraints and dedup_key not in seen:
                seen.add(dedup_key)
                records.append(StateRecord(
                    sample_id=str(sample["sample_id"]),
                    scenario_type=str(sample["scenario_type"]),
                    turn=turn,
                    target=target,
                    constraints=constraints,
                ))
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                message, boundary_used = customer_reply(
                    effective,
                    "other",
                    disclosed,
                    boundary_used,
                )
            last_asked = "other"
    return records


def select_states(states: Sequence[StateRecord], limit: int) -> list[StateRecord]:
    """Select a deterministic, scenario-mixed subset for a cheap screening run."""

    ordered = sorted(
        states,
        key=lambda item: (
            hashlib.sha256(item.state_id.encode("ascii")).hexdigest(),
            item.state_id,
        ),
    )
    if limit <= 0 or limit >= len(ordered):
        return ordered
    grouped: dict[str, list[StateRecord]] = {}
    for state in ordered:
        grouped.setdefault(state.scenario_type, []).append(state)
    selected: list[StateRecord] = []
    names = sorted(grouped)
    cursor = 0
    while len(selected) < limit:
        name = names[cursor % len(names)]
        if grouped[name]:
            selected.append(grouped[name].pop(0))
        if not any(grouped.values()):
            break
        cursor += 1
    return selected


def _cache_key(
    variant: str,
    constraints: Sequence[Mapping[str, object]],
    query_model: str,
    embedding_model: str,
    dimensions: int,
) -> str:
    material = "\n".join((
        PROMPT_SUITE_VERSION,
        variant,
        query_model if variant != DETERMINISTIC_VARIANT else DETERMINISTIC_VARIANT,
        embedding_model,
        str(dimensions),
        canonical_constraints(constraints),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def evaluate_one(
    state: StateRecord,
    variant: str,
    *,
    index: DenseCatalogIndex,
    cache: PromptCache,
    client: object | None,
    query_model: str,
    embedding_model: str,
    dimensions: int,
    top_k: int,
    offline: bool,
) -> EvaluatedQuery:
    key = _cache_key(
        variant,
        state.constraints,
        query_model,
        embedding_model,
        dimensions,
    )
    cached = cache.get(key)
    if cached is not None:
        query_text, embedding, prompt_tokens, completion_tokens, used_fallback = cached
        ranked = index.search(embedding, top_k)
        rank = ranked.index(state.target) + 1 if state.target in ranked else None
        return EvaluatedQuery(
            state.state_id,
            variant,
            query_text,
            rank,
            prompt_tokens,
            completion_tokens,
            used_fallback,
        )

    if offline:
        return EvaluatedQuery(
            state.state_id,
            variant,
            "",
            None,
            0,
            0,
            True,
            "not cached (offline mode)",
        )
    if client is None:
        return EvaluatedQuery(
            state.state_id,
            variant,
            "",
            None,
            0,
            0,
            True,
            "OpenAI client unavailable",
        )

    fallback = deterministic_query(state.constraints)
    query_text = fallback
    prompt_tokens = 0
    completion_tokens = 0
    used_fallback = variant == DETERMINISTIC_VARIANT
    errors: list[str] = []

    if variant != DETERMINISTIC_VARIANT:
        prompt = get_prompt_variant(variant)
        try:
            response = client.responses.create(
                model=query_model,
                instructions=prompt.instructions,
                input=build_rewrite_input(state.constraints),
                temperature=0,
                max_output_tokens=100,
            )
            raw = str(response.output_text or "")
            query_text = sanitize_query(raw, state.constraints)
            used_fallback = not raw.strip() or query_text == fallback
            usage = getattr(response, "usage", None)
            prompt_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        except Exception as exc:
            used_fallback = True
            errors.append(f"rewrite failed: {type(exc).__name__}: {exc}")

    try:
        response = client.embeddings.create(
            model=embedding_model,
            input=[query_text],
            dimensions=dimensions,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        usage = getattr(response, "usage", None)
        prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        embedding = cache.put(
            key,
            query_text,
            embedding,
            prompt_tokens,
            completion_tokens,
            used_fallback,
        )
    except Exception as exc:
        errors.append(f"embedding failed: {type(exc).__name__}: {exc}")
        return EvaluatedQuery(
            state.state_id,
            variant,
            query_text,
            None,
            prompt_tokens,
            completion_tokens,
            used_fallback,
            "; ".join(errors),
        )

    ranked = index.search(embedding, top_k)
    rank = ranked.index(state.target) + 1 if state.target in ranked else None
    return EvaluatedQuery(
        state.state_id,
        variant,
        query_text,
        rank,
        prompt_tokens,
        completion_tokens,
        used_fallback,
        "; ".join(errors) or None,
    )


def summarize(
    results: Sequence[EvaluatedQuery],
    variants: Sequence[str],
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        items = [item for item in results if item.variant == variant]
        valid = [item for item in items if item.error is None]
        ranks = [item.target_rank for item in valid]
        count = len(valid)
        recall = {
            cutoff: (
                sum(rank is not None and rank <= cutoff for rank in ranks) / count
                if count else 0.0
            )
            for cutoff in (1, 10, 50, 100)
            if cutoff <= top_k
        }
        mrr = (
            statistics.fmean(0.0 if rank is None else 1.0 / rank for rank in ranks)
            if count else 0.0
        )
        recall_at_limit = recall.get(top_k)
        if recall_at_limit is None:
            recall_at_limit = (
                sum(rank is not None for rank in ranks) / count if count else 0.0
            )
        rows.append({
            "variant": variant,
            "state_count": len(items),
            "valid_state_count": count,
            "error_count": len(items) - count,
            "fallback_rate": (
                sum(item.used_fallback for item in valid) / count if count else 0.0
            ),
            "mean_query_words": (
                statistics.fmean(len(item.query_text.split()) for item in valid)
                if count else 0.0
            ),
            "recall": {f"at_{key}": round(value, 6) for key, value in recall.items()},
            "mrr_at_top_k": round(mrr, 6),
            "selection_score": round(0.7 * recall_at_limit + 0.3 * mrr, 6),
            "prompt_tokens": sum(item.prompt_tokens for item in valid),
            "completion_tokens": sum(item.completion_tokens for item in valid),
        })
    rows.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -int(row["valid_state_count"]),
            str(row["variant"]),
        )
    )
    return rows


def paired_comparison(
    results: Sequence[EvaluatedQuery],
    top_k: int,
) -> dict[str, dict[str, int | float]]:
    baseline = {
        item.state_id: item.target_rank
        for item in results
        if item.variant == DETERMINISTIC_VARIANT and item.error is None
    }
    comparisons: dict[str, dict[str, int | float]] = {}
    variants = sorted({item.variant for item in results if item.variant != DETERMINISTIC_VARIANT})
    for variant in variants:
        improved = unchanged = worsened = 0
        delta_sum = 0.0
        compared = 0
        for item in results:
            if item.variant != variant or item.error is not None or item.state_id not in baseline:
                continue
            base_rank = baseline[item.state_id]
            base_value = top_k + 1 if base_rank is None else base_rank
            value = top_k + 1 if item.target_rank is None else item.target_rank
            compared += 1
            delta_sum += base_value - value
            if value < base_value:
                improved += 1
            elif value > base_value:
                worsened += 1
            else:
                unchanged += 1
        comparisons[variant] = {
            "compared": compared,
            "improved": improved,
            "unchanged": unchanged,
            "worsened": worsened,
            "mean_rank_gain_capped": round(delta_sum / compared, 6) if compared else 0.0,
        }
    return comparisons


def failure_cases(
    states: Sequence[StateRecord],
    results: Sequence[EvaluatedQuery],
    best_variant: str,
    top_k: int,
    limit: int = 12,
) -> list[dict[str, object]]:
    by_id = {state.state_id: state for state in states}
    baseline = {
        item.state_id: item for item in results
        if item.variant == DETERMINISTIC_VARIANT and item.error is None
    }
    candidates = [
        item for item in results
        if item.variant == best_variant and item.error is None
    ]
    candidates.sort(key=lambda item: (
        -(top_k + 1 if item.target_rank is None else item.target_rank),
        item.state_id,
    ))
    cases: list[dict[str, object]] = []
    for item in candidates:
        state = by_id[item.state_id]
        base = baseline.get(item.state_id)
        if item.target_rank is not None and not (
            base is not None
            and base.target_rank is not None
            and item.target_rank > base.target_rank
        ):
            continue
        cases.append({
            "sample_id": state.sample_id,
            "scenario_type": state.scenario_type,
            "turn": state.turn,
            "target": state.target,
            "constraints": list(state.constraints),
            "query": item.query_text,
            "target_rank": item.target_rank,
            "deterministic_rank": None if base is None else base.target_rank,
        })
        if len(cases) >= limit:
            break
    return cases


def parse_variants(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one prompt variant is required")
    available = {DETERMINISTIC_VARIANT, *(item.name for item in PROMPT_VARIANTS)}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variants {unknown}; choose from {sorted(available)}"
        )
    return list(dict.fromkeys(names))


def main() -> None:
    default_variants = ",".join([
        DETERMINISTIC_VARIANT,
        *(variant.name for variant in PROMPT_VARIANTS),
    ])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--embedding-path", default="artifacts/catalog_embeddings.npy")
    parser.add_argument("--ids-path", default="artifacts/catalog_embedding_ids.json")
    parser.add_argument(
        "--cache-path",
        default="artifacts/semantic_query_prompt_cache.sqlite",
    )
    parser.add_argument(
        "--output",
        default="artifacts/semantic_query_prompt_ablation.json",
    )
    parser.add_argument("--query-model", default=DEFAULT_QUERY_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--limit-states",
        type=int,
        default=0,
        help="deterministic scenario-stratified subset; 0 evaluates all states",
    )
    parser.add_argument("--variants", type=parse_variants, default=parse_variants(default_variants))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never call OpenAI; report only rows already present in the prompt cache",
    )
    args = parser.parse_args()

    if args.dimensions < 1 or args.top_k < 1 or args.workers < 1:
        raise SystemExit("dimensions, top-k, and workers must be positive")
    if not args.offline and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not configured. Set it, or pass --offline to use cached rows."
        )

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    states = select_states(public_states(samples, categories, products), args.limit_states)
    index = DenseCatalogIndex(args.embedding_path, args.ids_path, args.dimensions)
    cache = PromptCache(args.cache_path, args.dimensions)
    client: object | None = None
    if not args.offline:
        try:
            from openai import OpenAI

            client = OpenAI(timeout=30.0, max_retries=3)
        except Exception as exc:
            cache.close()
            raise SystemExit(f"OpenAI client unavailable: {exc}") from exc

    jobs = [(state, variant) for state in states for variant in args.variants]
    print(
        f"Evaluating {len(states)} public states x {len(args.variants)} variants "
        f"({len(jobs)} state-query pairs)",
        flush=True,
    )
    results: list[EvaluatedQuery] = []
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    evaluate_one,
                    state,
                    variant,
                    index=index,
                    cache=cache,
                    client=client,
                    query_model=args.query_model,
                    embedding_model=args.embedding_model,
                    dimensions=args.dimensions,
                    top_k=args.top_k,
                    offline=args.offline,
                )
                for state, variant in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 100 == 0 or completed == len(jobs):
                    print(f"Completed {completed}/{len(jobs)}", flush=True)
    finally:
        cache.close()

    summaries = summarize(results, args.variants, args.top_k)
    complete_summaries = [
        row for row in summaries
        if int(row["valid_state_count"]) == len(states)
    ]
    best_variant = str(
        (complete_summaries or summaries or [{"variant": DETERMINISTIC_VARIANT}])[0]["variant"]
    )
    report = {
        "configuration": {
            "prompt_suite_version": PROMPT_SUITE_VERSION,
            "query_model": args.query_model,
            "embedding_model": args.embedding_model,
            "dimensions": args.dimensions,
            "top_k": args.top_k,
            "offline": args.offline,
            "state_count": len(states),
            "variants": args.variants,
            "selection_rule": "0.7 * dense_recall@top_k + 0.3 * dense_MRR@top_k",
        },
        "best_variant": best_variant,
        "summary": summaries,
        "paired_vs_deterministic": paired_comparison(results, args.top_k),
        "failure_cases": failure_cases(states, results, best_variant, args.top_k),
        "errors": [
            {
                "state_id": item.state_id,
                "variant": item.variant,
                "error": item.error,
            }
            for item in results if item.error is not None
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failure_cases"}, indent=2))
    print(f"Wrote detailed report to {output}")


if __name__ == "__main__":
    main()
