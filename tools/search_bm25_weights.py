"""Tune SQLite FTS5 BM25 catalog-field weights with successive halving.

The search evaluates a reproducible randomized grid on progressively larger
stratified subsets. Selection uses only the training split; the holdout is
reported once at the end as an overfitting check.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig, BM25FieldWeights


FIELDS = ("title", "categories", "features", "details", "store", "description")
BASELINE = BM25FieldWeights()


def stratified_split(
    samples: list[dict], *, holdout_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(str(sample["scenario_type"]), []).append(sample)
    train: list[dict] = []
    holdout: list[dict] = []
    for name in sorted(groups):
        group = list(groups[name])
        rng.shuffle(group)
        count = max(1, round(len(group) * holdout_fraction))
        holdout.extend(group[:count])
        train.extend(group[count:])
    train.sort(key=lambda item: str(item["sample_id"]))
    holdout.sort(key=lambda item: str(item["sample_id"]))
    return train, holdout


def stratified_prefix(samples: list[dict], size: int, seed: int) -> list[dict]:
    """Return a deterministic proportionally stratified subset."""

    if size >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(str(sample["scenario_type"]), []).append(sample)
    for group in groups.values():
        rng.shuffle(group)
    total = len(samples)
    exact = {name: size * len(group) / total for name, group in groups.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = size - sum(quotas.values())
    for name in sorted(groups, key=lambda key: (exact[key] - quotas[key], key), reverse=True):
        if remaining == 0:
            break
        quotas[name] += 1
        remaining -= 1
    selected = [sample for name in sorted(groups) for sample in groups[name][: quotas[name]]]
    return selected


def _key(weights: BM25FieldWeights) -> tuple[float, ...]:
    return tuple(round(value, 8) for value in weights.as_tuple())


def candidate_grid(count: int, seed: int) -> list[BM25FieldWeights]:
    """Build a broad but bounded grid containing anchors and joint samples."""

    if count < 4:
        raise ValueError("candidate count must be at least four")
    rng = random.Random(seed)
    baseline = BASELINE.as_tuple()
    candidates: list[BM25FieldWeights] = []
    seen: set[tuple[float, ...]] = set()

    def add(values: Iterable[float]) -> None:
        weights = BM25FieldWeights(**dict(zip(FIELDS, values)))
        if _key(weights) not in seen:
            seen.add(_key(weights))
            candidates.append(weights)

    add(baseline)
    add((1.0,) * len(FIELDS))
    add((3.0,) * len(FIELDS))
    add((6.0,) * len(FIELDS))
    if len(candidates) >= count:
        return candidates[:count]

    # One-at-a-time perturbations make field effects interpretable.
    for field_index in range(len(FIELDS)):
        for multiplier in (0.5, 2.0):
            values = list(baseline)
            values[field_index] *= multiplier
            add(values)
            if len(candidates) >= count:
                return candidates

    # Joint randomized grid points capture interactions without 5^6 trials.
    log_steps = (-2, -1, 0, 1, 2)
    global_steps = (-1, 0, 1)
    while len(candidates) < count:
        scale = 2.0 ** rng.choice(global_steps)
        values = [
            max(0.0625, min(24.0, value * scale * (2.0 ** rng.choice(log_steps))))
            for value in baseline
        ]
        add(values)
    return candidates


def compact_metrics(result: dict) -> dict:
    return {
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart BM25 field-weight grid search")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/bm25_weight_search.json")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument(
        "--rung-sizes",
        default="40,100,all",
        help="comma-separated nested training sizes; final value may be all",
    )
    parser.add_argument(
        "--keep",
        default="10,4,1",
        help="candidate counts retained after each rung",
    )
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 0.5:
        raise SystemExit("--holdout-fraction must be between zero and 0.5")

    samples = load_jsonl(args.dataset)
    train, holdout = stratified_split(
        samples, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    sizes = [
        len(train) if value.strip() == "all" else int(value)
        for value in args.rung_sizes.split(",")
    ]
    keeps = [int(value) for value in args.keep.split(",")]
    if len(sizes) != len(keeps) or not sizes or sizes != sorted(sizes):
        raise SystemExit("--rung-sizes and --keep need equal lengths and ascending sizes")
    if any(size < 1 or size > len(train) for size in sizes):
        raise SystemExit(f"rung sizes must be between 1 and {len(train)}")
    if any(keep < 1 for keep in keeps):
        raise SystemExit("keep counts must be positive")

    catalog_ids, categories, products = catalog_index(args.catalog)
    base_config = AgentConfig()
    agent = Agent(args.catalog, config=base_config)
    candidates = candidate_grid(args.candidates, args.seed)
    history: list[dict] = []

    def run(weights: BM25FieldWeights, subset: list[dict], label: str) -> dict:
        agent.config = replace(base_config, bm25_weights=weights)
        agent._sessions.clear()
        result = evaluate(agent, subset, catalog_ids, categories, products)
        record = {
            "stage": label,
            "weights": weights.as_dict(),
            **compact_metrics(result),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        return record

    try:
        active = candidates
        train_rankings: list[dict] = []
        for rung, (size, keep) in enumerate(zip(sizes, keeps), start=1):
            subset = stratified_prefix(train, size, args.seed)
            train_rankings = [run(weights, subset, f"train_rung_{rung}") for weights in active]
            train_rankings.sort(
                key=lambda item: (
                    item["technical_score"],
                    item["mrr"],
                    item["hit_rate_at_10"],
                    -sum(item["weights"].values()),
                ),
                reverse=True,
            )
            active = [
                BM25FieldWeights(**item["weights"])
                for item in train_rankings[: min(keep, len(train_rankings))]
            ]

        winner = active[0]
        finalists = list(active)
        baseline_holdout = run(BASELINE, holdout, "holdout_baseline")
        finalist_holdouts = [run(weights, holdout, "holdout_finalist") for weights in finalists]
        full_winner = run(winner, samples, "full_set_winner")
    finally:
        agent.close()

    payload = {
        "method": {
            "name": "randomized_grid_successive_halving",
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "candidate_count": len(candidates),
            "rung_sizes": sizes,
            "keep": keeps,
            "selection_metric": "technical_score_then_mrr",
            "holdout_used_for_selection": False,
        },
        "split": {
            "train_count": len(train),
            "holdout_count": len(holdout),
            "train_sample_ids": [item["sample_id"] for item in train],
            "holdout_sample_ids": [item["sample_id"] for item in holdout],
        },
        "baseline_weights": BASELINE.as_dict(),
        "selected_weights": winner.as_dict(),
        "holdout_baseline": baseline_holdout,
        "holdout_finalists": finalist_holdouts,
        "full_set_selected": full_winner,
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Best training-selected configuration:")
    print(json.dumps({"weights": winner.as_dict(), "full_set": full_winner}, indent=2))


if __name__ == "__main__":
    main()
