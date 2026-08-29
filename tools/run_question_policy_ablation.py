"""Compare always-other with observable information-gain questioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent, AgentConfig
from starter.information_gain_policy import (
    InformationGainAgent,
    InformationGainConfig,
)


def _run(agent: Agent, samples: list[dict], catalog: tuple) -> dict:
    try:
        return evaluate(agent, samples, *catalog)
    finally:
        agent.close()


def _summary(result: dict) -> dict:
    keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "scenario_metrics",
    )
    return {key: result[key] for key in keys}


def _paired_changes(baseline: dict, experiment: dict) -> dict:
    before = {item["sample_id"]: item for item in baseline["sessions"]}
    after = {item["sample_id"]: item for item in experiment["sessions"]}
    improved: list[str] = []
    regressed: list[str] = []
    unchanged = 0
    for sample_id in sorted(before.keys() & after.keys()):
        left = (
            before[sample_id]["hit"],
            before[sample_id]["reciprocal_rank"],
            -(before[sample_id]["first_hit_turn"] or 11),
        )
        right = (
            after[sample_id]["hit"],
            after[sample_id]["reciprocal_rank"],
            -(after[sample_id]["first_hit_turn"] or 11),
        )
        if right > left:
            improved.append(sample_id)
        elif right < left:
            regressed.append(sample_id)
        else:
            unchanged += 1
    return {
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": unchanged,
        "improved_sample_ids": improved,
        "regressed_sample_ids": regressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate fixed always-other vs catalog information gain"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--output", default="artifacts/question_policy_ablation.json"
    )
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--allow-repeated", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog = catalog_index(args.catalog)
    common = AgentConfig(
        question_policy="always_other",
        use_typed_state=True,
        use_reranker=True,
        candidate_pool_size=args.candidate_pool_size,
    )

    baseline = _run(Agent(args.catalog, config=common), samples, catalog)
    policy_config = InformationGainConfig(
        candidate_limit=args.candidate_limit,
        mask_repeated=not args.allow_repeated,
    )
    information_gain_agent = InformationGainAgent(
        args.catalog,
        config=common,
        policy_config=policy_config,
    )
    information_gain = evaluate(information_gain_agent, samples, *catalog)
    policy_diagnostics = information_gain_agent.information_gain_policy.diagnostics()
    information_gain_agent.close()

    metric_keys = ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")
    result = {
        "baseline": baseline,
        "information_gain": information_gain,
        "comparison": {
            "baseline_summary": _summary(baseline),
            "information_gain_summary": _summary(information_gain),
            "delta_information_gain_minus_baseline": {
                key: round(information_gain[key] - baseline[key], 6)
                for key in metric_keys
            },
            "paired_changes": _paired_changes(baseline, information_gain),
        },
        "information_gain_policy": policy_diagnostics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        **result["comparison"],
        "information_gain_policy": policy_diagnostics,
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()

