"""Score the LLM pipeline against the sparse baseline on the public set.

Each component is switched independently so the contribution of every stage is
attributable, rather than reporting one number for "we added AI".

    python tools/run_llm_ablation.py --config baseline
    python tools/run_llm_ablation.py --config dense
    python tools/run_llm_ablation.py --config full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIGS = {
    # name        LLM   extract rewrite reply  dense
    "baseline":  ("0", "0", "0", "0", "0"),
    "dense":     ("0", "0", "0", "0", "1"),
    "rewrite":   ("1", "0", "1", "0", "1"),
    "extract":   ("1", "1", "0", "0", "1"),
    "reply":     ("1", "0", "0", "1", "1"),
    "full":      ("1", "1", "1", "1", "1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="full", choices=sorted(CONFIGS))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    llm, extract, rewrite, reply, dense = CONFIGS[args.config]
    os.environ["TECHJAM_LLM"] = llm
    os.environ["TECHJAM_EXTRACT"] = extract
    os.environ["TECHJAM_REWRITE"] = rewrite
    os.environ["TECHJAM_REPLY"] = reply
    os.environ["TECHJAM_DENSE"] = dense

    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from starter.llm_agent import LLMAgent

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]

    print(f"config: {args.config}  (llm={llm} extract={extract} "
          f"rewrite={rewrite} reply={reply} dense={dense})", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.perf_counter()
    agent = LLMAgent(args.catalog)
    print(f"  agent built in {time.perf_counter()-started:.1f}s "
          f"(llm={agent.llm is not None}, dense={agent.dense is not None})", flush=True)

    started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    elapsed = time.perf_counter() - started

    print(f"\n=== {args.config} ===")
    print(f"sessions        {result['sample_count']}")
    print(f"HitRate@10      {result['hit_rate_at_10']:.6f}")
    print(f"MRR             {result['mrr']:.6f}")
    print(f"MTTC            {result['mttc']:.4f}")
    print(f"TechnicalScore  {result['recommended_technical_score']:.6f}")
    print(f"wall time       {elapsed:.1f}s")
    for name, scenario in sorted(result.get("scenario_metrics", {}).items()):
        print(f"  {name:16} hit {scenario['hit_rate_at_10']:.4f}  "
              f"mrr {scenario['mrr']:.4f}  mttc {scenario['mttc']:.3f}")
    if agent.llm is not None:
        print(f"\nLLM usage       {agent.llm.usage()}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
