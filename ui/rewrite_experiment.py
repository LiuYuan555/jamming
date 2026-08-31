"""Does LLM query rewriting help the group-repo agent?

Our separate TechJam build credits a Qwen rolling rewriter with +0.088. That
build feeds BM25 the raw conversation, so removing conversational grammar is
worth a lot. This repo does not: ``ConversationState.retrieval_text()`` already
returns de-duplicated typed constraint values, which are catalog strings the
simulator quoted verbatim. The open question is whether rewriting still helps
here, or whether it destroys the exact-phrase evidence the reranker depends on.

The experiment patches ``retrieval_text`` -- the single method producing the
query -- so the agent, reranker, evaluator and scoring are otherwise untouched.

    python -m ui.rewrite_experiment --limit 0            # full 200 sessions
    python -m ui.rewrite_experiment --baseline           # no rewriting

Results are only meaningful over all 200 sessions; the first 40 are the hardest
block and report a wildly pessimistic number on their own.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.conversation_state import ConversationState  # noqa: E402
from ui.server import find_catalog  # noqa: E402


SYSTEM = (
    "You turn a shopper's accumulated requirements into a compact product "
    "search query. Reply with the query only -- no preamble, no punctuation "
    "beyond spaces, no explanation."
)

INSTRUCTION = (
    "Rewrite the requirements below as a short keyword search query for a "
    "clothing catalog.\n"
    "Keep every distinctive product term exactly as written.\n"
    "Drop marketing sentences, gift lists and boilerplate.\n"
    "Return at most 20 words.\n\n"
    "Requirements:\n"
)

WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


class QueryRewriter:
    """Local Qwen rewriter with a guard and a per-query cache."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.cache: dict[str, str] = {}
        self.calls = 0
        self.hits = 0
        self.rejected = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, source: str) -> str:
        if not source.strip():
            return source
        if source in self.cache:
            self.hits += 1
            return self.cache[source]

        import torch

        self.calls += 1
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION + source},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = output[0][inputs["input_ids"].shape[1] :]
        self.prompt_tokens += int(inputs["input_ids"].shape[1])
        self.completion_tokens += int(completion.shape[0])
        generated = self.tokenizer.decode(completion, skip_special_tokens=True)
        result = self._guard(generated, source)
        self.cache[source] = result
        return result

    def _guard(self, generated: str, source: str) -> str:
        """Reject a rewrite that invents vocabulary, matching src/rewrite.py."""

        cleaned = " ".join(generated.replace("\n", " ").split()).strip(' "\'`')
        if not cleaned:
            self.rejected += 1
            return source
        invented = _words(cleaned) - _words(source)
        # A few connective words are harmless; wholesale new vocabulary is not.
        if len(invented) > 3:
            self.rejected += 1
            return source
        return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--baseline", action="store_true", help="no rewriting")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 200")
    args = parser.parse_args()

    catalog_path = find_catalog(args.catalog)
    samples = load_jsonl(Path(__file__).resolve().parent.parent / args.dataset)
    if args.limit:
        samples = samples[: args.limit]

    rewriter = None
    if not args.baseline:
        print(f"loading {args.model} ...", flush=True)
        started = time.perf_counter()
        rewriter = QueryRewriter(args.model)
        print(f"  ready on {rewriter.device} in {time.perf_counter()-started:.1f}s",
              flush=True)

        original = ConversationState.retrieval_text

        def patched(self) -> str:
            return rewriter(original(self))

        ConversationState.retrieval_text = patched

    print("indexing catalog ...", flush=True)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(str(catalog_path))

    started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    elapsed = time.perf_counter() - started

    label = "baseline (no rewrite)" if args.baseline else f"rewrite: {args.model}"
    print(f"\n=== {label} ===")
    print(f"sessions        {result['sample_count']}")
    print(f"HitRate@10      {result['hit_rate_at_10']:.6f}")
    print(f"MRR             {result['mrr']:.6f}")
    print(f"MTTC            {result['mttc']:.4f}")
    print(f"TechnicalScore  {result['recommended_technical_score']:.6f}")
    print(f"wall time       {elapsed:.1f}s")
    for name, scenario in sorted(result.get("scenario_metrics", {}).items()):
        print(f"  {name:16} hit {scenario['hit_rate_at_10']:.4f}  "
              f"mrr {scenario['mrr']:.4f}  mttc {scenario['mttc']:.3f}")
    if rewriter is not None:
        print(f"\nrewrites        {rewriter.calls} generated, {rewriter.hits} cached, "
              f"{rewriter.rejected} rejected by guard")
        print(f"tokens          {rewriter.prompt_tokens} prompt / "
              f"{rewriter.completion_tokens} completion")


if __name__ == "__main__":
    main()
