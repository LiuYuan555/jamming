"""Generate leakage-safe conversations and train the tiny intent classifier.

The default target is the intent observable up to a turn. Hidden simulator
scenario is retained only as evaluation metadata. Products, not turns, are the
unit of train/validation/test splitting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    behavior_for,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.intent_classifier import (
    INTENTS,
    IntentClassifier,
    IntentVectorizer,
    observable_evidence,
    observable_intent,
)


SCENARIOS = ("buying", "browsing", "intent_override", "boundary")
ACTION_CYCLE = ("other", "feature", "material", "color", "size", "style")


@dataclass(frozen=True)
class IntentExample:
    messages: tuple[str, ...]
    asked_attributes: tuple[str, ...]
    observable_label: str
    hidden_scenario: str
    group_id: str
    split: str
    sample_id: str
    turn: int
    event_observable: bool


def stable_split(parent_asin: str, seed: str = "intent-v1") -> str:
    digest = hashlib.blake2b(
        f"{seed}\0{parent_asin}".encode("utf-8"), digest_size=8
    ).digest()
    fraction = int.from_bytes(digest, "big") / float(2**64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def group_hash(parent_asin: str, seed: str = "intent-v1") -> str:
    return hashlib.sha256(
        f"{seed}\0group\0{parent_asin}".encode("utf-8")
    ).hexdigest()[:20]


def _trajectory_examples(
    sample: Mapping[str, object],
    *,
    category: str,
    split: str,
    group_id: str,
    max_turns: int,
) -> list[IntentExample]:
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    messages = [initial_message(dict(sample), category, disclosed)]
    asked: list[str] = []
    result: list[IntentExample] = []
    scenario = str(sample["scenario_type"])
    sample_id = str(sample["sample_id"])
    rotation = int(hashlib.sha256(sample_id.encode()).hexdigest()[:4], 16)

    for turn in range(1, max_turns + 1):
        evidence = observable_evidence(messages)
        result.append(IntentExample(
            messages=tuple(messages),
            asked_attributes=tuple(asked),
            observable_label=observable_intent(messages),
            hidden_scenario=scenario,
            group_id=group_id,
            split=split,
            sample_id=sample_id,
            turn=turn,
            event_observable=bool(evidence["override"] or evidence["boundary"]),
        ))
        if turn == max_turns:
            break
        action = ACTION_CYCLE[(rotation + turn - 1) % len(ACTION_CYCLE)]
        override = dict(sample.get("behavior", {})).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            reply = str(
                override.get(
                    "message", "Actually, please ignore my earlier preference."
                )
            )
        else:
            reply, boundary_used = customer_reply(
                dict(sample), action, disclosed, boundary_used
            )
        asked.append(action)
        messages.append(reply)
    return result


def generate_catalog_examples(
    products: Mapping[str, dict],
    categories: Mapping[str, list[str]],
    *,
    product_count: int,
    seed: int,
    excluded_parent_asins: Iterable[str] = (),
    max_turns: int = 4,
) -> list[IntentExample]:
    """Generate all four scenarios for each sampled non-public product."""

    excluded = {str(value) for value in excluded_parent_asins}
    candidates = sorted(set(products) - excluded)
    if product_count < 1 or product_count > len(candidates):
        raise ValueError("product_count is outside the available catalog range")
    selected = random.Random(seed).sample(candidates, product_count)
    examples: list[IntentExample] = []
    for product_index, parent_asin in enumerate(selected):
        card = intent_card(products[parent_asin])
        split = stable_split(parent_asin)
        identifier = group_hash(parent_asin)
        for scenario in SCENARIOS:
            sample_id = f"intent_synthetic_{product_index:06d}_{scenario}"
            sample = {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "intent_card": copy.deepcopy(card),
                "behavior": behavior_for(
                    scenario, card, random.Random(f"{sample_id}\0{scenario}")
                ),
            }
            examples.extend(_trajectory_examples(
                sample,
                category=coarse_category(categories.get(parent_asin, [])),
                split=split,
                group_id=identifier,
                max_turns=max_turns,
            ))
    return examples


def generate_public_examples(
    samples: Sequence[dict],
    products: Mapping[str, dict],
    categories: Mapping[str, list[str]],
    *,
    max_turns: int = 4,
) -> list[IntentExample]:
    examples: list[IntentExample] = []
    for sample in samples:
        parent_asin = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, dict(products))
        effective = {**sample, "intent_card": card, "behavior": behavior}
        examples.extend(_trajectory_examples(
            effective,
            category=coarse_category(categories.get(parent_asin, [])),
            split="test",
            group_id=group_hash(parent_asin),
            max_turns=max_turns,
        ))
    return examples


def examples_to_arrays(
    examples: Sequence[IntentExample], vectorizer: IntentVectorizer
) -> tuple[np.ndarray, np.ndarray]:
    features = vectorizer.matrix(
        [row.messages for row in examples],
        [row.asked_attributes for row in examples],
    )
    labels = np.asarray(
        [INTENTS.index(row.observable_label) for row in examples], dtype=np.int64
    )
    return features, labels


def fit_softmax(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    epochs: int = 180,
    learning_rate: float = 0.04,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    patience: int = 25,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit weighted multinomial logistic regression with Adam."""

    if len(train_x) == 0:
        raise ValueError("training split is empty")
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(train_y, minlength=len(INTENTS)).astype(np.float64)
    class_weights = len(train_y) / (len(INTENTS) * np.maximum(class_counts, 1.0))
    weights = rng.normal(0.0, 0.01, (len(INTENTS), train_x.shape[1])).astype(np.float32)
    bias = np.zeros(len(INTENTS), dtype=np.float32)
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(bias)
    vb = np.zeros_like(bias)
    step = 0
    best_loss = float("inf")
    best = (weights.copy(), bias.copy())
    bad_epochs = 0

    def loss_for(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
        if len(x) == 0:
            return float("nan")
        logits = x @ w.T + b
        logits -= logits.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(logits).sum(axis=1))
        return float(np.mean(logsum - logits[np.arange(len(y)), y]))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_x[indices]
            y = train_y[indices]
            logits = x @ weights.T + bias
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            probabilities[np.arange(len(y)), y] -= 1.0
            sample_weights = class_weights[y].astype(np.float32)
            probabilities *= sample_weights[:, None]
            probabilities /= max(float(sample_weights.sum()), 1e-8)
            gradient_w = probabilities.T @ x + weight_decay * weights
            gradient_b = probabilities.sum(axis=0)
            step += 1
            beta1, beta2 = 0.9, 0.999
            mw = beta1 * mw + (1 - beta1) * gradient_w
            vw = beta2 * vw + (1 - beta2) * (gradient_w * gradient_w)
            mb = beta1 * mb + (1 - beta1) * gradient_b
            vb = beta2 * vb + (1 - beta2) * (gradient_b * gradient_b)
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            weights -= learning_rate * (mw / correction1) / (
                np.sqrt(vw / correction2) + 1e-8
            )
            bias -= learning_rate * (mb / correction1) / (
                np.sqrt(vb / correction2) + 1e-8
            )

        monitored = loss_for(
            validation_x if len(validation_x) else train_x,
            validation_y if len(validation_y) else train_y,
            weights,
            bias,
        )
        if monitored < best_loss - 1e-5:
            best_loss = monitored
            best = (weights.copy(), bias.copy())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    return best[0], best[1], {
        "epochs_run": epoch,
        "class_counts": {
            intent: int(class_counts[index]) for index, intent in enumerate(INTENTS)
        },
        "validation_nll": best_loss,
    }


def tune_temperature(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    if not len(logits):
        return 1.0, float("nan")
    best = (1.0, float("inf"))
    for temperature in np.geomspace(0.35, 3.0, 48):
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(scaled).sum(axis=1))
        nll = float(np.mean(logsum - scaled[np.arange(len(labels)), labels]))
        if nll < best[1]:
            best = (float(temperature), nll)
    return best


def serialize_examples(examples: Sequence[IntentExample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            record = {
                "model_input": {
                    "messages": list(row.messages),
                    "asked_attributes": list(row.asked_attributes),
                },
                "label": {
                    "observable_intent": row.observable_label,
                    "hidden_scenario": row.hidden_scenario,
                },
                "metadata": {
                    "product_group_id": row.group_id,
                    "split": row.split,
                    "sample_id": row.sample_id,
                    "turn": row.turn,
                    "event_observable": row.event_observable,
                },
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--public-set", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--products", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, categories, products = catalog_index(args.catalog)
    public_samples = load_jsonl(args.public_set)
    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public_samples
    }
    examples = generate_catalog_examples(
        products,
        categories,
        product_count=args.products,
        seed=args.seed,
        excluded_parent_asins=public_targets,
        max_turns=args.max_turns,
    )
    vectorizer = IntentVectorizer()
    splits = {
        name: [row for row in examples if row.split == name]
        for name in ("train", "validation", "test")
    }
    train_x, train_y = examples_to_arrays(splits["train"], vectorizer)
    validation_x, validation_y = examples_to_arrays(splits["validation"], vectorizer)
    weights, bias, fit_report = fit_softmax(
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=args.epochs,
        seed=args.seed,
    )
    temperature, calibrated_nll = tune_temperature(
        validation_x @ weights.T + bias, validation_y
    )
    model = IntentClassifier(
        weights,
        bias,
        hash_dimension=vectorizer.hash_dimension,
        temperature=temperature,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    if args.dataset_output:
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        serialize_examples(examples, args.dataset_output)

    report = {
        "model": str(args.output),
        "target_definition": "intent_observable_up_to_current_turn",
        "parameter_count": model.parameter_count,
        "artifact_bytes": args.output.stat().st_size,
        "hash_dimension": vectorizer.hash_dimension,
        "synthetic_products": args.products,
        "examples": len(examples),
        "groups_by_split": {
            name: len({row.group_id for row in rows}) for name, rows in splits.items()
        },
        "examples_by_split": {name: len(rows) for name, rows in splits.items()},
        "observable_labels": dict(Counter(row.observable_label for row in examples)),
        "hidden_scenarios": dict(Counter(row.hidden_scenario for row in examples)),
        "temperature": temperature,
        "calibrated_validation_nll": calibrated_nll,
        "fit": fit_report,
        "leakage_controls": [
            "public target parent_asins excluded from synthetic training",
            "product-grouped split before turn expansion",
            "parent_asin and hidden product fields excluded from model_input",
            "hidden scenario retained only for diagnostic evaluation",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
