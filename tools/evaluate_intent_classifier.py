"""Evaluate intent predictions against observable and hidden TechJam labels.

The two evaluations answer different questions. Observable intent measures the
deployable state classifier. Hidden-scenario accuracy quantifies how often that
state can stand in for the evaluator's latent scenario, including pre-event
turns where a causal classifier cannot know the future.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from starter.intent_classifier import INTENTS, IntentClassifier
from tools.train_intent_classifier import IntentExample, generate_public_examples


def confusion_matrix(truth: Sequence[str], predicted: Sequence[str]) -> list[list[int]]:
    matrix = [[0 for _ in INTENTS] for _ in INTENTS]
    for expected, actual in zip(truth, predicted):
        matrix[INTENTS.index(expected)][INTENTS.index(actual)] += 1
    return matrix


def classification_metrics(
    truth: Sequence[str], predicted: Sequence[str]
) -> dict[str, object]:
    if len(truth) != len(predicted):
        raise ValueError("truth and prediction lengths differ")
    matrix = confusion_matrix(truth, predicted)
    total = len(truth)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, intent in enumerate(INTENTS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        if support:
            f1_values.append(f1)
        per_class[intent] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "count": total,
        "accuracy": round(
            sum(matrix[index][index] for index in range(len(INTENTS))) / total,
            6,
        ) if total else None,
        "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else None,
        "labels": list(INTENTS),
        "confusion_matrix_truth_rows_prediction_columns": matrix,
        "per_class": per_class,
    }


def evaluate_slice(
    rows: Sequence[IntentExample],
    predictions: Sequence[str],
    selector: Callable[[IntentExample], bool],
) -> dict[str, object]:
    indices = [index for index, row in enumerate(rows) if selector(row)]
    subset = [rows[index] for index in indices]
    predicted = [predictions[index] for index in indices]
    return {
        "observable": classification_metrics(
            [row.observable_label for row in subset], predicted
        ),
        "hidden_scenario": classification_metrics(
            [row.hidden_scenario for row in subset], predicted
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument(
        "--samples", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl"
    )
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, categories, products = catalog_index(args.catalog)
    samples = load_jsonl(args.samples)
    rows = generate_public_examples(
        samples, products, categories, max_turns=args.max_turns
    )
    model = IntentClassifier.load(args.model)
    outputs = [
        model.predict(row.messages, row.asked_attributes) for row in rows
    ]
    predictions = [output.label for output in outputs]
    confidence = [output.confidence for output in outputs]
    correct_observable = [
        prediction == row.observable_label
        for prediction, row in zip(predictions, rows)
    ]
    # Expected calibration error for the deployable observable target.
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        indices = [
            index for index, value in enumerate(confidence)
            if lower <= value < lower + 0.1 or (lower == 0.9 and value == 1.0)
        ]
        if indices:
            bin_accuracy = np.mean([correct_observable[index] for index in indices])
            bin_confidence = np.mean([confidence[index] for index in indices])
            ece += len(indices) / len(rows) * abs(bin_accuracy - bin_confidence)

    report = {
        "model": str(args.model),
        "public_sessions": len(samples),
        "turn_examples": len(rows),
        "parameter_count": model.parameter_count,
        "target_definition": "intent_observable_up_to_current_turn",
        "all_turns": evaluate_slice(rows, predictions, lambda row: True),
        "turn_1": evaluate_slice(rows, predictions, lambda row: row.turn == 1),
        "event_observed": evaluate_slice(
            rows, predictions, lambda row: row.event_observable
        ),
        "pre_event_override_or_boundary": evaluate_slice(
            rows,
            predictions,
            lambda row: row.hidden_scenario in {"intent_override", "boundary"}
            and not row.event_observable,
        ),
        "pre_event_boundary": evaluate_slice(
            rows,
            predictions,
            lambda row: row.hidden_scenario == "boundary"
            and not row.event_observable,
        ),
        "pre_event_override": evaluate_slice(
            rows,
            predictions,
            lambda row: row.hidden_scenario == "intent_override"
            and not row.event_observable,
        ),
        "mean_confidence": round(float(np.mean(confidence)), 6),
        "provisional_rate": round(
            sum(output.provisional for output in outputs) / len(outputs), 6
        ),
        "observable_expected_calibration_error_10_bins": round(float(ece), 6),
        "interpretation": {
            "observable": (
                "Correct target for runtime routing: evidence available by this turn."
            ),
            "hidden_scenario": (
                "Diagnostic only. Boundary is identical to Browsing before the "
                "first boundary reply; Intent Override is not an observed change "
                "until its correction turn."
            ),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
