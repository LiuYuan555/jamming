"""Train the lightweight question policy from counterfactual JSONL records.

Accepted JSONL formats are either one action per row::

    {"state": {...}, "action": "material", "value": 0.72,
     "parent_asin": "B0...", "state_id": "B0...:2"}

or all counterfactual actions for a state::

    {"state": {...}, "action_values": {"material": 0.72, "other": 0.81},
     "parent_asin": "B0...", "state_id": "B0...:2"}

The counterfactual builder's namespaced v1 schema is accepted directly::

    {"model_input": {"state": {...}, "action": "material"},
     "label": {"training_reward": 0.72},
     "metadata": {"product_group_id": "...", "split": "train",
                  "sample_id": "...", "state_turn": 2}}

``parent_asin`` (or ``group_id``) is used only for a leakage-safe train/holdout
split.  It is never included in the feature vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.neural_question_policy import (
    ACTIONS,
    ActionValueNetwork,
    StateVectorizer,
    TrainingConfig,
)


@dataclass(frozen=True)
class Example:
    state: dict[str, object]
    action: str
    value: float
    group_id: str
    state_id: str
    split: str | None = None


def _value(record: Mapping[str, object]) -> float:
    for key in ("value", "q_value", "utility", "reward"):
        if key in record:
            result = float(record[key])
            if not np.isfinite(result):
                raise ValueError(f"non-finite {key}")
            return result
    raise ValueError("action row needs value, q_value, utility, or reward")


def load_examples(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                model_input = record.get("model_input", record)
                if not isinstance(model_input, dict):
                    raise ValueError("model_input must be an object")
                state = model_input["state"]
                if not isinstance(state, dict):
                    raise ValueError("state must be an object")
                metadata = record.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object")
                group_id = str(
                    record.get("group_id")
                    or record.get("parent_asin")
                    or record.get("sample_id")
                    or metadata.get("product_group_id")
                    or metadata.get("sample_id")
                    or f"line:{line_number}"
                )
                state_id = str(
                    record.get("state_id")
                    or (
                        f"{metadata.get('sample_id')}:{metadata.get('state_turn')}"
                        if metadata.get("sample_id") is not None
                        else f"line:{line_number}"
                    )
                )
                split_value = metadata.get("split", record.get("split"))
                split = str(split_value).casefold() if split_value is not None else None
                if split not in {None, "train", "validation", "test"}:
                    raise ValueError(f"invalid predefined split {split!r}")
                if "action_values" in record:
                    action_values = record["action_values"]
                    if not isinstance(action_values, dict) or not action_values:
                        raise ValueError("action_values must be a non-empty object")
                    rows = action_values.items()
                else:
                    label = record.get("label")
                    value_source = label if isinstance(label, dict) else record
                    if "training_reward" in value_source:
                        numeric_label = float(value_source["training_reward"])
                    else:
                        numeric_label = _value(value_source)
                    rows = ((model_input["action"], numeric_label),)
                for action_value, value in rows:
                    action = str(action_value).strip().casefold()
                    if action not in ACTIONS:
                        raise ValueError(f"invalid action {action!r}")
                    numeric_value = float(value)
                    if not np.isfinite(numeric_value):
                        raise ValueError("action value must be finite")
                    examples.append(
                        Example(
                            dict(state), action, numeric_value, group_id, state_id, split
                        )
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"counterfactual dataset is empty: {path}")
    return examples


def split_by_group(
    examples: list[Example], holdout_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    groups = sorted({example.group_id for example in examples})
    if holdout_fraction == 0.0 or len(groups) < 2:
        return list(examples), []
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    holdout_count = min(len(groups) - 1, max(1, round(len(groups) * holdout_fraction)))
    holdout_groups = set(shuffled[:holdout_count])
    training = [example for example in examples if example.group_id not in holdout_groups]
    holdout = [example for example in examples if example.group_id in holdout_groups]
    return training, holdout


def _arrays(
    examples: list[Example], vectorizer: StateVectorizer
) -> tuple[np.ndarray, np.ndarray]:
    features = vectorizer.matrix(
        [example.state for example in examples],
        [example.action for example in examples],
    )
    values = np.asarray([example.value for example in examples], dtype=np.float64)
    return features, values


def selection_metrics(
    model: ActionValueNetwork,
    vectorizer: StateVectorizer,
    examples: list[Example],
) -> tuple[float | None, float | None]:
    """Return optimal-action accuracy and regret for the learned policy."""

    by_state: dict[tuple[str, str], list[Example]] = {}
    for example in examples:
        by_state.setdefault((example.group_id, example.state_id), []).append(example)
    eligible = [rows for rows in by_state.values() if len(rows) >= 2]
    if not eligible:
        return None, None
    correct = 0
    total_regret = 0.0
    for rows in eligible:
        features, _ = _arrays(rows, vectorizer)
        predictions = model.predict(features)
        selected = rows[int(np.argmax(predictions))]
        best_value = max(row.value for row in rows)
        optimal_actions = {row.action for row in rows if row.value == best_value}
        correct += selected.action in optimal_actions
        total_regret += best_value - selected.value
    return correct / len(eligible), total_regret / len(eligible)


def selection_accuracy(
    model: ActionValueNetwork,
    vectorizer: StateVectorizer,
    examples: list[Example],
) -> float | None:
    """Backward-compatible accuracy-only view used by existing callers."""

    return selection_metrics(model, vectorizer, examples)[0]


def fixed_action_metrics(
    examples: list[Example], action: str = "other"
) -> tuple[float | None, float | None]:
    """Return optimal-action accuracy and mean regret for a fixed baseline."""

    by_state: dict[tuple[str, str], list[Example]] = {}
    for example in examples:
        by_state.setdefault((example.group_id, example.state_id), []).append(example)
    eligible = [
        rows for rows in by_state.values() if any(row.action == action for row in rows)
    ]
    if not eligible:
        return None, None
    correct = 0
    total_regret = 0.0
    for rows in eligible:
        best_value = max(row.value for row in rows)
        selected_value = next(row.value for row in rows if row.action == action)
        correct += selected_value == best_value
        total_regret += best_value - selected_value
    return correct / len(eligible), total_regret / len(eligible)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="report the predefined test split after model selection is complete",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset)
    predefined = {row.split for row in examples if row.split is not None}
    test_examples: list[Example] = []
    if predefined:
        if any(row.split is None for row in examples) or "train" not in predefined:
            raise ValueError(
                "predefined dataset splits must cover every row and include train"
            )
        training = [row for row in examples if row.split == "train"]
        holdout = [row for row in examples if row.split == "validation"]
        test_examples = [row for row in examples if row.split == "test"]
        split_strategy = "predefined_product_group"
    else:
        training, holdout = split_by_group(
            examples, args.holdout_fraction, args.seed
        )
        split_strategy = "random_product_group"
    vectorizer = StateVectorizer()
    train_x, train_y = _arrays(training, vectorizer)
    validation = _arrays(holdout, vectorizer) if holdout else None
    config = TrainingConfig(
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    model = ActionValueNetwork(vectorizer.dimension, config.hidden_size, seed=config.seed)
    result = model.fit(train_x, train_y, config=config, validation=validation)
    model.save(args.output)
    training_other_accuracy, training_other_regret = fixed_action_metrics(training)
    holdout_other_accuracy, holdout_other_regret = fixed_action_metrics(holdout)
    training_accuracy, training_regret = selection_metrics(
        model, vectorizer, training
    )
    holdout_accuracy, holdout_regret = selection_metrics(model, vectorizer, holdout)
    report = {
        "dataset": str(args.dataset),
        "model": str(args.output),
        "examples": len(examples),
        "training_examples": len(training),
        "holdout_examples": len(holdout),
        "excluded_test_examples": len(test_examples),
        "training_groups": len({row.group_id for row in training}),
        "holdout_groups": len({row.group_id for row in holdout}),
        "excluded_test_groups": len({row.group_id for row in test_examples}),
        "split_strategy": split_strategy,
        "feature_count": vectorizer.dimension,
        "hidden_size": model.hidden_size,
        "parameter_count": model.parameter_count,
        "epochs_run": result.epochs_run,
        "training_mse": result.training_mse,
        "holdout_mse": result.validation_mse,
        "training_action_accuracy": training_accuracy,
        "training_model_mean_regret": training_regret,
        "holdout_action_accuracy": holdout_accuracy,
        "holdout_model_mean_regret": holdout_regret,
        "training_always_other_accuracy": training_other_accuracy,
        "training_always_other_mean_regret": training_other_regret,
        "holdout_always_other_accuracy": holdout_other_accuracy,
        "holdout_always_other_mean_regret": holdout_other_regret,
    }
    if args.evaluate_test:
        test_accuracy, test_regret = selection_metrics(
            model, vectorizer, test_examples
        )
        test_other_accuracy, test_other_regret = fixed_action_metrics(test_examples)
        report.update({
            "test_action_accuracy": test_accuracy,
            "test_model_mean_regret": test_regret,
            "test_always_other_accuracy": test_other_accuracy,
            "test_always_other_mean_regret": test_other_regret,
        })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
