"""Lightweight learned clarification policy for offline TechJam evaluation.

The policy estimates ``Q(observable_state, ask_attribute)`` for every allowed
clarification action.  It deliberately consumes a compact structured state
rather than hidden evaluator fields or raw target-product data.  The model is a
small one-hidden-layer neural network implemented only with NumPy so that a
trained artifact remains cheap and network-free at submission time.

This module is intentionally not wired into :mod:`starter.agent`; it can be
trained and evaluated as an isolated experiment before replacing the strong
``always_other`` baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from starter.conversation_state import ConstraintStatus
from starter.question_policy import QuestionDecision, VALID_ATTRIBUTES


ACTIONS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
if frozenset(ACTIONS) != VALID_ATTRIBUTES:  # Keep evaluator controls aligned.
    raise RuntimeError("neural policy action vocabulary differs from question policy")

ACTION_MESSAGES = {
    "category": "Which product category should I focus on?",
    "material": "Do you have a preferred material?",
    "color": "Which colors would you prefer?",
    "size": "Do you have a preferred size or fit?",
    "style": "What style would you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I work within?",
    "feature": "Which features or qualities matter most to you?",
    "use_case": "What will you mainly use the product for?",
    "other": "What other preferences should I consider?",
}

# Every scalar is observable from the conversation or the current retrieval.
# Unknown values default to zero, keeping dataset generation incremental.
SCALAR_FEATURES = (
    "turn",
    "remaining_turns",
    "message_count",
    "active_constraint_count",
    "active_hard_constraint_count",
    "retracted_constraint_count",
    "asked_count",
    "no_preference_count",
    "query_term_count",
    "candidate_count",
    "top_score",
    "second_score",
    "score_gap",
    "score_entropy",
    "top_k_unique_categories",
    "top_k_attribute_diversity",
    "bm25_score",
    "dense_score",
    "has_override",
    "profile_average_prior_rating",
    "profile_preference_tag_count",
    "profile_summary_token_count",
    "message_kind_initial_count",
    "message_kind_clarification_count",
    "message_kind_no_preference_count",
    "message_kind_override_count",
    "message_kind_retry_prompt_count",
    "message_kind_free_text_count",
    "retrieval_route_global_count",
    "retrieval_route_category_count",
    "retrieval_route_clean_category_count",
    "retrieval_route_selected_sparse_count",
)

FEATURE_VERSION = 1


def _finite_number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _nested_scalar(state: Mapping[str, object], name: str) -> float:
    if name in state:
        return _finite_number(state[name])
    retrieval = state.get("retrieval")
    if isinstance(retrieval, Mapping) and name in retrieval:
        return _finite_number(retrieval[name])
    if name == "query_term_count":
        return _finite_number(state.get("retrieval_text_token_count"))
    if name == "asked_count":
        counts = state.get("prior_ask_attribute_counts")
        if isinstance(counts, Mapping):
            return sum(_finite_number(value) for value in counts.values())
    if name == "no_preference_count":
        counts = state.get("no_preference_type_counts")
        if isinstance(counts, Mapping):
            return sum(_finite_number(value) for value in counts.values())
    if name == "candidate_count":
        routes = state.get("retrieval_route_candidate_counts")
        if isinstance(routes, Mapping):
            selected = _finite_number(routes.get("selected_sparse"), default=-1.0)
            if selected >= 0:
                return selected
            return max((_finite_number(value) for value in routes.values()), default=0.0)
    if name.startswith("message_kind_") and name.endswith("_count"):
        kind = name[len("message_kind_") : -len("_count")]
        counts = state.get("message_kind_counts")
        if isinstance(counts, Mapping):
            return _finite_number(counts.get(kind))
    if name.startswith("retrieval_route_") and name.endswith("_count"):
        route = name[len("retrieval_route_") : -len("_count")]
        counts = state.get("retrieval_route_candidate_counts")
        if isinstance(counts, Mapping):
            return _finite_number(counts.get(route))
    if name == "has_override":
        counts = state.get("message_kind_counts")
        if isinstance(counts, Mapping):
            return float(_finite_number(counts.get("override")) > 0)
    return 0.0


def _attribute_counts(value: object) -> dict[str, float]:
    counts = {action: 0.0 for action in ACTIONS}
    if isinstance(value, Mapping):
        for action in ACTIONS:
            counts[action] = max(0.0, _finite_number(value.get(action)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            action = str(item).strip().casefold()
            if action in counts:
                counts[action] += 1.0
    return counts


class StateVectorizer:
    """Convert one observable state/action pair into a fixed numeric vector."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = list(SCALAR_FEATURES)
        for prefix in (
            "active_count",
            "active_mask",
            "asked_count",
            "asked_mask",
            "no_preference",
            "candidate_coverage",
            "last_asked",
            "action",
        ):
            names.extend(f"{prefix}:{action}" for action in ACTIONS)
        return tuple(names)

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def vectorize(self, state: Mapping[str, object], action: str) -> np.ndarray:
        action = str(action).strip().casefold()
        if action not in ACTIONS:
            raise ValueError(f"invalid ask_attribute: {action!r}")

        active_source = state.get(
            "active_constraint_counts", state.get("active_constraint_type_counts")
        )
        active = _attribute_counts(active_source)
        asked_source = state.get(
            "asked_attribute_counts",
            state.get("prior_ask_attribute_counts", state.get("asked_attributes")),
        )
        asked = _attribute_counts(asked_source)
        no_preference_source = state.get(
            "no_preference_attributes", state.get("no_preference_type_counts")
        )
        no_preference = _attribute_counts(no_preference_source)
        coverage = _attribute_counts(state.get("candidate_attribute_coverage"))
        last_asked = str(state.get("last_asked_attribute", "")).strip().casefold()

        values = [_nested_scalar(state, name) for name in SCALAR_FEATURES]
        values.extend(active[item] for item in ACTIONS)
        values.extend(float(active[item] > 0) for item in ACTIONS)
        values.extend(asked[item] for item in ACTIONS)
        values.extend(float(asked[item] > 0) for item in ACTIONS)
        values.extend(float(no_preference[item] > 0) for item in ACTIONS)
        values.extend(coverage[item] for item in ACTIONS)
        values.extend(float(last_asked == item) for item in ACTIONS)
        values.extend(float(action == item) for item in ACTIONS)
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.dimension,):
            raise RuntimeError("internal neural-policy feature dimension mismatch")
        return result

    def matrix(
        self,
        states: Sequence[Mapping[str, object]],
        actions: Sequence[str],
    ) -> np.ndarray:
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal length")
        if not states:
            return np.empty((0, self.dimension), dtype=np.float64)
        return np.stack(
            [self.vectorize(state, action) for state, action in zip(states, actions)]
        )


@dataclass(frozen=True)
class TrainingConfig:
    hidden_size: int = 32
    learning_rate: float = 0.003
    batch_size: int = 128
    epochs: int = 300
    weight_decay: float = 1e-5
    patience: int = 30
    seed: int = 7

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("learning rate, batch size, and epochs must be positive")
        if self.weight_decay < 0 or self.patience < 1:
            raise ValueError("weight_decay must be non-negative and patience positive")


@dataclass(frozen=True)
class TrainingResult:
    epochs_run: int
    training_mse: float
    validation_mse: float | None


class ActionValueNetwork:
    """A portable ReLU MLP mapping state/action vectors to scalar utility."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        seed: int = 7,
    ) -> None:
        if input_size < 1 or hidden_size < 1:
            raise ValueError("network dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.mean = np.zeros(input_size, dtype=np.float64)
        self.scale = np.ones(input_size, dtype=np.float64)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros(1, dtype=np.float64)

    @property
    def input_size(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.w1.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(f"features must have shape (n, {self.input_size})")
        return (features - self.mean) / self.scale

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = self._standardize(features)
        hidden = np.maximum(0.0, standardized @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def fit(
        self,
        features: np.ndarray,
        values: np.ndarray,
        *,
        config: TrainingConfig,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TrainingResult:
        features = np.asarray(features, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape != (len(values), self.input_size):
            raise ValueError("training features and values have incompatible shapes")
        if not len(values):
            raise ValueError("at least one training example is required")
        if not np.isfinite(features).all() or not np.isfinite(values).all():
            raise ValueError("training data must be finite")

        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale = np.where(scale < 1e-8, 1.0, scale)
        x_train = self._standardize(features)
        y_train = values
        if validation is not None:
            val_x, val_y = validation
            x_val = self._standardize(np.asarray(val_x, dtype=np.float64))
            y_val = np.asarray(val_y, dtype=np.float64).reshape(-1)
            if x_val.shape[0] != y_val.shape[0] or not len(y_val):
                raise ValueError("validation features and values are incompatible")
        else:
            x_val = y_val = None

        parameters = [self.w1, self.b1, self.w2, self.b2]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        rng = np.random.default_rng(config.seed)
        best_loss = math.inf
        best_parameters = [parameter.copy() for parameter in parameters]
        stale_epochs = 0
        step = 0

        for epoch in range(1, config.epochs + 1):
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), config.batch_size):
                indices = order[start : start + config.batch_size]
                batch_x = x_train[indices]
                batch_y = y_train[indices]
                pre_activation = batch_x @ self.w1 + self.b1
                hidden = np.maximum(0.0, pre_activation)
                predicted = (hidden @ self.w2 + self.b2).reshape(-1)
                output_gradient = (2.0 / len(indices)) * (predicted - batch_y)
                grad_w2 = hidden.T @ output_gradient[:, None]
                grad_w2 += 2.0 * config.weight_decay * self.w2
                grad_b2 = np.asarray([output_gradient.sum()])
                hidden_gradient = output_gradient[:, None] @ self.w2.T
                hidden_gradient[pre_activation <= 0.0] = 0.0
                grad_w1 = batch_x.T @ hidden_gradient
                grad_w1 += 2.0 * config.weight_decay * self.w1
                grad_b1 = hidden_gradient.sum(axis=0)
                gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

                step += 1
                for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    first_moments[index] *= beta1
                    first_moments[index] += (1.0 - beta1) * gradient
                    second_moments[index] *= beta2
                    second_moments[index] += (1.0 - beta2) * gradient * gradient
                    corrected_first = first_moments[index] / (1.0 - beta1**step)
                    corrected_second = second_moments[index] / (1.0 - beta2**step)
                    parameter -= config.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + epsilon
                    )

            training_mse = float(np.mean((self.predict(features) - values) ** 2))
            monitored_loss = training_mse
            if x_val is not None and y_val is not None:
                val_hidden = np.maximum(0.0, x_val @ self.w1 + self.b1)
                val_prediction = (val_hidden @ self.w2 + self.b2).reshape(-1)
                monitored_loss = float(np.mean((val_prediction - y_val) ** 2))
            if monitored_loss < best_loss - 1e-10:
                best_loss = monitored_loss
                best_parameters = [parameter.copy() for parameter in parameters]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

        self.w1[:], self.b1[:], self.w2[:], self.b2[:] = best_parameters
        training_mse = float(np.mean((self.predict(features) - values) ** 2))
        validation_mse = None
        if validation is not None:
            validation_mse = float(
                np.mean((self.predict(validation[0]) - np.asarray(validation[1])) ** 2)
            )
        return TrainingResult(epoch, training_mse, validation_mse)

    def save(self, path: str | Path) -> None:
        """Save a non-pickle model artifact suitable for offline loading."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "actions": ACTIONS,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
        }
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata)),
                mean=self.mean,
                scale=self.scale,
                w1=self.w1,
                b1=self.b1,
                w2=self.w2,
                b2=self.b2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ActionValueNetwork":
        with np.load(Path(path), allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            if metadata.get("feature_version") != FEATURE_VERSION:
                raise ValueError("unsupported neural-policy feature version")
            if tuple(metadata.get("actions", ())) != ACTIONS:
                raise ValueError("model action vocabulary does not match runtime")
            model = cls(
                int(metadata["input_size"]),
                int(metadata["hidden_size"]),
            )
            for name in ("mean", "scale", "w1", "b1", "w2", "b2"):
                loaded = np.asarray(artifact[name], dtype=np.float64)
                current = getattr(model, name)
                if loaded.shape != current.shape:
                    raise ValueError(f"invalid model tensor shape for {name}")
                current[:] = loaded
        return model


@dataclass(frozen=True)
class NeuralPolicyConfig:
    confidence_margin: float = 0.02
    minimum_value: float = -math.inf
    fallback_action: str = "other"
    mask_no_preference_actions: bool = True

    def __post_init__(self) -> None:
        if self.confidence_margin < 0 or not math.isfinite(self.confidence_margin):
            raise ValueError("confidence_margin must be finite and non-negative")
        if self.fallback_action not in ACTIONS:
            raise ValueError("fallback_action is invalid")


@dataclass(frozen=True)
class NeuralDecisionTrace:
    decision: QuestionDecision
    scores: dict[str, float]
    model_action: str | None
    used_fallback: bool
    confidence_margin: float


class NeuralQuestionPolicy:
    """Choose a clarification action, falling back to ``other`` when unsure."""

    def __init__(
        self,
        model: ActionValueNetwork,
        config: NeuralPolicyConfig = NeuralPolicyConfig(),
        vectorizer: StateVectorizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.vectorizer = vectorizer or StateVectorizer()
        if model.input_size != self.vectorizer.dimension:
            raise ValueError("model input dimension does not match state vectorizer")

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: NeuralPolicyConfig = NeuralPolicyConfig(),
    ) -> "NeuralQuestionPolicy":
        return cls(ActionValueNetwork.load(path), config=config)

    def decide(
        self,
        state: Mapping[str, object],
        allowed_actions: Iterable[str] | None = None,
    ) -> QuestionDecision:
        return self.decide_with_trace(state, allowed_actions).decision

    def decide_with_trace(
        self,
        state: Mapping[str, object],
        allowed_actions: Iterable[str] | None = None,
    ) -> NeuralDecisionTrace:
        allowed = set(ACTIONS if allowed_actions is None else allowed_actions)
        invalid = allowed.difference(ACTIONS)
        if invalid:
            raise ValueError(f"invalid allowed actions: {sorted(invalid)}")
        available = [action for action in ACTIONS if action in allowed]
        if self.config.mask_no_preference_actions:
            no_preference_source = state.get(
                "no_preference_attributes", state.get("no_preference_type_counts")
            )
            no_preference = _attribute_counts(no_preference_source)
            # Preserve the strong fallback even after a transient boundary-case
            # refusal.  In the public simulator the first ``other`` response can
            # be "no preference" even though later constraints remain hidden.
            available = [
                action
                for action in available
                if no_preference[action] <= 0
                or action == self.config.fallback_action
            ]
        if not available:
            decision = QuestionDecision("Here are the closest matches I found.", None)
            return NeuralDecisionTrace(decision, {}, None, False, 0.0)

        matrix = np.stack(
            [self.vectorizer.vectorize(state, action) for action in available]
        )
        predictions = self.model.predict(matrix)
        scores = {
            action: float(score) for action, score in zip(available, predictions)
        }
        ranked = sorted(available, key=lambda item: (-scores[item], ACTIONS.index(item)))
        model_action = ranked[0]
        margin = math.inf if len(ranked) == 1 else scores[ranked[0]] - scores[ranked[1]]
        low_confidence = (
            margin < self.config.confidence_margin
            or scores[model_action] < self.config.minimum_value
        )
        selected = model_action
        if low_confidence and self.config.fallback_action in available:
            selected = self.config.fallback_action
        decision = QuestionDecision(ACTION_MESSAGES[selected], selected)
        return NeuralDecisionTrace(
            decision=decision,
            scores=scores,
            model_action=model_action,
            used_fallback=selected != model_action,
            confidence_margin=float(margin),
        )


def observable_state_from_conversation(
    conversation: object,
    asked_attributes: Sequence[str],
    *,
    query_term_count: int = 0,
    retrieval: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build model input from the existing ``ConversationState`` public API.

    Retrieval diagnostics are supplied separately because the current agent's
    trace primarily stores ranked IDs.  Dataset generation and a future agent
    integration can add score gaps, entropy, and candidate coverage here.
    """

    constraints = list(getattr(conversation, "constraints", ()))
    active = [
        item
        for item in constraints
        if getattr(item, "status", None) is ConstraintStatus.ACTIVE
    ]
    retracted = [
        item
        for item in constraints
        if getattr(item, "status", None) is ConstraintStatus.RETRACTED
    ]
    active_counts = {action: 0 for action in ACTIONS}
    for item in active:
        attribute = getattr(getattr(item, "attribute", None), "value", None)
        if attribute in active_counts:
            active_counts[attribute] += 1
    messages = list(getattr(conversation, "messages", ()))
    no_preferences = [
        getattr(item, "value", str(item))
        for item in getattr(conversation, "no_preference_attributes", ())
    ]
    asked_counts = _attribute_counts(asked_attributes)
    return {
        "turn": max((getattr(item, "turn", 0) for item in messages), default=0),
        "message_count": len(messages),
        "active_constraint_count": len(active),
        "retracted_constraint_count": len(retracted),
        "asked_count": len(asked_attributes),
        "no_preference_count": len(no_preferences),
        "query_term_count": query_term_count,
        "has_override": any(
            getattr(getattr(item, "kind", None), "value", "") == "override"
            for item in messages
        ),
        "active_constraint_counts": active_counts,
        "asked_attribute_counts": asked_counts,
        "asked_attributes": list(asked_attributes),
        "last_asked_attribute": asked_attributes[-1] if asked_attributes else None,
        "no_preference_attributes": no_preferences,
        "retrieval": dict(retrieval or {}),
    }


__all__ = [
    "ACTIONS",
    "ACTION_MESSAGES",
    "ActionValueNetwork",
    "NeuralDecisionTrace",
    "NeuralPolicyConfig",
    "NeuralQuestionPolicy",
    "StateVectorizer",
    "TrainingConfig",
    "TrainingResult",
    "observable_state_from_conversation",
]
