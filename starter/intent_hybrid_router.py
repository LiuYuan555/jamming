"""Observable intent-conditioned hybrid routing and entropy question selection.

This module is deliberately an experimental adapter: it does not change the
production :mod:`starter.agent` pipeline.  A trained four-class classifier can
be injected through the small ``IntentClassifier`` protocol.  The default
classifier is a transparent, calibrated-to-the-released-simulator baseline so
the routing experiment can be run before a learned classifier artifact exists.

No target ID, hidden intent card, or evaluator scenario label is accepted by
any runtime API in this module.
"""

from __future__ import annotations

import contextvars
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from starter.agent import Agent, AgentConfig
from starter.constraint_reranker import CatalogReranker, Constraint
from starter.conversation_state import (
    Attribute,
    ConstraintSource,
    ConversationState,
)
from starter.hybrid_retrieval import HybridConfig
from starter.information_gain_policy import (
    ATTRIBUTE_MESSAGES,
    CatalogConstraintIndex,
)
from starter.question_policy import QuestionDecision


class ShoppingIntent(str, Enum):
    BROWSING = "browsing"
    BUYING = "buying"
    INTENT_OVERRIDE = "intent_override"
    BOUNDARY = "boundary"


INTENTS = tuple(ShoppingIntent)


@dataclass(frozen=True)
class IntentPrediction:
    probabilities: Mapping[ShoppingIntent, float]

    def __post_init__(self) -> None:
        normalized = {
            intent: float(self.probabilities.get(intent, 0.0)) for intent in INTENTS
        }
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
            raise ValueError("intent probabilities must be finite and non-negative")
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("at least one intent probability must be positive")
        object.__setattr__(
            self,
            "probabilities",
            {intent: value / total for intent, value in normalized.items()},
        )

    @property
    def intent(self) -> ShoppingIntent:
        return max(INTENTS, key=lambda item: self.probabilities[item])

    @property
    def confidence(self) -> float:
        return self.probabilities[self.intent]


class IntentClassifier(Protocol):
    def predict_proba(self, messages: Sequence[str]) -> IntentPrediction:
        """Classify the conversation using only messages observed so far."""


def _coerce_intent_prediction(value: object) -> IntentPrediction:
    """Adapt portable classifier predictions without importing its NumPy stack.

    The trained ``starter.intent_classifier.IntentClassifier`` returns an
    object whose probability keys are strings.  Experimental classifiers may
    return this module's native prediction, a mapping, or a probability vector
    in the portable classifier's documented order.
    """

    if isinstance(value, IntentPrediction):
        return value
    probabilities = getattr(value, "probabilities", value)
    if isinstance(probabilities, Mapping):
        converted: dict[ShoppingIntent, float] = {}
        for raw_key, raw_value in probabilities.items():
            key = raw_key.value if isinstance(raw_key, ShoppingIntent) else str(raw_key)
            try:
                converted[ShoppingIntent(key)] = float(raw_value)
            except ValueError:
                continue
        return IntentPrediction(converted)
    # This is intentionally duck-typed so a NumPy array remains optional here.
    try:
        values = [float(item) for item in probabilities]  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError("classifier prediction must expose intent probabilities") from exc
    portable_order = (
        ShoppingIntent.BUYING,
        ShoppingIntent.BROWSING,
        ShoppingIntent.INTENT_OVERRIDE,
        ShoppingIntent.BOUNDARY,
    )
    if len(values) != len(portable_order):
        raise ValueError("classifier probability vector must have four values")
    return IntentPrediction(dict(zip(portable_order, values)))


def _classify_observable_history(
    classifier: object,
    messages: Sequence[str],
    asked_attributes: Sequence[str],
) -> IntentPrediction:
    """Invoke either the portable trained API or the local baseline API."""

    predict = getattr(classifier, "predict", None)
    if callable(predict):
        return _coerce_intent_prediction(predict(messages, asked_attributes))
    predict_proba = getattr(classifier, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError("classifier must define predict or predict_proba")
    try:
        value = predict_proba(messages, asked_attributes)
    except TypeError:
        # ObservableIntentBaseline predates the asked-attribute feature and is
        # kept as the dependency-free fallback.
        value = predict_proba(messages)
    return _coerce_intent_prediction(value)


class ObservableIntentBaseline:
    """Small deterministic baseline for testing the router integration.

    It intentionally retains uncertainty when the released simulator emits an
    identical first message for browsing and boundary.  Replace this object
    with a trained classifier for the final experiment; the router API stays
    unchanged.
    """

    @staticmethod
    def _prediction(**values: float) -> IntentPrediction:
        return IntentPrediction({ShoppingIntent(key): value for key, value in values.items()})

    def predict_proba(self, messages: Sequence[str]) -> IntentPrediction:
        text = "\n".join(str(message).casefold() for message in messages)
        latest = str(messages[-1]).casefold() if messages else ""
        first = str(messages[0]).casefold() if messages else ""

        if latest.startswith("actually,") or "what i need is:" in latest:
            return self._prediction(
                browsing=0.01, buying=0.01, intent_override=0.97, boundary=0.01
            )
        if (
            "please use your judgment" in text
            or (
                "don't have a preference for" in latest
                and "additional preference" not in latest
            )
        ):
            return self._prediction(
                browsing=0.03, buying=0.02, intent_override=0.02, boundary=0.93
            )
        if "what i need is:" in text:
            return self._prediction(
                browsing=0.02, buying=0.08, intent_override=0.88, boundary=0.02
            )
        if "a key requirement is:" in first:
            return self._prediction(
                browsing=0.03, buying=0.92, intent_override=0.04, boundary=0.01
            )
        if "still exploring" in first and "for that, what matters is:" in text:
            # Observable state has moved from open-ended exploration to a
            # concrete requirement even if the hidden scenario remains named
            # "browsing". This matches the portable classifier's event labels.
            return self._prediction(
                browsing=0.07, buying=0.89, intent_override=0.02, boundary=0.02
            )
        if "still exploring" in first:
            # The public browsing and boundary initial messages are identical.
            # Their released mixture is 80:10, hence the non-zero boundary mass.
            return self._prediction(
                browsing=0.86, buying=0.02, intent_override=0.02, boundary=0.10
            )
        if first:
            # A category plus an unlabelled old preference is the observable
            # pre-override pattern, but keep buying probability for robustness.
            return self._prediction(
                browsing=0.04, buying=0.15, intent_override=0.78, boundary=0.03
            )
        return self._prediction(
            browsing=0.25, buying=0.25, intent_override=0.25, boundary=0.25
        )


@dataclass(frozen=True)
class IntentRoutingConfig:
    """Dense-weight schedules; the sparse weight is always ``1 - dense``.

    Schedule index zero applies to one observed constraint and the final value
    is reused after the schedule is exhausted.
    """

    browsing_dense: tuple[float, ...] = (0.45, 0.35, 0.25, 0.20)
    buying_dense: tuple[float, ...] = (0.30, 0.20, 0.10, 0.05)
    override_dense: tuple[float, ...] = (0.50, 0.35, 0.20)
    boundary_dense: tuple[float, ...] = (0.25, 0.20, 0.15, 0.15)
    safe_baseline_dense: float = 0.15
    confidence_floor: float = 0.25

    def __post_init__(self) -> None:
        schedules = (
            self.browsing_dense,
            self.buying_dense,
            self.override_dense,
            self.boundary_dense,
        )
        if any(not schedule for schedule in schedules):
            raise ValueError("intent schedules cannot be empty")
        values = [self.safe_baseline_dense, self.confidence_floor]
        values.extend(value for schedule in schedules for value in schedule)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("routing values must be finite values from zero to one")
        if self.confidence_floor >= 1:
            raise ValueError("confidence_floor must be less than one")

    @classmethod
    def conservative(cls) -> "IntentRoutingConfig":
        """Evidence-calibrated opt-in profile that keeps browsing relatively dense."""

        return cls(
            browsing_dense=(0.20, 0.15, 0.10, 0.05),
            buying_dense=(0.10, 0.05, 0.0, 0.0),
            override_dense=(0.20, 0.10, 0.0),
            boundary_dense=(0.10, 0.05, 0.05),
        )


@dataclass(frozen=True)
class RouteDecision:
    sparse_weight: float
    dense_weight: float
    predicted_intent: ShoppingIntent
    confidence: float
    information_count: int
    class_dense_weights: Mapping[ShoppingIntent, float]


class IntentHybridRouter:
    """Map calibrated intent probabilities to safe sparse/dense RRF weights."""

    def __init__(self, config: IntentRoutingConfig | None = None) -> None:
        self.config = config or IntentRoutingConfig()

    @staticmethod
    def _scheduled(schedule: Sequence[float], information_count: int) -> float:
        index = min(max(information_count, 1), len(schedule)) - 1
        return float(schedule[index])

    def route(
        self,
        prediction: IntentPrediction,
        information_count: int,
        *,
        turns_since_override: int | None = None,
    ) -> RouteDecision:
        config = self.config
        buying = self._scheduled(config.buying_dense, information_count)
        class_weights = {
            ShoppingIntent.BROWSING: self._scheduled(
                config.browsing_dense, information_count
            ),
            ShoppingIntent.BUYING: buying,
            ShoppingIntent.INTENT_OVERRIDE: (
                self._scheduled(config.override_dense, turns_since_override + 1)
                if turns_since_override is not None
                else config.override_dense[0]
            ),
            ShoppingIntent.BOUNDARY: self._scheduled(
                config.boundary_dense, information_count
            ),
        }
        probability_mixture = sum(
            prediction.probabilities[intent] * class_weights[intent]
            for intent in INTENTS
        )
        # At chance confidence, use the validated conservative 0.85/0.15 route.
        # Confidence smoothly unlocks the class-specific schedule.
        alpha = max(
            0.0,
            min(
                1.0,
                (prediction.confidence - config.confidence_floor)
                / (1.0 - config.confidence_floor),
            ),
        )
        dense = alpha * probability_mixture + (1.0 - alpha) * config.safe_baseline_dense
        dense = max(0.0, min(1.0, dense))
        return RouteDecision(
            sparse_weight=1.0 - dense,
            dense_weight=dense,
            predicted_intent=prediction.intent,
            confidence=prediction.confidence,
            information_count=information_count,
            class_dense_weights=class_weights,
        )


@dataclass(frozen=True)
class EntropyGateConfig:
    candidate_limit: int = 10
    rank_temperature: float = 8.0
    coverage_power: float = 0.5
    minimum_value: float = 0.015
    mask_repeated: bool = True
    fallback_attribute: str = "other"

    def __post_init__(self) -> None:
        if self.candidate_limit < 2:
            raise ValueError("candidate_limit must be at least two")
        if self.rank_temperature <= 0 or not math.isfinite(self.rank_temperature):
            raise ValueError("rank_temperature must be finite and positive")
        if self.coverage_power < 0 or not math.isfinite(self.coverage_power):
            raise ValueError("coverage_power must be finite and non-negative")
        if self.minimum_value < 0 or not math.isfinite(self.minimum_value):
            raise ValueError("minimum_value must be finite and non-negative")
        if self.fallback_attribute not in ATTRIBUTE_MESSAGES:
            raise ValueError("fallback_attribute must be askable")


@dataclass(frozen=True)
class EntropyAttributeScore:
    attribute: str
    prior_entropy_bits: float
    expected_posterior_entropy_bits: float
    information_gain_bits: float
    normalized_information_gain: float
    answer_coverage: float
    value: float
    masked: bool = False
    mask_reason: str | None = None


@dataclass(frozen=True)
class EntropyGateDecision:
    decision: QuestionDecision
    scores: tuple[EntropyAttributeScore, ...]
    used_fallback: bool


class ShannonEntropyQuestionGate:
    """Rank-aware expected information gain over the observable Top-K.

    ``other`` is evaluated using its real wildcard semantics: up to two
    remaining constraints of any type.  It is not treated as a tenth ordinary
    field and receives no artificial specificity discount.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        config: EntropyGateConfig | None = None,
    ) -> None:
        self.config = config or EntropyGateConfig()
        self.index = CatalogConstraintIndex(catalog_path)

    @staticmethod
    def _entropy(probabilities: Sequence[float]) -> float:
        return -sum(value * math.log2(value) for value in probabilities if value > 0)

    def _rank_probabilities(self, count: int) -> list[float]:
        values = [math.exp(-rank / self.config.rank_temperature) for rank in range(count)]
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def _active_values(state: ConversationState) -> set[str]:
        # CatalogConstraintIndex stores normalized alphanumeric strings.
        from starter.information_gain_policy import _normalize

        return {
            _normalize(record.value)
            for record in state.active_constraints()
            if record.attribute is not Attribute.CATEGORY and _normalize(record.value)
        }

    def _answer(
        self,
        parent_asin: str,
        attribute: str,
        disclosed: set[str],
    ) -> tuple[str, ...]:
        remaining = [
            (kind, value)
            for kind, value in self.index.constraints.get(parent_asin, ())
            if value not in disclosed
        ]
        if attribute != "other":
            remaining = [item for item in remaining if item[0] == attribute]
        return tuple(value for _, value in remaining[:2])

    def score_attributes(
        self,
        state: ConversationState,
        candidate_ids: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> tuple[EntropyAttributeScore, ...]:
        candidates = [
            item
            for item in dict.fromkeys(str(value) for value in candidate_ids)
            if item in self.index.constraints
        ][: self.config.candidate_limit]
        probabilities = self._rank_probabilities(len(candidates)) if candidates else []
        prior = self._entropy(probabilities)
        disclosed = self._active_values(state)
        refused = {item.value for item in state.no_preference_attributes}
        asked = set(asked_attributes)
        scores: list[EntropyAttributeScore] = []

        for attribute in ATTRIBUTE_MESSAGES:
            reason: str | None = None
            if attribute in refused:
                reason = "no_preference"
            elif self.config.mask_repeated and attribute in asked:
                reason = "already_asked"
            answers = [self._answer(item, attribute, disclosed) for item in candidates]
            coverage = sum(
                probability for probability, answer in zip(probabilities, answers) if answer
            )
            if reason is None and coverage == 0:
                reason = "zero_coverage"

            grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
            for probability, answer in zip(probabilities, answers):
                grouped[answer].append(probability)
            posterior = 0.0
            for group in grouped.values():
                mass = sum(group)
                if mass > 0:
                    posterior += mass * self._entropy([value / mass for value in group])
            gain = max(0.0, prior - posterior)
            normalized = gain / prior if prior > 0 else 0.0
            value = normalized * coverage ** self.config.coverage_power
            if reason is not None:
                value = float("-inf")
            scores.append(
                EntropyAttributeScore(
                    attribute,
                    prior,
                    posterior,
                    gain,
                    normalized,
                    coverage,
                    value,
                    reason is not None,
                    reason,
                )
            )
        return tuple(scores)

    def decide(
        self,
        state: ConversationState,
        candidate_ids: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> EntropyGateDecision:
        scores = self.score_attributes(state, candidate_ids, asked_attributes)
        viable = [item for item in scores if not item.masked]
        best = max(viable, key=lambda item: item.value) if viable else None
        used_fallback = best is None or best.value < self.config.minimum_value
        if used_fallback:
            fallback = next(
                (
                    item
                    for item in scores
                    if item.attribute == self.config.fallback_attribute and not item.masked
                ),
                best,
            )
            attribute = fallback.attribute if fallback is not None else self.config.fallback_attribute
        else:
            attribute = best.attribute
        return EntropyGateDecision(
            QuestionDecision(ATTRIBUTE_MESSAGES[attribute], attribute),
            scores,
            used_fallback,
        )


@dataclass(frozen=True)
class IntentTopKConfig:
    """Opt-in intent-specific presentation policy over a wider ranked pool."""

    pool_size: int = 30
    diversity_weight: float = 0.12
    original_rank_penalty: float = 0.03

    def __post_init__(self) -> None:
        if self.pool_size < 10:
            raise ValueError("Top-K policy pool_size must be at least ten")
        values = (self.diversity_weight, self.original_rank_penalty)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Top-K policy weights must be finite and non-negative")


@dataclass(frozen=True)
class IntentTopKDecision:
    recommendations: tuple[dict[str, str], ...]
    changed: bool
    intent: ShoppingIntent


class IntentTopKPolicy:
    """Apply intent-specific ordering without filtering the candidate pool.

    Buying preserves the constraint reranker's order. Browsing uses a small
    maximal-marginal-relevance penalty based on observable catalog metadata,
    while pinning the first result. Override reranks strictly from active
    post-retraction constraints. Boundary uses active category constraints only;
    a refusal is missing information and never becomes a negative constraint.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        config: IntentTopKConfig | None = None,
    ) -> None:
        self.config = config or IntentTopKConfig()
        self.reranker = CatalogReranker(catalog_path)
        self.signatures: dict[str, frozenset[str]] = {}
        for parent_asin, document in self.reranker.documents.items():
            tokens: set[str] = set()
            for field_name in ("categories", "store", "title"):
                tokens.update(
                    f"{field_name}:{token}"
                    for token in document.fields.get(field_name, "").split()
                )
            self.signatures[parent_asin] = frozenset(tokens)

    @staticmethod
    def _ids(recommendations: Sequence[Mapping[str, object]]) -> list[str]:
        return list(dict.fromkeys(
            str(item.get("parent_asin", ""))
            for item in recommendations
            if str(item.get("parent_asin", ""))
        ))

    @staticmethod
    def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _diversify(self, candidate_ids: Sequence[str], top_k: int) -> list[str]:
        remaining = list(candidate_ids[: self.config.pool_size])
        if not remaining or top_k <= 0:
            return []
        # Pin rank one: diversification should not sacrifice an existing best
        # answer merely to make the list look varied.
        selected = [remaining.pop(0)]
        original_rank = {value: rank for rank, value in enumerate(candidate_ids)}
        while remaining and len(selected) < top_k:
            def objective(parent_asin: str) -> tuple[float, int]:
                signature = self.signatures.get(parent_asin, frozenset())
                redundancy = max(
                    (
                        self._similarity(
                            signature,
                            self.signatures.get(chosen, frozenset()),
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                score = -(
                    self.config.original_rank_penalty * original_rank[parent_asin]
                    + self.config.diversity_weight * redundancy
                )
                return score, -original_rank[parent_asin]

            chosen = max(remaining, key=objective)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    @staticmethod
    def _constraints(
        state: ConversationState,
        *,
        category_only: bool = False,
    ) -> list[Constraint]:
        constraints: list[Constraint] = []
        for record in state.active_constraints():
            attribute = record.attribute.value
            if category_only and attribute != "category":
                continue
            constraints.append(
                Constraint(
                    attribute,
                    record.value,
                    hard=record.source
                    in {
                        ConstraintSource.INITIAL_CATEGORY,
                        ConstraintSource.INITIAL_REQUIREMENT,
                        ConstraintSource.OVERRIDE,
                    },
                )
            )
        return constraints

    def apply(
        self,
        intent: ShoppingIntent,
        recommendations: Sequence[Mapping[str, object]],
        state: ConversationState,
        *,
        top_k: int = 10,
    ) -> IntentTopKDecision:
        candidate_ids = self._ids(recommendations)
        original = candidate_ids[:top_k]
        if intent is ShoppingIntent.BROWSING:
            selected = self._diversify(candidate_ids, top_k)
        elif intent is ShoppingIntent.INTENT_OVERRIDE:
            # ConversationState.active_constraints() excludes retracted values.
            selected = [
                item.parent_asin
                for item in self.reranker.rerank(
                    candidate_ids, self._constraints(state), top_k=top_k
                )
            ]
        elif intent is ShoppingIntent.BOUNDARY:
            # A refusal is deliberately absent: category provides a safe broad
            # ordering and rerank() does not drop non-matches.
            selected = [
                item.parent_asin
                for item in self.reranker.rerank(
                    candidate_ids,
                    self._constraints(state, category_only=True),
                    top_k=top_k,
                )
            ]
        else:
            selected = original
        payload = tuple({"parent_asin": parent_asin} for parent_asin in selected)
        return IntentTopKDecision(payload, selected != original, intent)


@dataclass(frozen=True)
class _RoutingContext:
    prediction: IntentPrediction
    turns_since_override: int | None


class RoutedHybridConfig:
    """HybridConfig-compatible wrapper whose weights are set per response."""

    def __init__(self, base: HybridConfig, router: IntentHybridRouter) -> None:
        for key, value in vars(base).items():
            setattr(self, key, value)
        self.router = router
        self._context: contextvars.ContextVar[_RoutingContext | None] = (
            contextvars.ContextVar("intent_hybrid_route", default=None)
        )
        self.last_decision: RouteDecision | None = None

    def activate(self, context: _RoutingContext) -> contextvars.Token:
        return self._context.set(context)

    def deactivate(self, token: contextvars.Token) -> None:
        self._context.reset(token)

    def fusion_weights(
        self, constraints: Sequence[Mapping[str, object]]
    ) -> tuple[float, float]:
        context = self._context.get()
        if context is None:
            return float(self.sparse_weight), float(self.dense_weight)
        self.last_decision = self.router.route(
            context.prediction,
            len(constraints),
            turns_since_override=context.turns_since_override,
        )
        return self.last_decision.sparse_weight, self.last_decision.dense_weight

    def semantic_prompt_variant(self) -> str:
        """Route the tested dense-query style using observable intent only."""

        context = self._context.get()
        if context is None:
            return str(getattr(self, "query_prompt_variant", "natural_description"))
        if context.prediction.intent is ShoppingIntent.BOUNDARY:
            return "catalog_phrase"
        # The recall-oriented prompt performed best for Buying and is the safe
        # recovery choice for Browsing/Override when dense retrieval is active.
        return "natural_description"


class IntentConditionedHybridAgent(Agent):
    """Drop-in experimental Agent with intent routing and optional entropy gate."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        *,
        classifier: IntentClassifier | None = None,
        routing_config: IntentRoutingConfig | None = None,
        entropy_config: EntropyGateConfig | None = None,
        use_entropy_gate: bool = True,
        topk_config: IntentTopKConfig | None = None,
        use_intent_topk: bool = False,
    ) -> None:
        requested = config or AgentConfig(
            hybrid_config=HybridConfig(enabled=True, strict=False)
        )
        self.classifier = classifier or ObservableIntentBaseline()
        self.intent_router = IntentHybridRouter(routing_config)
        self.routed_hybrid_config = RoutedHybridConfig(
            requested.hybrid_config, self.intent_router
        )
        self.use_entropy_gate = use_entropy_gate
        self.use_intent_topk = use_intent_topk
        parent_config = replace(
            requested,
            question_policy="none" if use_entropy_gate else "always_other",
            hybrid_config=self.routed_hybrid_config,  # type: ignore[arg-type]
        )
        super().__init__(catalog_path, config=parent_config)
        self.entropy_gate = (
            ShannonEntropyQuestionGate(catalog_path, entropy_config)
            if use_entropy_gate
            else None
        )
        self.intent_topk_policy = (
            IntentTopKPolicy(catalog_path, topk_config) if use_intent_topk else None
        )
        self._intent_messages: dict[str, list[str]] = {}
        self._override_turn: dict[str, int] = {}
        self.route_history: dict[str, list[RouteDecision]] = defaultdict(list)
        self.intent_counts: Counter[str] = Counter()
        self.entropy_decision_counts: Counter[str] = Counter()
        self.topk_counts: Counter[str] = Counter()
        self.topk_changed_counts: Counter[str] = Counter()
        self.last_entropy_decision: dict[str, EntropyGateDecision] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self._intent_messages[session_id] = []
        self._override_turn.pop(session_id, None)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        history = self._intent_messages.setdefault(session_id, [])
        history.append(str(user_message))
        if str(user_message).strip().casefold().startswith("actually,"):
            self._override_turn[session_id] = turn
        override_turn = self._override_turn.get(session_id)
        override_age = None if override_turn is None else max(0, turn - override_turn)
        asked_attributes = self._sessions[session_id].asked_attributes
        prediction = _classify_observable_history(
            self.classifier, history, asked_attributes
        )
        self.intent_counts[prediction.intent.value] += 1
        token = self.routed_hybrid_config.activate(
            _RoutingContext(prediction, override_age)
        )
        try:
            retrieval_top_k = (
                max(top_k, self.intent_topk_policy.config.pool_size)
                if self.intent_topk_policy is not None
                else top_k
            )
            response = super().respond(
                session_id, user_message, turn, retrieval_top_k
            )
        finally:
            self.routed_hybrid_config.deactivate(token)
        route = self.routed_hybrid_config.last_decision
        if route is not None:
            self.route_history[session_id].append(route)

        session = self._sessions[session_id]
        if self.intent_topk_policy is not None:
            if session.structured is None:
                raise RuntimeError("intent Top-K policy requires typed conversation state")
            topk_decision = self.intent_topk_policy.apply(
                prediction.intent,
                response.get("recommendations", ()),
                session.structured,
                top_k=top_k,
            )
            response["recommendations"] = list(topk_decision.recommendations)
            self.topk_counts[prediction.intent.value] += 1
            if topk_decision.changed:
                self.topk_changed_counts[prediction.intent.value] += 1

        if self.entropy_gate is None:
            return response
        if session.structured is None:
            raise RuntimeError("entropy gating requires typed conversation state")
        candidate_ids = [
            str(item.get("parent_asin", ""))
            for item in response.get("recommendations", ())
            if isinstance(item, Mapping) and item.get("parent_asin")
        ]
        if len(candidate_ids) < 2:
            candidate_ids = list(session.last_retrieval.get("selected_sparse", ()))
        result = self.entropy_gate.decide(
            session.structured, candidate_ids, session.asked_attributes
        )
        decision = result.decision
        if decision.ask_attribute is not None:
            session.asked_attributes.append(decision.ask_attribute)
        session.last_asked_attribute = decision.ask_attribute
        self.entropy_decision_counts[str(decision.ask_attribute)] += 1
        self.last_entropy_decision[session_id] = result
        response["message"] = decision.message
        response["ask_attribute"] = decision.ask_attribute
        return response

    def diagnostics(self) -> dict[str, object]:
        route_counts: Counter[str] = Counter()
        dense_weights: list[float] = []
        for routes in self.route_history.values():
            for route in routes:
                route_counts[route.predicted_intent.value] += 1
                dense_weights.append(route.dense_weight)
        return {
            "intent_turn_counts": dict(sorted(self.intent_counts.items())),
            "routed_turn_counts": dict(sorted(route_counts.items())),
            "mean_dense_weight": (
                sum(dense_weights) / len(dense_weights) if dense_weights else None
            ),
            "entropy_decision_counts": dict(sorted(self.entropy_decision_counts.items())),
            "topk_turn_counts": dict(sorted(self.topk_counts.items())),
            "topk_changed_counts": dict(sorted(self.topk_changed_counts.items())),
            "schedules": {
                "browsing_dense": self.intent_router.config.browsing_dense,
                "buying_dense": self.intent_router.config.buying_dense,
                "intent_override_dense_by_age": self.intent_router.config.override_dense,
                "boundary_dense": self.intent_router.config.boundary_dense,
                "safe_baseline_dense": self.intent_router.config.safe_baseline_dense,
            },
        }


__all__ = [
    "EntropyAttributeScore",
    "EntropyGateConfig",
    "EntropyGateDecision",
    "IntentClassifier",
    "IntentConditionedHybridAgent",
    "IntentHybridRouter",
    "IntentPrediction",
    "IntentRoutingConfig",
    "IntentTopKConfig",
    "IntentTopKDecision",
    "IntentTopKPolicy",
    "ObservableIntentBaseline",
    "RouteDecision",
    "ShannonEntropyQuestionGate",
    "ShoppingIntent",
]
