"""Observable, catalog-statistics question selection for TechJam.

The policy in this module deliberately has no target-product or scenario input.
It estimates the value of a clarification from the products that the existing
retriever considers plausible, then chooses the attribute whose possible
answers most reduce uncertainty over that candidate set.

``InformationGainAgent`` is a drop-in evaluator adapter.  It leaves retrieval
and reranking in :class:`starter.agent.Agent` unchanged and replaces only its
fixed clarification decision.  Keeping the adapter here makes the experiment
possible without changing the production agent during an ablation.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from starter.agent import Agent, AgentConfig
from starter.conversation_state import Attribute, ConversationState
from starter.question_policy import QuestionDecision


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

# Category is present in every initial simulator message.  ``other`` remains a
# candidate because the public API explicitly supports it, but receives a small
# specificity discount so a named attribute wins when it partitions candidates
# equally well.  These are priors, not fitted values.
DEFAULT_ACTION_WEIGHTS: Mapping[str, float] = {
    "material": 1.00,
    "color": 1.00,
    "size": 1.00,
    "style": 1.00,
    "brand": 0.90,
    "budget": 0.95,
    "feature": 1.00,
    "use_case": 1.00,
    "other": 0.92,
}

ATTRIBUTE_MESSAGES: Mapping[str, str] = {
    "material": "Do you have a preferred material?",
    "color": "Which colors would you prefer?",
    "size": "What size or fit should I look for?",
    "style": "Is there a particular style you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I stay within?",
    "feature": "Which features or qualities matter most to you?",
    "use_case": "What will you mainly use the product for?",
    "other": "What other preferences should I consider?",
}


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _flatten_text(value: object) -> str:
    return " ".join(_flatten_values(value))


def _normalize(value: object, limit: int = 180) -> str:
    text = " ".join(TOKEN_RE.findall(str(value).casefold()))
    return text[:limit].rstrip()


def _searchable_text(product: Mapping[str, object]) -> str:
    return " ".join(_flatten_text(product.get(field_name)) for field_name in SEARCH_FIELDS)


def _classify(value: str) -> str:
    lowered = value.casefold()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered) or "color" in lowered:
        return "color"
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in SIZE_WORDS):
        return "size"
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in STYLE_WORDS):
        return "style"
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in USE_CASE_WORDS):
        return "use_case"
    return "feature"


def _intent_like_constraints(product: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Approximate possible replies from observable catalog metadata.

    The ordering mirrors the released simulator at a high level: explicit
    material/color mentions precede feature/detail values, and price is last.
    It does not inspect a sample, target, hidden intent card, or behavior label.
    """

    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).casefold())
    if color:
        candidates.insert(1, f"color: {color.group(1).casefold()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append((_classify(str(candidate)), normalized))
        # Only the first four values can be disclosed by the released intent
        # generator, so later metadata should not inflate estimated coverage.
        if len(result) == 4:
            break
    return tuple(result)


@dataclass(frozen=True)
class InformationGainConfig:
    """Hyperparameters for the deterministic question baseline."""

    candidate_limit: int = 100
    max_values_per_answer: int = 2
    coverage_power: float = 0.5
    mask_repeated: bool = True
    mask_zero_coverage: bool = True
    action_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ACTION_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.candidate_limit < 2:
            raise ValueError("candidate_limit must be at least two")
        if self.max_values_per_answer < 1:
            raise ValueError("max_values_per_answer must be positive")
        if not math.isfinite(self.coverage_power) or self.coverage_power < 0:
            raise ValueError("coverage_power must be finite and non-negative")
        unknown = set(self.action_weights) - set(ATTRIBUTE_MESSAGES)
        if unknown:
            raise ValueError(f"unknown question attributes: {sorted(unknown)}")
        if not self.action_weights:
            raise ValueError("at least one question attribute is required")
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self.action_weights.values()
        ):
            raise ValueError("action weights must be finite and non-negative")


@dataclass(frozen=True)
class AttributeScore:
    attribute: str
    score: float
    information_gain: float
    normalized_information_gain: float
    answer_rate: float
    masked: bool = False
    mask_reason: str | None = None


@dataclass(frozen=True)
class InformationGainDecision:
    decision: QuestionDecision
    scores: tuple[AttributeScore, ...]
    candidate_count: int


class CatalogConstraintIndex:
    """Compact possible-answer index built solely from the product catalog."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.constraints: dict[str, tuple[tuple[str, str], ...]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.constraints[parent_asin] = _intent_like_constraints(product)


class InformationGainQuestionPolicy:
    """Choose a question by expected candidate-entropy reduction."""

    def __init__(
        self,
        catalog_path: str | Path,
        config: InformationGainConfig | None = None,
    ) -> None:
        self.config = config or InformationGainConfig()
        self.index = CatalogConstraintIndex(catalog_path)
        self.decision_counts: Counter[str] = Counter()
        self.mask_counts: Counter[str] = Counter()

    @staticmethod
    def _active_values(state: ConversationState) -> set[str]:
        return {
            _normalize(record.value)
            for record in state.active_constraints()
            if record.attribute is not Attribute.CATEGORY and _normalize(record.value)
        }

    def _possible_answer(
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
            remaining = [(kind, value) for kind, value in remaining if kind == attribute]
        return tuple(
            value for _, value in remaining[: self.config.max_values_per_answer]
        )

    @staticmethod
    def _entropy_reduction(groups: Counter[tuple[str, ...]], total: int) -> float:
        if total < 2:
            return 0.0
        prior_entropy = math.log(total)
        posterior_entropy = sum(
            (count / total) * math.log(count) for count in groups.values()
        )
        return max(0.0, prior_entropy - posterior_entropy)

    def score_attributes(
        self,
        state: ConversationState,
        candidate_ids: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> tuple[AttributeScore, ...]:
        # Preserve retrieval order and ignore IDs absent from the observable
        # catalog index.  No target identifier is accepted by this API.
        candidates = list(
            dict.fromkeys(
                parent_asin
                for parent_asin in candidate_ids[: self.config.candidate_limit]
                if parent_asin in self.index.constraints
            )
        )
        no_preference = {
            attribute.value for attribute in state.no_preference_attributes
        }
        asked = set(asked_attributes)
        disclosed = self._active_values(state)
        scores: list[AttributeScore] = []

        for attribute, action_weight in self.config.action_weights.items():
            mask_reason: str | None = None
            if attribute in no_preference:
                mask_reason = "no_preference"
            elif self.config.mask_repeated and attribute in asked:
                mask_reason = "already_asked"

            answers = [
                self._possible_answer(parent_asin, attribute, disclosed)
                for parent_asin in candidates
            ]
            answer_count = sum(bool(answer) for answer in answers)
            answer_rate = answer_count / len(candidates) if candidates else 0.0
            if (
                mask_reason is None
                and self.config.mask_zero_coverage
                and answer_count == 0
            ):
                mask_reason = "zero_coverage"

            groups: Counter[tuple[str, ...]] = Counter(answers)
            information_gain = self._entropy_reduction(groups, len(candidates))
            normalizer = math.log(len(candidates)) if len(candidates) >= 2 else 1.0
            normalized = information_gain / normalizer
            score = action_weight * normalized * answer_rate ** self.config.coverage_power
            if mask_reason is not None:
                score = float("-inf")
                self.mask_counts[mask_reason] += 1
            scores.append(
                AttributeScore(
                    attribute=attribute,
                    score=score,
                    information_gain=information_gain,
                    normalized_information_gain=normalized,
                    answer_rate=answer_rate,
                    masked=mask_reason is not None,
                    mask_reason=mask_reason,
                )
            )
        return tuple(scores)

    def decide(
        self,
        state: ConversationState,
        candidate_ids: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> InformationGainDecision:
        scores = self.score_attributes(state, candidate_ids, asked_attributes)
        viable = [score for score in scores if not score.masked]
        if viable:
            # Mapping order provides a stable tie break.
            chosen = max(viable, key=lambda item: item.score).attribute
        else:
            # All attributes can become masked late in a conversation.  A
            # repeated broad question is preferable to returning an invalid
            # control signal, and mirrors the fixed baseline's fallback.
            chosen = "other" if "other" in ATTRIBUTE_MESSAGES else next(iter(ATTRIBUTE_MESSAGES))
        self.decision_counts[chosen] += 1
        return InformationGainDecision(
            decision=QuestionDecision(ATTRIBUTE_MESSAGES[chosen], chosen),
            scores=scores,
            candidate_count=min(len(dict.fromkeys(candidate_ids)), self.config.candidate_limit),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "decision_counts": dict(sorted(self.decision_counts.items())),
            "mask_counts": dict(sorted(self.mask_counts.items())),
            "config": {
                "candidate_limit": self.config.candidate_limit,
                "max_values_per_answer": self.config.max_values_per_answer,
                "coverage_power": self.config.coverage_power,
                "mask_repeated": self.config.mask_repeated,
                "mask_zero_coverage": self.config.mask_zero_coverage,
                "action_weights": dict(self.config.action_weights),
            },
        }


class InformationGainAgent(Agent):
    """Use the existing pipeline with an information-gain question policy."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        policy_config: InformationGainConfig | None = None,
    ) -> None:
        requested = config or AgentConfig()
        # ``none`` prevents the parent class from mutating asked-question state;
        # this adapter installs its decision after retrieval/reranking finish.
        super().__init__(catalog_path, config=replace(requested, question_policy="none"))
        self.information_gain_policy = InformationGainQuestionPolicy(
            catalog_path, config=policy_config
        )
        self.last_question_diagnostics: dict[str, InformationGainDecision] = {}

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        session = self._sessions[session_id]
        if session.structured is None:
            raise RuntimeError("information-gain policy requires typed conversation state")
        candidate_ids = session.last_retrieval.get("selected_sparse", ())
        if not candidate_ids:
            candidate_ids = tuple(
                str(item.get("parent_asin", ""))
                for item in response.get("recommendations", ())
                if isinstance(item, dict)
            )
        result = self.information_gain_policy.decide(
            session.structured,
            candidate_ids,
            session.asked_attributes,
        )
        decision = result.decision
        session.asked_attributes.append(str(decision.ask_attribute))
        session.last_asked_attribute = decision.ask_attribute
        self.last_question_diagnostics[session_id] = result
        response["message"] = decision.message
        response["ask_attribute"] = decision.ask_attribute
        return response

