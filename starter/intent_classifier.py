"""Tiny, network-free classifier for the intent observable in a conversation.

The four labels resemble the evaluator's hidden scenario names, but the
runtime contract is intentionally narrower: it predicts what has become
observable *up to the current turn*.  In particular, an undisclosed boundary
session is indistinguishable from browsing, and an intent override does not
exist as an observable event until the customer corrects the earlier request.

The model is multinomial logistic regression over deterministic hashed text
features and a handful of structural conversation features.  A JSON artifact
is sufficient at runtime; no model server, tokenizer download, or network call
is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


INTENTS = ("buying", "browsing", "intent_override", "boundary")
MODEL_VERSION = 1
DEFAULT_HASH_DIMENSION = 384
ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)

_TOKEN_RE = re.compile(r"[a-z0-9$]+(?:'[a-z]+)?", re.IGNORECASE)
_OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|changed my mind|ignore (?:my )?(?:earlier|previous|old))\b",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:please use your judgment|use your judgement|you decide|no preference)\b",
    re.IGNORECASE,
)
_EXPLORING_RE = re.compile(
    r"\b(?:still exploring|just browsing|not sure yet|open to ideas)\b",
    re.IGNORECASE,
)
_REQUIREMENT_RE = re.compile(
    r"\b(?:key requirement|what matters is|must have|need is|looking for)\b",
    re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(r"\bfor that,? what matters is\b", re.IGNORECASE)
_NO_ADDITIONAL_RE = re.compile(
    r"\bi don['\u2019]?t have an additional preference\b", re.IGNORECASE
)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def observable_evidence(messages: Sequence[str]) -> dict[str, bool]:
    """Return semantic evidence flags used by training and runtime safeguards."""

    joined = "\n".join(str(message) for message in messages)
    latest = str(messages[-1]) if messages else ""
    return {
        "override": bool(_OVERRIDE_RE.search(joined)),
        # "No additional preference" is an ordinary exhausted-slot response;
        # TechJam boundary evidence contains a stronger delegation phrase.
        "boundary": bool(_BOUNDARY_RE.search(joined))
        and not bool(_NO_ADDITIONAL_RE.search(latest)),
        "exploring": bool(_EXPLORING_RE.search(joined)),
        "requirement": bool(_REQUIREMENT_RE.search(joined)),
        "clarification": bool(_CLARIFICATION_RE.search(joined)),
        "no_additional": bool(_NO_ADDITIONAL_RE.search(joined)),
    }


def observable_intent(messages: Sequence[str]) -> str:
    """Deterministic definition of the trainable, observable target label."""

    evidence = observable_evidence(messages)
    if evidence["override"]:
        return "intent_override"
    if evidence["boundary"]:
        return "boundary"
    # Browsing becomes Buying after the customer supplies a concrete answer.
    if evidence["exploring"] and not evidence["clarification"]:
        return "browsing"
    return "buying"


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    probabilities: dict[str, float]
    confidence: float
    provisional: bool
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "provisional": self.provisional,
            "evidence": list(self.evidence),
        }


class IntentVectorizer:
    """Hash a conversation history into a small fixed-size numeric vector."""

    STRUCTURAL_NAMES = (
        "bias", "turn_count", "token_count", "latest_token_count",
        "asked_count", "unique_asked_count", "has_override", "latest_override",
        "has_boundary", "latest_boundary", "has_exploring", "has_requirement",
        "has_clarification", "latest_clarification", "has_no_additional",
    )

    def __init__(self, hash_dimension: int = DEFAULT_HASH_DIMENSION) -> None:
        if hash_dimension < 32:
            raise ValueError("hash_dimension must be at least 32")
        self.hash_dimension = int(hash_dimension)

    @property
    def dimension(self) -> int:
        return self.hash_dimension + len(self.STRUCTURAL_NAMES) + len(ASK_ATTRIBUTES)

    @staticmethod
    def _bucket(feature: str, dimension: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        return number % dimension, 1.0 if (number >> 63) == 0 else -1.0

    def vectorize(
        self,
        messages: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> np.ndarray:
        clean_messages = [str(message) for message in messages]
        vector = np.zeros(self.dimension, dtype=np.float32)
        all_tokens: list[str] = []
        for message_index, message in enumerate(clean_messages):
            items = _tokens(message)
            all_tokens.extend(items)
            recency = 1.0 if message_index == len(clean_messages) - 1 else 0.65
            features = [f"u:{item}" for item in items]
            features.extend(
                f"b:{first}_{second}" for first, second in zip(items, items[1:])
            )
            if message_index == len(clean_messages) - 1:
                features.extend(f"last:{item}" for item in items)
            for feature in features:
                bucket, sign = self._bucket(feature, self.hash_dimension)
                vector[bucket] += np.float32(sign * recency)

        norm = float(np.linalg.norm(vector[: self.hash_dimension]))
        if norm > 0:
            vector[: self.hash_dimension] /= norm

        evidence = observable_evidence(clean_messages)
        latest = clean_messages[-1] if clean_messages else ""
        latest_evidence = observable_evidence([latest])
        structural = (
            1.0,
            min(len(clean_messages), 10) / 10.0,
            min(len(all_tokens), 200) / 200.0,
            min(len(_tokens(latest)), 80) / 80.0,
            min(len(asked_attributes), 10) / 10.0,
            len({str(value).casefold() for value in asked_attributes}) / 10.0,
            float(evidence["override"]),
            float(latest_evidence["override"]),
            float(evidence["boundary"]),
            float(latest_evidence["boundary"]),
            float(evidence["exploring"]),
            float(evidence["requirement"]),
            float(evidence["clarification"]),
            float(latest_evidence["clarification"]),
            float(evidence["no_additional"]),
        )
        offset = self.hash_dimension
        vector[offset : offset + len(structural)] = structural
        offset += len(structural)
        asked = {str(value).strip().casefold() for value in asked_attributes}
        for index, attribute in enumerate(ASK_ATTRIBUTES):
            vector[offset + index] = float(attribute in asked)
        return vector

    def matrix(
        self,
        conversations: Sequence[Sequence[str]],
        asked_attributes: Sequence[Sequence[str]],
    ) -> np.ndarray:
        if len(conversations) != len(asked_attributes):
            raise ValueError("conversation and asked-attribute lengths differ")
        if not conversations:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([
            self.vectorize(messages, asks)
            for messages, asks in zip(conversations, asked_attributes)
        ])


class IntentClassifier:
    """Portable four-class softmax model with observability safeguards."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        *,
        hash_dimension: int = DEFAULT_HASH_DIMENSION,
        temperature: float = 1.0,
        confidence_threshold: float = 0.62,
    ) -> None:
        self.vectorizer = IntentVectorizer(hash_dimension)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        expected = (len(INTENTS), self.vectorizer.dimension)
        if self.weights.shape != expected or self.bias.shape != (len(INTENTS),):
            raise ValueError(f"model shapes must be {expected} and {(len(INTENTS),)}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size + self.bias.size)

    def predict_proba(
        self,
        messages: Sequence[str],
        asked_attributes: Sequence[str] = (),
        *,
        enforce_observability: bool = True,
    ) -> np.ndarray:
        vector = self.vectorizer.vectorize(messages, asked_attributes)
        logits = (self.weights @ vector + self.bias) / self.temperature
        evidence = observable_evidence(messages)
        if enforce_observability:
            # Boundary/override are event labels, not guesses about future
            # hidden simulator state. Their probability is zero pre-event.
            if not evidence["override"]:
                logits[INTENTS.index("intent_override")] = -1e9
            if not evidence["boundary"]:
                logits[INTENTS.index("boundary")] = -1e9
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def predict(
        self,
        messages: Sequence[str],
        asked_attributes: Sequence[str] = (),
    ) -> IntentPrediction:
        probabilities = self.predict_proba(messages, asked_attributes)
        index = int(np.argmax(probabilities))
        label = INTENTS[index]
        confidence = float(probabilities[index])
        flags = observable_evidence(messages)
        evidence = tuple(name for name, present in flags.items() if present)
        event_observed = flags["override"] or flags["boundary"]
        return IntentPrediction(
            label=label,
            probabilities={
                intent: float(probabilities[position])
                for position, intent in enumerate(INTENTS)
            },
            confidence=confidence,
            # Buying/Browsing cannot rule out a future boundary or override.
            provisional=(not event_observed) or confidence < self.confidence_threshold,
            evidence=evidence,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "model_version": MODEL_VERSION,
            "intents": list(INTENTS),
            "hash_dimension": self.vectorizer.hash_dimension,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("unsupported intent-classifier model version")
        if tuple(payload.get("intents", ())) != INTENTS:
            raise ValueError("intent vocabulary differs from runtime")
        return cls(
            np.asarray(payload["weights"], dtype=np.float32),
            np.asarray(payload["bias"], dtype=np.float32),
            hash_dimension=int(payload["hash_dimension"]),
            temperature=float(payload.get("temperature", 1.0)),
            confidence_threshold=float(payload.get("confidence_threshold", 0.62)),
        )


__all__ = [
    "ASK_ATTRIBUTES", "DEFAULT_HASH_DIMENSION", "INTENTS", "IntentClassifier",
    "IntentPrediction", "IntentVectorizer", "observable_evidence",
    "observable_intent",
]
