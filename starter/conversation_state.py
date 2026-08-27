"""Typed, auditable conversation state for the TechJam shopping agent.

This module intentionally has no retrieval or question-policy dependencies.  It
turns the public simulator's messages into a small structured state that a
retriever can consume, while retaining the original messages and every state
transition for debugging.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Attribute(str, Enum):
    CATEGORY = "category"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    STYLE = "style"
    BRAND = "brand"
    BUDGET = "budget"
    FEATURE = "feature"
    USE_CASE = "use_case"
    OTHER = "other"


class ConstraintStatus(str, Enum):
    ACTIVE = "active"
    RETRACTED = "retracted"


class ConstraintSource(str, Enum):
    INITIAL_CATEGORY = "initial_category"
    INITIAL_REQUIREMENT = "initial_requirement"
    INITIAL_PREFERENCE = "initial_preference"
    CLARIFICATION = "clarification"
    OVERRIDE = "override"
    FREE_TEXT = "free_text"


class MessageKind(str, Enum):
    INITIAL = "initial"
    CLARIFICATION = "clarification"
    NO_PREFERENCE = "no_preference"
    OVERRIDE = "override"
    RETRY_PROMPT = "retry_prompt"
    FREE_TEXT = "free_text"


@dataclass
class MessageRecord:
    turn: int
    text: str
    kind: MessageKind
    asked_attribute: Attribute | None = None


@dataclass
class ConstraintRecord:
    value: str
    attribute: Attribute
    source: ConstraintSource
    turn: int
    status: ConstraintStatus = ConstraintStatus.ACTIVE
    retracted_turn: int | None = None
    retraction_reason: str | None = None

    @property
    def normalized_value(self) -> str:
        return _normalize(self.value)


@dataclass
class NoPreferenceRecord:
    attribute: Attribute
    turn: int
    message: str
    status: ConstraintStatus = ConstraintStatus.ACTIVE
    retracted_turn: int | None = None


@dataclass
class StateUpdate:
    """The changes produced by one call to :meth:`ConversationState.observe`."""

    message: MessageRecord
    added: list[ConstraintRecord] = field(default_factory=list)
    retracted: list[ConstraintRecord] = field(default_factory=list)
    no_preferences_added: list[NoPreferenceRecord] = field(default_factory=list)


MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange",
)
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")

_BUYING_RE = re.compile(
    r"^i(?:'| a)?m looking for (?P<category>.+?)\.\s*"
    r"a key requirement is:\s*(?P<requirement>.+?)\.?$",
    re.IGNORECASE,
)
_BROWSING_RE = re.compile(
    r"^i(?:'| a)?m looking for (?P<category>.+?),\s*but i(?:'| a)?m still exploring\.?$",
    re.IGNORECASE,
)
_INITIAL_PREFERENCE_RE = re.compile(
    r"^i(?:'| a)?m looking for (?P<category>.+?)\.\s*(?P<preference>.+?)\.?$",
    re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(
    r"^for that,\s*what matters is:\s*(?P<payload>.+?)\.?$",
    re.IGNORECASE,
)
_OVERRIDE_RE = re.compile(
    r"^actually,\s*(?P<retraction>.+?)\.\s*what i need is:\s*(?P<payload>.+?)\.?$",
    re.IGNORECASE,
)
_NO_PREFERENCE_RE = re.compile(
    r"^i don['’]t have (?:an additional |a )?preference for\s+"
    r"(?P<attribute>[a-z_]+)(?:;[^.]*)?\.?$",
    re.IGNORECASE,
)
_RETRY_PREFIX = "those options are not quite right"
_VALID_ATTRIBUTES = {attribute.value: attribute for attribute in Attribute}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n-;,.")


def _normalize(value: str) -> str:
    return _clean(value).casefold()


def _contains_word(text: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def classify_constraint(value: str) -> Attribute:
    """Classify a disclosed constraint using the evaluator's public taxonomy."""

    lowered = value.casefold()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return Attribute.BUDGET
    if _contains_word(lowered, MATERIALS):
        return Attribute.MATERIAL
    if "color" in lowered or _contains_word(lowered, COLORS):
        return Attribute.COLOR
    if _contains_word(lowered, SIZE_WORDS):
        return Attribute.SIZE
    if _contains_word(lowered, STYLE_WORDS):
        return Attribute.STYLE
    if _contains_word(lowered, USE_CASE_WORDS):
        return Attribute.USE_CASE
    return Attribute.FEATURE


class ConversationState:
    """State for one shopping session.

    ``observe`` is the integration entry point.  The optional
    ``asked_attribute`` is useful when the customer reply itself omits the
    attribute name; positive constraints are still classified from their text.
    """

    def __init__(self, user_profile: dict) -> None:
        self.user_profile = copy.deepcopy(user_profile)
        self.messages: list[MessageRecord] = []
        self.constraints: list[ConstraintRecord] = []
        self.no_preference_history: list[NoPreferenceRecord] = []

    @property
    def no_preference_attributes(self) -> set[Attribute]:
        return {
            record.attribute
            for record in self.no_preference_history
            if record.status is ConstraintStatus.ACTIVE
        }

    def active_constraints(
        self, attribute: Attribute | str | None = None
    ) -> list[ConstraintRecord]:
        wanted = self._coerce_attribute(attribute)
        return [
            constraint
            for constraint in self.constraints
            if constraint.status is ConstraintStatus.ACTIVE
            and (wanted is None or constraint.attribute is wanted)
        ]

    def retrieval_text(self) -> str:
        """Return active, de-duplicated constraint text in disclosure order."""

        values: list[str] = []
        seen: set[str] = set()
        for constraint in self.active_constraints():
            normalized = constraint.normalized_value
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(constraint.value)
        return " ".join(values)

    def observe(
        self,
        user_message: str,
        turn: int,
        asked_attribute: Attribute | str | None = None,
    ) -> StateUpdate:
        if turn < 1:
            raise ValueError("turn must be at least 1")
        asked = self._coerce_attribute(asked_attribute)
        text = str(user_message)
        stripped = text.strip()
        lowered = stripped.casefold()

        no_preference = _NO_PREFERENCE_RE.fullmatch(stripped)
        if no_preference:
            explicit = self._coerce_attribute(no_preference.group("attribute"))
            attribute = explicit or asked
            message = MessageRecord(turn, text, MessageKind.NO_PREFERENCE, asked)
            update = StateUpdate(message)
            if attribute is not None:
                record = NoPreferenceRecord(attribute, turn, text)
                self.no_preference_history.append(record)
                update.no_preferences_added.append(record)
            self.messages.append(message)
            return update

        if lowered.startswith(_RETRY_PREFIX):
            message = MessageRecord(turn, text, MessageKind.RETRY_PROMPT, asked)
            self.messages.append(message)
            return StateUpdate(message)

        override = _OVERRIDE_RE.fullmatch(stripped)
        if override:
            message = MessageRecord(turn, text, MessageKind.OVERRIDE, asked)
            update = StateUpdate(message)
            update.retracted.extend(
                self._apply_retraction(override.group("retraction"), turn)
            )
            record = self._add_constraint(
                override.group("payload"), ConstraintSource.OVERRIDE, turn
            )
            if record is not None:
                update.added.append(record)
                self._supersede_no_preference(record.attribute, turn)
            self.messages.append(message)
            return update

        clarification = _CLARIFICATION_RE.fullmatch(stripped)
        if clarification:
            message = MessageRecord(turn, text, MessageKind.CLARIFICATION, asked)
            update = StateUpdate(message)
            for value in clarification.group("payload").split(";"):
                record = self._add_constraint(
                    value, ConstraintSource.CLARIFICATION, turn
                )
                if record is not None:
                    update.added.append(record)
                    self._supersede_no_preference(record.attribute, turn)
            self.messages.append(message)
            return update

        buying = _BUYING_RE.fullmatch(stripped)
        if buying:
            message = MessageRecord(turn, text, MessageKind.INITIAL, asked)
            update = StateUpdate(message)
            self._append_if_new(
                update,
                buying.group("category"),
                Attribute.CATEGORY,
                ConstraintSource.INITIAL_CATEGORY,
                turn,
            )
            self._append_if_new(
                update,
                buying.group("requirement"),
                None,
                ConstraintSource.INITIAL_REQUIREMENT,
                turn,
            )
            self.messages.append(message)
            return update

        browsing = _BROWSING_RE.fullmatch(stripped)
        if browsing:
            message = MessageRecord(turn, text, MessageKind.INITIAL, asked)
            update = StateUpdate(message)
            self._append_if_new(
                update,
                browsing.group("category"),
                Attribute.CATEGORY,
                ConstraintSource.INITIAL_CATEGORY,
                turn,
            )
            self.messages.append(message)
            return update

        initial_preference = _INITIAL_PREFERENCE_RE.fullmatch(stripped)
        if initial_preference:
            message = MessageRecord(turn, text, MessageKind.INITIAL, asked)
            update = StateUpdate(message)
            self._append_if_new(
                update,
                initial_preference.group("category"),
                Attribute.CATEGORY,
                ConstraintSource.INITIAL_CATEGORY,
                turn,
            )
            self._append_if_new(
                update,
                initial_preference.group("preference"),
                None,
                ConstraintSource.INITIAL_PREFERENCE,
                turn,
            )
            self.messages.append(message)
            return update

        # Conservative fallback: retain an unknown meaningful utterance as one
        # soft feature rather than inventing several structured constraints.
        message = MessageRecord(turn, text, MessageKind.FREE_TEXT, asked)
        update = StateUpdate(message)
        record = self._add_constraint(stripped, ConstraintSource.FREE_TEXT, turn)
        if record is not None:
            update.added.append(record)
            self._supersede_no_preference(record.attribute, turn)
        self.messages.append(message)
        return update

    def _append_if_new(
        self,
        update: StateUpdate,
        value: str,
        attribute: Attribute | None,
        source: ConstraintSource,
        turn: int,
    ) -> None:
        record = self._add_constraint(value, source, turn, attribute)
        if record is not None:
            update.added.append(record)

    def _add_constraint(
        self,
        value: str,
        source: ConstraintSource,
        turn: int,
        attribute: Attribute | None = None,
    ) -> ConstraintRecord | None:
        cleaned = _clean(value)
        normalized = _normalize(cleaned)
        if not normalized:
            return None
        for existing in self.active_constraints():
            if existing.normalized_value == normalized:
                return None
        record = ConstraintRecord(
            value=cleaned,
            attribute=attribute or classify_constraint(cleaned),
            source=source,
            turn=turn,
        )
        self.constraints.append(record)
        return record

    def _apply_retraction(self, phrase: str, turn: int) -> list[ConstraintRecord]:
        normalized = _normalize(phrase)
        targets: list[ConstraintRecord] = []

        # This is the exact public-simulator meaning: the original preference
        # is replaced, while category and clarification answers stay valid.
        if re.search(r"\b(?:earlier|previous|old) preference\b", normalized):
            candidates = [
                item for item in self.active_constraints()
                if item.source is ConstraintSource.INITIAL_PREFERENCE
            ]
            if candidates:
                targets = [candidates[-1]]
        else:
            # Support explicit retractions such as "ignore cotton" without
            # guessing when the phrase cannot be matched to a known value.
            explicit = re.sub(
                r"^(?:please\s+)?ignore\s+(?:my\s+)?", "", normalized
            ).strip()
            targets = [
                item for item in self.active_constraints()
                if explicit and (
                    explicit == item.normalized_value
                    or explicit in item.normalized_value
                    or item.normalized_value in explicit
                )
            ]

        for target in targets:
            target.status = ConstraintStatus.RETRACTED
            target.retracted_turn = turn
            target.retraction_reason = _clean(phrase)
        return targets

    def _supersede_no_preference(self, attribute: Attribute, turn: int) -> None:
        for record in self.no_preference_history:
            if (
                record.status is ConstraintStatus.ACTIVE
                and record.attribute in {attribute, Attribute.OTHER}
            ):
                record.status = ConstraintStatus.RETRACTED
                record.retracted_turn = turn

    @staticmethod
    def _coerce_attribute(value: Attribute | str | None) -> Attribute | None:
        if isinstance(value, Attribute):
            return value
        if isinstance(value, str):
            return _VALID_ATTRIBUTES.get(value.strip().casefold())
        return None


__all__ = [
    "Attribute",
    "ConstraintRecord",
    "ConstraintSource",
    "ConstraintStatus",
    "ConversationState",
    "MessageKind",
    "MessageRecord",
    "NoPreferenceRecord",
    "StateUpdate",
    "classify_constraint",
]
