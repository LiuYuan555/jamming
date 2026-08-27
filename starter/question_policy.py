from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


VALID_ATTRIBUTES = frozenset(
    {
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
    }
)


ATTRIBUTE_MESSAGES = {
    "material": "Do you have a preferred material?",
    "color": "Which colors would you prefer?",
    "feature": "Which features or qualities matter most to you?",
    "other": "What other preferences should I consider?",
}


@dataclass(frozen=True)
class QuestionDecision:
    """One human-facing question and its evaluator-facing control signal."""

    message: str
    ask_attribute: str | None


@dataclass(frozen=True)
class QuestionPolicyConfig:
    """Configuration for an independently selectable clarification policy."""

    name: str = "always_other"


class QuestionPolicy:
    """Choose clarification attributes without changing retrieval behavior.

    The object is stateless: conversation state remains owned by the agent and
    is supplied through ``asked_attributes``. This lets one policy instance be
    safely shared by all evaluator sessions.
    """

    _SEQUENCES: dict[str, tuple[tuple[str | None, ...], str | None]] = {
        "none": ((None,), None),
        "always_other": (("other",), "other"),
        "always_feature": (("feature",), "feature"),
        "always_material": (("material",), "material"),
        "feature_first": (("feature", "material", "color", "other"), "other"),
        "other_first": (("other", "feature", "material", "color"), "other"),
    }

    def __init__(self, config: QuestionPolicyConfig | str = QuestionPolicyConfig()) -> None:
        if isinstance(config, str):
            config = QuestionPolicyConfig(name=config)
        if config.name not in self._SEQUENCES:
            choices = ", ".join(sorted(self._SEQUENCES))
            raise ValueError(f"unknown question policy {config.name!r}; choose one of: {choices}")
        self.config = config

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._SEQUENCES))

    def decide(self, asked_attributes: Sequence[str]) -> QuestionDecision:
        """Return the next clarification decision for one session.

        A fixed policy repeats its sole choice. A sequence advances according
        to the number of earlier clarification questions and repeats its final
        choice after the sequence is exhausted.
        """

        sequence, fallback = self._SEQUENCES[self.config.name]
        question_number = len(asked_attributes)
        attribute = sequence[question_number] if question_number < len(sequence) else fallback
        if attribute is None:
            return QuestionDecision(
                message="Here are the closest matches I found.",
                ask_attribute=None,
            )
        if attribute not in VALID_ATTRIBUTES:  # Defensive guard for future edits.
            raise RuntimeError(f"policy selected invalid ask_attribute: {attribute}")
        return QuestionDecision(
            message=ATTRIBUTE_MESSAGES[attribute],
            ask_attribute=attribute,
        )
