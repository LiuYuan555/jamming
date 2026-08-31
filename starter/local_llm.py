"""Local language model for extraction, query rewriting and customer replies.

Model choice: ``Qwen/Qwen2.5-1.5B-Instruct`` -- Apache-2.0, free, ~3.1GB at
fp16, runs on a consumer laptop GPU at ~13 tok/s and on CPU without one. It is
loaded once and shared by all three tasks below, so the marginal cost of a task
is a short prompt rather than another model.

Three jobs, each deliberately narrow:

1. :meth:`LocalLLM.extract`  -- turn a free-text customer message into typed
   constraints. The public simulator emits fixed templates that a regex parses
   perfectly, so this exists for the deployment case the benchmark does not
   test: a real shopper writes "something warm for hiking, nothing leather".
2. :meth:`LocalLLM.rewrite`  -- turn active constraints *and the customer's
   profile preference tags* into one compact semantic query for dense
   retrieval.
3. :meth:`LocalLLM.reply`    -- one short sentence to the customer, so the
   agent never answers with a bare list of products.

Every method validates its own output and returns ``None`` on any failure, so
the caller falls back to deterministic behaviour. A small model can therefore
fail to help, but it cannot corrupt the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ASIN_RE = re.compile(r"B0[A-Z0-9]{8}")
WORD_RE = re.compile(r"[a-z0-9]+")
# Excluded from the grounding overlap test: matching on "for" or "the" would
# let any invented value through.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "for", "in", "is", "it", "of", "on", "or",
    "something", "the", "to", "too", "with", "i", "need", "want", "nothing",
    "looking", "me", "my", "that", "this", "some", "would", "like",
})

# The evaluator's published enum. Anything outside it is silently coerced to
# "other" by the scorer, so extraction must never invent a facet name.
ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

EXTRACT_SYSTEM = (
    "You extract shopping requirements from a customer message. "
    "You reply with one JSON object and nothing else."
)

EXTRACT_INSTRUCTION = """Extract the concrete product requirements the customer states.

Return JSON: {"constraints": [{"attribute": "...", "value": "..."}]}

"attribute" must be one of: category, material, color, size, style, brand, budget, feature, use_case
"value" must be words the customer actually used. Never invent a requirement.
If the message states nothing concrete, return {"constraints": []}.

Customer message:
"""

REWRITE_SYSTEM = (
    "You write short product search queries. Reply with the query text only."
)

REWRITE_INSTRUCTION = """Write one product search query for a clothing catalog.

Output KEYWORDS ONLY, like a search box entry.
Never write a question, a sentence, or the words "find", "where", "catalog".
Use every stated requirement verbatim -- these are exact catalog terms.
The profile hints are weak background preferences: append them only if they do
not conflict with a stated requirement, and never let them replace one.
At most 20 words.

Example input:  requirements = category: running shoes, size: wide
                profile hints = breathable
Example output: running shoes wide breathable

"""

REPLY_SYSTEM = (
    "You are a helpful shopping assistant. Reply with one JSON object and "
    "nothing else."
)


def _words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a completion."""

    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


class LocalLLM:
    """One shared local model behind three narrow, validated tasks."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.error: str | None = None
        self._model = None
        self._tokenizer = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.rejects = 0
        self._cache: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            # Local cache first: from_pretrained otherwise contacts the hub on
            # every start, which is slow and can hang without network.
            for local_only in (True, False):
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, local_files_only=local_only
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, dtype=dtype, local_files_only=local_only
                    ).to(self.device)
                    break
                except Exception:
                    if not local_only:
                        raise
            self._model.eval()
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str | None:
        key = (system, user)
        if key in self._cache:
            return self._cache[key]
        if not self.load():
            return None
        try:
            import torch

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy: reproducible, and better JSON
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            completion = output[0][inputs["input_ids"].shape[1] :]
            self.calls += 1
            self.prompt_tokens += int(inputs["input_ids"].shape[1])
            self.completion_tokens += int(completion.shape[0])
            result = self._tokenizer.decode(completion, skip_special_tokens=True)
            self._cache[key] = result
            return result
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None

    # --------------------------------------------------------------- extract
    def extract(self, message: str) -> list[tuple[str, str]] | None:
        """Return [(attribute, value), ...] the customer actually stated."""

        if not message.strip():
            return None
        raw = self._generate(EXTRACT_SYSTEM, EXTRACT_INSTRUCTION + message.strip(), 160)
        if raw is None:
            return None
        parsed = extract_json(raw)
        if not parsed:
            self.rejects += 1
            return None

        source = _words(message)
        found: list[tuple[str, str]] = []
        for item in parsed.get("constraints") or []:
            if not isinstance(item, dict):
                continue
            attribute = str(item.get("attribute") or "").strip().lower()
            value = " ".join(str(item.get("value") or "").split())
            if attribute not in ATTRIBUTES or not value:
                continue
            # Grounding: every extracted value must share a content word with
            # the message. Requiring full containment rejected legitimate
            # paraphrases; requiring nothing let a 1.5B turn "warm, nothing too
            # heavy" into material:wool and size:lightweight, both invented.
            value_words = _words(value) - STOPWORDS
            if not value_words or not (value_words & source):
                self.rejects += 1
                continue
            if (attribute, value) not in found:
                found.append((attribute, value))
        return found or None

    # --------------------------------------------------------------- rewrite
    def rewrite(
        self,
        constraints: Sequence[tuple[str, str]],
        profile_tags: Sequence[str] = (),
    ) -> str | None:
        """Compose one dense-retrieval query from constraints plus profile."""

        if not constraints and not profile_tags:
            return None
        lines = ["Stated requirements:"]
        lines.extend(f"- {attribute}: {value}" for attribute, value in constraints)
        if not constraints:
            lines.append("- (none yet)")
        if profile_tags:
            lines.append("")
            lines.append("Profile hints (weak): " + ", ".join(profile_tags))
        raw = self._generate(REWRITE_SYSTEM, REWRITE_INSTRUCTION + "\n".join(lines), 60)
        if raw is None:
            return None

        cleaned = " ".join(raw.replace("\n", " ").split()).strip(' "\'`')
        if not cleaned:
            self.rejects += 1
            return None
        # Guard: the query must stay anchored to the customer's own vocabulary.
        allowed = set()
        for _, value in constraints:
            allowed |= _words(value)
        for tag in profile_tags:
            allowed |= _words(tag)
        # Only novel *content* words matter. Counting connectives ("made of",
        # "with", "and") as invented rejected perfectly good queries.
        invented = _words(cleaned) - allowed - STOPWORDS
        if len(invented) > 6:
            self.rejects += 1
            return None
        return cleaned

    # ----------------------------------------------------------------- reply
    def reply(
        self,
        constraints: Sequence[tuple[str, str]],
        result_count: int,
    ) -> str | None:
        """One short sentence confirming what the agent understood."""

        if not constraints:
            return None
        lines = ["Customer requirements:"]
        lines.extend(f"- {attribute}: {value}" for attribute, value in constraints)
        lines.append("")
        lines.append(f"You are showing them {result_count} matching products.")
        lines.append("")
        lines.append('Return JSON: {"reply": "..."}')
        lines.append(
            '  "reply": ONE sentence, spoken to the customer, beginning with '
            '"You\'re looking for", restating ONLY the requirements above.'
        )
        lines.append("")
        lines.append(
            'Example: {"reply": "You\'re looking for waterproof hiking boots in '
            'a wide fit."}'
        )
        lines.append(
            "Do not ask a question. Never mention a product, brand or product "
            "code. Under 25 words."
        )
        raw = self._generate(REPLY_SYSTEM, "\n".join(lines), 72)
        if raw is None:
            return None

        parsed = extract_json(raw)
        if parsed:
            text = parsed.get("reply")
        elif "{" not in raw and "}" not in raw:
            text = raw  # a clean bare sentence is still usable
        else:
            self.rejects += 1
            return None
        if not isinstance(text, str) or not text.strip():
            self.rejects += 1
            return None
        # A bare sentence often arrives wrapped in the quotes the model would
        # have used inside JSON; they must not reach the customer.
        text = " ".join(text.split()).strip(" \"'`")[:400]
        if ASIN_RE.search(text):
            self.rejects += 1  # reciting candidate ids at the customer
            return None
        return text

    # ----------------------------------------------------------------- usage
    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "rejects": self.rejects,
        }
