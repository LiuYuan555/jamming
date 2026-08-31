"""Local grounded response generation for the demo console.

The scored agent does not use this module, and must not: the evaluator reads
``ask_attribute`` and discards ``message`` entirely, so generation cannot move
TechnicalScore -- it can only add latency and timeout risk to a scoring run.
This exists to make the assistant's replies presentable in a demo, and it is
opt-in (``python -m ui.server --llm``).

Grounding is enforced by validation, not by prompting:

* a ``parent_asin`` the model did not receive is dropped;
* a reason that does not match a constraint the customer actually stated is
  dropped;
* any parse or schema failure returns ``None`` so the caller falls back to the
  deterministic template.

A small model cannot therefore invent a product or a requirement, only fail to
produce one -- which degrades to the existing behaviour.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_REPLY_CHARS = 400
MAX_ITEMS = 5
ASIN_RE = re.compile(r"B0[A-Z0-9]{8}")

SYSTEM = (
    "You are a helpful shopping assistant. You reply with a single JSON object "
    "and nothing else. No markdown, no code fences, no commentary."
)


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model completion."""

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


def build_prompt(
    constraints: Sequence[tuple[str, str]],
    products: Sequence[dict],
    ask_attribute: str | None,
) -> str:
    lines = ["Customer requirements so far:"]
    if constraints:
        for attribute, value in constraints:
            lines.append(f"- {attribute}: {value}")
    else:
        lines.append("- (none stated yet)")

    # Deliberately NOT given the product list.  Per-product evidence is
    # produced by exact catalog matching, which is instant and cannot
    # hallucinate; asking a 1.5B model to describe products it can only see as
    # titles invents attributes ("features a black color") and recites raw
    # ASINs at the customer.  The model's only job is natural phrasing.
    lines.append("")
    lines.append(f"You are showing them {len(products)} matching products.")
    lines.append("")
    lines.append('Return JSON: {"reply": "..."}')
    lines.append(
        '  "reply": ONE short sentence, spoken directly TO the customer using '
        '"you", confirming what you understood they are looking for.'
    )
    lines.append("")
    lines.append(
        'Example of a complete valid answer:'
        ' {"reply": "You\'re after waterproof hiking boots in a wide fit."}'
    )
    # The clarification question is appended by the caller from the agent's own
    # template.  It drives `ask_attribute` in the API response, so it has to
    # match that field exactly -- a model free to phrase it could ask about
    # colour while the structured field says material.
    lines.append("")
    lines.append(
        'The reply must begin with the words "You\'re looking for" and restate '
        "ONLY the requirements listed above, as one natural sentence. Add no "
        "detail, explanation, or question. Never mention a specific product, "
        "brand, or product code. Keep it under 25 words."
    )
    return "\n".join(lines)


class ResponseGenerator:
    """Lazily loaded local causal LM. Never raises into the request path."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 72,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.error: str | None = None
        self._model = None
        self._tokenizer = None
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Prefer the local cache. from_pretrained otherwise contacts the
            # HF hub to check for updated files on every start, which adds a
            # network round-trip and can hang outright on a flaky connection.
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
                        raise  # already tried the network; report the failure
            self._model.eval()
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def ready(self) -> bool:
        return self._model is not None

    def generate(
        self,
        constraints: Sequence[tuple[str, str]],
        products: Sequence[dict],
        ask_attribute: str | None,
    ) -> dict | None:
        """Return {"reply": str, "reasons": {asin: [str, ...]}} or None."""

        # With nothing stated there is nothing to confirm, and a small model
        # asked to confirm an empty list invents requirements wholesale
        # ("jackets in your size and style"). Let the template handle turn one.
        if not products or not constraints or not self.load():
            return None
        try:
            import torch

            prompt = build_prompt(constraints, products, ask_attribute)
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy: reproducible, and better JSON
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            completion = output[0][inputs["input_ids"].shape[1] :]
            self.last_usage = {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(completion.shape[0]),
            }
            raw = self._tokenizer.decode(completion, skip_special_tokens=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None

        return self._validate(raw, constraints, products)

    @staticmethod
    def _validate(
        raw: str,
        constraints: Sequence[tuple[str, str]],
        products: Sequence[dict],
    ) -> dict | None:
        """Drop anything the model was not given. Grounding lives here."""

        # Degradation ladder: valid JSON -> a clean bare sentence -> template.
        # Small models often answer the question correctly while ignoring the
        # envelope; discarding that would waste a usable reply.
        parsed = _extract_json(raw)
        if parsed:
            reply = parsed.get("reply")
        elif "{" not in raw and "}" not in raw:
            reply = raw
        else:
            return None
        if not isinstance(reply, str) or not reply.strip():
            return None
        reply = " ".join(reply.split())[:MAX_REPLY_CHARS]
        # A reply naming a product code is reciting the candidate list at the
        # customer rather than talking to them; fall back to the template.
        if ASIN_RE.search(reply):
            return None

        known_ids = {str(p["parent_asin"]) for p in products}
        # Map normalized constraint text back to the customer's exact wording,
        # so a paraphrase by the model cannot become a displayed "reason".
        known_values = {_norm(value): value for _, value in constraints}

        reasons: dict[str, list[str]] = {}
        for item in parsed.get("items") or []:
            if not isinstance(item, dict):
                continue
            asin = str(item.get("parent_asin") or "")
            if asin not in known_ids:
                continue
            kept: list[str] = []
            for because in item.get("because") or []:
                exact = known_values.get(_norm(because))
                if exact and exact not in kept:
                    kept.append(exact)
            if kept:
                reasons[asin] = kept[:3]

        return {"reply": reply, "reasons": reasons}
