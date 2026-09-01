"""The full pipeline: typed state + LLM understanding + hybrid retrieval.

``LLMAgent`` layers four local-model components onto the measured sparse agent,
with every one of them optional and every one failing back to the deterministic
path underneath:

    message ─┬─ regex template parse            (exact, instant)
             └─ LLM extraction on fall-through  (deployment robustness)
                        │
                 typed constraint state
                        │
             ┌──────────┴───────────┐
        BM25 sparse            LLM query rewrite  ← + profile preference_tags
             │                      │
             │                 MiniLM dense
             └──────────┬───────────┘
                weighted RRF fusion
                        │
              constraint reranker → Top 10
                        │
                LLM customer reply

Why extraction is a *fall-through* rather than a replacement: the public
simulator emits fixed templates, and the regex parser reads them perfectly and
instantly. Sending those to a 1.5B model could only lose information. The LLM
earns its place on the messages the templates do not cover -- which is every
message in a real deployment.

Everything is switchable by environment variable so any configuration can be
scored through the official command without editing code:

    TECHJAM_LLM=0        disable all language-model components
    TECHJAM_EXTRACT=0    disable LLM extraction only
    TECHJAM_REWRITE=0    disable LLM query rewriting only
    TECHJAM_REPLY=0      disable LLM customer replies only
    TECHJAM_DENSE=0      disable dense retrieval (sparse-only baseline)
"""

from __future__ import annotations

import os
from pathlib import Path

from starter.agent import Agent, AgentConfig, _terms
from starter.constraint_reranker import Constraint
from starter.conversation_state import (
    ConstraintSource,
    ConversationState,
    MessageKind,
)
from starter.hybrid_retrieval import weighted_rrf
from starter.local_dense import LocalDenseRetriever
from starter.local_llm import LocalLLM, explicit_finish_intent


def _flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# Message kinds whose requirements the language model extracts. The remaining
# kinds are dialogue *acts*, not requirements: NO_PREFERENCE drives Boundary
# handling and OVERRIDE drives retraction, and both are detected structurally
# so a model can never silently drop a retraction.
EXTRACTABLE = frozenset({
    MessageKind.INITIAL,
    MessageKind.CLARIFICATION,
    MessageKind.FREE_TEXT,
})


class LLMConversationState(ConversationState):
    """Typed state whose requirements are extracted by a language model.

    The structural parser always classifies the dialogue act first, so
    retraction and refusal remain deterministic. By default, its constraints
    stand for benchmark templates, while unmatched ``FREE_TEXT`` falls through
    to grounded LLM extraction. An opt-in extraction-first mode sends all
    requirement-bearing messages through the model. A model or validation
    failure preserves the parser result; a valid empty extraction clears a
    parser guess for greetings and non-shopping chatter.
    """

    def __init__(
        self,
        user_profile: dict,
        llm: LocalLLM | None = None,
        extract_first: bool = True,
    ) -> None:
        super().__init__(user_profile)
        self.llm = llm
        self.extract_first = extract_first
        self.llm_extractions = 0
        self.llm_fallbacks = 0

    def observe(self, user_message, turn, asked_attribute=None):
        update = super().observe(user_message, turn, asked_attribute)
        if self.llm is None:
            return update
        if update.message.kind not in EXTRACTABLE:
            return update
        if not self.extract_first and update.message.kind is not MessageKind.FREE_TEXT:
            return update

        extracted = self.llm.extract(str(user_message))
        # ``None`` means the model failed validation and the parser must stand.
        # An empty list is a successful extraction of *no requirement* (for
        # example, "hello") and must clear the parser's free-text guess.
        if extracted is None:
            self.llm_fallbacks += 1  # parser's constraints stand
            return update

        # Replace the parser's constraints for this turn with the model's typed
        # ones. Only this turn's records are touched; accumulated state from
        # earlier turns is never disturbed.
        source = update.added[0].source if update.added else ConstraintSource.FREE_TEXT
        for record in update.added:
            if record in self.constraints:
                self.constraints.remove(record)
        update.added.clear()

        for attribute, value in extracted:
            record = self._add_constraint(
                value, source, turn, self._coerce_attribute(attribute)
            )
            if record is not None:
                update.added.append(record)
                self._supersede_no_preference(record.attribute, turn)
        self.llm_extractions += 1
        return update


class LLMAgent(Agent):
    """Sparse + dense retrieval with local-model understanding and replies."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        *,
        use_llm: bool | None = None,
        use_dense: bool | None = None,
        llm_model: str | None = None,
    ) -> None:
        super().__init__(catalog_path, config)

        self.use_llm = _flag("TECHJAM_LLM") if use_llm is None else use_llm
        self.use_extract = self.use_llm and _flag("TECHJAM_EXTRACT")
        self.use_rewrite = self.use_llm and _flag("TECHJAM_REWRITE")
        self.use_reply = self.use_llm and _flag("TECHJAM_REPLY")
        # Default: the parser owns the phrasings it recognises and the model
        # extracts from everything else. TECHJAM_EXTRACT_FIRST=1 puts the model
        # in front of every requirement-bearing message instead -- built, but
        # not the measured default.
        self.extract_first = _flag("TECHJAM_EXTRACT_FIRST", default=False)
        want_dense = _flag("TECHJAM_DENSE") if use_dense is None else use_dense

        self.llm: LocalLLM | None = None
        self.llm_unavailable_reason: str | None = None
        if self.use_llm:
            candidate = LocalLLM(model_name=llm_model) if llm_model else LocalLLM()
            self.llm = candidate if candidate.load() else None
            if self.llm is None:
                self.llm_unavailable_reason = candidate.error
                self.use_extract = self.use_rewrite = self.use_reply = False

        self.dense: LocalDenseRetriever | None = None
        self.dense_unavailable_reason: str | None = None
        if want_dense:
            candidate = LocalDenseRetriever(top_k=self.config.candidate_pool_size)
            self.dense = candidate if candidate.available else None
            if self.dense is None:
                self.dense_unavailable_reason = candidate.unavailable_reason

        self.last_diagnostics: dict = {}
        self._dialogue_history: dict[str, list[dict[str, str]]] = {}

    # ------------------------------------------------------------------ state
    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self._dialogue_history[session_id] = []
        state = self._sessions[session_id]
        if self.config.use_typed_state:
            state.structured = LLMConversationState(
                user_profile,
                self.llm if self.use_extract else None,
                extract_first=self.extract_first,
            )

    # -------------------------------------------------------------- profile
    @staticmethod
    def _profile_tags(profile: dict) -> list[str]:
        """Weak background preferences, used only to colour the dense query."""

        tags = profile.get("preference_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return [str(tag).strip() for tag in tags if str(tag).strip()][:6]

    # -------------------------------------------------------------- respond
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        usage_before = self.llm.usage() if self.llm is not None else {}
        state_before = self._sessions.get(session_id)
        extraction_count_before = getattr(
            getattr(state_before, "structured", None), "llm_extractions", 0
        )
        fallback_count_before = getattr(
            getattr(state_before, "structured", None), "llm_fallbacks", 0
        )
        reply = super().respond(session_id, user_message, turn, top_k)
        state = self._sessions.get(session_id)
        if state is None or state.structured is None:
            return reply

        active = [
            (record.attribute.value, record.value)
            for record in state.structured.active_constraints()
        ]
        tags = self._profile_tags(state.user_profile)
        extraction_count = getattr(state.structured, "llm_extractions", 0)
        fallback_count = getattr(state.structured, "llm_fallbacks", 0)
        diagnostics: dict = {
            "constraints": active,
            "profile_tags": tags,
            "llm_extractions": extraction_count,
            "llm_fallbacks": fallback_count,
            "extraction_route": (
                "llm_fallthrough"
                if extraction_count > extraction_count_before
                else "llm_rejected_parser_fallback"
                if fallback_count > fallback_count_before
                else "deterministic_parser"
            ),
        }

        # ---- dense route, fused with the sparse pool the parent produced ----
        if self.dense is not None and active:
            query_text = None
            if self.use_rewrite and self.llm is not None:
                query_text = self.llm.rewrite(active, tags)
                diagnostics["rewritten_query"] = query_text
            semantic = [
                {"attribute": attribute, "value": value, "hard": True}
                for attribute, value in active
            ]
            dense_result = self.dense.search(
                semantic,
                top_k=self.config.candidate_pool_size,
                query_text=query_text,
            )
            sparse_ids = list(state.last_retrieval.get("selected_sparse", ()))
            if dense_result.ranked_ids and sparse_ids:
                sparse_weight, dense_weight = self.config.hybrid_config.fusion_weights(
                    semantic
                )
                fused = weighted_rrf(
                    sparse_ids,
                    list(dense_result.ranked_ids),
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    rrf_k=self.config.hybrid_config.rrf_k,
                    limit=self.config.candidate_pool_size,
                )
                diagnostics["dense_count"] = len(dense_result.ranked_ids)
                diagnostics["fusion"] = [sparse_weight, dense_weight]
                if self.reranker is not None:
                    constraints = [
                        Constraint(attribute, value, hard=True)
                        for attribute, value in active
                        if _terms(value)
                    ]
                    reply["recommendations"] = [
                        candidate.recommendation()
                        for candidate in self.reranker.rerank(
                            fused, constraints, top_k=top_k
                        )
                    ]

        # ---- dialogue-aware customer response ----
        if self.use_reply and self.llm is not None:
            history = list(self._dialogue_history.get(session_id, ()))
            history.append({"role": "user", "content": str(user_message)})
            required_action = (
                "finish" if explicit_finish_intent(user_message) else None
            )
            plan = self.llm.reply(
                active,
                len(reply.get("recommendations") or []),
                conversation=history,
                proposed_question=(
                    None
                    if required_action == "finish"
                    else str(reply.get("message") or "") or None
                ),
                required_action=required_action,
            )
            if plan and plan["action"] == "finish" and required_action != "finish":
                # A generative model may not terminate a session without an
                # explicit high-precision cue in the latest customer turn.
                diagnostics["model_finish_rejected"] = True
                plan = None
            if required_action == "finish" and not plan:
                # Protocol-safe fallback: an explicit terminal cue must never
                # be followed by another sales question just because generation
                # or JSON validation failed.
                plan = {
                    "action": "finish",
                    "reply": "Glad I could help—enjoy your selection!",
                }
                diagnostics["reply_fallback"] = "explicit_finish"
            if plan:
                sentence = plan["reply"]
                action = plan["action"]
                diagnostics["dialogue_action"] = action
                diagnostics["generated_reply"] = True
                if action == "finish":
                    # The base agent selected a question before the dialogue
                    # controller ran. Remove that unshown action so a later
                    # resumed turn does not inherit false question history.
                    selected = reply.get("ask_attribute")
                    if selected and state.asked_attributes:
                        state.asked_attributes.pop()
                    state.last_asked_attribute = (
                        state.asked_attributes[-1] if state.asked_attributes else None
                    )
                    reply["message"] = sentence
                    reply["ask_attribute"] = None
                    diagnostics["conversation_complete"] = True
                else:
                    question = reply.get("message") or ""
                    # The protocol question stays deterministic and therefore
                    # cannot disagree with ask_attribute.
                    reply["message"] = f"{sentence} {question}".strip()
                    diagnostics["customer_confirmation"] = sentence
                    diagnostics["conversation_complete"] = False

        final_message = str(reply.get("message") or "")
        dialogue = self._dialogue_history.setdefault(session_id, [])
        dialogue.append({"role": "user", "content": str(user_message)})
        if final_message:
            dialogue.append({"role": "assistant", "content": final_message})

        if self.llm is not None:
            usage_after = self.llm.usage()
            reply["usage"] = {
                "prompt_tokens": usage_after["prompt_tokens"]
                - int(usage_before.get("prompt_tokens", 0)),
                "completion_tokens": usage_after["completion_tokens"]
                - int(usage_before.get("completion_tokens", 0)),
            }
            diagnostics["llm_calls"] = usage_after["calls"] - int(
                usage_before.get("calls", 0)
            )
            diagnostics["llm_rejects"] = usage_after["rejects"] - int(
                usage_before.get("rejects", 0)
            )
            diagnostics["llm_generate_ms"] = round(
                usage_after["generate_ms"]
                - float(usage_before.get("generate_ms", 0.0)),
                1,
            )
        self.last_diagnostics = diagnostics
        return reply

    def close(self) -> None:
        if self.dense is not None:
            self.dense.close()
        super().close()
