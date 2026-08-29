"""Build counterfactual training rows for an ``ask_attribute`` policy.

The policy input is deliberately limited to state that the agent can observe at
runtime. Hidden intent cards and target products are used only by the offline
simulator to generate customer replies and score the downstream retrieval.

Each output row has three explicit namespaces:

``model_input``
    Safe-to-train-on state features plus one candidate action.
``label``
    The rank and score observed after taking that action.
``metadata``
    A hashed product group and its deterministic split. This namespace must not
    be passed to a model.

This first prototype uses one-step counterfactual outcomes with a small rank@100
shaping term. It follows an ``other`` reference rollout to collect later states;
it does not yet solve the complete action tree by dynamic programming.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    behavior_for,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent
from starter.conversation_state import ConstraintSource, MessageKind


ACTION_ORDER = (
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
SCENARIO_CYCLE = (
    "buying",
    "browsing",
    "buying",
    "browsing",
    "intent_override",
    "buying",
    "browsing",
    "buying",
    "browsing",
    "intent_override",
    "buying",
    "browsing",
    "buying",
    "browsing",
    "intent_override",
    "buying",
    "browsing",
    "buying",
    "browsing",
    "boundary",
)
SCHEMA_VERSION = "question-policy-counterfactual-v1"

if set(ACTION_ORDER) != ALLOWED_ATTRIBUTES:
    raise RuntimeError("ACTION_ORDER must enumerate the evaluator's allowed attributes")


def _stable_fraction(value: str, seed: str) -> float:
    digest = hashlib.blake2b(
        f"{seed}\0{value}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def split_for_parent_asin(
    parent_asin: str,
    *,
    seed: str = "techjam-question-policy-v1",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> str:
    """Return a deterministic product-disjoint train/validation/test split."""

    if not math.isfinite(train_ratio) or not math.isfinite(validation_ratio):
        raise ValueError("split ratios must be finite")
    if train_ratio < 0 or validation_ratio < 0 or train_ratio + validation_ratio > 1:
        raise ValueError("split ratios must be non-negative and sum to at most one")
    fraction = _stable_fraction(str(parent_asin), seed)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + validation_ratio:
        return "validation"
    return "test"


def product_group_id(parent_asin: str, seed: str) -> str:
    """Hash the target grouping key so the raw identifier is never serialized."""

    return hashlib.sha256(
        f"{seed}\0product-group\0{parent_asin}".encode("utf-8")
    ).hexdigest()[:20]


def synthesize_catalog_samples(
    products: Mapping[str, dict],
    *,
    count: int,
    seed: int,
    profiles: Sequence[dict] = (),
    excluded_parent_asins: Iterable[str] = (),
) -> list[dict]:
    """Create catalog-derived sessions using the public simulator primitives."""

    if count < 0:
        raise ValueError("count must be non-negative")
    excluded = {str(value) for value in excluded_parent_asins}
    identifiers = sorted(set(products) - excluded)
    if count > len(identifiers):
        raise ValueError(
            f"requested {count} catalog samples but only {len(identifiers)} are available"
        )
    rng = random.Random(seed)
    selected = rng.sample(identifiers, count)
    result: list[dict] = []
    for index, parent_asin in enumerate(selected):
        scenario = SCENARIO_CYCLE[index % len(SCENARIO_CYCLE)]
        sample_id = f"synthetic_{seed}_{index:06d}"
        card = intent_card(products[parent_asin])
        behavior = behavior_for(
            scenario,
            card,
            random.Random(f"{sample_id}\0{scenario}"),
        )
        profile = copy.deepcopy(profiles[index % len(profiles)]) if profiles else {}
        result.append(
            {
                "sample_id": sample_id,
                "scenario_type": scenario,
                "user_profile": profile,
                "ground_truth": {"parent_asin": parent_asin},
                "intent_card": card,
                "behavior": behavior,
            }
        )
    return result


def _ranked_ids(response: object, catalog_ids: set[str], limit: int) -> list[str]:
    if not isinstance(response, dict) or not isinstance(response.get("recommendations"), list):
        return []
    ranked: list[str] = []
    seen: set[str] = set()
    for item in response["recommendations"]:
        value = item.get("parent_asin") if isinstance(item, dict) else item
        parent_asin = str(value or "").strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        ranked.append(parent_asin)
        if len(ranked) >= limit:
            break
    return ranked


def _safe_respond(
    agent: Agent,
    session_id: str,
    user_message: str,
    *,
    turn: int,
    top_k: int,
) -> tuple[dict, bool]:
    """Mirror the evaluator's exception-to-empty-response behavior."""

    try:
        response = agent.respond(session_id, user_message, turn=turn, top_k=top_k)
    except Exception:
        return {"message": "", "ask_attribute": None, "recommendations": []}, True
    if not isinstance(response, dict):
        return {"message": "", "ask_attribute": None, "recommendations": []}, True
    return response, False


def _rank_of(target: str, ranking: Sequence[str]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def _fixed_counts(values: Iterable[str], names: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: int(counts.get(name, 0)) for name in names}


def observable_features(agent: Agent, session_id: str, turn: int, prior_actions: Sequence[str]) -> dict:
    """Extract numeric/categorical state without consulting target-side data."""

    state = agent._sessions[session_id]
    structured = state.structured
    if structured is None:
        raise ValueError("counterfactual dataset generation requires typed conversation state")

    active = structured.active_constraints()
    active_types = [record.attribute.value for record in active]
    hard_sources = {
        ConstraintSource.INITIAL_CATEGORY,
        ConstraintSource.INITIAL_REQUIREMENT,
        ConstraintSource.OVERRIDE,
    }
    message_kinds = [record.kind.value for record in structured.messages]
    no_preferences = [record.attribute.value for record in structured.no_preference_history]
    trace = state.last_retrieval
    profile = state.user_profile
    average_rating = profile.get("average_prior_rating")
    try:
        average_rating = float(average_rating)
    except (TypeError, ValueError):
        average_rating = None
    tags = profile.get("preference_tags")
    tag_count = len(tags) if isinstance(tags, list) else 0
    summary = str(profile.get("summary") or "")

    active_counts = _fixed_counts(active_types, ACTION_ORDER)
    asked_counts = _fixed_counts(prior_actions, ACTION_ORDER)
    no_preference_counts = _fixed_counts(no_preferences, ACTION_ORDER)
    selected_candidate_count = len(trace.get("selected_sparse", ()))
    return {
        "turn": turn,
        "remaining_turns": MAX_TURNS - turn,
        "message_count": len(structured.messages),
        "message_kind_counts": _fixed_counts(
            message_kinds, tuple(kind.value for kind in MessageKind)
        ),
        "active_constraint_count": len(active),
        "active_hard_constraint_count": sum(
            record.source in hard_sources for record in active
        ),
        "retracted_constraint_count": sum(
            getattr(record.status, "value", "") == "retracted"
            for record in structured.constraints
        ),
        "active_constraint_counts": active_counts,
        "active_constraint_type_counts": active_counts,
        "known_attribute_flags": {
            name: int(name in active_types) for name in ACTION_ORDER
        },
        "no_preference_attributes": list(no_preferences),
        "no_preference_count": len(no_preferences),
        "no_preference_type_counts": no_preference_counts,
        "asked_attributes": list(prior_actions),
        "asked_count": len(prior_actions),
        "asked_attribute_counts": asked_counts,
        "prior_ask_attribute_counts": asked_counts,
        "last_asked_attribute": prior_actions[-1] if prior_actions else None,
        "query_term_count": len(structured.retrieval_text().split()),
        "retrieval_text_token_count": len(structured.retrieval_text().split()),
        "candidate_count": selected_candidate_count,
        "has_override": int("override" in message_kinds),
        "retrieval": {"candidate_count": selected_candidate_count},
        "retrieval_route_candidate_counts": {
            name: len(trace.get(name, ()))
            for name in ("global", "category", "clean_category", "selected_sparse")
        },
        "profile_average_prior_rating": average_rating,
        "profile_preference_tag_count": tag_count,
        "profile_summary_token_count": len(summary.split()),
    }


def _transition(
    sample: dict,
    action: str,
    disclosed: set[str],
    boundary_used: bool,
    override_applied: bool,
    next_turn: int,
) -> tuple[str, set[str], bool, bool]:
    """Apply the evaluator's next-message transition for one candidate action."""

    next_disclosed = set(disclosed)
    next_boundary_used = boundary_used
    next_override_applied = override_applied
    override = sample.get("behavior", {}).get("override") or {}
    if not override_applied and next_turn == int(override.get("turn", -1)):
        next_override_applied = True
        new_value = str(override.get("new_value", ""))
        if new_value:
            next_disclosed.add(new_value)
        message = str(
            override.get(
                "message", "Actually, please ignore my earlier preference."
            )
        )
    else:
        message, next_boundary_used = customer_reply(
            sample, action, next_disclosed, next_boundary_used
        )
    return message, next_disclosed, next_boundary_used, next_override_applied


def _technical_utility(rank: int | None, turn: int, eligible: bool) -> float:
    if not eligible or rank is None or rank > 10:
        return 0.0
    efficiency = (MAX_TURNS + 1 - turn) / MAX_TURNS
    return 0.50 + 0.30 / rank + 0.20 * efficiency


def build_counterfactual_rows(
    agent: Agent,
    samples: Sequence[dict],
    catalog_ids: set[str],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict],
    *,
    split_seed: str = "techjam-question-policy-v1",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    max_states_per_sample: int = 3,
    candidate_depth: int = 100,
    rank_shaping_weight: float = 0.05,
) -> list[dict]:
    """Enumerate one-step question outcomes along an ``other`` rollout."""

    if max_states_per_sample < 1:
        raise ValueError("max_states_per_sample must be at least one")
    if candidate_depth < 10:
        raise ValueError("candidate_depth must be at least ten")
    if not math.isfinite(rank_shaping_weight) or rank_shaping_weight < 0:
        raise ValueError("rank_shaping_weight must be finite and non-negative")

    rows: list[dict] = []
    for sample_index, original_sample in enumerate(samples):
        target = str(original_sample["ground_truth"]["parent_asin"])
        if target not in products:
            continue
        card, behavior = materialize_hidden_fields(original_sample, dict(products))
        sample = {**original_sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            sample, coarse_category(categories.get(target, [])), disclosed
        )
        session_id = f"policy_data_{sample_index}_{uuid.uuid4().hex}"
        agent.reset(session_id, sample.get("user_profile") or {})
        prior_actions: list[str] = []
        group_id = product_group_id(target, split_seed)
        split = split_for_parent_asin(
            target,
            seed=split_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        states_written = 0

        for turn in range(1, MAX_TURNS + 1):
            response, current_agent_error = _safe_respond(
                agent,
                session_id,
                user_message,
                turn=turn,
                top_k=candidate_depth,
            )
            current_ranking = _ranked_ids(response, catalog_ids, candidate_depth)
            current_rank = _rank_of(target, current_ranking)
            if override_applied and current_rank is not None and current_rank <= 10:
                break
            if turn == MAX_TURNS or states_written >= max_states_per_sample:
                break

            features = observable_features(agent, session_id, turn, prior_actions)
            features["current_agent_error"] = int(current_agent_error)
            state_rows: list[dict] = []
            for action in ACTION_ORDER:
                (
                    next_message,
                    action_disclosed,
                    action_boundary_used,
                    action_override_applied,
                ) = _transition(
                    sample,
                    action,
                    disclosed,
                    boundary_used,
                    override_applied,
                    turn + 1,
                )
                branch_id = f"{session_id}_{turn}_{action}"
                branch_state = copy.deepcopy(agent._sessions[session_id])
                # ``respond`` already recorded its configured question. Replace
                # that decision with the counterfactual action before observing
                # the resulting customer turn.
                if branch_state.asked_attributes:
                    branch_state.asked_attributes[-1] = action
                else:
                    branch_state.asked_attributes.append(action)
                branch_state.last_asked_attribute = action
                agent._sessions[branch_id] = branch_state
                next_response, action_agent_error = _safe_respond(
                    agent,
                    branch_id,
                    next_message,
                    turn=turn + 1,
                    top_k=candidate_depth,
                )
                next_ranking = _ranked_ids(next_response, catalog_ids, candidate_depth)
                next_rank = _rank_of(target, next_ranking)
                official_utility = _technical_utility(
                    next_rank, turn + 1, action_override_applied
                )
                reciprocal_rank_at_depth = (
                    0.0 if next_rank is None else 1.0 / next_rank
                )
                training_reward = (
                    official_utility
                    + rank_shaping_weight * reciprocal_rank_at_depth
                )
                state_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model_input": {
                            "state": copy.deepcopy(features),
                            "action": action,
                        },
                        "label": {
                            "training_reward": training_reward,
                            "official_next_turn_utility": official_utility,
                            "target_rank_at_depth": next_rank,
                            "hit_at_10": bool(
                                action_override_applied
                                and next_rank is not None
                                and next_rank <= 10
                            ),
                            "reciprocal_rank_at_depth": reciprocal_rank_at_depth,
                            "evaluator_eligible": action_override_applied,
                            "revealed_constraint_count": len(action_disclosed)
                            - len(disclosed),
                            "boundary_no_preference": (
                                action_boundary_used and not boundary_used
                            ),
                            "agent_error": action_agent_error,
                            "is_best_action": False,
                        },
                        "metadata": {
                            "split": split,
                            "product_group_id": group_id,
                            "sample_id": str(sample.get("sample_id", sample_index)),
                            "state_turn": turn,
                            "reference_policy": "always_other",
                        },
                    }
                )
                del agent._sessions[branch_id]

            best_reward = max(row["label"]["training_reward"] for row in state_rows)
            for row in state_rows:
                row["label"]["is_best_action"] = math.isclose(
                    row["label"]["training_reward"],
                    best_reward,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            rows.extend(state_rows)
            states_written += 1

            (
                user_message,
                disclosed,
                boundary_used,
                override_applied,
            ) = _transition(
                sample,
                "other",
                disclosed,
                boundary_used,
                override_applied,
                turn + 1,
            )
            prior_actions.append("other")

        del agent._sessions[session_id]
    return rows


def dataset_manifest(rows: Sequence[dict], config: Mapping[str, object]) -> dict:
    split_rows = Counter(row["metadata"]["split"] for row in rows)
    split_states: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        metadata = row["metadata"]
        split = metadata["split"]
        split_states[split].add(
            (
                metadata["product_group_id"],
                metadata["state_turn"],
                metadata["sample_id"],
            )
        )
        split_groups[split].add(metadata["product_group_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "state_count": sum(len(values) for values in split_states.values()),
        "product_group_count": len(
            set().union(*(values for values in split_groups.values()))
        ),
        "split_row_counts": dict(sorted(split_rows.items())),
        "split_state_counts": {
            key: len(value) for key, value in sorted(split_states.items())
        },
        "split_product_group_counts": {
            key: len(value) for key, value in sorted(split_groups.items())
        },
        "actions": list(ACTION_ORDER),
        "model_input_policy": (
            "Train only on model_input. label and metadata use hidden offline "
            "evaluation state and must never be model features."
        ),
        "config": dict(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build product-disjoint counterfactual ask-attribute data"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--catalog-samples",
        type=int,
        default=0,
        help="also synthesize this many targets outside the labeled public targets",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help=(
            "write only catalog-synthesized targets, preserving all labeled "
            "public targets as an external evaluation set"
        ),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-states-per-sample", type=int, default=3)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--rank-shaping-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--split-seed", default="techjam-question-policy-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    public_samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    synthetic_samples: list[dict] = []
    if args.catalog_samples:
        synthetic_samples = synthesize_catalog_samples(
            products,
            count=args.catalog_samples,
            seed=args.seed,
            profiles=[sample.get("user_profile") or {} for sample in public_samples],
            excluded_parent_asins={
                str(sample["ground_truth"]["parent_asin"])
                for sample in public_samples
            },
        )
    if args.catalog_only and not synthetic_samples:
        parser.error("--catalog-only requires a positive --catalog-samples")
    samples = synthetic_samples if args.catalog_only else [*public_samples, *synthetic_samples]
    if args.max_samples is not None:
        if args.max_samples < 0:
            parser.error("--max-samples must be non-negative")
        samples = samples[: args.max_samples]

    config = {
        "catalog": str(args.catalog),
        "dataset": str(args.dataset),
        "public_sample_count": sum(
            not str(sample.get("sample_id", "")).startswith("synthetic_")
            for sample in samples
        ),
        "catalog_sample_count": sum(
            str(sample.get("sample_id", "")).startswith("synthetic_")
            for sample in samples
        ),
        "catalog_only": args.catalog_only,
        "max_states_per_sample": args.max_states_per_sample,
        "candidate_depth": args.candidate_depth,
        "rank_shaping_weight": args.rank_shaping_weight,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
    }
    agent = Agent(args.catalog)
    try:
        rows = build_counterfactual_rows(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            max_states_per_sample=args.max_states_per_sample,
            candidate_depth=args.candidate_depth,
            rank_shaping_weight=args.rank_shaping_weight,
        )
    finally:
        agent.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = dataset_manifest(rows, config)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
