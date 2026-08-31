"""Defensive introspection of a running Agent.

This module is the reason the UI survives changes to the agent.  It reads
*nothing* it is not allowed to fail at reading: every accessor is wrapped, and a
section that cannot be produced is dropped rather than raising.  The frontend
renders whatever sections arrive, so adding a field to the agent makes it appear
in the inspector with no frontend edit, and removing one makes its panel vanish.

The only hard dependency is the frozen organizer contract:

    Agent(catalog_path)
    Agent.reset(session_id, user_profile) -> None
    Agent.respond(session_id, user_message, turn, top_k) -> dict

Everything below that line is best-effort.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


MAX_DEPTH = 6
MAX_ITEMS = 60


def safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run ``fn`` and swallow anything it raises.

    Used for every single agent attribute access in this file.  A refactor that
    renames or deletes an internal simply yields ``default`` here.
    """

    try:
        return fn()
    except Exception:
        return default


def plain(value: Any, depth: int = 0) -> Any:
    """Convert arbitrary agent internals into JSON-safe data.

    Handles dataclasses, enums, mappings and sequences generically so that new
    configuration fields serialize without this function being updated.
    """

    if depth > MAX_DEPTH:
        return "..."
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return safe(lambda: value.value, str(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: plain(safe(lambda f=f: getattr(value, f.name)), depth + 1)
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        out = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_ITEMS:
                out["..."] = f"{len(value) - MAX_ITEMS} more"
                break
            out[str(key)] = plain(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        rendered = [plain(item, depth + 1) for item in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            rendered.append(f"... {len(items) - MAX_ITEMS} more")
        return rendered
    return str(value)


# --------------------------------------------------------------------------
# Section builders.  Each returns a section dict or None; none of them raise.
# --------------------------------------------------------------------------


def _session(agent: Any, session_id: str) -> Any:
    """Locate the agent's per-session record without assuming its name."""

    for attribute in ("_sessions", "sessions", "_state", "states"):
        store = safe(lambda attribute=attribute: getattr(agent, attribute))
        if isinstance(store, Mapping) and session_id in store:
            return store[session_id]
    return None


def constraints_section(session: Any) -> dict | None:
    """Typed constraint records: value, attribute, source, turn, status."""

    records = safe(lambda: list(session.structured.constraints))
    if not records:
        return None

    rows = []
    for record in records:
        status = safe(lambda r=record: r.status.value, "active")
        rows.append(
            {
                "value": safe(lambda r=record: str(r.value), ""),
                "attribute": safe(lambda r=record: r.attribute.value, "unknown"),
                "source": safe(lambda r=record: r.source.value, ""),
                "turn": safe(lambda r=record: r.turn),
                "status": status,
                "retracted_turn": safe(lambda r=record: r.retracted_turn),
                "reason": safe(lambda r=record: r.retraction_reason),
            }
        )
    if not any(row["value"] for row in rows):
        return None

    active = sum(1 for row in rows if row["status"] == "active")
    return {
        "id": "constraints",
        "title": "Typed state",
        "kind": "constraints",
        "badge": f"{active} active",
        "rows": rows,
    }


def no_preference_section(session: Any) -> dict | None:
    """Attributes the customer explicitly declined to constrain."""

    records = safe(lambda: list(session.structured.no_preference_history))
    if not records:
        return None
    rows = [
        {
            "label": safe(lambda r=record: r.attribute.value, "unknown"),
            "turn": safe(lambda r=record: r.turn),
            "muted": safe(lambda r=record: r.status.value, "active") != "active",
        }
        for record in records
    ]
    return {
        "id": "no_preference",
        "title": "Declined attributes",
        "kind": "chips",
        "badge": f"{len(rows)}",
        "hint": "Absence of information, not a negative constraint.",
        "rows": rows,
    }


def routes_section(session: Any) -> dict | None:
    """Retrieval routes and their pool sizes.

    Fully generic over the trace dictionary, so a new retrieval route appears
    here automatically the moment the agent records it.
    """

    trace = safe(lambda: session.last_retrieval)
    if not isinstance(trace, Mapping) or not trace:
        return None
    rows = []
    for name, ids in trace.items():
        try:
            ids = list(ids)
        except Exception:
            continue
        rows.append(
            {
                "label": str(name).replace("_", " "),
                "count": len(ids),
                "head": [str(value) for value in ids[:3]],
                "empty": not ids,
            }
        )
    if not rows:
        return None
    live = sum(1 for row in rows if not row["empty"])
    return {
        "id": "routes",
        "title": "Retrieval routes",
        "kind": "routes",
        "badge": f"{live}/{len(rows)} firing",
        "rows": rows,
    }


def query_section(session: Any) -> dict | None:
    """The text actually handed to the retriever this turn."""

    text = safe(lambda: session.structured.retrieval_text())
    if not text:
        terms = safe(lambda: list(session.query_terms))
        text = " ".join(str(term) for term in terms) if terms else None
    if not text:
        return None
    return {
        "id": "query",
        "title": "Retrieval query",
        "kind": "text",
        "text": str(text),
    }


def config_section(agent: Any) -> dict | None:
    """Whole agent configuration, walked generically."""

    config = safe(lambda: agent.config)
    if config is None:
        return None
    data = plain(config)
    if not isinstance(data, dict) or not data:
        return None
    return {
        "id": "config",
        "title": "Active configuration",
        "kind": "tree",
        "collapsed": True,
        "rows": data,
    }


def components_section(agent: Any) -> dict | None:
    """Which optional components were actually constructed."""

    checks = {
        "constraint reranker": safe(lambda: agent.reranker is not None, False),
        "hybrid dense route": safe(lambda: agent.hybrid_retriever is not None, False),
        "typed state": safe(lambda: agent.config.use_typed_state, False),
    }
    rows = [
        {"label": name, "on": bool(value)} for name, value in checks.items()
    ]
    return {
        "id": "components",
        "title": "Components",
        "kind": "toggles",
        "rows": rows,
    }


SESSION_SECTIONS: Sequence[Callable[[Any], dict | None]] = (
    constraints_section,
    query_section,
    routes_section,
    no_preference_section,
)

AGENT_SECTIONS: Sequence[Callable[[Any], dict | None]] = (
    components_section,
    config_section,
)


def inspect(agent: Any, session_id: str) -> list[dict]:
    """Return every section that could be built, in display order."""

    sections: list[dict] = []
    session = safe(lambda: _session(agent, session_id))
    if session is not None:
        for builder in SESSION_SECTIONS:
            section = safe(lambda b=builder: b(session))
            if section:
                sections.append(section)
    for builder in AGENT_SECTIONS:
        section = safe(lambda b=builder: b(agent))
        if section:
            sections.append(section)
    return sections


def active_values(agent: Any, session_id: str) -> list[tuple[str, str]]:
    """Active (attribute, value) pairs, for the "why this matched" chips.

    Returns an empty list rather than raising if the agent has no typed state,
    so the chips simply do not appear.
    """

    session = safe(lambda: _session(agent, session_id))
    records = safe(lambda: list(session.structured.active_constraints()), [])
    pairs = []
    for record in records or []:
        value = safe(lambda r=record: str(r.value), "")
        attribute = safe(lambda r=record: r.attribute.value, "other")
        if value:
            pairs.append((attribute, value))
    return pairs


def describe(agent: Any) -> dict:
    """One-time description of the agent, for the header."""

    return {
        "class": safe(lambda: type(agent).__name__, "Agent"),
        "module": safe(lambda: type(agent).__module__, "unknown"),
        "question_policy": safe(lambda: agent.config.question_policy),
        "pool": safe(lambda: agent.config.candidate_pool_size),
        "mode": safe(lambda: agent.config.category_retrieval.mode),
    }
