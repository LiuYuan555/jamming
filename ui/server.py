"""Local demo server for the TechJam shopping copilot.

Standard library only, matching the agent it wraps -- no third-party package is
introduced, so the property that makes the submission safe under the organizer's
"no network, CPU only" clause is preserved.

The server binds to the frozen organizer contract and nothing else:

    Agent(catalog_path) / reset(session_id, profile) / respond(...)

Every richer thing the inspector shows is produced by ``ui.probe``, which fails
soft.  Swap the retrieval stack, flip the question policy, or wire in the intent
router and this file keeps working unchanged.

Usage:
    python -m ui.server                      # auto-discovers the catalog
    python -m ui.server --catalog PATH --port 8000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter.agent import Agent  # noqa: E402
from ui import probe  # noqa: E402
from ui.generate import DEFAULT_MODEL, ResponseGenerator  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Opening lines mirroring the evaluated scenarios, for one-click demos.
#
# Ordering matters for a live walkthrough: "Stated preference" creates an
# INITIAL_PREFERENCE record, which is the only source "Intent override" can
# retract. Opening with "Buying" instead produces INITIAL_REQUIREMENT, which the
# override deliberately leaves standing.
PRESETS = [
    {
        "label": "Buying",
        "text": "I'm looking for women's combat boots. A key requirement is: Rubber sole.",
    },
    {
        "label": "Browsing",
        "text": "I'm looking for a winter jacket, but I'm still exploring.",
    },
    {
        "label": "Stated preference",
        "text": "I'm looking for women's combat boots. I prefer a zipper closure.",
    },
    {
        "label": "Clarification reply",
        "text": "For that, what matters is: leather; Imported",
    },
    {
        "label": "Intent override",
        "text": "Actually, ignore my earlier preference. What I need is: waterproof",
    },
    {
        "label": "No preference",
        "text": "I don't have a preference for color.",
    },
]


def find_catalog(explicit: str | None) -> Path:
    """Locate catalog.jsonl without forcing a 60MB copy into the repo."""

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"catalog not found: {path}")
        return path
    candidates = [
        REPO / "data" / "catalog.jsonl",
        REPO.parent / "techjam-conversational-search" / "data" / "catalog.jsonl",
        REPO.parent / "data" / "catalog.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "catalog.jsonl not found. Pass --catalog PATH, or place it at data/catalog.jsonl"
    )


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None so browsers can parse the response."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _norm(value: object) -> str:
    """Lowercase alphanumeric text, matching how the catalog is searched."""

    return " ".join(TOKEN_RE.findall(str(value or "").lower()))


def _flat(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value or "")


class Display:
    """Slim ASIN -> presentable fields index, for product cards.

    Also keeps a normalized text blob per product so the UI can show *why* a
    product was recommended.  The match test is a plain phrase containment
    check against the catalog's own text -- it deliberately does not import the
    reranker, so it cannot drift when the scoring changes.
    """

    def __init__(self, catalog_path: Path) -> None:
        self.items: dict[str, dict] = {}
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except ValueError:
                    continue
                asin = str(product.get("parent_asin", ""))
                if not asin:
                    continue
                categories = product.get("categories") or []
                blob = " ".join(
                    (
                        _norm(product.get("title")),
                        _norm(_flat(product.get("features"))),
                        _norm(_flat(product.get("details"))),
                        _norm(product.get("store")),
                    )
                )
                self.items[asin] = {
                    "title": str(product.get("title") or "")[:180],
                    "price": product.get("price"),
                    "store": str(product.get("store") or ""),
                    "rating": product.get("average_rating"),
                    "ratings": product.get("rating_number"),
                    "category": " › ".join(str(c) for c in categories[-2:]),
                    "blob": f" {blob} "[:2400],
                }

    def card(
        self,
        asin: str,
        rank: int,
        previous: dict[str, int],
        wants: Sequence[tuple[str, str]] = (),
    ) -> dict:
        item = self.items.get(asin, {})
        was = previous.get(asin)
        blob = item.get("blob", "")
        reasons = []
        for attribute, value in wants:
            phrase = _norm(value)
            if phrase and f" {phrase} " in blob:
                reasons.append({"attribute": attribute, "value": value[:40]})
            if len(reasons) >= 3:
                break
        return {
            "rank": rank,
            "parent_asin": asin,
            "title": item.get("title") or "(not in catalog)",
            "price": item.get("price"),
            "store": item.get("store") or "",
            "rating": item.get("rating"),
            "ratings": item.get("ratings"),
            "category": item.get("category") or "",
            "reasons": reasons,
            # Rank movement is derived purely from the contract's output, so it
            # keeps working no matter how retrieval changes underneath.
            "delta": (was - rank) if was is not None else None,
            "isNew": was is None,
        }


class Session:
    def __init__(self, session_id: str, profile: dict) -> None:
        self.id = session_id
        self.profile = profile
        self.turn = 0
        self.previous_ranks: dict[str, int] = {}
        self.transcript: list[dict] = []


class App:
    def __init__(
        self,
        catalog_path: Path,
        llm: bool = False,
        llm_model: str = DEFAULT_MODEL,
        entropy: bool = True,
        dense: bool = False,
    ) -> None:
        # The agent owns a SQLite connection, which is bound to the thread that
        # created it. The HTTP server is threaded (a browser holding a
        # keep-alive connection must not block every other request), so all
        # agent work is built on, and dispatched to, one dedicated worker.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")
        self._pool.submit(self._build, catalog_path, llm, llm_model, entropy, dense).result()

    def _build(
        self, catalog_path: Path, llm: bool, llm_model: str, entropy: bool, dense: bool
    ) -> None:
        started = time.perf_counter()
        print(f"  catalog   {catalog_path}")
        print("  building FTS5 index and loading reranker catalog ...", flush=True)
        self.agent = Agent(str(catalog_path))
        agent_ready = time.perf_counter()
        print(f"  agent     ready in {agent_ready - started:.1f}s", flush=True)
        print("  loading display index ...", flush=True)
        self.display = Display(catalog_path)
        print(
            f"  display   {len(self.display.items):,} products"
            f" in {time.perf_counter() - agent_ready:.1f}s",
            flush=True,
        )
        self.sessions: dict[str, Session] = {}

        # Optional, opt-in, and never on the scoring path -- the evaluator
        # discards `message`, so this can only cost latency during scoring.
        self.generator: ResponseGenerator | None = None
        if llm:
            print(f"  loading {llm_model} ...", flush=True)
            generator = ResponseGenerator(llm_model)
            if generator.load():
                self.generator = generator
                print(f"  llm       ready on {generator.device}", flush=True)
            else:
                print(f"  llm       UNAVAILABLE -> {generator.error}", flush=True)
                print("            falling back to template replies", flush=True)

        # Shannon-entropy question selection, replacing the agent's fixed
        # `always_other`. Applied in this layer so the scored agent is
        # untouched -- HISTORY.md measures entropy questions at 0.827465 vs
        # 0.853670, so it must not become the scoring default. It exists here
        # because asking a varied, motivated question is what a shopping
        # assistant should look like, and repeating "other" ten times is not.
        self.gate = None
        if entropy:
            print("  building entropy question gate ...", flush=True)
            gate_started = time.perf_counter()
            try:
                from starter.intent_hybrid_router import ShannonEntropyQuestionGate

                self.gate = ShannonEntropyQuestionGate(str(catalog_path))
                print(
                    f"  entropy   ready in {time.perf_counter() - gate_started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"  entropy   UNAVAILABLE -> {type(exc).__name__}: {exc}", flush=True)

        # Offline dense retrieval. The agent fuses `hybrid_retriever` results
        # with BM25 by weighted RRF whenever the attribute is not None, so we
        # attach the local MiniLM retriever rather than the OpenAI one -- no
        # API key, no network. Weights come from AgentConfig.hybrid_config
        # (0.85 sparse / 0.15 dense by default).
        self.dense = False
        if dense:
            print("  loading offline dense retriever (MiniLM) ...", flush=True)
            try:
                from starter.hf_hybrid_retrieval import HFHybridRetriever

                retriever = HFHybridRetriever(top_k=100)
                if getattr(retriever, "available", False):
                    self.agent.hybrid_retriever = retriever
                    self.dense = True
                    sparse_w, dense_w = self.agent.config.hybrid_config.fusion_weights([])
                    print(f"  dense     ready ({sparse_w:.2f} sparse / {dense_w:.2f} dense RRF)",
                          flush=True)
                else:
                    print(f"  dense     UNAVAILABLE -> "
                          f"{getattr(retriever, 'unavailable_reason', 'unknown')}", flush=True)
            except Exception as exc:
                print(f"  dense     UNAVAILABLE -> {type(exc).__name__}: {exc}", flush=True)

        self.meta = probe.describe(self.agent)
        self.meta["dense"] = "MiniLM-L6-v2 (offline)" if self.dense else None
        self.meta["question_policy"] = (
            "shannon_entropy" if self.gate else self.meta.get("question_policy")
        )
        self.meta["catalog_size"] = len(self.display.items)
        self.meta["presets"] = PRESETS
        self.meta["llm"] = (
            {"model": llm_model, "device": self.generator.device}
            if self.generator
            else None
        )

    def start(self, session_id: str, profile: dict | None) -> dict:
        return self._pool.submit(self._start, session_id, profile).result()

    def message(self, session_id: str, text: str, top_k: int = 10) -> dict:
        return self._pool.submit(self._message, session_id, text, top_k).result()

    def _start(self, session_id: str, profile: dict | None) -> dict:
        session = Session(session_id, profile or {})
        self.sessions[session_id] = session
        self.agent.reset(session_id, session.profile)
        return {"session_id": session_id, "turn": 0}

    def _ask_by_entropy(
        self, session_id: str, asins: Sequence[str]
    ) -> tuple[str, str, list[dict]] | None:
        """Pick the next question by expected information gain.

        Returns ``None`` on any failure so the agent's own template stands.
        The agent's asked-attribute bookkeeping is corrected too: it recorded
        `other`, but the customer is being asked something else, and the state
        tracker uses that field to interpret the next reply.
        """

        try:
            agent_state = self.agent._sessions[session_id]
            structured = agent_state.structured
            if structured is None:
                return None
            decision = self.gate.decide(
                structured, list(asins), agent_state.asked_attributes[:-1]
            )
            attribute = decision.decision.ask_attribute
            if not attribute:
                return None
            if agent_state.asked_attributes:
                agent_state.asked_attributes[-1] = attribute
            agent_state.last_asked_attribute = attribute
            scores = [
                {
                    "attribute": item.attribute,
                    "gain": round(item.information_gain_bits, 3),
                    "coverage": round(item.answer_coverage, 3),
                    "value": round(item.value, 4),
                    "masked": item.masked,
                    "reason": item.mask_reason,
                }
                for item in sorted(decision.scores, key=lambda i: -i.value)
            ]
            return decision.decision.message, attribute, scores
        except Exception:
            return None

    def _message(self, session_id: str, text: str, top_k: int = 10) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            self._start(session_id, {})
            session = self.sessions[session_id]

        session.turn += 1
        started = time.perf_counter()
        reply = self.agent.respond(session_id, text, session.turn, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Defensive: the evaluator tolerates a malformed reply, so we do too.
        if not isinstance(reply, dict):
            reply = {}
        raw = reply.get("recommendations") or []
        asins: list[str] = []
        for entry in raw:
            if isinstance(entry, dict) and entry.get("parent_asin"):
                asins.append(str(entry["parent_asin"]))
            elif isinstance(entry, str):
                asins.append(entry)

        wants = probe.active_values(self.agent, session_id)
        cards = [
            self.display.card(asin, index, session.previous_ranks, wants)
            for index, asin in enumerate(asins, start=1)
        ]
        session.previous_ranks = {asin: i for i, asin in enumerate(asins, start=1)}

        message = reply.get("message")
        ask = reply.get("ask_attribute")
        entropy_scores: list[dict] = []
        if self.gate is not None:
            override = self._ask_by_entropy(session_id, asins)
            if override:
                message, ask, entropy_scores = override

        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        generated = False

        generate_ms = 0.0
        if self.generator is not None:
            gen_started = time.perf_counter()
            result = self.generator.generate(wants, cards, ask if isinstance(ask, str) else None)
            generate_ms = (time.perf_counter() - gen_started) * 1000.0
            if result:
                # Model writes the confirmation; the agent's own template still
                # supplies the question, so the prose and `ask_attribute` can
                # never disagree.
                question = message if isinstance(message, str) else ""
                message = f"{result['reply']} {question}".strip()
                generated = True
                for card in cards:
                    grounded = result["reasons"].get(card["parent_asin"])
                    if grounded:
                        card["reasons"] = [
                            {"attribute": "stated", "value": value} for value in grounded
                        ]
                        card["explained"] = True
            prompt_tokens += self.generator.last_usage["prompt_tokens"]
            completion_tokens += self.generator.last_usage["completion_tokens"]

        session.transcript.append({"role": "user", "text": text, "turn": session.turn})
        session.transcript.append(
            {
                "role": "agent",
                "text": message if isinstance(message, str) else "",
                "ask": ask,
                "turn": session.turn,
                "count": len(cards),
            }
        )

        return {
            "turn": session.turn,
            "message": message if isinstance(message, str) else "",
            "ask_attribute": ask,
            "recommendations": cards,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "generated": generated,
            "entropy_scores": entropy_scores,
            "latency_ms": round(elapsed_ms, 1),          # retrieval only
            "generate_ms": round(generate_ms, 1),        # local LLM only
            "total_ms": round(elapsed_ms + generate_ms, 1),
            "sections": probe.inspect(self.agent, session_id),
        }


class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        if "api/message" in str(args):
            sys.stderr.write(f"  turn -> {args[0]}\n")

    def _send(self, payload: dict, status: int = 200) -> None:
        # json.dumps emits Infinity/NaN, which Python parses back happily but
        # JSON.parse in the browser rejects outright -- the response arrives as
        # HTTP 200 and then fails to decode. The entropy gate scores masked
        # attributes at -inf, so this is reachable from ordinary input.
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._file(HERE / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._send(self.app.meta)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self._send({"error": "invalid JSON"}, 400)
            return

        try:
            if self.path == "/api/session":
                self._send(
                    self.app.start(
                        str(payload.get("session_id") or f"ui-{int(time.time()*1000)}"),
                        payload.get("user_profile") or {},
                    )
                )
            elif self.path == "/api/message":
                text = str(payload.get("message") or "").strip()
                if not text:
                    self._send({"error": "empty message"}, 400)
                    return
                self._send(
                    self.app.message(str(payload.get("session_id") or "ui"), text)
                )
            else:
                self.send_error(404)
        except Exception as exc:  # never take the server down mid-demo
            import traceback

            traceback.print_exc()
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shopping copilot demo UI")
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="generate grounded replies with a local model (demo only; "
        "the evaluator discards `message`, so this cannot affect the score)",
    )
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="fuse offline MiniLM dense retrieval with BM25 (needs "
        "tools/build_hf_embeddings.py to have been run)",
    )
    parser.add_argument(
        "--no-entropy",
        action="store_true",
        help="use the agent's fixed always_other policy instead of "
        "Shannon-entropy question selection",
    )
    args = parser.parse_args()

    print("\n  TechJam shopping copilot - demo console")
    print("  " + "-" * 44)
    Handler.app = App(
        find_catalog(args.catalog),
        llm=args.llm,
        llm_model=args.llm_model,
        entropy=not args.no_entropy,
        dense=args.dense,
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  ready     {url}\n")
    if not args.no_browser:
        webbrowser.open(url)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")


if __name__ == "__main__":
    main()
