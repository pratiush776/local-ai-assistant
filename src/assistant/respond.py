"""The shared brain: one surface-agnostic turn of the assistant.

`respond()` takes the clean conversation so far plus a new user message and
*yields events* describing what it is doing — deciding the route, searching
files, streaming the answer, citing sources, timing each step. The CLI prints
these events; the GUI renders them. Neither owns the logic; this module does.

Routing per turn:
  decide() -> fact  : retrieve + rerank + grounded generation over your files
            -> local: answer with the local model (chat / reason / action)
            -> ask  : ambiguous -> emit one clarifying question (next turn resolves it)
  force_escalate     : skip routing, answer with Claude (user-authorized)
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import httpx

from . import rag
from .claude import client as claude_client
from .context import messages_to_text
from .decide import decide

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
PRIMARY_LOCAL_MODEL = "qwen3.5:9b"
ESCALATE_MODEL = "claude-haiku-5-5"
SYSTEM_PROMPT = "You are a helpful assistant whose main job is to help users ethically."

EventKind = Literal["status", "route", "delta", "citations", "timing", "done","retrieval"]


@dataclass
class Event:
    """One thing that happened this turn. `text` carries human-readable status or
    a streamed token; `data` carries structured detail for a GUI to render."""

    kind: EventKind
    text: str = ""
    data: dict = field(default_factory=dict)


# ---------- streaming helpers: yield ("delta", piece) then ("stats", {...}) ----------
def _stream_ollama(messages: list[dict], model: str) -> Iterator[tuple[str, object]]:
    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": True,
        "options": {"temperature": 0.7},
    }
    with httpx.stream("POST", OLLAMA_CHAT_URL, json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            yield "delta", chunk["message"]["content"]
            if chunk["done"]:
                out = chunk["eval_count"]
                secs = chunk["eval_duration"] / 1e9
                yield (
                    "stats",
                    {
                        "input_tokens": chunk["prompt_eval_count"],
                        "output_tokens": out,
                        "tokens_per_sec": round(out / secs, 1) if secs else None,
                        "ttft_s": round(
                            (chunk["load_duration"] + chunk["prompt_eval_duration"])
                            / 1e9,
                            1,
                        ),
                    },
                )


def _stream_claude(
    messages: list[dict], system: str, model: str
) -> Iterator[tuple[str, object]]:
    with claude_client.messages.stream(
        model=model, max_tokens=1024, messages=messages, system=system
    ) as stream:
        for text in stream.text_stream:
            yield "delta", text
        final = stream.get_final_message()
    yield "stats", {
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
        "tokens_per_sec": None,
    }


def _stream(messages, system, model) -> Iterator[tuple[str, object]]:
    """Dispatch to Claude or Ollama; Ollama takes the system prompt as a message."""
    if model.startswith("claude"):
        yield from _stream_claude(messages, system, model)
    else:
        yield from _stream_ollama(
            [{"role": "system", "content": system}] + messages, model
        )


# ---------------------------- the turn ----------------------------
def respond(
    user_msg: str, history: list[dict], *, force_escalate: bool = False
) -> Iterator[Event]:
    """Run one turn. `history` is the prior clean conversation (list of
    {role, content}); it is not mutated. Yields Events; the final Event(kind="done")
    carries the answer and metadata so the caller can append it to its own history."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- route ----
    yield Event("status", "Reading your message…")
    if force_escalate:
        label, route, conf, candidates = "escalate", "escalate", 1.0, []
    else:
        decision = decide(user_msg, messages_to_text(history))
        label = decision.label or ""
        route, conf, candidates = (
            decision.route,
            decision.confidence,
            decision.candidates,
        )
    timings["route_ms"] = round((time.perf_counter() - t0) * 1000)
    yield Event(
        "route",
        text=f"route={route} label={label or '—'} conf={conf:.2f}",
        data={
            "route": route,
            "label": label,
            "confidence": conf,
            "candidates": candidates,
        },
    )

    # ---- ambiguous: ask one clarifying question, resolve next turn ----
    if route == "ask":
        yield Event("status", "I need a bit more detail…")
        sys = (
            "You are the assistant's clarifier. The router could not confidently tell "
            f"whether this request is {candidates or 'one thing or another'}. In 1-2 "
            "friendly sentences, ask the user the single question that would resolve it. "
            "Do not answer the request itself."
        )
        answer = ""
        for kind, payload in _stream(
            [{"role": "user", "content": user_msg}], sys, PRIMARY_LOCAL_MODEL
        ):
            if kind == "delta":
                answer += payload
                yield Event("delta", text=payload)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
        yield Event("timing", data=timings)
        yield Event(
            "done",
            text=answer,
            data={"label": "ask", "clarifying": True, "model": PRIMARY_LOCAL_MODEL},
        )
        return

    # ---- fact: retrieve + rerank + grounded generation ----
    if label == "fact":
        yield Event("status", "Searching your files…")
        ts = time.perf_counter()
        chunks = rag.rag_retrieve(user_msg)
        timings["retrieve_rerank_ms"] = round((time.perf_counter() - ts) * 1000)
        #-- kept or dropped with its score
        rows = [
            {
                "id": cid,
                "score": round(float(score),4),
                "preview":text[:80],
                "kept": score >= rag.RERANK_THRESHOLD
            } for cid, text, score in chunks
        ]
        kept_n = sum(r["kept"] for r in rows)
        yield Event(kind="retrieval",text=f"{kept_n}/{len(chunks)} kept", data={"chunks":rows})
        hits = [(cid,text) for cid,text,score in chunks if score >= rag.RERANK_THRESHOLD]
        if not hits:
            yield Event("status", "Nothing relevant in your files.")
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
            yield Event("timing", data=timings)
            yield Event(
                "done",
                text="I don't have that in your files.",
                data={"label": "fact", "not_found": True, "model": PRIMARY_LOCAL_MODEL},
            )
            return
        yield Event(
            "status", f"Found {len(hits)} relevant passage(s) — writing the answer…"
        )
        ctx = rag.format_context(hits)
        call_msgs = history + [
            {"role": "user", "content": f"{ctx}\n\nQuestion: {user_msg}"}
        ]
        answer, gen_stats = "", {}
        ts = time.perf_counter()
        for kind, payload in _stream(
            call_msgs, rag.RAG_SYSTEM_PROMPT, PRIMARY_LOCAL_MODEL
        ):
            if kind == "delta":
                answer += payload
                yield Event("delta", text=payload)
            elif kind == "stats":
                gen_stats = payload
        timings["generate_ms"] = round((time.perf_counter() - ts) * 1000)
        cites = [cid for cid, _ in hits]
        texts = [text for _, text in hits]
        yield Event(
            "citations",
            text=f"{len(cites)} source(s)",
            data={"ids": cites, "texts": texts},
        )
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
        yield Event("timing", data={**timings, **gen_stats})
        yield Event(
            "done",
            text=answer,
            data={"label": "fact", "citations": cites, "model": PRIMARY_LOCAL_MODEL},
        )
        return

    # ---- local / escalate: plain generation ----
    model = ESCALATE_MODEL if route == "escalate" else PRIMARY_LOCAL_MODEL
    yield Event(
        "status", "Asking Claude…" if route == "escalate" else "Writing a reply…"
    )
    call_msgs = history + [{"role": "user", "content": user_msg}]
    answer, gen_stats = "", {}
    ts = time.perf_counter()
    for kind, payload in _stream(call_msgs, SYSTEM_PROMPT, model):
        if kind == "delta":
            answer += payload
            yield Event("delta", text=payload)
        elif kind == "stats":
            gen_stats = payload
    timings["generate_ms"] = round((time.perf_counter() - ts) * 1000)
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
    yield Event("timing", data={**timings, **gen_stats})
    yield Event("done", text=answer, data={"label": label or route, "model": model})
