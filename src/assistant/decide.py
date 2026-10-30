"""The router: decide how to handle a request.

One local model (MLX qwen) reads the label distribution from a single-token
logprobs call, then route_from() turns that distribution into a decision:
  - one clear winner        -> handle it locally
  - no clear winner         -> surface the candidates to the user, who picks
                               clarify (answer it) or escalate (send to Claude)

Escalation is user-authorized by default; `auto_escalate` is the hook for
headless use later (phone, scheduled jobs), to be gated by a cost budget.
"""

import math
from collections import defaultdict
from typing import Literal, cast

import httpx
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"  # the MLX build (bare tag; the GGUF is -q4_K_M)

WINNER_THRESHOLD = 0.65  # a label this high (and clearly ahead) = one clear winner
LIVE_CUTOFF = 0.15  # a label this high is a real contender ("alive")

Label = Literal["action", "fact", "reason", "chat"]
Route = Literal["local", "ask", "escalate"]
LABELS: set[Label] = {"action", "fact", "reason", "chat"}

LABEL_RULES = """Labels (what the NEXT step needs that a plain local chat model doesn't have):
- action: needs tools and the user's approval; it changes something in the world (send, book, set, buy).
- fact: needs outside or personal info not already in the conversation (files, notes, the web, anything recent).
- reason: needs multi-step thinking; all the information is already present.
- chat: needs nothing extra; answerable directly, including writing, rewriting, summarising, translating.
If several fit, pick the first in this order: action, fact, reason, chat."""


class Decision(BaseModel):
    route: Route
    label: Label | None = None  # the winner when route == "local"
    confidence: float  # the top label's probability
    candidates: list[str] = []  # the live labels when route != "local"


def label_scores(request: str, context: str = "") -> dict[str, float]:
    """One forward pass: read the first-token logprobs, return raw
    distribution over the 4 labels."""
    prompt = f"""Classify the NEXT step this request needs, given the conversation so far.
{LABEL_RULES}

Conversation so far:
{context or "(none)"}

Request: {request}
Reply with only the one word."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 8,
        "options": {"temperature": 0, "num_predict": 1},
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    first = r.json()["logprobs"][0]

    probs: dict[str, float] = defaultdict(float)
    for cand in first["top_logprobs"]:
        tok = cand["token"].strip().lower()
        if tok in LABELS:
            probs[tok] += math.exp(cand["logprob"])
    return probs


def route_from(scores: dict[str, float]) -> tuple[Route, list[str]]:
    """turn the label distribution into a routing decision."""
    alive = []
    for label, score in scores.items():
        if score >= WINNER_THRESHOLD:
            return "local", [label]
        if score > LIVE_CUTOFF:
            alive.append(label)
    return "ask", sorted(alive, key=lambda l: scores[l], reverse=True)


def choose_strategy() -> str:
    """Memory-aware hook. Single model on this machine; two-model path is for
    the cloud box (#10) where memory isn't the constraint. Kept trivial for now."""
    return "single"


def decide(request: str, context: str = "", auto_escalate: bool = False) -> Decision:
    scores = label_scores(request, context)
    route, candidates = route_from(scores)
    confidence = max(scores.values(), default=0.0)
    if route == "local":
        return Decision(
            route="local", label=cast(Label, candidates[0]), confidence=confidence
        )
    if auto_escalate:
        return Decision(route="escalate", confidence=confidence, candidates=candidates)
    return Decision(route="ask", confidence=confidence, candidates=candidates)
