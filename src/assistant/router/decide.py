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

LABEL_RULES = """- action: needs a tool and your approval to change something in the world (send, book, set, buy, schedule, delete).
- fact: the user is asking for a specific real-world fact, name, date, number, or current/personal info they do not provide — choose fact EVEN IF you think you know it. (Asking you to create, write, suggest, or brainstorm something is NOT fact.)
- reason: answering needs calculation or multi-step deduction — choose reason even if it looks simple.
- chat: needs nothing external and no calculation — writing, rewriting, summarising, translating, giving an opinion, or creating/brainstorming/suggesting open-ended content (ideas, plans, lists), even if it draws on general knowledge.
If several fit, pick the first in this order: action, fact, reason, chat."""

LABEL_EXAMPLES = """Request: Write a short poem about autumn.
Answer: chat
Request: Create a workout plan to strengthen my upper body.
Answer: chat
Request: Give me some ideas for a 5-year-old's birthday party.
Answer: chat
Request: Who won the Nobel Prize in Physics in 2019?
Answer: fact
Request: What is the current population of Tokyo?
Answer: fact
Request: A shirt costs $40 after a 20% discount. What was the original price?
Answer: reason
Request: I have meetings at 2pm, 3pm and 4pm, each 45 minutes. Do any overlap?
Answer: reason
Request: Send an email to Sam asking to reschedule Friday.
Answer: action
Request: Set a reminder for my dentist appointment tomorrow at 9am.
Answer: action"""


class Decision(BaseModel):
    route: Route
    label: Label | None = None  # the winner when route == "local"
    confidence: float = 0.0  # the top label's probability
    candidates: list[str] = []  # the live labels when route != "local"


def label_scores(request: str, context: str = "") -> dict[str, float]:
    """One forward pass: read the first-token logprobs, return raw
    distribution over the 4 labels."""
    prompt = f"""Classify what the user's NEXT step needs. Reply with ONE word: action, fact, reason, or chat.
{LABEL_RULES}

Examples:
{LABEL_EXAMPLES}

Conversation so far:
{context or "(none)"}

Request: {request}
Answer:"""
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
