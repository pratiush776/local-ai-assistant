"""LLM-as-judge: grade one answer against the expected answer -> PASS/FAIL + reason.

Validated before it is trusted: agreement() scores the judge against my own hand
labels (method 5.4). Local model first; escalate to Haiku only if TPR or TNR < 0.80.
"""
import httpx

from ..claude import client as claude_client

LOCAL_JUDGE = "qwen3.5:9b"
HAIKU_JUDGE = "claude-haiku-5-5"
JUDGE_MODEL = LOCAL_JUDGE  # default; validation may escalate to HAIKU_JUDGE
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


# ---------- glue (Claude wrote this): one non-streaming completion ----------
def _complete(system: str, user: str, model: str = JUDGE_MODEL) -> str:
    if model.startswith("claude"):  # Anthropic API (no temperature: Haiku 5.5 400s)
        reply = claude_client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in reply.content if b.type == "text")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": False,
        "options": {"temperature": 0},  # deterministic grading
    }
    r = httpx.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ======================= KERNEL — you type below =======================

# (1) The judge's instructions. Tell it: it is grading whether `actual` answers the
#     question as well as `expected` does; semantic match, not word-for-word; a right
#     paraphrase PASSes, a missing/wrong fact FAILs. Make it reply on the FIRST line
#     with exactly `PASS` or `FAIL`, then a one-line reason.
JUDGE_SYSTEM = """You grade whether a model's ANSWER is correct, using the EXPECTED \
answer as the ground truth.
Judge meaning, not wording: a correct paraphrase or an equivalent number, date or name \
PASSes; a missing, added or wrong fact, or any hallucination, FAILs.
Reply with exactly PASS or FAIL on the first line, then one short line with the reason."""


# (2) Grade one answer. Build the user message from (question, expected, actual),
#     call _complete(JUDGE_SYSTEM, ...), then read the verdict off the reply:
#     PASS if the reply starts with "pass" (case-insensitive), else FAIL.
def judge_one(
    question: str, expected: str, actual: str, model: str = JUDGE_MODEL
) -> tuple[str, str]:
    """Return ('PASS'|'FAIL', one-line reason)."""
    user = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}"
    )
    reply = _complete(JUDGE_SYSTEM, user, model).strip()
    verdict = "PASS" if reply.lower().startswith("pass") else "FAIL"
    lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
    reason = next((ln for ln in lines if ln.upper() not in ("PASS", "FAIL")), "")
    return verdict, reason

# (3) Score the judge against my labels. Both lists are "PASS"/"FAIL", same order.
#     TPR = of the rows I labelled PASS, the fraction the judge also called PASS.
#     TNR = of the rows I labelled FAIL, the fraction the judge also called FAIL.
def agreement(verdicts: list[str], labels: list[str]) -> tuple[float, float]:
    """Return (tpr, tnr)."""
    on_pass = [v for v, label in zip(verdicts, labels) if label == "PASS"]
    on_fail = [v for v, label in zip(verdicts, labels) if label == "FAIL"]
    tpr = on_pass.count("PASS") / len(on_pass) if on_pass else 0.0
    tnr = on_fail.count("FAIL") / len(on_fail) if on_fail else 0.0
    return tpr, tnr

# =======================================================================
