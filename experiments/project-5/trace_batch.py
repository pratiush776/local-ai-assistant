"""Batch real questions through the RAG pipeline → one JSONL trace per call
(scored retrieval + threshold decision + answer) for #5 error analysis.

Harness, not product: it drives rag.* directly and never touches the chat loop.
Prereqs to run: Postgres (ParadeDB) + Ollama up, corpus already ingested.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from assistant import rag
from assistant.ollama import ask_without_stream

# GENERATE=False → fast retrieval+rerank pass (RAM-safe on 18GB, see #4 memory note).
# Flip to True for the full pass that also captures the grounded answer (slow: ~86s/call
# when the laptop swaps).
GENERATE = True

# --- your three buckets: edit these to YOUR real usage (this part is yours, not AI's) ---
QUESTIONS = [
    # answerable from notes.md / personal_info.pdf
    "When is my dentist appointment?",
    "Who is my dentist?",
    "What is the recommended medication dose?",
    "What was Q3 revenue?",
    "How much did revenue grow in Q3?",
    "What are the monthly cloud hosting costs?",
    "When is my trip to Lisbon?",
    "What is my flight number to Lisbon?",
    "Where am I staying in Lisbon?",
    "What side project idea did I write down?",
    "What car do I drive?",
    "What is my car model?",
    # known failures (from #4)
    "Do I have an appointment anytime soon?",
    "What is my trip to France?",
    # should be not-found (honest-refusal check)
    "What is my dog's name?",
    "What is my bank account balance?",
    "What time is my flight home from Tokyo?",
]


def trace_one(question: str) -> dict:
    # STEP 1 — scored retrieval: the (cid, text, score) list rag_search hides  [THE LESSON]
    t = time.perf_counter()
    scored = rag.rag_retrieve(question)
    retrieve_ms = round((time.perf_counter() - t) * 1000)

    # STEP 2 — apply the threshold ourselves → kept vs dropped  [THE LESSON]
    kept = [(cid, text) for cid, text, s in scored if s >= rag.RERANK_THRESHOLD]

    # STEP 3 — grounded generation (or skip for a fast pass)
    answer, generate_ms = "", None
    if GENERATE and kept:
        ctx = rag.format_context(kept)
        t = time.perf_counter()
        answer = ask_without_stream(
            [
                {"role": "system", "content": rag.RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"},
            ],
            rag.GEN_MODEL,
        )
        generate_ms = round((time.perf_counter() - t) * 1000)

    # STEP 4 — the trace record. float(s) casts np.float32 → JSON-serializable  [THE LESSON]
    return {
        "query": question,
        "retrieved": [
            {"cid": cid, "score": round(float(s), 4)} for cid, _text, s in scored
        ],
        "kept": [cid for cid, _ in kept],
        "answer": answer,
        # retrieval-empty refusal only; a generation-refusal (LLM says "I don't have
        # that" despite kept chunks) is NOT caught here — you catch it in error analysis.
        "refused": not kept,
        "timings_ms": {"retrieve": retrieve_ms, "generate": generate_ms},
    }


def main() -> None:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = Path(__file__).parent / "traces" / f"run-{ts}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        for q in QUESTIONS:
            f.write(json.dumps(trace_one(q)) + "\n")
            f.flush()  # write-as-you-go: a crash on Q12 keeps Q1–11
            print(f"✓ {q[:55]}")
    print(f"\nwrote {out}  ({len(QUESTIONS)} traces, GENERATE={GENERATE})")


if __name__ == "__main__":
    main()
