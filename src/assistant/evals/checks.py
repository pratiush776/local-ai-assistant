"""Objective pass/fail checks for one eval row — pure functions, no model, no DB.

Each answers a yes/no that code can settle on its own (method 5.4, code checks vs
judge):
  hit_at_k        - was the right source file retrieved?   (answerable rows only)
  refusal_correct - did it refuse exactly when it should?  (all rows)
  citation_valid  - does every [n] point to a real retrieved chunk?

The semantic question "is the answer right?" is NOT here — that is judge.py's job.
"""
from os.path import basename

# Refusal signatures the RAG system prompt steers the model toward ("say you don't
# have it truthfully"). This is a code check on surface phrasing; prose refusals that
# dodge every marker are caught later by the judge.
REFUSAL_MARKERS = (
    "does not contain",
    "no mention",
    "do not have",
    "not in the",
    "don't have",
)


def looks_like_refusal(answer: str, not_found: bool) -> bool:
    """True if the turn declined to answer: either the retrieval-empty flag, or the
    answer text carrying one of the refusal markers above."""
    if not_found:
        return True
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def hit_at_k(
    retrieved_sources: list[str], expected_sources: list[str], k: int = 5
) -> bool:
    """Was any expected source file among the top-k retrieved? Compares bare
    filenames, because the store keeps full paths and golden.jsonl keeps names.
    `retrieved_sources` is best-first; only call this on answerable rows."""
    topk = {basename(s) for s in retrieved_sources[:k]}
    want = {basename(s) for s in expected_sources}
    return bool(topk & want)


def refusal_correct(answer: str, not_found: bool, answerable: bool) -> bool:
    """Right behaviour = refuse exactly when the question is unanswerable.
    Refused on an answerable row, or answered an unanswerable one, is wrong."""
    return looks_like_refusal(answer, not_found) != answerable


def citation_valid(citation_ids: list[str], retrieved_ids: list[str]) -> bool:
    """Every cited chunk must be one that was actually retrieved — no invented [n].
    Vacuously true when there are no citations (e.g. a refusal cites nothing)."""
    retrieved = set(retrieved_ids)
    return all(cid in retrieved for cid in citation_ids)
