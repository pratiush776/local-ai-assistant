"""RAG orchestration — backend-agnostic.

This module owns everything that is the same no matter where chunks live:
embedding, the cross-encoder reranker, the kept-set thresholds, RRF_K, and the
grounding prompt. The actual storage + hybrid query live behind a
`RetrievalBackend` (see `retrieval/`), chosen by the RETRIEVAL_BACKEND env var.
The six functions at the bottom just delegate to the active backend, so swapping
ParadeDB <-> SQLite never touches the callers (respond, tools, gui, trace, evals).
"""

from typing import Literal

import httpx
import numpy as np
from sentence_transformers import CrossEncoder

from assistant.retrieval import get_backend

# --- config (single source of truth, backend-agnostic) ---
OLLAMA_URL = "http://localhost:11434/api/"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3.5:9b"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RRF_K = 60
# recall-favouring pre-filter, not the honesty gate: no single threshold separates
# relevant from noise (France/Lisbon noise scored 0.31 > real answers at 0.24), so we
# keep this low and let grounded generation refuse near-miss noise. Provisional until
# the personal-files eval (#5) calibrates it. See ROADMAP "To review".
RERANK_THRESHOLD = 0.01
# KEEP_N: when set, keep the top-N chunks by rank and ignore RERANK_THRESHOLD.
# None -> the old score-threshold behaviour. A fixed cut is kept-set-size-invariant,
# so it isolates the reranker (5.5) and sidesteps the uncalibrated threshold.
KEEP_N: int | None = None
TOP_K = 10
# The cross-encoder is OFF by default: it costs ~2.3 GB RAM + a cold ~7 s load on an
# 18 GB laptop that already swaps. Off -> the hybrid RRF order goes straight to
# generation. Lesson 5.5 measures whether it earns its memory and may flip this.
RERANK_ENABLED = False
PERSONA_TODAY = "2024-05-15"  # the assistant's clock while it runs as the persona
_RERANKER = None  # module-level cache, starts empty
RAG_SYSTEM_PROMPT = """Answer using ONLY the numbered context below. Cite the source number [n] with exact text used from the context to create the response
for reference so that the user can trace it back for every fact respectively.
If the context does not contain the answer, say you don't have it truthfully."""


def embed(
    texts: list[str], task_prefix: Literal["search_query", "search_document"]
) -> np.ndarray:
    payload = {
        "model": EMBED_MODEL,
        "input": [f"{task_prefix}: {text}" for text in texts],
    }
    r = httpx.post(OLLAMA_URL + "embed", json=payload, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


# ---------------------------- reranking / kept-set (backend-agnostic) ----------------------------
def _reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_NAME)
    return _RERANKER


def warmup_reranker() -> None:
    _reranker()


def rag_retrieve(question: str) -> list[tuple[str, str, float]]:
    """Every candidate as (chunk_id, text, score), best first.
    Reranker ON  -> score is the cross-encoder relevance score.
    Reranker OFF -> score is the hybrid RRF score (order unchanged)."""
    hits = hybrid_retrieve(question, TOP_K)  # (cid, text, rrf)
    if not RERANK_ENABLED:
        return hits
    scores = _reranker().predict([[question, text] for _, text, _ in hits])
    reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [(cid, text, float(score)) for (cid, text, _), score in reranked]


def rag_search(question: str) -> list[tuple[str, str]]:
    chunks = rag_retrieve(question)  # best-first
    if KEEP_N is not None:
        return [(cid, text) for cid, text, _ in chunks[:KEEP_N]]
    return [(cid, text) for cid, text, score in chunks if score >= RERANK_THRESHOLD]


def format_context(hits: list[tuple[str, str]]) -> str:
    return "".join(
        [f"[{i + 1}] (source={cid}) {text}\n" for i, (cid, text) in enumerate(hits)]
    )


# ---------------------------- backend delegation ----------------------------
# These six + wipe are the only storage-specific calls; the active backend
# (ParadeDB or SQLite) implements them. Kept as module-level names so tests can
# still monkeypatch rag.hybrid_retrieve.
def hybrid_retrieve(question: str, k: int = TOP_K) -> list[tuple[str, str, float]]:
    return get_backend().hybrid_retrieve(question, k)


def ingest(path: str) -> dict:
    return get_backend().ingest(path)


def wipe() -> int:
    return get_backend().wipe()


def list_sources() -> list[tuple[str, int]]:
    return get_backend().list_sources()


def chunks_for_source(source: str) -> list[tuple[str, str]]:
    return get_backend().chunks_for_source(source)


def sources_for_chunks(cids: list[str]) -> dict[str, str]:
    return get_backend().sources_for_chunks(cids)


def texts_for_chunks(cids: list[str]) -> dict[str, str]:
    return get_backend().texts_for_chunks(cids)
