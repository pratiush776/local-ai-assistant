"""The retrieval backend seam.

`rag.py` owns everything backend-agnostic (embedding, the reranker, thresholds,
RRF_K, the kept-set logic). A *backend* owns only what is specific to one storage
engine: how chunks are stored, how the hybrid vector+keyword query runs, and the
read-only views the GUI/trace need. Swapping engines means writing one new class
that satisfies this Protocol and flipping an env var — never a refactor.

Today: ParadeDB (Postgres: pgvector + pg_search) for a hosted/large corpus,
SQLite (FTS5 bm25 + sqlite-vec brute-force) for a light local corpus.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrievalBackend(Protocol):
    """One storage engine behind the chunk store. All scores/ranks are the
    backend's own; rag.py reranks and thresholds on top, identically for both."""

    def hybrid_retrieve(self, question: str, k: int) -> list[tuple[str, str, float]]:
        """Top-k (chunk_id, text, rrf_score), best first — the fused
        vector + keyword result. The one method on the hot path."""
        ...

    def ingest(self, path: str) -> dict:
        """Chunk + embed + upsert one file. Returns {inserted, deleted, total}."""
        ...

    def wipe(self) -> int:
        """Empty the chunk store; return how many chunks were removed.
        Backs `ingest_folder --fresh` without reaching into a DSN."""
        ...

    # ---- read-only views (GUI Memory panel + citation tracing) ----
    def list_sources(self) -> list[tuple[str, int]]:
        """Every ingested file and its chunk count."""
        ...

    def chunks_for_source(self, source: str) -> list[tuple[str, str]]:
        """The (chunk_id, text) units of one file."""
        ...

    def sources_for_chunks(self, cids: list[str]) -> dict[str, str]:
        """Map cited chunk ids back to their source file."""
        ...

    def texts_for_chunks(self, cids: list[str]) -> dict[str, str]:
        """Map chunk ids to their full text (the trace sidecar freezes these)."""
        ...
