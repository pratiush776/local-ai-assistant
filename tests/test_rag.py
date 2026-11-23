from assistant.knowledge import rag


def test_format_context_numbers_hits_1_indexed():
    hits = [("aaa", "the first chunk"), ("bbb", "the second chunk")]
    out = rag.format_context(hits)

    assert "[1] (source=aaa) the first chunk" in out
    assert "[2] (source=bbb) the second chunk" in out
    # strongest hit (passed first) is numbered [1]
    assert out.index("[1]") < out.index("[2]")


def test_format_context_empty_is_empty_string():
    assert rag.format_context([]) == ""


def _stub(monkeypatch, hits, scores, *, rerank=True):
    """Make rag_search deterministic: canned hits + a reranker returning canned scores,
    so we test the sort/threshold logic without Postgres, Ollama, or the real model.
    `hits` are (cid, text); hybrid_retrieve now also carries an rrf score, stubbed here."""
    monkeypatch.setattr(rag, "RERANK_ENABLED", rerank)
    monkeypatch.setattr(
        rag, "hybrid_retrieve", lambda q, k=rag.TOP_K: [(c, t, 0.02) for c, t in hits]
    )

    class _FakeReranker:
        def predict(self, pairs):
            return scores

    monkeypatch.setattr(rag, "_reranker", lambda: _FakeReranker())


def test_rag_search_keeps_only_above_threshold(monkeypatch):
    hits = [("a", "relevant"), ("b", "noise")]
    scores = [rag.RERANK_THRESHOLD + 0.1, rag.RERANK_THRESHOLD - 0.05]
    _stub(monkeypatch, hits, scores)

    assert rag.rag_search("q") == [("a", "relevant")]


def test_rag_search_sorts_by_score_descending(monkeypatch):
    hits = [("a", "weaker"), ("b", "stronger")]
    scores = [rag.RERANK_THRESHOLD + 0.1, rag.RERANK_THRESHOLD + 0.5]
    _stub(monkeypatch, hits, scores)

    # best score first, regardless of retrieve order
    assert rag.rag_search("q") == [("b", "stronger"), ("a", "weaker")]


def test_rag_search_all_below_threshold_is_not_found(monkeypatch):
    hits = [("a", "x"), ("b", "y")]
    scores = [rag.RERANK_THRESHOLD - 0.01, 0.0]
    _stub(monkeypatch, hits, scores)

    assert rag.rag_search("q") == []


def test_rag_search_rerank_off_keeps_all_in_hybrid_order(monkeypatch):
    # reranker OFF (the default): hybrid order goes straight through, nothing dropped,
    # and the cross-encoder is never consulted (its "scores" would reject everything).
    hits = [("a", "first"), ("b", "second")]
    _stub(monkeypatch, hits, scores=[0.0, 0.0], rerank=False)

    assert rag.rag_search("q") == [("a", "first"), ("b", "second")]
