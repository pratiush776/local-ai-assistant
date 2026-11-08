"""using rrf to fuse retrival from vector ranker and bm25 ranker"""

from collections import defaultdict
import numpy as np
import httpx
import bm25s


# vector ranker brute force
def brute_force_vector_search(sentences: list[str], query: str):
    URL = "http://localhost:11434/api/embed"
    MODEL = "nomic-embed-text"

    sentences1 = [f"search_document: {sentence}" for sentence in sentences]
    sentences1.append(f"search_query: {query}")

    r = httpx.post(URL, json={"model": MODEL, "input": sentences1})
    r.raise_for_status()
    data = r.json()

    embeddings = np.array(data["embeddings"])

    q = embeddings[-1]
    D = embeddings[:-1]
    scores = D @ q
    rank1 = np.argsort(scores)[::-1][:10]
    return rank1


# BM25
def bm25_search(sentences: list[str], query: str, threshold: int = 0):
    search_corpus = bm25s.tokenize(sentences, stopwords="en_plus")
    query_tokenized = bm25s.tokenize(query, stopwords="en_plus")

    retriever = bm25s.BM25()
    retriever.index(search_corpus)

    rank2, scores = retriever.retrieve(query_tokenized, k=10)
    return [rank for rank, score in zip(rank2[0], scores[0]) if score > threshold]


# rrf
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=lambda k: scores[k], reverse=True)


def main() -> None:
    query = input("Enter your query: ").strip()
    while query != "quit":
        sentences = open("data/hundred_sentences.txt").read().splitlines()
        rank1 = brute_force_vector_search(sentences, query)
        rank2 = bm25_search(sentences, query)
        rrf_scores = reciprocal_rank_fusion([list(rank1), rank2], 60)

        print("")
        for i in rrf_scores:
            print(f"{sentences[i]}")

        print("\nEnter 'quit' to exit.\n")
        query = input("Enter your query: ").strip()


if __name__ == "__main__":
    main()
