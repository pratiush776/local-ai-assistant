"""keyword search lane using bm25s"""

import bm25s

sentences = open("data/hundred_sentences.txt").read().splitlines()

# tokanize the corpus and build the BM25 index(once)
corpus_tokens = bm25s.tokenize(sentences, stopwords="en")  # splits + drops stopwords
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# query - tokenize the query
query = "how do I take care of my pet dog"
query_tokens = bm25s.tokenize(query, stopwords="en")

# retrieve
idx, scores = retriever.retrieve(query_tokens, k=5)
for i, x in zip(idx[0], scores[0]):
    print(f"{x:.3f} {sentences[i]}")
