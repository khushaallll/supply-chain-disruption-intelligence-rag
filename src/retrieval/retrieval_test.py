"""
test_retrieval.py

Loads the ChromaDB collection built by embed_corpus.py, rebuilds the BM25
index (cheap — seconds, not hours, so it's rebuilt fresh rather than saved
to disk), and exposes hybrid_search() plus a few inspection helpers.

This file is meant to be edited — add your 20 hand-written test queries at
the bottom and read through what comes back.
"""

import json
import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "supply_chain_docs"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return text.lower().split()


# --------------------------------------------------------------------------- #
# Setup — runs once when the script starts
# --------------------------------------------------------------------------- #

print("Loading chunks...")
chunks = load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks")

print("Building BM25 index (fast, rebuilt fresh each run)...")
tokenized_corpus = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

print("Loading embedding model + connecting to ChromaDB...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Chroma collection count: {collection.count()}")
if collection.count() != len(chunks):
    print(f"WARNING: Chroma has {collection.count()} chunks but the corpus file "
          f"has {len(chunks)}. Did embed_corpus.py finish? Same file on both sides?")


# --------------------------------------------------------------------------- #
# Retrieval logic — same as validated earlier, with one change: n_candidates
# bounds how deep each retriever looks before fusing, instead of asking
# Chroma to rank the entire corpus (fine at 5 chunks, wasteful at 111k).
# --------------------------------------------------------------------------- #

def matches_filter(chunk, filters):
    return all(chunk.get(key) == value for key, value in filters.items())


def bm25_rank(query, filters=None):
    scores = bm25.get_scores(tokenize(query))
    candidates = list(zip(chunks, scores))
    if filters:
        candidates = [(c, s) for c, s in candidates if matches_filter(c, filters)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, s in candidates]


def chroma_rank(query, filters=None, n_results=50):
    query_embedding = model.encode(f"search_query: {query}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=filters if filters else None,
    )
    return results["ids"][0]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, filters=None, top_n=5, n_candidates=50, k=60):
    bm25_ranking = bm25_rank(query, filters=filters)[:n_candidates]
    chroma_ranking = chroma_rank(query, filters=filters, n_results=n_candidates)
    fused = reciprocal_rank_fusion([bm25_ranking, chroma_ranking], k=k)
    return fused[:top_n]


# --------------------------------------------------------------------------- #
# Inspection helpers — for manually reading through results
# --------------------------------------------------------------------------- #

_chunk_lookup = {c["chunk_id"]: c for c in chunks}


def get_chunk_by_id(chunk_id):
    return _chunk_lookup.get(chunk_id)


def print_results(query, filters=None, top_n=5):
    print(f"\nQuery: '{query}'  |  filters: {filters}")
    results = hybrid_search(query, filters=filters, top_n=top_n)
    if not results:
        print("  (no results)")
    for chunk_id, score in results:
        chunk = get_chunk_by_id(chunk_id)
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"  {score:.4f} | {chunk['company']:20s} | {chunk_id}")
        print(f"           {preview}...")


def collection_stats():
    print(f"Total chunks in corpus file: {len(chunks)}")
    print(f"Total chunks in Chroma:      {collection.count()}")
    print(f"Distinct companies: {len(set(c['company'] for c in chunks))}")


# --------------------------------------------------------------------------- #
# Your 20 test queries go here
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    collection_stats()

    print_results("audit committee financial expert")
    print_results("audit committee financial expert", filters={"company": "panasonic"})

    # Add the rest of your 20 hand-written queries below, one print_results()
    # call each — that's the actual Day 6 gate check.