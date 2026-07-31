import torch
from sentence_transformers import SentenceTransformer
import json
import time
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"

def load_chunks(path, limit=None):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks

chunks = load_chunks(CHUNKS_PATH, limit=5)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
print("Max sequence length:", model.max_seq_length)

# ---------------- TESTING EMBEDDING
# test_text = "search_document: Sony's Kumamoto facility produces image sensors."
# embedding = model.encode(test_text)

# print("Type:", type(embedding))
# print("Shape:", embedding.shape)
# print("First 5 values:", embedding[:5])

# ---------------- TESTING EMBEDDING SMALL RUN
# texts = [f"search_document: {c['text']}" for c in chunks]
# batch_size = 8  # small on purpose — this is a first test, not the real run

# start = time.time()
# embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
# elapsed = time.time() - start

# print("Shape:", embeddings.shape)
# print(f"Embedded {len(texts)} chunks in {elapsed:.2f}s "
#       f"({len(texts)/elapsed:.2f} chunks/sec)")

# ---------------- TESTING EMBEDDING TIMINGS
# sample_texts = [f"search_document: {c['text']}" for c in chunks] * 10
# print(f"Benchmarking on {len(sample_texts)} texts")

# # Warm-up: run once, small, and throw the result away — this pays the
# # one-time setup cost BEFORE we start the clock
# _ = model.encode(sample_texts[:4], batch_size=4)

# # Now time the real run
# start = time.time()
# embeddings = model.encode(sample_texts, batch_size=8, show_progress_bar=True)
# elapsed = time.time() - start

# rate = len(sample_texts) / elapsed
# print(f"Embedded {len(sample_texts)} chunks in {elapsed:.2f}s ({rate:.2f} chunks/sec)")

# hours_full_corpus = 111029 / rate / 3600
# print(f"Projected time for full 111,029-chunk corpus: {hours_full_corpus:.1f} hours")

# ---------------- TESTING CHROMADB - PUTTING TEXT AND GETTING IT BACK

client = chromadb.PersistentClient(path="./chroma_db", settings=chromadb.Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="supply_chain_docs")

# Re-embed just the 5 real chunks (not the x10 benchmark copies from last piece)
real_texts = [f"search_document: {c['text']}" for c in chunks]
real_embeddings = model.encode(real_texts, batch_size=8)

# Chroma metadata only accepts str/int/float/bool — not lists.
# relevant_events is a list, so it has to be flattened to a string first,
# or the .add() call below will throw an error.
metadatas = [{
    "company": c["company"],
    "source_type": c["source_type"],
    "form": c["form"],
    "published_int": c["published_int"],
    "section": c["section"],
    "relevant_events": ",".join(c["relevant_events"]),
} for c in chunks]

ids = [c["chunk_id"] for c in chunks]

collection.add(
    ids=ids,
    embeddings=real_embeddings.tolist(),  # Chroma wants plain lists, not numpy arrays
    documents=[c["text"] for c in chunks],
    metadatas=metadatas,
)

print("Collection count:", collection.count())
# Read one back and check it matches what went in
check = collection.get(ids=[chunks[0]["chunk_id"]], include=["metadatas", "documents"])


for chunk in chunks:
    if 'audit committee financial expert' in chunk['text'].lower():
        print("Chunk found: ",chunk['chunk_id'])

# ---------------- BUILD THE BM25 INDEX
def tokenize(text):
    return text.lower().split()

tokenized_corpus = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_rank(query, chunks, bm25_index):
    scores = bm25_index.get_scores(tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, score in ranked]

query = "audit committee financial expert"
bm25_ranking = bm25_rank(query, chunks, bm25)
print(f"BM25 Ranking: \n{bm25_ranking}\n")


def chroma_rank(query, collection, model, n_results=5):
    query_embedding = model.encode(f"search_query: {query}")
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    return results["ids"][0]

chroma_ranking = chroma_rank(query, collection, model)
print(f"ChromaDB Ranking: \n{chroma_ranking}\n")

def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

fused_result = reciprocal_rank_fusion([bm25_ranking, chroma_ranking])
print("Reciprocal Rank Fusion: \n")
for chunk_id, score in fused_result:
    print(f"{score:.4f} | {chunk_id}")

print("-------------------------------------------")
def matches_filter(chunk, filters):
    return all(chunk.get(key) == value for key, value in filters.items())

def bm25_rank(query, chunks, bm25_index, filters=None):
    scores = bm25_index.get_scores(tokenize(query))
    candidates = list(zip(chunks, scores))
    if filters:
        candidates = [(c, s) for c, s in candidates if matches_filter(c, filters)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c["chunk_id"] for c, s in candidates]

def chroma_rank(query, collection, model, filters=None, n_results=5):
    query_embedding = model.encode(f"search_query: {query}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=filters if filters else None,
    )
    return results["ids"][0]

def hybrid_search(query, chunks, bm25_index, collection, model, filters=None, top_n=5, k=60):
    bm25_ranking = bm25_rank(query, chunks, bm25_index, filters=filters)
    chroma_ranking = chroma_rank(query, collection, model, filters=filters, n_results=len(chunks))
    fused = reciprocal_rank_fusion([bm25_ranking, chroma_ranking], k=k)
    return fused[:top_n]

# Test 1: no filter — should match Piece 12's result
print(hybrid_search(query, chunks, bm25, collection, model))

# Test 2: a filter that should match everything here (sanity check the plumbing)
print(hybrid_search(query, chunks, bm25, collection, model, filters={"company": "panasonic"}))

# Test 3: a filter that matches nothing — should come back empty, not crash
print(hybrid_search(query, chunks, bm25, collection, model, filters={"company": "toyota"}))