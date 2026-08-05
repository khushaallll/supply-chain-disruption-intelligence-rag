"""
embed_corpus.py

Embeds the full chunked knowledge base into a persistent ChromaDB collection.

Design notes:
- Resumable: chunk_ids already present in the Chroma collection are skipped,
  so an interrupted run (crash, laptop sleep, cluster session timeout) can
  be restarted without re-embedding everything from scratch.
- Batched: chunks are embedded and inserted in fixed-size batches, so memory
  use stays bounded regardless of corpus size.
- Logged in real time: every batch writes a line to the console AND a
  timestamped file under data/logs/, so progress can be watched from a
  second terminal (tail -f) while this runs unattended.

Usage:
    python src/retrieval/embed_corpus.py
    python src/retrieval/embed_corpus.py --batch_size 256      # bigger batch on GPU
    python src/retrieval/embed_corpus.py --limit 500           # quick test run first
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------- #
# Configuration — the values you're likely to change between runs.
# --------------------------------------------------------------------------- #

DEFAULT_CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "supply_chain_docs"
DEFAULT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_BATCH_SIZE = 8            # safe default for CPU — raise this on GPU
CHECKPOINT_EVERY_N_BATCHES = 20   # how often to sanity-check persisted count


def parse_args():
    parser = argparse.ArgumentParser(description="Embed corpus chunks into ChromaDB.")
    parser.add_argument("--chunks_path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--chroma_path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection_name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only embed the first N chunks — for a quick test run.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Logging — writes to console AND a timestamped file under data/logs/.
# Python's logging module flushes to the file on every call by default, so
# this is genuinely real-time, not buffered until the process exits.
# --------------------------------------------------------------------------- #

def setup_logging():
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"embed_corpus_{datetime.now():%Y%m%d_%H%M%S}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Backstop for a known chromadb/posthog version-mismatch bug: if the env
    # var above doesn't fully suppress it, strip just this one noisy line so
    # the log file stays readable during an unattended multi-hour run.
    class SuppressChromaTelemetryNoise(logging.Filter):
        def filter(self, record):
            return "Failed to send telemetry event" not in record.getMessage()

    for handler in logging.getLogger().handlers:
        handler.addFilter(SuppressChromaTelemetryNoise())

    logging.info(f"Logging to {log_path}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_chunks(path, limit=None):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def chunk_to_metadata(chunk):
    """Chroma metadata values must be str/int/float/bool — relevant_events
    is a list, so it's flattened to a comma-joined string before insert."""
    return {
        "company": chunk["company"],
        "source_type": chunk["source_type"],
        "form": chunk["form"],
        "published_int": chunk["published_int"],
        "section": chunk["section"],
        "relevant_events": ",".join(chunk.get("relevant_events", [])),
    }


# --------------------------------------------------------------------------- #
# Resumability — skip chunk_ids already in the collection.
# --------------------------------------------------------------------------- #

def get_already_embedded_ids(collection):
    # include=[] skips fetching documents/metadatas/embeddings — ids are
    # always returned regardless, so this stays a lightweight call even
    # once the collection holds 100k+ entries.
    existing = collection.get(include=[])
    return set(existing["ids"])


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    setup_logging()

    logging.info("=== Corpus embedding run starting ===")
    logging.info(f"chunks_path={args.chunks_path} batch_size={args.batch_size} "
                 f"model={args.model_name} limit={args.limit}")

    # ---- Load chunks --------------------------------------------------- #
    chunks = load_chunks(args.chunks_path, limit=args.limit)
    total_chunks = len(chunks)
    logging.info(f"Loaded {total_chunks} chunks from {args.chunks_path}")
    if total_chunks == 0:
        logging.error("Zero chunks loaded — check chunks_path before going further.")
        return

    # ---- Device + model -------------------------------------------------- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if device == "cpu":
        logging.warning("Running on CPU — this will be slow for the full corpus "
                         "(benchmark: ~0.8 chunks/sec measured locally).")

    model = SentenceTransformer(args.model_name, trust_remote_code=True, device=device)
    expected_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Model loaded. Max sequence length: {model.max_seq_length}, "
                 f"embedding dimension: {expected_dim}")

    # ---- Chroma setup ------------------------------------------------------ #
    client = chromadb.PersistentClient(
        path=args.chroma_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=args.collection_name)
    logging.info(f"Chroma collection '{args.collection_name}' opened at "
                 f"'{args.chroma_path}'. Existing count: {collection.count()}")

    # ---- Resumability: drop chunks already embedded ------------------------ #
    already_embedded = get_already_embedded_ids(collection)
    if already_embedded:
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in already_embedded]
        logging.info(f"Resuming: {before - len(chunks)} chunks already embedded, "
                     f"skipping. {len(chunks)} remaining.")
    if not chunks:
        logging.info("Nothing left to embed — collection is already up to date.")
        return

    # ---- Batch loop --------------------------------------------------- #
    start_time = time.time()
    n_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    chunks_done = 0
    errors = 0

    for batch_num in range(n_batches):
        batch = chunks[batch_num * args.batch_size: (batch_num + 1) * args.batch_size]

        try:
            texts = [f"search_document: {c['text']}" for c in batch]
            embeddings = model.encode(texts, batch_size=args.batch_size)

            # --- Health checks on this batch, before trusting the output --- #
            if embeddings.shape[0] != len(batch):
                raise ValueError(f"Row count mismatch: got {embeddings.shape[0]} "
                                 f"embeddings for {len(batch)} chunks.")
            if embeddings.shape[1] != expected_dim:
                raise ValueError(f"Embedding dim mismatch: got {embeddings.shape[1]}, "
                                 f"expected {expected_dim}.")
            if np.isnan(embeddings).any():
                raise ValueError("NaN values found in embeddings for this batch.")

            collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings.tolist(),
                documents=[c["text"] for c in batch],
                metadatas=[chunk_to_metadata(c) for c in batch],
            )
            chunks_done += len(batch)

        except Exception as e:
            errors += 1
            failed_ids = [c["chunk_id"] for c in batch]
            logging.error(f"Batch {batch_num + 1}/{n_batches} FAILED: {e}. "
                          f"chunk_ids affected: {failed_ids}")
            if device == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logging.info(f"Cleared CUDA cache after OOM on batch {batch_num + 1}.")
            continue  # one bad batch shouldn't cost the whole multi-hour run

        # --- Progress signal, every batch --- #
        elapsed = time.time() - start_time
        rate = chunks_done / elapsed if elapsed > 0 else 0
        remaining = len(chunks) - chunks_done
        eta_minutes = (remaining / rate / 60) if rate > 0 else float("inf")

        logging.info(f"Batch {batch_num + 1}/{n_batches} | "
                     f"{chunks_done}/{len(chunks)} chunks | "
                     f"{rate:.2f} chunks/sec | ETA {eta_minutes:.1f} min")

        # --- Periodic checkpoint: confirm persisted count matches what we
        #     think we've inserted — catches a silent Chroma insert failure --- #
        if (batch_num + 1) % CHECKPOINT_EVERY_N_BATCHES == 0:
            persisted = collection.count()
            expected = len(already_embedded) + chunks_done
            status = "OK" if persisted == expected else "MISMATCH"
            logging.info(f"CHECKPOINT | persisted={persisted} expected={expected} "
                         f"[{status}]")

        # --- Periodic GPU memory cleanup: release cached-but-unallocated
        #     memory back to the pool, to prevent fragmentation building up
        #     over hundreds of batches (root cause of the CUDA OOM at batch 511) --- #
        if device == "cuda" and (batch_num + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # ---- Final summary --------------------------------------------------- #
    total_elapsed = time.time() - start_time
    final_count = collection.count()
    expected_final = len(already_embedded) + chunks_done

    logging.info("=== Run complete ===")
    logging.info(f"Chunks embedded this run: {chunks_done} | Errors: {errors} | "
                 f"Time: {total_elapsed/60:.1f} min | "
                 f"Avg rate: {chunks_done/total_elapsed:.2f} chunks/sec")
    logging.info(f"Final collection count: {final_count} (expected {expected_final})")

    if final_count != expected_final:
        logging.warning("Final count does not match expected total — "
                        "investigate before treating this run as complete.")
    else:
        logging.info("SUCCESS: persisted count matches expected total.")

    # --- Second, independent check: does the collection match the FULL
    #     corpus file, not just what this run attempted? This is the check
    #     that would have caught the 39,168-chunk gap from failed batches. --- #
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        total_corpus_chunks = sum(1 for _ in f)

    if final_count != total_corpus_chunks:
        missing = total_corpus_chunks - final_count
        logging.warning(f"CORPUS INCOMPLETE: collection has {final_count} chunks, "
                        f"but {args.chunks_path} has {total_corpus_chunks} — "
                        f"{missing} chunks still missing. Re-run this script to "
                        f"retry only the missing ones (resumability will skip "
                        f"everything already embedded).")
    else:
        logging.info(f"VERIFIED: collection count ({final_count}) matches the "
                     f"full corpus file ({total_corpus_chunks}). Nothing missing.")


if __name__ == "__main__":
    main()