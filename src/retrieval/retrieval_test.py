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


CHUNKS_PATH = "/home/shubham.ghanmode/supply-chain-disruption-intelligence-rag/data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"
CHROMA_PATH = "/home/shubham.ghanmode/supply-chain-disruption-intelligence-rag/chroma_db"
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
    for key, value in filters.items():
        if key == "published_int_max":
            if chunk.get("published_int", 0) > value:
                return False
        else:
            if chunk.get(key) != value:
                return False
    return True

def build_chroma_where(filters):
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if key == "published_int_max":
            conditions.append({"published_int": {"$lte": value}})
        else:
            conditions.append({key: value})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}

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
        where=build_chroma_where(filters),
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
        print(f"  {score:.4f} | {chunk['company']:20s} | {chunk['published_int']} | {chunk_id}")
        print(f"           {preview}...")


def collection_stats():
    print(f"Total chunks in corpus file: {len(chunks)}")
    print(f"Total chunks in Chroma:      {collection.count()}")
    print(f"Distinct companies: {len(set(c['company'] for c in chunks))}")


def resolve_company(keyword):
    """Find the exact `company` metadata string matching a keyword.

    Filters do an EXACT match against chunk["company"], so guessing the
    wrong casing/spelling (e.g. "toyota" vs "toyota motor") makes a filter
    silently return zero results — indistinguishable from a genuine
    confirmed-absent company unless checked first. Prints what it found
    so a wrong guess is visible before it's mistaken for a real finding.
    """
    unique_companies = sorted(set(c["company"] for c in chunks))
    matches = [c for c in unique_companies if keyword.lower() in c.lower()]
    if not matches:
        print(f"  [resolve_company] '{keyword}': NOT FOUND in corpus company list")
        return keyword  # fall back to the raw guess — for confirmed-absent
                        # tests this is fine, since any spelling returns empty
    if len(matches) > 1:
        print(f"  [resolve_company] '{keyword}': multiple matches {matches} — using '{matches[0]}'")
    else:
        print(f"  [resolve_company] '{keyword}' -> '{matches[0]}'")
    return matches[0]


# --------------------------------------------------------------------------- #
# The 20 hand-written test queries — the Day 6 gate check.
#
# Every query below is grounded in a specific field from
# ground_truth_batch{1,2,3}_final.json or CORPUS_COVERAGE_PROBLEM.md — not
# guessed. Each comment states: which event it comes from, what field backs
# the expectation, and what a PASS actually looks like. Read the printed
# results against that expectation — don't just check that something came
# back, check that the RIGHT thing came back.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    collection_stats()

    # Resolve the exact corpus spelling for companies used in filtered tests
    # below, once, up front — so a filter test failing later is a genuine
    # finding, not a silently wrong string.
    print("\nResolving company name strings used in filtered tests...")
    TOYOTA = resolve_company("toyota")
    PANASONIC = resolve_company("panasonic")

    # ----------------------------------------------------------------- #
    # Category 1 — Positive matches: has_corpus_doc == true
    # A real, verified document for this company exists pre-dating the
    # event. PASS = that company's chunks appear in the top results.
    # A boilerplate risk-factors chunk from the RIGHT company still counts
    # as a pass — it doesn't need to be the single most dramatic sentence.
    # ----------------------------------------------------------------- #

    # print("\n=== 1. Toyota — event 2 (nippon_steel_2011), has_corpus_doc=true ===")
    # print_results("Toyota Motor Corporation supply chain earthquake risk")

    # print("\n=== 2. Sony — event 2, has_corpus_doc=true ===")
    # print_results("Sony Corporation Japan supply disruption")

    # print("\n=== 3. Panasonic — event 2, has_corpus_doc=true (same company as the 5-chunk prototype test) ===")
    # print_results("Panasonic Corporation production risk Japan")

    # print("\n=== 4. Hitachi — event 2, has_corpus_doc=true ===")
    # print_results("Hitachi Ltd earthquake production halt")

    # print("\n=== 5. General Motors — events 2 and 47 (china_steel_2018), has_corpus_doc=true in both ===")
    # print_results("General Motors steel tariff commodity cost")

    # print("\n=== 6. Ford — event 47, has_corpus_doc=true. Real phrase from Ford's actual 10-Q "
    #       "per the Day 4-5 disclosure-lag finding ===")
    # print_results("Ford Motor Company commodity cost tariff")

    # print("\n=== 7. ExxonMobil — event 4 (dow_2017), has_corpus_doc=true ===")
    # print_results("Exxon Mobil hurricane refinery disruption")

    # print("\n=== 8. Alcoa — event 35 (gazprom_pjsc_2022), has_corpus_doc=true ===")
    # print_results("Alcoa aluminum smelter energy supply")

    # print("\n=== 9. Rio Tinto — events 11 (aurizon_2010) and 57 (rusal_2018), has_corpus_doc=true in both ===")
    # print_results("Rio Tinto coal export rail disruption")

    # # ----------------------------------------------------------------- #
    # # Category 2 — Restraint: events with ZERO affected companies
    # # ground_truth_affected is genuinely empty for these three. PASS =
    # # no confident, specific affected-company match — the system should
    # # show weak/generic results, not confidently name a company as harmed.
    # # ----------------------------------------------------------------- #

    # print("\n=== 10. RESTRAINT — event 18 (chevron_2012), ground_truth_affected is empty ===")
    # print_results("Chevron Richmond refinery fire 2012")

    # print("\n=== 11. RESTRAINT — event 33 (kyushu_electric_power_2018), ground_truth_affected is empty ===")
    # print_results("Kyushu Electric solar curtailment 2018")

    # print("\n=== 12. RESTRAINT — event 34 (korea_electric_power_2011), ground_truth_affected is empty ===")
    # print_results("Korea Electric Power blackout 2011")

    # # ----------------------------------------------------------------- #
    # # Category 3 — Filters on companies confirmed absent from the corpus
    # # PASS = empty result. Any spelling works here — if the company isn't
    # # in the corpus at all, no filter string will accidentally match it.
    # # ----------------------------------------------------------------- #

    # print("\n=== 13. Filter — SUMCO, event 21: graph-present but has_corpus_doc=false. "
    #       "PASS = empty (no document exists, even if the graph knows the entity) ===")
    # print_results("audit committee risk factors", filters={"company": "sumco corporation"})

    # print("\n=== 14. Filter — Korinox, event 1: in_graph_context=false, private Korean company, "
    #       "no SEC presence. PASS = empty ===")
    # print_results("supply chain disruption impact", filters={"company": "korinox"})

    # print("\n=== 15. Filter — Unimicron, event 40: in_graph_context=false. PASS = empty ===")
    # print_results("supply chain disruption impact", filters={"company": "unimicron technology corp"})

    # print("\n=== 16. Filter — Transocean, event 24 (Deepwater Horizon 'trap' event): "
    #       "in_graph_context=false. PASS = empty ===")
    # print_results("drilling rig explosion supply impact", filters={"company": "transocean"})

    # # ----------------------------------------------------------------- #
    # # Category 4 — Filter plumbing: filtered vs. unfiltered, same query
    # # PASS = the filtered run returns Toyota-only chunks; the unfiltered
    # # run is free to return anyone. Tests the filter mechanism itself,
    # # not a ground-truth claim.
    # # ----------------------------------------------------------------- #

    # print("\n=== 17a. Same query, NO filter — can return any company ===")
    # print_results("supply chain risk factors")

    # print("\n=== 17b. Same query, FILTERED to Toyota — every result's company must equal the resolved Toyota string ===")
    # print_results("supply chain risk factors", filters={"company": TOYOTA})

    # # ----------------------------------------------------------------- #
    # # Category 5 — Date-boundary leakage check
    # # PASS = every returned chunk's published_int is on or before the
    # # event's date_news_first. Any chunk dated AFTER the event date is a
    # # leakage bug — the system would be "predicting" something it read
    # # about after the fact.
    # # ----------------------------------------------------------------- #

    print("\n=== 18. Leakage check — event 2 date_news_first = 2011-03-11. "
          "Manually inspect published_int on every result below; none should exceed 20110311 ===")
    print_results("Tohoku earthquake production impact Japan", filters={"published_int_max": 20110311})

    print("\n=== 19. Leakage check — event 47 date_news_first = 2018-03-01. "
          "None of the results below should have published_int after 20180301 ===")
    print_results("steel tariff Section 232 cost increase", filters={"published_int_max": 20180301})




    # # ----------------------------------------------------------------- #
    # # Category 6 — Exact-phrase / generic-boilerplate stress test
    # # This re-runs the Day 6 distance-metric finding (Section 7 of the
    # # implementation log) at full corpus scale instead of 5 chunks.
    # # Watch specifically whether hybrid fusion still corrects for dense
    # # retrieval's weakness on short, jargon-heavy, low-topic-diversity text
    # # now that real company variety exists in the corpus.
    # # ----------------------------------------------------------------- #

    # print("\n=== 20a. Stress test — exact phrase, unfiltered. The original 5-chunk finding, "
    #       "now at full scale ===")
    # print_results("audit committee financial expert")

    # print("\n=== 20b. Stress test — same phrase, filtered to Panasonic (the original test's company) ===")
    # print_results("audit committee financial expert", filters={"company": PANASONIC})

    # print("\nDone — 20 queries run. Read each block above against its stated PASS "
    #       "condition, not just for 'did something come back'.")