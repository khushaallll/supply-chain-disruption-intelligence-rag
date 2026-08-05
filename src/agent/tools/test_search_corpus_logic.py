"""
test_search_corpus_logic.py

Tests everything in search_corpus_tool.py that does NOT require the real
embedding model, a real ChromaDB collection, or the real corpus file --
fusion, filtering, leakage-guard translation, company resolution, and
result/status wrapping.

This does NOT validate retrieval QUALITY (is the right chunk actually
ranked highly for a given real query). That was already validated
separately via the 20 hand-written test queries in retrieval_test.py, using
the real corpus, the real embedding model, and the real Chroma collection.
This file only checks that search_corpus() wraps that already-validated
machinery correctly -- that the leakage guard actually excludes what it
should, that fusion combines two ranked lists the way reciprocal-rank
fusion is supposed to, that a misspelled company filter is reported
distinctly from a genuine empty result, and so on.

Strategy: never call the real __init__ (which needs chromadb/torch/
sentence_transformers -- none of which are installed in this environment).
Instead, build a store via CorpusSearchStore.__new__(...) and set its
attributes directly to small, hand-controlled fakes:
  - FakeBM25 replaces only the score-generation step; the REAL
    _bm25_rank() code (including matches_filter, the leakage guard) still
    runs untouched.
  - FakeCollection replaces the real Chroma network call and returns a
    fixed, pre-specified id list, while recording the `where` clause it
    was given -- so build_chroma_where() is exercised end-to-end too.

Run: python test_search_corpus_logic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_corpus_tool import (
    CorpusSearchStore, date_to_published_int, matches_filter,
    build_chroma_where, reciprocal_rank_fusion, resolve_company_name,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ===========================================================================
# PART A -- pure function tests, no class involved at all
# ===========================================================================
print("=" * 70)
print("PART A -- pure function tests")
print("=" * 70)

check("date_to_published_int('2011-03-11') == 20110311",
      date_to_published_int("2011-03-11") == 20110311,
      date_to_published_int("2011-03-11"))

check("matches_filter: chunk before cutoff passes",
      matches_filter({"published_int": 20110101}, {"published_int_max": 20110311}))
check("matches_filter: chunk AFTER cutoff is excluded (the leakage guard, in isolation)",
      not matches_filter({"published_int": 20110401}, {"published_int_max": 20110311}))
check("matches_filter: chunk exactly ON the cutoff date passes (inclusive)",
      matches_filter({"published_int": 20110311}, {"published_int_max": 20110311}))
check("matches_filter: company equality filter matches",
      matches_filter({"company": "toyota motor"}, {"company": "toyota motor"}))
check("matches_filter: company equality filter rejects a near-miss",
      not matches_filter({"company": "toyota motor"}, {"company": "toyota"}))

where1 = build_chroma_where({"published_int_max": 20110311})
check("build_chroma_where: single filter becomes a bare $lte clause",
      where1 == {"published_int": {"$lte": 20110311}}, where1)

where2 = build_chroma_where({"published_int_max": 20110311, "company": "toyota motor"})
check("build_chroma_where: two filters get wrapped in $and",
      where2 == {"$and": [{"published_int": {"$lte": 20110311}}, {"company": "toyota motor"}]},
      where2)

check("build_chroma_where: no filters returns None",
      build_chroma_where(None) is None)
check("build_chroma_where: empty dict returns None",
      build_chroma_where({}) is None)

fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
fused_ids = [cid for cid, _ in fused]
check("RRF: 'b' ranks first -- appears near the top of both input lists",
      fused_ids[0] == "b", fused_ids)
check("RRF: items appearing in only one list still make it into the fused result",
      "a" in fused_ids and "d" in fused_ids, fused_ids)
check("RRF: empty input lists produce an empty fused result",
      reciprocal_rank_fusion([[], []]) == [])

check("resolve_company_name: case-insensitive substring match",
      resolve_company_name("TOYOTA", ["toyota motor", "sony corp"]) == "toyota motor")
check("resolve_company_name: no match returns None, not an empty string",
      resolve_company_name("nonexistent_xyz", ["toyota motor", "sony corp"]) is None)


# ===========================================================================
# PART B -- search_corpus() orchestration, via a fake store
# ===========================================================================
print()
print("=" * 70)
print("PART B -- search_corpus() orchestration, with fake rankers")
print("=" * 70)

FAKE_CHUNKS = [
    {"chunk_id": "c1", "doc_id": "d1", "company": "toyota motor",
     "text": "Toyota chunk dated before the event.",
     "published_date": "2010-01-01", "published_int": 20100101,
     "form": "10-K", "url": "https://example.com/1"},
    {"chunk_id": "c2", "doc_id": "d1", "company": "toyota motor",
     "text": "Toyota chunk dated AFTER the event -- must be excluded by the leakage guard.",
     "published_date": "2012-01-01", "published_int": 20120101,
     "form": "10-K", "url": "https://example.com/1"},
    {"chunk_id": "c3", "doc_id": "d2", "company": "sony corp",
     "text": "Sony chunk dated before the event.",
     "published_date": "2010-06-01", "published_int": 20100601,
     "form": "20-F", "url": "https://example.com/2"},
    {"chunk_id": "c4", "doc_id": "d3", "company": "panasonic",
     "text": "Panasonic chunk dated before the event.",
     "published_date": "2010-03-01", "published_int": 20100301,
     "form": "20-F", "url": "https://example.com/3"},
]


class FakeBM25:
    """Real .get_scores() is never invoked. Everything downstream of the
    scores -- filtering via matches_filter, sorting -- is the REAL code
    from search_corpus.py, genuinely exercised."""
    def __init__(self, score_by_id):
        self.score_by_id = score_by_id

    def get_scores(self, tokens):
        return [self.score_by_id.get(c["chunk_id"], 0.0) for c in FAKE_CHUNKS]


class FakeCollection:
    """Records the `where` clause it received (so build_chroma_where's
    output is exercised end-to-end) and returns a fixed id list. Does NOT
    reimplement Chroma's own filtering/similarity logic -- that's Chroma's
    job and was already validated against the real thing separately."""
    def __init__(self, fixed_ids):
        self.fixed_ids = fixed_ids
        self.last_where = "NOT_CALLED"

    def query(self, query_embeddings, n_results, where):
        self.last_where = where
        return {"ids": [self.fixed_ids[:n_results]]}

    def count(self):
        return len(FAKE_CHUNKS)


class FakeModel:
    def encode(self, text):
        class _Vec:
            def tolist(self_inner):
                return [0.0]
        return _Vec()


def make_fake_store(bm25_scores, chroma_ids):
    store = CorpusSearchStore.__new__(CorpusSearchStore)  # skip the real __init__ entirely
    store.chunks = FAKE_CHUNKS
    store._chunk_lookup = {c["chunk_id"]: c for c in FAKE_CHUNKS}
    store._companies = sorted(set(c["company"] for c in FAKE_CHUNKS))
    store.bm25 = FakeBM25(bm25_scores)
    store.model = FakeModel()
    store.collection = FakeCollection(chroma_ids)
    return store


# --- B1: leakage guard fires even against the HIGHEST bm25 score ----------
# c2 (Toyota, post-event) is deliberately given the highest score of all,
# specifically to prove the FILTER is doing the excluding, not just the sort.
store = make_fake_store(
    bm25_scores={"c1": 5.0, "c2": 100.0, "c3": 3.0, "c4": 1.0},
    chroma_ids=["c3", "c1", "c4"],
)
r = store.search_corpus("earthquake supply impact", event_date="2011-03-11", top_k=5)
returned_ids = [h.chunk_id for h in r.hits]
check("B1 status == found", r.status == "found", r.status)
check("B1 c2 excluded despite the highest bm25 score (leakage guard fired correctly)",
      "c2" not in returned_ids, returned_ids)
check("B1 c1 and c3 both present in the fused result",
      "c1" in returned_ids and "c3" in returned_ids, returned_ids)
check("B1 distinct_companies correctly includes both toyota and sony",
      set(r.distinct_companies) >= {"toyota motor", "sony corp"}, r.distinct_companies)
check("B1 every hit is citable -- doc_id and url survived the chunk-lookup join",
      all(h.doc_id and h.url for h in r.hits))
check("B1 chroma path received the correctly-translated leakage filter",
      store.collection.last_where == {"published_int": {"$lte": 20110311}},
      store.collection.last_where)

# --- B2: genuinely no results -- achieved via the leakage guard itself -----
# NOTE ON WHY THIS TEST INITIALLY FAILED: an "empty" bm25_scores dict does
# NOT produce zero bm25 candidates. FakeBM25.get_scores() (matching real
# BM25Okapi behaviour) returns a score for EVERY chunk in the corpus,
# defaulting unseen ids to 0.0 -- it never returns an empty list. So handing
# it {} still yields all 4 chunks, tied at score 0.0, and the fused result
# is non-empty. A genuinely empty result only happens when the FILTER
# eliminates every candidate -- here, an event_date earlier than any fake
# chunk's published_date, so the leakage guard excludes all four before
# ranking even matters.
store2 = make_fake_store(bm25_scores={}, chroma_ids=[])
r2 = store2.search_corpus("nonsense query matching nothing", event_date="2005-01-01")
check("B2 status == no_results", r2.status == "no_results", r2.status)
check("B2 hits list is empty", r2.hits == [])

# --- B3: company filter resolves via fuzzy substring, with a warning -------
store3 = make_fake_store(bm25_scores={"c1": 1.0}, chroma_ids=["c1"])
r3 = store3.search_corpus("supply risk", event_date="2011-01-01", company="toyota")
check("B3 status == found", r3.status == "found", r3.status)
check("B3 company_filter_resolved == 'toyota motor'",
      r3.company_filter_resolved == "toyota motor", r3.company_filter_resolved)
check("B3 a warning is recorded because the input string != resolved string",
      len(r3.warnings) == 1, r3.warnings)

# --- B4: company filter matches nothing -> its own distinct status ---------
store4 = make_fake_store(bm25_scores={}, chroma_ids=[])
r4 = store4.search_corpus("supply risk", event_date="2011-01-01", company="totally_made_up_xyz")
check("B4 status == company_not_resolved (not the generic no_results)",
      r4.status == "company_not_resolved", r4.status)
check("B4 no ranking was even attempted once resolution failed",
      store4.collection.last_where == "NOT_CALLED", store4.collection.last_where)

# --- B5: as_observation() is safe for every status seen so far -------------
for result in (r, r2, r3, r4):
    obs = result.as_observation()
    check(f"B5 as_observation() non-empty for status={result.status}", len(obs) > 0)


print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
