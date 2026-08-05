"""
test_search_corpus_real.py

Runs the SAME 20 hand-written queries from retrieval_test.py, but through
the real search_corpus() TOOL (CorpusSearchStore.search_corpus()) instead
of the bare hybrid_search() function. This does two jobs in one pass:

  1. Re-confirms retrieval quality on real data, against the same PASS
     conditions you already wrote in retrieval_test.py.
  2. Provides the real-data parity check that test_search_corpus_logic.py's
     fakes could never give -- does the wrapped tool surface the same real
     chunks the validated hybrid_search() found?

One deliberate difference from retrieval_test.py, worth reading closely:
search_corpus() requires event_date on every call (the leakage guard is
mandatory, not optional). Your original queries 1-9, 13-17, and 20 never
passed a date filter at all -- only 18-19 tested leakage on purpose. This
script closes that gap: every single query below now runs with its real
event's date applied, not just the two that were originally checking for
it.

Event dates for every query below were read directly out of your real
ground_truth_batch{1,2,3}_final.json files (schema confirmed by direct
inspection: `event_id`, `date_news_first`) -- see VERIFIED_EVENT_DATES.
The runtime loader is kept as a secondary mechanism so any event you add
later gets picked up automatically, but nothing in this script depends on
it succeeding -- every query used here already has a verified date.

NOT executed in this environment -- no chromadb/torch/rank_bm25 installed
here, and not your real corpus/Chroma collection. Run this on your own
machine and paste the output back if you want help reading it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_corpus_tool import CorpusSearchStore


# --------------------------------------------------------------------------- #
# Same real paths as retrieval_test.py -- adjust if yours differ
# --------------------------------------------------------------------------- #

CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "supply_chain_docs"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# --------------------------------------------------------------------------- #
# Event dates -- verified directly against your real
# ground_truth_batch{1,2,3}_final.json files (schema confirmed by direct
# inspection: top-level list, fields `event_id` and `date_news_first`).
# Every date below was read straight out of those files, not guessed.
# --------------------------------------------------------------------------- #

VERIFIED_EVENT_DATES = {
    1: "2022-09-06",    # 1_posco_2022
    2: "2011-03-11",    # 2_nippon_steel_2011
    4: "2017-08-25",    # 4_dow_2017
    11: "2010-12-25",   # 11_aurizon_2010
    18: "2012-08-06",   # 18_chevron_2012
    21: "2014-01-09",   # 21_mitsubishi_materials_2014
    24: "2010-04-20",   # 24_bp_2010 (Deepwater Horizon)
    33: "2018-10-13",   # 33_kyushu_electric_power_2018
    34: "2011-09-15",   # 34_korea_electric_power_2011
    35: "2022-09-02",   # 35_gazprom_pjsc_2022
    40: "2021-09-23",   # 40_state_grid_corp_of_china_2021
    47: "2018-03-01",   # 47_china_steel_2018
    57: "2018-04-06",   # 57_united_co_rusal_international_pjsc_2018
}

# Cross-checked directly against the real files: all four "confirmed absent"
# company/event pairings used below actually match --
#   SUMCO           -> event 21, has_corpus_doc=False, in_graph_context=hop1
#   Korinox         -> event 1,  has_corpus_doc=False, in_graph_context=False
#   Unimicron       -> event 40, has_corpus_doc=False, in_graph_context=False
#   Transocean      -> event 24, has_corpus_doc=False, in_graph_context=False
# So categories 13-16 below are testing exactly what their comments claim.

# The path below is a guess at your real repo layout -- the FIELD NAMES are
# confirmed correct, the PATH is not. If this loader finds 0 events, the
# VERIFIED_EVENT_DATES table above still covers every query in this script,
# so nothing breaks -- but worth fixing the path anyway for future events.
GROUND_TRUTH_PATHS = [
    "data/ground_truth/ground_truth_batch1_final.json",
    "data/ground_truth/ground_truth_batch2_final.json",
    "data/ground_truth/ground_truth_batch3_final.json",
]


def load_event_dates(paths):
    """Returns {event_id: date_string} freshly read from the real files, for
    any FUTURE event this script doesn't already have a verified date for.
    Confirmed field names: `event_id` (e.g. '2_nippon_steel_2011') and
    `date_news_first` (e.g. '2011-03-11')."""
    dates = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  [load_event_dates] not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            events = json.load(f)
        for ev in events:
            eid, date = ev.get("event_id"), ev.get("date_news_first")
            if eid is None or date is None:
                continue
            head = eid.split("_")[0]
            numeric_id = int(head) if head.isdigit() else eid
            dates[numeric_id] = date
    print(f"  [load_event_dates] loaded {len(dates)} event date(s) from real ground truth files")
    return dates


EVENT_DATES = load_event_dates(GROUND_TRUTH_PATHS)
for eid, date in VERIFIED_EVENT_DATES.items():
    EVENT_DATES.setdefault(eid, date)  # verified table wins only where the loader found nothing


def event_date_for(event_id):
    date = EVENT_DATES.get(event_id)
    if date is None:
        print(f"  [WARN] no known date for event {event_id} -- add it to "
              f"GROUND_TRUTH_PATHS (preferred) or FALLBACK_EVENT_DATES")
    return date


# --------------------------------------------------------------------------- #
# Setup -- loads the REAL chunks, REAL BM25, REAL embedding model, REAL
# Chroma collection. This is the one-time expensive step.
# --------------------------------------------------------------------------- #

print("Loading real corpus, BM25 index, embedding model, and Chroma collection...")
store = CorpusSearchStore(
    chunks_path=CHUNKS_PATH,
    chroma_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    model_name=MODEL_NAME,
)
print(f"Loaded {len(store.chunks)} chunks, {len(store._companies)} distinct companies")


# --------------------------------------------------------------------------- #
# Same shape as retrieval_test.py's print_results(), reading off the
# structured CorpusSearchResult instead of raw (chunk_id, score) tuples
# --------------------------------------------------------------------------- #

def print_results(query, event_id, company=None, top_k=5):
    event_date = event_date_for(event_id)
    if event_date is None:
        print(f"\nSKIPPED (no date for event {event_id}): '{query}'")
        return None

    result = store.search_corpus(query, event_date=event_date, company=company, top_k=top_k)
    print(f"\nQuery: '{query}'  |  event {event_id} ({event_date})  |  company filter: {company}")
    print(f"  status: {result.status}")
    if result.warnings:
        print(f"  warnings: {result.warnings}")
    if result.status != "found":
        return result

    for h in result.hits:
        preview = h.text[:150].replace("\n", " ")
        print(f"  {h.score:.4f} | {h.company:20s} | {h.published_int} | {h.chunk_id}")
        print(f"           {preview}...")
    return result


# --------------------------------------------------------------------------- #
# The same 20 scenarios, same grouping, run through search_corpus()
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print(f"\nTotal chunks: {len(store.chunks)}")
    print(f"Distinct companies: {len(store._companies)}")

    print("\n=== Category 1 -- positive matches (has_corpus_doc == true) ===")

    print("\n--- 1. Toyota -- event 2 ---")
    print_results("Toyota Motor Corporation supply chain earthquake risk", event_id=2)

    print("\n--- 2. Sony -- event 2 ---")
    print_results("Sony Corporation Japan supply disruption", event_id=2)

    print("\n--- 3. Panasonic -- event 2 ---")
    print_results("Panasonic Corporation production risk Japan", event_id=2)

    print("\n--- 4. Hitachi -- event 2 ---")
    print_results("Hitachi Ltd earthquake production halt", event_id=2)

    print("\n--- 5. General Motors -- events 2 and 47 ---")
    print_results("General Motors steel tariff commodity cost", event_id=2)
    print_results("General Motors steel tariff commodity cost", event_id=47)

    print("\n--- 6. Ford -- event 47 ---")
    print_results("Ford Motor Company commodity cost tariff", event_id=47)

    print("\n--- 7. ExxonMobil -- event 4 (dow_2017) ---")
    print_results("Exxon Mobil hurricane refinery disruption", event_id=4)

    print("\n--- 8. Alcoa -- event 35 (gazprom_pjsc_2022) ---")
    print_results("Alcoa aluminum smelter energy supply", event_id=35)

    print("\n--- 9. Rio Tinto -- events 11 and 57 ---")
    print_results("Rio Tinto coal export rail disruption", event_id=11)
    print_results("Rio Tinto coal export rail disruption", event_id=57)

    print("\n=== Category 2 -- restraint (ground_truth_affected is empty) ===")

    print("\n--- 10. RESTRAINT -- event 18 (chevron_2012) ---")
    print_results("Chevron Richmond refinery fire 2012", event_id=18)

    print("\n--- 11. RESTRAINT -- event 33 (kyushu_electric_power_2018) ---")
    print_results("Kyushu Electric solar curtailment 2018", event_id=33)

    print("\n--- 12. RESTRAINT -- event 34 (korea_electric_power_2011) ---")
    print_results("Korea Electric Power blackout 2011", event_id=34)

    print("\n=== Category 3 -- companies confirmed absent from the corpus "
          "(PASS = company_not_resolved, or found=empty) ===")

    print("\n--- 13. SUMCO -- event 21 ---")
    print_results("audit committee risk factors", event_id=21, company="sumco corporation")

    print("\n--- 14. Korinox -- event 1 ---")
    print_results("supply chain disruption impact", event_id=1, company="korinox")

    print("\n--- 15. Unimicron -- event 40 ---")
    print_results("supply chain disruption impact", event_id=40, company="unimicron technology corp")

    print("\n--- 16. Transocean -- event 24 (Deepwater Horizon) ---")
    print_results("drilling rig explosion supply impact", event_id=24, company="transocean")

    print("\n=== Category 4 -- filter plumbing ===")

    print("\n--- 17a. No company filter -- event 2 ---")
    print_results("supply chain risk factors", event_id=2)

    print("\n--- 17b. Filtered to Toyota -- event 2 ---")
    print_results("supply chain risk factors", event_id=2, company="toyota")

    print("\n=== Category 5 -- leakage checks (now automated, not manual) ===")

    print("\n--- 18. Leakage -- event 2, cutoff 2011-03-11 ---")
    r18 = print_results("Tohoku earthquake production impact Japan", event_id=2)
    if r18 and r18.status == "found":
        bad = [h for h in r18.hits if h.published_int and h.published_int > 20110311]
        print(f"  leakage check: {'PASS -- none after cutoff' if not bad else f'FAIL -- {len(bad)} chunk(s) after cutoff'}")

    print("\n--- 19. Leakage -- event 47, cutoff 2018-03-01 ---")
    r19 = print_results("steel tariff Section 232 cost increase", event_id=47)
    if r19 and r19.status == "found":
        bad = [h for h in r19.hits if h.published_int and h.published_int > 20180301]
        print(f"  leakage check: {'PASS -- none after cutoff' if not bad else f'FAIL -- {len(bad)} chunk(s) after cutoff'}")

    print("\n=== Category 6 -- exact-phrase / boilerplate stress test ===")

    print("\n--- 20a. Unfiltered -- event 2 ---")
    print_results("audit committee financial expert", event_id=2)

    print("\n--- 20b. Filtered to Panasonic -- event 2 ---")
    print_results("audit committee financial expert", event_id=2, company="panasonic")

    print("\nDone -- 20 scenarios run through the real search_corpus() tool.")
    print("Read each block against the same PASS conditions you used in")
    print("retrieval_test.py -- and additionally: does search_corpus() surface")
    print("the SAME top chunks hybrid_search() found, now with citations and")
    print("an explicit status attached, instead of a bare (chunk_id, score) tuple?")
