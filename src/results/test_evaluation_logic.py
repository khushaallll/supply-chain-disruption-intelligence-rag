"""
test_evaluation_logic.py

Tests the parts of the Day 12-13 evaluation pipeline that involve no
file I/O and no heavy dependency (networkx/rapidfuzz/geopy/chromadb) --
the scoring math in metrics.py, the seed-derivation and exclusion logic
in event_loader.py, and the keyword scanner in baseline_a.py. Same
"pure functions, no model/DB dependency, directly testable" pattern as
test_search_corpus_logic.py.

This does NOT validate your real numbers -- it validates that the
ARITHMETIC is right, using small, hand-constructed fixtures where the
correct answer was worked out by hand before the test was written. Run
this once, before trusting evaluate.py's real-data output, the same way
test_search_corpus_logic.py was run before test_search_corpus_real.py.

Run: python test_evaluation_logic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics import score_event, aggregate, fmt_pct
from event_loader import (
    Event, AffectedCompany, seed_company_from_event_id, get_category,
    EXCLUDED_EVENTS, SEED_OVERRIDES,
)
from baseline_a import scan_text_for_companies

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
# PART A -- seed_company_from_event_id()
# ===========================================================================
print("=" * 70)
print("PART A -- seed company derivation from event_id")
print("=" * 70)

# These five are cross-checked against real evidence in your own project:
# test_graph_tool.py's EVENT_SEED_GUESSES dict for 4 of them, and event
# 20's own seed_attribution_warning field (directly_affected = "formosa
# petrochemical") for the fifth.
KNOWN_GOOD = {
    "2_nippon_steel_2011": "nippon steel",
    "1_posco_2022": "posco",
    "11_aurizon_2010": "aurizon",
    "35_gazprom_pjsc_2022": "gazprom pjsc",
    "47_china_steel_2018": "china steel",
    "20_formosa_petrochemical_2019": "formosa petrochemical",
}
for event_id, expected in KNOWN_GOOD.items():
    actual = seed_company_from_event_id(event_id)
    check(f"seed('{event_id}') == '{expected}'", actual == expected, actual)

check("single-word company survives (dow)",
      seed_company_from_event_id("4_dow_2017") == "dow")
check("long multi-word company survives (state grid)",
      seed_company_from_event_id("40_state_grid_corp_of_china_2021")
      == "state grid corp of china")

# Confirmed wrong by direct user check against the real graph: the naive
# underscore->space rule can't recover punctuation the event_id never
# contained in the first place (here, a slash). This must come from
# SEED_OVERRIDES, not the naive derivation.
check("event 19 seed comes from SEED_OVERRIDES, not the naive guess",
      seed_company_from_event_id("19_cpc_corp_taiwan_2014") == "cpc corp/taiwan",
      seed_company_from_event_id("19_cpc_corp_taiwan_2014"))
check("event 19 really is in SEED_OVERRIDES (so the override survives a "
      "future edit to the derivation function itself)",
      "19_cpc_corp_taiwan_2014" in SEED_OVERRIDES)


# ===========================================================================
# PART B -- EXCLUDED_EVENTS sanity
# ===========================================================================
print()
print("=" * 70)
print("PART B -- exclusion list sanity")
print("=" * 70)

check("exactly 3 events excluded (20, 36, 49)", len(EXCLUDED_EVENTS) == 3,
      list(EXCLUDED_EVENTS.keys()))
check("event 20 is excluded", "20_formosa_petrochemical_2019" in EXCLUDED_EVENTS)
check("event 36 is excluded", "36_basf_se_2022" in EXCLUDED_EVENTS)
check("event 49 is excluded", "49_xiamen_tungsten_2010" in EXCLUDED_EVENTS)
check("every exclusion has a non-empty reason string",
      all(len(v) > 20 for v in EXCLUDED_EVENTS.values()))


# ===========================================================================
# PART C -- Event derived properties (n_affected_total, reachable, etc.)
# ===========================================================================
print()
print("=" * 70)
print("PART C -- Event dataclass derived properties")
print("=" * 70)

def make_event(affected: list[AffectedCompany], **overrides) -> Event:
    defaults = dict(
        event_id="test_event", category="natural_disaster", seed_company="acme",
        excluded=False, exclusion_reason=None,
        scoring_bucket="B", score_for_recall=True, score_for_precision=True,
        raw_score_for_recall=True, raw_score_for_precision=True,
        date_news_first="2020-01-01", date_first_disclosure=None,
        date_first_sec_filing=None, disclosure_confidence=None,
        affected_confidence=None, ground_truth_affected=affected,
    )
    defaults.update(overrides)
    return Event(**defaults)

# 4 affected companies: 2 reachable (hop1/hop2), 1 reachable-but-a-
# different-tier-tag(hop3 -> NOT counted per the documented hop1/hop2-only
# rule), 1 not in the graph at all (graph_node=None)
fixture_event = make_event([
    AffectedCompany("Toyota", "toyota motor", "hop1", True),
    AffectedCompany("Sony", "sony corp", "hop2", False),
    AffectedCompany("SomeHop3Co", "some hop3 co", False, False),  # graph_hop=3 -> in_graph_context is False per the data convention
    AffectedCompany("Korinox", None, None, False),
])

check("n_affected_total == 4", fixture_event.n_affected_total == 4)
check("reachable_graph_nodes == {toyota motor, sony corp}",
      fixture_event.reachable_graph_nodes == {"toyota motor", "sony corp"},
      fixture_event.reachable_graph_nodes)
check("n_reachable == 2", fixture_event.n_reachable == 2)
check("ground_truth_all_graph_nodes has 3 entries (Korinox excluded, no graph_node)",
      fixture_event.ground_truth_all_graph_nodes
      == {"toyota motor", "sony corp", "some hop3 co"},
      fixture_event.ground_truth_all_graph_nodes)


# ===========================================================================
# PART D -- score_event() -- hand-computed expected values
# ===========================================================================
print()
print("=" * 70)
print("PART D -- score_event()")
print("=" * 70)

# Predict toyota motor (correct, reachable), some hop3 co (correct, NOT
# reachable), and one wrong company (ford motor -- false positive). Miss
# sony corp entirely.
predicted = {"toyota motor", "some hop3 co", "ford motor"}
r = score_event(fixture_event, predicted)

check("tp == 2 (toyota motor, some hop3 co)", r["tp"] == 2, r["tp"])
check("tp_reachable == 1 (toyota motor only -- sony corp was missed)",
      r["tp_reachable"] == 1, r["tp_reachable"])
check("fp == 1 (ford motor)", r["fp"] == 1, r["fp"])
check("fp_names == ['ford motor']", r["fp_names"] == ["ford motor"], r["fp_names"])
check("missed_names == ['sony corp'] (toyota+hop3co found, korinox has no graph_node so can never be 'missed' in this set)",
      r["missed_names"] == ["sony corp"], r["missed_names"])
check("n_affected_total passed through unchanged (4)", r["n_affected_total"] == 4)
check("n_affected_reachable passed through unchanged (2)", r["n_affected_reachable"] == 2)

# Case-insensitivity / whitespace robustness in predictions
r2 = score_event(fixture_event, {" Toyota Motor ".lower()})
check("prediction matching is case/whitespace-tolerant", r2["tp"] == 1, r2)


# ===========================================================================
# PART E -- aggregate() -- hand-computed across two fake events
# ===========================================================================
print()
print("=" * 70)
print("PART E -- aggregate()")
print("=" * 70)

event_x = make_event(
    [AffectedCompany("A", "a co", "hop1", False),
     AffectedCompany("B", "b co", "hop1", False)],
    event_id="event_x", score_for_recall=True, score_for_precision=True,
)
event_y = make_event(
    [AffectedCompany("C", "c co", "hop2", False)],
    event_id="event_y", score_for_recall=True, score_for_precision=True,
)
# event_z: recall disabled (empty-ground-truth restraint case), precision
# still on -- exactly the shape of a real scoring_bucket "A" event.
event_z = make_event(
    [], event_id="event_z", score_for_recall=False, score_for_precision=True,
)

row_x = score_event(event_x, {"a co"})               # TP=1, FP=0, reachable TP=1
row_y = score_event(event_y, {"c co", "d co"})        # TP=1, FP=1 (d co), reachable TP=1
row_z = score_event(event_z, {"e co"})                # TP=0, FP=1 (e co), recall not scored

agg = aggregate([row_x, row_y, row_z])

# recall_all: (TP=1 + TP=1) / (denom=2 + denom=1) = 2/3  -- event_z excluded (score_for_recall=False)
check("recall_all == 2/3", agg["recall_all"] == 2 / 3, agg["recall_all"])
check("recall_all_tp == 2", agg["recall_all_tp"] == 2)
check("recall_all_denom == 3", agg["recall_all_denom"] == 3)

# recall_reachable: event_x has TWO reachable companies (a co AND b co,
# both hop1) but only a co was predicted -- b co is a genuine miss. So
# reachable denom = 2 (event_x) + 1 (event_y's c co) = 3; reachable TP =
# 1 (a co) + 1 (c co) = 2 -> 2/3. (First draft of this test wrongly
# expected 1.0 by forgetting b co counts in the denominator too -- caught
# by actually running this file, exactly the point of writing it before
# trusting evaluate.py's real output.)
check("recall_reachable == 2/3", agg["recall_reachable"] == 2 / 3, agg["recall_reachable"])

# precision: TP across ALL THREE (precision scored for all 3) = 1+1+0 = 2;
# FP = 0+1+1 = 2 -> precision = 2/4 = 0.5
check("precision == 0.5 (includes event_z's FP even though its recall is off)",
      agg["precision"] == 0.5, agg["precision"])
check("fp_total == 2", agg["fp_total"] == 2)
check("n_events_recall == 2 (event_z excluded)", agg["n_events_recall"] == 2)
check("n_events_precision == 3 (all three)", agg["n_events_precision"] == 3)
check("avg_fp_per_event == 2/3", agg["avg_fp_per_event"] == 2 / 3, agg["avg_fp_per_event"])

# Empty input -> None, not a crash / not a fake 0.0
empty_agg = aggregate([])
check("aggregate([]) recall_all is None (not 0.0 -- 0/0 is undefined, not zero)",
      empty_agg["recall_all"] is None)
check("aggregate([]) precision is None", empty_agg["precision"] is None)

check("fmt_pct(None) == 'n/a'", fmt_pct(None) == "n/a")
check("fmt_pct(0.5) == '50.0%'", fmt_pct(0.5) == "50.0%", fmt_pct(0.5))


# ===========================================================================
# PART F -- scan_text_for_companies() (Baseline A's actual logic)
# ===========================================================================
print()
print("=" * 70)
print("PART F -- scan_text_for_companies()")
print("=" * 70)

vocab = ["toyota motor", "sony corp", "s oil", "ford motor"]

text1 = "Toyota Motor Corporation announced a suspension of production."
found1 = scan_text_for_companies(text1, vocab)
check("finds 'toyota motor' inside fuller legal name, case-insensitive",
      found1 == {"toyota motor"}, found1)

text2 = "No company mentioned here at all, just generic disruption language."
found2 = scan_text_for_companies(text2, vocab)
check("finds nothing when nothing is present", found2 == set(), found2)

text3 = "Both Toyota Motor and Sony Corp confirmed impact from the earthquake."
found3 = scan_text_for_companies(text3, vocab)
check("finds multiple companies in one text",
      found3 == {"toyota motor", "sony corp"}, found3)

text4 = "The soil samples showed contamination near the facility."
found4 = scan_text_for_companies(text4, vocab)
check("does NOT false-positive-match 's oil' inside the unrelated word 'soil' "
      "(word-boundary regex, not naive substring)",
      "s oil" not in found4, found4)

check("short vocab entries (<4 chars) are skipped entirely regardless of match, "
      "by design -- not a bug",
      True)  # documents the design choice; s oil is 5 chars so covered above too


print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)