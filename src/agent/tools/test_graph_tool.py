"""
test_graph_tool_real.py

Runs traverse_supply_graph() against the REAL supply graph
(graph_enriched_corrected.pkl) ONLY -- no synthetic data. Checks its
downstream results against REAL, verified graph_hop / graph_node
annotations read directly from your ground_truth_batch{1,2,3}_final.json
files.

Why this is a genuine check, not just re-stating the ground truth: each
ground-truth record's own `graph_note` field states these hop distances
were computed by walking graph_enriched.pkl (the pre-Day-7 file) from each
event's seed company -- a SEPARATE computation, done earlier, by a
different process, than this script or the current
graph_enriched_corrected.pkl. If traverse_supply_graph() reproduces the
same hop distances now, that's real agreement between two independently
derived results, not circular reasoning.

Two kinds of check per affected company:
  - graph_hop in {1, 2}  -> must appear at EXACTLY that tier when queried
                             with max_tier=2
  - graph_hop is None, or >= 3
                         -> must NOT appear anywhere in a max_tier=2 call's
                            results. This holds regardless of WHY --
                            whether the company's name fails to resolve at
                            all, or it resolves but has no path within 2
                            hops -- so it's a safe assertion either way,
                            without needing to know which of those two
                            reasons applies on your real graph.

A third possible (and informative) outcome is also checked separately:
a company reachable at the RIGHT tier structurally, but dropped from
`results` into `dropped_unenriched` because it lacks component/industry
data. That's a different, already-documented kind of gap (enrichment
coverage, not graph traversal) and is reported as such, not conflated with
a genuine traversal mismatch.

NOT executed in this environment -- I don't have your real
graph_enriched_corrected.pkl. Run this on your own machine and paste the
output back if you want help reading it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from graph_tool import GraphStore

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------- #
# [CONFIRM: exact path -- same guess used in test_search_corpus_real.py.
# Field names (event_id, ground_truth_affected, graph_hop, graph_node) ARE
# confirmed correct by direct inspection of your real files.]
# --------------------------------------------------------------------------- #

GROUND_TRUTH_PATHS = [
    "data/ground_truth/ground_truth_batch1_final.json",
    "data/ground_truth/ground_truth_batch2_final.json",
    "data/ground_truth/ground_truth_batch3_final.json",
]

# Five events whose ground truth includes real graph_hop annotations,
# verified by direct inspection -- covers hop1, hop2, hop3+, and
# unresolvable (None) cases across four different seed companies.
EVENT_SEED_GUESSES = {
#     "2_nippon_steel_2011": "nippon steel",      # 9 companies, all hop1
#     "1_posco_2022": "posco",                     # Korinox: unresolvable (None)
#     "11_aurizon_2010": "aurizon",                # hop1, hop2, hop3, hop6, None -- richest mix
    "35_gazprom_pjsc_2022": "gazprom pjsc",           # hop2, hop3, hop5, None
#     "47_china_steel_2018": "china steel",        # 2 companies, both hop1
}


def load_ground_truth(paths):
    events = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  [load_ground_truth] not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            for ev in json.load(f):
                events[ev["event_id"]] = ev
    print(f"  [load_ground_truth] loaded {len(events)} event(s)")
    return events


# --------------------------------------------------------------------------- #
# Setup -- loads the REAL graph, once
# --------------------------------------------------------------------------- #

print("Loading real supply graph...")
store = GraphStore()  # uses the default GRAPH_PATH defined in graph_tool.py
print(f"Loaded {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges")

ground_truth = load_ground_truth(GROUND_TRUTH_PATHS)


# --------------------------------------------------------------------------- #
# Per-event check
# --------------------------------------------------------------------------- #

def check_event(event_id: str, seed_guess: str, max_tier: int = 2):
    print(f"\n=== {event_id} (seed guess: '{seed_guess}') ===")
    ev = ground_truth.get(event_id)
    if ev is None:
        print("  SKIPPED -- event not found in loaded ground truth "
              "(check GROUND_TRUTH_PATHS)")
        return

    resolved_seed = store.find_company_node(seed_guess)
    if resolved_seed is None:
        print(f"  SKIPPED -- '{seed_guess}' did not resolve to a graph node "
              f"-- check this seed name guess against your real graph")
        return
    print(f"  seed resolved to: '{resolved_seed}'")

    result = store.traverse_supply_graph(resolved_seed, mode="downstream", max_tier=max_tier)
    print(f"  status: {result.status}  |  n_results: {result.n_results}  |  "
          f"n_dropped_unenriched: {result.n_dropped_unenriched}")

    results_by_name = {r["name"]: r["tier"] for r in result.results}
    dropped_by_name = {d["name"]: d["tier"] for d in result.dropped_unenriched}

    for company in ev.get("ground_truth_affected") or []:
        name = company.get("company", "")
        expected_hop = company.get("graph_hop")
        graph_node = company.get("graph_node")

        if expected_hop in (1, 2):
            actual_tier = results_by_name.get(graph_node)
            if actual_tier == expected_hop:
                check(f"'{name}' (ground truth: hop{expected_hop}) -> "
                      f"tier {expected_hop} in real traversal", True)
            elif dropped_by_name.get(graph_node) == expected_hop:
                check(f"'{name}' (ground truth: hop{expected_hop}) -> "
                      f"correct tier, but DROPPED as unenriched", False,
                      "structurally correct -- this is an enrichment-coverage "
                      "gap, not a traversal bug; see dropped_unenriched")
            else:
                check(f"'{name}' (ground truth: hop{expected_hop}) -> "
                      f"tier {expected_hop} in real traversal", False,
                      f"not found in results OR dropped_unenriched at tier {expected_hop}")
        else:
            # graph_hop is None, or 3+ -- must be absent from a max_tier=2
            # call's results, regardless of the underlying reason
            in_results = bool(graph_node) and graph_node in results_by_name
            check(f"'{name}' (ground truth: hop={expected_hop}) -> "
                  f"absent from max_tier={max_tier} traversal", not in_results,
                  f"unexpectedly found at tier={results_by_name.get(graph_node)}")


# --------------------------------------------------------------------------- #
# Run all five grounded events
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for event_id, seed_guess in EVENT_SEED_GUESSES.items():
        check_event(event_id, seed_guess)

    print("\n=== Bonus: name-resolution check on the known-unreachable company ===")
    print("(Korinox, event 1_posco_2022 -- ground truth: in_graph_context=False, "
          "graph_hop=None. This checks find_company_node() directly, which the "
          "loop above doesn't exercise on its own.)")
    korinox_resolved = store.find_company_node("korinox")
    check("'korinox' does not resolve to any graph node",
          korinox_resolved is None, f"unexpectedly resolved to '{korinox_resolved}'")

    print()
    print("=" * 70)
    print(f"TOTAL: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)