"""
name_resolution_audit.py -- Evaluation #3: is missed recall a graph
problem or a name-matching problem?

"""

from __future__ import annotations

import csv
from pathlib import Path

from event_loader import load_events
from graph_tool import GraphStore, GRAPH_PATH
from phonebook_lite import SimplePhonebook

RESULTS_DIR = Path("results")


def collect_names(events) -> dict[str, dict]:
    """Returns {name: {"type": "seed"|"affected_company", "example_event_id":
    ..., "gt_graph_node": ... or None}}. Deduplicated by exact name string --
    if the same company name is used as a seed in one event and an
    affected-company entry in another, it's tracked once as "seed" (seeds
    are the more load-bearing case: get baseline_b.py's whole event wrong
    if unresolved, vs. one missed company in a list)."""
    names: dict[str, dict] = {}

    for ev in events:
        if ev.seed_company not in names:
            names[ev.seed_company] = {
                "type": "seed", "example_event_id": ev.event_id, "gt_graph_node": None,
            }

    for ev in events:
        for c in ev.ground_truth_affected:
            if not c.company or c.company in names:
                continue
            names[c.company] = {
                "type": "affected_company", "example_event_id": ev.event_id,
                "gt_graph_node": c.graph_node,
            }

    return names


def main():
    events = load_events()
    names = collect_names(events)
    print(f"Checking {len(names)} distinct names "
          f"({sum(1 for v in names.values() if v['type'] == 'seed')} seeds, "
          f"{sum(1 for v in names.values() if v['type'] == 'affected_company')} "
          f"affected-company names).\n")

    store = GraphStore(GRAPH_PATH)  # loaded once, phonebook toggled below
    pb = SimplePhonebook()
    print(f"Phonebook loaded: {pb.n_rows} rows, {pb.n_graph_entries} with a graph_name.\n")

    rows = []
    for name, meta in names.items():
        store.phonebook = None
        without_pb = store.find_company_node(name)

        store.phonebook = pb
        with_pb = store.find_company_node(name)

        # graph_tool.py's find_company_node() returns the phonebook's answer
        # directly (Step 0) with NO check that it's actually a node in the
        # currently-loaded graph -- confirmed by reading the real source,
        # not assumed. Harmless as long as the phonebook and the graph file
        # stay in sync, but worth knowing before wiring the phonebook in:
        # if they ever drift (graph regenerated, a node renamed), this path
        # would hand back a name that isn't in self.G at all, and the NEXT
        # call (e.g. self.G.successors(node) inside downstream_from) would
        # raise a NetworkX error rather than a clean "not_resolved" status.
        with_pb_is_real_node = (with_pb is not None and store.G.has_node(with_pb))

        gt_node = meta["gt_graph_node"]
        rows.append({
            "name": name,
            "type": meta["type"],
            "example_event_id": meta["example_event_id"],
            "resolved_without_phonebook": without_pb,
            "resolved_with_phonebook": with_pb,
            "with_pb_result_is_real_graph_node": with_pb_is_real_node,
            "gt_stated_graph_node": gt_node,
            "phonebook_recovered_a_miss": (without_pb is None and with_pb is not None),
            "phonebook_changed_the_answer": (
                without_pb is not None and with_pb is not None and without_pb != with_pb
            ),
            "without_pb_disagrees_with_gt": (
                gt_node is not None and without_pb is not None and without_pb != gt_node
            ),
        })

    store.phonebook = None  # leave the store the way baseline_b.py would find it

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(RESULTS_DIR / "name_resolution_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    seeds = [r for r in rows if r["type"] == "seed"]
    affected = [r for r in rows if r["type"] == "affected_company"]

    def resolved_count(subset, key):
        return sum(1 for r in subset if r[key] is not None)

    print("=== Resolution rate, without vs. with phonebook ===\n")
    for label, subset in [("Seeds", seeds), ("Affected-company names", affected)]:
        n = len(subset)
        wo = resolved_count(subset, "resolved_without_phonebook")
        w_ = resolved_count(subset, "resolved_with_phonebook")
        print(f"  {label:26s} without: {wo:3d}/{n:<3d}   with: {w_:3d}/{n:<3d}")

    recovered = [r for r in rows if r["phonebook_recovered_a_miss"]]
    print(f"\n=== Names the phonebook recovers that plain fuzzy matching misses "
          f"({len(recovered)}) ===")
    print("These would resolve correctly TODAY if GraphStore were constructed "
          "with the phonebook attached -- currently baseline_b.py and "
          "evaluate.py both call GraphStore(GRAPH_PATH) with no phonebook, "
          "so none of these recoveries are happening in your actual reported "
          "numbers yet.")
    for r in recovered[:20]:
        flag = "" if r["with_pb_result_is_real_graph_node"] else "  *** NOT A REAL NODE IN THIS GRAPH FILE ***"
        print(f"  [{r['type']:16s}] '{r['name']}' -> '{r['resolved_with_phonebook']}' "
              f"(event: {r['example_event_id']}){flag}")
    if len(recovered) > 20:
        print(f"  ... and {len(recovered) - 20} more -- see the CSV.")

    stale = [r for r in rows if r["resolved_with_phonebook"] is not None
             and not r["with_pb_result_is_real_graph_node"]]
    if stale:
        print(f"\n  *** {len(stale)} phonebook answer(s) point to a node that "
              f"does NOT exist in the currently-loaded graph file. This is a "
              f"real property of find_company_node()'s phonebook path (Step "
              f"0 returns the phonebook's answer without checking "
              f"self.G.has_node() on it) -- confirmed by reading graph_tool.py "
              f"directly, not a guess. If you wire the phonebook into "
              f"baseline_b.py, these specific names would likely crash "
              f"downstream_from() with a NetworkX error rather than fail "
              f"cleanly. Check whether your phonebook.csv was built against "
              f"the SAME graph_enriched_corrected.pkl you're loading now.")
        for r in stale[:10]:
            print(f"    '{r['name']}' -> phonebook says '{r['resolved_with_phonebook']}' "
                  f"(not in graph)")

    changed = [r for r in rows if r["phonebook_changed_the_answer"]]
    print(f"\n=== Names where the phonebook gives a DIFFERENT answer than fuzzy "
          f"matching, not just a recovered miss ({len(changed)}) ===")
    print("Worth a manual look at each -- this means the un-assisted fuzzy "
          "match (what your scripts currently use) landed on some node, but "
          "the phonebook's pre-reviewed answer disagrees with it.")
    for r in changed[:20]:
        print(f"  [{r['type']:16s}] '{r['name']}': without={r['resolved_without_phonebook']!r}  "
              f"with={r['resolved_with_phonebook']!r}  (event: {r['example_event_id']})")

    disagree = [r for r in rows if r["without_pb_disagrees_with_gt"]]
    print(f"\n=== Current (no-phonebook) resolution disagrees with what ground "
          f"truth already recorded ({len(disagree)}) ===")
    print("These are cases where baseline_b.py's CURRENT behaviour resolves a "
          "name to a DIFFERENT graph node than the one ground truth's own "
          "graph_node field already states -- meaning a scoring comparison "
          "for this company could silently be checking the wrong node right "
          "now, in either direction.")
    for r in disagree[:20]:
        print(f"  [{r['type']:16s}] '{r['name']}': resolved={r['resolved_without_phonebook']!r}  "
              f"ground_truth_says={r['gt_stated_graph_node']!r}  (event: {r['example_event_id']})")

    unresolved_seeds = [r for r in seeds if r["resolved_without_phonebook"] is None
                        and r["resolved_with_phonebook"] is None]
    print(f"\n=== Seeds that STILL don't resolve even with the phonebook "
          f"({len(unresolved_seeds)}) ===")
    print("These events genuinely have no matching node in the graph at all "
          "(or the seed name itself is wrong, the way event 19 was before "
          "the SEED_OVERRIDES fix) -- worth checking each one by hand, the "
          "same way event 19 got fixed.")
    for r in unresolved_seeds:
        print(f"  '{r['name']}'  (event: {r['example_event_id']})")

    print(f"\nWritten: {RESULTS_DIR}/name_resolution_audit.csv")


if __name__ == "__main__":
    main()
