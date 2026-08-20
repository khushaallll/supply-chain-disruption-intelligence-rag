"""
select_final_enrichment_targets.py

Builds the final list of companies to enrich (city, lat/lon, component label)

Selection logic:
  1. MANDATORY  - the 29 real event epicenters. Non-negotiable.
  2. OEM        - the top-N automotive OEMs (industry == 'Automotive') reachable
                  within a real 2-hop path from ANY epicenter, ranked by how many
                  of the 30 events can reach them. These are the "headline result"
                  companies -- the ones that make an alert worth reading
                  ("a typhoon at POSCO threatens BMW's production").
  3. BRIDGE     - the specific 1-hop companies needed to connect an epicenter to
                  a selected OEM. Without these, the OEM is enriched but
                  structurally disconnected from any usable evidence path.
  4. FALLBACK   - one event (perusahaan...listrik negara / PLN Indonesia) has
                  ZERO automotive OEMs reachable at 2 hops -- it's a pure
                  electric-utility disruption with no car-industry story. For
                  this event only, we fall back to its own highest-degree real
                  1-hop and 2-hop customers (a proxy for "economically
                  significant", industry-agnostic), so the event isn't left
                  with nothing beyond its own epicenter.

Output: final_enrichment_targets.csv with columns:
  company, reason, notes
"""

import json
import pickle
import csv
from collections import Counter

GRAPH_PATH = "data/processed/graph_clean.pkl"
EVENTS_PATH = "data/events/events_selected_30.json"
OUTPUT_PATH = "data/enrichments/final_enrichment_targets.csv"
TOP_N_OEMS = 200


def load_data():
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    with open(EVENTS_PATH) as f:
        events = json.load(f)
    return graph, events


def get_mandatory(events):
    mandatory = set()
    for e in events:
        mandatory.update(e["companies_affected"])
    return mandatory


def compute_oem_paths(graph, events):
    """
    For every event, trace real 2-hop paths (epicenter -> bridge -> destination)
    and record which destinations are automotive OEMs, and which bridge(s)
    reach each one. Also records, per epicenter, whether it reached ANY OEM
    at all (to detect orphaned events).
    """
    oem_freq = Counter()
    oem_bridges = {}  # oem -> set of bridge companies that reach it
    epicenter_oem_count = {}  # epicenter -> how many distinct OEMs it reaches

    for e in events:
        epicenter = e["companies_affected"][0]
        if epicenter not in graph.nodes():
            continue
        bridges = set(graph.successors(epicenter))
        reached = set()
        for bridge in bridges:
            for dest in graph.successors(bridge):
                if dest == epicenter or dest in bridges:
                    continue
                if graph.nodes[dest].get("industry") == "Automotive":
                    reached.add(dest)
                    oem_bridges.setdefault(dest, set()).add(bridge)
        epicenter_oem_count[epicenter] = len(reached)
        for oem in reached:
            oem_freq[oem] += 1

    return oem_freq, oem_bridges, epicenter_oem_count


def get_fallback_targets(graph, epicenter, top_k=5):
    """
    For an event with no automotive OEMs reachable, fall back to its own
    highest-degree real 1-hop and 2-hop customers as a proxy for
    "economically significant company worth enriching", regardless of industry.
    """
    if epicenter not in graph.nodes():
        return set(), set()

    direct = list(graph.successors(epicenter))
    direct_ranked = sorted(direct, key=lambda c: -graph.out_degree(c))
    fallback_1hop = set(direct_ranked[:top_k])

    two_hop_candidates = set()
    for b in direct:
        two_hop_candidates.update(graph.successors(b))
    two_hop_candidates -= set(direct)
    two_hop_candidates.discard(epicenter)
    two_hop_ranked = sorted(two_hop_candidates, key=lambda c: (-graph.out_degree(c), c))
    fallback_2hop = set(two_hop_ranked[:top_k])

    return fallback_1hop, fallback_2hop


def build_final_list(graph, events, top_n_oems=TOP_N_OEMS):
    mandatory = get_mandatory(events)
    oem_freq, oem_bridges, epicenter_oem_count = compute_oem_paths(graph, events)

    selected = {}  # company -> (reason, notes)

    # 1. Mandatory epicenters
    for company in mandatory:
        selected[company] = ("mandatory", "event epicenter")

    # 2. Top-N OEMs by cross-event frequency (ties broken alphabetically for
    #    reproducibility -- without this, Python's hash randomization can
    #    silently change which company lands on the N-th spot between runs)
    ranked_oems = sorted(oem_freq, key=lambda c: (-oem_freq[c], c))[:top_n_oems]
    for oem in ranked_oems:
        selected[oem] = ("oem", f"reachable from {oem_freq[oem]} of 30 events")

    # 3. Bridges required to connect epicenters to the chosen OEMs
    required_bridges = set()
    for oem in ranked_oems:
        required_bridges.update(oem_bridges.get(oem, set()))
    for bridge in required_bridges:
        if bridge not in selected:
            selected[bridge] = ("bridge", "connects an epicenter to a selected OEM")

    # 4. Fallback for orphaned events (zero automotive OEMs reachable)
    orphans = [ep for ep, count in epicenter_oem_count.items() if count == 0]
    for epicenter in orphans:
        fb_1hop, fb_2hop = get_fallback_targets(graph, epicenter)
        for c in fb_1hop:
            if c not in selected:
                selected[c] = ("fallback_1hop", f"no automotive OEM path from {epicenter}; high-degree direct customer")
        for c in fb_2hop:
            if c not in selected:
                selected[c] = ("fallback_2hop", f"no automotive OEM path from {epicenter}; high-degree 2-hop customer")

    return selected, orphans


def write_csv(selected, graph, output_path=OUTPUT_PATH):
    priority = {"mandatory": 0, "oem": 1, "bridge": 2, "fallback_1hop": 3, "fallback_2hop": 4}
    rows = [
        {"company": c, "reason": reason, "notes": notes, "in_graph": c in graph.nodes()}
        for c, (reason, notes) in selected.items()
    ]
    rows.sort(key=lambda r: priority.get(r["reason"], 9))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "reason", "notes", "in_graph"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    graph, events = load_data()
    selected, orphans = build_final_list(graph, events)
    rows = write_csv(selected, graph)

    counts = Counter(r["reason"] for r in rows)
    print(f"Total companies selected: {len(rows)}")
    for reason in ["mandatory", "oem", "bridge", "fallback_1hop", "fallback_2hop"]:
        if counts.get(reason):
            print(f"  - {reason}: {counts[reason]}")
    print(f"\nOrphaned events (no automotive OEM path, used fallback): {orphans}")

    not_in_graph = [r["company"] for r in rows if not r["in_graph"]]
    if not_in_graph:
        print(f"WARNING: {len(not_in_graph)} companies not found in graph: {not_in_graph}")

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()