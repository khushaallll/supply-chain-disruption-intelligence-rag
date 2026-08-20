"""
graph_reach_analysis.py -- Evaluation #1: the graph's recall ceiling.

"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from event_loader import load_events, Event
from agent.tools.graph_tool import GraphStore, GRAPH_PATH

RESULTS_DIR = Path("results")
MAX_TIER = 4


def analyze_event(store: GraphStore, event: Event) -> list[dict]:
    """One row per ground-truth-affected company for this event."""
    rows = []

    seed_result = store.traverse_supply_graph(
        event.seed_company, mode="downstream", max_tier=MAX_TIER
    )
    seed_resolved = seed_result.status != "not_resolved"

    tier_by_node: dict[str, int] = {}
    enriched_nodes: set[str] = set()
    for r in seed_result.results:
        tier_by_node[r["name"]] = r["tier"]
        enriched_nodes.add(r["name"])
    for d in seed_result.dropped_unenriched:
        # a node could in principle appear at more than one tier via
        # different paths -- keep the SMALLEST tier seen, since that's
        # the true shortest-path distance and what "how many hops away"
        # should mean
        tier_by_node[d["name"]] = min(tier_by_node.get(d["name"], d["tier"]), d["tier"])

    for c in event.ground_truth_affected:
        base = {
            "event_id": event.event_id,
            "category": event.category,
            "excluded": event.excluded,
            "company": c.company,
            "graph_node": c.graph_node,
            "gt_stated_hop": c.in_graph_context,  # "hop1" / "hop2" / False, as ground truth recorded it
        }

        if not seed_resolved:
            rows.append({**base, "live_status": "seed_unresolved", "live_tier": None})
            continue

        if not c.graph_node:
            rows.append({**base, "live_status": "no_graph_node", "live_tier": None})
            continue

        tier = tier_by_node.get(c.graph_node)
        if tier is None:
            rows.append({**base, "live_status": "not_found_within_tier4", "live_tier": None})
        elif c.graph_node in enriched_nodes:
            rows.append({**base, "live_status": "found_enriched", "live_tier": tier})
        else:
            rows.append({**base, "live_status": "found_unenriched", "live_tier": tier})

    return rows


def summarize(rows: list[dict]) -> dict:
    """Cumulative reach at each tier, as a % of the scored company pool.
    Only rows from events where score_for_recall is True are included --
    same pool the headline recall_all number is computed over, so this
    ceiling is directly comparable to that number, not a different
    population."""
    n = len(rows)
    if n == 0:
        return {}

    def pct(cond) -> float:
        return round(100 * sum(1 for r in rows if cond(r)) / n, 1)

    return {
        "n_companies": n,
        "found_by_tier_1_enriched": pct(lambda r: r["live_status"] == "found_enriched" and r["live_tier"] <= 1),
        "found_by_tier_2_enriched": pct(lambda r: r["live_status"] == "found_enriched" and r["live_tier"] <= 2),
        "found_by_tier_3_enriched": pct(lambda r: r["live_status"] == "found_enriched" and r["live_tier"] <= 3),
        "found_by_tier_4_enriched": pct(lambda r: r["live_status"] == "found_enriched" and r["live_tier"] <= 4),
        "found_but_unenriched_any_tier": pct(lambda r: r["live_status"] == "found_unenriched"),
        "not_found_within_tier4": pct(lambda r: r["live_status"] == "not_found_within_tier4"),
        "no_graph_node_at_all": pct(lambda r: r["live_status"] == "no_graph_node"),
        "seed_unresolved": pct(lambda r: r["live_status"] == "seed_unresolved"),
    }


def main():
    events = load_events()
    store = GraphStore(GRAPH_PATH)
    print(f"Loaded {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges\n")

    all_rows = []
    for ev in events:
        all_rows.extend(analyze_event(store, ev))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "category", "excluded", "company", "graph_node",
                  "gt_stated_hop", "live_status", "live_tier"]
    with open(RESULTS_DIR / "graph_reach_per_company.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    # --- the headline ceiling table: same pool as recall_all (score_for_recall events only) ---
    scored_rows = [r for r, ev in
                   ((row, next(e for e in events if e.event_id == row["event_id"])) for row in all_rows)
                   if ev.score_for_recall]
    ceiling = summarize(scored_rows)

    print("=== Recall ceiling (same company pool as the headline recall_all number) ===")
    print(f"  n = {ceiling.get('n_companies', 0)} real affected companies across "
          f"{sum(1 for e in events if e.score_for_recall)} scored events\n")
    for label, key in [
        ("Findable at tier <= 1 (enriched, usable by Baseline B today)", "found_by_tier_1_enriched"),
        ("Findable at tier <= 2 (enriched)", "found_by_tier_2_enriched"),
        ("Findable at tier <= 3 (enriched)", "found_by_tier_3_enriched"),
        ("Findable at tier <= 4 (enriched)", "found_by_tier_4_enriched"),
        ("Structurally reachable but NOT enriched (invisible to any system reading .results)",
         "found_but_unenriched_any_tier"),
        ("Not found within 4 hops downstream at all", "not_found_within_tier4"),
        ("Never had a graph node in the first place", "no_graph_node_at_all"),
        ("Event's seed company itself didn't resolve", "seed_unresolved"),
    ]:
        print(f"  {label:70s} {ceiling.get(key, 0):5.1f}%")

    ceiling_within_4 = (ceiling.get("found_by_tier_4_enriched", 0))
    print(f"\n  ABSOLUTE CEILING: no downstream-graph system, however smart its "
          f"reasoning, can score above ~{ceiling_within_4:.1f}% recall_all on "
          f"this dataset using ENRICHED companies only within 4 hops -- the "
          f"rest are either not in the graph, not enriched, or further than "
          f"4 hops away. Quote this number next to any recall_all figure.")

    # --- ground-truth-hop vs live-hop disagreement check ---
    print("\n=== Cross-check: live tier vs. ground truth's own stated hop ===")
    disagreements = []
    for r in all_rows:
        if r["live_status"] not in ("found_enriched", "found_unenriched"):
            continue
        stated = r["gt_stated_hop"]
        stated_tier = {"hop1": 1, "hop2": 2}.get(stated)
        if stated_tier is not None and stated_tier != r["live_tier"]:
            disagreements.append(r)
    print(f"  {len(disagreements)} companies where the live-recomputed tier "
          f"disagrees with the ground truth's stated hop1/hop2 tag.")
    for r in disagreements[:15]:
        print(f"    {r['event_id']:40s} {r['company']:35s} "
              f"stated={r['gt_stated_hop']}  live=tier{r['live_tier']}")
    if len(disagreements) > 15:
        print(f"    ... and {len(disagreements) - 15} more -- see the CSV.")
    if disagreements:
        print("  Worth a look: could mean the graph changed between when "
              "ground truth's hop was computed (graph_enriched.pkl) and now "
              "(graph_enriched_corrected.pkl), or a genuine seed-derivation "
              "difference. Doesn't invalidate either number on its own.")

    print(f"\nWritten: {RESULTS_DIR}/graph_reach_per_company.csv")


if __name__ == "__main__":
    main()
