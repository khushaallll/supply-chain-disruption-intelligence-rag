"""
baseline_b.py -- System B: single-hop supplier graph lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.tools.graph_tool import GraphStore, GRAPH_PATH


@dataclass
class BaselineBResult:
    event_id: str
    seed_input: str
    status: str                 # found | no_results | not_resolved
    seed_resolved: Optional[str]
    predicted_companies: set    # graph node names, tier 1 only, enriched only
    n_dropped_unenriched: int = 0
    dropped_unenriched_names: list = field(default_factory=list)


def run_baseline_b(
    store: GraphStore,
    event_id: str,
    seed_company: str,
    max_tier: int = 1,
) -> BaselineBResult:
    """
    max_tier defaults to 1 because that IS Baseline B's definition.
    Parameterised (rather than hardcoded) only so the same function can
    also be used later for a diagnostic look at how far the graph reaches
    at tier 2+ -- NOT the agent's own tier-depth experiment from
    Days 12-13, which caps System C's hop count and needs the agent.
    Calling this with max_tier=2 tells you about the GRAPH's reach, not
    about multi-hop AGENT reasoning -- don't conflate the two in the
    write-up.
    """
    result = store.traverse_supply_graph(
        seed_company, mode="downstream", max_tier=max_tier
    )

    if result.status == "not_resolved":
        return BaselineBResult(
            event_id=event_id, seed_input=seed_company,
            status="not_resolved", seed_resolved=None,
            predicted_companies=set(),
        )

    predicted = {r["name"] for r in result.results}
    dropped_names = [d["name"] for d in result.dropped_unenriched]

    return BaselineBResult(
        event_id=event_id, seed_input=seed_company,
        status=result.status, seed_resolved=result.seed,
        predicted_companies=predicted,
        n_dropped_unenriched=result.n_dropped_unenriched,
        dropped_unenriched_names=dropped_names,
    )


def run_baseline_b_for_all_events(store: GraphStore, events, max_tier: int = 1
                                   ) -> dict[str, BaselineBResult]:
    """events: list[Event] from event_loader.py. Runs on EVERY loaded
    event, including excluded ones -- evaluate.py is responsible for
    filtering by score_for_recall/score_for_precision at scoring time, not
    this function. Keeping this function exclusion-agnostic means the
    same results dict can also be inspected for the excluded events (e.g.
    to write the event-20 qualitative note) without a second run."""
    out = {}
    for ev in events:
        out[ev.event_id] = run_baseline_b(store, ev.event_id, ev.seed_company, max_tier)
    return out


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from event_loader import load_events

    print("Loading real supply graph...")
    store = GraphStore(GRAPH_PATH)
    print(f"Loaded {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges\n")

    events = load_events()
    results = run_baseline_b_for_all_events(store, events, max_tier=1)

    print("=== Baseline B (single-hop) -- per event ===")
    n_not_resolved = 0
    for ev in events:
        r = results[ev.event_id]
        flag = "" if ev.score_for_recall or ev.score_for_precision else "  (not scored)"
        print(f"  {ev.event_id:45s} seed='{ev.seed_company}' -> "
              f"status={r.status:12s} n_predicted={len(r.predicted_companies):3d} "
              f"n_dropped_unenriched={r.n_dropped_unenriched:3d}{flag}")
        if r.status == "not_resolved":
            n_not_resolved += 1

    print(f"\n{n_not_resolved} / {len(events)} seed companies did not resolve to any "
          f"graph node at all -- worth checking these against find_company_node "
          f"directly, since a wrong seed guess here silently zeroes out that "
          f"event's Baseline B score.")
