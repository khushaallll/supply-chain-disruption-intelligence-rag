"""
screen_events.py

For a given company (or list of companies affected
by an event), checks whether the supplier graph has enough structure around
that company to be useful for the agent later:

  - downstream_customer_count : how many companies it supplies directly
                                 (1 hop out, since edges run supplier -> customer)
  - two_hop_oem_reach         : which companies show up 2 hops out
                                 (its customers' customers)

Two ways to use it:
  1. Single company lookup   -> python screen_events.py company "Formosa Plastics"
  2. Batch over your 60      -> python screen_events.py batch events_candidates.json
     candidate events, adding visibility fields to each and writing the
     result back out.
"""

import argparse
import json
import pickle
import networkx as nx


GRAPH_PKL = "data/processed/graph_clean.pkl"  # matches your real graph location


def load_graph(path: str = GRAPH_PKL) -> nx.DiGraph:
    """
    Loads the supplier graph from disk.

    NetworkX graphs aren't plain text -- they're Python objects (nodes, edges,
    attributes all bundled together), so they get saved/loaded with `pickle`
    rather than something like json or csv.
    """
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def screen_single_company(G: nx.DiGraph, company_name: str) -> dict:
    """
    Runs the visibility test for one company.

    NetworkX concepts used here:
      - G.has_node(name)      -> does this node exist in the graph at all?
      - G.successors(name)    -> nodes this node has an OUTGOING edge to.
                                  Since our edges are supplier -> customer,
                                  successors of a company = its customers.
      - Going one more layer (successors of successors) = 2 hops out.
    """
    result = {
        "company": company_name,
        "found_in_graph": False,
        "downstream_customer_count": 0,
        "two_hop_oem_reach": [],
        "passes_visibility": False,
    }

    # Guard clause: if the name isn't an exact match to a graph node, stop here.
    # This is exactly the check that catches naming mismatches from Day 1
    # (e.g. "TSMC" vs. the graph's actual cleaned entity name).
    if not G.has_node(company_name):
        return result

    result["found_in_graph"] = True

    # --- 1 hop: direct downstream customers ---
    # list() turns the successors iterator into an actual list we can use twice.
    direct_customers = list(G.successors(company_name))
    result["downstream_customer_count"] = len(direct_customers)

    # --- 2 hops: customers of those customers ---
    # We use a set() here, not a list, because the same 2-hop company can be
    # reachable through more than one 1-hop customer (e.g. two different
    # suppliers both feed into Toyota) -- a set automatically removes duplicates.
    two_hop_set = set()
    for customer in direct_customers:
        for second_hop in G.successors(customer):
            # Exclude the original company itself, in case the graph has a
            # cycle that loops back (Day 1 confirmed cycles exist).
            if second_hop != company_name:
                two_hop_set.add(second_hop)

    result["two_hop_oem_reach"] = sorted(two_hop_set)

    # --- Pass/fail rule from the Day 2 plan ---
    # "At least one directly-affected company must have downstream customers."
    result["passes_visibility"] = result["downstream_customer_count"] > 0

    return result


def screen_event(G: nx.DiGraph, event: dict) -> dict:
    """
    Runs the visibility test across every company listed in an event's
    'companies_affected' list, and folds the results back into the event dict.

    An event passes overall if ANY of its affected companies pass individually
    -- matching the plan's rule ("at least one directly-affected company").
    """
    company_results = [
        screen_single_company(G, name) for name in event.get("companies_affected", [])
    ]

    # Take the best-connected result across the affected companies, so the
    # numbers we report reflect the strongest anchor, not an average.
    best = max(
        company_results,
        key=lambda r: r["downstream_customer_count"],
        default=None,
    )

    event["downstream_customer_count"] = best["downstream_customer_count"] if best else 0
    event["two_hop_oem_reach"] = best["two_hop_oem_reach"] if best else []
    event["passes_visibility"] = any(r["passes_visibility"] for r in company_results)

    return event


def run_batch(events_path: str, output_path: str = None):
    """
    Loads events_candidates.json, re-screens every event against the (cleaned)
    graph, and writes the updated file back out with fresh visibility numbers.
    """
    G = load_graph()

    with open(events_path, "r") as f:
        events = json.load(f)

    updated_events = [screen_event(G, event) for event in events]

    output_path = output_path or events_path
    with open(output_path, "w") as f:
        json.dump(updated_events, f, indent=2)

    passed = sum(1 for e in updated_events if e["passes_visibility"])
    print(f"Screened {len(updated_events)} events.")
    print(f"Passed visibility test: {passed}")
    print(f"Failed visibility test: {len(updated_events) - passed}")

    # Quick per-category breakdown, since you're balancing across 4 categories.
    by_category = {}
    for e in updated_events:
        cat = e.get("category", "Unknown")
        by_category.setdefault(cat, {"pass": 0, "fail": 0})
        key = "pass" if e["passes_visibility"] else "fail"
        by_category[cat][key] += 1

    print("\nBy category:")
    for cat, counts in by_category.items():
        print(f"  {cat}: {counts['pass']} passed, {counts['fail']} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 2 visibility test.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    company_parser = subparsers.add_parser("company", help="Screen a single company.")
    company_parser.add_argument("name", help="Exact company name as it appears in the graph.")

    batch_parser = subparsers.add_parser("batch", help="Screen all events in a JSON file.")
    batch_parser.add_argument("events_json", help="Path to events_candidates.json")
    batch_parser.add_argument(
        "--output", default=None, help="Where to write results (defaults to overwriting input)."
    )

    args = parser.parse_args()

    if args.mode == "company":
        graph = load_graph()
        print(json.dumps(screen_single_company(graph, args.name), indent=2))
    elif args.mode == "batch":
        run_batch(args.events_json, args.output)
        