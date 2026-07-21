"""Review-before-removal workflow for extraction-noise nodes in graph_clean.pkl.

Two modes, run via CLI:
    python src/clean_suspect_nodes.py detect --min-total-degree 2
    python src/clean_suspect_nodes.py apply

detect flags suspect nodes (missing industry/country, or low total degree)
and writes them to data/review/suspect_nodes.csv with an empty "decision"
column for manual "keep"/"remove" review. apply reads that file back after
review, removes only the rows marked "remove", re-saves the graph, and logs
what was removed to data/review/removal_log.csv. Nothing is ever removed
automatically — apply only acts on decisions already recorded by hand.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import networkx as nx

GRAPH_PKL = Path("data/processed/graph_clean.pkl")
REVIEW_DIR = Path("data/review")
SUSPECT_CSV = REVIEW_DIR / "suspect_nodes.csv"
REMOVAL_LOG_CSV = REVIEW_DIR / "removal_log.csv"

DEFAULT_MIN_TOTAL_DEGREE = 2


def load_graph(path):
    if not path.exists():
        print(f"Error: graph file not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_graph(g, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)


def is_missing(value):
    return value is None or str(value).strip() == ""


def classify_node(g, node, min_total_degree):
    """Return a list of reason labels (empty if the node isn't suspect)."""
    attrs = g.nodes[node]
    industry = attrs.get("industry")
    country = attrs.get("country")
    in_degree = g.in_degree(node)
    out_degree = g.out_degree(node)

    reasons = []
    if is_missing(industry) or is_missing(country):
        reasons.append("missing attrs")
    if (in_degree + out_degree) < min_total_degree:
        reasons.append("low degree")
    return reasons, industry, country, in_degree, out_degree


def cmd_detect(args):
    """Flag suspect nodes and write them to data/review/suspect_nodes.csv
    for manual keep/remove review. Never modifies the graph."""
    g = load_graph(GRAPH_PKL)

    rows = []
    for node in g.nodes():
        reasons, industry, country, in_degree, out_degree = classify_node(g, node, args.min_total_degree)
        if reasons:
            rows.append({
                "node": node,
                "industry": industry if industry is not None else "",
                "country": country if country is not None else "",
                "in_degree": in_degree,
                "out_degree": out_degree,
                "flag_reason": "; ".join(reasons),
                "decision": "",
            })

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUSPECT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node", "industry", "country", "in_degree", "out_degree", "flag_reason", "decision"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Flagged {len(rows)} suspect node(s) out of {g.number_of_nodes()} total.")
    print(f"Written to: {SUSPECT_CSV}")
    print('Fill in the "decision" column with keep/remove, then run: apply')


def cmd_apply(args):
    """Read reviewed data/review/suspect_nodes.csv, remove only rows marked
    "remove", re-save the graph, and log what was removed."""
    if not SUSPECT_CSV.exists():
        print(f"Error: {SUSPECT_CSV} not found. Run `detect` first.", file=sys.stderr)
        sys.exit(1)

    with open(SUSPECT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    to_remove_rows = [r for r in rows if r.get("decision", "").strip().lower() == "remove"]
    undecided = sum(1 for r in rows if r.get("decision", "").strip().lower() not in ("keep", "remove"))
    if undecided:
        print(f'Note: {undecided} row(s) have no keep/remove decision yet and will be left in the graph.')

    g = load_graph(GRAPH_PKL)
    before_nodes, before_edges = g.number_of_nodes(), g.number_of_edges()

    existing = set(g.nodes())
    nodes_to_remove = [r["node"] for r in to_remove_rows]
    actually_present = [r for r in to_remove_rows if r["node"] in existing]
    missing = [r["node"] for r in to_remove_rows if r["node"] not in existing]
    if missing:
        print(f"Note: {len(missing)} node(s) marked remove were already absent from the graph: {missing}")

    # remove_nodes_from (not remove_node) so a mismatched or already-gone
    # name doesn't raise — it's silently skipped.
    g.remove_nodes_from(nodes_to_remove)

    save_graph(g, GRAPH_PKL)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(REMOVAL_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node", "industry", "country", "in_degree", "out_degree", "removed_reason"])
        writer.writeheader()
        for r in actually_present:
            writer.writerow({
                "node": r["node"],
                "industry": r["industry"],
                "country": r["country"],
                "in_degree": r["in_degree"],
                "out_degree": r["out_degree"],
                "removed_reason": f'flagged: {r["flag_reason"]}',
            })

    print(f"Removed {len(actually_present)} node(s).")
    print(f"Graph: {before_nodes} -> {g.number_of_nodes()} nodes, {before_edges} -> {g.number_of_edges()} edges")
    print(f"Saved to: {GRAPH_PKL}")
    print(f"Removal log written to: {REMOVAL_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Review-before-removal workflow for suspect nodes in graph_clean.pkl.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    detect_parser = subparsers.add_parser("detect", help="Flag suspect nodes for manual review.")
    detect_parser.add_argument(
        "--min-total-degree", type=int, default=DEFAULT_MIN_TOTAL_DEGREE,
        help=f"Nodes with in_degree + out_degree below this are flagged (default: {DEFAULT_MIN_TOTAL_DEGREE}).",
    )
    detect_parser.set_defaults(func=cmd_detect)

    apply_parser = subparsers.add_parser("apply", help="Remove nodes marked 'remove' in suspect_nodes.csv.")
    apply_parser.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
