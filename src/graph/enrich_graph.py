"""
enrich_graph.py

Applies the geocoded locations.csv and components.csv onto graph_clean.pkl,
producing graph_enriched.pkl. graph_clean.pkl itself is left untouched as a
pre-enrichment checkpoint.

Usage:
    python enrich_graph.py \
        --graph graph_clean.pkl \
        --locations locations.csv \
        --components components.csv \
        --out graph_enriched.pkl

Nodes outside the enriched company set simply don't get lat/lon/component
attributes - anything reading these later should guard with .get(...) rather
than assume every node has them.
"""

import argparse
import pickle

import networkx as nx
import pandas as pd


def build_attr_dict(df, key_col, value_cols, require_notna=None):
    """Build a {node_name: {attr: value, ...}} dict from a dataframe, keyed on
    a stripped company name. Rows failing `require_notna` (if given) are skipped."""
    attrs = {}
    for _, row in df.iterrows():
        if require_notna is not None and pd.isna(row[require_notna]):
            continue
        company = str(row[key_col]).strip()
        attrs[company] = {col: row[col] for col in value_cols}
    return attrs


def report_unmatched(attrs, G, label):
    unmatched = set(attrs.keys()) - set(G.nodes())
    if unmatched:
        print(f"WARNING: {len(unmatched)} companies in {label} not found as graph nodes (typo/rename mismatch?):")
        for name in sorted(unmatched):
            print(f"  - {name!r}")
    else:
        print(f"All {label} companies matched a graph node.")
    return unmatched


def main():
    parser = argparse.ArgumentParser(description="Enrich graph_clean.pkl with geocoded location + component attributes")
    parser.add_argument("--graph", default="graph_clean.pkl")
    parser.add_argument("--locations", default="locations.csv")
    parser.add_argument("--components", default="components.csv")
    parser.add_argument("--out", default="graph_enriched.pkl")
    args = parser.parse_args()

    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    locations = pd.read_csv(args.locations)
    components = pd.read_csv(args.components)

    # skip rows that failed to geocode (lat is NaN) rather than writing null coordinates
    loc_attrs = build_attr_dict(locations, key_col="company", value_cols=["lat", "lon"], require_notna="lat")
    comp_attrs = build_attr_dict(components, key_col="company", value_cols=["component", "confidence"])

    report_unmatched(loc_attrs, G, args.locations)
    report_unmatched(comp_attrs, G, args.components)

    n_skipped_geocode = len(locations) - len(loc_attrs)
    if n_skipped_geocode:
        print(f"NOTE: {n_skipped_geocode} rows in {args.locations} had no lat/lon (failed geocode) and were skipped.")

    nx.set_node_attributes(G, loc_attrs)
    nx.set_node_attributes(G, comp_attrs)

    n_with_location = sum(1 for _, d in G.nodes(data=True) if "lat" in d)
    n_with_component = sum(1 for _, d in G.nodes(data=True) if "component" in d)
    print()
    print(f"Nodes with lat/lon:     {n_with_location} / {G.number_of_nodes()}")
    print(f"Nodes with component:  {n_with_component} / {G.number_of_nodes()}")

    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved enriched graph to {args.out}")

    # acceptance check - spot-check a few known companies
    print("\nSpot-check:")
    for name in ["denso", "ford motor", "oci"]:
        if name in G:
            print(f"  {name}: {G.nodes[name]}")
        else:
            print(f"  {name}: NOT IN GRAPH (check exact node name)")


if __name__ == "__main__":
    main()
