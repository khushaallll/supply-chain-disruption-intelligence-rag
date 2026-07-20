"""Build the supply chain graph from cleaned data with reviewed merges applied.

Re-runs the same company/country normalization as clean.py, then applies the
approved rows of data/review/merge_decisions.csv (rejected pairs are left
untouched) before building a networkx.DiGraph: edges point supplier ->
customer, and each node carries industry/country attributes resolved by
majority vote over its (post-merge) rows. Saved as data/processed/graph_clean.pkl.
"""

import pickle
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from clean import RAW_CSV, clean_countries, normalize_company_name

DECISIONS_CSV = Path("data/review/merge_decisions.csv")
GRAPH_PKL = Path("data/processed/graph_clean.pkl")


def build_merge_map(decisions_csv):
    """(company -> final canonical name) for every company touched by an
    approved merge. Approved pairs are unioned into connected components
    (handles chains like A~B, B~C), and each component's label is the
    majority-vote (case-insensitive) canonical_name across its member pairs
    — so one inconsistent/typo'd canonical entry on a single pair can't
    split or mislabel the group. Labels are lowercased to match the
    lowercase company_clean convention used everywhere else."""
    decisions = pd.read_csv(decisions_csv)
    approved = decisions[decisions["decision"] == "approve"]

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in approved.itertuples(index=False):
        union(row.company_a, row.company_b)

    votes = {}
    for row in approved.itertuples(index=False):
        root = find(row.company_a)
        key = str(row.canonical_name).strip().lower()
        votes.setdefault(root, {}).setdefault(key, 0)
        votes[root][key] += 1

    final_label = {}
    for root, counts in votes.items():
        max_count = max(counts.values())
        final_label[root] = sorted(k for k, v in counts.items() if v == max_count)[0]

    return {member: final_label[find(member)] for member in parent}


def build_relationships(merge_map):
    """Row-level table with the same cleaning as clean.py (drop unused
    relationship columns, normalize countries, normalize company names) plus
    the approved merges relabeled onto Supplier/Customer."""
    df = pd.read_csv(RAW_CSV)
    df = df.drop(columns=["Relationship2", "Relationship3"])
    df = clean_countries(df)

    def clean_and_merge(name):
        cleaned = normalize_company_name(name)
        return merge_map.get(cleaned, cleaned)

    df["Supplier"] = df["Supplier"].apply(clean_and_merge)
    df["Customer"] = df["Customer"].apply(clean_and_merge)
    return df


def resolve_node_attributes(df):
    """Majority-vote industry/country per final (post-merge) company name,
    over all rows where that company appears as either supplier or customer."""
    supplier_side = df[["Supplier", "Supplier Industry", "Supplier Country"]].rename(
        columns={"Supplier": "company", "Supplier Industry": "industry", "Supplier Country": "country"}
    )
    customer_side = df[["Customer", "Customer Industry", "Customer Country"]].rename(
        columns={"Customer": "company", "Customer Industry": "industry", "Customer Country": "country"}
    )
    long_df = pd.concat([supplier_side, customer_side], ignore_index=True)

    attrs = {}
    for company, group in long_df.groupby("company"):
        attrs[company] = {
            "industry": group["industry"].value_counts().idxmax(),
            "country": group["country"].value_counts().idxmax(),
        }
    return attrs


def build_graph(df, node_attrs):
    g = nx.DiGraph()
    for company, attrs in node_attrs.items():
        g.add_node(company, **attrs)
    for row in df.itertuples(index=False):
        g.add_edge(row.Supplier, row.Customer, relationship=row.Relationship1)
    return g


def find_suppliers(g, company, country=None, industry=None, max_tier=2):
    """Upstream suppliers of `company` (via predecessors, since edges are
    supplier -> customer) within `max_tier` hops. A global `visited` set
    guards traversal so cycles in the graph can't cause an infinite loop or
    revisit a node.

    Filters only affect which results are *returned*, not which nodes are
    traversed through — a non-matching tier-1 supplier can still lead to a
    matching tier-2 one.

    No `component` (physical component/product type, e.g. "microcontroller")
    filter: the source data has no such field. Relationship1 is a two-value
    relationship-type enum (suppliesTo/owns) and Relationship2/Relationship3
    are constant across every row (locatedIn/operatesIn) — none of them
    describe what's being supplied. Revisit if a data source with real
    product/component-level detail becomes available.
    """
    if company not in g:
        return []

    results = []
    visited = {company}
    frontier = [(company, 0)]

    while frontier:
        node, tier = frontier.pop(0)
        if tier >= max_tier:
            continue
        for supplier in g.predecessors(node):
            if supplier in visited:
                continue
            visited.add(supplier)
            frontier.append((supplier, tier + 1))

            attrs = g.nodes[supplier]
            if country is not None and attrs.get("country") != country:
                continue
            if industry is not None and attrs.get("industry") != industry:
                continue

            results.append({
                "company": supplier,
                "tier": tier + 1,
                "industry": attrs.get("industry"),
                "country": attrs.get("country"),
            })

    return results


def run_acceptance_checks(g, df):
    print("\n=== Acceptance checks ===")

    denso_rows = ((df["Supplier"] == "denso") | (df["Customer"] == "denso")).sum()
    denso_degree = g.in_degree("denso") + g.out_degree("denso") if "denso" in g else 0
    print(f"denso: {denso_rows} merged relationship rows (expected ~160 = 140+20); "
          f"{denso_degree} unique graph edges (in+out degree)")

    ford_suppliers = find_suppliers(g, "ford motor", max_tier=1)
    print(f'"ford motor" Tier-1 supplier count: {len(ford_suppliers)}')

    is_dag = nx.is_directed_acyclic_graph(g)
    print(f"Graph is a DAG: {is_dag}")
    if not is_dag:
        cycle = nx.find_cycle(g)
        cycle_node = cycle[0][0]
        start = time.perf_counter()
        result = find_suppliers(g, cycle_node, max_tier=5)
        elapsed = time.perf_counter() - start
        print(f"Cycle found through {cycle_node!r} (e.g. {cycle[:3]}...); "
              f"find_suppliers(max_tier=5) from it returned {len(result)} results "
              f"in {elapsed:.4f}s without hanging")


def main():
    merge_map = build_merge_map(DECISIONS_CSV)
    df = build_relationships(merge_map)
    node_attrs = resolve_node_attributes(df)
    g = build_graph(df, node_attrs)

    GRAPH_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(g, f)

    print("=== Graph summary ===")
    print(f"Companies merged via approved decisions: {len(merge_map)}")
    print(f"Nodes: {g.number_of_nodes()}")
    print(f"Edges: {g.number_of_edges()}")
    print(f"Saved to: {GRAPH_PKL}")

    run_acceptance_checks(g, df)


if __name__ == "__main__":
    main()
