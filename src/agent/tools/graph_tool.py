"""
graph_tool.py — Layer 3 tool: traverse_supply_graph()

Loads the supplier graph and provides:
  - find_company_node()      reliable name lookup (exact -> suffix-strip -> fuzzy)
  - traverse_supply_graph()  the actual agent-facing tool, two modes:
        mode="downstream"   -> who is at risk downstream of a company
        mode="substitutes"  -> who else makes a similar component, far enough away
  - filter_to_names()        helper to check specific companies against a result,
                              e.g. one event's ground-truth affected list

Uses graph_enriched_corrected.pkl (not graph_enriched.pkl) - the coordinate
override fix from Day 7 (see coordinate_overrides.py / coordinate_override_log.txt)
is already baked into this file, so no override step is needed at load time.
"""

import pickle
import networkx as nx
from rapidfuzz import process, fuzz
from geopy.distance import geodesic

GRAPH_PATH = "data/processed/graph_enriched_corrected.pkl"

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)


# ---------------------------------------------------------------------------
# Company name lookup
# ---------------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "inc", "ltd", "llc", "plc", "sa", "ag", "nv", "bv", "spa",
    "co", "corp", "gmbh", "group", "holdings",
]


def strip_suffix(name: str) -> str:
    """Remove a trailing legal-entity suffix, if present (e.g. 'posco holdings' -> 'posco')."""
    words = name.lower().strip().split()
    if len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def find_company_node(name: str, threshold: int = 90) -> str | None:
    """
    Resolve a possibly-messy company name to an exact graph node name.
    Tries, in order: exact match -> exact match after suffix-stripping -> fuzzy match.
    Returns None if nothing is confident enough, rather than guessing.
    """
    name_lower = name.lower().strip()

    if G.has_node(name_lower):
        return name_lower

    stripped = strip_suffix(name_lower)
    if stripped != name_lower and G.has_node(stripped):
        return stripped

    all_nodes = list(G.nodes())
    match, score, _ = process.extractOne(stripped, all_nodes, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return match
    return None


# ---------------------------------------------------------------------------
# Downstream traversal (Job 1 - "who's at risk")
# ---------------------------------------------------------------------------

def downstream_from(resolved_name: str, max_tier: int = 2) -> dict[int, set[str]]:
    """
    Walk forward (supplier -> customer) from an already-resolved node, tier by tier.
    Returns {1: {tier-1 companies}, 2: {tier-2 companies}, ...}.
    Cycle-safe via the `visited` set (the graph is not guaranteed acyclic - Day 1 log).
    """
    visited = {resolved_name}
    current_tier_nodes = {resolved_name}
    tiers = {}

    for tier in range(1, max_tier + 1):
        next_tier_nodes = set()
        for node in current_tier_nodes:
            for successor in G.successors(node):
                if successor not in visited:
                    next_tier_nodes.add(successor)

        visited.update(next_tier_nodes)
        tiers[tier] = next_tier_nodes
        current_tier_nodes = next_tier_nodes

        if not next_tier_nodes:
            break

    return tiers


# ---------------------------------------------------------------------------
# Substitute-supplier search (Job 2 - "who else makes this")
# ---------------------------------------------------------------------------

def get_coords(node_name: str) -> tuple[float, float] | None:
    attrs = G.nodes[node_name]
    lat, lon = attrs.get("lat"), attrs.get("lon")
    if lat is None or lon is None:
        return None
    return (lat, lon)


def find_alternative_suppliers(
    seed_node: str, component: str = None, min_distance_km: float = 500
) -> list[tuple[str, str, str, float]]:
    """
    Flat scan (not a walk) for companies making a similar component, far enough
    from the seed company's location to plausibly be unaffected by the same event.
    Uses real coordinates rather than the country label (country attributes are
    known to be unreliable for some rows - see Day 3 log).
    Only scans the 402 enriched nodes, since only they carry `component`/lat/lon.
    """
    seed_coords = get_coords(seed_node)
    if seed_coords is None:
        return []

    matches = []
    for node, attrs in G.nodes(data=True):
        if node == seed_node:
            continue

        node_component = attrs.get("component")
        if component is not None:
            if node_component is None or component.lower() not in node_component.lower():
                continue

        node_coords = get_coords(node)
        if node_coords is None:
            continue

        distance_km = geodesic(seed_coords, node_coords).km
        if distance_km < min_distance_km:
            continue

        matches.append((node, node_component, attrs.get("country"), round(distance_km)))

    matches.sort(key=lambda x: x[3])
    return matches


# ---------------------------------------------------------------------------
# The actual agent-facing tool
# ---------------------------------------------------------------------------

def traverse_supply_graph(
    company_name: str,
    mode: str = "downstream",
    max_tier: int = 2,
    component: str = None,
    min_distance_km: float = 500,
) -> dict:
    """
    mode="downstream"  -> enriched companies at risk downstream of this company
    mode="substitutes" -> enriched companies making a similar component, far enough away

    Only companies with graph enrichment (component/industry attached) are
    returned in "downstream" mode - unenriched nodes are dropped, since there
    is no basis to reason about them. Their names are kept (not just a count)
    so it's possible to tell "unenriched" apart from "genuinely absent" later.
    """
    resolved = find_company_node(company_name)
    if resolved is None:
        return {"error": f"Could not resolve '{company_name}' to a graph node"}

    if mode == "downstream":
        tiers = downstream_from(resolved, max_tier=max_tier)
        results = []
        dropped = []
        for tier, companies in tiers.items():
            for c in companies:
                attrs = G.nodes[c]
                if attrs.get("component") is None:
                    dropped.append({"name": c, "tier": tier})
                    continue
                results.append({
                    "name": c,
                    "tier": tier,
                    "industry": attrs.get("industry"),
                    "country": attrs.get("country"),
                    "component": attrs.get("component"),
                    "confidence": attrs.get("confidence"),
                })
        return {
            "mode": "downstream",
            "seed": resolved,
            "results": results,
            "dropped_unenriched": dropped,
        }

    elif mode == "substitutes":
        matches = find_alternative_suppliers(resolved, component=component, min_distance_km=min_distance_km)
        results = []
        for n, comp, country, dist in matches:
            attrs = G.nodes[n]
            results.append({
                "name": n,
                "component": comp,
                "country": country,
                "distance_km": dist,
                "confidence": attrs.get("confidence"),
            })
        return {"mode": "substitutes", "seed": resolved, "results": results}

    else:
        return {"error": f"Unknown mode '{mode}'"}


# ---------------------------------------------------------------------------
# Helper: scope a result down to specific companies of interest
# ---------------------------------------------------------------------------

def filter_to_names(traverse_result: dict, names_of_interest: list[str]) -> tuple[list[dict], set[str]]:
    """
    Given a traverse_supply_graph() result, keep only the entries matching a
    specific set of company names - e.g. the ground-truth affected companies
    for one event, rather than every enriched company reachable from the seed.
    Returns (kept_entries, names_not_found).
    """
    wanted = {n.lower().strip() for n in names_of_interest}
    kept = [r for r in traverse_result["results"] if r["name"] in wanted]
    missing = wanted - {r["name"] for r in traverse_result["results"]}
    return kept, missing


# ---------------------------------------------------------------------------
# Tests / verification - only runs when this file is executed directly,
# not when it's imported by agent.py or anything else.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== Basic load check ===")
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    print("\n=== Name lookup checks ===")
    for n in ["POSCO", "posco holdings", "some totally fake company xyz"]:
        print(f"  '{n}' -> {find_company_node(n)}")

    print("\n=== Downstream mode: POSCO ===")
    downstream_result = traverse_supply_graph("posco", mode="downstream", max_tier=2)

    by_tier = {}
    for r in downstream_result["results"]:
        by_tier.setdefault(r["tier"], 0)
        by_tier[r["tier"]] += 1
    print("  Counts per tier (enriched only):", by_tier)
    print("  Total kept:", len(downstream_result["results"]))
    print("  Total dropped (unenriched):", len(downstream_result["dropped_unenriched"]))

    print("\n  Known-companies check:")
    tier1_names = {r["name"] for r in downstream_result["results"] if r["tier"] == 1}
    tier2_names = {r["name"] for r in downstream_result["results"] if r["tier"] == 2}
    for name in ["hyundai motor", "kia", "ford motor", "general motors", "tesla"]:
        resolved = find_company_node(name)
        if resolved in tier1_names:
            print(f"    '{name}' -> tier 1")
        elif resolved in tier2_names:
            print(f"    '{name}' -> tier 2")
        else:
            print(f"    '{name}' -> DROPPED or unreachable")

    print("\n=== Substitutes mode: POSCO, component='steel' ===")
    sub_result = traverse_supply_graph("posco", mode="substitutes", component="steel", min_distance_km=500)
    print(f"  Found {len(sub_result['results'])} matches")
    for r in sub_result["results"][:10]:
        print(f"    {r['name']:30s} | {r['country']:15s} | {r['distance_km']:>6} km | conf={r['confidence']}")

    print("\n=== filter_to_names(): check specific event-affected companies ===")
    event_affected = ["ford motor", "general motors", "tesla", "korinox"]  # korinox = known unreachable
    kept, missing = filter_to_names(downstream_result, event_affected)

    print("  Found in traversal:")
    for r in kept:
        print(f"    {r['name']:20s} | tier {r['tier']} | confidence={r['confidence']}")

    print("  Not found:", missing)
    dropped_names = {d["name"] for d in downstream_result["dropped_unenriched"]}
    for name in missing:
        resolved = find_company_node(name)
        if resolved in dropped_names:
            print(f"    '{name}' -> reachable but NOT enriched (dropped)")
        else:
            print(f"    '{name}' -> genuinely absent from the graph")