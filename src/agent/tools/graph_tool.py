"""
graph_tool.py — Layer 3 tool: traverse_supply_graph()

Revision notes (this pass specifically brought this tool in line with
get_supplier_info() and search_corpus() -- see the six numbered points
below, each mapped to a concrete change):

  1. STATUS FIELD. traverse_supply_graph() used to return a bare dict, with
     failure signalled only by an {"error": ...} key -- a different shape
     from the other two tools' explicit `status` field. Now returns a
     GraphTraversalResult with status in {"found", "no_results",
     "not_resolved", "invalid_mode"} -- four outcomes, not one, for the
     same reason get_supplier_info() distinguishes "company not on file"
     from "company on file but nothing recent enough": each means
     something different, and a stopping condition needs to tell them apart.

  2. READABLE OBSERVATION. Added GraphTraversalResult.as_observation(),
     mirroring SupplierInfoResult and CorpusSearchResult -- turns the
     structured result into one paragraph. Previously this tool had no
     text-rendering step at all; its dict would have been dumped to the
     model as a raw Python repr.

  3. LLM-FACING TOOL DESCRIPTION. The main traverse_supply_graph() method
     now has a docstring written for the MODEL that will call it -- when to
     use downstream vs. substitutes mode, and what it's for -- separate
     from the DEVELOPER-facing design-rationale comments elsewhere in this
     file. This is deliberately NOT the same thing as the system prompt:
     the system prompt (Day 8) is cross-tool STRATEGY ("call this tool
     first, stop when..."); a tool's own docstring is what a LangGraph tool
     wrapper surfaces to the model as that ONE tool's spec, and is
     addable now, tool by tool, independent of the agent loop existing yet.

  4. LOAD-ONCE STORE. The graph used to load at module IMPORT time, via a
     bare module-level `with open(...) as f: G = pickle.load(f)` and a
     hardcoded path. Now wrapped in GraphStore, constructed explicitly by
     the caller (mirrors SupplierInfoStore / CorpusSearchStore) -- plus a
     loud, fatal check that the loaded graph isn't empty, same "absence
     must be fatal and loud" principle as CorpusSearchStore's Chroma-count
     check.

  5. COMPANY NAME AMBIGUITY ACROSS TOOLS. Deliberately UNCHANGED here --
     this tool still resolves names via fuzzy matching (threshold 90), as
     before. Reconciling that against get_supplier_info()'s exact-match-only
     policy is explicitly deferred, not addressed in this revision.

  6. STOPPING-CONDITION-RELEVANT CHANGES. The 4-hop cap and any looping
     logic are explicitly OUT OF SCOPE here -- that's agent-state, built
     during Day 8, not something a single tool call should track. What
     WAS added, because it lives naturally inside this tool's own result
     rather than the agent: `n_results` and `n_dropped_unenriched` as
     directly-readable counts on GraphTraversalResult, so a stopping
     condition can read "how much did this hop find" at a glance -- the
     same role distinct_companies/hit-counts play on CorpusSearchResult --
     without re-deriving it by counting a list itself. The `status` field
     from point 1 is the single most important stopping-condition-relevant
     change: it's what lets the agent's stopping logic read one consistent
     signal shape across all three tools, rather than three different ones.

Uses graph_enriched_corrected.pkl (not graph_enriched.pkl) -- the
coordinate override fix from Day 7 is already baked into this file, so no
override step is needed at load time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
from rapidfuzz import process, fuzz
from geopy.distance import geodesic

GRAPH_PATH = "data/processed/graph_enriched_corrected.pkl"

LEGAL_SUFFIXES = [
    "inc", "ltd", "llc", "plc", "sa", "ag", "nv", "bv", "spa",
    "co", "corp", "gmbh", "group", "holdings",
]


def strip_suffix(name: str) -> str:
    """Remove a trailing legal-entity suffix, if present (e.g. 'posco holdings' -> 'posco').
    Pure function, no graph needed -- kept at module level, same pattern as
    resolve_company_name() in search_corpus.py."""
    words = name.lower().strip().split()
    if len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
        words = words[:-1]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class GraphTraversalResult:
    status: str                    # found | no_results | not_resolved | invalid_mode
    mode: str
    company_name: str              # as given by the caller, before resolution
    seed: Optional[str] = None     # resolved graph node name
    results: list = field(default_factory=list)
    dropped_unenriched: list = field(default_factory=list)  # downstream mode only
    n_results: int = 0
    n_dropped_unenriched: int = 0

    def as_observation(self) -> str:
        """Plain-text form to hand back to the LangGraph agent -- same role
        as SupplierInfoResult.as_observation() and CorpusSearchResult.as_observation()."""
        if self.status == "not_resolved":
            return f"Could not find '{self.company_name}' in the supply graph."

        if self.status == "invalid_mode":
            return (f"'{self.mode}' is not a recognized traversal mode -- "
                    f"use 'downstream' or 'substitutes'.")

        if self.status == "no_results":
            if self.mode == "downstream":
                return (f"'{self.seed}' was found in the graph, but no enriched "
                        f"companies were reachable downstream within the tier "
                        f"limit used.")
            return (f"'{self.seed}' was found in the graph, but no alternative "
                    f"suppliers matched the given component/distance constraints.")

        if self.mode == "downstream":
            lines = [f"[downstream of '{self.seed}' | {self.n_results} enriched "
                     f"compan{'y' if self.n_results == 1 else 'ies'} found, "
                     f"{self.n_dropped_unenriched} reachable but unenriched]"]
            by_tier: dict[int, list] = {}
            for r in self.results:
                by_tier.setdefault(r["tier"], []).append(r)
            for tier in sorted(by_tier):
                lines.append(f"\nTier {tier}:")
                for r in by_tier[tier]:
                    lines.append(f"  - {r['name']} ({r['industry']}, {r['country']}) "
                                 f"makes {r['component']}, confidence={r['confidence']}")
            if self.dropped_unenriched:
                dropped_names = ", ".join(d["name"] for d in self.dropped_unenriched)
                lines.append(f"\nReachable but not enriched (no basis to reason "
                             f"about them further): {dropped_names}")
            return "\n".join(lines)

        # mode == "substitutes"
        lines = [f"[alternative suppliers to '{self.seed}' | {self.n_results} found]"]
        for r in self.results:
            lines.append(f"  - {r['name']} ({r['country']}, {r['distance_km']} km away) "
                         f"makes {r['component']}, confidence={r['confidence']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The store -- loads the graph ONCE, exposes traverse_supply_graph() as the tool
# ---------------------------------------------------------------------------

class GraphStore:
    """
    Loads the supplier graph once at construction time -- mirrors
    SupplierInfoStore (supplier_info.py) and CorpusSearchStore
    (search_corpus.py) -- instead of the previous module-level
    `with open(...) as f: G = pickle.load(f)`, which fired the moment this
    file was merely imported, using a hardcoded path with no caller control.
    """

    def __init__(self, graph_path: str | Path = GRAPH_PATH, phonebook=None):
        with open(graph_path, "rb") as f:
            self.G: nx.DiGraph = pickle.load(f)

        self.phonebook = phonebook  # optional Phonebook -- see phonebook.py

        if self.G.number_of_nodes() == 0:
            raise RuntimeError(
                f"Loaded graph from {graph_path} has ZERO nodes -- refusing "
                f"to proceed silently. A silently-empty graph would make "
                f"every future traverse_supply_graph() call report "
                f"'not_resolved' for every company, which looks like normal "
                f"tool behaviour rather than a loading failure. Same "
                f"'absence must be fatal and loud' principle as "
                f"CorpusSearchStore's Chroma/corpus count check."
            )

    # -- company name lookup ---------------------------------------------------
    def find_company_node(self, name: str, threshold: int = 90) -> str | None:
        """
        Resolve a possibly-messy company name to an exact graph node name.
        Tries, in order: exact match -> exact match after suffix-stripping ->
        fuzzy match. Returns None if nothing is confident enough, rather
        than guessing. UNCHANGED from the original -- see point 5 above,
        reconciling this against get_supplier_info()'s exact-match-only
        policy is explicitly deferred.
        """
        name_lower = name.lower().strip()

        # Step 0: phonebook lookup, if one was provided -- a pre-reviewed
        # answer beats any guess, fuzzy or otherwise. A miss here just
        # means this name isn't one of the 111 known ones; fall through
        # to the tool's own matching, unchanged, exactly as before.
        if self.phonebook is not None:
            ph_match = self.phonebook.graph_name(name)
            if ph_match is not None:
                return ph_match

        if self.G.has_node(name_lower):
            return name_lower

        stripped = strip_suffix(name_lower)
        if stripped != name_lower and self.G.has_node(stripped):
            return stripped

        all_nodes = list(self.G.nodes())
        match, score, _ = process.extractOne(stripped, all_nodes, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return match
        return None

    # -- downstream traversal (Job 1 - "who's at risk") ------------------------
    def downstream_from(self, resolved_name: str, max_tier: int = 2) -> dict[int, set[str]]:
        """
        Walk forward (supplier -> customer) from an already-resolved node,
        tier by tier. Returns {1: {tier-1 companies}, 2: {tier-2 companies}, ...}.
        Cycle-safe via the `visited` set (the graph is not guaranteed
        acyclic - Day 1 log). UNCHANGED logic from the original.
        """
        visited = {resolved_name}
        current_tier_nodes = {resolved_name}
        tiers: dict[int, set[str]] = {}

        for tier in range(1, max_tier + 1):
            next_tier_nodes = set()
            for node in current_tier_nodes:
                for successor in self.G.successors(node):
                    if successor not in visited:
                        next_tier_nodes.add(successor)

            visited.update(next_tier_nodes)
            tiers[tier] = next_tier_nodes
            current_tier_nodes = next_tier_nodes

            if not next_tier_nodes:
                break

        return tiers

    # -- substitute-supplier search (Job 2 - "who else makes this") ------------
    def get_coords(self, node_name: str) -> tuple[float, float] | None:
        attrs = self.G.nodes[node_name]
        lat, lon = attrs.get("lat"), attrs.get("lon")
        if lat is None or lon is None:
            return None
        return (lat, lon)

    def find_alternative_suppliers(
        self, seed_node: str, component: str = None, min_distance_km: float = 500
    ) -> list[tuple[str, str, str, float]]:
        """
        Flat scan (not a walk) for companies making a similar component,
        far enough from the seed company's location to plausibly be
        unaffected by the same event. Uses real coordinates rather than the
        country label (country attributes are known to be unreliable for
        some rows - see Day 3 log). Only scans the enriched nodes, since
        only they carry `component`/lat/lon. UNCHANGED logic from the original.
        """
        seed_coords = self.get_coords(seed_node)
        if seed_coords is None:
            return []

        matches = []
        for node, attrs in self.G.nodes(data=True):
            if node == seed_node:
                continue

            node_component = attrs.get("component")
            if component is not None:
                if node_component is None or component.lower() not in node_component.lower():
                    continue

            node_coords = self.get_coords(node)
            if node_coords is None:
                continue

            distance_km = geodesic(seed_coords, node_coords).km
            if distance_km < min_distance_km:
                continue

            matches.append((node, node_component, attrs.get("country"), round(distance_km)))

        matches.sort(key=lambda x: x[3])
        return matches

    # -- the tool ------------------------------------------------------------------
    def traverse_supply_graph(
        self,
        company_name: str,
        mode: str = "downstream",
        max_tier: int = 2,
        component: str = None,
        min_distance_km: float = 500,
    ) -> GraphTraversalResult:
        """
        Find companies structurally connected to a given company in the
        supply graph. Use this FIRST when investigating a disruption,
        before checking any specific company's filings or searching the
        corpus by topic -- this tool answers "who is even relevant here?"

        mode="downstream" (default): companies at risk of being affected BY
            a disruption at company_name (its customers, and their
            customers, up to max_tier steps away). Use when a company has
            just been hit by an event and you need to find who might be
            impacted downstream.
        mode="substitutes": other companies making a similar component to
            company_name, far enough away (min_distance_km) to plausibly be
            unaffected by the same event. Use when checking whether an
            affected company's customers have an alternative source.

        Only ENRICHED companies (known industry/component data) are
        returned in `results` for downstream mode -- structurally-connected
        but unenriched companies are still counted and named in
        `dropped_unenriched`, not silently discarded, so a later step can
        tell "no signal" apart from "genuinely nothing there."
        """
        resolved = self.find_company_node(company_name)
        if resolved is None:
            return GraphTraversalResult(
                status="not_resolved", mode=mode, company_name=company_name,
            )

        if mode == "downstream":
            tiers = self.downstream_from(resolved, max_tier=max_tier)
            results = []
            dropped = []
            for tier, companies in tiers.items():
                for c in companies:
                    attrs = self.G.nodes[c]
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

            # REVISION (post-30-event real-batch review): sort by
            # confidence, highest first, WITHIN each tier. Tier grouping
            # itself (see as_observation()) is unchanged -- this only
            # orders companies inside a tier, it does not reorder tiers.
            #
            # Why this matters: downstream_from() returns each tier as a
            # Python SET, which has no defined iteration order at all --
            # `results` was being built in essentially arbitrary order
            # before this fix, even though every entry already carries a
            # real `confidence` score. The agent's only signal for "which
            # candidate is strongest" was therefore buried in an unsorted
            # wall of text it had to read and compare by hand, on events
            # where a single tier can hold dozens of candidates.
            #
            # Confirmed on the real 30-event trace batch: 70% of the
            # agent's get_supplier_info calls (26 of 37) targeted a small
            # set of globally well-known companies (Toyota, Apple, Tesla,
            # Ford, Hyundai...) rather than the graph's own
            # highest-confidence candidate for that specific event --
            # consistent with the model falling back on training-data
            # familiarity to fill the gap left by an unordered list,
            # rather than genuinely reading the graph's own confidence
            # field. This costs nothing to fix -- same companies, same
            # tiers, same data, only the ORDER changes -- and gives the
            # agent's limited evidence-gathering turns a principled
            # "check the strongest candidates first" signal instead.
            #
            # None-confidence entries sort last within their tier (via the
            # `is None` primary key), not first and not scattered randomly
            # among the real, scored entries.
            results.sort(key=lambda r: (r["confidence"] is None, -(r["confidence"] or 0)))

            status = "found" if results else "no_results"
            return GraphTraversalResult(
                status=status, mode=mode, company_name=company_name, seed=resolved,
                results=results, dropped_unenriched=dropped,
                n_results=len(results), n_dropped_unenriched=len(dropped),
            )

        elif mode == "substitutes":
            matches = self.find_alternative_suppliers(
                resolved, component=component, min_distance_km=min_distance_km
            )
            results = []
            for n, comp, country, dist in matches:
                attrs = self.G.nodes[n]
                results.append({
                    "name": n,
                    "component": comp,
                    "country": country,
                    "distance_km": dist,
                    "confidence": attrs.get("confidence"),
                })
            status = "found" if results else "no_results"
            return GraphTraversalResult(
                status=status, mode=mode, company_name=company_name, seed=resolved,
                results=results, n_results=len(results),
            )

        else:
            return GraphTraversalResult(
                status="invalid_mode", mode=mode, company_name=company_name, seed=resolved,
            )


# ---------------------------------------------------------------------------
# Helper: scope a result down to specific companies of interest
# ---------------------------------------------------------------------------

def filter_to_names(
    traverse_result: GraphTraversalResult, names_of_interest: list[str]
) -> tuple[list[dict], set[str]]:
    """
    Given a traverse_supply_graph() result, keep only the entries matching a
    specific set of company names - e.g. the ground-truth affected companies
    for one event, rather than every enriched company reachable from the seed.
    Returns (kept_entries, names_not_found). Pure function, no graph needed --
    unchanged in logic, only updated to read `.results` off the dataclass
    instead of indexing a plain dict.
    """
    wanted = {n.lower().strip() for n in names_of_interest}
    kept = [r for r in traverse_result.results if r["name"] in wanted]
    missing = wanted - {r["name"] for r in traverse_result.results}
    return kept, missing


# ---------------------------------------------------------------------------
# Tests / verification - only runs when this file is executed directly,
# not when it's imported by agent.py or anything else.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    store = GraphStore()

    print("=== Basic load check ===")
    print("Nodes:", store.G.number_of_nodes(), "Edges:", store.G.number_of_edges())

    print("\n=== Name lookup checks ===")
    for n in ["POSCO", "posco holdings", "some totally fake company xyz"]:
        print(f"  '{n}' -> {store.find_company_node(n)}")

    print("\n=== Downstream mode: POSCO ===")
    downstream_result = store.traverse_supply_graph("posco", mode="downstream", max_tier=2)
    print("  status:", downstream_result.status)

    by_tier = {}
    for r in downstream_result.results:
        by_tier.setdefault(r["tier"], 0)
        by_tier[r["tier"]] += 1
    print("  Counts per tier (enriched only):", by_tier)
    print("  Total kept:", downstream_result.n_results)
    print("  Total dropped (unenriched):", downstream_result.n_dropped_unenriched)

    print("\n  --- as_observation() preview ---")
    print(downstream_result.as_observation())

    print("\n  Known-companies check:")
    tier1_names = {r["name"] for r in downstream_result.results if r["tier"] == 1}
    tier2_names = {r["name"] for r in downstream_result.results if r["tier"] == 2}
    for name in ["hyundai motor", "kia", "ford motor", "general motors", "tesla"]:
        resolved = store.find_company_node(name)
        if resolved in tier1_names:
            print(f"    '{name}' -> tier 1")
        elif resolved in tier2_names:
            print(f"    '{name}' -> tier 2")
        else:
            print(f"    '{name}' -> DROPPED or unreachable")

    print("\n=== Substitutes mode: POSCO, component='steel' ===")
    sub_result = store.traverse_supply_graph("posco", mode="substitutes", component="steel", min_distance_km=500)
    print("  status:", sub_result.status)
    print(f"  Found {sub_result.n_results} matches")
    for r in sub_result.results[:10]:
        print(f"    {r['name']:30s} | {r['country']:15s} | {r['distance_km']:>6} km | conf={r['confidence']}")

    print("\n  --- as_observation() preview (truncated in this print, not in the real return) ---")
    print(sub_result.as_observation()[:400], "...")

    print("\n=== filter_to_names(): check specific event-affected companies ===")
    event_affected = ["ford motor", "general motors", "tesla", "korinox"]  # korinox = known unreachable
    kept, missing = filter_to_names(downstream_result, event_affected)

    print("  Found in traversal:")
    for r in kept:
        print(f"    {r['name']:20s} | tier {r['tier']} | confidence={r['confidence']}")

    print("  Not found:", missing)
    dropped_names = {d["name"] for d in downstream_result.dropped_unenriched}
    for name in missing:
        resolved = store.find_company_node(name)
        if resolved in dropped_names:
            print(f"    '{name}' -> reachable but NOT enriched (dropped)")
        else:
            print(f"    '{name}' -> genuinely absent from the graph")

    print("\n=== Status-path checks (not_resolved / invalid_mode / no_results) ===")
    bad_company = store.traverse_supply_graph("totally fake company xyz", mode="downstream")
    print("  unresolvable company -> status:", bad_company.status)
    print("   ", bad_company.as_observation())

    bad_mode = store.traverse_supply_graph("posco", mode="not_a_real_mode")
    print("  invalid mode -> status:", bad_mode.status)
    print("   ", bad_mode.as_observation())