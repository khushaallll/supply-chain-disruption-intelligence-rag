"""
stopping_condition.py -- the two-check stopping condition
(Coverage, Evidence quality), plus the significance/recency defaults it
needs to actually evaluate those checks.

These are pure functions -- no LangGraph, no store, no I/O -- so they're
directly unit-testable against small hand-built result objects, the same
"pure function first" pattern already used elsewhere in this project
(search_corpus_tool.py's date_to_published_int/matches_filter, graph_tool.py's
strip_suffix). See test_agent_mocked.py, Part A.

--------------------------------------------------------------------------
OPEN DECISIONS THIS FILE MAKES EXPLICIT (the Day 8 brief asked for defaults
to be proposed and reasoned about, not silently picked):

SIGNIFICANT = tier-1 AND tier-2 downstream companies from the most recent
downstream traverse_supply_graph() call. Not substitutes-mode results, and
not tier-3+ (already out of scope at the graph-tool level itself --
traverse_supply_graph's own max_tier default is 2, so tier-3+ normally
never even appears in a result to begin with; SIGNIFICANT_TIER_MAX exists
mainly to stay correct if a caller ever passes a larger max_tier).

  REVISED from an earlier tier-1-only draft (see day8_implementation_log.md,
  "Revision: tier-2 significance"). That draft under-scoped this on a
  flawed cost argument (it assumed tier-2 discovery itself cost an extra
  hop -- it doesn't; one traverse_supply_graph(max_tier=2) call finds both
  tiers at once). The real constraint isn't discovery, it's EVIDENCE-
  GATHERING breadth: each required company still needs its own
  get_supplier_info/search_corpus call, one hop each, and a 4-hop budget
  genuinely cannot always afford individual evidence for every tier-1 +
  tier-2 company on a large event (POSCO's own reference trace names 5:
  Hyundai, Kia, Ford, GM, Tesla).

  Explicit decision (confirmed by the project owner, not assumed): both
  tiers stay REQUIRED anyway, evidence-per-company stays mandatory, and a
  4-hop run on a large event will often end in `hop_cap_reached` with real,
  named gaps still open -- and that is the CORRECT, intended outcome here,
  not a bug to design around. It is exactly the behaviour the Day 8 brief
  itself describes: "when coverage is incomplete the agent must flag the
  gap, not fabricate confidence." A capped run with 2 of 5 companies
  honestly flagged as unconfirmed is a more useful and more truthful
  result than a run that quietly narrows its own definition of "covered"
  just to report a clean finish.

  `dropped_unenriched` companies are never required, regardless of tier:
  there is no component/industry data to gather further evidence against,
  and graph_tool.py's own docstring already documents them as a distinct,
  known coverage gap, not a traversal failure.

RECENT_ENOUGH_DAYS = 730 (2 years).
  Reasoning: get_supplier_info() returns the most recent QUALIFYING
  (pre-event) SEC filing, and annual reports are filed roughly once a
  year, so 730 days is "at most one missed filing cycle" -- generous but
  not meaningless. A 10-K older than that is unlikely to still describe
  the supplier relationship accurately by the time of the event, so it's
  flagged, though not treated as equivalent to no evidence at all -- see
  below.

EVIDENCE QUALITY for one company is satisfied by EITHER:
  - a get_supplier_info() call with status == 'found' AND
    staleness_days <= RECENT_ENOUGH_DAYS, OR
  - a search_corpus() call with status == 'found' whose distinct_companies
    includes this company. (search_corpus evidence has no separate
    staleness check: every hit is already dated on/before the event by
    construction of the leakage guard, so "recent enough" doesn't apply
    the same way it does to a single filing.)
  A company CAN satisfy the coverage check with only a STALE
  get_supplier_info() hit (status == 'found', staleness_days > 730) --
  something real was found and read, so it counts as "covered," but the
  staleness is surfaced as its own, separate gap
  ("only stale evidence"), never folded into "no evidence found." Two
  different problems get two different messages -- the same principle the
  tools' own multi-status contracts already apply (e.g.
  get_supplier_info's three distinct "nothing found" statuses).
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

SIGNIFICANT_TIER_MAX = 2
RECENT_ENOUGH_DAYS = 730

# Which tools count as EVIDENCE for the coverage check, as opposed to
# merely NAMING a company as significant. traverse_supply_graph is what
# makes a company 'significant' in the first place (it populates
# `coverage` with a tier) -- but structural adjacency in the graph is not
# itself a grounded claim about impact, so a company that has ONLY ever
# been touched by traverse_supply_graph must still read as "no evidence
# gathered," not as covered. Only get_supplier_info and search_corpus
# actually retrieve primary-source text, so only they count here. This
# tool still gets recorded in evidence_sources (useful for the trace --
# "when was this company first named") -- it's just excluded from the
# evidence-quality check specifically, below.
EVIDENCE_TOOLS = {"get_supplier_info", "search_corpus"}


# --------------------------------------------------------------------------- #
# Reading a raw tool result into the two things coverage.py needs from it
# --------------------------------------------------------------------------- #

def extract_significant_companies(graph_result) -> list[dict]:
    """Given a GraphTraversalResult, return the subset of .results that
    counts as 'significant' (see module docstring): tier <= SIGNIFICANT_TIER_MAX,
    downstream mode, status == 'found'. Anything else -- substitutes mode,
    a not-found status, tier-3+ entries -- returns []."""
    if graph_result is None:
        return []
    if getattr(graph_result, "status", None) != "found":
        return []
    if getattr(graph_result, "mode", None) != "downstream":
        return []
    return [r for r in graph_result.results if r["tier"] <= SIGNIFICANT_TIER_MAX]


# --------------------------------------------------------------------------- #
# Updating the per-company coverage row -- called once per company touched
# by ANY tool result (traversal, supplier info, or corpus search)
# --------------------------------------------------------------------------- #

def record_evidence(
    coverage: dict,
    company_name: str,
    tool: str,
    *,
    tier: Optional[int] = None,
    hop: int = 0,
    staleness_days: Optional[int] = None,
    tag_mismatch: Optional[bool] = None,
) -> dict:
    """Update (in place) and return `coverage`'s row for one company.

    Never overwrites a known tier with None: a company can be first named
    by search_corpus (no tier info at all), then later confirmed at tier 1
    by traverse_supply_graph -- once a tier is known, it should stick, not
    get clobbered by a subsequent call that doesn't carry tier info.
    """
    key = company_name.strip().lower()
    row = coverage.get(key)
    if row is None:
        row = {
            "name": company_name,
            "tier": tier,
            "first_seen_hop": hop,
            "evidence_sources": [],
            "best_staleness_days": None,
            "tag_mismatch": tag_mismatch,
        }
        coverage[key] = row

    if tier is not None and row["tier"] is None:
        row["tier"] = tier
    if tool not in row["evidence_sources"]:
        row["evidence_sources"].append(tool)
    if staleness_days is not None:
        if row["best_staleness_days"] is None or staleness_days < row["best_staleness_days"]:
            row["best_staleness_days"] = staleness_days
    if tag_mismatch is not None:
        row["tag_mismatch"] = tag_mismatch

    return row


# --------------------------------------------------------------------------- #
# The stopping condition itself -- reads ONLY state['coverage'] / hop_count /
# max_hops, per the brief's instruction to read the fields already tracked,
# not re-derive coverage/quality signals from raw tool results each time.
# --------------------------------------------------------------------------- #

def check_stopping_condition(state) -> tuple[bool, str, list]:
    """
    Returns (should_stop, reason, gap_flags).

    reason in {'hop_cap_reached', 'coverage_and_quality_met', 'continue'}.

    gap_flags is always computed and returned, even when should_stop is
    False and reason == 'continue' -- so a caller (e.g. a live trace
    viewer) can show the current gap list mid-run, not just at the end.
    """
    hop_count = state["hop_count"]
    max_hops = state["max_hops"]
    coverage = state["coverage"]

    gaps: list[str] = []
    required_rows = [
        row for row in coverage.values()
        if row["tier"] is not None and row["tier"] <= SIGNIFICANT_TIER_MAX
    ]
    for row in required_rows:
        has_evidence = any(src in EVIDENCE_TOOLS for src in row["evidence_sources"])
        if not has_evidence:
            gaps.append(f"{row['name']}: no evidence gathered")
        elif (row["best_staleness_days"] is not None
              and row["best_staleness_days"] > RECENT_ENOUGH_DAYS):
            gaps.append(f"{row['name']}: only stale evidence "
                        f"({row['best_staleness_days']} days old)")

    # An EMPTY (or all-non-required) coverage dict must NOT read as "0 gaps
    # -> done": if no downstream traversal has even been run yet, there is
    # nothing to be confident about, so this must not stop on hop 0/1 just
    # because nothing exists yet to flag as missing.
    coverage_and_quality_met = len(gaps) == 0 and len(required_rows) > 0

    # Coverage is checked BEFORE the hop cap, deliberately: if coverage and
    # quality happen to become fully satisfied on the very hop that also
    # hits the cap, the honest reason is "the investigation finished," not
    # "the investigation was cut off" -- those are different facts about
    # the run and a trace reader (or Layer 5) should see the true one.
    if coverage_and_quality_met:
        return True, "coverage_and_quality_met", gaps
    if hop_count >= max_hops:
        return True, "hop_cap_reached", gaps
    return False, "continue", gaps
