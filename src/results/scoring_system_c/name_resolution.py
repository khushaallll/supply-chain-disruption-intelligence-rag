"""
name_resolution.py -- resolving ground-truth company names against a
system's predicted-company keys.

This is deliberately system-agnostic: it takes a set of "predicted name"
strings (already lowercased, however the calling system's extraction
produced them) and a ground-truth affected-company record, and returns a
match or None. System C's own coverage keys are already lowercased graph-
or model-echoed strings; Baseline A/B (when built) will produce their own
predicted-name strings in whatever form is natural to them, and can reuse
this module unchanged as long as they pass in lowercased strings.

Three resolution paths, tried in order, per the project's own real-data
findings (day8_9_full_implementation_log.md §2.6, §7.6):

1. ground_truth_affected[i]['graph_node'] -- the ground-truth curator's
   OWN pre-resolved graph-node spelling for this company, computed
   against the exact same graph_enriched.pkl the agent's traverse_supply_graph
   tool reads from. When present, this is the single most reliable
   signal available -- both sides of the comparison (agent coverage keys,
   ground truth) are keyed to the same underlying graph's own spelling.
   Tried first as an exact match, then (since the agent may have echoed
   the name back through get_supplier_info/search_corpus under a
   corporate-suffix variant the code-level canonicalization in
   agent_graph.py's _canonicalize_company_name didn't fully collapse --
   e.g. if the model's evidence-tool call used a spelling that isn't a
   clean word-prefix of the graph's own node name) via the same
   word-prefix variant check used inside the agent itself.

2. company_phonebook.csv -- the Day 7 shared phonebook, keyed on the
   ground truth's own 'company' string (curator-written, human prose,
   e.g. "BHP Billiton"). Consulted when graph_node is null/missing
   (company absent from the graph, or never resolved by the curator),
   since the phonebook independently records 111 reviewed spellings and
   may still supply a usable graph-side name distinct from what
   graph_node happened to capture.

3. Raw word-prefix variant match on the ground truth's own 'company'
   string, lowercased, against every predicted key -- last-resort, for
   residual mismatches neither graph_node nor the phonebook caught. Uses
   the SAME `_is_name_variant` logic as agent_graph.py, not a generic
   fuzzy-match library, because the project's own empirical check (log
   §2.6) found that a generic threshold (rapidfuzz token_sort_ratio)
   cannot separate real pairs like 'bhp'/'bhp group' (same company, low
   score) from 'china steel'/'china motor' (different companies, higher
   score) -- word-prefix matching is deliberately narrow and was the one
   approach that separated all eight real observed pairs cleanly.

Each candidate string is tried both as an exact key match and as a
variant match against the full predicted-key set, so "toyota motor" (an
exact predicted key) and "toyota motor corporation" (a variant) both
resolve to the same predicted company correctly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional


def _is_name_variant(a: str, b: str) -> bool:
    """True if one name's word sequence is a literal prefix of the
    other's (after lowercasing/splitting on whitespace) -- e.g. 'bhp' is
    a prefix of 'bhp group'; 'toyota motor' is a prefix of 'toyota motor
    corporation'. Copied verbatim from agent_graph.py's own logic (not
    imported, so this evaluation script has no dependency on langgraph
    or any of System C's runtime machinery -- see module docstring)."""
    a_words, b_words = a.split(), b.split()
    shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    return bool(shorter) and longer[: len(shorter)] == shorter


def load_phonebook(path: str) -> dict:
    """Returns {lowercased ground_truth_name: lowercased graph_name}.
    Rows whose graph_match_type is 'not_found' (empty graph_name) are
    kept out of the map entirely -- an empty string is not a usable
    resolution target and would otherwise match everything via
    _is_name_variant's `bool(shorter)` guard failing safe, but there's
    no reason to even attempt it."""
    phonebook: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_name = (row.get("ground_truth_name") or "").strip().lower()
            graph_name = (row.get("graph_name") or "").strip().lower()
            if gt_name and graph_name:
                phonebook[gt_name] = graph_name
    return phonebook


@dataclass
class MatchResult:
    matched_key: Optional[str]   # the predicted-name key that matched, or None
    method: str                  # audit trail: how the match was found


def resolve_gt_company(
    gt_entry: dict,
    predicted_keys: set,
    phonebook: dict,
) -> MatchResult:
    """gt_entry: one element of a ground-truth event's 'ground_truth_affected'
    list (has 'company', and usually 'graph_node'). predicted_keys: the
    system's predicted-company name strings, already lowercased. Returns
    the matched predicted key, or None with method='unmatched'."""

    candidates: list[tuple[str, str]] = []  # (candidate_string, source_label)

    graph_node = gt_entry.get("graph_node")
    if graph_node:
        candidates.append((graph_node.strip().lower(), "graph_node"))

    company_raw = (gt_entry.get("company") or "").strip().lower()
    phonebook_name = phonebook.get(company_raw)
    if phonebook_name:
        candidates.append((phonebook_name, "phonebook"))

    if company_raw:
        candidates.append((company_raw, "raw_name"))

    # Pass 1: exact matches, in candidate priority order (graph_node beats
    # phonebook beats raw name -- see module docstring for why).
    for cand, source in candidates:
        if cand in predicted_keys:
            return MatchResult(matched_key=cand, method=f"{source}_exact")

    # Pass 2: word-prefix variant matches, same priority order. Done as a
    # separate pass (not interleaved with pass 1) so an exact match from a
    # lower-priority candidate never loses to a variant match from a
    # higher-priority one -- exact beats variant, regardless of source.
    for cand, source in candidates:
        for key in predicted_keys:
            if _is_name_variant(cand, key):
                return MatchResult(matched_key=key, method=f"{source}_variant")

    return MatchResult(matched_key=None, method="unmatched")
