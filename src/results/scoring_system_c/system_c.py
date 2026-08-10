"""
system_c.py -- the ONLY file in this evaluation package that knows
anything about System C's trace JSON shape. Baselines A and B will each
get their own small sibling module (baseline_a.py, baseline_b.py) instead
of this file growing branches -- scoring.py and name_resolution.py stay
untouched either way.
"""

from __future__ import annotations

EVIDENCE_TOOLS = {"get_supplier_info", "search_corpus"}  # from stopping_condition.py


def extract_system_c_predictions(trace: dict) -> list:
    """Returns the lowercased predicted-company-name list for one
    System-C trace: coverage rows with at least one EVIDENCE_TOOLS
    source. See scoring.py's module docstring for the full reasoning on
    why this is the bar, rather than every row in `coverage`."""
    coverage = trace.get("coverage", {}) or {}
    predicted = []
    for key, row in coverage.items():
        sources = row.get("evidence_sources", []) or []
        if any(s in EVIDENCE_TOOLS for s in sources):
            predicted.append(key)
    return predicted


def is_seed_unresolved(trace: dict) -> bool:
    """True iff the trace's coverage dict is completely empty -- the
    brief's requirement-4 category, distinct from an ordinary recall
    miss. Read directly off the trace file itself (not off the batch
    summary's n_companies_in_coverage field) so this script has no
    dependency on run_all_events_summary_*.json existing or being in
    sync with the individual trace files."""
    return len(trace.get("coverage", {}) or {}) == 0
