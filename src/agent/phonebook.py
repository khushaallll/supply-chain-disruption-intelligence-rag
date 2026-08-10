"""
phonebook.py -- shared company-name lookup for all three Layer 3 tools.

Loads company_phonebook.csv (produced once by build_phonebook.py -- see
that file's docstring for how the CSV itself is built and reviewed) and
answers one question fast, with no fuzzy matching involved: "given any
spelling of this company that appears ANYWHERE in the phonebook, what's
the correct spelling for the graph, and what's the correct spelling for
the corpus?"

Indexed by every known spelling of each company (ground-truth name, graph
name, AND corpus name, all lowercased) -> the same row. This means a tool
can look a name up using whichever spelling it happens to have on hand --
the name a previous tool call just returned, or the raw ground-truth
name -- and still land on the right row, not just the one exact key it
was built from.

This only covers the 111 company names across the 30 ground-truth events
(see build_phonebook.py). A lookup miss is expected and normal for any
name outside that set -- callers should fall back to their own existing
matching logic in that case, not treat a miss as an error.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


class Phonebook:
    def __init__(self, csv_path: str | Path):
        self._by_any_name: dict[str, dict] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            for key_field in ("ground_truth_name", "graph_name", "corpus_name"):
                value = (row.get(key_field) or "").strip()
                if value:
                    self._by_any_name[value.lower()] = row

        self.n_rows = len(rows)

    def _lookup_row(self, name: str) -> Optional[dict]:
        return self._by_any_name.get(name.strip().lower())

    def graph_name(self, name: str) -> Optional[str]:
        """The correct graph spelling for this company, or None if this
        name isn't in the phonebook at all (fall back to the graph tool's
        own matching in that case)."""
        row = self._lookup_row(name)
        if row is None:
            return None
        value = (row.get("graph_name") or "").strip()
        return value or None  # a phonebook row can legitimately have no
                               # graph_name (a company confirmed absent
                               # from the graph) -- return None either way,
                               # the caller doesn't need to tell these two
                               # "no match" cases apart to do the right thing

    def corpus_name(self, name: str) -> Optional[str]:
        """The correct corpus spelling for this company, or None if this
        name isn't in the phonebook, or is but has no corpus document."""
        row = self._lookup_row(name)
        if row is None:
            return None
        value = (row.get("corpus_name") or "").strip()
        return value or None
