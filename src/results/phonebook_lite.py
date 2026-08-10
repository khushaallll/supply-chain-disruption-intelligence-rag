"""
phonebook_lite.py -- a minimal, compatible stand-in for your real
phonebook.py, built directly from the actual company_phonebook.csv you
uploaded (confirmed structure: 111 rows, columns including
ground_truth_name, graph_name, corpus_name -- 74/111 rows have a
non-empty, pre-reviewed graph_name).

*** I do NOT have your real phonebook.py, only the CSV it produces. ***
Your existing tools (graph_tool.py, supplier_info_tool.py,
search_corpus_tool.py) all accept an optional `phonebook` object with
`.graph_name(name)` / `.corpus_name(name)` methods -- "a pre-reviewed
answer for a known set of 111 companies... a phonebook miss falls
straight through" per supplier_info_tool.py's own docstring. This class
implements that exact contract (exact match, case/whitespace-normalised,
against the ground_truth_name column; a miss returns None, not a guess)
so it's usable anywhere your real code expects a phonebook object. If
your real phonebook.py does anything more (fuzzy fallback, alias lists),
swap that in instead -- this is deliberately the simplest thing that
satisfies the contract, not a reimplementation of logic I can't see.

Used by name_resolution_audit.py to answer: "would resolution have
worked if the phonebook had been wired in?" None of your other scripts
(baseline_b.py, evaluate.py) currently pass a phonebook to GraphStore --
that's not a bug, just a choice nobody has revisited yet. See this
audit's own output for whether it's worth revisiting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

PHONEBOOK_PATH = "data/company_phonebook.csv"


class SimplePhonebook:
    def __init__(self, csv_path: str | Path = PHONEBOOK_PATH):
        self._graph_lookup: dict[str, str] = {}
        self._corpus_lookup: dict[str, str] = {}
        self.n_rows = 0
        self.n_graph_entries = 0
        self.n_corpus_entries = 0

        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Phonebook not found at {csv_path} -- name_resolution_audit.py "
                f"needs this to check whether phonebook-assisted lookup would "
                f"have helped. Fix the path or skip the phonebook comparison."
            )

        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.n_rows += 1
                key = row["ground_truth_name"].strip().lower()
                graph_name = row.get("graph_name", "").strip()
                corpus_name = row.get("corpus_name", "").strip()
                if graph_name:
                    self._graph_lookup[key] = graph_name
                    self.n_graph_entries += 1
                if corpus_name:
                    self._corpus_lookup[key] = corpus_name
                    self.n_corpus_entries += 1

    def graph_name(self, name: str) -> Optional[str]:
        return self._graph_lookup.get(name.strip().lower())

    def corpus_name(self, name: str) -> Optional[str]:
        return self._corpus_lookup.get(name.strip().lower())


if __name__ == "__main__":
    pb = SimplePhonebook()
    print(f"Loaded {pb.n_rows} phonebook rows "
          f"({pb.n_graph_entries} with a graph_name, "
          f"{pb.n_corpus_entries} with a corpus_name).")
    for name in ["Alcoa Corporation", "totally made up company xyz"]:
        print(f"  graph_name('{name}') -> {pb.graph_name(name)}")
