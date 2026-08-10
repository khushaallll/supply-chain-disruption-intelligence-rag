"""
baseline_a.py -- System A: naive keyword co-occurrence baseline.

Per the implementation plan: "Baseline A is the simplest possible approach
-- a keyword search that flags any company whose name or location appears
in the same GDELT article as the disruption event." Since the full GDELT
pipeline was descoped (implementation_plan.md's scope decisions -- "~30-40
events... not full GDELT pipeline"), the natural stand-in is the one
synthetic TRIGGER document per event that Day 5's corpus build created
specifically to play this role: "one document per event: the event
description, location, category and directly-affected company," dated on
the event date (day4-5_implementation_log.md, Section 2.2). Scanning that
document for known company names is the same operation Baseline A is
supposed to perform on a real news article -- just against your project's
actual data source instead of a GDELT article that was never built.

*** ONE GENUINE UNKNOWN IN THIS FILE -- READ THIS BEFORE RUNNING ***
I do not have direct access to your corpus chunks file
(chunks_600_80_fulltext_sentence_tagged.jsonl) or a manifest that lists
the 30 trigger documents specifically -- the manifest_fulltext_sentence.csv
you uploaded contains only the 605 SEC annual-report documents (source_type
"annual_report" throughout, confirmed by direct inspection), not the
triggers. So `load_trigger_texts()` below has to GUESS which field
identifies a trigger chunk in the real chunks file. It tries three
heuristics in order and tells you on stdout which one worked, or if none
did. Run `python baseline_a.py` and read that output FIRST -- if it says
"0 trigger documents found," open the chunks file yourself and check what
actually marks a trigger doc (likely candidates: a `form` value like
"trigger" or "event_trigger", or a `source_type` field, or a `doc_id`
pattern), then fix the one function `load_trigger_texts()` -- nothing
else in this file depends on how that lookup works internally.

If you'd rather not chase that down right now, `run_baseline_a()` will
accept a plain dict you build by hand ({event_id: trigger_text}) instead
of calling the loader -- see the `trigger_texts` parameter.

DELIBERATELY NOT using ground_truth_affected[].impact text as a stand-in
for the trigger document, even though it was tempting and would have
worked without any file-path guessing at all: that text was written BY
the ground-truth researcher, describing exactly which companies were
affected. Scanning it for company names doesn't test whether Baseline A
can find companies from a news-like description -- it tests whether
Baseline A can find companies whose names are ALREADY IN the answer key,
which would make Baseline A trivially close to 100% recall for the wrong
reason. This is the same circularity risk your project's own memory
already flags for Cowork-sourced ground truth, and grading Baseline A
against its own answer key would be a worse version of exactly that
problem.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Deliberately NOT importing graph_tool at module level -- scan_text_for_
# companies(), load_trigger_texts(), and run_baseline_a() below need none
# of networkx/rapidfuzz/geopy, only something that HAS them (GraphStore)
# needs to hand them a company-name vocabulary. Same reason
# search_corpus_tool.py defers chromadb/torch/sentence_transformers to
# inside __init__ rather than the module top: it keeps this file's pure
# logic importable and testable (see test_evaluation_logic.py) without
# your real environment installed. GraphStore is imported only inside the
# __main__ block at the bottom, where it's actually needed.

CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"


# ---------------------------------------------------------------------------
# Pure function -- no file I/O, no graph load required, directly testable
# (see test_evaluation_logic.py). This is the actual "keyword baseline"
# logic; everything else in this file is plumbing to get text into it.
# ---------------------------------------------------------------------------

def scan_text_for_companies(text: str, known_company_names: list[str]) -> set[str]:
    """
    known_company_names: canonical graph-node names (already lowercase,
    already suffix-stripped, e.g. "toyota motor" not "Toyota Motor
    Corporation") -- pass store.G.nodes() when calling this for real.

    Matching strategy: whole-word/phrase substring match on the
    lowercased text, e.g. "toyota motor" matches inside "...Toyota Motor
    Corporation announced..." because the canonical node name is a
    prefix of the fuller legal name written in real text. Word-boundary
    regex, not a bare `in` check, specifically to avoid a short node name
    like "s oil" matching inside an unrelated word.

    A minimum-length guard on the SEARCH TERM (not the input text) skips
    node names under 4 characters entirely -- these are too likely to
    produce spurious matches for a naive keyword scan to be a fair
    representation of "simplest possible approach." This is a deliberate,
    documented choice, not an oversight: a real GDELT-keyword system would
    have the exact same short-name noise problem, so excluding them
    doesn't flatter Baseline A, it just avoids drowning the result in
    single acronym false positives that would make the baseline
    uninterpretable rather than genuinely weak.
    """
    text_lower = text.lower()
    found = set()
    for name in known_company_names:
        name = name.strip()
        if len(name) < 4:
            continue
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text_lower):
            found.add(name)
    return found


# ---------------------------------------------------------------------------
# Trigger-document loading -- the one genuinely uncertain part, see the
# module docstring. Three heuristics, tried in order.
# ---------------------------------------------------------------------------

def load_trigger_texts(chunks_path: str = CHUNKS_PATH) -> dict[str, str]:
    """
    Returns {event_id: concatenated_trigger_text}. Empty dict if the file
    isn't found or no heuristic matches anything -- caller must check for
    that, not assume this worked.
    """
    p = Path(chunks_path)
    if not p.exists():
        print(f"  [load_trigger_texts] NOT FOUND: {chunks_path}")
        return {}

    chunks = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    # Heuristic 1: an explicit field says so (form/source_type in a small
    # set of plausible trigger-marker values).
    trigger_markers = {"trigger", "event_trigger", "trigger_document", "gdelt"}
    h1 = [c for c in chunks
          if str(c.get("form", "")).lower() in trigger_markers
          or str(c.get("source_type", "")).lower() in trigger_markers]

    # Heuristic 2: doc_id or chunk_id contains the event_id verbatim, e.g.
    # a doc_id like "2_nippon_steel_2011_trigger".
    h2 = [c for c in chunks if "trigger" in str(c.get("doc_id", "")).lower()]

    # Heuristic 3: company field is null/empty AND relevant_events has
    # exactly one entry -- a document not tied to any single company but
    # tagged to one specific event is a plausible trigger-doc signature,
    # since every OTHER document in the corpus (per manifest_fulltext_
    # sentence.csv, confirmed 605/605 rows) is an annual report with a
    # real company name.
    h3 = [c for c in chunks
          if not c.get("company")
          and len(c.get("relevant_events") or []) == 1]

    for label, hits in [("form/source_type == trigger", h1),
                         ("doc_id contains 'trigger'", h2),
                         ("no company + single relevant_event tag", h3)]:
        if hits:
            print(f"  [load_trigger_texts] heuristic matched: {label} "
                  f"({len(hits)} chunks)")
            by_event: dict[str, list[str]] = {}
            for c in hits:
                for eid in (c.get("relevant_events") or [c.get("doc_id", "")]):
                    by_event.setdefault(eid, []).append(c.get("text", ""))
            return {eid: "\n".join(texts) for eid, texts in by_event.items()}

    print("  [load_trigger_texts] NONE of the three heuristics matched "
          "anything. Open the chunks file yourself, find what marks a "
          "trigger document, and either fix this function or build the "
          "{event_id: text} dict by hand and pass it to run_baseline_a() "
          "directly via the trigger_texts parameter.")
    return {}


# ---------------------------------------------------------------------------
# The baseline itself
# ---------------------------------------------------------------------------

@dataclass
class BaselineAResult:
    event_id: str
    status: str                  # found | no_trigger_text
    predicted_companies: set = field(default_factory=set)


def run_baseline_a(
    events,
    known_company_names: list[str],
    trigger_texts: Optional[dict[str, str]] = None,
) -> dict[str, BaselineAResult]:
    """
    events: list[Event] from event_loader.py
    known_company_names: store.G.nodes() from a loaded GraphStore -- using
        the same vocabulary as Baseline B so the two are directly
        comparable, and so a company Baseline A "finds" is guaranteed to
        be checkable against the same graph-node identity ground truth
        uses for its graph_node field.
    trigger_texts: pass this in yourself if load_trigger_texts()'s
        heuristics didn't work on your real file.
    """
    if trigger_texts is None:
        trigger_texts = load_trigger_texts()

    out = {}
    for ev in events:
        text = trigger_texts.get(ev.event_id)
        if not text:
            out[ev.event_id] = BaselineAResult(event_id=ev.event_id, status="no_trigger_text")
            continue
        predicted = scan_text_for_companies(text, known_company_names)
        out[ev.event_id] = BaselineAResult(
            event_id=ev.event_id, status="found", predicted_companies=predicted,
        )
    return out


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from event_loader import load_events
    from agent.tools.graph_tool import GraphStore, GRAPH_PATH  # only needed here

    events = load_events()
    trigger_texts = load_trigger_texts()

    print(f"\nTrigger text found for {len(trigger_texts)} / {len(events)} events.")
    if not trigger_texts:
        print("Stopping here -- fix load_trigger_texts() before continuing "
              "(see the module docstring).")
    else:
        print("\nLoading graph for the company-name vocabulary...")
        store = GraphStore(GRAPH_PATH)
        known_names = list(store.G.nodes())
        print(f"Vocabulary size: {len(known_names)} graph nodes\n")

        results = run_baseline_a(events, known_names, trigger_texts)

        print("=== Baseline A (keyword) -- per event ===")
        for ev in events:
            r = results[ev.event_id]
            print(f"  {ev.event_id:45s} status={r.status:16s} "
                  f"n_predicted={len(r.predicted_companies):3d}  "
                  f"{sorted(r.predicted_companies)[:5]}")
