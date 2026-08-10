"""
get_supplier_info(company_name, event_date, ...) — Layer 3 agent tool.

Answers: "what does this specific company's own filings say, as of this
event date?" It is a metadata lookup over manifest_fulltext.csv +
chunks_600_80_fulltext.jsonl. No embeddings, no ChromaDB.

Design decisions (see conversation for full rationale):
  - Exact company match (case-insensitive), no fuzzy matching inside this tool.
    Company-identity resolution already happened once, deliberately, on Day 1
    (rapidfuzz + manual review). Re-fuzzy-matching here would create a second,
    undocumented identity decision at query time. An optional shared
    `Phonebook` (phonebook.py) can be passed in to translate a near-miss
    spelling to the correct one BEFORE the exact match runs -- this is not
    fuzzy matching at query time, it's a pre-reviewed answer for a known set
    of 111 companies (see phonebook.py / build_phonebook.py); a phonebook
    miss falls straight through to the same exact-match-only behaviour as
    before.
  - Leakage guard enforced inside the tool: published_date <= event_date.
    Mirrors the Day 5 corpus rule; not left to the agent to remember.
  - Most-recent-qualifying-document selection, with an explicit, single
    tie-break rule (n_chunks desc, then doc_id asc) so the result is
    reproducible rather than "roughly the newest."
  - relevant_events (pre-computed at Day 5 corpus-build time, against a fixed
    30-event set) is used as a DIAGNOSTIC cross-check only, never as a gate.
    Gating on it would (a) break on any event outside the original 30, and
    (b) silently inherit any Day 5 tagging bugs -- e.g. the known 2 truncated
    48-char fetch IDs -- as false "no evidence" results.
  - Three distinct "nothing found" outcomes, not one, because they mean
    different things to the agent's stopping-condition logic:
        no_manifest_entry      -- company not in the manifest at all
        no_predating_document  -- company known, but nothing predates event
        chunks_missing         -- manifest says a doc exists, but the chunks
                                   file has no matching doc_id (a join
                                   failure -- e.g. the truncated-ID bug)
        found                  -- normal case
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class SupplierInfoResult:
    status: str                      # found | no_manifest_entry | no_predating_document | chunks_missing
    company: str
    event_id: Optional[str]
    event_date: str

    doc_id: Optional[str] = None
    published_date: Optional[str] = None
    form: Optional[str] = None
    url: Optional[str] = None
    accession: Optional[str] = None
    staleness_days: Optional[int] = None

    text: str = ""
    n_chunks_returned: int = 0
    n_chunks_total_in_doc: int = 0
    any_hard_split: bool = False

    tag_mismatch: Optional[bool] = None   # None if event_id wasn't supplied
    candidate_documents_considered: int = 0

    def as_observation(self) -> str:
        """Plain-text form to hand back to the LangGraph agent as a tool observation."""
        if self.status == "no_manifest_entry":
            return f"No filings on record for '{self.company}'."
        if self.status == "no_predating_document":
            return (f"'{self.company}' has filings on record, but none dated "
                     f"on or before {self.event_date}.")
        if self.status == "chunks_missing":
            return (f"Manifest lists a document for '{self.company}' "
                     f"(doc_id={self.doc_id}, {self.published_date}), but its "
                     f"chunk text could not be located -- likely a join failure.")
        header = (f"[{self.company} | {self.form} filed {self.published_date} "
                  f"({self.staleness_days} days before event) | doc_id={self.doc_id}]")
        return header + "\n\n" + self.text


# ---------------------------------------------------------------------------
# Loading (do this once, reuse across calls -- these files are read many
# times over a 30-event evaluation run, so don't re-parse them every call)
# ---------------------------------------------------------------------------

def _to_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def load_manifest(manifest_path: str | Path) -> list[dict]:
    """One row per document. Loaded once, kept in memory."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_chunk_index(chunks_path: str | Path) -> dict[str, list[dict]]:
    """
    doc_id -> list of chunk dicts, sorted by their chunk_id sequence number
    (chunk_id looks like '{doc_id}:{section}:s0000', so the trailing sNNNN
    gives a stable, cheap sort key without needing a separate index field).
    """
    index: dict[str, list[dict]] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            index.setdefault(chunk["doc_id"], []).append(chunk)

    def _seq(chunk: dict) -> int:
        tail = chunk["chunk_id"].rsplit(":s", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    for doc_id in index:
        index[doc_id].sort(key=_seq)

    return index


# ---------------------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------------------

class SupplierInfoStore:
    """
    Wraps a loaded manifest + chunk index. Build one of these once per agent
    run (or once per process), then call .get_supplier_info() many times --
    this avoids re-reading multi-hundred-MB files on every agent hop.
    """

    def __init__(self, manifest_path: str | Path, chunks_path: str | Path, phonebook=None):
        self.manifest = load_manifest(manifest_path)
        self.chunk_index = build_chunk_index(chunks_path)
        self.phonebook = phonebook  # optional Phonebook -- see phonebook.py

        if not self.manifest or not self.chunk_index:
            raise RuntimeError(
                f"Loaded manifest has {len(self.manifest)} row(s) and chunk "
                f"index has {len(self.chunk_index)} document(s) -- refusing "
                f"to proceed with an empty load. A silently-empty manifest "
                f"or chunk index would make every future get_supplier_info() "
                f"call report 'no_manifest_entry' or 'chunks_missing' for "
                f"every company, which looks like normal tool behaviour "
                f"rather than a loading failure. Same 'absence must be "
                f"fatal and loud' principle as CorpusSearchStore's "
                f"Chroma-count check and GraphStore's zero-node check."
            )

    def get_supplier_info(
        self,
        company_name: str,
        event_date: str,
        event_id: Optional[str] = None,
        max_chunks: int = 8,
    ) -> SupplierInfoResult:
        """
        Look up what a SPECIFIC, already-named company says about itself in
        its own filings. Use this AFTER you already have a company name
        (e.g. from traverse_supply_graph or search_corpus) and want to
        ground a claim about that one company in primary-source text --
        this tool cannot discover a company you haven't already named, and
        does not search by topic (use search_corpus for that instead).

        company_name : exact match against the manifest 'company' field
                        (case-insensitive; NOT fuzzy -- a near-miss spelling
                        is reported as no_manifest_entry, not guessed at)
        event_date   : 'YYYY-MM-DD' -- the disruption date being investigated;
                        only documents dated on or before this are eligible
        event_id     : optional -- current event's ID, used only to compute
                        the tag_mismatch diagnostic against relevant_events
        max_chunks   : cap on how many chunks of the chosen document to return
        """
        # Step 0: phonebook lookup, if one was provided -- if this exact
        # name is one of the 111 known ones, use its pre-reviewed corpus
        # spelling instead of the raw input. This tool is exact-match-only
        # by design (module docstring), so this is the one place a
        # near-miss spelling gets a chance to still resolve correctly --
        # without adding fuzzy matching to the tool itself. A phonebook
        # miss changes nothing: falls through to the exact match below,
        # exactly as before.
        lookup_name = company_name
        if self.phonebook is not None:
            ph_match = self.phonebook.corpus_name(company_name)
            if ph_match is not None:
                lookup_name = ph_match

        company_key = lookup_name.strip().lower()
        event_dt = _to_date(event_date)

        # --- Step 1: does this company exist in the manifest at all? -------
        candidates = [row for row in self.manifest
                      if row["company"].strip().lower() == company_key]

        if not candidates:
            return SupplierInfoResult(
                status="no_manifest_entry",
                company=company_name, event_id=event_id, event_date=event_date,
            )

        # --- Step 2: leakage guard -- keep only docs dated on/before event,
        # AND only docs that actually have real text behind them.
        #
        # REVISION (post-30-event real-batch review): added the
        # status != "FAILED" condition below. The manifest can contain
        # rows for documents whose original SEC fetch failed at
        # corpus-build time -- status == 'FAILED', n_chunks == '0',
        # chars == '0' -- while still looking like an ordinary,
        # selectable row otherwise. Confirmed on the real manifest: 8 of
        # 605 rows are FAILED this way, across 4 companies (ENI x3, Korea
        # Electric Power x3, Shell x1, Rio Tinto x1).
        #
        # Before this fix, Step 3 below (pick the single most recent
        # qualifying document) could still select a FAILED row purely
        # because of its date, then this call would report
        # "chunks_missing" -- which reads as a data-join bug, when the
        # real story is simpler: the source document itself never
        # successfully downloaded. Confirmed concretely on two real
        # events: Korea Electric Power's and ENI's most-recent
        # pre-event filing were BOTH their FAILED 2022 row, while a
        # perfectly good, fully-chunked filing from the year before sat
        # right behind it, unconsidered, for both companies.
        #
        # Filtering FAILED rows out HERE, at the same step as the
        # existing leakage-guard date filter, means Step 3's "most
        # recent" selection naturally falls through to the next real,
        # working filing instead -- no other logic needs to change. If
        # every predating filing for a company happens to be FAILED,
        # this now correctly reports "no_predating_document" (an honest
        # description) rather than "chunks_missing" (which implies a
        # join bug that isn't actually what happened in that case).
        predating = [row for row in candidates
                     if _to_date(row["published_date"]) <= event_dt
                     and row.get("status") != "FAILED"]

        if not predating:
            return SupplierInfoResult(
                status="no_predating_document",
                company=company_name, event_id=event_id, event_date=event_date,
                candidate_documents_considered=len(candidates),
            )

        # --- Step 3: pick most recent; tie-break n_chunks desc, doc_id asc -
        predating.sort(
            key=lambda r: (_to_date(r["published_date"]), int(r["n_chunks"]), r["doc_id"]),
            reverse=True,
        )
        chosen = predating[0]
        doc_id = chosen["doc_id"]

        # --- Step 4: join against the chunk file ----------------------------
        chunks = self.chunk_index.get(doc_id)
        if not chunks:
            return SupplierInfoResult(
                status="chunks_missing",
                company=company_name, event_id=event_id, event_date=event_date,
                doc_id=doc_id, published_date=chosen["published_date"],
                form=chosen["form"], url=chosen["url"], accession=chosen["accession"],
                candidate_documents_considered=len(candidates),
            )

        selected = chunks[:max_chunks]
        text = "\n\n".join(c["text"] for c in selected)
        any_hard_split = any(c.get("hard_split") for c in selected)

        # --- Step 5: relevant_events is a diagnostic, not a gate ------------
        tag_mismatch = None
        if event_id is not None:
            tagged_events = set()
            for c in selected:
                tagged_events.update(c.get("relevant_events") or [])
            tag_mismatch = event_id not in tagged_events

        staleness = (event_dt - _to_date(chosen["published_date"])).days

        return SupplierInfoResult(
            status="found",
            company=company_name, event_id=event_id, event_date=event_date,
            doc_id=doc_id, published_date=chosen["published_date"],
            form=chosen["form"], url=chosen["url"], accession=chosen["accession"],
            staleness_days=staleness,
            text=text,
            n_chunks_returned=len(selected),
            n_chunks_total_in_doc=len(chunks),
            any_hard_split=any_hard_split,
            tag_mismatch=tag_mismatch,
            candidate_documents_considered=len(candidates),
        )


def get_supplier_info(
    company_name: str,
    event_date: str,
    store: SupplierInfoStore,
    event_id: Optional[str] = None,
    max_chunks: int = 8,
) -> SupplierInfoResult:
    """Free-function wrapper -- this is the shape LangGraph will actually bind as a tool."""
    return store.get_supplier_info(company_name, event_date, event_id=event_id, max_chunks=max_chunks)