"""
get_supplier_info(company_name, event_date, ...) — Layer 3 agent tool.

Answers: "what does this specific company's own filings say, as of this
event date?" It is a metadata lookup over manifest_fulltext.csv +
chunks_600_80_fulltext.jsonl. No embeddings, no ChromaDB.

Design decisions (see conversation for full rationale):
  - Exact company match (case-insensitive), no fuzzy matching inside this tool.
    Company-identity resolution already happened once, deliberately, on Day 1
    (rapidfuzz + manual review). Re-fuzzy-matching here would create a second,
    undocumented identity decision at query time.
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

    def __init__(self, manifest_path: str | Path, chunks_path: str | Path):
        self.manifest = load_manifest(manifest_path)
        self.chunk_index = build_chunk_index(chunks_path)

    def get_supplier_info(
        self,
        company_name: str,
        event_date: str,
        event_id: Optional[str] = None,
        max_chunks: int = 8,
    ) -> SupplierInfoResult:
        """
        company_name : exact match against manifest 'company' field
                        (case-insensitive; NOT fuzzy -- see module docstring)
        event_date   : 'YYYY-MM-DD' -- the disruption date being investigated
        event_id     : optional -- current event's ID, used only to compute
                        the tag_mismatch diagnostic against relevant_events
        max_chunks   : cap on how many chunks of the chosen document to return
        """
        company_key = company_name.strip().lower()
        event_dt = _to_date(event_date)

        # --- Step 1: does this company exist in the manifest at all? -------
        candidates = [row for row in self.manifest
                      if row["company"].strip().lower() == company_key]

        if not candidates:
            return SupplierInfoResult(
                status="no_manifest_entry",
                company=company_name, event_id=event_id, event_date=event_date,
            )

        # --- Step 2: leakage guard -- keep only docs dated on/before event -
        predating = [row for row in candidates
                     if _to_date(row["published_date"]) <= event_dt]

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