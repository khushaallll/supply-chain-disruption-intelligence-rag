"""
Tests for get_supplier_info(), in the same "named acceptance check" style
already used elsewhere in this project (Denso=137, Ford=97, etc.) --
each test states the exact expected value and prints PASS/FAIL, rather
than being a silent pytest suite.

Two groups:
  PART A -- real-data smoke tests, against the actual uploaded
            manifest_fulltext_sentence.csv + chunks_600_80_fulltext_dummy.jsonl
  PART B -- synthetic fixture tests, for edge cases the real (small) dummy
            file doesn't happen to contain, e.g. a genuine tie-break

Run: python test_supplier_info.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

from supplier_info_tool import SupplierInfoStore

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ===========================================================================
# PART A -- real-data smoke tests
# ===========================================================================

MANIFEST_PATH = "data/corpus/manifest_fulltext_sentence.csv"
CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"

print("=" * 70)
print("PART A -- real-data smoke tests")
print("=" * 70)

store = SupplierInfoStore(MANIFEST_PATH, CHUNKS_PATH)

# A1 -- normal case: panasonic has a 2004-09-13 20-F, chunks ARE present in
# the dummy file for this exact doc_id. Event dated a year later.
r = store.get_supplier_info(
    company_name="panasonic",
    event_date="2005-08-29",
    event_id="8_exxon_mobil_2005",
)
check("A1 status == found", r.status == "found", r.status)
check("A1 doc_id == 2e9b044a0764f2b8", r.doc_id == "2e9b044a0764f2b8", r.doc_id)
check("A1 published_date == 2004-09-13", r.published_date == "2004-09-13", r.published_date)
check("A1 staleness_days == 350", r.staleness_days == 350, r.staleness_days)
check("A1 tag_mismatch is False (chunk IS tagged for this event)", r.tag_mismatch is False, r.tag_mismatch)
check("A1 text is non-empty", len(r.text) > 0)
check("A1 n_chunks_returned == 8 (capped)", r.n_chunks_returned == 8, r.n_chunks_returned)
# Self-consistency check rather than a magic number: whatever chunks file is
# loaded (small dummy or full real corpus), the tool's reported total must
# match what's actually sitting in the index for this doc_id.
actual_chunks_for_doc = len(store.chunk_index.get(r.doc_id, []))
check("A1 n_chunks_total_in_doc matches the loaded chunk index for this doc",
      r.n_chunks_total_in_doc == actual_chunks_for_doc,
      f"reported {r.n_chunks_total_in_doc}, index has {actual_chunks_for_doc}")
# Sanity cross-check against the manifest's own claimed count, when the full
# corpus is loaded these should agree exactly (both point to 142 for Panasonic
# 2004-09-13). On the small dummy file they will legitimately disagree --
# that disagreement is expected there, not a failure, so it isn't asserted.
manifest_claimed = next(int(row["n_chunks"]) for row in store.manifest if row["doc_id"] == r.doc_id)
print(f"        (info: manifest claims {manifest_claimed} chunks for this doc; "
      f"loaded index actually has {actual_chunks_for_doc})")

# A2 -- tag_mismatch diagnostic: same lookup, but claim it's for an event
# that was never tagged on this chunk. Should NOT block the result.
r2 = store.get_supplier_info(
    company_name="panasonic",
    event_date="2005-08-29",
    event_id="99_fake_event_not_in_corpus",
)
check("A2 status still == found (mismatch does not gate)", r2.status == "found", r2.status)
check("A2 tag_mismatch is True", r2.tag_mismatch is True, r2.tag_mismatch)

# A3 -- no event_id supplied at all -> tag_mismatch should be None, not False/True
r3 = store.get_supplier_info(company_name="panasonic", event_date="2005-08-29")
check("A3 tag_mismatch is None when event_id omitted", r3.tag_mismatch is None, r3.tag_mismatch)

# A4 -- leakage guard: event dated BEFORE panasonic's earliest filing
r4 = store.get_supplier_info(company_name="panasonic", event_date="2003-01-01")
check("A4 status == no_predating_document", r4.status == "no_predating_document", r4.status)
check("A4 text is empty", r4.text == "", r4.text)

# A5 -- company not in manifest at all
r5 = store.get_supplier_info(company_name="totally_made_up_company_xyz", event_date="2020-01-01")
check("A5 status == no_manifest_entry", r5.status == "no_manifest_entry", r5.status)

# A6 -- company IS in manifest with a real, newer filing (panasonic has a
# 2011-06-30 20-F per the manifest). Whether this correctly resolves to
# 'found' or 'chunks_missing' depends entirely on whether THAT SPECIFIC
# doc_id's chunks are present in whichever chunks file was loaded --
# so the expectation is derived from the loaded index, not hard-coded.
# On the small dummy file (only has the 2004 doc's chunks) -> chunks_missing.
# On the real full corpus (should have every doc's chunks) -> found.
# If this ever reports chunks_missing on the REAL corpus, that is a genuine
# join failure worth investigating -- e.g. the 2 known truncated-fetch-ID
# events -- not a test artifact.
r6 = store.get_supplier_info(company_name="panasonic", event_date="2012-01-01")
expected_doc_id = "8f3d6e90aac845c7"  # panasonic 2011-06-30, per the manifest
doc_actually_in_index = expected_doc_id in store.chunk_index
expected_status = "found" if doc_actually_in_index else "chunks_missing"
check(f"A6 status == {expected_status} (based on what's actually loaded)",
      r6.status == expected_status, r6.status)
check("A6 doc_id is the 2011-06-30 filing (most recent before 2012-01-01)",
      r6.published_date == "2011-06-30", r6.published_date)
check("A6 candidate_documents_considered == 6 (panasonic has 6 filings total)",
      r6.candidate_documents_considered == 6, r6.candidate_documents_considered)
if not doc_actually_in_index:
    print("        (info: running against a partial/dummy chunks file -- "
          "chunks_missing here is expected, not a real bug)")

# A7 -- case-insensitivity (not fuzzy -- just normalization)
r7 = store.get_supplier_info(company_name="PANASONIC", event_date="2005-08-29")
check("A7 case-insensitive match still finds the company", r7.status == "found", r7.status)

# A8 -- observation string doesn't crash for each status type
for result in (r, r4, r5, r6):
    obs = result.as_observation()
    check(f"A8 as_observation() non-empty for status={result.status}", len(obs) > 0)


# ===========================================================================
# PART B -- synthetic fixtures, for cases the small real dummy file can't
# exercise (a genuine same-date tie between two documents for one company)
# ===========================================================================

print()
print("=" * 70)
print("PART B -- synthetic fixture tests (tie-break, multi-doc)")
print("=" * 70)

tmpdir = Path(tempfile.mkdtemp())
manifest_fixture = tmpdir / "manifest.csv"
chunks_fixture = tmpdir / "chunks.jsonl"

# Two documents for "acme corp", same published_date, different n_chunks.
# Tie-break rule: n_chunks desc -> doc_b (5 chunks) should win over doc_a (2 chunks).
manifest_rows = [
    {"doc_id": "doc_a", "company": "acme corp", "cik": "1", "source_type": "annual_report",
     "form": "10-K", "published_date": "2015-01-01", "accession": "acc-a",
     "sections": "full_document", "chars": "1000", "n_chunks": "2",
     "status": "rechunked", "url": "https://example.com/a"},
    {"doc_id": "doc_b", "company": "acme corp", "cik": "1", "source_type": "annual_report",
     "form": "10-K", "published_date": "2015-01-01", "accession": "acc-b",
     "sections": "full_document", "chars": "3000", "n_chunks": "5",
     "status": "rechunked", "url": "https://example.com/b"},
]
with open(manifest_fixture, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    w.writeheader()
    w.writerows(manifest_rows)

chunk_rows = []
for i in range(2):
    chunk_rows.append({
        "chunk_id": f"doc_a:full_document:s{i:04d}", "doc_id": "doc_a",
        "text": f"doc_a chunk {i}", "company": "acme corp", "cik": "1",
        "source_type": "annual_report", "form": "10-K", "published_date": "2015-01-01",
        "section": "full_document", "relevant_events": ["99_synthetic_event"],
        "url": "https://example.com/a", "n_sentences": 3, "hard_split": False,
        "published_int": 20150101,
    })
for i in range(5):
    chunk_rows.append({
        "chunk_id": f"doc_b:full_document:s{i:04d}", "doc_id": "doc_b",
        "text": f"doc_b chunk {i}", "company": "acme corp", "cik": "1",
        "source_type": "annual_report", "form": "10-K", "published_date": "2015-01-01",
        "section": "full_document", "relevant_events": [],
        "url": "https://example.com/b", "n_sentences": 3,
        "hard_split": (i == 4),  # last chunk of doc_b is a hard split
        "published_int": 20150101,
    })
with open(chunks_fixture, "w") as f:
    for c in chunk_rows:
        f.write(json.dumps(c) + "\n")

synth_store = SupplierInfoStore(manifest_fixture, chunks_fixture)

rb = synth_store.get_supplier_info(company_name="acme corp", event_date="2015-06-01", max_chunks=10)
check("B1 tie-break picks doc_b (higher n_chunks)", rb.doc_id == "doc_b", rb.doc_id)
check("B1 n_chunks_returned == 5", rb.n_chunks_returned == 5, rb.n_chunks_returned)
check("B1 any_hard_split == True (last chunk of doc_b)", rb.any_hard_split is True, rb.any_hard_split)

# B2 -- max_chunks cap actually caps
rb2 = synth_store.get_supplier_info(company_name="acme corp", event_date="2015-06-01", max_chunks=3)
check("B2 n_chunks_returned == 3 when capped", rb2.n_chunks_returned == 3, rb2.n_chunks_returned)
check("B2 n_chunks_total_in_doc still reports 5 (true total, not the cap)",
      rb2.n_chunks_total_in_doc == 5, rb2.n_chunks_total_in_doc)

# B3 -- staleness_days computed correctly (2015-06-01 minus 2015-01-01 = 151 days)
check("B3 staleness_days == 151", rb.staleness_days == 151, rb.staleness_days)


# ===========================================================================
print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)