"""
test_phonebook_integration.py

A SIMPLE end-to-end check: does wiring the phonebook into each of the
three tools actually change their behavior for the better -- not just
"does the code run", but "does a real name that used to fail now work,
and does a name the phonebook doesn't cover still behave exactly as
before"?

For each tool, the SAME lookup is run twice -- once with phonebook=None
(old behavior) and once with the real phonebook wired in -- so the
before/after difference is visible directly, not just asserted.

Uses small synthetic graph/manifest/chunks fixtures (this environment
doesn't have your real data files), but the phonebook itself is your real,
reviewed company_phonebook.csv -- so the NAMES being resolved are real,
even though the underlying graph/corpus data they point into is not.

Run: python test_phonebook_integration.py
"""

import csv
import json
import pickle
import sys
import tempfile
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from phonebook import Phonebook
from src.agent.tools.graph_tool import GraphStore
from src.agent.tools.supplier_info_tool import SupplierInfoStore

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


pb = Phonebook("data/company_phonebook.csv")
print(f"Phonebook loaded: {pb.n_rows} rows\n")

tmpdir = Path(tempfile.mkdtemp())


# ===========================================================================
# 1. graph_tool.py -- the CPC Corp Taiwan / CSBC Corp Taiwan near-miss
# ===========================================================================
print("=== 1. graph_tool.py ===")

G = nx.DiGraph()
G.add_node("cpc corp/taiwan", component="fuel", industry="oil refining",
           country="Taiwan", confidence=0.8, lat=22.6, lon=120.3)
G.add_node("csbc corp taiwan", component="ships", industry="shipbuilding",
           country="Taiwan", confidence=0.8, lat=22.6, lon=120.3)
graph_path = tmpdir / "graph.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)

store_no_pb = GraphStore(graph_path=graph_path, phonebook=None)
store_with_pb = GraphStore(graph_path=graph_path, phonebook=pb)

without = store_no_pb.find_company_node("cpc corp taiwan")
with_pb = store_with_pb.find_company_node("cpc corp taiwan")

check("WITHOUT phonebook: same 'cpc corp taiwan' still resolves to the WRONG company",
      without == "csbc corp taiwan", f"got '{without}'")
check("WITH phonebook: same input now resolves to the CORRECT company",
      with_pb == "cpc corp/taiwan", f"got '{with_pb}'")


# ===========================================================================
# 2. supplier_info.py -- "Toyota Motor Corporation" (ground truth's spelling)
#    vs "toyota motor" (the real corpus spelling)
# ===========================================================================
print("\n=== 2. supplier_info.py ===")

manifest_path = tmpdir / "manifest.csv"
with open(manifest_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["doc_id", "company", "cik", "source_type", "form",
                                        "published_date", "accession", "sections", "chars",
                                        "n_chunks", "status", "url"])
    w.writeheader()
    w.writerow({"doc_id": "doc1", "company": "toyota motor", "cik": "1", "source_type": "annual_report",
                "form": "20-F", "published_date": "2010-06-25", "accession": "acc1",
                "sections": "full_document", "chars": "1000", "n_chunks": "1",
                "status": "rechunked", "url": "https://example.com/toyota"})

chunks_path = tmpdir / "chunks.jsonl"
with open(chunks_path, "w") as f:
    f.write(json.dumps({"chunk_id": "doc1:full_document:s0000", "doc_id": "doc1",
                          "text": "Toyota purchases parts from many suppliers.",
                          "company": "toyota motor", "published_date": "2010-06-25",
                          "published_int": 20100625, "form": "20-F", "url": "https://example.com/toyota",
                          "relevant_events": [], "hard_split": False, "n_sentences": 1}) + "\n")

store_no_pb = SupplierInfoStore(manifest_path, chunks_path, phonebook=None)
store_with_pb = SupplierInfoStore(manifest_path, chunks_path, phonebook=pb)

r_without = store_no_pb.get_supplier_info("Toyota Motor Corporation", event_date="2011-03-11")
r_with = store_with_pb.get_supplier_info("Toyota Motor Corporation", event_date="2011-03-11")

check("WITHOUT phonebook: ground truth's exact spelling doesn't match the manifest",
      r_without.status == "no_manifest_entry", f"got status='{r_without.status}'")
check("WITH phonebook: same input now finds the real filing",
      r_with.status == "found", f"got status='{r_with.status}'")


# ===========================================================================
# 3. search_corpus.py -- same Toyota case, different tool, different matcher
# ===========================================================================
print("\n=== 3. search_corpus.py ===")
print("  (uses a lightweight stand-in for CorpusSearchStore -- this tool needs")
print("   chromadb/torch/a real embedding model, none of which exist here;")
print("   only .resolve_company() itself is being tested, so only the pieces")
print("   that method touches are set up)")

from src.agent.tools.search_corpus_tool import CorpusSearchStore

fake_companies = ["toyota motor", "sony", "panasonic"]

def make_bare_store(phonebook):
    s = CorpusSearchStore.__new__(CorpusSearchStore)
    s._companies = fake_companies
    s.phonebook = phonebook
    return s

store_no_pb = make_bare_store(None)
store_with_pb = make_bare_store(pb)

without = store_no_pb.resolve_company("Toyota Motor Corporation")
with_pb = store_with_pb.resolve_company("Toyota Motor Corporation")

check("WITHOUT phonebook: 'Toyota Motor Corporation' is LONGER than 'toyota motor', "
      "so the tool's own substring check can never match it",
      without is None, f"got '{without}'")
check("WITH phonebook: same input now resolves correctly",
      with_pb == "toyota motor", f"got '{with_pb}'")


# ===========================================================================
# 4. Honesty check -- a name the phonebook genuinely doesn't cover
# ===========================================================================
print("\n=== 4. Honesty check: names outside the phonebook behave exactly as before ===")

store_with_pb = GraphStore(graph_path=graph_path, phonebook=pb)
result = store_with_pb.find_company_node("some totally unrelated company")
check("A name with no phonebook entry at all still falls through cleanly (returns None, no crash)",
      result is None, f"got '{result}'")

# The known, documented boundary: a TRUNCATED form of a real phonebook name
# is still a miss, because the phonebook matches exact known spellings only.
ph_result = pb.graph_name("gazprom")            # truncated -- not a phonebook key
ph_result_full = pb.graph_name("gazprom pjsc")  # full form -- IS a phonebook key
check("Phonebook correctly does NOT guess at a truncated name ('gazprom')",
      ph_result is None, f"got '{ph_result}'")
check("Phonebook correctly resolves the full, real name ('gazprom pjsc')",
      ph_result_full == "gazprom pjsc", f"got '{ph_result_full}'")


print()
print("=" * 60)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL else 0)
