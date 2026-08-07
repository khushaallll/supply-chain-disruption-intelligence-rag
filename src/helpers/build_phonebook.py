"""
build_phonebook.py

Builds a single, human-reviewed name-mapping table ("the phonebook") for
every company that matters across the 30 ground-truth events -- so all
three Layer 3 tools can look up a name once, correctly, instead of each
guessing independently at query time.

Sources:
  - ground_truth_batch{1,2,3}_final.json  -- 30 events; each affected
    company already carries a `graph_node` field (pre-resolved by whoever
    built the ground truth) plus `graph_hop`/`in_graph_context`/
    `has_corpus_doc`. Seed companies have no equivalent field and are
    derived from event_id, then resolved here.
  - company_nodes.txt      -- authoritative, full list of all 6,238 real
    graph node names.
  - corpus_companies.txt   -- authoritative, full list of all 84 real
    corpus company names, extracted directly from the final
    chunks_600_80_fulltext_sentence_tagged.jsonl. Supersedes the earlier
    64-company manifest snapshot used in the first version of this script
    -- a "not found" result against this file IS now conclusive, the same
    way a graph "not found" already was.

For every name, three confidence tiers are used, in this order:
  1. exact match (case-insensitive)
  2. suffix-stripped match (same LEGAL_SUFFIXES list graph_tool.py uses)
  3. fuzzy match (rapidfuzz, token_sort_ratio) -- ALWAYS flagged for human
     review regardless of score, because this is exactly the failure mode
     the phonebook exists to eliminate: a fuzzy guess made silently, once,
     with no human check.

MANUAL_OVERRIDES below records every hand-verified correction found during
review, keyed by ground-truth name, so re-running this script (e.g. after
a source file updates) never loses a correction that already required a
human look -- unlike patching the output CSV directly, which a rebuild
would silently overwrite.
"""

import csv
import json
from pathlib import Path
from rapidfuzz import process, fuzz


LEGAL_SUFFIXES = [
    "inc", "ltd", "llc", "plc", "sa", "ag", "nv", "bv", "spa",
    "co", "corp", "gmbh", "group", "holdings",
    # Unabbreviated forms -- ground truth tends to write these out in full
    # ("Toyota Motor CORPORATION") while the graph/corpus use the
    # abbreviated form ("corp"). Missing these caused 10 real companies
    # (Toyota, Ford, GM, Sony, Hitachi, Komatsu, Panasonic, Alcoa, Exxon
    # Mobil, Shell) to silently fail corpus resolution -- found by directly
    # inspecting the output, not by the script's own review-flag logic,
    # which is the second bug fixed below.
    "corporation", "company", "limited", "incorporated",
]

# --------------------------------------------------------------------------- #
# Hand-verified corrections found during human review. Applied AFTER
# auto-resolution, so they survive every future rebuild.
# --------------------------------------------------------------------------- #

MANUAL_OVERRIDES = {
    "cpc corp taiwan": {
        "graph_name": "cpc corp/taiwan",
        "graph_match_type": "human_confirmed",
        "graph_match_score": 100.0,
        "resolution_note": (
            'Fuzzy matcher picked "csbc corp taiwan" (score 90.3) -- a DIFFERENT '
            'real Taiwanese state company (shipbuilding, not oil refining). '
            'Manually checked company_nodes.txt directly: the real node is '
            '"cpc corp/taiwan" (a literal slash, not a space, between "corp" '
            'and "taiwan"), which the fuzzy tokenizer scored lower than the '
            'wrong candidate because the slash merges "corp" and "taiwan" into '
            'one token instead of two. This is the exact failure mode the '
            'phonebook exists to catch.'
        ),
    },
    "shell (royal dutch shell / shell oil)": {
        "corpus_name": "shell",
        "corpus_match_type": "human_confirmed",
        "corpus_match_score": 100.0,
        "resolution_note": (
            'Ground truth wrote a compound/parenthetical name covering '
            'multiple historical aliases. Suffix-stripping and the fuzzy '
            'threshold both correctly failed to guess this (score 38.7) -- '
            'confirmed directly against corpus_companies.txt: the real '
            'corpus entry is simply "shell".'
        ),
    },
    "kaiser aluminum corporation": {
        "graph_name": "kaiser aluminium",
        "graph_match_type": "human_confirmed",
        "graph_match_score": 100.0,
        "resolution_note": (
            'Fuzzy match (96.8) flagged for review on principle, not because '
            'of any real doubt: company_nodes.txt has exactly ONE "kaiser" '
            'entry, "kaiser aluminium" -- American vs British spelling '
            '("aluminum" vs "aluminium") of the same real company, no '
            'competing candidate exists.'
        ),
    },
}


def strip_suffix(name: str) -> str:
    """Strip a trailing legal-entity word, tolerating trailing punctuation
    (e.g. 'Ltd.' or 'Hitachi,') that a plain membership check would miss."""
    words = name.lower().strip().split()
    if len(words) > 1:
        last_word_clean = words[-1].rstrip(".,")
        if last_word_clean in LEGAL_SUFFIXES:
            words = words[:-1]
            # also strip a trailing comma left on the new last word, e.g.
            # "hitachi, ltd." -> ["hitachi,"] -> "hitachi"
            words[-1] = words[-1].rstrip(",")
    return " ".join(words)


def resolve(name: str, candidates_lower_to_orig: dict, threshold: int = 90):
    """Returns (matched_name, match_type, score). match_type in
    {'exact','suffix_strip','fuzzy','not_found'}."""
    name_lower = name.lower().strip()

    if name_lower in candidates_lower_to_orig:
        return candidates_lower_to_orig[name_lower], "exact", 100.0

    stripped = strip_suffix(name_lower)
    if stripped != name_lower and stripped in candidates_lower_to_orig:
        return candidates_lower_to_orig[stripped], "suffix_strip", 100.0

    all_lower = list(candidates_lower_to_orig.keys())
    match, score, _ = process.extractOne(stripped, all_lower, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return candidates_lower_to_orig[match], "fuzzy", round(score, 1)

    return None, "not_found", round(score, 1)


# --------------------------------------------------------------------------- #
# Load sources
# --------------------------------------------------------------------------- #

with open("company_nodes.txt", encoding="utf-8") as f:
    graph_names = [line.strip() for line in f if line.strip()]
graph_lookup = {n.lower(): n for n in graph_names}
print(f"Graph: {len(graph_names)} node names loaded")

with open("corpus_companies.txt", encoding="utf-8") as f:
    corpus_names = [line.strip() for line in f if line.strip()]
corpus_lookup = {n.lower(): n for n in corpus_names}
print(f"Corpus: {len(corpus_names)} company names loaded (full corpus, not a partial snapshot)")

events = []
for fname in ["ground_truth_batch1_final.json", "ground_truth_batch2_final.json", "ground_truth_batch3_final.json"]:
    with open("data/ground_truth/"+fname, encoding="utf-8") as f:
        events.extend(json.load(f))
print(f"Ground truth: {len(events)} events loaded")

# --------------------------------------------------------------------------- #
# Collect every name that matters: 30 seeds + every affected company
# --------------------------------------------------------------------------- #

entries = {}

def get_or_create(name, role):
    key = name.strip().lower()
    if key not in entries:
        entries[key] = {
            "ground_truth_name": name.strip(),
            "roles": set(),
            "events": set(),
            "gt_graph_node": None,
            "gt_graph_hop": None,
            "gt_in_graph_context": None,
            "gt_has_corpus_doc": None,
        }
    entries[key]["roles"].add(role)
    return entries[key]

for ev in events:
    event_id = ev["event_id"]

    parts = event_id.split("_")
    seed_guess_parts = parts[1:]
    if seed_guess_parts and seed_guess_parts[-1].isdigit() and len(seed_guess_parts[-1]) == 4:
        seed_guess_parts = seed_guess_parts[:-1]
    seed_guess = " ".join(seed_guess_parts)

    rec = get_or_create(seed_guess, "seed")
    rec["events"].add(event_id)

    for company in (ev.get("ground_truth_affected") or []):
        rec = get_or_create(company["company"], "affected")
        rec["events"].add(event_id)
        if company.get("graph_node"):
            rec["gt_graph_node"] = company["graph_node"]
        rec["gt_graph_hop"] = company.get("graph_hop")
        rec["gt_in_graph_context"] = company.get("in_graph_context")
        rec["gt_has_corpus_doc"] = company.get("has_corpus_doc")

print(f"\nUnique company names to resolve: {len(entries)}")

# --------------------------------------------------------------------------- #
# Resolve graph + corpus spelling for every entry
# --------------------------------------------------------------------------- #

rows = []
for rec in entries.values():
    name = rec["ground_truth_name"]
    override = MANUAL_OVERRIDES.get(name.strip().lower())

    # --- graph resolution ---
    if override and "graph_name" in override:
        graph_name = override["graph_name"]
        graph_match = override["graph_match_type"]
        graph_score = override["graph_match_score"]
    elif rec["gt_graph_node"]:
        if rec["gt_graph_node"].lower() in graph_lookup:
            graph_name, graph_match, graph_score = rec["gt_graph_node"], "ground_truth_provided", 100.0
        else:
            graph_name, graph_match, graph_score = None, "ground_truth_provided_BUT_STALE", 0.0
    else:
        graph_name, graph_match, graph_score = resolve(name, graph_lookup)

    # --- corpus resolution (always self-resolved -- ground truth has no name field for this) ---
    if override and "corpus_name" in override:
        corpus_name = override["corpus_name"]
        corpus_match = override["corpus_match_type"]
        corpus_score = override["corpus_match_score"]
    else:
        corpus_name, corpus_match, corpus_score = resolve(name, corpus_lookup)

    needs_review = (
        graph_match == "fuzzy"
        or corpus_match == "fuzzy"
        or graph_match == "ground_truth_provided_BUT_STALE"
        or (graph_match == "not_found" and rec["gt_in_graph_context"] not in (False, None))
        or (corpus_match == "not_found" and rec["gt_has_corpus_doc"] is True)
    )

    rows.append({
        "ground_truth_name": name,
        "roles": ",".join(sorted(rec["roles"])),
        "n_events": len(rec["events"]),
        "events": ";".join(sorted(rec["events"])),
        "graph_name": graph_name or "",
        "graph_match_type": graph_match,
        "graph_match_score": graph_score,
        "gt_graph_hop": rec["gt_graph_hop"] if rec["gt_graph_hop"] is not None else "",
        "corpus_name": corpus_name or "",
        "corpus_match_type": corpus_match,
        "corpus_match_score": corpus_score,
        "gt_has_corpus_doc": rec["gt_has_corpus_doc"] if rec["gt_has_corpus_doc"] is not None else "",
        "needs_human_review": needs_review,
        "resolution_note": override.get("resolution_note", "") if override else "",
    })

rows.sort(key=lambda r: (not r["needs_human_review"], r["ground_truth_name"]))

# --------------------------------------------------------------------------- #
# Write CSV
# --------------------------------------------------------------------------- #

out_path = Path("data/company_phonebook.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n_review = sum(1 for r in rows if r["needs_human_review"])
n_graph_found = sum(1 for r in rows if r["graph_match_type"] not in ("not_found", "ground_truth_provided_BUT_STALE"))
n_corpus_found = sum(1 for r in rows if r["corpus_match_type"] not in ("not_found",))

print(f"\nWrote {len(rows)} rows to {out_path}")
print(f"  Resolved in graph:  {n_graph_found} / {len(rows)}")
print(f"  Resolved in corpus (full, real corpus list): {n_corpus_found} / {len(rows)}")
print(f"  Flagged for human review: {n_review} / {len(rows)}")