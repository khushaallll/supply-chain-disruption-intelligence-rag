#!/usr/bin/env python3
"""
tag_chunks.py
=============

Adds two fields to a chunks file:

  relevant_events   which events this document is the newest annual report for.
                    Same rule build_corpus.py used: for each company and each
                    event, take the most recent report filed strictly BEFORE the
                    event date. Triggers already carry their own event and are
                    left alone.

  published_int     the filing date as an integer, e.g. 20220906.
                    Chroma's $lte filter is reliable on numbers and fiddly on
                    date strings, so this is what you filter on at query time.
                    published_date stays as-is for reading and citing.

It streams the file line by line, so a 350 MB corpus is fine on any machine.

USAGE
    python tag_chunks.py                       # sentence chunks, writes a new file
    python tag_chunks.py --in-place            # replace the file, keeps a .bak
    python tag_chunks.py --chunks data/corpus/chunks_600_80_fulltext.jsonl
    python tag_chunks.py --inclusive           # use <= event date instead of <
"""

import argparse, csv, json, os, shutil, sys
from collections import defaultdict
from pathlib import Path

SEARCH_DIRS = [".", "data", "data/corpus", "data/ground_truth", "data/events",
               "data/raw", "inputs", "../data"]


def find(name, extra=None):
    for d in ([extra] if extra else []) + SEARCH_DIRS:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def require(name, extra=None):
    p = find(name, extra)
    if p:
        return p
    looked = "\n".join(f"    {(Path(d)/name).resolve()}"
                       for d in ([extra] if extra else []) + SEARCH_DIRS)
    sys.exit(f"\nCannot find '{name}'. Looked in:\n{looked}\n"
             f"Pass the path explicitly or edit SEARCH_DIRS.\n")


def to_int(d):
    """'2022-09-06' -> 20220906. Returns None if the date is unusable."""
    try:
        return int(str(d)[:10].replace("-", ""))
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/corpus/chunks_600_80_fulltext_sentence.jsonl")
    ap.add_argument("--manifest", default=None,
                    help="defaults to manifest_<mode>_sentence.csv next to the chunks")
    ap.add_argument("--output", default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the chunks file (a .bak copy is kept)")
    ap.add_argument("--inclusive", action="store_true",
                    help="treat a report filed ON the event date as available (default: strictly before)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()

    chunks_path = Path(a.chunks)
    if not chunks_path.exists():
        found = find(chunks_path.name, a.data_dir)
        if not found:
            sys.exit(f"Cannot find {a.chunks}")
        chunks_path = found

    # manifest: prefer the one matching this chunks file, else any manifest
    if a.manifest:
        man_path = Path(a.manifest)
    else:
        stem = chunks_path.stem                         # chunks_600_80_fulltext_sentence
        mode = "sentence" if stem.endswith("_sentence") else ""
        guesses = ([f"manifest_fulltext_sentence.csv", "manifest_fulltext.csv"]
                   if "fulltext" in stem else
                   [f"manifest_sections_sentence.csv", "manifest_sections.csv"])
        man_path = None
        for g in guesses:
            man_path = find(g, str(chunks_path.parent)) or find(g, a.data_dir)
            if man_path:
                break
        if not man_path:
            sys.exit("Cannot find a manifest CSV. Pass --manifest.")

    print("=" * 70)
    print("TAG CHUNKS")
    print("=" * 70)
    print(f"  chunks   {chunks_path}")
    print(f"  manifest {man_path}")

    # ---- 1. event dates -------------------------------------------------
    events = {}
    for i in (1, 2, 3):
        p = find(f"ground_truth_batch{i}_final.json", a.data_dir) \
            or find(f"ground_truth_batch{i}.json", a.data_dir)
        if p:
            for e in json.load(open(p, encoding="utf-8")):
                if e.get("date_news_first"):
                    events[e["event_id"]] = e["date_news_first"][:10]
    if not events:
        for i in (1, 2, 3):
            p = require(f"events_batch{i}.json", a.data_dir)
            for e in json.load(open(p, encoding="utf-8")):
                events[e["event_id"]] = e["date"][:10]
    print(f"  events   {len(events)}")

    # ---- 2. per company, that company's reports in date order -----------
    by_company = defaultdict(list)
    for r in csv.DictReader(open(man_path, encoding="utf-8")):
        if r.get("source_type") != "annual_report":
            continue
        if r.get("status") == "FAILED" or not r.get("published_date"):
            continue
        by_company[r["company"]].append((r["published_date"][:10], r["doc_id"]))
    for c in by_company:
        by_company[c].sort()
    print(f"  companies with reports  {len(by_company)}")

    # ---- 3. newest report per company, per event ------------------------
    doc_events = defaultdict(list)
    for eid, edate in sorted(events.items(), key=lambda kv: kv[1]):
        for company, reports in by_company.items():
            prior = [(d, i) for d, i in reports
                     if (d <= edate if a.inclusive else d < edate)]
            if prior:
                doc_events[prior[-1][1]].append(eid)
    print(f"  documents tagged        {len(doc_events)}")

    # doc_id -> published date, for the verification pass
    doc_date = {i: d for reports in by_company.values() for d, i in reports}

    # ---- 4. stream and patch --------------------------------------------
    if a.in_place:
        out_path = chunks_path.with_suffix(".jsonl.tmp")
    else:
        out_path = Path(a.output) if a.output else \
            chunks_path.with_name(chunks_path.stem + "_tagged.jsonl")

    n = n_int_fail = n_tagged = n_kept = n_trig = 0
    bad_leak = 0
    with open(chunks_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            c = json.loads(line)
            n += 1

            pi = to_int(c.get("published_date"))
            if pi is None:
                n_int_fail += 1
            c["published_int"] = pi

            if c.get("source_type") == "trigger":
                n_trig += 1                              # already has its event
            elif c.get("relevant_events"):
                n_kept += 1                              # already tagged, leave it
            else:
                evs = doc_events.get(c.get("doc_id"), [])
                c["relevant_events"] = evs
                if evs:
                    n_tagged += 1

            # safety: an event must never be dated at or before the document
            for eid in c.get("relevant_events") or []:
                ed = events.get(eid)
                if ed and c.get("published_date") and c["published_date"][:10] >= ed:
                    bad_leak += 1

            fout.write(json.dumps(c, ensure_ascii=False) + "\n")
            if n % 25000 == 0:
                print(f"     {n:,} chunks...")

    if a.in_place:
        bak = chunks_path.with_suffix(".jsonl.bak")
        if not bak.exists():
            shutil.copyfile(chunks_path, bak)
            print(f"\n  backup  {bak}")
        os.replace(out_path, chunks_path)
        out_path = chunks_path

    # ---- 5. report ------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  chunks processed          {n:,}")
    print(f"  published_int added       {n - n_int_fail:,}"
          + (f"   ({n_int_fail} failed - check those dates)" if n_int_fail else ""))
    print(f"  relevant_events filled    {n_tagged:,}")
    print(f"  already had events        {n_kept:,}")
    print(f"  triggers left alone       {n_trig:,}")
    print(f"  LEAKAGE CHECK             {bad_leak}   (must be 0)")
    if bad_leak:
        print("  !! a chunk is tagged to an event dated on or before its own filing date")
    print("=" * 70)

    ev_counts = defaultdict(int)
    for d, evs in doc_events.items():
        for e in evs:
            ev_counts[e] += 1
    lo = sorted(ev_counts.items(), key=lambda kv: kv[1])[:3]
    hi = sorted(ev_counts.items(), key=lambda kv: -kv[1])[:3]
    print(f"\n  documents per event:  fewest {[(e.split('_')[0], v) for e, v in lo]}")
    print(f"                        most   {[(e.split('_')[0], v) for e, v in hi]}")
    missing = [e for e in events if e not in ev_counts]
    print(f"  events with no documents: {len(missing)} {missing if missing else ''}")
    print(f"\n  wrote {out_path}\n")
    print("  At query time, filter on published_int:")
    print("     where={'published_int': {'$lte': 20220906}}\n")


if __name__ == "__main__":
    main()
