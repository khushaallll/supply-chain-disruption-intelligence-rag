#!/usr/bin/env python3
"""
build_corpus.py
===============

Builds the RAG document store. Step 1 of the pipeline.

WHAT IT DOES
------------
For every company in your supplier graph, it downloads the most recent annual
report that was filed BEFORE each event date, extracts the sections that
actually talk about supply chains, and chunks them into passages.

Every passage carries a published_date, so at query time you can filter to
"only what existed before the disruption" and avoid reading the future.

WHAT IT DOES NOT DO
-------------------
No embeddings. No vector database. It stops at files on disk plus a manifest.
Embedding is a separate script, because you will want to swap embedding models
without re-downloading a thousand filings.

OUTPUT
------
    data/corpus/
        raw/<doc_id>.html          the document as downloaded
        text/<doc_id>.txt          cleaned text, useful sections only
        chunks.jsonl               one JSON object per passage
        manifest.csv               one row per document
        skipped_companies.csv      who we could not cover, and why

REQUIREMENTS
------------
    pip install requests
    Python 3.9+

Put this next to enumerate_disclosures.py so it can reuse the CIK overrides.
Set USER_AGENT below to your real name and email.

USAGE
-----
    python build_corpus.py                      # everything (slow, hours)
    python build_corpus.py --events 3           # first 3 events, to test
    python build_corpus.py --max-docs 50        # cap the downloads
    python build_corpus.py --skip-fetch         # re-chunk what is already saved
    python build_corpus.py --chunk-size 500 --overlap 80
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install requests")


# ============================================================================
# CONFIG
# ============================================================================

USER_AGENT = os.getenv("USER_AGENT")
GROUND_TRUTH = ["data/ground_truth/ground_truth_batch1.json",
                "data/ground_truth/ground_truth_batch2.json",
                "data/ground_truth/ground_truth_batch3.json"]
EVENT_FILES = ["data/events/events_batch1.json", "data/events/events_batch2.json", "data/events/events_batch3.json"]
COMPANY_LOOKUP = "data/ground_truth/company_lookup.csv"

# Where to hunt for the input files, relative to the working directory.
# Add your own paths here, or pass --data-dir on the command line.
SEARCH_DIRS = [".", "data", "data/raw", "data/ground_truth", "data/inputs",
               "inputs", "../data", "../data/raw", "src/data"]

OUT = Path("data/corpus")
CACHE = OUT / "_cache"

ANNUAL_FORMS = {"10-K", "10-K405", "20-F", "40-F"}   # yearly reports only
CHUNK_WORDS = 600         # ~800 tokens
OVERLAP_WORDS = 80
MAX_DOC_BYTES = 25_000_000
REQ_PER_SEC = 7

# The parts of an annual report that discuss suppliers, customers and plants.
# Everything else is mostly accounting and is dropped.
SECTION_PATTERNS = {
    "business":      r"item\s*1\s*[\.\-—:]?\s*business",
    "risk_factors":  r"item\s*1a\s*[\.\-—:]?\s*risk\s*factors",
    "mdna":          r"item\s*7\s*[\.\-—:]?\s*management",
    # 20-F equivalents
    "f_risk":        r"item\s*3\s*[\.\-—:]?\s*key\s*information",
    "f_business":    r"item\s*4\s*[\.\-—:]?\s*information\s*on\s*the\s*company",
    "f_mdna":        r"item\s*5\s*[\.\-—:]?\s*operating\s*and\s*financial",
}
SECTION_ENDS = [
    r"item\s*1b\s*[\.\-—:]?\s*unresolved",
    r"item\s*2\s*[\.\-—:]?\s*propert",
    r"item\s*8\s*[\.\-—:]?\s*financial\s*statements",
    r"item\s*6\s*[\.\-—:]?\s*(selected|directors)",
    r"item\s*9\s*[\.\-—:]?",
]

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_CIK_LOOKUP = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUB_PAGE = "https://data.sec.gov/submissions/{name}"
SEC_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

SUFFIXES = ["incorporated","corporation","company","holdings","holding","group",
            "limited","inc","corp","co","ltd","plc","nv","n v","se","ag","sa",
            "s a","asa","ab","oyj","llc","lp","pjsc","spa","tbk","pt","bhd","pcl"]

_last = [0.0]


# ============================================================================
# plumbing
# ============================================================================

def throttle():
    gap = 1.0 / REQ_PER_SEC
    d = time.time() - _last[0]
    if d < gap:
        time.sleep(gap - d)
    _last[0] = time.time()


def get(url, as_json=False, cache_key=None, max_bytes=MAX_DOC_BYTES, retries=3):
    if cache_key:
        p = CACHE / cache_key
        if p.exists():
            raw = p.read_bytes()
            return json.loads(raw) if as_json else raw.decode("utf-8", "ignore")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    for attempt in range(retries):
        try:
            throttle()
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code in (403, 404):
                if r.status_code == 403:
                    print("   ! 403 - set USER_AGENT to a real name and email")
                return None
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            data = r.content
            if max_bytes and len(data) > max_bytes:
                return None
            if cache_key:
                p = CACHE / cache_key
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            return json.loads(data) if as_json else data.decode("utf-8", "ignore")
        except Exception as exc:
            if attempt == retries - 1:
                print(f"   ! {type(exc).__name__} on {url[:80]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def find_file(name, extra=None):
    """Look for an input file in the usual places. Returns a Path or None."""
    dirs = ([extra] if extra else []) + SEARCH_DIRS
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def require(name, extra=None, hint=""):
    """Same, but stop with a useful message instead of failing silently."""
    p = find_file(name, extra)
    if p:
        return p
    looked = "\n".join(f"    {Path(d).resolve() / name}"
                       for d in (([extra] if extra else []) + SEARCH_DIRS))
    sys.exit(f"\nCannot find '{name}'. Looked in:\n{looked}\n\n"
             f"Fix: pass --data-dir <folder containing it>, or add its folder to\n"
             f"SEARCH_DIRS at the top of this script.{hint}\n")


def normalize(name):
    n = re.sub(r"\([^)]*\)", " ", name.lower())
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    w = re.sub(r"\s+", " ", n).strip().split()
    while w and w[-1] in SUFFIXES:
        w.pop()
    return " ".join(w)


def strip_html(t):
    t = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8217;", "'")
          .replace("&#8220;", '"').replace("&#8221;", '"').replace("&#151;", "-"))
    t = re.sub(r"&#\d+;|&#x[0-9a-fA-F]+;|&[a-z]+;", " ", t)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", t)).strip()


MIN_SECTION_CHARS = 8_000        # a real Business/MD&A section is at least this long
MAX_SECTION_CHARS = 500_000
FALLBACK_RATIO = 0.15            # if we kept less than this share, keep the whole doc


def extract_sections(text, form):
    """
    Keep only Business / Risk Factors / MD&A, falling back to the whole document
    when extraction looks unreliable.

    Two traps this guards against:

    1. Table of contents. "Item 1. Business" appears there first, so we try each
       occurrence from the last backwards and reject any that yields a short body.
    2. Cross-references. Item 1 routinely says "see Item 2, Properties", which
       would truncate the section immediately. So we only start looking for the
       section end MIN_SECTION_CHARS after the heading.
    """
    low = text.lower()
    keys = (["f_business", "f_risk", "f_mdna"] if form.startswith("20-F")
            else ["business", "risk_factors", "mdna"])
    out = []
    for k in keys:
        starts = [m.start() for m in re.finditer(SECTION_PATTERNS[k], low)]
        best = None
        for s in reversed(starts):              # last heading is usually the real one
            frm = s + MIN_SECTION_CHARS
            ends = [e.start() for pat in SECTION_ENDS
                    for e in re.finditer(pat, low[frm:frm + MAX_SECTION_CHARS])]
            stop = (frm + min(ends)) if ends else min(s + MAX_SECTION_CHARS, len(text))
            body = text[s:stop]
            if len(body) >= MIN_SECTION_CHARS:
                best = body
                break
        if best:
            out.append((k, best))

    kept = sum(len(b) for _, b in out)
    if not out or (kept < FALLBACK_RATIO * len(text) and kept < 25_000):
        return [("full_document", text)], False     # unreliable, keep everything
    return out, True


def chunk_words(text, size, overlap):
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)] if w else []
    step = max(1, size - overlap)
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step) if w[i:i + size]]


# ============================================================================
# company -> CIK
# ============================================================================

def load_overrides():
    """Reuse the CIK list already curated in enumerate_disclosures.py."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import enumerate_disclosures as ed
        print(f"  reusing {len(ed.CIK_OVERRIDES)} CIK overrides and "
              f"{len(ed.KNOWN_NON_FILERS)} known non-filers from enumerate_disclosures.py")
        return dict(ed.CIK_OVERRIDES), dict(ed.KNOWN_NON_FILERS)
    except Exception:
        print("  (enumerate_disclosures.py not importable - continuing without overrides)")
        return {}, {}


def build_cik_index():
    idx = {}
    t = get(SEC_TICKERS, as_json=True, cache_key="company_tickers.json", max_bytes=None)
    if t:
        for row in t.values():
            k = normalize(row.get("title", ""))
            if k and k not in idx:
                idx[k] = int(row["cik_str"])
    raw = get(SEC_CIK_LOOKUP, cache_key="cik-lookup-data.txt", max_bytes=None)
    if raw:
        for line in raw.splitlines():
            p = line.rstrip(":").split(":")
            if len(p) >= 2 and p[-1].isdigit():
                k = normalize(p[0])
                if k and k not in idx:
                    idx[k] = int(p[-1])
    if not idx:
        sys.exit("Could not build the SEC company index. Check USER_AGENT and network.")
    print(f"  SEC name index: {len(idx):,} entries")
    return idx


def annual_filings(cik):
    """Every annual report this company has ever filed, oldest first."""
    data = get(SEC_SUBMISSIONS.format(cik=cik), as_json=True,
               cache_key=f"sub/CIK{cik:010d}.json", max_bytes=None)
    if not data:
        return []
    blocks = [data.get("filings", {}).get("recent", {})]
    for page in data.get("filings", {}).get("files", []):
        extra = get(SEC_SUB_PAGE.format(name=page["name"]), as_json=True,
                    cache_key=f"sub/{page['name']}", max_bytes=None)
        if extra:
            blocks.append(extra)
    out = []
    for b in blocks:
        for i, form in enumerate(b.get("form", [])):
            if form in ANNUAL_FORMS:
                doc = b.get("primaryDocument", [""] * (i + 1))[i]
                if doc:
                    out.append({"date": b["filingDate"][i], "form": form,
                                "accession": b["accessionNumber"][i], "doc": doc})
    out.sort(key=lambda f: f["date"])
    return out


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=None, help="only first N events")
    ap.add_argument("--max-docs", type=int, default=None, help="cap documents fetched")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="re-chunk already-downloaded text, no network")
    ap.add_argument("--full-text", action="store_true",
                    help="keep whole documents instead of extracting sections")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="folder holding company_lookup.csv, events_batch*.json "
                         "and ground_truth_batch*.json")
    a = ap.parse_args()

    if "example.com" in USER_AGENT:
        sys.exit("Set USER_AGENT to your real name and email.")

    # Keep the two modes in separate folders and files so you can build both
    # and compare retrieval quality without one clobbering the other.
    MODE = "fulltext" if a.full_text else "sections"
    # IMPORTANT: this cache holds the FULL cleaned text, never the extracted
    # sections. It is shared by both modes. If it held processed output, a
    # second run would re-process already-processed text and the chunk count
    # would drift upward every time - the run would not be reproducible.
    TEXT_DIR = OUT / "text_clean"
    for d in [OUT / "raw", TEXT_DIR, CACHE]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CORPUS BUILDER | chunk={a.chunk_size}w overlap={a.overlap}w "
          f"| sections={'off' if a.full_text else 'on'}")
    print("=" * 70)

    # ---- locate inputs, loudly -------------------------------------------
    print("\nInput files:")
    gt_paths = [require(f, a.data_dir) for f in GROUND_TRUTH]
    lookup_path = require(COMPANY_LOOKUP, a.data_dir)
    ev_paths = [find_file(f, a.data_dir) for f in EVENT_FILES]
    for p in gt_paths + [lookup_path]:
        print(f"   found  {p}")
    for f, p in zip(EVENT_FILES, ev_paths):
        print(f"   {'found  ' + str(p) if p else 'MISSING ' + f + '  (trigger docs will be skipped)'}")

    # ---- events ----------------------------------------------------------
    events = []
    for p in gt_paths:
        for e in json.load(open(p, encoding="utf-8")):
            if e.get("date_news_first"):
                events.append({"event_id": e["event_id"],
                               "date": e["date_news_first"][:10]})
    events.sort(key=lambda e: e["date"])
    if a.events:
        events = events[:a.events]
    print(f"\nEvents: {len(events)}  ({events[0]['date']} to {events[-1]['date']})")

    # ---- company universe ------------------------------------------------
    companies = []
    for r in csv.DictReader(open(lookup_path, encoding="utf-8")):
        if r.get("company"):
            companies.append(r["company"].strip())
    if not companies:
        sys.exit(f"'{lookup_path}' has no usable 'company' column. "
                 f"Columns seen: {list(next(csv.DictReader(open(lookup_path, encoding='utf-8'))).keys())}")
    seen = set()
    companies = [c for c in companies if not (c.lower() in seen or seen.add(c.lower()))]
    print(f"Companies in supplier graph: {len(companies)}")

    overrides, non_filers = load_overrides()
    idx = build_cik_index()

    # ---- resolve ---------------------------------------------------------
    resolved, skipped = {}, []
    ov_norm = {normalize(k): v for k, v in overrides.items()}
    nf_norm = {normalize(k): v for k, v in non_filers.items()}
    for c in companies:
        k = normalize(c)
        if k in nf_norm:
            skipped.append([c, "known non-filer", nf_norm[k]]); continue
        cik = overrides.get(c) or ov_norm.get(k) or idx.get(k)
        if not cik and len(k.split()) > 2:
            cik = idx.get(" ".join(k.split()[:2]))
        if cik:
            resolved[c] = cik
        else:
            skipped.append([c, "no CIK found", ""])
    print(f"Resolved to SEC filers: {len(resolved)}   not covered: {len(skipped)}")

    # ---- plan: newest annual report before each event date ---------------
    print("\nPlanning (one submissions file per company)...")
    plan = {}
    for n, (name, cik) in enumerate(sorted(resolved.items()), 1):
        if n % 25 == 0:
            print(f"   {n}/{len(resolved)}  plan={len(plan)} docs")
        filings = annual_filings(cik)
        if not filings:
            continue
        for ev in events:
            prior = [f for f in filings if f["date"] < ev["date"]]
            if prior:
                f = prior[-1]
                plan.setdefault(f["accession"], {
                    "company": name, "cik": cik, "form": f["form"],
                    "date": f["date"], "doc": f["doc"], "events": []})
                plan[f["accession"]]["events"].append(ev["event_id"])
    print(f"   unique annual reports to fetch: {len(plan)}")
    if a.max_docs:
        plan = dict(list(plan.items())[:a.max_docs])
        print(f"   capped at {len(plan)}")

    # ---- fetch, extract, chunk -------------------------------------------
    manifest, chunks_out, nchunk = [], [], 0
    chunk_path = OUT / f"chunks_{a.chunk_size}_{a.overlap}_{MODE}.jsonl"
    cf = open(chunk_path, "w", encoding="utf-8")

    print(f"\nFetching and chunking {len(plan)} documents...")
    for n, (acc, meta) in enumerate(sorted(plan.items(), key=lambda kv: kv[1]["date"]), 1):
        doc_id = hashlib.sha256(acc.encode()).hexdigest()[:16]
        url = SEC_DOC.format(cik=meta["cik"], acc=acc.replace("-", ""), doc=meta["doc"])
        txt_path = TEXT_DIR / f"{doc_id}.txt"

        if txt_path.exists():
            text, status = txt_path.read_text(encoding="utf-8"), "cached"
        elif a.skip_fetch:
            continue
        else:
            raw = get(url, cache_key=f"doc/{doc_id}.html")
            if not raw:
                manifest.append([doc_id, meta["company"], meta["cik"], "annual_report",
                                 meta["form"], meta["date"], acc, "", 0, 0, "FAILED", url])
                continue
            (OUT / "raw" / f"{doc_id}.html").write_text(raw, encoding="utf-8")
            text, status = strip_html(raw), "fetched"
            # Save the FULL cleaned text, before any section extraction, so that
            # re-runs and mode switches always start from the same input.
            txt_path.write_text(text, encoding="utf-8")

        # Section extraction always runs on the full text, never on a previous
        # run's output. This is what makes repeated runs give identical results.
        secs, ok = ([("full_document", text)], False) if a.full_text \
            else extract_sections(text, meta["form"])

        doc_chunks = 0
        for sec_name, body in secs:
            for i, ch in enumerate(chunk_words(body, a.chunk_size, a.overlap)):
                cf.write(json.dumps({
                    "chunk_id": f"{doc_id}:{sec_name}:{i:04d}",
                    "doc_id": doc_id,
                    "text": ch,
                    "company": meta["company"],
                    "cik": meta["cik"],
                    "source_type": "annual_report",
                    "form": meta["form"],
                    "published_date": meta["date"],
                    "section": sec_name,
                    "relevant_events": meta["events"],
                    "url": url,
                }, ensure_ascii=False) + "\n")
                doc_chunks += 1
        nchunk += doc_chunks
        manifest.append([doc_id, meta["company"], meta["cik"], "annual_report",
                         meta["form"], meta["date"], acc,
                         "|".join(s for s, _ in secs), len(text), doc_chunks,
                         status + ("" if ok or a.full_text else " (no sections found)"), url])
        if n % 20 == 0 or n == len(plan):
            print(f"   {n}/{len(plan)}  chunks={nchunk:,}")

    # ---- trigger documents ------------------------------------------------
    print("\nWriting trigger documents (one per event, dated on the event day)...")
    raw_events = {}
    for p in ev_paths:
        if p:
            for e in json.load(open(p, encoding="utf-8")):
                raw_events[e["event_id"]] = e
    if not raw_events:
        print("   ! no events_batch*.json found - skipping trigger documents.")
        print("     Without them the agent has no signal that an event occurred.")
    for ev in events:
        src = raw_events.get(ev["event_id"], {})
        body = (f"{src.get('description','')} "
                f"Location: {src.get('location','')}. "
                f"Category: {src.get('category','')}. "
                f"Date: {ev['date']}.").strip()
        if not src:
            continue
        doc_id = "trigger_" + ev["event_id"][:40]
        cf.write(json.dumps({
            "chunk_id": f"{doc_id}:0000", "doc_id": doc_id, "text": body,
            "company": src.get("directly_affected", ""), "cik": None,
            "source_type": "trigger", "form": "news_summary",
            "published_date": ev["date"], "section": "event_description",
            "relevant_events": [ev["event_id"]], "url": src.get("seed_source", ""),
        }, ensure_ascii=False) + "\n")
        nchunk += 1
        manifest.append([doc_id, src.get("directly_affected", ""), "", "trigger",
                         "news_summary", ev["date"], "", "event_description",
                         len(body), 1, "generated", src.get("seed_source", "")])
    cf.close()

    # ---- write manifest + skip log ---------------------------------------
    with open(OUT / f"manifest_{MODE}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "company", "cik", "source_type", "form",
                    "published_date", "accession", "sections", "chars",
                    "n_chunks", "status", "url"])
        w.writerows(manifest)
    with open(OUT / f"skipped_companies_{MODE}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["company", "reason", "detail"])
        w.writerows(skipped)

    ok_docs = [m for m in manifest if m[10] != "FAILED"]
    print("\n" + "=" * 70)
    print(f"Documents in corpus     : {len(ok_docs):,}   (failed: {len(manifest)-len(ok_docs)})")
    print(f"Chunks written          : {nchunk:,}")
    print(f"Companies covered       : {len({m[1] for m in ok_docs}):,}")
    print(f"Companies not covered   : {len(skipped):,}  (see skipped_companies_{MODE}.csv)")
    if ok_docs:
        ds = sorted(m[5] for m in ok_docs)
        print(f"Publication dates       : {ds[0]} to {ds[-1]}")
    print("=" * 70)
    print(f"\n  {chunk_path}\n  {OUT/f'manifest_{MODE}.csv'}\n  {OUT/f'skipped_companies_{MODE}.csv'}")
    print("\nEvery chunk carries published_date. At query time, keep only chunks")
    print("where published_date <= the event date. That is your leakage guard.\n")


if __name__ == "__main__":
    main()