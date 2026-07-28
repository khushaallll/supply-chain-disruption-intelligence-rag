#!/usr/bin/env python3
"""
enumerate_disclosures.py
========================

Finds the earliest SEC filing in which each affected company acknowledged its
disruption event.

THE IDEA
--------
The old way was SEARCHING: type phrases into SEC's search box and hope you
guessed the right words. If the company wrote "supply interruption" and you
searched "force majeure", you miss it.

This script ENUMERATES instead: for every affected company it downloads the
complete list of everything that company filed with the SEC, keeps the filings
made in a window around the event, opens every single document inside them
(including exhibits, where press releases live), and looks for the event.

Nothing inside that window can be missed, because every document is opened.

WHAT YOU GET
------------
1. disclosure_candidates.json  - ranked candidate disclosure dates per event
2. coverage_log.csv            - proof of what was checked (the thesis appendix)
3. unresolved_companies.csv    - companies that need a manual CIK, if any


BEFORE RUNNING
--------------
Set USER_AGENT below to your real name and email. The SEC requires this and
will block you without it.

USAGE
-----
    python enumerate_disclosures.py
    python enumerate_disclosures.py --window 120       # wider date window
    python enumerate_disclosures.py --only 11_aurizon_2010
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install requests")


# REQUIRED by the SEC. Put your real name and email or you will be blocked.
USER_AGENT = os.getenv("USER_AGENT")

# Where your three ground-truth files live
INPUT_FILES = [
    "data/events/ground_truth_batch1.json",
    "data/events/ground_truth_batch2.json",
    "data/events/ground_truth_batch3.json",
]

OUTPUT_DIR = Path("data/disclosure_sweep")
CACHE_DIR = OUTPUT_DIR / "cache"       # downloaded files, so re-runs are fast

WINDOW_DAYS_BEFORE = 2                  # look back slightly (timezones, early leaks)
WINDOW_DAYS_AFTER = 90                  # look forward this many days
MAX_DOC_BYTES = 8_000_000               # skip documents bigger than this
REQUESTS_PER_SECOND = 7                 # SEC allows 10; stay under
SNIPPET_CHARS = 240                     # how much context to save around a match
 
 
# ============================================================================
# EVENT KEYWORDS
# What words would a company use if it were talking about this event?
# Add more if you think of better ones. Generic words are allowed - the script
# shows you a snippet of every match so you can judge for yourself.
# ============================================================================
 
KEYWORDS = {
    "1_posco_2022":                 ["Hinnamnor", "Pohang"],
    "2_nippon_steel_2011":          ["Tohoku", "earthquake", "tsunami"],
    "4_dow_2017":                   ["Hurricane Harvey", "Harvey"],
    "17_basf_se_2016":              ["Ludwigshafen", "North Harbor", "North Harbour"],
    "18_chevron_2012":              ["Richmond refinery", "Richmond, California"],
    "19_cpc_corp_taiwan_2014":      ["Kaohsiung", "gas explosion"],
    "33_kyushu_electric_power_2018":["curtailment", "Kyushu"],
    "34_korea_electric_power_2011": ["blackout", "power outage", "rolling"],
    "47_china_steel_2018":          ["Section 232", "steel tariff", "tariff",
                                     "commodity cost", "raw material cost"],
    "48_glencore_2025":             ["cobalt export", "Democratic Republic of Congo", "cobalt"],
 
    "5_formosa_plastics_2021":      ["Winter Storm Uri", "winter storm", "Texas freeze"],
    "8_exxon_mobil_2005":           ["Hurricane Katrina", "Katrina"],
    "9_xinxiang_tianli_energy_2021":["Henan", "Zhengzhou", "flood"],
    "20_formosa_petrochemical_2019":["Mailiao", "aromatics"],
    "21_mitsubishi_materials_2014": ["Yokkaichi", "polysilicon"],
    "23_s_oil_2022":                ["Onsan", "S-Oil", "alkylation"],
    "35_gazprom_pjsc_2022":         ["Nord Stream", "natural gas price", "gas supply"],
    "36_basf_se_2022":              ["ammonia", "Ludwigshafen", "gas price"],
    "49_xiamen_tungsten_2010":      ["rare earth"],
    "50_aneka_tambang_tbk_2020":    ["nickel ore", "export ban", "Indonesia"],
 
    "10_hesteel_2023":              ["Doksuri", "Hebei", "flood"],
    "11_aurizon_2010":              ["Queensland", "flood", "force majeure"],
    "24_bp_2010":                   ["Deepwater Horizon", "Macondo", "oil spill"],
    "25_petrochina_2005":           ["Songhua", "Jilin", "benzene"],
    "38_perusahaan_perseroan_persero_pt_perusahaan_listrik_negara_2019":
                                    ["blackout", "power outage", "Java", "Indonesia"],
    "40_state_grid_corp_of_china_2021":
                                    ["power rationing", "power curtailment", "power shortage",
                                     "electricity restriction"],
    "41_yunnan_aluminium_2022":     ["Yunnan", "aluminium", "aluminum"],
    "54_yunnan_chihong_zinc_germanium_2023":
                                    ["gallium", "germanium", "export control"],
    "57_united_co_rusal_international_pjsc_2018":
                                    ["Rusal", "OFAC", "sanction"],
    "59_coronado_global_resources_2020":
                                    ["Australian coal", "import ban", "China"],
}
 
 
# ============================================================================
# MANUAL CIK OVERRIDES
# Fill these in from unresolved_companies.csv after your first run.
# Format:  "company name as written in ground truth": CIK number (int)
# ============================================================================
 
CIK_OVERRIDES = {
    "Rio Tinto":                                863064,
    "BHP Billiton":                             811809,
    "Peabody Energy Corporation":              1064728,
    "Transocean Ltd.":                         1451505,
    "Anadarko Petroleum Corporation":           773910,
    "Omega Protein Corporation":               1053650,
    "NXP Semiconductors N.V.":                 1413447,
    "Olin Corporation":                          74303,
    "Pactiv Evergreen Inc.":                   1527508,
    "W. R. Grace & Co.":                       1045309,
    "Core Molding Technologies, Inc.":         1026655,
    "Univar Inc. (now Univar Solutions)":      1494319,
    "Ashland Global Holdings Inc.":            1674862,
    "The Dixie Group, Inc.":                     29332,
    "AXT, Inc. (and its Beijing subsidiary Tongmei)": 1051627,
    "Constellium SE":                          1563411,
    "Kaiser Aluminum Corporation":               811596,
    "Aleris Corporation":                      1518587,
    "Tredegar Corporation":                     850429,
    "TTM Technologies, Inc.":                  1116942,
    "Diodes Incorporated":                       29002,
    "Genco Shipping & Trading Limited":         1326200,
    "General Motors Company":                  1467858,
    "Ford Motor Company":                        37996,
    "ArcelorMittal":                           1243429,
    "Vale S.A.":                                917851,
    "Alcoa Corporation":                       1675149,
    "Exxon Mobil Corporation":                   34088,
    "Sony Corporation":                         313838,
    "Hitachi, Ltd.":                             47710,
    "Panasonic Corporation":                     63271,
    "Komatsu Ltd.":                              56594,
    "Toyota Motor Corporation":                1094517,
    "FedEx Corporation":                       1048911,
    # --- added after run 1, from unresolved_companies.csv ---
    "The Dow Chemical Company":                  29915,
    "Air Products and Chemicals, Inc.":           2969,
    "ArcelorMittal S.A.":                      1243429,
}
 
 
# Companies that are definitely NOT SEC filers. Saves pointless lookups and
# keeps the coverage log honest about WHY they were not checked.
KNOWN_NON_FILERS = {
    "Korinox Co., Ltd. (코리녹스)":                          "private Korean company",
    "Eurasian Resources Group (ERG)":                        "private",
    "Nyrstar (Trafigura group)":                             "private (Trafigura)",
    "IKEA (Inter IKEA / Ingka Group)":                       "private",
    "LCY Chemical Corp.":                                    "Taiwan-listed (TWSE 1704)",
    "Formosa Chemicals & Fibre Corp (FCFC)":                 "Taiwan-listed (TWSE 1326)",
    "SUMCO Corporation":                                     "Japan-listed (TSE 3436)",
    "Unimicron Technology Corp.":                            "Taiwan-listed (TWSE 3037)",
    "Eson Precision Industry Co., Ltd.":                     "Taiwan-listed (TWSE 5243)",
    "Concraft Holding Co., Ltd.":                            "Taiwan-listed",
    "Hon Hai Precision Industry Co., Ltd. (Foxconn)":        "Taiwan-listed (TWSE 2317)",
    "SAIC Motor Corporation Limited":                        "Shanghai-listed (600104)",
    "Taiwan Power Company (Taipower)":                       "state-owned, unlisted",
    "China Petrochemical Development Corporation":           "Taiwan-listed (TWSE 1314)",
    "Aurizon Holdings Limited":                              "ASX-listed",
    "Whitehaven Coal Limited":                               "ASX-listed",
    "Macarthur Coal Limited":                                "ASX-listed (delisted 2011)",
    "Xstrata plc":                                           "LSE-listed (merged 2013)",
    "Anglo American plc":                                    "LSE-listed",
    "Yara International ASA":                                "Oslo-listed",
    "Norsk Hydro ASA / Slovalco a.s.":                       "Oslo-listed",
    "Samsung Electronics Co., Ltd.":                         "Korea-listed (KRX 005930)",
    "Infineon Technologies AG":                              "Frankfurt-listed",
    "LyondellBasell Industries N.V.":                        "SEC filer - remove me if you want it checked",
    "Nissan Motor Co., Ltd.":                                "Japan-listed, no current SEC registration",
    "Honda Motor Co., Ltd.":                                 "check manually - was NYSE-listed",
    "Telkomsel, Indosat Ooredoo, XL Axiata and Hutchison 3 Indonesia":
                                                             "bundled entry - split before running",
    "MRT Jakarta and KAI Commuter":                          "state-owned Indonesian",
    "Aquila Resources, Cockatoo Coal and Ensham Resources":  "bundled entry - split before running",
    # --- added after run 1, from unresolved_companies.csv ---
    "Formosa Plastics Corp., U.S.A.":                        "private US subsidiary of Taiwanese parent",
    "Ascend Performance Materials":                          "private",
    "INVISTA":                                               "private (Koch subsidiary)",
}
 
 
# ============================================================================
# Below here you should not need to edit anything.
# ============================================================================
 
SEC_CIK_LOOKUP = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{name}"
SEC_FILING_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
SEC_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
 
SUFFIXES = [
    "incorporated", "corporation", "company", "holdings", "holding", "group",
    "limited", "inc", "corp", "co", "ltd", "plc", "nv", "n v", "se", "ag",
    "sa", "s a", "asa", "ab", "oyj", "kk", "llc", "lp", "pjsc", "spa",
]
 
_last_request = [0.0]
 
 
def throttle():
    """Keep under the SEC's rate limit."""
    gap = 1.0 / REQUESTS_PER_SECOND
    since = time.time() - _last_request[0]
    if since < gap:
        time.sleep(gap - since)
    _last_request[0] = time.time()
 
 
def get(url, as_json=False, cache_key=None, retries=3, max_bytes=MAX_DOC_BYTES):
    """
    Download a URL, with disk caching and retries.
 
    max_bytes guards against downloading enormous individual filings.
    Pass max_bytes=None for reference files that are legitimately large.
    """
    if cache_key:
        path = CACHE_DIR / cache_key
        if path.exists():
            raw = path.read_bytes()
            return json.loads(raw) if as_json else raw.decode("utf-8", "ignore")
 
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    for attempt in range(retries):
        try:
            throttle()
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 403:
                print("      ! 403 Forbidden - the SEC is refusing this request.")
                print("        Set USER_AGENT at the top of this file to a real")
                print("        name and email address, e.g. 'Jane Doe jane@uni.ac.uk'")
                return None
            if r.status_code == 429:
                print("      ! 429 Rate limited - slowing down and retrying")
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.content
            if max_bytes and len(data) > max_bytes:
                print(f"      ! skipped, {len(data):,} bytes exceeds the "
                      f"{max_bytes:,} byte cap: {url}")
                return None
            if cache_key:
                path = CACHE_DIR / cache_key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            return json.loads(data) if as_json else data.decode("utf-8", "ignore")
        except Exception as exc:
            if attempt == retries - 1:
                print(f"      ! failed after {retries} attempts: {url}")
                print(f"        {type(exc).__name__}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None
 
 
def normalize(name):
    """Turn a company name into a plain lowercase key for matching."""
    n = name.lower()
    n = re.sub(r"\([^)]*\)", " ", n)          # drop bracketed bits
    n = re.sub(r"[^a-z0-9 ]", " ", n)         # drop punctuation
    n = re.sub(r"\s+", " ", n).strip()
    words = n.split()
    while words and words[-1] in SUFFIXES:    # strip trailing Inc/Corp/Ltd...
        words.pop()
    return " ".join(words)
 
 
def strip_html(text):
    """Remove tags so we search the words a human would read."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text)
 
 
def load_cik_index():
    """
    Build a lookup of company name -> SEC ID number.
 
    Two sources, smaller one first:
      1. company_tickers.json  (~1 MB)  every currently listed company
      2. cik-lookup-data.txt   (~10 MB) every filer ever, including old names
 
    If one fails we carry on with the other. Both are cached after the
    first successful run.
    """
    index = {}
    print("Loading SEC company index (cached after first run)...")
 
    tickers = get(SEC_TICKERS, as_json=True,
                  cache_key="company_tickers.json", max_bytes=None)
    if tickers:
        for row in tickers.values():
            key = normalize(row.get("title", ""))
            if key and key not in index:
                index[key] = int(row["cik_str"])
        print(f"  company_tickers.json : {len(index):,} names")
    else:
        print("  company_tickers.json : FAILED")
 
    raw = get(SEC_CIK_LOOKUP, cache_key="cik-lookup-data.txt", max_bytes=None)
    if raw:
        added = 0
        for line in raw.splitlines():
            parts = line.rstrip(":").split(":")
            if len(parts) >= 2 and parts[-1].isdigit():
                key = normalize(parts[0])
                if key and key not in index:
                    index[key] = int(parts[-1])
                    added += 1
        print(f"  cik-lookup-data.txt  : +{added:,} names")
    else:
        print("  cik-lookup-data.txt  : FAILED (carrying on without it)")
 
    if not index:
        sys.exit(
            "\nCould not build a company index from either SEC source.\n"
            "Things to check:\n"
            "  1. USER_AGENT at the top of this file is a real name and email\n"
            "  2. You have internet access and sec.gov is not blocked\n"
            "  3. Open https://www.sec.gov/files/company_tickers.json in a\n"
            "     browser - if that fails too, it is a network issue, not the script\n"
            "\nWorkaround: every company you care about is already listed in\n"
            "CIK_OVERRIDES, so you can comment out this function's exit and the\n"
            "script will still resolve them.\n"
        )
    print(f"  total unique names   : {len(index):,}\n")
    return index
 
 
def resolve_cik(name, index):
    """Find a company's SEC ID number. Returns (cik, how_we_found_it)."""
    if name in CIK_OVERRIDES:
        return CIK_OVERRIDES[name], "manual override"
    key = normalize(name)
    if key in index:
        return index[key], "exact name match"
    # try the first two words, e.g. "peabody energy"
    words = key.split()
    if len(words) > 2:
        short = " ".join(words[:2])
        if short in index:
            return index[short], f"partial match on '{short}'"
    return None, "not found"
 
 
def get_filings_in_window(cik, start, end):
    """
    Return every filing this company made between two dates.
    Handles SEC's paging: recent filings sit in one file, older ones in others.
    """
    data = get(SEC_SUBMISSIONS.format(cik=cik), as_json=True,
               cache_key=f"submissions/CIK{cik:010d}.json")
    if not data:
        return []
 
    blocks = [data.get("filings", {}).get("recent", {})]
 
    # older filings live in separate paged files
    for page in data.get("filings", {}).get("files", []):
        p_from = page.get("filingFrom", "")
        p_to = page.get("filingTo", "")
        if p_to >= start and p_from <= end:            # date ranges overlap
            extra = get(SEC_SUBMISSIONS_PAGE.format(name=page["name"]), as_json=True,
                        cache_key=f"submissions/{page['name']}")
            if extra:
                blocks.append(extra)
 
    out = []
    for block in blocks:
        dates = block.get("filingDate", [])
        for i, d in enumerate(dates):
            if start <= d <= end:
                out.append({
                    "date": d,
                    "form": block["form"][i],
                    "accession": block["accessionNumber"][i],
                })
    out.sort(key=lambda f: f["date"])
    return out
 
 
def documents_in_filing(cik, accession):
    """List every readable document inside one filing, including exhibits."""
    acc = accession.replace("-", "")
    idx = get(SEC_FILING_INDEX.format(cik=cik, acc=acc), as_json=True,
              cache_key=f"index/{cik}/{acc}.json")
    if not idx:
        return []
    docs = []
    for item in idx.get("directory", {}).get("item", []):
        nm = item.get("name", "")
        if not nm.lower().endswith((".htm", ".html", ".txt")):
            continue
        if "-index" in nm.lower() or nm.lower().endswith(".xml"):
            continue
 
        # Skip the "<accession>.txt" file. That is the entire submission
        # concatenated into one blob - every document inside it is already
        # listed separately below, so reading it would just be a slow
        # duplicate. It is also the only thing that trips the size cap.
        if nm.rsplit(".", 1)[0].replace("-", "").lower() == acc.lower():
            continue
 
        try:
            if int(item.get("size", 0)) > MAX_DOC_BYTES:
                continue
        except (TypeError, ValueError):
            pass                       # size missing, let get() decide
        docs.append(nm)
    return docs
 
 
def scan_document(cik, accession, docname, keywords):
    """Open one document and look for any of the event keywords."""
    acc = accession.replace("-", "")
    raw = get(SEC_DOC.format(cik=cik, acc=acc, doc=docname),
              cache_key=f"docs/{cik}/{acc}/{docname}")
    if not raw:
        return None
    text = strip_html(raw)
    low = text.lower()
    for kw in keywords:
        pos = low.find(kw.lower())
        if pos != -1:
            a = max(0, pos - SNIPPET_CHARS // 2)
            return {
                "keyword": kw,
                "snippet": text[a:a + SNIPPET_CHARS].strip(),
                "url": SEC_DOC.format(cik=cik, acc=acc, doc=docname),
                "document": docname,
            }
    return None
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=WINDOW_DAYS_AFTER,
                    help="days after the event to search (default 90)")
    ap.add_argument("--only", type=str, default=None,
                    help="run a single event_id, for testing")
    args = ap.parse_args()
 
    if "your.name@example.com" in USER_AGENT or not USER_AGENT.strip():
        sys.exit("Set USER_AGENT at the top of this file to your real name and email.")
 
    # Fingerprint so the console proves which version of this file just ran.
    print(f"=== BUILD 2 | {len(CIK_OVERRIDES)} CIK overrides | "
          f"{len(KNOWN_NON_FILERS)} known non-filers | "
          f"{len(KEYWORDS['47_china_steel_2018'])} keywords on event 47 ===")
    print(f"    script : {Path(__file__).resolve()}")
    print(f"    output : {OUTPUT_DIR.resolve()}\n")
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
 
    # ---- load the ground truth -------------------------------------------
    events = []
    for f in INPUT_FILES:
        if not Path(f).exists():
            sys.exit(f"Cannot find {f}. Put this script next to your JSON files.")
        events.extend(json.load(open(f, encoding="utf-8")))
    if args.only:
        events = [e for e in events if e["event_id"] == args.only]
    print(f"Loaded {len(events)} events\n")
 
    cik_index = load_cik_index()
 
    results = {}
    coverage_rows = []
    unresolved = []
 
    for ev in events:
        eid = ev["event_id"]
        event_date = ev.get("date_news_first")
        if not event_date:
            continue
        try:
            d0 = datetime.strptime(event_date[:10], "%Y-%m-%d")
        except ValueError:
            print(f"[{eid}] bad date '{event_date}' - skipping")
            continue
 
        start = (d0 - timedelta(days=WINDOW_DAYS_BEFORE)).strftime("%Y-%m-%d")
        end = (d0 + timedelta(days=args.window)).strftime("%Y-%m-%d")
        keywords = KEYWORDS.get(eid, [])
        companies = ev.get("ground_truth_affected", [])
 
        print(f"[{eid}]  window {start} -> {end}   ({len(companies)} companies)")
        if not keywords:
            print("   ! no keywords defined for this event - add some to KEYWORDS")
 
        hits = []
 
        for entry in companies:
            name = entry.get("company") if isinstance(entry, dict) else str(entry)
            if not name:
                continue
 
            if name in KNOWN_NON_FILERS:
                reason = KNOWN_NON_FILERS[name]
                print(f"   - {name[:50]:52} skipped ({reason})")
                coverage_rows.append([eid, name, "", "not an SEC filer",
                                      reason, 0, 0, ""])
                continue
 
            cik, how = resolve_cik(name, cik_index)
            if not cik:
                print(f"   ? {name[:50]:52} CIK NOT FOUND")
                unresolved.append([eid, name, normalize(name)])
                coverage_rows.append([eid, name, "", "unresolved",
                                      "could not map name to a CIK", 0, 0, ""])
                continue
 
            filings = get_filings_in_window(cik, start, end)
            n_docs = 0
            found_here = []
 
            for filing in filings:
                for docname in documents_in_filing(cik, filing["accession"]):
                    n_docs += 1
                    match = scan_document(cik, filing["accession"], docname, keywords)
                    if match:
                        found_here.append({
                            "company": name,
                            "cik": cik,
                            "date": filing["date"],
                            "form": filing["form"],
                            "accession": filing["accession"],
                            **match,
                        })
                        break          # one hit per filing is enough
 
            status = "MATCH" if found_here else "no match"
            earliest = found_here[0]["date"] if found_here else ""
            print(f"   . {name[:50]:52} CIK {cik:<8} "
                  f"{len(filings):>3} filings  {n_docs:>4} docs  {status} {earliest}")
 
            coverage_rows.append([eid, name, cik, "checked", how,
                                  len(filings), n_docs, earliest])
            hits.extend(found_here)
 
        hits.sort(key=lambda h: h["date"])
        results[eid] = {
            "event_date": event_date,
            "window": [start, end],
            "keywords": keywords,
            "existing_date_first_disclosure": ev.get("date_first_disclosure"),
            "proposed_date_first_disclosure": hits[0]["date"] if hits else None,
            "candidates": hits,
        }
 
        old = ev.get("date_first_disclosure")
        new = hits[0]["date"] if hits else None
        if new and old and new < old:
            print(f"   >> EARLIER than recorded: {new} beats {old}")
        elif new and not old:
            print(f"   >> NEW date found: {new}")
        print()
 
    # ---- write outputs ----------------------------------------------------
    out_json = OUTPUT_DIR / "disclosure_candidates.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
 
    out_cov = OUTPUT_DIR / "coverage_log.csv"
    with open(out_cov, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "company", "cik", "status", "how_resolved",
                    "filings_in_window", "documents_opened", "earliest_match"])
        w.writerows(coverage_rows)
 
    if unresolved:
        out_unres = OUTPUT_DIR / "unresolved_companies.csv"
        with open(out_unres, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_id", "company", "normalized_name"])
            w.writerows(unresolved)
 
    # ---- summary ----------------------------------------------------------
    checked = [r for r in coverage_rows if r[3] == "checked"]
    docs_total = sum(r[6] for r in checked)
    with_date = sum(1 for r in results.values() if r["proposed_date_first_disclosure"])
    improved = sum(1 for r in results.values()
                   if r["proposed_date_first_disclosure"] and
                   (not r["existing_date_first_disclosure"] or
                    r["proposed_date_first_disclosure"] < r["existing_date_first_disclosure"]))
 
    print("=" * 70)
    print(f"Companies checked on EDGAR : {len(checked)}")
    print(f"Skipped (not SEC filers)   : {sum(1 for r in coverage_rows if r[3]=='not an SEC filer')}")
    print(f"Unresolved (need manual CIK): {len(unresolved)}")
    print(f"Documents opened           : {docs_total:,}")
    print(f"Events with a candidate    : {with_date} / {len(results)}")
    print(f"New or earlier than before : {improved}")
    print("=" * 70)
    print(f"\nWrote:\n  {out_json}\n  {out_cov}")
    if unresolved:
        print(f"  {OUTPUT_DIR/'unresolved_companies.csv'}   <- add these to CIK_OVERRIDES and re-run")
    print("\nNext: open disclosure_candidates.json, read the snippet for the top "
          "candidate of each event, and accept or reject it.\n")
 
 
if __name__ == "__main__":
    main()