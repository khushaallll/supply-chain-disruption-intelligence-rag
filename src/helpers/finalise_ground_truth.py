#!/usr/bin/env python3
"""
finalise_ground_truth.py
========================

Applies the four Section-1 fixes to the ground truth and writes final files.

WHAT IT FIXES
-------------
1. Graph tags       - recomputes graph_node / graph_hop / in_graph_context from
                      graph_enriched.pkl instead of trusting whatever is in the file
2. Bundled rows     - splits rows that hold several companies into one row each
3. Scoring bucket   - labels every event A / B / C / D so your scorer knows which
                      events can be scored for recall and which cannot
4. Negative type    - for empty events, records whether nobody was hurt
                      (provable_negative) or whether we simply could not find out
                      (not_found). Scoring these the same corrupts your numbers.

IT IS IDEMPOTENT. Running it twice gives byte-identical output, because every
derived field is recomputed from source rather than updated in place.

INPUTS  (searched in ., data, data/ground_truth, data/raw, data/corpus, ...)
    ground_truth_batch1.json, ground_truth_batch2.json, ground_truth_batch3.json
    events_batch1.json, events_batch2.json, events_batch3.json
    graph_enriched.pkl
    manifest_fulltext.csv        (optional - without it, B and C cannot be told apart)

OUTPUTS
    ground_truth_batch{1,2,3}_final.json
    ground_truth_summary.csv     one row per event, for your records

USAGE
    pip install networkx
    python finalise_ground_truth.py
    python finalise_ground_truth.py --data-dir data/ground_truth
"""

import argparse, csv, json, pickle, re, sys
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    sys.exit("Missing dependency. Run:  pip install networkx")

SEARCH_DIRS = [".", "data", "data/processed","data/ground_truth", "data/raw", "data/events",
               "data/corpus", "data/graph", "inputs", "../data"]

# ---------------------------------------------------------------------------
# FIX 2 - rows that hold more than one company.
# Left-hand side must match the "company" string exactly.
# ---------------------------------------------------------------------------
BUNDLED = {
    "Aquila Resources, Cockatoo Coal and Ensham Resources":
        ["Aquila Resources", "Cockatoo Coal", "Ensham Resources"],
    "Telkomsel, Indosat Ooredoo, XL Axiata and Hutchison 3 Indonesia":
        ["Telkomsel", "Indosat Ooredoo", "XL Axiata", "Hutchison 3 Indonesia"],
    "MRT Jakarta and KAI Commuter":
        ["MRT Jakarta", "KAI Commuter"],
    "Norsk Hydro ASA / Slovalco a.s.":
        ["Norsk Hydro ASA", "Slovalco a.s."],
}

# ---------------------------------------------------------------------------
# FIX 4 - why an event has no affected companies.
# provable_negative = there is a documented reason nobody downstream was hurt.
# not_found         = we searched and found nothing, but something may exist.
# Only provable_negative should be scored as a true negative.
# ---------------------------------------------------------------------------
NEGATIVE_TYPE = {
    "33_kyushu_electric_power_2018": ("provable_negative",
        "Curtailment harms electricity GENERATORS. Every graph candidate is an "
        "electricity CONSUMER, so the direction of harm is inverted. Any positive "
        "prediction here is a false positive by construction."),
    "34_korea_electric_power_2011": ("provable_negative",
        "Korean reporting states that large listed firms with backup generation and "
        "priority fabs were explicitly EXEMPTED from load shedding. The losses fell on "
        "small and medium enterprises outside the graph."),
    "18_chevron_2012":              ("not_found", "Price-mediated event; no named victim located."),
    "23_s_oil_2022":                ("not_found", "Price-mediated event; no named victim located."),
    "36_basf_se_2022":              ("not_found", "Price-mediated event; no named victim located."),
    "49_xiamen_tungsten_2010":      ("not_found",
        "No named victim located. Note also that peer-reviewed work disputes whether the "
        "embargo occurred at all - consider excluding this event entirely."),
    "50_aneka_tambang_tbk_2020":    ("not_found", "Price-mediated event; no named victim located."),
    "10_hesteel_2023":              ("not_found", "Price-mediated event; no named victim located."),
    "25_petrochina_2005":           ("not_found",
        "Harm was environmental and municipal rather than commercial; no named corporate victim."),
    "41_yunnan_aluminium_2022":     ("not_found", "Price-mediated event; no named victim located."),
}

BUCKET_MEANING = {
    "A": "No affected companies. Scores RESTRAINT only - can the agent stay quiet? "
         "Recall is undefined here; do not include it in a recall average.",
    "B": "Graph can reach at least one correct company AND the corpus has a document "
         "for it. Full end-to-end test. Score recall and precision.",
    "C": "Graph can reach at least one correct company but no corpus document exists. "
         "Tests reasoning from graph attributes. Score recall and precision.",
    "D": "No correct company is reachable in the graph within 2 hops. The agent CANNOT "
         "succeed. Exclude from recall; report as a graph-coverage finding.",
}

SUFFIXES = {"incorporated","corporation","company","holdings","holding","group","limited",
            "inc","corp","co","ltd","plc","nv","n v","se","ag","sa","s a","asa","ab","llc",
            "lp","pjsc","tbk","pt","a s","as"}

# Manual name -> graph node, for cases the normaliser cannot bridge
ALIAS = {
    "the dow chemical company": "dow",
    "hon hai precision industry co., ltd. (foxconn)": "hon hai precision industry",
    "nyrstar (trafigura group)": "nyrstar",
    "shell (royal dutch shell / shell oil)": "shell",
    "occidental chemical (oxychem)": "occidental petroleum",
    "axt, inc. (and its beijing subsidiary tongmei)": "axt",
    "formosa plastics corp., u.s.a.": "formosa plastics",
    "eurasian resources group (erg)": "eurasian resources",
    "korinox co., ltd. (코리녹스)": "korinox",
    "formosa chemicals & fibre corp (fcfc)": "formosa chemicals & fibre",
    "taiwan power company (taipower)": "taiwan power",
    "univar inc. (now univar solutions)": "univar",
}


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
             f"Pass --data-dir <folder> or edit SEARCH_DIRS.\n")


def norm(s):
    s = re.sub(r"\([^)]*\)", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    w = re.sub(r"\s+", " ", s).strip().split()
    while w and w[-1] in SUFFIXES:
        w.pop()
    return " ".join(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("FINALISE GROUND TRUTH")
    print("=" * 74)

    gt_paths = [require(f"ground_truth_batch{i}.json", a.data_dir) for i in (1, 2, 3)]
    ev_paths = [require(f"events_batch{i}.json", a.data_dir) for i in (1, 2, 3)]
    g_path   = require("graph_enriched.pkl", a.data_dir)
    m_path   = find("manifest_fulltext.csv", a.data_dir) or find("manifest_sections.csv", a.data_dir)
    for p in gt_paths + ev_paths + [g_path]:
        print(f"  found  {p}")
    print(f"  {'found  ' + str(m_path) if m_path else 'MISSING manifest - B and C cannot be distinguished'}")

    g = pickle.load(open(g_path, "rb"))
    node_idx = {}
    for n in g.nodes():
        node_idx.setdefault(norm(n), n)
    print(f"\n  graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    corpus = set()
    if m_path:
        for r in csv.DictReader(open(m_path, encoding="utf-8")):
            if r.get("status") != "FAILED" and r.get("source_type") == "annual_report":
                corpus.add(norm(r["company"]))
        print(f"  corpus: documents for {len(corpus)} companies")

    seeds = {}
    for p in ev_paths:
        for e in json.load(open(p, encoding="utf-8")):
            seeds[e["event_id"]] = e["directly_affected"]

    def resolve(name):
        a_ = ALIAS.get(name.lower())
        if a_ and norm(a_) in node_idx:
            return node_idx[norm(a_)]
        k = norm(name)
        if k in node_idx:
            return node_idx[k]
        for c in (" ".join(k.split()[:2]), " ".join(k.split()[:1])):
            if c and c in node_idx:
                return node_idx[c]
        return None

    GRAPH_NOTE = ("Recomputed from graph_enriched.pkl: shortest directed path from the event "
                  "seed. Only hop1 and hop2 count as in-graph, because hop3 already reaches "
                  "~35% of the 6,233-node graph and hop4+ reaches 81-99%, so greater distances "
                  "carry no discriminative signal.")

    splits, rows, buckets = [], [], {}
    for i, p in zip((1, 2, 3), gt_paths):
        data = json.load(open(p, encoding="utf-8"))
        for ev in data:
            eid = ev["event_id"]
            seed = seeds.get(eid)

            # ---- FIX 2: split bundled rows -------------------------------
            expanded = []
            for entry in ev["ground_truth_affected"]:
                names = BUNDLED.get(entry["company"])
                if names:
                    splits.append((eid, entry["company"], len(names)))
                    for nm in names:
                        e2 = dict(entry)
                        e2["company"] = nm
                        e2["split_from"] = entry["company"]
                        expanded.append(e2)
                else:
                    expanded.append(dict(entry))

            # ---- FIX 1: recompute graph position -------------------------
            for entry in expanded:
                node = resolve(entry["company"])
                hop = None
                if node and seed in g:
                    try:
                        hop = nx.shortest_path_length(g, seed, node)
                    except Exception:
                        hop = None
                entry["graph_node"] = node
                entry["graph_hop"] = hop
                entry["in_graph_context"] = f"hop{hop}" if (hop is not None and hop <= 2) else False
                entry["has_corpus_doc"] = (norm(entry["company"]) in corpus) if m_path else None
                entry["graph_note"] = GRAPH_NOTE
            ev["ground_truth_affected"] = expanded

            # ---- FIX 3 + 4: bucket and negative type ---------------------
            n = len(expanded)
            reach = [e for e in expanded if e["graph_hop"] is not None and e["graph_hop"] <= 2]
            withdoc = [e for e in reach if e["has_corpus_doc"]]
            if n == 0:
                bucket = "A"
                kind, why = NEGATIVE_TYPE.get(eid, ("not_found", "Not classified - review."))
                ev["negative_type"] = kind
                ev["negative_reason"] = why
            elif withdoc:
                bucket = "B"
            elif reach:
                bucket = "C"
            else:
                bucket = "D"
            ev["scoring_bucket"] = bucket
            ev["scoring_note"] = BUCKET_MEANING[bucket]
            ev["score_for_recall"] = bucket in ("B", "C")
            ev["score_for_precision"] = True

            buckets[eid] = bucket
            rows.append([eid, bucket, n, len(reach), len(withdoc),
                         ev.get("negative_type", ""), ev.get("date_first_sec_filing") or "",
                         "yes" if bucket in ("B", "C") else "no"])

        json.dump(data, open(out / f"ground_truth_batch{i}_final.json", "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    with open(out / "ground_truth_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "scoring_bucket", "n_companies", "n_reachable_hop1_2",
                    "n_with_corpus_doc", "negative_type", "date_first_sec_filing",
                    "score_for_recall"])
        w.writerows(rows)

    # ---- report -----------------------------------------------------------
    print(f"\n  FIX 2: split {len(splits)} bundled rows")
    for eid, orig, k in splits:
        print(f"     {eid[:30]:<32}{orig[:44]:<46}-> {k} rows")

    tot = sum(r[2] for r in rows)
    reach = sum(r[3] for r in rows)
    docs = sum(r[4] for r in rows)
    bc = collections_count(rows)
    print(f"\n  FIX 3: bucket counts   " + "  ".join(f"{k}={v}" for k, v in sorted(bc.items())))
    neg = [r for r in rows if r[1] == "A"]
    pn = sum(1 for r in neg if r[5] == "provable_negative")
    print(f"  FIX 4: of {len(neg)} empty events, {pn} provable_negative, {len(neg)-pn} not_found")

    print(f"\n  companies after splitting : {tot}   (was 80)")
    print(f"  reachable in graph hop1-2 : {reach}")
    print(f"  with a corpus document    : {docs}")
    rec_events = [r for r in rows if r[7] == "yes"]
    rec_n = sum(r[2] for r in rec_events)
    rec_r = sum(r[3] for r in rec_events)
    print(f"\n  RECALL SET: {len(rec_events)} events, {rec_n} companies, "
          f"{rec_r} reachable -> ceiling {100*rec_r/rec_n:.0f}%")
    print(f"  EXCLUDED  : {sum(1 for r in rows if r[1]=='D')} events (bucket D, unreachable), "
          f"{len(neg)} events (bucket A, nothing to find)")

    print(f"\n  wrote {out/'ground_truth_batch1_final.json'} (+2 more)")
    print(f"        {out/'ground_truth_summary.csv'}")
    print("\n  Run this twice - the output should be identical both times.\n")


def collections_count(rows):
    d = {}
    for r in rows:
        d[r[1]] = d.get(r[1], 0) + 1
    return d


if __name__ == "__main__":
    main()
