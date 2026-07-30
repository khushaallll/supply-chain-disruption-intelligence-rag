#!/usr/bin/env python3
"""
rechunk_sentences.py
====================

Re-chunks the corpus on sentence boundaries instead of fixed word windows.

It reads the already-cleaned text in data/corpus/text_clean/ - so there is no
downloading and no network. A full re-chunk takes a few minutes.

WHY
---
The original chunker cut every 600 words regardless of where sentences ended,
so most chunks began mid-sentence. This one packs whole sentences up to a word
budget, and overlaps by whole sentences too. Nothing is ever cut mid-sentence
unless a single "sentence" is itself longer than the budget (tables, long lists),
in which case it is split by words as a fallback and flagged.

OUTPUT
------
    chunks_<size>_<overlap>_<mode>_sentence.jsonl

Same fields as the word-based file, plus:
    n_sentences   how many sentences are in this chunk
    hard_split    true if a single oversized sentence had to be cut by words

Keep both files. Running retrieval against each is a cheap ablation:
"does sentence-aware chunking change retrieval quality?"

USAGE
-----
    python rechunk_sentences.py                     # full-text mode
    python rechunk_sentences.py --mode sections     # section-filtered mode
    python rechunk_sentences.py --chunk-size 400 --overlap 60
"""

import argparse, csv, json, re, sys
from pathlib import Path

OUT = Path("data/corpus")
TEXT_DIR = OUT / "text_clean"

# ---------------------------------------------------------------------------
# Sentence splitting
#
# SEC filings are full of things that look like sentence ends but are not:
#   "Autoliv Inc. is a Delaware holding corporation"
#   "$1.5 billion", "3.4%", "No. 2 Hot Rolling Mill"
#   "U.S. operations", "e.g. tariffs"
# The guard below protects those, then splits only where a period, question mark
# or exclamation mark is followed by whitespace and then something that looks
# like the start of a new sentence.
# ---------------------------------------------------------------------------

ABBREV = [
    "Inc", "Corp", "Ltd", "Co", "LLC", "L.L.C", "LP", "PLC", "plc", "N.V", "S.A",
    "A.G", "S.p.A", "Pty", "Bhd", "Pte", "GmbH", "AB", "AG", "SE", "NV", "SA",
    "U.S", "U.S.A", "U.K", "E.U", "N.A", "St", "Mt",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Jr", "Sr", "Hon",
    "No", "Nos", "vs", "etc", "approx", "est", "Fig", "Sec", "Art", "Ref", "cf",
    "e.g", "i.e", "al", "Ph.D", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Sept", "Oct", "Nov", "Dec",
]
_MARK = "\x00"                                   # placeholder for a protected dot

_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ABBREV) + r")\.", re.I)
_INITIAL_RE = re.compile(r"\b([A-Z])\.")          # single-letter initials: "J. Smith"
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")           # 1.5, 3.4%
_ITEM_RE = re.compile(r"\b(Item|Note|Section|Part)\s+(\d+[A-Za-z]?)\.", re.I)
_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text):
    """Split text into sentences. Dependency-free, tuned for SEC filings."""
    t = _ABBREV_RE.sub(lambda m: m.group(1) + _MARK, text)
    t = _INITIAL_RE.sub(lambda m: m.group(1) + _MARK, t)
    t = _DECIMAL_RE.sub(lambda m: m.group(1) + _MARK + m.group(2), t)
    t = _ITEM_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}{_MARK}", t)

    parts = _SPLIT_RE.split(t)

    out = []
    for p in parts:
        p = p.replace(_MARK, ".").strip()
        if p:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Packing sentences into chunks
# ---------------------------------------------------------------------------

def pack_sentences(sentences, target_words, overlap_words):
    """
    Fill a chunk with whole sentences up to target_words, then start the next
    chunk with enough trailing sentences to cover overlap_words.

    Returns a list of (text, n_sentences, hard_split).
    """
    chunks, cur, cur_words = [], [], 0

    def flush(hard=False):
        if cur:
            chunks.append((" ".join(cur), len(cur), hard))

    i = 0
    while i < len(sentences):
        s = sentences[i]
        w = len(s.split())

        # A single sentence longer than the budget (usually a table row or a
        # run-on list). Emit what we have, then hard-split it by words.
        if w > target_words:
            flush()
            cur, cur_words = [], 0
            words = s.split()
            step = max(1, target_words - overlap_words)
            for j in range(0, len(words), step):
                piece = words[j:j + target_words]
                if piece:
                    chunks.append((" ".join(piece), 1, True))
            i += 1
            continue

        if cur_words + w > target_words and cur:
            flush()
            # carry back whole sentences until the overlap budget is covered
            back, back_words = [], 0
            for prev in reversed(cur):
                if back_words >= overlap_words:
                    break
                back.insert(0, prev)
                back_words += len(prev.split())
            cur, cur_words = list(back), back_words

        cur.append(s)
        cur_words += w
        i += 1

    flush()
    return chunks


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fulltext", "sections"], default="fulltext")
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--corpus-dir", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.corpus_dir)
    text_dir = out / "text_clean"
    manifest_in = out / f"manifest_{a.mode}.csv"

    if not text_dir.exists():
        sys.exit(f"Cannot find {text_dir}. Run build_corpus.py first.")
    if not manifest_in.exists():
        sys.exit(f"Cannot find {manifest_in}. Run build_corpus.py --{'full-text' if a.mode=='fulltext' else ''} first.")

    extract = None
    if a.mode == "sections":
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from build_corpus import extract_sections as extract
        except Exception:
            sys.exit("Need extract_sections from build_corpus.py for --mode sections. "
                     "Put this script next to it.")

    rows = list(csv.DictReader(open(manifest_in, encoding="utf-8")))
    out_path = out / f"chunks_{a.chunk_size}_{a.overlap}_{a.mode}_sentence.jsonl"

    print("=" * 70)
    print(f"SENTENCE RE-CHUNK | mode={a.mode} size={a.chunk_size}w overlap={a.overlap}w")
    print(f"reading {text_dir}   ({len(rows)} manifest rows)")
    print("=" * 70)

    n_chunks = n_docs = n_hard = n_sent = 0
    new_manifest = []
    with open(out_path, "w", encoding="utf-8") as f:
        for k, r in enumerate(rows, 1):
            # triggers are single short documents - copy through unchanged
            if r["source_type"] == "trigger":
                continue
            if r["status"] == "FAILED":
                new_manifest.append([r["doc_id"], r["company"], r["cik"], r["source_type"],
                                     r["form"], r["published_date"], r["accession"], "", 0, 0,
                                     "FAILED", r["url"]])
                continue

            p = text_dir / f"{r['doc_id']}.txt"
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")

            secs = ([("full_document", text)] if a.mode == "fulltext"
                    else extract(text, r["form"])[0])

            doc_chunks = 0
            for sec_name, body in secs:
                sents = split_sentences(body)
                n_sent += len(sents)
                for i, (ct, ns, hard) in enumerate(
                        pack_sentences(sents, a.chunk_size, a.overlap)):
                    f.write(json.dumps({
                        "chunk_id": f"{r['doc_id']}:{sec_name}:s{i:04d}",
                        "doc_id": r["doc_id"],
                        "text": ct,
                        "company": r["company"],
                        "cik": r["cik"],
                        "source_type": r["source_type"],
                        "form": r["form"],
                        "published_date": r["published_date"],
                        "section": sec_name,
                        "relevant_events": [],
                        "url": r["url"],
                        "n_sentences": ns,
                        "hard_split": hard,
                    }, ensure_ascii=False) + "\n")
                    doc_chunks += 1
                    if hard:
                        n_hard += 1
            n_chunks += doc_chunks
            n_docs += 1
            new_manifest.append([r["doc_id"], r["company"], r["cik"], r["source_type"],
                                 r["form"], r["published_date"], r["accession"],
                                 "|".join(s for s, _ in secs), len(text), doc_chunks,
                                 "rechunked", r["url"]])
            if k % 100 == 0:
                print(f"   {k}/{len(rows)}  chunks={n_chunks:,}")

        # triggers, copied straight from the original chunk file
        src = out / f"chunks_{a.chunk_size}_{a.overlap}_{a.mode}.jsonl"
        n_trig = 0
        if src.exists():
            for line in open(src, encoding="utf-8"):
                c = json.loads(line)
                if c.get("source_type") == "trigger":
                    c["n_sentences"] = len(split_sentences(c["text"]))
                    c["hard_split"] = False
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    n_trig += 1
                    n_chunks += 1
        else:
            print(f"   ! {src.name} not found - trigger documents not copied")

    with open(out / f"manifest_{a.mode}_sentence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "company", "cik", "source_type", "form", "published_date",
                    "accession", "sections", "chars", "n_chunks", "status", "url"])
        w.writerows(new_manifest)

    # ---- comparison with the word-based file ------------------------------
    print("\n" + "=" * 70)
    print(f"documents re-chunked : {n_docs:,}")
    print(f"sentences found      : {n_sent:,}")
    print(f"chunks written       : {n_chunks:,}   (incl. {n_trig} triggers)")
    print(f"hard-split chunks    : {n_hard:,}  ({100*n_hard/max(1,n_chunks):.1f}%  - oversized tables/lists)")

    def pct_mid(path, limit=4000):
        mid = tot = 0
        for line in open(path, encoding="utf-8"):
            c = json.loads(line)
            if c.get("source_type") == "trigger":
                continue
            t = c["text"].lstrip()
            if t:
                tot += 1
                if t[0].islower():
                    mid += 1
            if tot >= limit:
                break
        return 100 * mid / max(1, tot)

    if src.exists():
        print(f"\nchunks starting mid-sentence:")
        print(f"   word-based file     : {pct_mid(src):.0f}%")
    print(f"   sentence-based file : {pct_mid(out_path):.0f}%")
    print("=" * 70)
    print(f"\n  {out_path}")
    print(f"  {out / f'manifest_{a.mode}_sentence.csv'}\n")


if __name__ == "__main__":
    main()
