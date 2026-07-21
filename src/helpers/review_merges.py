"""Interactive terminal reviewer for data/review/merge_candidates.csv.

Walks candidate pairs score-descending, one keypress per decision
(y=approve, n=reject, s=skip/decide later, q=quit). Decisions are keyed by
(company_a, company_b) and the full decisions file is rewritten after every
single decision, so re-deciding a pair updates its existing row instead of
appending a duplicate. Re-running resumes: pairs already approved or
rejected are skipped; pairs previously skipped are shown again.
"""

import csv
from pathlib import Path

import pandas as pd

CANDIDATES_CSV = Path("data/review/merge_candidates.csv")
DECISIONS_CSV = Path("data/review/merge_decisions.csv")
DECISIONS_HEADER = ["company_a", "company_b", "similarity_score", "decision", "canonical_name"]

# "skip" is a deferral, not a final answer, so it does not block a pair from
# being re-shown on the next run — only approve/reject do.
FINAL_DECISIONS = {"approve", "reject"}


def read_key():
    """Single keypress, no Enter required on Windows (msvcrt); falls back
    to line input if msvcrt isn't available."""
    try:
        import msvcrt
    except ImportError:
        raw = input().strip().lower()
        return raw[0] if raw else ""

    while True:
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            msvcrt.getch()  # discard second byte of arrow/function keys
            continue
        try:
            key = ch.decode("utf-8").lower()
        except UnicodeDecodeError:
            continue
        if key:
            print(key)
            return key


def load_decisions():
    """One record per (company_a, company_b), keyed by file order so a
    later row (e.g. a skip later upgraded to approve) wins — guards against
    stale duplicate rows already on disk even before this run starts."""
    if not DECISIONS_CSV.exists():
        return {}
    df = pd.read_csv(DECISIONS_CSV, dtype=str)
    decisions = {}
    for row in df.itertuples(index=False):
        decisions[(row.company_a, row.company_b)] = {
            "similarity_score": row.similarity_score,
            "decision": row.decision,
            "canonical_name": row.canonical_name if pd.notna(row.canonical_name) else "",
        }
    return decisions


def save_decisions(decisions):
    """Rewrite the full decisions file from the in-memory dict. Called after
    every single decision so a pair always has exactly one row — re-deciding
    it updates that row in place rather than appending a duplicate."""
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DECISIONS_HEADER)
        for (company_a, company_b), rec in decisions.items():
            writer.writerow([company_a, company_b, rec["similarity_score"], rec["decision"], rec["canonical_name"]])


def prompt_canonical(company_a, company_b, count_a, count_b):
    default = company_a if count_a >= count_b else company_b
    typed = input(f'    Canonical name [default: "{default}"] (Enter to accept, or type override): ').strip()
    return typed if typed else default


def review(df):
    decisions = load_decisions()
    total = len(df)

    print("Commands: y=approve  n=reject  s=skip/decide later  q=quit\n")

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            key = (row.company_a, row.company_b)
            if decisions.get(key, {}).get("decision") in FINAL_DECISIONS:
                continue

            print(f"[{i}/{total}] score: {row.similarity_score:.1f}")
            print(f'A: "{row.company_a}"'.ljust(24) + f"({row.count_a} relationships)")
            print(f'B: "{row.company_b}"'.ljust(24) + f"({row.count_b} relationships)")

            while True:
                print("> ", end="", flush=True)
                choice = read_key()
                if choice in ("y", "n", "s", "q"):
                    break
                print("Invalid key, press y/n/s/q.")

            if choice == "q":
                break

            score = float(row.similarity_score)
            if choice == "y":
                canonical = prompt_canonical(row.company_a, row.company_b, row.count_a, row.count_b)
                decisions[key] = {"similarity_score": score, "decision": "approve", "canonical_name": canonical}
            elif choice == "n":
                decisions[key] = {"similarity_score": score, "decision": "reject", "canonical_name": ""}
            else:  # "s"
                decisions[key] = {"similarity_score": score, "decision": "skip", "canonical_name": ""}

            save_decisions(decisions)
            print()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    approved = sum(1 for r in decisions.values() if r["decision"] == "approve")
    rejected = sum(1 for r in decisions.values() if r["decision"] == "reject")
    skipped = sum(1 for r in decisions.values() if r["decision"] == "skip")
    remaining = total - approved - rejected - skipped
    print("=== Review summary ===")
    print(f"Total pairs: {total}")
    print(f"Approved:    {approved}")
    print(f"Rejected:    {rejected}")
    print(f"Skipped:     {skipped}")
    print(f"Remaining:   {remaining}")
    print(f"Decisions saved to: {DECISIONS_CSV}")


def main():
    df = pd.read_csv(CANDIDATES_CSV)
    df = df.sort_values("similarity_score", ascending=False, ignore_index=True)
    review(df)


if __name__ == "__main__":
    main()
