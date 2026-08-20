"""
disclosure_lag.py -- RQ1: disclosure lag analysis.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from event_loader import load_events, Event

RESULTS_DIR = Path("results")


def _parse(d: str) -> datetime:
    return datetime.strptime(d[:10], "%Y-%m-%d")


def compute_lag_rows(events: list[Event]) -> list[dict]:
    rows = []
    for ev in events:
        if ev.excluded:
            continue
        if not ev.date_news_first or not ev.date_first_disclosure:
            continue
        try:
            news_dt = _parse(ev.date_news_first)
            disc_dt = _parse(ev.date_first_disclosure)
        except ValueError as exc:
            print(f"  [disclosure_lag] could not parse a date for "
                  f"{ev.event_id}: {exc} -- skipped")
            continue

        lag_days = (disc_dt - news_dt).days
        if lag_days < 0:
            print(f"  [disclosure_lag] WARNING: {ev.event_id} has a "
                  f"NEGATIVE lag ({lag_days} days) -- disclosure date is "
                  f"before the news date. Check date_first_disclosure and "
                  f"date_news_first for this event by hand before using "
                  f"it; not auto-corrected here since guessing which one "
                  f"is wrong would be worse than flagging it.")

        rows.append({
            "event_id": ev.event_id,
            "category": ev.category,
            "date_news_first": ev.date_news_first,
            "date_first_disclosure": ev.date_first_disclosure,
            "date_first_sec_filing": ev.date_first_sec_filing,
            "lag_days": lag_days,
            "disclosure_confidence": ev.disclosure_confidence,
        })
    return rows


def summarize(rows: list[dict], min_confidence: set[str]) -> dict:
    subset = [r for r in rows if r["disclosure_confidence"] in min_confidence]
    lags = [r["lag_days"] for r in subset]
    if not lags:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(lags),
        "median": statistics.median(lags),
        "mean": round(statistics.mean(lags), 1),
        "min": min(lags),
        "max": max(lags),
    }


def summarize_by_category(rows: list[dict], min_confidence: set[str]) -> dict:
    categories = sorted({r["category"] for r in rows})
    out = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        out[cat] = summarize(cat_rows, min_confidence)
    return out


def main():
    events = load_events()
    rows = compute_lag_rows(events)

    print(f"{len(rows)} / {len(events)} events have a usable disclosure lag "
          f"(a non-null date_first_disclosure, event not excluded).\n")

    print("=== Overall lag, by confidence threshold ===")
    for label, confs in [
        ("high only", {"high"}),
        ("high + medium", {"high", "medium"}),
        ("all (incl. low)", {"high", "medium", "low", None}),
    ]:
        s = summarize(rows, confs)
        if s["n"] == 0:
            print(f"  {label:16s} n=0")
        else:
            print(f"  {label:16s} n={s['n']:2d}  median={s['median']:.0f} days  "
                  f"mean={s['mean']:.1f} days  range=[{s['min']}, {s['max']}]")
    print()

    print("=== Lag by category (high-confidence only) ===")
    print("*** Events_Limitation.md 1.2 already documented that this table "
          "will be sparse and uneven -- do not report a median for any row "
          "with n<3. ***\n")
    by_cat_high = summarize_by_category(rows, {"high"})
    for cat, s in by_cat_high.items():
        if s["n"] == 0:
            print(f"  {cat:22s} n=0")
        elif s["n"] < 3:
            print(f"  {cat:22s} n={s['n']}  (too few to report a median -- "
                  f"list individually instead: "
                  f"{[r['lag_days'] for r in rows if r['category'] == cat and r['disclosure_confidence'] == 'high']})")
        else:
            print(f"  {cat:22s} n={s['n']:2d}  median={s['median']:.0f} days  "
                  f"mean={s['mean']:.1f} days")
    print()

    print("=== Per-event detail (high confidence) ===")
    for r in sorted(rows, key=lambda r: r["lag_days"]):
        if r["disclosure_confidence"] != "high":
            continue
        print(f"  {r['event_id']:45s} {r['category']:20s} "
              f"lag={r['lag_days']:4d}d  "
              f"({r['date_news_first']} -> {r['date_first_disclosure']})")

    # --- write outputs ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "disclosure_lag_per_event.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["event_id", "category", "date_news_first",
                      "date_first_disclosure", "date_first_sec_filing",
                      "lag_days", "disclosure_confidence"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {
        "n_events_total": len(events),
        "n_events_with_lag": len(rows),
        "overall": {
            "high_only": summarize(rows, {"high"}),
            "high_and_medium": summarize(rows, {"high", "medium"}),
            "all": summarize(rows, {"high", "medium", "low", None}),
        },
        "by_category_high_only": by_cat_high,
        "caveat": (
            "High-confidence disclosure dates are not evenly distributed "
            "across event categories (Events_Limitation.md 1.2). Report "
            "medians only for categories with n>=3 at your chosen "
            "confidence threshold; list individual cases otherwise. Frame "
            "RQ1 as a case-based finding with primary-source citations per "
            "event, not as a statistical distribution -- the underlying "
            "sample (30 events, ~12 with any disclosure date) is too small "
            "to support confidence intervals (Events_Limitation.md 7.1)."
        ),
    }
    with open(RESULTS_DIR / "disclosure_lag_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWritten: {RESULTS_DIR}/disclosure_lag_per_event.csv")
    print(f"Written: {RESULTS_DIR}/disclosure_lag_summary.json")


if __name__ == "__main__":
    main()
