"""
tier_depth_sweep.py -- Evaluation #2: what happens to Baseline B as you
let it search further from the disruption?

This is the graph-only preview of the implementation plan's "System C
capped at Tier 1-4" experiment -- same question (does precision collapse
with depth?), asked of the plain graph traversal instead of the agent,
because the agent doesn't exist yet and this does. When System C's own
version of this experiment runs later, this table is the baseline it
should be compared against: how much of the precision collapse is just
"the graph fans out" versus how much the agent's reasoning actually holds
back?

Reuses run_baseline_b_for_all_events() and metrics.score_event/aggregate()
completely unchanged -- this file contains zero new scoring logic, only a
loop over max_tier. If evaluate.py's numbers are trustworthy, this sweep's
tier-1 row will exactly match evaluate.py's Baseline B row; that's a
built-in consistency check, not a coincidence.

Files used: event_loader.py, metrics.py, graph_tool.py, baseline_b.py.

NOT executed against your real graph in this environment. Run:
    python tier_depth_sweep.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from event_loader import load_events
from metrics import score_event, aggregate, fmt_pct
from agent.tools.graph_tool import GraphStore, GRAPH_PATH
from baseline_b import run_baseline_b_for_all_events

RESULTS_DIR = Path("results")
TIERS_TO_SWEEP = [1, 2, 3, 4]


def main():
    events = load_events()
    store = GraphStore(GRAPH_PATH)
    print(f"Loaded {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges\n")

    sweep_rows = []
    per_event_rows = []

    for tier in TIERS_TO_SWEEP:
        print(f"Running Baseline B at max_tier={tier}...")
        b_results = run_baseline_b_for_all_events(store, events, max_tier=tier)

        rows = []
        for ev in events:
            predicted = b_results[ev.event_id].predicted_companies
            row = score_event(ev, predicted)
            row["n_predicted_raw"] = len(predicted)  # before TP/FP split, for the "graph fan-out" story
            rows.append(row)
            per_event_rows.append({"max_tier": tier, **row})

        agg = aggregate(rows)
        n_precision_events = agg["n_events_precision"]
        avg_predicted = (sum(r["n_predicted_raw"] for r in rows if r["score_for_precision"])
                          / n_precision_events) if n_precision_events else None

        sweep_rows.append({
            "max_tier": tier,
            "recall_all": agg["recall_all"],
            "recall_all_tp": agg["recall_all_tp"],
            "recall_all_denom": agg["recall_all_denom"],
            "precision": agg["precision"],
            "fp_total": agg["fp_total"],
            "avg_fp_per_event": agg["avg_fp_per_event"],
            "avg_companies_predicted_per_event": round(avg_predicted, 1) if avg_predicted else None,
        })

    print("\n=== Baseline B, swept across tier depth ===\n")
    header = (f"{'tier':6s} {'recall_all':12s} {'precision':11s} {'fp_total':10s} "
              f"{'avg_fp/ev':10s} {'avg_predicted/ev':18s}")
    print(header)
    print("-" * len(header))
    for r in sweep_rows:
        avg_fp = "n/a" if r["avg_fp_per_event"] is None else f"{r['avg_fp_per_event']:.2f}"
        avg_pred = "n/a" if r["avg_companies_predicted_per_event"] is None else f"{r['avg_companies_predicted_per_event']:.1f}"
        print(f"{r['max_tier']:<6d} {fmt_pct(r['recall_all']):12s} "
              f"{fmt_pct(r['precision']):11s} {r['fp_total']:<10d} {avg_fp:10s} {avg_pred:18s}")

    if len(sweep_rows) >= 2:
        first, last = sweep_rows[0], sweep_rows[-1]
        if first["precision"] and last["precision"] is not None:
            print(f"\n  Precision at tier {first['max_tier']}: {fmt_pct(first['precision'])}  "
                  f"-> tier {last['max_tier']}: {fmt_pct(last['precision'])}")
        if first["recall_all"] is not None and last["recall_all"] is not None:
            print(f"  recall_all at tier {first['max_tier']}: {fmt_pct(first['recall_all'])}  "
                  f"-> tier {last['max_tier']}: {fmt_pct(last['recall_all'])}")
        print("  Expect recall_all to rise and precision to fall as tier increases --"
              " if precision does NOT collapse noticeably, that's worth double-checking"
              " rather than reporting as a clean result; it would be an unusual finding"
              " given how graph fan-out normally behaves.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "tier_depth_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(sweep_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)

    with open(RESULTS_DIR / "tier_depth_sweep_per_event.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["max_tier", "event_id", "category", "score_for_recall",
                      "score_for_precision", "n_predicted_raw", "n_affected_total",
                      "n_affected_reachable", "tp", "tp_reachable", "fp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_event_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    print(f"\nWritten: {RESULTS_DIR}/tier_depth_sweep_summary.csv")
    print(f"Written: {RESULTS_DIR}/tier_depth_sweep_per_event.csv")


if __name__ == "__main__":
    main()
