"""
evaluate.py -- Day 12-13 deliverable, the part that doesn't need the agent.

Runs Baseline A (keyword) and Baseline B (single-hop graph) against all
scoreable events, and produces the numbers your dissertation needs:
recall_all, recall_reachable, precision, and false-positive counts --
overall and broken down by event category -- for each system.

System C slots in later with ZERO changes to metrics.py or event_loader.py:
once agent.py exists, write one more function that returns
{event_id: set_of_predicted_company_names} the same shape run_baseline_a()
and run_baseline_b_for_all_events() already return, add "C" to the
SYSTEMS dict below, and re-run this file. Every table this script produces
already has a spare "system" column waiting for it.

Usage:
    python event_loader.py     # run this FIRST, read the diagnostic output
    python baseline_b.py       # sanity-check Baseline B alone
    python baseline_a.py       # sanity-check Baseline A alone (fix the
                                # trigger-doc loader first if it says
                                # "0 trigger documents found")
    python evaluate.py         # this file -- the real scoring run

Outputs (written to ./results/):
    per_event_scores.csv       one row per (event, system) -- the rawest
                                form of the numbers, for appendix tables
    summary_overall.csv        recall/precision/FP per system, overall
    summary_by_category.csv    recall/precision/FP per system, per category
    evaluation_summary.json    same numbers as summary_*.csv, machine-
                                readable, for quoting exact figures in text

NOT executed against your real files in this environment -- I don't have
your graph_enriched_corrected.pkl, your corpus, or a live Python
environment with networkx/rapidfuzz/geopy installed. Run this on your
machine and read the printed tables plus the CSVs before trusting any
number from it in the dissertation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from event_loader import load_events, Event
from metrics import score_event, aggregate, fmt_pct
from agent.tools.graph_tool import GraphStore, GRAPH_PATH
from baseline_a import run_baseline_a, load_trigger_texts
from baseline_b import run_baseline_b_for_all_events

RESULTS_DIR = Path("results")


def run_all_systems(events: list[Event]) -> dict[str, dict[str, set]]:
    """Returns {system_name: {event_id: set_of_predicted_names}}."""
    print("Loading supply graph...")
    store = GraphStore(GRAPH_PATH)
    print(f"  {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges\n")

    print("Running Baseline B (single-hop graph, max_tier=1)...")
    b_results = run_baseline_b_for_all_events(store, events, max_tier=1)
    b_predictions = {eid: r.predicted_companies for eid, r in b_results.items()}
    n_not_resolved = sum(1 for r in b_results.values() if r.status == "not_resolved")
    print(f"  {n_not_resolved} / {len(events)} seed companies did not resolve "
          f"in the graph at all (zero predictions for those events, not an "
          f"error -- verify against find_company_node directly if this "
          f"number looks high)\n")

    print("Running Baseline A (keyword co-occurrence)...")
    known_names = list(store.G.nodes())
    trigger_texts = load_trigger_texts()
    if not trigger_texts:
        print("  WARNING: no trigger texts loaded -- Baseline A will "
              "predict nothing for every event until you fix "
              "baseline_a.load_trigger_texts() (see that file's module "
              "docstring). Continuing anyway so Baseline B's numbers "
              "still come out of this run.")
    a_results = run_baseline_a(events, known_names, trigger_texts)
    a_predictions = {eid: r.predicted_companies for eid, r in a_results.items()}
    print()

    # System C: once agent.py exists, add a third entry here, e.g.
    #   c_predictions = run_system_c_for_all_events(events)
    #   return {"A": a_predictions, "B": b_predictions, "C": c_predictions}
    return {"A": a_predictions, "B": b_predictions}


def score_all(events: list[Event], predictions: dict[str, dict[str, set]]
              ) -> list[dict]:
    """Returns a flat list of per-(event, system) score rows, EXCLUDED
    events included but flagged with score_for_recall/score_for_precision
    already False (from event_loader's override), so they naturally drop
    out of every aggregate() call without any special-casing here."""
    rows = []
    for system, preds_by_event in predictions.items():
        for ev in events:
            predicted = preds_by_event.get(ev.event_id, set())
            row = score_event(ev, predicted)
            row["system"] = system
            rows.append(row)
    return rows


def build_summary_tables(rows: list[dict]) -> tuple[dict, dict]:
    """Returns (overall_by_system, by_category_by_system)."""
    systems = sorted({r["system"] for r in rows})
    categories = sorted({r["category"] for r in rows})

    overall = {}
    for system in systems:
        system_rows = [r for r in rows if r["system"] == system]
        overall[system] = aggregate(system_rows)

    by_category = defaultdict(dict)
    for system in systems:
        for category in categories:
            group_rows = [r for r in rows
                          if r["system"] == system and r["category"] == category]
            by_category[category][system] = aggregate(group_rows)

    return overall, dict(by_category)


def print_overall_table(overall: dict):
    print("=== Overall results, per system ===\n")
    header = (f"{'system':8s} {'recall_all':12s} {'recall_reach':14s} "
              f"{'precision':11s} {'fp_total':10s} {'avg_fp/ev':10s} "
              f"{'n_recall_ev':12s} {'n_prec_ev':10s}")
    print(header)
    print("-" * len(header))
    for system, s in sorted(overall.items()):
        avg_fp = "n/a" if s["avg_fp_per_event"] is None else f"{s['avg_fp_per_event']:.2f}"
        print(f"{system:8s} {fmt_pct(s['recall_all']):12s} "
              f"{fmt_pct(s['recall_reachable']):14s} "
              f"{fmt_pct(s['precision']):11s} {s['fp_total']:<10d} "
              f"{avg_fp:10s} {s['n_events_recall']:<12d} {s['n_events_precision']:<10d}")
    print()


def print_category_table(by_category: dict):
    print("=== Results by event category, per system ===\n")
    for category in sorted(by_category.keys()):
        print(f"--- {category} ---")
        for system, s in sorted(by_category[category].items()):
            if s["n_events_total"] == 0:
                continue
            avg_fp = "n/a" if s["avg_fp_per_event"] is None else f"{s['avg_fp_per_event']:.2f}"
            print(f"  {system:6s} recall_all={fmt_pct(s['recall_all']):8s} "
                  f"recall_reachable={fmt_pct(s['recall_reachable']):8s} "
                  f"precision={fmt_pct(s['precision']):8s} "
                  f"fp_total={s['fp_total']:<3d} avg_fp/ev={avg_fp:5s} "
                  f"(n_recall={s['n_events_recall']}, n_precision={s['n_events_precision']})")
        print()


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def main():
    events = load_events()
    n_excluded = sum(1 for e in events if e.excluded)
    print(f"Loaded {len(events)} events ({n_excluded} excluded from scoring "
          f"-- see event_loader.EXCLUDED_EVENTS).\n")

    predictions = run_all_systems(events)
    rows = score_all(events, predictions)

    overall, by_category = build_summary_tables(rows)

    print_overall_table(overall)
    print_category_table(by_category)

    # --- per-event CSV, appendix-table grade detail ---
    per_event_fields = [
        "event_id", "category", "system", "scoring_bucket",
        "score_for_recall", "score_for_precision",
        "n_predicted", "n_affected_total", "n_affected_reachable",
        "tp", "tp_reachable", "fp",
    ]
    write_csv(rows, RESULTS_DIR / "per_event_scores.csv", per_event_fields)

    # --- also dump full names (TP/FP/missed) alongside, for failure analysis ---
    detail_fields = per_event_fields + ["tp_names", "fp_names", "missed_names"]
    write_csv(rows, RESULTS_DIR / "per_event_scores_with_names.csv", detail_fields)

    # --- overall summary CSV ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary_overall.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["system"] + list(next(iter(overall.values())).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for system, s in sorted(overall.items()):
            w.writerow({"system": system, **s})

    # --- by-category summary CSV ---
    with open(RESULTS_DIR / "summary_by_category.csv", "w", newline="", encoding="utf-8") as f:
        sample = next(iter(next(iter(by_category.values())).values()))
        fieldnames = ["category", "system"] + list(sample.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for category, per_system in sorted(by_category.items()):
            for system, s in sorted(per_system.items()):
                w.writerow({"category": category, "system": system, **s})

    # --- everything, machine-readable, for quoting exact figures in text ---
    with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_events_total": len(events),
            "n_events_excluded": n_excluded,
            "excluded_events": {e.event_id: e.exclusion_reason
                                 for e in events if e.excluded},
            "overall": overall,
            "by_category": by_category,
        }, f, indent=2, default=str)

    print(f"Written: {RESULTS_DIR}/per_event_scores.csv")
    print(f"Written: {RESULTS_DIR}/per_event_scores_with_names.csv")
    print(f"Written: {RESULTS_DIR}/summary_overall.csv")
    print(f"Written: {RESULTS_DIR}/summary_by_category.csv")
    print(f"Written: {RESULTS_DIR}/evaluation_summary.json")


if __name__ == "__main__":
    main()
