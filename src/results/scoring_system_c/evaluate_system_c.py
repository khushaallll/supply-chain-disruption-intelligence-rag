#!/usr/bin/env python3
"""
evaluate_system_c.py -- scores System C's real 30-event trace batch
against ground truth.

All paths are set as constants immediately below (edit these to match
your local layout), so this can just be run with no arguments:

    python evaluate_system_c.py

CLI flags are still accepted and override the constants below if given,
in case you ever want to point this at a second folder (e.g. a re-run)
without editing the file -- but for normal use, editing the constants is
the intended path.

Loops the traces directory generically (every *.json file whose stem
matches a ground-truth event_id); does not assume a fixed count or a
fixed set of filenames, so this doesn't need editing when pointed at the
real 30-file folder, beyond setting TRACES_DIR itself once below.

See scoring.py's module docstring for the precision-denominator design
decision, and system_c.py for the System-C-specific "predicted company"
extraction it implements that decision with.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from ground_truth import load_ground_truth
from name_resolution import load_phonebook
from scoring import aggregate, score_event
from system_c import extract_system_c_predictions, is_seed_unresolved


# --------------------------------------------------------------------------- #
# LOCAL PATHS -- edit these four to match your machine, then just run
# `python evaluate_system_c.py` with no arguments.
# --------------------------------------------------------------------------- #

TRACES_DIR = "results/traces_7"                 # the 30 real System C trace JSON files
GT_DIR = "data/ground_truth"                                   # directory containing ground_truth_batch{1,2,3}_final.json
PHONEBOOK_PATH = "data/company_phonebook.csv"
OUT_PATH = "results/system_c/system_c_eval_7.json"        # set to None to skip writing a file and only print the report


def load_traces(traces_dir: str) -> dict:
    """Returns {event_id: trace_dict} for every *.json file in
    traces_dir. event_id is taken from the filename stem, matching this
    project's existing convention (e.g. 11_aurizon_2010.json ->
    event_id '11_aurizon_2010') -- the same convention run_all_events.py
    already uses for per-event output files."""
    traces = {}
    for path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            traces[stem] = json.load(f)
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-dir", default=TRACES_DIR, help="Directory of per-event System C trace JSON files")
    ap.add_argument("--gt-dir", default=GT_DIR, help="Directory containing ground_truth_batch*_final.json")
    ap.add_argument("--phonebook", default=PHONEBOOK_PATH, help="Path to company_phonebook.csv")
    ap.add_argument("--out", default=OUT_PATH, help="Where to write the full JSON results (pass '' to skip writing a file)")
    args = ap.parse_args()
    out_path = args.out or None

    ground_truth = load_ground_truth(args.gt_dir)
    phonebook = load_phonebook(args.phonebook)
    traces = load_traces(args.traces_dir)

    if not traces:
        print(f"No trace files found in {args.traces_dir!r}. Check TRACES_DIR at the "
              f"top of this file (or pass --traces-dir).", file=sys.stderr)
        sys.exit(1)

    matched_event_ids = set(traces) & set(ground_truth)
    trace_only = set(traces) - set(ground_truth)
    gt_only = set(ground_truth) - set(traces)
    if trace_only:
        print(f"WARNING: {len(trace_only)} trace file(s) have no matching ground-truth "
              f"event_id, skipped: {sorted(trace_only)}", file=sys.stderr)
    if gt_only:
        print(f"WARNING: {len(gt_only)} ground-truth event(s) have no matching trace "
              f"file, skipped: {sorted(gt_only)}", file=sys.stderr)

    scores = []
    for event_id in sorted(matched_event_ids):
        trace = traces[event_id]
        gt_record = ground_truth[event_id]
        seed_unresolved = is_seed_unresolved(trace)
        predicted = extract_system_c_predictions(trace) if not seed_unresolved else []
        scores.append(score_event(
            event_id=event_id,
            predicted_names=predicted,
            gt_record=gt_record,
            phonebook=phonebook,
            seed_unresolved=seed_unresolved,
        ))

    agg = aggregate(scores)

    results = {
        "n_ground_truth_events": len(ground_truth),
        "n_traces_loaded": len(traces),
        "n_events_scored": len(scores),
        "trace_files_without_ground_truth": sorted(trace_only),
        "ground_truth_events_without_trace": sorted(gt_only),
        "aggregate": agg,
        "per_event": [
            {
                "event_id": s.event_id,
                "seed_unresolved": s.seed_unresolved,
                "scoring_bucket": s.scoring_bucket,
                "score_for_recall": s.score_for_recall,
                "score_for_precision": s.score_for_precision,
                "gt_all_n": s.gt_all_n,
                "tp_all": s.tp_all,
                "gt_reachable_n": s.gt_reachable_n,
                "tp_reachable": s.tp_reachable,
                "predicted_n": s.predicted_n,
                "tp_precision": s.tp_precision,
                "fp_precision": s.fp_precision,
                "unmatched_predicted": s.unmatched_predicted,
                "gt_matches": s.gt_matches,
            }
            for s in scores
        ],
    }

    _print_report(results)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {out_path}")


def _print_report(results: dict) -> None:
    agg = results["aggregate"]
    r, p = agg["recall"], agg["precision"]

    def pct(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    print("=" * 72)
    print("SYSTEM C -- EVALUATION REPORT")
    print("=" * 72)
    print(f"Ground-truth events: {results['n_ground_truth_events']}  |  "
          f"Traces loaded: {results['n_traces_loaded']}  |  "
          f"Events scored: {results['n_events_scored']}")

    print(f"\n--- Seed-resolution failures (own category, excluded from recall/precision) ---")
    print(f"  {agg['n_seed_unresolved']} event(s): {agg['seed_unresolved_event_ids']}")

    print(f"\n--- Recall (score_for_recall == true events only, seed-unresolved excluded) ---")
    print(f"  Eligible events: {r['n_recall_eligible_events']}")
    print(f"  recall_all        : {pct(r['recall_all_micro'])}  "
          f"({r['recall_all_tp']}/{r['recall_all_denominator']})  "
          f"[macro mean: {pct(r['recall_all_macro_mean_per_event'])}]")
    print(f"  recall_reachable  : {pct(r['recall_reachable_micro'])}  "
          f"({r['recall_reachable_tp']}/{r['recall_reachable_denominator']})  "
          f"[macro mean: {pct(r['recall_reachable_macro_mean_per_event'])}]")

    print(f"\n--- Precision (score_for_precision == true events only, seed-unresolved excluded) ---")
    print(f"  Eligible events: {p['n_precision_eligible_events']}  "
          f"| scored (predicted_n > 0): {p['n_events_scored_for_precision']}  "
          f"| zero-prediction events: {p['n_events_with_zero_predictions']}")
    print(f"  precision         : {pct(p['precision_micro'])}  "
          f"({p['precision_tp']}/{p['precision_denominator']})  "
          f"[macro mean: {pct(p['precision_macro_mean_per_event'])}]")
    print(f"  total FP: {p['precision_fp']}  |  avg FP per scored event: "
          f"{p['avg_fp_per_scored_event']:.2f}" if p['avg_fp_per_scored_event'] is not None
          else "  total FP: 0")
    print("=" * 72)


if __name__ == "__main__":
    main()