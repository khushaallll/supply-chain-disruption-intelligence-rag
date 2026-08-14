"""
token_usage_analysis.py -- summarizes token usage across a batch of real
System C traces, now that every trace carries prompt_tokens/
completion_tokens/total_tokens/token_usage_log (see the token-usage
instrumentation added to agent_state.py/agent_graph.py/run_event.py).

Produces the same KIND of numbers your literature reference reports for
its own system ("token usage decreases from 30K to 16K") so System C's
own cost profile can be stated in comparable terms.

Run:
    python token_usage_analysis.py results/traces_8
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys

EVIDENCE_TOOLS = {"get_supplier_info", "search_corpus"}


def load_traces(traces_dir: str) -> dict:
    traces = {}
    for path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        event_id = os.path.splitext(os.path.basename(path))[0]
        import json
        with open(path, encoding="utf-8") as f:
            traces[event_id] = json.load(f)
    return traces


def summarize(traces_dir: str) -> list[dict]:
    traces = load_traces(traces_dir)
    rows = []

    for event_id, t in traces.items():
        n_evidenced = sum(
            1 for row in (t.get("coverage") or {}).values()
            if any(s in EVIDENCE_TOOLS for s in (row.get("evidence_sources") or []))
        )
        total = t.get("total_tokens")
        rows.append({
            "event_id": event_id,
            "total_tokens": total,
            "prompt_tokens": t.get("prompt_tokens"),
            "completion_tokens": t.get("completion_tokens"),
            "hop_count": t.get("hop_count"),
            "n_evidenced_companies": n_evidenced,
            "tokens_per_evidenced_company": round(total / n_evidenced, 1)
                if (total and n_evidenced) else None,
        })

    have_totals = [r["total_tokens"] for r in rows if r["total_tokens"] is not None]
    print(f"Events with usable token data: {len(have_totals)} / {len(rows)}")
    if have_totals:
        print(f"  mean total_tokens/event   : {statistics.mean(have_totals):,.0f}")
        print(f"  median total_tokens/event : {statistics.median(have_totals):,.0f}")
        print(f"  min / max                 : {min(have_totals):,} / {max(have_totals):,}")
        print(f"  total across all events   : {sum(have_totals):,}")

    # Cost by node -- how much goes to deciding what to check ("agent")
    # vs the one wrap-up call at the end ("summarize"). Not a hop-budget
    # split (summarize isn't a hop at all) -- a pure cost split.
    node_totals: dict[str, int] = {}
    for t in traces.values():
        for entry in (t.get("token_usage_log") or []):
            node = entry.get("node", "unknown")
            cost = (entry.get("prompt_tokens") or 0) + (entry.get("completion_tokens") or 0)
            node_totals[node] = node_totals.get(node, 0) + cost
    print("\nToken cost by node:")
    for node, total in sorted(node_totals.items()):
        print(f"  {node}: {total:,}")

    return rows


if __name__ == "__main__":
    traces_dir = sys.argv[1] if len(sys.argv) > 1 else "results/traces_8"
    rows = summarize(traces_dir)

    out_path = "results/system_c/token_usage_per_event.csv"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_path}")
