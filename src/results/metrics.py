"""
metrics.py -- pure recall/precision/false-positive scoring functions.

No networkx/rapidfuzz/geopy/chromadb imports here on purpose, same reason
search_corpus_tool.py defers its heavy imports to inside __init__: this
lets test_evaluation_logic.py exercise the actual scoring math in an
environment that doesn't have your real graph/corpus libraries installed
at all, and lets you hand-verify the numbers against the real
ground-truth JSON without loading anything expensive.

Metric definitions used throughout (state these explicitly in the
methodology chapter, exact wording below):

  recall_all       = TP / (every recorded affected company, whether or
                      not it has a graph_node) -- summed only over events
                      where score_for_recall is True.
  recall_reachable = TP / (affected companies tagged hop1 or hop2 in the
                      ground truth) -- same event pool. This decomposition
                      is not new -- it's the Day 4-5 log's D4.2/D4.3
                      decision, carried over unchanged: recall_all is
                      bounded above by how much of the real world your
                      graph covers at all (~30% per Events_Limitation.md
                      4.1); recall_reachable asks the fairer question of
                      how well the SYSTEM does on the subset it could
                      possibly reach.
  precision         = TP / (TP + FP) -- summed only over events where
                      score_for_precision is True.
  false positives   = reported as a raw total AND as an average per
                      scored event, NOT as a classical rate with a fixed
                      negative-class denominator. There is no natural
                      fixed universe of "companies that were definitely
                      NOT affected" to divide by in this open, thousands-
                      of-nodes graph -- picking one (e.g. "every node in
                      the graph") would be an arbitrary modelling choice
                      dressed up as a rate. Total and per-event-average FP
                      counts are the honest, defensible version of the
                      "false positive rate" the implementation plan asks
                      for; say so explicitly wherever this number appears.
"""

from __future__ import annotations


def score_event(event, predicted: set[str]) -> dict:
    """event: an event_loader.Event. predicted: set of graph-node-style
    company name strings (lowercase), from either baseline."""
    predicted = {p.lower().strip() for p in predicted}
    gt_all = event.ground_truth_all_graph_nodes
    gt_reachable = event.reachable_graph_nodes

    tp = predicted & gt_all
    tp_reachable = predicted & gt_reachable
    fp = predicted - gt_all

    return {
        "event_id": event.event_id,
        "category": event.category,
        "scoring_bucket": event.scoring_bucket,
        "score_for_recall": event.score_for_recall,
        "score_for_precision": event.score_for_precision,
        "n_predicted": len(predicted),
        "n_affected_total": event.n_affected_total,
        "n_affected_reachable": event.n_reachable,
        "tp": len(tp),
        "tp_reachable": len(tp_reachable),
        "fp": len(fp),
        "tp_names": sorted(tp),
        "fp_names": sorted(fp),
        "missed_names": sorted(gt_all - predicted),
    }


def aggregate(rows: list[dict]) -> dict:
    """rows: output of score_event(), already filtered to whichever group
    you want a summary for (overall, or one category) -- this function
    itself does the score_for_recall/score_for_precision filtering, so
    pass it every row in the group and it'll pick the right subset for
    each metric."""
    recall_rows = [r for r in rows if r["score_for_recall"]]
    precision_rows = [r for r in rows if r["score_for_precision"]]

    tp_sum = sum(r["tp"] for r in recall_rows)
    denom_all = sum(r["n_affected_total"] for r in recall_rows)
    recall_all = (tp_sum / denom_all) if denom_all else None

    tp_reach_sum = sum(r["tp_reachable"] for r in recall_rows)
    denom_reach = sum(r["n_affected_reachable"] for r in recall_rows)
    recall_reachable = (tp_reach_sum / denom_reach) if denom_reach else None

    tp_p_sum = sum(r["tp"] for r in precision_rows)
    fp_sum = sum(r["fp"] for r in precision_rows)
    precision = (tp_p_sum / (tp_p_sum + fp_sum)) if (tp_p_sum + fp_sum) else None
    avg_fp = (fp_sum / len(precision_rows)) if precision_rows else None

    return {
        "n_events_total": len(rows),
        "n_events_recall": len(recall_rows),
        "n_events_precision": len(precision_rows),
        "recall_all": recall_all,
        "recall_all_tp": tp_sum,
        "recall_all_denom": denom_all,
        "recall_reachable": recall_reachable,
        "recall_reachable_tp": tp_reach_sum,
        "recall_reachable_denom": denom_reach,
        "precision": precision,
        "precision_tp": tp_p_sum,
        "fp_total": fp_sum,
        "avg_fp_per_event": avg_fp,
    }


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"
