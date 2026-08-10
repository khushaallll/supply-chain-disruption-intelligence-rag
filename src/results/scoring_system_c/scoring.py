"""
scoring.py -- the generic recall/precision scorer, deliberately written
to take a plain list of predicted-company-name strings rather than
anything System-C-shaped. This is the piece meant to be reused, unchanged,
when Baselines A and B are built: each just needs its own
`extract_<system>_predictions(raw_output) -> list[str]` function (see
system_c.py's `extract_system_c_predictions` for the pattern), and can
call `score_event()` below with the result.

--------------------------------------------------------------------------
DESIGN DECISION: what counts as a "predicted company" for precision --
stated explicitly before computing anything, per the brief's own
requirement.

For System C specifically (this reasoning is System-C-specific; a future
baseline's extractor may differ and should state its own reasoning in its
own module), the predicted set is NOT "every company in the coverage
dict." It is the subset of coverage rows that have at least one
EVIDENCE_TOOLS source (get_supplier_info or search_corpus with
status=='found'), i.e. rows where the agent actually retrieved and read
primary-source text about the company -- not merely rows that
traverse_supply_graph happened to name as structurally adjacent.

Reasoning: stopping_condition.py's own module docstring draws this exact
line for a reason directly relevant to evaluation, not just to the
stopping logic -- "structural adjacency in the graph is not itself a
grounded claim about impact." A single traverse_supply_graph(max_tier=2)
call can name anywhere from 15 to over 300 companies (day8_9 log §8.1) as
merely "significant" (tier <= 2); treating ALL of those as "the agent's
prediction" would make precision measure the graph's own branching factor
for a given seed company, not anything about the agent's reasoning --
and would make recall trivially inflatable on any event with a
well-connected seed, for the same reason. Requiring an EVIDENCE_TOOLS hit
is what makes "predicted" mean "the agent looked, found something, and is
reporting it as a finding" -- which is also the plain-English reading a
methodology-chapter reader would give the word "predicted."

This same predicted-set definition is used for BOTH recall and precision
(not a looser one for recall and a stricter one for precision), so the
two numbers stay internally consistent with each other -- a company only
ever counts as "found" once, under one consistent bar, everywhere.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from name_resolution import resolve_gt_company


@dataclass
class EventScore:
    event_id: str
    seed_unresolved: bool
    score_for_recall: bool
    score_for_precision: bool
    scoring_bucket: Optional[str]

    # Recall bookkeeping (all = every gt company; reachable = gt company's
    # own graph_hop <= 2, i.e. within what the agent could structurally
    # ever reach -- see brief requirement 2)
    gt_all_n: int = 0
    tp_all: int = 0
    gt_reachable_n: int = 0
    tp_reachable: int = 0

    # Precision bookkeeping
    predicted_n: int = 0
    tp_precision: int = 0
    fp_precision: int = 0

    # Audit trail -- every gt company's resolution outcome, and every
    # predicted company that went unmatched (the false positives, named).
    gt_matches: list = field(default_factory=list)     # list of dicts
    unmatched_predicted: list = field(default_factory=list)  # list of names


def score_event(
    event_id: str,
    predicted_names: list,
    gt_record: dict,
    phonebook: dict,
    seed_unresolved: bool,
) -> EventScore:
    """predicted_names: lowercased strings, the system's own predicted-
    affected-company set for this event (see module docstring for what
    that means for System C specifically). gt_record: this event's full
    ground-truth record (one element of ground_truth.load_ground_truth()'s
    values). phonebook: name_resolution.load_phonebook()'s output."""

    predicted_keys = {p.strip().lower() for p in predicted_names if p and p.strip()}
    gt_all = gt_record.get("ground_truth_affected", []) or []
    gt_reachable = [c for c in gt_all if c.get("graph_hop") is not None and c["graph_hop"] <= 2]

    score = EventScore(
        event_id=event_id,
        seed_unresolved=seed_unresolved,
        score_for_recall=bool(gt_record.get("score_for_recall")),
        score_for_precision=bool(gt_record.get("score_for_precision")),
        scoring_bucket=gt_record.get("scoring_bucket"),
        gt_all_n=len(gt_all),
        gt_reachable_n=len(gt_reachable),
        predicted_n=len(predicted_keys),
    )

    if seed_unresolved:
        # Nothing to score -- see brief requirement 4. Leave every
        # numerator/denominator at 0 rather than computing a misleading
        # 0/0-shaped "0% recall" for an event that never got a real
        # chance to find anything. The caller keeps this event out of
        # both the recall and precision aggregates entirely, on its own
        # explicit accounting line.
        return score

    reachable_ids = {id(c) for c in gt_reachable}
    matched_predicted_keys = set()

    for gt_entry in gt_all:
        result = resolve_gt_company(gt_entry, predicted_keys, phonebook)
        hit = result.matched_key is not None
        if hit:
            score.tp_all += 1
            if id(gt_entry) in reachable_ids:
                score.tp_reachable += 1
            matched_predicted_keys.add(result.matched_key)
        score.gt_matches.append({
            "company": gt_entry.get("company"),
            "graph_node": gt_entry.get("graph_node"),
            "graph_hop": gt_entry.get("graph_hop"),
            "reachable": id(gt_entry) in reachable_ids,
            "matched": hit,
            "matched_key": result.matched_key,
            "method": result.method,
        })

    score.tp_precision = len(matched_predicted_keys)
    score.fp_precision = len(predicted_keys - matched_predicted_keys)
    score.unmatched_predicted = sorted(predicted_keys - matched_predicted_keys)

    return score


# --------------------------------------------------------------------------- #
# Aggregation across events
# --------------------------------------------------------------------------- #

def aggregate(scores: list) -> dict:
    """Micro-averaged (pooled-count) aggregates, matching the convention
    already established by this project's own Baseline B evaluation
    (single overall recall_all / recall_reachable percentages, not a mean
    of per-event percentages). Macro (mean-of-per-event) numbers are
    included alongside as a secondary diagnostic, since micro and macro
    can tell different stories when event sizes vary as widely as they do
    here (gt_all_n ranges from 0 to 11 across the 30 events)."""

    seed_unresolved = [s for s in scores if s.seed_unresolved]
    recall_eligible = [s for s in scores if s.score_for_recall and not s.seed_unresolved]
    precision_eligible = [s for s in scores if s.score_for_precision and not s.seed_unresolved]
    precision_scored = [s for s in precision_eligible if s.predicted_n > 0]

    def _ratio(num, den):
        return (num / den) if den else None

    recall_all_num = sum(s.tp_all for s in recall_eligible)
    recall_all_den = sum(s.gt_all_n for s in recall_eligible)
    recall_reachable_num = sum(s.tp_reachable for s in recall_eligible)
    recall_reachable_den = sum(s.gt_reachable_n for s in recall_eligible)

    precision_num = sum(s.tp_precision for s in precision_scored)
    precision_den = sum(s.tp_precision + s.fp_precision for s in precision_scored)

    per_event_recall_all = [
        _ratio(s.tp_all, s.gt_all_n) for s in recall_eligible if s.gt_all_n > 0
    ]
    per_event_recall_reachable = [
        _ratio(s.tp_reachable, s.gt_reachable_n) for s in recall_eligible if s.gt_reachable_n > 0
    ]
    per_event_precision = [
        _ratio(s.tp_precision, s.tp_precision + s.fp_precision) for s in precision_scored
    ]

    return {
        "n_events_scored": len(scores),
        "n_seed_unresolved": len(seed_unresolved),
        "seed_unresolved_event_ids": [s.event_id for s in seed_unresolved],

        "recall": {
            "n_recall_eligible_events": len(recall_eligible),
            "n_recall_eligible_excluding_seed_unresolved": len(recall_eligible),
            "recall_all_micro": _ratio(recall_all_num, recall_all_den),
            "recall_all_tp": recall_all_num,
            "recall_all_denominator": recall_all_den,
            "recall_all_macro_mean_per_event": (
                sum(per_event_recall_all) / len(per_event_recall_all)
                if per_event_recall_all else None
            ),
            "recall_reachable_micro": _ratio(recall_reachable_num, recall_reachable_den),
            "recall_reachable_tp": recall_reachable_num,
            "recall_reachable_denominator": recall_reachable_den,
            "recall_reachable_macro_mean_per_event": (
                sum(per_event_recall_reachable) / len(per_event_recall_reachable)
                if per_event_recall_reachable else None
            ),
        },

        "precision": {
            "n_precision_eligible_events": len(precision_eligible),
            "n_events_with_zero_predictions": len(precision_eligible) - len(precision_scored),
            "n_events_scored_for_precision": len(precision_scored),
            "precision_micro": _ratio(precision_num, precision_den),
            "precision_tp": precision_num,
            "precision_fp": precision_den - precision_num,
            "precision_denominator": precision_den,
            "avg_fp_per_scored_event": (
                (precision_den - precision_num) / len(precision_scored)
                if precision_scored else None
            ),
            "precision_macro_mean_per_event": (
                sum(per_event_precision) / len(per_event_precision)
                if per_event_precision else None
            ),
        },
    }
