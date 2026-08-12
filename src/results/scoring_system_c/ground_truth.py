"""
ground_truth.py -- loads and indexes the 30-event ground truth (three
batch files) by event_id. No System-C-specific logic here; every scoring
script (System C today, Baselines A/B later) shares this loader.

--------------------------------------------------------------------------
REVISION: applies the same 3-event manual exclusion override that
event_loader.py (the Baseline A/B pipeline) already applies, so every
system is scored over the identical event pool.

Found by direct comparison: event_loader.py's EXCLUDED_EVENTS overrides
score_for_recall and score_for_precision to False for exactly 3 events
(20_formosa_petrochemical_2019 -- wrong seed company recorded;
36_basf_se_2022 -- same underlying shock as event 35, double-counts it;
49_xiamen_tungsten_2010 -- disputed/contested event), layered ON TOP of
the ground truth's own raw flags, not a replacement for them (see
event_loader.py's own comment block for the full per-event reasoning --
copied verbatim below so this file doesn't silently drift out of sync
with it). This loader never applied that same override -- every scoring
script built on top of it (scoring.py, system_c.py,
evaluate_system_c.py) reads score_for_recall/score_for_precision
straight from the raw ground-truth JSON, with no exclusion layer at all.
Confirmed concretely: without this fix, System C's own evaluation scores
20_formosa_petrochemical_2019 (recall-eligible under the raw flag),
contributing a guaranteed-wrong recall attempt and 8 pure false
positives that Baseline A/B's numbers never carry, since Baseline A/B
excludes this event entirely.

This dict is deliberately copied verbatim, not imported from
event_loader.py -- ground_truth.py has no dependency on any baseline
file, by design (see this module's own opening line: "no
System-C-specific logic here"), and the reverse dependency (importing
System C's loader into the baseline pipeline) would be equally wrong.
Two small, verbatim copies in two independent files is the accepted
tradeoff here, NOT an oversight -- but it does mean: if this exclusion
list is ever revised, both copies (this one and event_loader.py's) must
be updated together, or the two pipelines will silently drift apart
again exactly as they did before this fix.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import glob
import json
import os

# Verbatim copy of event_loader.py's EXCLUDED_EVENTS -- see this module's
# docstring for why this is a copy, not a shared import, and for the
# maintenance obligation that comes with that choice.
EXCLUDED_EVENTS: dict[str, str] = {
    "20_formosa_petrochemical_2019": (
        "Wrong seed company recorded: the ARO-3 unit that burned is "
        "operated by Formosa Chemicals & Fibre Corp (FCFC), not Formosa "
        "Petrochemical Corp as the event's directly-affected field "
        "states. FCFC also appears in this event's own hop1 candidate "
        "list, so scoring it as a downstream hit would score the direct "
        "victim as if it were a correct prediction. See this event's own "
        "seed_attribution_warning field."
    ),
    "36_basf_se_2022": (
        "Same underlying shock as event 35 (2022 European gas crisis); "
        "event 36's own date_news_first_note explicitly flags the "
        "overlap and calls event 36 'arguably a downstream consequence' "
        "of event 35. Scoring both risks inflating recall if both "
        "carried positive examples, and unfairly penalises precision "
        "here specifically, since this row's ground truth is empty."
    ),
    "49_xiamen_tungsten_2010": (
        "Whether a targeted rare-earth embargo actually occurred is "
        "disputed in the peer-reviewed and policy literature (Japanese "
        "customs data shows no uniform import drop attributable to it). "
        "Excluded from quantitative recall/precision; report separately "
        "as a qualitative case study of how the system handles a widely "
        "reported but contested event."
    ),
}


def load_ground_truth(gt_dir: str, pattern: str = "ground_truth_batch*_final.json") -> dict:
    """Returns {event_id: ground_truth_record}. Loads every file matching
    `pattern` in `gt_dir` and merges them. Raises if two files define the
    same event_id with different content-shape surprises would rather be
    loud than silently overwritten -- but duplicate event_ids across
    batch files are not expected (each event lives in exactly one batch),
    so this only raises on an actual collision, not on rerun-safety
    grounds.

    Applies the EXCLUDED_EVENTS override to every loaded record before
    returning: score_for_recall and score_for_precision are ANDed with
    "not excluded", exactly matching event_loader.py's own
    `(raw_recall and not excluded)` logic, so a scoring script built on
    this loader can never see a different recall/precision-eligibility
    verdict than the baseline pipeline does for the same event. The raw,
    pre-override values are preserved under raw_score_for_recall /
    raw_score_for_precision for auditing, matching event_loader.py's own
    Event dataclass fields of the same name; `excluded` and
    `exclusion_reason` are added for the same reason."""
    records: dict = {}
    paths = sorted(glob.glob(os.path.join(gt_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No ground-truth files matching {pattern!r} found in {gt_dir!r}"
        )
    for path in paths:
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        for ev in batch:
            eid = ev["event_id"]
            if eid in records:
                raise ValueError(
                    f"Duplicate event_id {eid!r} found in both a prior batch "
                    f"file and {path!r} -- ground truth files should not overlap."
                )

            excluded = eid in EXCLUDED_EVENTS
            raw_recall = bool(ev.get("score_for_recall", False))
            raw_precision = bool(ev.get("score_for_precision", False))

            ev["excluded"] = excluded
            ev["exclusion_reason"] = EXCLUDED_EVENTS.get(eid)
            ev["raw_score_for_recall"] = raw_recall
            ev["raw_score_for_precision"] = raw_precision
            ev["score_for_recall"] = raw_recall and not excluded
            ev["score_for_precision"] = raw_precision and not excluded

            records[eid] = ev
    return records