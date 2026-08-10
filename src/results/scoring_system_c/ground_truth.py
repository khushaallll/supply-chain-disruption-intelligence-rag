"""
ground_truth.py -- loads and indexes the 30-event ground truth (three
batch files) by event_id. No System-C-specific logic here; every scoring
script (System C today, Baselines A/B later) shares this loader.
"""

from __future__ import annotations

import glob
import json
import os


def load_ground_truth(gt_dir: str, pattern: str = "ground_truth_batch*_final.json") -> dict:
    """Returns {event_id: ground_truth_record}. Loads every file matching
    `pattern` in `gt_dir` and merges them. Raises if two files define the
    same event_id with different content-shape surprises would rather be
    loud than silently overwritten -- but duplicate event_ids across
    batch files are not expected (each event lives in exactly one batch),
    so this only raises on an actual collision, not on rerun-safety
    grounds."""
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
            records[eid] = ev
    return records
