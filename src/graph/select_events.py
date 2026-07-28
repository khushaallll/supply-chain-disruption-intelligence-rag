"""
select_events.py

Day 2, final step: pick 30 events out of the 60 candidates, split 8/8/7/7
across categories (Natural disaster / Industrial accident / Energy-utility /
Geopolitical-policy), following two rules:

  1. Prefer stronger events first (higher downstream_customer_count).
  2. Skip an event if it overlaps too heavily with something already picked
     in the same category -- overlap is measured on two_hop_oem_reach, since
     two events sharing almost the same set of downstream companies test the
     same part of the graph twice.

Certain events can be marked "protected" (e.g. Kumamoto, since it's already
your flagship worked example) -- protected events are always kept regardless
of rank or overlap.

Usage:
  python select_events.py events_candidates.json \
      --targets "Natural disaster:8,Industrial accident:8,Energy / utility:7,Geopolitical / policy:7" \
      --protect 16 \
      --overlap-threshold 0.5 \
      --output events_selected_30.json
"""

import argparse
import json


def load_events(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def overlap_ratio(event_a: dict, event_b: dict) -> float:
    """
    How much do two events' 2-hop OEM reach overlap, as a fraction of the
    SMALLER event's list?

    Using the smaller list as the denominator (rather than a plain union/
    intersection ratio) answers the question we actually care about: "if I
    already have event A, does event B add anything new, or is B basically
    a subset of A?" A small event that's 90% contained inside a big event's
    reach is redundant, even if the big event's total size makes the overlap
    look small the other way round.
    """
    set_a = set(event_a.get("two_hop_oem_reach", []))
    set_b = set(event_b.get("two_hop_oem_reach", []))

    smaller = min(len(set_a), len(set_b))
    if smaller == 0:
        return 0.0  # nothing to overlap on; treat as fully distinct

    intersection = set_a & set_b
    return len(intersection) / smaller


def select_events(
    events: list,
    category_targets: dict,
    protected_ids: set,
    overlap_threshold: float,
) -> list:
    """
    Runs the ranking + overlap-dedup selection, one category at a time.
    """
    # Group candidates by category
    by_category = {}
    for e in events:
        by_category.setdefault(e["category"], []).append(e)

    selected = []

    for category, target_count in category_targets.items():
        candidates = by_category.get(category, [])

        # Protected events go in first, regardless of score or overlap
        protected = [e for e in candidates if e["id"] in protected_ids]
        remaining = [e for e in candidates if e["id"] not in protected_ids]

        # Sort remaining by strength, strongest first
        remaining.sort(key=lambda e: e["downstream_customer_count"], reverse=True)

        chosen = list(protected)

        for candidate in remaining:
            if len(chosen) >= target_count:
                break

            # Check overlap against everything already chosen in this category
            too_similar = any(
                overlap_ratio(candidate, picked) > overlap_threshold
                for picked in chosen
            )

            if not too_similar:
                chosen.append(candidate)

        # If overlap-skipping left us short of the target, backfill with the
        # next-strongest events regardless of overlap, rather than silently
        # under-filling the category.
        if len(chosen) < target_count:
            chosen_ids = {e["id"] for e in chosen}
            for candidate in remaining:
                if len(chosen) >= target_count:
                    break
                if candidate["id"] not in chosen_ids:
                    chosen.append(candidate)
                    chosen_ids.add(candidate["id"])

        selected.extend(chosen[:target_count])

        print(
            f"{category}: selected {len(chosen[:target_count])} / target {target_count} "
            f"(from {len(candidates)} candidates)"
        )

    return selected


def parse_targets(targets_str: str) -> dict:
    """Parses '--targets' CLI string like 'CategoryA:8,CategoryB:7' into a dict."""
    targets = {}
    for pair in targets_str.split(","):
        name, count = pair.rsplit(":", 1)
        targets[name.strip()] = int(count)
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select final 30 events from 60 candidates.")
    parser.add_argument("events_json", help="Path to events_candidates.json (60 events)")
    parser.add_argument(
        "--targets",
        default=(
            "Natural disaster:8,"
            "Industrial accident:8,"
            "Energy / utility:7,"
            "Geopolitical / policy:7"
        ),
        help="Comma-separated Category:count pairs.",
    )
    parser.add_argument(
        "--protect",
        default="",
        help="Comma-separated event IDs to always include regardless of rank/overlap, e.g. '16,33'",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Skip a candidate if its overlap with an already-picked event in the same "
        "category exceeds this fraction (default 0.5 = 50%%).",
    )
    parser.add_argument("--output", default="events_selected_30.json")

    args = parser.parse_args()

    events = load_events(args.events_json)
    category_targets = parse_targets(args.targets)
    protected_ids = {int(x) for x in args.protect.split(",") if x.strip()}

    selected = select_events(events, category_targets, protected_ids, args.overlap_threshold)

    with open(args.output, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nTotal selected: {len(selected)}")
    print(f"Written to {args.output}") 