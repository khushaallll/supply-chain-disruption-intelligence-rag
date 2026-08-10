"""
run_all_events.py -- Day 9: runs System C across all 30 ground-truth events
in one pass, reusing a single built app (see run_event.build_app()) rather
than reloading the graph/corpus/embedding model 30 times.

WHERE THE SEED COMPANY AND DATE COME FROM, and why this is a heuristic:
ground_truth_batch{1,2,3}_final.json do NOT carry a clean, universal
"which company is the disrupted seed" field -- checked directly: only 3 of
30 events tag one explicitly (via an edgar_excerpts entry whose
relation_to_event says "SEED COMPANY"). For the other 27, the seed name is
derived from event_id itself (e.g. '11_aurizon_2010' -> 'aurizon'), the
same heuristic test_graph_tool.py's EVENT_SEED_GUESSES already uses by
hand for a handful of events. This is a GUESS, not a confirmed value --
every guess is checked against the real graph via
graph_store.find_company_node() BEFORE spending an API call on it; if it
doesn't resolve, that event is skipped and logged, not silently run on a
wrong seed. event_date comes from the confirmed 'date_news_first' field.

WHAT ELSE GETS CARRIED THROUGH: each event's real 'scoring_bucket',
'confidence', and score_for_recall/score_for_precision flags (see the Day
9 discussion -- the ground truth already has a difficulty/testability
classification per event, e.g. bucket D events are explicitly marked
"agent cannot succeed within 2 hops, exclude from recall"). These are
written into the summary alongside each run's result specifically so a
later evaluation script (or you, reading results by hand) doesn't have to
re-derive that context -- it's real signal already sitting in your data
that would otherwise get lost between this script's output and the
ground-truth file.

RATE LIMITS AND TRANSIENT GENERATION ERRORS: if you're running this against
a free hosted API (see llm_setup.py), you WILL plausibly hit 429 errors
partway through 30 events -- that's expected, not a bug (see llm_setup.py's
token-budget math). Separately, the model itself can occasionally generate
malformed JSON for a tool call's own arguments (confirmed on a real run --
see run_event_with_retry's _looks_like_transient_generation_error) -- this
is not a code bug either, just an LLM generation slip, and is retried
short rather than with the long rate-limit backoff, since there's nothing
to wait out. Each event gets a small number of retries for either case; an
event that still fails after that is logged as an error and the batch
continues with the next one, rather than the whole run dying partway
through. Note: a retry re-runs the WHOLE event from hop 1, not just the
failed hop -- simpler to reason about, at the cost of redoing any
already-successful hops on a retry.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from run_event import build_app, run_one_event, save_trace

# [CONFIRM: same convention as run_event.py -- adjust to your real layout]
GROUND_TRUTH_PATHS = [
    "data/ground_truth/ground_truth_batch1_final.json",
    "data/ground_truth/ground_truth_batch2_final.json",
    "data/ground_truth/ground_truth_batch3_final.json",
]

TRACE_DIR = Path("results/traces_4")
SUMMARY_PATH = Path("results/run_all_events_summary_4.json")

MAX_RETRIES = 3
# For transient tool-call-generation errors specifically (see
# _looks_like_transient_generation_error): confirmed on a second real run
# that 3 total attempts is not always enough -- '40_state_grid_corp_of_china_2021'
# hit the same malformed-JSON error class as '47_china_steel_2018' and
# exhausted all 3 attempts, while china_steel's identical error class
# recovered within 3 on its own run. This suggests the failure is not
# purely random per-call noise for every query shape -- some queries may
# trigger it closer to deterministically. More attempts is a cheap,
# low-risk way to reduce (not guaranteed to eliminate) this residual
# failure rate without the complexity of varying generation parameters
# between retries.
MAX_GENERATION_ERROR_RETRIES = 5
RETRY_BACKOFF_SECONDS = 20  # doubled each retry -- 20s, 40s, 80s


# --------------------------------------------------------------------------- #
# Loading events + deriving the seed-company guess
# --------------------------------------------------------------------------- #

_EVENT_ID_PATTERN = re.compile(r"^\d+_(.+)_\d{4}$")

# Manual overrides for events where the event_id-derived guess doesn't
# resolve against the real graph. Confirmed directly against
# company_nodes.txt (the real node list), not re-guessed -- same pattern
# test_graph_tool.py's EVENT_SEED_GUESSES already uses for exactly this
# kind of case. Add to this table as more get found; don't hand-edit
# guess_seed_company()'s heuristic to special-case them.
SEED_NAME_OVERRIDES = {
    "23_s_oil_2022": "s-oil",                                             # heuristic guessed 's oil' (space); real node uses a hyphen
    "54_yunnan_chihong_zinc_germanium_2023": "yunnan chihong zinc&germanium",  # real node has no spaces around '&', not "zinc and germanium"
}


def guess_seed_company(event_id: str) -> Optional[str]:
    """'11_aurizon_2010' -> 'aurizon'. Returns None if event_id doesn't
    match the expected '{number}_{slug}_{year}' shape at all -- a
    malformed event_id should be skipped and logged, not guessed at
    further."""
    match = _EVENT_ID_PATTERN.match(event_id)
    if not match:
        return None
    return match.group(1).replace("_", " ")


def load_events(paths: list[str]) -> list[dict]:
    events = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  [load_events] not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            events.extend(json.load(f))
    print(f"  [load_events] loaded {len(events)} event(s)")
    return events


# --------------------------------------------------------------------------- #
# Per-event run, with retry on rate-limit-shaped failures
# --------------------------------------------------------------------------- #

def _looks_like_rate_limit(exc: Exception) -> bool:
    """No guaranteed exception TYPE across providers for a 429 -- string-
    matching the message is unglamorous but is what actually works across
    langchain-groq / langchain-ollama without importing provider-specific
    exception classes here. False positives just mean an unnecessary
    retry, which is harmless; false negatives just mean a real rate limit
    gets logged as a generic error instead of retried, which is also
    survivable, so this doesn't need to be perfect."""
    msg = str(exc).lower()
    return any(term in msg for term in ("429", "rate limit", "rate_limit", "too many requests"))


def _looks_like_transient_generation_error(exc: Exception) -> bool:
    """Confirmed on TWO independent real runs: '47_china_steel_2018' and
    '40_state_grid_corp_of_china_2021' both failed with 'error parsing
    tool call ... invalid character ... looking for beginning of object
    key string' -- Ollama's own JSON parser rejecting tool-call arguments
    the MODEL generated, not a bug in our dispatch code, which never even
    ran here. china_steel recovered within 3 attempts; state_grid did not
    -- evidence this is not purely random per-call noise for every query
    shape, hence the larger MAX_GENERATION_ERROR_RETRIES budget and a
    SHORT retry delay (there is nothing to wait out, unlike a rate
    limit's long backoff)."""
    msg = str(exc).lower()
    return any(term in msg for term in (
        "error parsing tool call", "invalid character", "invalid json",
        "cannot unmarshal", "unexpected end of json",
    ))


def run_event_with_retry(app, company_name: str, event_date: str, event_id: str) -> dict:
    last_exc = None
    max_attempts = max(MAX_RETRIES, MAX_GENERATION_ERROR_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            return run_one_event(app, company_name, event_date, event_id=event_id)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES and _looks_like_rate_limit(exc):
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"    rate-limit-shaped error on attempt {attempt}/{MAX_RETRIES}, "
                      f"waiting {wait}s before retry: {exc}")
                time.sleep(wait)
                continue
            if attempt < MAX_GENERATION_ERROR_RETRIES and _looks_like_transient_generation_error(exc):
                wait = 3
                print(f"    transient tool-call-generation error on attempt "
                      f"{attempt}/{MAX_GENERATION_ERROR_RETRIES}, "
                      f"retrying in {wait}s (nothing to wait out, just trying again): {exc}")
                time.sleep(wait)
                continue
            break
    raise last_exc


# --------------------------------------------------------------------------- #
# The batch run
# --------------------------------------------------------------------------- #

def run_all_events(provider: str = "ollama", model: Optional[str] = None,
                    max_hops: int = 4) -> list[dict]:
    events = load_events(GROUND_TRUTH_PATHS)
    if not events:
        raise RuntimeError("No ground-truth events loaded -- check GROUND_TRUTH_PATHS.")

    print("Building app (loads the real graph, corpus, and embedding model -- "
          "this happens ONCE for all 30 events, not once per event)...")
    app, graph_store = build_app(provider=provider, model=model)
    print("App built.\n")

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for i, ev in enumerate(events, start=1):
        event_id = ev["event_id"]
        event_date = ev.get("date_news_first")
        seed_guess = SEED_NAME_OVERRIDES.get(event_id) or guess_seed_company(event_id)

        print(f"[{i}/{len(events)}] {event_id}")

        if event_date is None:
            print("    SKIPPED -- no date_news_first on this event record")
            summary.append({"event_id": event_id, "status": "skipped_no_date"})
            continue

        if seed_guess is None:
            print(f"    SKIPPED -- event_id doesn't match the expected naming pattern")
            summary.append({"event_id": event_id, "status": "skipped_bad_event_id"})
            continue

        if event_id in SEED_NAME_OVERRIDES:
            print(f"    using manual seed override: '{seed_guess}' (confirmed against company_nodes.txt)")

        resolved_seed = graph_store.find_company_node(seed_guess)
        if resolved_seed is None:
            print(f"    SKIPPED -- guessed seed '{seed_guess}' did not resolve to any "
                  f"graph node; needs a manual seed name, same as "
                  f"test_graph_tool.py's EVENT_SEED_GUESSES handles by hand")
            summary.append({"event_id": event_id, "status": "skipped_unresolved_seed",
                             "seed_guess": seed_guess})
            continue

        print(f"    seed resolved: '{seed_guess}' -> '{resolved_seed}' | date: {event_date}")

        try:
            result = run_event_with_retry(app, resolved_seed, event_date, event_id)
        except Exception as exc:
            print(f"    ERROR -- {exc}")
            summary.append({"event_id": event_id, "status": "error", "error": str(exc),
                             "seed_used": resolved_seed})
            continue

        save_trace(result, TRACE_DIR / f"{event_id}.json")
        print(f"    done -- stop_reason={result['stop_reason']} | "
              f"hops={result['hop_count']}/{result['max_hops']} | "
              f"gaps={len(result['gap_flags'])}")

        summary.append({
            "event_id": event_id,
            "status": "completed",
            "seed_used": resolved_seed,
            "event_date": event_date,
            "stop_reason": result["stop_reason"],
            "hop_count": result["hop_count"],
            "n_gap_flags": len(result["gap_flags"]),
            "n_companies_in_coverage": len(result["coverage"]),
            # carried through from the ground-truth file itself -- real,
            # pre-existing testability signal, not derived by this script
            "scoring_bucket": ev.get("scoring_bucket"),
            "score_for_recall": ev.get("score_for_recall"),
            "score_for_precision": ev.get("score_for_precision"),
            "confidence": ev.get("confidence"),
        })

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {SUMMARY_PATH}")

    completed = sum(1 for s in summary if s["status"] == "completed")
    skipped = sum(1 for s in summary if s["status"].startswith("skipped"))
    errored = sum(1 for s in summary if s["status"] == "error")
    print(f"{completed} completed, {skipped} skipped, {errored} errored, "
          f"out of {len(events)} total events")

    return summary


if __name__ == "__main__":
    run_all_events()