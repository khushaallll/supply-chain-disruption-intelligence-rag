"""
event_loader.py -- shared ground-truth loading for the Day 12-13 evaluation
scripts (evaluate.py, disclosure_lag.py). This is the ONE place that:

  1. reads the three ground_truth_batch{1,2,3}_final.json files
  2. applies the manual exclusion list for events 20, 36, 49 -- ON TOP OF,
     not instead of, each event's own built-in score_for_recall /
     score_for_precision flags (see EXCLUDED_EVENTS docstring below: a
     direct check of your real files shows the automatic flags do NOT
     already cover all three cases)
  3. derives each event's seed company from its event_id
  4. attaches a best-effort event category for the per-category
     breakdowns the dissertation wants (see CATEGORY_MAP -- READ THE
     WARNING ON THAT DICT, it is inferred, not read from an authoritative
     field, because no category field exists in the JSON as uploaded)

Verified directly against your four uploaded ground_truth_batch*_final.json
files before writing this (not guessed):
  - 30 events total, 10 per batch
  - top-level fields per event: event_id, date_news_first,
    date_news_first_note, date_first_disclosure, date_first_sec_filing,
    confidence (nested: ground_truth_affected / date_first_disclosure,
    each "high"/"medium"/"low"), scoring_bucket (A/B/C/D),
    score_for_recall, score_for_precision, ground_truth_affected (list)
  - scoring_bucket counts: A=10 (empty ground truth, restraint test),
    B=5 (graph reaches + corpus has doc), C=7 (graph reaches, no corpus
    doc), D=8 (graph cannot reach within 2 hops)
  - score_for_recall: True=12, False=18 (BEFORE the manual exclusions
    below); score_for_precision: True=30 (all events, before exclusions)
  - bundled company entries (e.g. "Aquila Resources, Cockatoo Coal and
    Ensham Resources") are ALREADY split into individual dicts in this
    "_final" version, each carrying a `split_from` field showing the
    original bundled string -- so, unlike the Day 4-5 log's open item,
    no bundle-splitting code is needed here. Verified by direct search
    across all 30 events for any remaining comma-joined `company` value:
    none found.

NOT executed against your real files in this environment. Run this
directly (`python event_loader.py`) against your real
data/ground_truth/*.json first and read the printed summary before
trusting evaluate.py or disclosure_lag.py's output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


GROUND_TRUTH_PATHS = [
    "data/ground_truth/ground_truth_batch1_final.json",
    "data/ground_truth/ground_truth_batch2_final.json",
    "data/ground_truth/ground_truth_batch3_final.json",
]


# ---------------------------------------------------------------------------
# The three manual exclusions -- see Events_Limitation.md Section 5.2-5.4
# and the conversation that decided these.
#
# These are deliberately layered ON TOP OF the built-in score_for_recall /
# score_for_precision flags already in the JSON, not a replacement for
# them, because checking the real files directly shows the automatic
# flags do not already handle all three:
#
#   - event 20 (formosa_petrochemical_2019): score_for_recall is
#     currently TRUE in the file as uploaded. But the event's own
#     seed_attribution_warning field says the wrong seed company was
#     recorded (the ARO-3 unit that burned is operated by Formosa
#     Chemicals & Fibre Corp, not Formosa Petrochemical Corp as
#     directly_affected states), and Events_Limitation.md 5.2 says this
#     event is "already marked unusable for scoring." The prose note and
#     the machine-readable flag disagree -- this override wins.
#   - event 36 (basf_se_2022): score_for_recall is already False (bucket
#     A, empty ground_truth_affected), but score_for_precision is still
#     True. Left as-is, a system that correctly flags a company also
#     genuinely affected by the near-identical event 35 (e.g. Alcoa,
#     ArcelorMittal, both appear in event 35's real ground truth) would
#     be scored as a FALSE POSITIVE here purely because event 36's answer
#     key is empty -- an artifact of one shock being split into two rows,
#     not a real system error. Excluded entirely rather than partially.
#   - event 49 (xiamen_tungsten_2010): whether the underlying embargo
#     happened at all is disputed in the literature (Events_Limitation.md
#     5.4; the ground truth's own note calls it "contested rather than
#     empty"). Excluded from all quantitative scoring; write it up
#     separately as a qualitative case study instead.
#
# This dict is deliberately the single, dated, documented place this
# decision lives -- quote it directly in the methodology chapter appendix.
# ---------------------------------------------------------------------------

EXCLUDED_EVENTS: dict[str, str] = {
    "20_formosa_petrochemical_2019": (
        "Wrong seed company recorded: the ARO-3 unit that burned is "
        "operated by Formosa Chemicals & Fibre Corp (FCFC), not Formosa "
        "Petrochemical Corp as the event's directly-affected field states. "
        "FCFC also appears in this event's own hop1 candidate list, so "
        "scoring it as a downstream hit would score the direct victim as "
        "if it were a correct prediction. See this event's own "
        "seed_attribution_warning field."
    ),
    "36_basf_se_2022": (
        "Same underlying shock as event 35 (2022 European gas crisis); "
        "event 36's own date_news_first_note explicitly flags the overlap "
        "and calls event 36 'arguably a downstream consequence' of event "
        "35. Scoring both risks inflating recall if both carried positive "
        "examples, and unfairly penalises precision here specifically, "
        "since this row's ground truth is empty."
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


# ---------------------------------------------------------------------------
# Event category map.
#
# *** READ THIS BEFORE TRUSTING THE PER-CATEGORY NUMBERS ***
# No category field exists anywhere in the ground_truth_batch*_final.json
# files as uploaded (checked directly -- every top-level and nested key
# was enumerated). The categories below are INFERRED from each event's
# real-world subject matter, not read from an authoritative source. Most
# are unambiguous (an earthquake is a natural disaster). A handful are
# marked [CONFIRM] because I'm genuinely not certain -- check these
# against whatever source you originally used to plan the 8/8/7/7 balance
# mentioned in the implementation plan, if one exists as a file, and swap
# it in here instead of trusting my guesses.
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "1_posco_2022": "natural_disaster",            # Typhoon Hinnamnor flooding, Pohang
    "2_nippon_steel_2011": "natural_disaster",     # Tohoku earthquake/tsunami
    "4_dow_2017": "natural_disaster",              # Hurricane Harvey
    "5_formosa_plastics_2021": "natural_disaster", # Texas Winter Storm Uri
    "8_exxon_mobil_2005": "natural_disaster",      # Hurricane Katrina
    "9_xinxiang_tianli_energy_2021": "natural_disaster",  # [CONFIRM] Henan/Zhengzhou flooding, 2021
    "10_hesteel_2023": "natural_disaster",         # [CONFIRM] Hebei 2023 -- likely Typhoon Doksuri flooding
    "11_aurizon_2010": "natural_disaster",         # Queensland floods
    "17_basf_se_2016": "industrial_accident",      # Ludwigshafen pipeline explosion
    "18_chevron_2012": "industrial_accident",      # Richmond refinery fire
    "19_cpc_corp_taiwan_2014": "industrial_accident",  # [CONFIRM] Kaohsiung gas pipeline explosion
    "20_formosa_petrochemical_2019": "industrial_accident",  # Mailiao ARO-3 fire (excluded from scoring anyway)
    "21_mitsubishi_materials_2014": "industrial_accident",   # [CONFIRM]
    "23_s_oil_2022": "industrial_accident",        # [CONFIRM] Onsan refinery incident
    "24_bp_2010": "industrial_accident",           # Deepwater Horizon
    "25_petrochina_2005": "industrial_accident",   # Jilin petrochemical explosion
    "33_kyushu_electric_power_2018": "energy",     # solar curtailment
    "34_korea_electric_power_2011": "energy",      # rolling blackout
    "35_gazprom_pjsc_2022": "geopolitical_policy", # Nord Stream 1 halt
    "36_basf_se_2022": "energy",                   # ammonia curtailment (excluded from scoring anyway)
    "38_perusahaan_perseroan_persero_pt_perusahaan_listrik_negara_2019": "energy",  # Java-Bali blackout
    "40_state_grid_corp_of_china_2021": "energy",  # China power crunch
    "41_yunnan_aluminium_2022": "energy",          # drought-driven power rationing
    "47_china_steel_2018": "geopolitical_policy",  # Section 232 tariffs
    "48_glencore_2025": "uncategorized",           # [CONFIRM] genuinely unknown to me -- check this one directly
    "49_xiamen_tungsten_2010": "geopolitical_policy",  # rare-earth dispute (excluded from scoring anyway)
    "50_aneka_tambang_tbk_2020": "geopolitical_policy",  # Indonesia nickel export ban
    "54_yunnan_chihong_zinc_germanium_2023": "geopolitical_policy",  # China gallium/germanium export controls
    "57_united_co_rusal_international_pjsc_2018": "geopolitical_policy",  # US sanctions on Rusal
    "59_coronado_global_resources_2020": "geopolitical_policy",  # China ban on Australian coal
}


def get_category(event_id: str) -> str:
    return CATEGORY_MAP.get(event_id, "uncategorized")


# ---------------------------------------------------------------------------
# Seed-company derivation from event_id.
#
# Verified, not guessed: your own test_graph_tool.py's EVENT_SEED_GUESSES
# dict already establishes this exact convention --
#   "2_nippon_steel_2011"  -> "nippon steel"
#   "1_posco_2022"         -> "posco"
#   "11_aurizon_2010"      -> "aurizon"
#   "35_gazprom_pjsc_2022" -> "gazprom pjsc"
#   "47_china_steel_2018"  -> "china steel"
# and event 20's own seed_attribution_warning field independently confirms
# it again: event_id "20_formosa_petrochemical_2019" pairs with a
# directly_affected value of "formosa petrochemical". The rule: strip the
# leading numeric event id and a trailing 4-digit year, join what's left
# with spaces.
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


# ---------------------------------------------------------------------------
# Manual seed-name overrides.
#
# seed_company_from_event_id() is a best-effort GUESS reconstructed from
# the event_id string, and event_ids can't represent every character a
# real company name contains -- slashes, commas, periods, and ampersands
# all get lost when the event_id was built. Confirmed wrong so far:
#
#   event 19 (19_cpc_corp_taiwan_2014): the real company is
#   "CPC Corp/Taiwan" (Taiwan's state oil refiner, formally "CPC
#   Corporation, Taiwan") -- the naive derivation guesses "cpc corp
#   taiwan" (space), losing the slash entirely.
#
# Add any further corrections you find while spot-checking the rest
# against find_company_node() here, in this one dict -- not as a scattered
# fix inside baseline_b.py or anywhere else that calls
# seed_company_from_event_id().
# ---------------------------------------------------------------------------

SEED_OVERRIDES: dict[str, str] = {
    "19_cpc_corp_taiwan_2014": "cpc corp/taiwan",
}


def seed_company_from_event_id(event_id: str) -> str:
    if event_id in SEED_OVERRIDES:
        return SEED_OVERRIDES[event_id]
    parts = event_id.split("_")
    if len(parts) < 2:
        return event_id
    start = 1 if parts[0].isdigit() else 0
    end = len(parts)
    if _YEAR_RE.match(parts[-1]):
        end -= 1
    return " ".join(parts[start:end])


# ---------------------------------------------------------------------------
# Event contract
# ---------------------------------------------------------------------------

@dataclass
class AffectedCompany:
    company: str
    graph_node: Optional[str]
    in_graph_context: object   # "hop1" | "hop2" | False
    has_corpus_doc: bool


@dataclass
class Event:
    event_id: str
    category: str
    seed_company: str

    excluded: bool
    exclusion_reason: Optional[str]

    scoring_bucket: str
    score_for_recall: bool     # AFTER applying the manual exclusion override
    score_for_precision: bool  # AFTER applying the manual exclusion override
    raw_score_for_recall: bool     # BEFORE the override -- kept for auditing
    raw_score_for_precision: bool  # BEFORE the override

    date_news_first: Optional[str]
    date_first_disclosure: Optional[str]
    date_first_sec_filing: Optional[str]
    disclosure_confidence: Optional[str]
    affected_confidence: Optional[str]

    ground_truth_affected: list[AffectedCompany] = field(default_factory=list)

    @property
    def n_affected_total(self) -> int:
        """Recall_all denominator -- every recorded affected company,
        whether or not it resolved to a graph node."""
        return len(self.ground_truth_affected)

    @property
    def ground_truth_all_graph_nodes(self) -> set[str]:
        return {c.graph_node for c in self.ground_truth_affected if c.graph_node}

    @property
    def reachable_graph_nodes(self) -> set[str]:
        """Recall_reachable numerator/denominator pool -- companies tagged
        hop1 or hop2. Matches the Day 4-5 log's recall_all vs
        recall_reachable decomposition (D4.2/D4.3): only hop1/hop2 are
        treated as 'in graph context', by deliberate, documented choice
        (hop3 already reaches ~35% of the whole graph, hop4+ reaches
        81-99% -- greater distances carry no discriminative signal)."""
        return {
            c.graph_node for c in self.ground_truth_affected
            if c.graph_node and c.in_graph_context in ("hop1", "hop2")
        }

    @property
    def n_reachable(self) -> int:
        return len(self.reachable_graph_nodes)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw(paths: list[str]) -> list[dict]:
    events = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  [event_loader] NOT FOUND: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            events.extend(json.load(f))
    return events


def load_events(paths: list[str] = GROUND_TRUTH_PATHS) -> list[Event]:
    raw_events = _load_raw(paths)
    if not raw_events:
        raise RuntimeError(
            f"Loaded ZERO events from {paths} -- refusing to proceed "
            f"silently. Check the paths above against your real repo "
            f"layout before re-running. A silently-empty event list would "
            f"make evaluate.py report perfect (undefined/0-of-0) scores "
            f"for every system, which looks like success, not failure."
        )

    events: list[Event] = []
    for ev in raw_events:
        event_id = ev["event_id"]
        excluded = event_id in EXCLUDED_EVENTS
        exclusion_reason = EXCLUDED_EVENTS.get(event_id)

        raw_recall = bool(ev.get("score_for_recall", False))
        raw_precision = bool(ev.get("score_for_precision", False))

        affected = [
            AffectedCompany(
                company=c.get("company", ""),
                graph_node=c.get("graph_node"),
                in_graph_context=c.get("in_graph_context"),
                has_corpus_doc=bool(c.get("has_corpus_doc", False)),
            )
            for c in ev.get("ground_truth_affected", [])
        ]

        confidence = ev.get("confidence", {}) or {}

        events.append(Event(
            event_id=event_id,
            category=get_category(event_id),
            seed_company=seed_company_from_event_id(event_id),
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            scoring_bucket=ev.get("scoring_bucket", ""),
            score_for_recall=(raw_recall and not excluded),
            score_for_precision=(raw_precision and not excluded),
            raw_score_for_recall=raw_recall,
            raw_score_for_precision=raw_precision,
            date_news_first=ev.get("date_news_first"),
            date_first_disclosure=ev.get("date_first_disclosure"),
            date_first_sec_filing=ev.get("date_first_sec_filing"),
            disclosure_confidence=confidence.get("date_first_disclosure"),
            affected_confidence=confidence.get("ground_truth_affected"),
            ground_truth_affected=affected,
        ))

    return events


# ---------------------------------------------------------------------------
# Standalone check -- run this FIRST, before evaluate.py or
# disclosure_lag.py, and read the output.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    events = load_events()

    print(f"Loaded {len(events)} events.\n")

    print("=== Exclusion check ===")
    for e in events:
        if e.excluded:
            print(f"  EXCLUDED: {e.event_id}")
            print(f"    reason: {e.exclusion_reason}")
            print(f"    (raw flags before override: recall={e.raw_score_for_recall}, "
                  f"precision={e.raw_score_for_precision})")

    print(f"\n=== Category coverage ===")
    from collections import Counter
    cat_counts = Counter(e.category for e in events)
    for cat, n in sorted(cat_counts.items()):
        flag = "  <-- check CATEGORY_MAP" if cat == "uncategorized" else ""
        print(f"  {cat:20s} {n:2d}{flag}")

    print(f"\n=== Seed-company derivation spot check ===")
    for e in events[:8]:
        tag = " (manual override)" if e.event_id in SEED_OVERRIDES else " (guessed)"
        print(f"  {e.event_id:45s} -> seed: '{e.seed_company}'{tag}")
    print(f"  {len(SEED_OVERRIDES)} manual override(s) currently on record: "
          f"{list(SEED_OVERRIDES.keys())}")
    print("  Every other seed above is still a GUESS from the event_id -- "
          "verify each one resolves correctly via GraphStore.find_company_node "
          "before trusting baseline_b.py's output, and add any wrong ones to "
          "SEED_OVERRIDES the same way event 19 was fixed.")

    print(f"\n=== Scoring pool sizes AFTER exclusions ===")
    n_recall = sum(1 for e in events if e.score_for_recall)
    n_precision = sum(1 for e in events if e.score_for_precision)
    print(f"  events scored for recall:    {n_recall} / {len(events)}")
    print(f"  events scored for precision: {n_precision} / {len(events)}")

    print(f"\n=== Disclosure-date availability ===")
    with_date = [e for e in events if e.date_first_disclosure and not e.excluded]
    print(f"  events with a disclosure date (post-exclusion): {len(with_date)}")
    from collections import Counter as C2
    print("  by confidence:", C2(e.disclosure_confidence for e in with_date))