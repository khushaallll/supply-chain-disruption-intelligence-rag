# Limitations of the Ground-Truth Dataset

**Dataset:** 30 historical supply-chain disruption events
**Files:** `ground_truth_batch1.json`, `ground_truth_batch2.json`, `ground_truth_batch3.json`
**Contents:** 30 events · ~80 confirmed affected companies · 248 source documents · 12 disclosure dates
**Revision:** v2, 2026-07-27 — updated after the systematic SEC disclosure sweep
(`enumerate_disclosures.py`: 47 companies, 1,036 filings, 2,074 documents opened, 0 unresolved).
Items closed or downgraded by that sweep are marked **[RESOLVED]** or **[REDUCED]**.

---

## How to read this document

This is a complete, honest account of everything wrong with the dataset, written in plain language.

It is long on purpose. The point is not to make the dataset look bad — it is to make sure that when someone asks you a hard question in a viva, you have already thought about it and written the answer down.

Every limitation below has three parts:

1. **What it is** — plain English
2. **What it does to your results** — the specific number or claim it affects
3. **What to do** — how to handle it

Each is rated:

| Rating | Meaning |
|---|---|
| **HIGH** | Could invalidate a headline claim if you don't address it |
| **MEDIUM** | Must be disclosed; changes how results should be read |
| **LOW** | Worth noting; doesn't change any conclusion |

---

# Section 1 — How the events were chosen

## 1.1 The events were not selected for this evaluation — HIGH

**What it is.** The 30 events arrived already finalised. Their internal IDs run up to 59, so roughly 30 were picked from a pool of about 60. No inclusion or exclusion criteria were recorded.

Looking at the data, the selection rule is visible: every event is named after a company that already exists in the supplier graph — `posco`, `nippon steel`, `dow`, `basf se`, `chevron`, `kyushu electric`, `china steel`, `glencore`. Events were chosen because the **directly hit** company was in the graph.

Nobody checked whether the **downstream** damage was ever publicly documented. That is the single root cause of most problems below.

**What it does to your results.** The set over-represents events with a graph-present direct victim and under-represents events with well-documented downstream victims. These are not the same thing. Your recall figure is shaped by a selection decision made before evaluation began.

**What to do.** State the selection rule explicitly in your methods chapter. Say that it was inherited, that it optimised for seed-company graph presence, and that this was not the same criterion as "downstream impact is documented."

## 1.2 Category balance looks good but collapses under conditioning — MEDIUM

**What it is.** By design the set is nicely balanced: 8 natural disasters, 8 industrial accidents, 7 energy events, 7 geopolitical events.

But once you keep only events with a high-confidence disclosure date, it becomes: 3 natural, 1 industrial, 0 energy, 2 geopolitical.

**What it does to your results.** You cannot compare detection performance across disruption types on the disclosure-date metric. The energy bar of that chart would be empty. Any such chart would be fiction.

**What to do.** Do not produce a per-category comparison for the disclosure metric. You *can* produce one for the company-identification metric, where all four categories still have events.

## 1.3 Selecting future events on graph membership would be circular — HIGH (a trap to avoid)

**What it is.** It is tempting to swap weak events for ones where the graph already contains the victims. This would raise recall from ~30% toward ~100%.

**What it does to your results.** It would invalidate them. Choosing test questions because your system can answer them is writing the exam after seeing what the student revised. It is the first thing an examiner probes.

**What to do.** Don't. If you ever add events, select on *"was a downstream company publicly named?"* — a criterion independent of your graph. Selecting on measurability is normal. Selecting on your own system's coverage is not.

---

# Section 2 — How the ground truth was built

## 2.1 One annotator, no agreement check — HIGH

**What it is.** Every judgement in this dataset was made by a single annotator (an AI assistant) in one pass. No second person re-labelled any event. There is no inter-annotator agreement statistic.

**What it does to your results.** Every label carries one person's judgement about what counts as "affected," which sources are credible, and when to stop searching. Systematic bias cannot be detected.

**What to do.** Cheapest fix available: have a human re-annotate 10 randomly chosen events and report the agreement rate. Half a day, and it removes the most common objection. If you can't, say plainly that the dataset is single-annotator and that this is a limitation.

## 2.2 The ground truth and the system may read the same sources — HIGH

**What it is.** I built the answer key largely by searching the web. If your RAG agent also ingests news articles, then "did the agent find the affected company?" partly reduces to "did the agent read the same article I did?"

**What it does to your results.** Agreement is partly tautological. It inflates apparent performance on any event where my evidence was a news article.

**What to do.** Two things. First, weight the evidence types differently — entries backed by SEC filings and corporate press releases have independent evidentiary value; news-only entries do not. Second, if you can, exclude my exact source URLs from the agent's corpus, or report performance separately for events where the agent found the company through a *different* document than my citation.

## 2.3 The search protocol was not systematic — MEDIUM  [REDUCED]

**What it is.** Search terms were chosen per event, by judgement. Roughly 4 to 10 searches per event. There was no fixed query template, no fixed stopping rule, and no record of every query run.

Worse: effort was not uniform. Early events in batch 1 received more searching before I understood the time budget. Later events received less.

**What it does to your results.** The completeness of ground truth varies by event in a way that correlates with when it was researched, not with how much evidence exists. An event may look "clean" simply because it was researched late.

**RESOLVED FOR ONE FIELD.** The sweep of 2026-07-27 replaced judgement-based searching with exhaustive
filing enumeration for the disclosure-date field: every affected company resolved to a CIK, every filing
in a 92-day window enumerated, every constituent document including exhibits opened. 47 companies,
1,036 filings, 2,074 documents, 0 unresolved. That field is now reproducible.

**STILL OPEN FOR THE OTHER FIELD.** `ground_truth_affected` was still built by judgement-based searching
with variable effort per event. Do not let the rigour of the disclosure field imply rigour in the
affected-company field.

## 2.4 "Affected" was never tightly defined — MEDIUM

**What it is.** I accepted a company as affected if a source described delays, shortages, production halts, cost increases, force majeure, or supply allocation. That is a wide net. A stricter definition (say, "quantified financial impact only") would produce a much smaller set. A looser one would produce a larger one.

**What it does to your results.** Both your precision and recall depend on a threshold that was never written down. Another researcher applying "reasonable" judgement could produce a materially different answer key.

**What to do.** Write the definition down retrospectively and state it. Something like: *"A company is recorded as affected where a cited source states it experienced a production halt, supply delay, shortage, allocation, force majeure, or a cost increase attributable to the event."* Then acknowledge that the threshold is inclusive.

## 2.5 The count of "80 companies" is approximate — LOW

**What it is.** Some entries bundle several companies into one record. For example, one entry reads "Aquila Resources, Cockatoo Coal and Ensham Resources," and another reads "Telkomsel, Indosat Ooredoo, XL Axiata and Hutchison 3 Indonesia."

**What it does to your results.** The denominator for recall is fuzzy by a handful of companies. It will not change any conclusion, but it will make your numbers slightly irreproducible if someone recounts.

**Confirmed by the sweep.** Two bundled entries were flagged automatically and appear in
`coverage_log.csv` with the reason `bundled entry - split before running`. Those are the exact rows to fix.

**What to do.** Before scoring, split bundled entries into individual companies and fix the count. Twenty minutes.

## 2.6 Direction of impact is not always verified — LOW  [REDUCED]

**What it is.** For some entries I confirmed only that a company *discussed* the event, not that it was
*harmed* by it. Four were flagged as identified by phrase-search and never read.

**Three are now RESOLVED.** The sweep retrieved the matched passage in each case and all three assert a
real operational impact:

- **Pactiv Evergreen** — Uri "damaged buildings and equipment in, interfered with the operations of, and
  sharply increased the energy" costs. Confidence low → **high**.
- **TTM Technologies** — Q4 guidance cut because "potential power rationing in China" was expected to hurt
  profitability. Confidence low → **high**.
- **Genco Shipping** — Capesize earnings pulled back "following a reported ban by China on Australian coal
  shipments". Confidence low → **high**.

**One remains open:** Diodes Incorporated (event 40), which was not the accepted candidate.

There is a related problem in the opposite direction. In several events the companies most affected were *beneficiaries* — steel producers gained from the 2018 tariffs, nickel miners gained from the Indonesian export ban, Chilean Cobalt framed the DRC ban as good news. I excluded these, but the line between "affected" and "benefited" required judgement.

**What it does to your results.** A small number of entries may be false positives in your answer key — companies marked as hurt that were merely commenting, or that gained.

**What to do.** Read the four flagged passages before scoring. That is an hour. For the rest, disclose that beneficiary-versus-victim was a judgement call.

## 2.7 Some claims rest on second-hand or weak sources — MEDIUM

**What it is.** Specific examples:

- The SUMCO entry (Mitsubishi Materials event) rests on a Nikkei figure reached through a Wikipedia citation. I never read the Nikkei article.
- The Korinox entry (POSCO event) rests on a single company-friendly feature article whose framing is that POSCO ultimately *prevented* the delay.
- The Dow entry (BASF 2016 event) rests on trade-press market commentary, not on any Dow statement.

**What it does to your results.** These entries are weaker than the rest. If your system misses them, that is arguably not a failure.

**What to do.** These are already marked `low` confidence in the `confidence` field. Consider reporting your metrics twice — once over all entries, once over `high` and `medium` confidence entries only.

## 2.8 Several sources are paywalled — LOW

**What it is.** Some claims cite ICIS, S&P Global Platts, Caixin Global, Fortune or Wood Mackenzie. I could read headlines and snippets but not always the full article. Each is flagged with `full_text_available: false`.

**What it does to your results.** A reader cannot fully verify those specific claims without a subscription.

**What to do.** They are individually flagged. Where a paywalled source is the *only* support for an entry, consider downgrading confidence.

## 2.9 Link rot — LOW

**What it is.** 248 URLs. Over the life of a dissertation, some will die.

**What to do.** Archive them. The corpus-building script you're about to run saves the text locally, which solves this automatically.

---

# Section 3 — The disclosure-date field

This is the field you said matters most, so it gets the most scrutiny.

## 3.1 Eighteen of thirty are blank — HIGH (but the blanks are now evidenced)  [REDUCED]

**What it is.** Only 12 of 30 events have a `date_first_disclosure`. Of those 12, only 6 are high confidence.

The blanks break down as:

| Reason | Count |
|---|---|
| No affected company was ever identified, so nobody could disclose | 10 |
| Affected companies are private; no filings exist anywhere | 2 |
| Affected companies file in Taiwan or Japan, outside search scope | 3 |
| A US-listed affected company exists and may have been missed | 3 |

**What it does to your results.** Any lead-time analysis has a usable sample of 12, or 6 if you restrict to high confidence. That is a case-study sample, not a statistical one.

**Status after the sweep.** Still 18 blanks, but they are no longer assertions of failure. Every affected
company was enumerated on EDGAR and every document in the window opened. Each blank now carries a
`date_first_sec_filing_note` recording how many companies, filings and documents were checked for that
event. "We could not find one" has become "we opened everything reachable and there is none."

**What to do.** Never compute a mean lead time. Publish a table of individual cases. Underneath it,
publish the breakdown above so every one of the 18 blanks is accounted for. Cite `coverage_log.csv`.

## 3.2 The field only measures American disclosure — HIGH

**What it is.** All 12 dates came from SEC filings or US/UK corporate press releases. Zero came from Korea's DART, Taiwan's MOPS, Japan's TDnet or EDINET, Indonesia's IDX, or the Shanghai and Shenzhen exchanges.

**What it does to your results.** The field is not "earliest official disclosure." It is "earliest official disclosure discoverable through EDGAR." An event that harmed only Asian-listed companies will show a blank even where a filing definitely exists.

Worse, this bias is not random. It correlates with where events happened, which correlates with event category. So it can masquerade as a finding about disruption types.

**What to do.** Rename the field in your thesis to something honest — for example `date_first_disclosure_sec` — or state the restriction wherever it appears. Never compare Asian and Western events on this metric.

## 3.3 One known case is a month late — HIGH

**What it is.** For the 2021 China power rationing event, the recorded date is 2021-10-27, from a TTM Technologies filing. But Unimicron and Eson Precision both filed Taiwan Stock Exchange announcements on **2021-09-26** — a month earlier. Those filings are known to exist; I simply could not retrieve them.

**What it does to your results.** That single row would make your system appear roughly one month better than it actually is on that event.

**What to do.** Either retrieve the two MOPS filings, or exclude this event from lead-time analysis, or mark it as an upper bound. It is already flagged `low` confidence in the file.

## 3.4 Earnings-call transcripts were not systematically checked — MEDIUM

**What it is.** The original brief asked for earnings-call transcripts as a valid disclosure channel. In practice I checked filing indexes, not transcripts. Transcripts are commonly paywalled and were not reliably reachable.

**What it does to your results.** In several events an executive likely mentioned the disruption on a quarterly call *before* it appeared in a written filing. Those earlier acknowledgements are missing.

**What to do.** Disclose it. If you have library access to a transcript database, a targeted check on the 12 populated events would tighten those dates.

## 3.5 The earliest true disclosure is often structurally invisible — MEDIUM

**What it is.** In many of these events the genuine first acknowledgement was a **force majeure notice sent privately to customers**. LyondellBasell on 15 Feb 2021, Olin on 16 Feb 2021, and Rio Tinto's contract notices in April 2018 are examples. These are commercial letters, not public documents. They surface only when a journalist reports them.

**What it does to your results.** There is a floor on how early your measured disclosure date can ever be, and it sits *later* than reality. Your system may genuinely be detecting before disclosure by more than you can prove.

**What to do.** State it. It is a limitation that works *against* your system, which makes it a safe and credible one to raise.

## 3.6 Filing dates and disclosure dates are not the same thing — MEDIUM  [UPGRADED]

**What it is.** EDGAR stamps documents submitted after 17:30 Eastern with the *next business day*. Transocean's Deepwater Horizon press release was submitted at 17:37 on 22 April 2010 and carries a file date of 23 April. The company acknowledged on the 22nd.

**What it does to your results.** Any date taken from EDGAR's `file_date` field can be one day late.

**A second instance found by the sweep.** Transocean's 8-K states in its own text: "On April 21, 2010,
the Company announced that it had a fire and explosion onboard its semisubmersible drilling rig Deepwater
Horizon." So the announcement was 2010-04-21, EDGAR's file date is 2010-04-23, and the submission
timestamp is 2010-04-22 17:37. Three defensible dates for one disclosure. `date_first_disclosure` was
corrected to 2010-04-21; `date_first_sec_filing` retains EDGAR's 2010-04-23.

**What to do.** Use `date_first_sec_filing` (EDGAR date) for the reproducible metric and
`date_first_disclosure` (true announcement date) for the substantive claim. Never mix them in one column.

---

# Section 4 — The supplier graph

## 4.1 The graph reaches about 30% of confirmed victims — HIGH

**What it is.** Of ~80 confirmed affected companies, only about 24 appear anywhere in the graph — at hop1, hop2 or hop3. Names like SUMCO, AXT, Unimicron, Transocean, Nyrstar, Peabody, Dixie Group and NXP are genuinely affected and entirely absent.

**What it does to your results.** A perfect system tops out near 30% recall. That number measures your graph, not your agent.

**What to do.** Report two figures: recall over all ground truth, and recall over graph-reachable ground truth. Every affected company already carries an `in_graph_context` tag, so this costs about ten lines of code. Reporting both decomposes the failure instead of hiding it.

## 4.2 In-graph coverage is highly concentrated — MEDIUM

**What it is.** Of the ~24 reachable victims, about 15 come from just three events: the 2011 Tohoku earthquake, the 2010 Queensland floods, and Hurricane Harvey.

**What it does to your results.** Any metric averaged over company-event pairs is effectively a measurement of performance on those three events.

**What to do.** Average per event (macro-average), not per company-event pair. Report the per-event spread, not just the mean.

## 4.3 The graph contains temporal impossibilities — MEDIUM

**What it is.** For the 2010 rare-earth event the graph proposes CATL at hop1. CATL was founded in 2011. Gotion, Sunwoda and CALB are similarly anachronistic for that date.

**What it does to your results.** Some candidates cannot possibly be correct, so a portion of your false-positive rate is attributable to the graph having no time dimension.

**What to do.** State that the graph is a static snapshot with no validity dates. If feasible, filter candidates by company founding date before scoring.

## 4.4 The graph has direction errors — MEDIUM

**What it is.** For the 2018 Kyushu solar curtailment, the hop1 list contains 68 electricity *consumers* — Toyota, Sony, Denso and so on. But curtailment harms electricity *generators*. The arrow points the wrong way.

The 2010 Deepwater Horizon event is worse: all eleven hop1 candidates are BP's fuel customers, none of whom lost supply, because Macondo was an exploration well and not a producing asset.

**What it does to your results.** These are guaranteed false positives for any system that trusts graph proximity.

**What to do.** Keep these events. They are your best test of whether the agent reasons about *mechanism* rather than *adjacency*. This is a strength disguised as a limitation.

## 4.5 There are circular edges between your own events — LOW

**What it is.** Yunnan Aluminium is a hop1 candidate for event 54, and Yunnan Chihong Zinc & Germanium is a hop1 candidate for event 41. Each is the seed of the other's event.

**What it does to your results.** Scoring both events independently double-counts one graph edge.

**What to do.** Note it. If you report per-event results it makes no material difference.

## 4.6 Hop3 was never researched — MEDIUM

**What it is.** The hop3 candidate lists run to several hundred names per event. Per the original brief, I did not research them exhaustively.

**What it does to your results.** If your system returns hop3 candidates, you have no ground truth to score them against. A correct hop3 prediction may be scored as a false positive.

**What to do.** Either restrict scoring to hop1 and hop2 predictions, or treat hop3 predictions as unscored rather than wrong.

---

# Section 5 — Problems with the events themselves

## 5.1 Three events had the wrong kind of date — MEDIUM (corrected)

**What it is.** Three events recorded the date a policy *took effect* rather than the date it was *announced*:

| Event | Was | Corrected to | Gap |
|---|---|---|---|
| Section 232 tariffs | 2018-03-23 | 2018-03-01 | 3 weeks |
| Indonesia nickel ban | 2020-01-01 | 2019-08-30 | 4 months |
| Gallium/germanium controls | 2023-08-01 | 2023-07-03 | 4 weeks |

**What it does to your results.** If uncorrected, a four-month error would make any lead-time measurement meaningless for that event.

**What to do.** Already corrected in the files. Worth flagging that this pattern appeared in 3 of 30 events, so any future events should be checked for it specifically.

## 5.2 One seed company appears to be wrong — MEDIUM

**What it is.** Event 20 records the Mailiao ARO-3 fire against Formosa Petrochemical. The unit is operated by Formosa Chemicals & Fibre — which also appears in that event's own hop1 candidate list.

**What it does to your results.** If the seed is wrong, then the one company recorded as affected is actually the *directly* hit party, not a downstream one. Scoring it as a downstream hit would be an error.

**What to do.** This event is already marked unusable for scoring. Either fix the seed or leave it excluded.

## 5.3 Two events are the same shock — MEDIUM

**What it is.** Event 35 (Nord Stream 1 halt) and event 36 (BASF gas-driven ammonia curtailment) are both the 2022 European gas crisis. Event 36 is arguably a *consequence* of event 35.

**What it does to your results.** Scoring both independently inflates apparent recall, because a system that understands the gas crisis gets credit twice.

**What to do.** Drop one, or merge them, or report them as a single case.

## 5.4 One event may not have happened — MEDIUM

**What it is.** Event 49 records China's 2010 rare-earth embargo against Japan. Peer-reviewed and policy literature disputes that a targeted embargo occurred at all: Japanese customs data shows no uniform drop in imports attributable to the incident.

**What it does to your results.** You cannot score precision or recall on an event whose existence is contested without first taking a position on it.

**What to do.** Either take a documented position, or exclude the event. It is an excellent *qualitative* case — a monitoring system will find dozens of confident news assertions about an event that may be a media artefact.

## 5.5 Several events have no single date — MEDIUM

**What it is.** The Queensland floods built over weeks. The China power rationing escalated over months. Nord Stream was cut progressively from June to September 2022. Winter Storm Uri has a defensible date on either 14 or 15 February.

**What it does to your results.** Lead time is measured from a date that, for these events, is a judgement call. Shifting the event date by a week changes the answer.

**What to do.** For these events, report lead time as a range rather than a number, or exclude them from precise timing analysis.

## 5.6 One event date appears to be a placeholder — LOW

**What it is.** Event 36's original date of 2022-07-01 corresponds to no reported event. It looks like a month-start placeholder. BASF's actual ammonia curtailment announcement was 2021-09-27.

**What to do.** Already flagged. This event is a candidate for removal under 5.3 anyway.

---

# Section 6 — The empty events

## 6.1 Ten events have no affected companies — HIGH

**What it is.** Ten of thirty events record zero affected companies: Chevron 2012, Kyushu 2018, Korea 2011, S-Oil 2022, BASF 2022, rare earths 2010, Indonesia nickel 2020, Hebei 2023, Jilin 2005, Yunnan 2022.

**Why.** There is a clear pattern, and it is a genuine finding rather than a research failure.

- **Contract-type disruptions** — a plant stops, a supplier declares force majeure, an allocation is imposed — produce *named* victims. Journalists write "Company X cannot get its parts."
- **Price-type disruptions** — a commodity gets more expensive — produce *no* named victims. Nobody writes "Company Y paid 3% more for aluminium this quarter."

All ten empties are the price type.

**What it does to your results.** A third of your events contribute no positive examples. A system that outputs nothing scores perfectly on them.

**What to do.** Report this as a finding about *evaluability*, not as a gap. It tells future researchers which disruption types can and cannot be evaluated with public evidence.

## 6.2 Two different meanings of "empty" are pooled — HIGH

**What it is.** "No affected companies" currently means two very different things:

- **Provable negative** — there is a documented reason nobody was hurt. Kyushu 2018 (the harmed parties are generators, not the listed consumers) and Korea 2011 (large listed firms were explicitly exempted from load shedding) are of this kind.
- **Not found** — I searched and found nothing, but something may exist. Chevron 2012, Yunnan 2022, Hebei 2023 are of this kind.

**What it does to your results.** If you treat a "not found" as a true negative, you punish a system that *correctly* identifies a real victim. That corrupts precision and recall at the same time, in opposite directions.

**What to do.** Split them explicitly. Score against the provable negatives. Exclude the not-founds. Seven events are already marked unusable in the inventory for this reason.

---

# Section 7 — Sample size and statistics

## 7.1 The sample is too small for distributional claims — HIGH

**What it is.** 30 events. 23 scoreable. 12 with a disclosure date. 6 with a high-confidence disclosure date.

**What it does to your results.** You cannot report a mean lead time with a confidence interval. You cannot claim statistical significance for a difference between system variants unless the difference is very large. Confidence intervals on 12 heterogeneous cases would be enormous and misleading.

**What to do.** Frame the evaluation as case-based, not statistical. Report each case. Where you compare system variants, report per-event results side by side rather than a single averaged score, so the reader can see whether a difference is consistent or driven by one event.

## 7.2 The events are not comparable to each other — MEDIUM

**What it is.** A refinery fire, a currency-scale trade policy, a typhoon and a sanctions designation are not the same kind of thing. Their propagation mechanisms, timescales and evidence trails differ completely.

**What it does to your results.** Averaging across them produces a number with no clear meaning.

**What to do.** Macro-average per event and report the spread. Consider grouping by mechanism (contract-type vs price-type) rather than by the existing category labels, since that split predicts evaluability far better.

---

# Section 8 — Tooling and coverage boundaries

## 8.1 Non-US disclosure systems were unreachable — HIGH

**What it is.** MOPS (Taiwan), DART (Korea), TDnet and EDINET (Japan), IDX (Indonesia), and the Shanghai and Shenzhen exchanges were all outside reach. Known filings — for example Unimicron's on 2021-09-26 — could not be retrieved.

**What it does to your results.** This is the direct cause of limitation 3.2. It is a tooling boundary, not an absence of evidence.

**What to do.** State it as a scope boundary rather than a finding. Write: *"disclosure evidence was collected from SEC EDGAR only; filings on Asian exchanges were out of scope."*

## 8.2 Searching was predominantly in English — MEDIUM

**What it is.** Most searches were English-language. A small number were run in Korean and Chinese. Japanese, Taiwanese and Indonesian trade press was not systematically searched.

**What it does to your results.** Compounds the geographic bias in 3.2. Asian events have systematically thinner ground truth for reasons unrelated to what actually happened.

**What to do.** Disclose. Note that this affects the affected-company field as well as the disclosure field.

## 8.3 The sweep fixed one field, not all of them — MEDIUM  [now executed]

**What it is.** The sweep (resolve company → SEC ID → full filing list → filter to a 92-day window →
open every document) ran on 2026-07-27 and made the **disclosure-date** field exhaustive and reproducible
within the SEC universe.

It does **not** fix the affected-company field. That was still built by judgement-based searching.

**What it does to your results.** After running the script you can say the disclosure field was collected systematically. You cannot say the same of the affected-company field.

**What to do.** Be precise about which claim applies to which field. Do not let the rigour of one field imply rigour in the other.

## 8.4 The script covers less than half the companies — MEDIUM

**What it is.** Of the 80 event-company pairs, **47 were checked on EDGAR** and **33 were not SEC
registrants**: Taiwan-listed 6, private 6, ASX-listed 3, state-owned 3, LSE-listed 2, Oslo-listed 2,
Japan-listed 2, bundled entries 2, Korea-listed 1, Shanghai-listed 1, Frankfurt-listed 1, other 4.
Every one is logged individually in `coverage_log.csv` with its reason.

**What it does to your results.** "Exhaustive" means exhaustive within the SEC universe, which is a minority of your ground truth.

**What to do.** The script's coverage log should list, per company, whether it was covered and why not if it wasn't. Publish that log as an appendix. It converts a vague boundary into a documented one.

## 8.5 Enumeration is keyword-bounded — MEDIUM  [NEW, discovered by the sweep]

**What it is.** Enumeration removes guessing about *which documents* to open. It does not remove guessing
about *which words* to search for.

**Evidence from the run.** Event 47 (Section 232 tariffs) initially returned **nothing**, despite 291
documents being opened across GM and Ford, because the keyword list held only "Section 232" and "steel
tariff" — and Ford's filing says "commodity cost". Broadening the list surfaced the real disclosure, but
three of the five resulting candidates were false positives on generic words: a proxy statement discussing
the previous fiscal year, and governance boilerplate about "controlling raw material costs". On event 59
the top-ranked candidate matched "Australian coal" inside a **director's biography**.

**What it does to your results.** A null means "not found with these keywords", not "does not exist". And
a machine-proposed date is not trustworthy until a human reads the snippet.

**What to do.** Publish the per-event keyword list (stored in `disclosure_candidates.json` under
`keywords`). State that every accepted date was confirmed by reading the matched passage, and that two
machine proposals were rejected.

## 8.6 Older events have thinner surviving evidence — LOW

**What it is.** The events span 2005 to 2025. Web coverage of 2005 events is much thinner today than coverage of 2023 events — pages have been deleted, archives moved, trade press retired.

**What it does to your results.** Ground-truth completeness correlates with event recency for reasons unrelated to the event's actual impact.

**What to do.** Note it. EDGAR full-text search covers 2001 onward, so the SEC-based portion of the dataset is not affected by this.

---

# Section 9 — Things that look like limitations but are not

It is as important not to over-apologise as it is not to over-claim. The following are **not** weaknesses:

**The 30% graph coverage figure.** This is a *result*, not a flaw. "Commercial supplier graphs reach roughly a third of the firms that public evidence shows were affected" is a useful empirical finding for anyone building these systems.

**The ten empty events.** These are your negative class. Without them you cannot demonstrate that your system knows when to stay silent — which is a harder capability than finding something.

**The trap events.** Deepwater Horizon, Kyushu solar, rare earths 2010 and Korea 2011 are the most valuable events in the set. A system that scores well elsewhere and still avoids these has demonstrated genuine reasoning rather than graph proximity.

**Nulls in the disclosure field.** A null that is fully accounted for is not a gap. After the enumeration script runs, each null will be backed by a log showing exactly what was checked.

**Small sample size, if framed correctly.** Twelve documented case studies with primary-source citations is a legitimate contribution. Twelve data points presented as a distribution is not. The problem is the framing, not the number.

---

# Section 10 — The short version for your thesis

If you need four sentences:

> The ground-truth dataset was constructed by a single annotator in one pass, without an inter-annotator
> agreement check. Disclosure dates were obtained by a systematic sweep of SEC EDGAR in which every
> affected company was resolved to a filer identifier and every document it filed within ninety days of
> the event was opened and searched — 47 companies, 1,036 filings and 2,074 documents across thirty
> events. Because the sweep is confined to SEC registrants, 18 of 30 events carry no disclosure date and
> the field is systematically biased toward events involving US-listed companies; the 33 companies outside
> that universe are individually logged with the reason. The supplier graph contains approximately 30% of
> the companies that public sources confirm were affected, placing an upper bound on achievable recall
> that is a property of the graph rather than of the system under evaluation. Ten events have no
> identified affected company, reflecting a distinction between contract-driven disruptions, which produce
> named victims in public reporting, and price-driven disruptions, which do not.

---

# Appendix — Priority order for fixing

If you have limited time, this is the order that removes the most risk per hour:

| Priority | Action | Time | Removes |
|---|---|---|---|
| 1 | Split "provable negative" from "not found" in the empty events | 30 min | 6.2 (HIGH) |
| 2 | Compute both recall figures using the `in_graph_context` tags | 1 hr | 4.1 (HIGH) |
| ~~3~~ | ~~Run the enumeration script and publish the coverage log~~ — **DONE 2026-07-27** | — | 2.3, 3.1, 8.4 |
| ~~4~~ | ~~Read the four unverified passages~~ — **3 of 4 DONE by the sweep**, Diodes remains | 10 min | 2.6 (now LOW) |
| 5 | Split bundled company entries and fix the count | 20 min | 2.5 (LOW) |
| 6 | Decide on events 35/36 overlap and event 49's contested status | 30 min | 5.3, 5.4 |
| 7 | Have a human re-annotate 10 random events | half day | 2.1 (HIGH) |

Items 1 through 6 total about one working day. Item 7 is optional but removes the most common objection.
