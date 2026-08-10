"""
test_agent_mocked.py

Tests the Day 8 agent loop's MECHANICS -- stopping condition, hop cap,
tool-call sequencing, gap-flagging -- entirely against mocked tool outputs
and a scripted fake LLM. No real graph/corpus/manifest files, no chromadb/
torch/sentence-transformers, no API key, no network. This does NOT test
retrieval quality or real reasoning quality (that's Day 9, against real
events) -- it tests that the LOOP itself does what it's supposed to do,
in the same "named acceptance check" style already used elsewhere in this
project (test_supplier_info.py, test_search_corpus_logic.py): each check
states the exact expected value and prints PASS/FAIL.

PART A -- stopping_condition.py's pure functions, directly, against real
          dataclass instances (GraphTraversalResult / SupplierInfoResult /
          CorpusSearchResult imported from the actual project files -- no
          store construction needed, since these are just dataclasses).

PART B -- the full compiled LangGraph graph (agent_graph.build_agent_graph),
          driven by a ScriptedLLM (a stand-in chat model whose .invoke()
          pops pre-written AIMessages off a queue -- no bind_tools schema
          validation beyond langchain_core's own, no real model call) and
          fake stores (return pre-built dataclass results in sequence, the
          same "swap the __init__-loaded parts, keep the real code path"
          strategy test_search_corpus_logic.py's FakeBM25/FakeCollection
          already used). This exercises the REAL routing functions, the
          REAL tools_node absorbers, and the REAL finalize_node -- only the
          LLM and the three stores are fake.

Run: python test_agent_mocked.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import AIMessage

from agent.tools.graph_tool import GraphTraversalResult
from agent.tools.supplier_info_tool import SupplierInfoResult
from agent.tools.search_corpus_tool import CorpusSearchResult

from agent_state import make_initial_state
from stopping_condition import (
    check_stopping_condition,
    extract_significant_companies,
    record_evidence,
)
from agent_tools import build_tool_bindings
from agent_graph import build_agent_graph

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ===========================================================================
# PART A -- stopping_condition.py, pure functions
# ===========================================================================
print("=" * 70)
print("PART A -- stopping_condition.py pure-function tests")
print("=" * 70)

# --- extract_significant_companies -----------------------------------------
downstream_found = GraphTraversalResult(
    status="found", mode="downstream", company_name="posco", seed="posco",
    results=[
        {"name": "hyundai motor", "tier": 1, "industry": "auto", "country": "South Korea",
         "component": "steel", "confidence": 0.95},
        {"name": "kia", "tier": 1, "industry": "auto", "country": "South Korea",
         "component": "steel", "confidence": 0.90},
        {"name": "ford motor", "tier": 2, "industry": "auto", "country": "USA",
         "component": "steel", "confidence": 0.70},
        {"name": "some tier3 co", "tier": 3, "industry": "auto", "country": "USA",
         "component": "steel", "confidence": 0.40},
    ],
    dropped_unenriched=[{"name": "daechang steel", "tier": 1}],
    n_results=4, n_dropped_unenriched=1,
)

sig = extract_significant_companies(downstream_found)
sig_names = {c["name"] for c in sig}
check("A1 tier-1 AND tier-2 companies both count as 'significant'",
      sig_names == {"hyundai motor", "kia", "ford motor"}, sig_names)
check("A2 tier-3 company excluded from significance",
      "some tier3 co" not in sig_names)

substitutes_result = GraphTraversalResult(
    status="found", mode="substitutes", company_name="posco", seed="posco",
    results=[{"name": "china steel", "component": "steel", "country": "Taiwan",
              "distance_km": 900, "confidence": 0.8}],
    n_results=1,
)
check("A3 substitutes-mode results never count as significant",
      extract_significant_companies(substitutes_result) == [])

not_resolved = GraphTraversalResult(status="not_resolved", mode="downstream", company_name="xyz")
check("A4 not_resolved status -> no significant companies",
      extract_significant_companies(not_resolved) == [])

# --- record_evidence ---------------------------------------------------------
coverage: dict = {}
record_evidence(coverage, "Hyundai Motor", "traverse_supply_graph", tier=1, hop=1)
check("A5 record_evidence creates a new row, keyed lowercase",
      "hyundai motor" in coverage, coverage.keys())
check("A6 tier recorded on first insert", coverage["hyundai motor"]["tier"] == 1)
check("A7 evidence_sources starts with the tool that created the row",
      coverage["hyundai motor"]["evidence_sources"] == ["traverse_supply_graph"])

# search_corpus later mentions the same company with NO tier info -- tier
# must not be clobbered back to None
record_evidence(coverage, "hyundai motor", "search_corpus", hop=2)
check("A8 known tier is NOT overwritten by a later call with tier=None",
      coverage["hyundai motor"]["tier"] == 1)
check("A9 a second tool is appended to evidence_sources, not replacing the first",
      coverage["hyundai motor"]["evidence_sources"] == ["traverse_supply_graph", "search_corpus"])

# get_supplier_info reports staleness -- best (smallest) staleness should win
record_evidence(coverage, "hyundai motor", "get_supplier_info", hop=3, staleness_days=400)
record_evidence(coverage, "hyundai motor", "get_supplier_info", hop=4, staleness_days=100)
check("A10 best_staleness_days keeps the SMALLEST value seen",
      coverage["hyundai motor"]["best_staleness_days"] == 100,
      coverage["hyundai motor"]["best_staleness_days"])

record_evidence(coverage, "hyundai motor", "get_supplier_info", hop=5, staleness_days=900)
check("A11 a LARGER staleness_days on a later call does not overwrite the best one",
      coverage["hyundai motor"]["best_staleness_days"] == 100)

# --- check_stopping_condition -----------------------------------------------
# B1: empty coverage must not read as "done" on hop 0
empty_state = {"hop_count": 0, "max_hops": 4, "coverage": {}}
should_stop, reason, gaps = check_stopping_condition(empty_state)
check("A12 empty coverage on hop 0 -> should_stop is False",
      should_stop is False, (should_stop, reason))
check("A13 empty coverage on hop 0 -> reason == 'continue'", reason == "continue", reason)

# B2: one required company, no evidence yet -> a gap, and NOT stopped
partial_state = {
    "hop_count": 1, "max_hops": 4,
    "coverage": {"hyundai motor": {"name": "Hyundai Motor", "tier": 1, "first_seen_hop": 1,
                                    "evidence_sources": [], "best_staleness_days": None,
                                    "tag_mismatch": None}},
}
should_stop, reason, gaps = check_stopping_condition(partial_state)
check("A14 required company with zero evidence -> should_stop False", should_stop is False)
check("A15 gap message names the company and says 'no evidence gathered'",
      gaps == ["Hyundai Motor: no evidence gathered"], gaps)

# B3: required company WITH fresh evidence -> coverage_and_quality_met
covered_state = {
    "hop_count": 2, "max_hops": 4,
    "coverage": {"hyundai motor": {"name": "Hyundai Motor", "tier": 1, "first_seen_hop": 1,
                                    "evidence_sources": ["get_supplier_info"],
                                    "best_staleness_days": 200, "tag_mismatch": False}},
}
should_stop, reason, gaps = check_stopping_condition(covered_state)
check("A16 fully covered, fresh evidence -> should_stop True", should_stop is True)
check("A17 reason == 'coverage_and_quality_met'", reason == "coverage_and_quality_met", reason)
check("A18 no gaps reported when genuinely covered", gaps == [], gaps)

# B4: required company with STALE evidence -> covered but flagged, not "no evidence"
stale_state = {
    "hop_count": 2, "max_hops": 4,
    "coverage": {"hyundai motor": {"name": "Hyundai Motor", "tier": 1, "first_seen_hop": 1,
                                    "evidence_sources": ["get_supplier_info"],
                                    "best_staleness_days": 900, "tag_mismatch": False}},
}
should_stop, reason, gaps = check_stopping_condition(stale_state)
check("A19 stale-only evidence -> should_stop False (not good enough to finish)",
      should_stop is False)
check("A20 stale gap message is distinct from the 'no evidence' message",
      gaps == ["Hyundai Motor: only stale evidence (900 days old)"], gaps)

# B5: hop cap forces a stop even with real gaps remaining
capped_state = {
    "hop_count": 4, "max_hops": 4,
    "coverage": {"hyundai motor": {"name": "Hyundai Motor", "tier": 1, "first_seen_hop": 1,
                                    "evidence_sources": [], "best_staleness_days": None,
                                    "tag_mismatch": None}},
}
should_stop, reason, gaps = check_stopping_condition(capped_state)
check("A21 hop cap forces should_stop True even with a real gap", should_stop is True)
check("A22 reason == 'hop_cap_reached'", reason == "hop_cap_reached", reason)
check("A23 the gap is still reported even though we're stopping", len(gaps) == 1, gaps)

# B6: coverage completing EXACTLY on the capped hop reports completion, not the cap
exact_state = {
    "hop_count": 4, "max_hops": 4,
    "coverage": {"hyundai motor": {"name": "Hyundai Motor", "tier": 1, "first_seen_hop": 1,
                                    "evidence_sources": ["get_supplier_info"],
                                    "best_staleness_days": 50, "tag_mismatch": False}},
}
should_stop, reason, gaps = check_stopping_condition(exact_state)
check("A24 coverage met on the exact cap hop -> reason is completion, not the cap",
      reason == "coverage_and_quality_met", reason)

# B7: tier-3-only companies never gate stopping, however many there are
# (tier-1 and tier-2 ARE now required -- see the module docstring's
# "Revision: tier-2 significance" note -- so this check has to move to
# tier-3 to still be testing "something beyond the significance bar")
tier3_only_state = {
    "hop_count": 1, "max_hops": 4,
    "coverage": {"some tier3 co": {"name": "Some Tier3 Co", "tier": 3, "first_seen_hop": 1,
                                    "evidence_sources": [], "best_staleness_days": None,
                                    "tag_mismatch": None}},
}
should_stop, reason, gaps = check_stopping_condition(tier3_only_state)
check("A25 a tier-3-only coverage dict has NO gaps (tier-3 isn't required)",
      gaps == [], gaps)
check("A26 ...but also does not count as 'done' (nothing required has been confirmed yet)",
      should_stop is False, (should_stop, reason))


# ===========================================================================
# PART B -- full graph mechanics, via a scripted fake LLM + fake stores
# ===========================================================================
print()
print("=" * 70)
print("PART B -- full LangGraph mechanics (scripted LLM, fake stores)")
print("=" * 70)


class ScriptedLLM:
    """Stand-in chat model: .bind_tools() is a no-op returning self;
    .invoke() pops the next pre-written AIMessage off a queue. Lets the
    REAL graph (real routing functions, real tools_node, real
    finalize_node) run end-to-end with no API key and no real model."""
    def __init__(self, scripted_responses):
        self._queue = list(scripted_responses)
        self.n_invocations = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.n_invocations += 1
        if not self._queue:
            raise AssertionError(
                "ScriptedLLM ran out of scripted responses -- the graph "
                "asked the 'agent' node for another turn than the test "
                "anticipated (hop cap or stopping condition likely didn't "
                "fire when expected)."
            )
        return self._queue.pop(0)


def ai_tool_call(tool_name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


def ai_parallel_tool_calls(calls):
    """calls: list of (tool_name, args, call_id)."""
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": a, "id": i} for n, a, i in calls
    ])


def ai_final_answer(text):
    return AIMessage(content=text)


class FakeGraphStore:
    def __init__(self, responses):
        self._responses = list(responses)
    def traverse_supply_graph(self, *a, **kw):
        return self._responses.pop(0)


class FakeSupplierStore:
    def __init__(self, responses):
        self._responses = list(responses)
    def get_supplier_info(self, *a, **kw):
        return self._responses.pop(0)


class FakeCorpusStore:
    def __init__(self, responses):
        self._responses = list(responses)
    def search_corpus(self, *a, **kw):
        return self._responses.pop(0)


def make_traverse_result(pairs, seed="posco"):
    """pairs: list of (name, tier)."""
    return GraphTraversalResult(
        status="found", mode="downstream", company_name=seed, seed=seed,
        results=[{"name": n, "tier": t, "industry": "auto", "country": "?",
                  "component": "steel", "confidence": 0.9} for n, t in pairs],
        n_results=len(pairs),
    )


def make_supplier_found(company, staleness_days):
    return SupplierInfoResult(
        status="found", company=company, event_id=None, event_date="2022-09-06",
        doc_id="d1", published_date="2022-01-01", form="10-K", url="https://x",
        accession="a1", staleness_days=staleness_days, text="filing text",
        n_chunks_returned=3, n_chunks_total_in_doc=3, tag_mismatch=None,
        candidate_documents_considered=1,
    )


# --------------------------------------------------------------------------- #
# B1 -- stops on its own, BEFORE the hop cap, once coverage+quality are met
# --------------------------------------------------------------------------- #
print("\n--- B1: stops early once coverage+quality are genuinely met ---")

llm_b1 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_tool_call("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
    ai_tool_call("get_supplier_info", {"company_name": "kia", "event_date": "2022-09-06"}, "c3"),
])
graph_store_b1 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b1 = FakeSupplierStore([
    make_supplier_found("hyundai motor", staleness_days=200),
    make_supplier_found("kia", staleness_days=100),
])
corpus_store_b1 = FakeCorpusStore([])

llm_tools, raw_dispatch = build_tool_bindings(graph_store_b1, supplier_store_b1, corpus_store_b1)
app_b1 = build_agent_graph(llm_b1, llm_tools, raw_dispatch)
result_b1 = app_b1.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B1 stops before exhausting the hop budget", result_b1["hop_count"] < 4, result_b1["hop_count"])
check("B1 hop_count == 3 (exactly the 3 scripted tool-calling turns)",
      result_b1["hop_count"] == 3, result_b1["hop_count"])
check("B1 stop_reason == 'coverage_and_quality_met'",
      result_b1["stop_reason"] == "coverage_and_quality_met", result_b1["stop_reason"])
check("B1 no gap flags on a genuinely complete run", result_b1["gap_flags"] == [], result_b1["gap_flags"])
check("B1 ScriptedLLM was never asked for a 4th turn (graph didn't over-run)",
      llm_b1.n_invocations == 3, llm_b1.n_invocations)
check("B1 both tier-1 companies ended up in coverage",
      set(result_b1["coverage"].keys()) >= {"hyundai motor", "kia"}, result_b1["coverage"].keys())

# --------------------------------------------------------------------------- #
# B2 -- hop cap forces a stop while real gaps remain, and the loop never
#        exceeds the cap even though the scripted model keeps requesting tools
# --------------------------------------------------------------------------- #
print("\n--- B2: hard 4-hop cap forces a stop with gaps still open ---")

llm_b2 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_tool_call("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
    ai_tool_call("search_corpus", {"query": "steel supply", "event_date": "2022-09-06"}, "c3"),
    ai_tool_call("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c4"),
    # a 5th response is deliberately included -- if the graph asks for it,
    # that's a hop-cap bug, and llm_b2.n_invocations will read 5, not 4
    ai_tool_call("get_supplier_info", {"company_name": "kia", "event_date": "2022-09-06"}, "c5"),
])
graph_store_b2 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b2 = FakeSupplierStore([
    make_supplier_found("hyundai motor", staleness_days=50),
    make_supplier_found("hyundai motor", staleness_days=40),  # kia never gets covered
])
corpus_store_b2 = FakeCorpusStore([
    CorpusSearchResult(status="no_results", query="steel supply", event_date="2022-09-06"),
])

llm_tools2, raw_dispatch2 = build_tool_bindings(graph_store_b2, supplier_store_b2, corpus_store_b2)
app_b2 = build_agent_graph(llm_b2, llm_tools2, raw_dispatch2)
result_b2 = app_b2.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B2 hop_count is capped at exactly 4", result_b2["hop_count"] == 4, result_b2["hop_count"])
check("B2 stop_reason == 'hop_cap_reached'", result_b2["stop_reason"] == "hop_cap_reached", result_b2["stop_reason"])
check("B2 kia's gap is still reported, not silently dropped",
      any("kia" in g.lower() for g in result_b2["gap_flags"]), result_b2["gap_flags"])
check("B2 ScriptedLLM was invoked exactly 4 times, never a 5th (hop cap is real, not advisory)",
      llm_b2.n_invocations == 4, llm_b2.n_invocations)

# --------------------------------------------------------------------------- #
# B3 -- the model ends the loop on its own, before coverage is complete;
#        finalize() must still surface the gap, per requirement 4
# --------------------------------------------------------------------------- #
print("\n--- B3: model stops early without full coverage -> gap still surfaces ---")

llm_b3 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_final_answer("I believe I've covered the disruption sufficiently."),
])
graph_store_b3 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b3 = FakeSupplierStore([])
corpus_store_b3 = FakeCorpusStore([])

llm_tools3, raw_dispatch3 = build_tool_bindings(graph_store_b3, supplier_store_b3, corpus_store_b3)
app_b3 = build_agent_graph(llm_b3, llm_tools3, raw_dispatch3)
result_b3 = app_b3.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B3 only 1 hop was actually used (model chose to stop, not the cap)",
      result_b3["hop_count"] == 1, result_b3["hop_count"])
check("B3 stop_reason == 'agent_ended_early' (not silently relabeled as success)",
      result_b3["stop_reason"] == "agent_ended_early", result_b3["stop_reason"])
check("B3 both companies show up as unflagged gaps despite the model's confidence",
      len(result_b3["gap_flags"]) == 2, result_b3["gap_flags"])
check("B3 the model's own text is preserved in the final report",
      "I believe I've covered" in result_b3["final_report"], result_b3["final_report"])

# --------------------------------------------------------------------------- #
# B4 -- parallel tool calls in one AI turn cost exactly ONE hop
# --------------------------------------------------------------------------- #
print("\n--- B4: two tool calls in a single turn == one hop, not two ---")

llm_b4 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_parallel_tool_calls([
        ("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
        ("get_supplier_info", {"company_name": "kia", "event_date": "2022-09-06"}, "c3"),
    ]),
])
graph_store_b4 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b4 = FakeSupplierStore([
    make_supplier_found("hyundai motor", staleness_days=50),
    make_supplier_found("kia", staleness_days=60),
])
corpus_store_b4 = FakeCorpusStore([])

llm_tools4, raw_dispatch4 = build_tool_bindings(graph_store_b4, supplier_store_b4, corpus_store_b4)
app_b4 = build_agent_graph(llm_b4, llm_tools4, raw_dispatch4)
result_b4 = app_b4.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B4 two parallel tool calls in one turn used exactly 2 hops total (1 per turn)",
      result_b4["hop_count"] == 2, result_b4["hop_count"])
check("B4 both companies from the parallel turn are covered",
      result_b4["stop_reason"] == "coverage_and_quality_met", result_b4["stop_reason"])

# --------------------------------------------------------------------------- #
# B5 -- an unknown tool name and a tool that raises don't crash the loop,
#        and neither one spuriously marks a company as covered
# --------------------------------------------------------------------------- #
print("\n--- B5: unknown tool name / tool exception are handled, not fatal ---")

def _raising_supplier_info(*a, **kw):
    raise ValueError("simulated bad call")

class RaisingSupplierStore:
    def get_supplier_info(self, *a, **kw):
        return _raising_supplier_info()

llm_b5 = ScriptedLLM([
    ai_parallel_tool_calls([
        ("totally_made_up_tool", {"foo": "bar"}, "c1"),
        ("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
    ]),
    ai_final_answer("Could not gather usable evidence."),
])
graph_store_b5 = FakeGraphStore([])
corpus_store_b5 = FakeCorpusStore([])

llm_tools5, raw_dispatch5 = build_tool_bindings(graph_store_b5, RaisingSupplierStore(), corpus_store_b5)
app_b5 = build_agent_graph(llm_b5, llm_tools5, raw_dispatch5)
result_b5 = app_b5.invoke(make_initial_state("Investigate a disruption.", max_hops=4))

check("B5 the run completes without raising (unknown tool + exception both absorbed)",
      result_b5["hop_count"] == 1, result_b5["hop_count"])
check("B5 neither the unknown-tool call nor the raising call added anything to coverage",
      result_b5["coverage"] == {}, result_b5["coverage"])
error_statuses = {e["status"] for e in result_b5["evidence_log"]}
check("B5 evidence_log records both the unknown-tool and the error outcome",
      error_statuses == {"unknown_tool", "error"}, error_statuses)

# --------------------------------------------------------------------------- #
# B6 -- sensible tool ordering: a realistic trace (graph, then supplier
#        info, then corpus) runs through the loop in that order and the
#        evidence_log preserves it faithfully, hop by hop
# --------------------------------------------------------------------------- #
print("\n--- B6: a realistic identify-then-evidence trace is preserved in order ---")
# NOTE: this scenario deliberately keeps a SECOND tier-1 company (kia)
# uncovered until the last hop. If there were only one required company,
# the loop would (correctly) stop as soon as hyundai motor got its first
# adequate source -- the stopping condition only requires ONE adequate
# source per company (see stopping_condition.py's EVIDENCE QUALITY note);
# "two sources where available" is a system-prompt instruction to the
# model (system_prompt.py, requirement 2), not something the stopping
# condition itself forces. Keeping kia open is what makes a 4-hop trace
# actually run here, so ordering across all 4 hops can be checked.

llm_b6 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_tool_call("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
    ai_tool_call("search_corpus", {"query": "steel supply risk", "event_date": "2022-09-06",
                                    "company": "hyundai motor"}, "c3"),
    ai_tool_call("get_supplier_info", {"company_name": "kia", "event_date": "2022-09-06"}, "c4"),
])
graph_store_b6 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b6 = FakeSupplierStore([
    make_supplier_found("hyundai motor", staleness_days=50),
    make_supplier_found("kia", staleness_days=60),
])
corpus_store_b6 = FakeCorpusStore([
    CorpusSearchResult(status="found", query="steel supply risk", event_date="2022-09-06",
                        hits=[], distinct_companies=["hyundai motor"],
                        bm25_hit_count=1, semantic_hit_count=1, fused_hit_count=1),
])

llm_tools6, raw_dispatch6 = build_tool_bindings(graph_store_b6, supplier_store_b6, corpus_store_b6)
app_b6 = build_agent_graph(llm_b6, llm_tools6, raw_dispatch6)
result_b6 = app_b6.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

logged_order = [e["tool"] for e in result_b6["evidence_log"]]
check("B6 tool call order in evidence_log matches the scripted identify -> evidence order",
      logged_order == ["traverse_supply_graph", "get_supplier_info", "search_corpus", "get_supplier_info"],
      logged_order)
check("B6 hop numbers in evidence_log increment 1, 2, 3, 4 -- one per turn",
      [e["hop"] for e in result_b6["evidence_log"]] == [1, 2, 3, 4],
      [e["hop"] for e in result_b6["evidence_log"]])
check("B6 hyundai motor accumulated evidence from BOTH get_supplier_info and search_corpus "
      "(requirement 2: two sources where available) -- traverse_supply_graph is also present "
      "in the list (it named the company first) but is not itself an evidence source",
      set(result_b6["coverage"]["hyundai motor"]["evidence_sources"]) >=
      {"get_supplier_info", "search_corpus"},
      result_b6["coverage"]["hyundai motor"]["evidence_sources"])
check("B6 run finishes with coverage_and_quality_met",
      result_b6["stop_reason"] == "coverage_and_quality_met", result_b6["stop_reason"])
check("B6 used all 4 scripted hops (stopped only once kia was also covered)",
      result_b6["hop_count"] == 4, result_b6["hop_count"])

# --------------------------------------------------------------------------- #
# B7 -- a POSCO-sized event (5 significant companies: 2 tier-1, 3 tier-2 --
#        the same shape as the project's own reference trace: Hyundai, Kia
#        direct; Ford, GM, Tesla two hops out) genuinely cannot get
#        individual evidence for all 5 inside 4 hops (1 for the traversal,
#        3 left for evidence). This is the explicitly CHOSEN outcome (see
#        stopping_condition.py's "Revision: tier-2 significance" note): the
#        run should end honestly at the cap, with the uncovered companies
#        named as real gaps -- not silently redefined as "not required" to
#        force a clean finish.
# --------------------------------------------------------------------------- #
print("\n--- B7: a large (5-company) event hits the cap with 2 honest gaps ---")

llm_b7 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_tool_call("get_supplier_info", {"company_name": "hyundai motor", "event_date": "2022-09-06"}, "c2"),
    ai_tool_call("get_supplier_info", {"company_name": "kia", "event_date": "2022-09-06"}, "c3"),
    ai_tool_call("get_supplier_info", {"company_name": "ford motor", "event_date": "2022-09-06"}, "c4"),
])
graph_store_b7 = FakeGraphStore([make_traverse_result([
    ("hyundai motor", 1), ("kia", 1), ("ford motor", 2), ("gm", 2), ("tesla", 2),
])])
supplier_store_b7 = FakeSupplierStore([
    make_supplier_found("hyundai motor", staleness_days=50),
    make_supplier_found("kia", staleness_days=60),
    make_supplier_found("ford motor", staleness_days=70),
    # gm and tesla never get an evidence-gathering call at all -- the
    # script runs out of hops before reaching them, exactly like a real
    # 4-hop-capped run on a big event would.
])
corpus_store_b7 = FakeCorpusStore([])

llm_tools7, raw_dispatch7 = build_tool_bindings(graph_store_b7, supplier_store_b7, corpus_store_b7)
app_b7 = build_agent_graph(llm_b7, llm_tools7, raw_dispatch7)
result_b7 = app_b7.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B7 hop_count capped at 4 (1 traversal + 3 evidence calls, exactly what fits)",
      result_b7["hop_count"] == 4, result_b7["hop_count"])
check("B7 stop_reason == 'hop_cap_reached' (honest -- not relabeled as complete)",
      result_b7["stop_reason"] == "hop_cap_reached", result_b7["stop_reason"])
check("B7 exactly 2 gaps reported -- gm and tesla, the two that ran out of budget",
      len(result_b7["gap_flags"]) == 2, result_b7["gap_flags"])
check("B7 the gaps name gm and tesla specifically, not a vague count",
      any("gm" in g.lower() for g in result_b7["gap_flags"])
      and any("tesla" in g.lower() for g in result_b7["gap_flags"]),
      result_b7["gap_flags"])
check("B7 the 3 companies that DID get evidence are not also flagged as gaps",
      not any(name in " ".join(result_b7["gap_flags"]).lower()
              for name in ("hyundai", "kia", "ford")),
      result_b7["gap_flags"])

# --------------------------------------------------------------------------- #
# B8 -- a model that tries to request another tool call but emits it as
#        plain JSON text instead of a structured tool_calls entry (the
#        exact failure mode seen on a real Aurizon trace with a small
#        local model). This must be labeled honestly -- NOT as
#        'agent_ended_early', which would wrongly imply the model judged
#        the investigation complete.
# --------------------------------------------------------------------------- #
print("\n--- B8: malformed tool-call-as-text is labeled honestly, not as a deliberate stop ---")

malformed_json_text = (
    '{\n  "name": "get_supplier_info",\n  "arguments": {\n'
    '    "company_name": "kia",\n    "event_date": "2022-09-06"\n  }\n}'
)
llm_b8 = ScriptedLLM([
    ai_tool_call("traverse_supply_graph", {"company_name": "posco"}, "c1"),
    ai_final_answer(malformed_json_text),  # no real tool_calls -- this is the bug being tested
])
graph_store_b8 = FakeGraphStore([make_traverse_result([("hyundai motor", 1), ("kia", 1)])])
supplier_store_b8 = FakeSupplierStore([])
corpus_store_b8 = FakeCorpusStore([])

llm_tools8, raw_dispatch8 = build_tool_bindings(graph_store_b8, supplier_store_b8, corpus_store_b8)
app_b8 = build_agent_graph(llm_b8, llm_tools8, raw_dispatch8)
result_b8 = app_b8.invoke(make_initial_state("Investigate the POSCO disruption.", max_hops=4))

check("B8 stop_reason == 'malformed_tool_call_output' (NOT 'agent_ended_early')",
      result_b8["stop_reason"] == "malformed_tool_call_output", result_b8["stop_reason"])
check("B8 the garbled JSON is not shown as if it were a real analyst summary",
      "Analyst summary" not in result_b8["final_report"], result_b8["final_report"])
check("B8 the report explicitly notes the investigation did not actually finish",
      "did not actually finish" in result_b8["final_report"] or "not" in result_b8["final_report"].lower(),
      result_b8["final_report"])
check("B8 both companies are still correctly flagged as gaps",
      len(result_b8["gap_flags"]) == 2, result_b8["gap_flags"])


# ===========================================================================
print()
print("=" * 70)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)