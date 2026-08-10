"""
agent_state.py -- Day 8: shared LangGraph state for the multi-hop agent.

WHY hop-count and coverage/evidence tracking live here, in agent state, and
not inside any of the three tools: graph_tool.py, supplier_info_tool.py, and
search_corpus_tool.py were all deliberately built stateless -- each call is
self-contained, with no memory of prior calls (see each tool's own revision
notes, e.g. graph_tool.py point 6: "the 4-hop cap and any looping logic are
explicitly OUT OF SCOPE here -- that's agent-state, built during Day 8, not
something a single tool call should track"). So the hop cap and "have we
covered everyone" tracking belong here, updated once per hop by the agent
loop itself, never inside GraphStore / SupplierInfoStore / CorpusSearchStore.

`coverage` and `evidence_log` are plain dicts/lists (not a custom reducer)
because every node that touches them (agent_graph.py's tools_node,
finalize_node) reads the current value out of state, builds the FULL new
value itself, and returns it -- LangGraph's default "last write wins" merge
behaviour is exactly what's wanted; there is never more than one node
writing to these keys in the same step, so no reducer is needed. Only
`messages` needs a reducer (add_messages), since messages must ACCUMULATE
across every node, not be overwritten by each node's own return value.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages


class CompanyCoverage(TypedDict):
    """One row per company the agent has encountered as potentially
    'significant' (see stopping_condition.py for the exact definition).
    Tracks whether -- and how well -- that company has since been backed
    by evidence, across however many tool calls have touched it."""
    name: str
    tier: Optional[int]            # tier at first/best confirmation (1, 2, ...); None if only ever surfaced by search_corpus, never confirmed by a graph traversal
    first_seen_hop: int
    evidence_sources: list         # tool names that returned USABLE evidence for this company, e.g. ['get_supplier_info', 'search_corpus']
    best_staleness_days: Optional[int]   # smallest staleness_days seen across any get_supplier_info evidence for this company
    tag_mismatch: Optional[bool]


class EvidenceLogEntry(TypedDict):
    """One row per tool CALL (not per company) -- the raw trace, kept
    separately from `coverage` (which is per-company, deduplicated, and
    is what the stopping condition actually reads)."""
    hop: int
    tool: str
    status: str
    company_name: Optional[str]
    summary: str


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    hop_count: int
    max_hops: int

    coverage: dict            # str (lowercased company name) -> CompanyCoverage
    evidence_log: list        # list[EvidenceLogEntry]

    stop_reason: Optional[str]
    gap_flags: list           # list[str], set by finalize_node
    final_report: Optional[str]


def make_initial_state(user_query: str, max_hops: int = 4) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=user_query)],
        hop_count=0,
        max_hops=max_hops,
        coverage={},
        evidence_log=[],
        stop_reason=None,
        gap_flags=[],
        final_report=None,
    )
