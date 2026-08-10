"""
agent_graph.py -- Day 8: the LangGraph ReAct scaffold.

Graph shape:

               +---------+   tool_calls?   +-------+
    START ---> | agent   | --------------> | tools | --+
               +---------+                 +-------+   |
                    ^                          |        | should_stop?
                    |            continue       |        |
                    +---------------------------+        v
                                                     +----------+
                        (no tool_calls) -----------> | finalize | --> END
                                                     +----------+

Two places decide when to stop, and neither is "trust the LLM's own
judgement" alone:

  - route_after_agent: if the model's turn has NO tool calls (it believes
    it's done), go straight to `finalize` -- but finalize() itself still
    runs check_stopping_condition() and will surface any gap the model
    missed, so an early "I'm done" from the model can never silently skip
    the gap check (requirement 4: gap-flagging is a property of the
    OUTPUT, not of whether the model remembered to mention it).

  - route_after_tools: after EVERY tool-executing hop, check_stopping_
    condition() decides -- hop cap reached, or coverage+quality genuinely
    met -> `finalize`; otherwise -> back to `agent` for another hop. This
    is where the hard 4-hop cap actually lives: once hop_count >= max_hops,
    the graph structurally cannot route back to `agent` again, regardless
    of what the model wants to do next. The cap is enforced by GRAPH
    TOPOLOGY, not by a tool refusing to run or a prompt asking nicely.

One hop == one AI turn == one visit to the `tools` node, regardless of how
many individual tool calls the model made in that turn (a model issuing two
parallel tool calls in one turn still only costs 1 of the 4 hops). This
matches the "4-hop cap tracked in agent state" framing in the Day 8 brief:
the cap bounds round trips through the loop, not the raw count of
individual tool invocations.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent_state import AgentState
from stopping_condition import (
    SIGNIFICANT_TIER_MAX,
    check_stopping_condition,
    extract_significant_companies,
    record_evidence,
)
from system_prompt import render_system_prompt


# --------------------------------------------------------------------------- #
# Node: agent (the LLM turn)
# --------------------------------------------------------------------------- #

def make_agent_node(llm, llm_tools):
    """llm: any LangChain chat model supporting .bind_tools(). Bound once,
    at graph-build time, not per call."""
    bound_llm = llm.bind_tools(llm_tools)

    def agent_node(state: AgentState) -> dict:
        system = SystemMessage(content=render_system_prompt(state["max_hops"]))
        response = bound_llm.invoke([system, *state["messages"]])
        return {"messages": [response]}

    return agent_node


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


# --------------------------------------------------------------------------- #
# Node: tools -- execution AND structured bookkeeping in one place. See
# agent_tools.py's module docstring for why this bypasses prebuilt ToolNode:
# it needs the dataclass result, not just the text ToolMessage.content.
# --------------------------------------------------------------------------- #

# One absorber per tool: reads that tool's structured result and folds the
# company-level facts it contains into `coverage`, via record_evidence().
# Keeps tools_node itself tool-agnostic dispatch, not a pile of per-tool
# if/elif branches.

def _absorb_traverse_result(result, coverage: dict, hop: int) -> None:
    # Required (tier <= SIGNIFICANT_TIER_MAX, currently tier-1 AND tier-2)
    # companies -- these are what the stopping condition actually gates on.
    for c in extract_significant_companies(result):
        record_evidence(coverage, c["name"], "traverse_supply_graph", tier=c["tier"], hop=hop)
    # Anything beyond SIGNIFICANT_TIER_MAX -- recorded (so it shows up in
    # the trace / a future gap flag if the agent spends a hop on it) but
    # NOT required; see stopping_condition.py's module docstring for the
    # significance bar and its reasoning. Read from the same constant
    # extract_significant_companies() uses, rather than a hardcoded number,
    # so this can't silently drift out of sync if that constant changes.
    if getattr(result, "status", None) == "found" and getattr(result, "mode", None) == "downstream":
        for r in result.results:
            if r["tier"] > SIGNIFICANT_TIER_MAX:
                record_evidence(coverage, r["name"], "traverse_supply_graph", tier=r["tier"], hop=hop)


def _absorb_supplier_info_result(result, coverage: dict, hop: int) -> None:
    if getattr(result, "status", None) == "found":
        record_evidence(
            coverage, result.company, "get_supplier_info", hop=hop,
            staleness_days=result.staleness_days, tag_mismatch=result.tag_mismatch,
        )
    # no_manifest_entry / no_predating_document / chunks_missing: nothing to
    # absorb -- these ARE the "no evidence" case, which coverage already
    # detects by a company simply having zero evidence_sources. No separate
    # bookkeeping needed for the negative cases.


def _absorb_search_corpus_result(result, coverage: dict, hop: int) -> None:
    if getattr(result, "status", None) == "found":
        for company in result.distinct_companies:
            record_evidence(coverage, company, "search_corpus", hop=hop)


_ABSORBERS = {
    "traverse_supply_graph": _absorb_traverse_result,
    "get_supplier_info": _absorb_supplier_info_result,
    "search_corpus": _absorb_search_corpus_result,
}


def make_tools_node(raw_dispatch: dict):
    """raw_dispatch: agent_tools.build_tool_bindings()'s second return value."""

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        hop = state["hop_count"] + 1
        # Copy one level deeper than a bare dict(...) -- coverage's VALUES
        # are themselves dicts that record_evidence() mutates in place;
        # copying only the outer dict would leave those inner dicts shared
        # with the previous hop's state, which record_evidence() would then
        # silently mutate too. Copying each row keeps every hop's state
        # genuinely its own snapshot.
        coverage = {k: dict(v) for k, v in state["coverage"].items()}
        evidence_log = list(state["evidence_log"])
        new_messages = []

        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            call_id = call["id"]
            dispatch_fn = raw_dispatch.get(name)

            if dispatch_fn is None:
                new_messages.append(ToolMessage(
                    content=f"Unknown tool '{name}' -- no such tool is available.",
                    tool_call_id=call_id,
                ))
                evidence_log.append({
                    "hop": hop, "tool": name, "status": "unknown_tool",
                    "company_name": args.get("company_name") or args.get("company"),
                    "summary": f"unknown tool requested: {name}",
                })
                continue

            try:
                result = dispatch_fn(**args)
            except Exception as exc:
                # A single bad call (malformed args, etc.) must not crash
                # the whole run -- report it to the model as an observation
                # and log it, same as any other tool outcome.
                new_messages.append(ToolMessage(
                    content=f"Tool '{name}' raised an error: {exc}",
                    tool_call_id=call_id,
                ))
                evidence_log.append({
                    "hop": hop, "tool": name, "status": "error",
                    "company_name": args.get("company_name") or args.get("company"),
                    "summary": f"error: {exc}",
                })
                continue

            observation = result.as_observation()
            new_messages.append(ToolMessage(content=observation, tool_call_id=call_id))

            absorber = _ABSORBERS.get(name)
            if absorber is not None:
                absorber(result, coverage, hop)

            evidence_log.append({
                "hop": hop, "tool": name,
                "status": getattr(result, "status", "unknown"),
                "company_name": args.get("company_name") or args.get("company"),
                "summary": observation.splitlines()[0] if observation else "",
            })

        return {
            "messages": new_messages,
            "hop_count": hop,
            "coverage": coverage,
            "evidence_log": evidence_log,
        }

    return tools_node


def route_after_tools(state: AgentState) -> str:
    should_stop, _reason, _gaps = check_stopping_condition(state)
    return "finalize" if should_stop else "agent"


# --------------------------------------------------------------------------- #
# Node: finalize -- ALWAYS re-runs the stopping-condition check itself, even
# when the model believed it was done. This is what makes gap-flagging a
# property of the graph's OUTPUT rather than of the model's own diligence:
# route_after_agent's "no tool calls -> finalize" path means a model can end
# the loop early, but it cannot make finalize() skip the coverage check.
# --------------------------------------------------------------------------- #

def finalize_node(state: AgentState) -> dict:
    should_stop, reason, gaps = check_stopping_condition(state)
    # `should_stop` can come back False here (e.g. the model stopped calling
    # tools before coverage was ever met, or before hop 1 ran at all) --
    # finalize still runs, because the GRAPH is ending regardless (that's
    # why we're in this node). In that case `reason` would otherwise say
    # "continue", which is not an honest description of what actually
    # happened -- relabel it so stop_reason always reflects reality.
    if reason == "continue":
        reason = "agent_ended_early"

    last = state["messages"][-1]
    model_answer = (
        last.content
        if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None)
        else None
    )

    lines = [f"[stop_reason: {reason} | hops used: {state['hop_count']}/{state['max_hops']}]"]
    if model_answer:
        lines.append("\nAnalyst summary:\n" + model_answer)

    if gaps:
        lines.append(f"\nCOVERAGE GAPS ({len(gaps)}) -- reported explicitly, not omitted:")
        for g in gaps:
            lines.append(f"  - {g}")
    else:
        lines.append("\nNo coverage gaps: every tier-1 company found had usable, "
                     "sufficiently recent evidence.")

    report = "\n".join(lines)
    return {
        "stop_reason": reason,
        "gap_flags": gaps,
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_agent_graph(llm, llm_tools, raw_dispatch):
    """
    llm          : any LangChain chat model, unbound (bind_tools happens
                   inside make_agent_node)
    llm_tools    : agent_tools.build_tool_bindings(...)[0]
    raw_dispatch : agent_tools.build_tool_bindings(...)[1]
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm, llm_tools))
    graph.add_node("tools", make_tools_node(raw_dispatch))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()
