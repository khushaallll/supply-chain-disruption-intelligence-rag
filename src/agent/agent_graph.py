"""
agent_graph.py -- the LangGraph ReAct scaffold.


Three places decide when to stop, and none of them is "trust the LLM's own
judgement" alone:

  - route_after_agent: if the model's turn has NO tool calls (it believes
    it's done), go straight to `finalize` -- but finalize() itself still
    runs check_stopping_condition() and will surface any gap the model
    missed, so an early "I'm done" from the model can never silently skip
    the gap check (requirement 4: gap-flagging is a property of the
    OUTPUT, not of whether the model remembered to mention it). This path
    already has real text from the model (or a malformed-tool-call-shaped
    text -- see _looks_like_malformed_tool_call), so it goes straight to
    finalize with no extra step needed.

  - route_after_tools: after EVERY tool-executing hop, check_stopping_
    condition() decides -- hop cap reached, or coverage+quality genuinely
    met -> `summarize`; otherwise -> back to `agent` for another hop. This
    is where the hard 4-hop cap actually lives: once hop_count >= max_hops,
    the graph structurally cannot route back to `agent` again, regardless
    of what the model wants to do next. The cap is enforced by GRAPH
    TOPOLOGY, not by a tool refusing to run or a prompt asking nicely.

  - summarize: added after a real trace exposed a real gap -- when the
    hop cap (or a coverage-complete stop) is reached right after a tool
    executes, the LAST message is a ToolMessage, not an AIMessage, so the
    model had never actually been asked to write anything up. Without this
    node, finalize_node's report jumped straight from a raw tool
    observation to the mechanical stop_reason/gap list with ZERO synthesis
    -- confirmed on a real Aurizon run (322 companies found, hop cap
    reached, no analyst summary at all in the output). Given how sprawling
    real events turn out to be, hitting the hop cap mid-tool-use is the
    COMMON case, not an edge case, so this needed fixing, not deferring.
    Deliberately calls the RAW (unbound) llm, not llm.bind_tools(llm_tools)
    -- this turn must produce text, not another tool-call attempt, so
    tools are structurally unavailable rather than merely discouraged by
    prompt wording. Not counted as a hop: it is a wrap-up call, not
    evidence-gathering, so "hops used: X/4" in the report keeps meaning
    exactly what it means everywhere else.

One hop == one AI turn == one visit to the `tools` node, regardless of how
many individual tool calls the model made in that turn (a model issuing two
parallel tool calls in one turn still only costs 1 of the 4 hops). This
matches the "4-hop cap tracked in agent state" framing in the Day 8 brief:
the cap bounds round trips through the loop, not the raw count of
individual tool invocations.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
# Token usage -- shared by every node that actually calls the LLM (agent_node
# and summarize_node; tools_node and finalize_node never do). Added to
# measure real LLM cost per event, alongside hop_count's existing measure
# of real LLM CALL COUNT. See agent_state.py's own AgentState docstring for
# why these are cumulative fields with no reducer, same pattern as
# coverage/evidence_log.
# --------------------------------------------------------------------------- #

def _extract_token_usage(response) -> tuple[Optional[int], Optional[int]]:
    """Returns (prompt_tokens, completion_tokens) for one LLM response.

    Tries LangChain's own standardized `usage_metadata` first (the
    input_tokens/output_tokens shape most current chat-model integrations,
    including recent langchain-ollama versions, populate). Falls back to
    Ollama's raw `response_metadata` keys (`prompt_eval_count`/
    `eval_count`) for older integrations that don't populate
    usage_metadata yet -- confirmed by direct inspection of what
    langchain-ollama actually returns as of Day 9's model-selection work
    (see llm_setup.py), not assumed to be one shape or the other in
    advance.

    Returns (None, None), NOT (0, 0), if neither is found. A real
    zero-token response and "the model/provider never reported usage at
    all" are different facts -- collapsing them to 0 would make a missing
    measurement silently indistinguishable from a genuinely free call,
    which would corrupt any total built from it without any visible sign
    that had happened.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens")

    meta = getattr(response, "response_metadata", None) or {}
    prompt = meta.get("prompt_eval_count")
    completion = meta.get("eval_count")
    if prompt is not None or completion is not None:
        return prompt, completion

    return None, None


def _accumulate_tokens(state: AgentState, node: str, response) -> dict:
    """Returns the token-tracking fields to merge into a node's return
    dict. Called from both agent_node and summarize_node -- the only two
    places in the whole graph that invoke the LLM. `None` usage values
    are treated as 0 for the RUNNING TOTAL specifically (a total has to
    be a number to stay useful), but the raw None is still preserved,
    per-call, in token_usage_log -- so "this call's usage was unknown" is
    never lost, only excluded from the sum."""
    prompt_tokens, completion_tokens = _extract_token_usage(response)
    log_entry = {
        "node": node,
        "hop": state["hop_count"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    new_prompt_total = state["prompt_tokens"] + (prompt_tokens or 0)
    new_completion_total = state["completion_tokens"] + (completion_tokens or 0)
    return {
        "prompt_tokens": new_prompt_total,
        "completion_tokens": new_completion_total,
        "total_tokens": new_prompt_total + new_completion_total,
        "token_usage_log": state["token_usage_log"] + [log_entry],
    }


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
        return {"messages": [response], **_accumulate_tokens(state, "agent", response)}

    return agent_node


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


# --------------------------------------------------------------------------- #
# Company-name canonicalization for coverage bookkeeping -- see the Day 9
# trace review that found this bug: three independent real traces
# (BHP/"BHP Group", volkswagen/"Volkswagen AG", and most clearly toyota
# motor/"Toyota Motor Corp"/"Toyota Motor Corporation") each showed the
# SAME real company splitting into multiple coverage rows because
# get_supplier_info/search_corpus echo back whatever string the model
# passed in (by design -- see supplier_info_tool.py), not a canonical
# name. record_evidence() keys purely on that raw string, so a company
# already named "toyota motor" by traverse_supply_graph and a company
# named "toyota motor corporation" by a later get_supplier_info call
# become two disconnected rows -- one still gapped, one holding real,
# genuinely-found evidence that never counts toward closing that gap.
# On the Nippon Steel trace this suppressed the single strongest,
# highest-confidence ground-truth company (Toyota) even though the agent
# had already found and read Toyota's own 20-F.
#
# Fix: before recording evidence under a name, check whether it is a
# corporate-suffix variant of an EXISTING coverage row (one name's word
# sequence is a prefix of the other's -- "toyota motor" is a prefix of
# "toyota motor corporation") and reuse that row's key if so, rather than
# creating a second, disconnected one. Deliberately NOT a generic fuzzy-
# match threshold: token_sort_ratio scores 'bhp' vs 'bhp group' at only
# 50 (too low to trust) while scoring 'china steel' vs 'china motor' at
# 63.6 (too high to safely ignore) -- there is no single ratio threshold
# that separates "same company, added suffix" from "different company,
# shared word" on real examples. Word-prefix matching does separate them
# cleanly on every case found in real traces, including the negatives.
# --------------------------------------------------------------------------- #

def _is_name_variant(a: str, b: str) -> bool:
    """True if a and b look like the same company differing only by a
    trailing corporate suffix or similar addition (Group, AG, Corp,
    Corporation, Company, etc.) -- checked as: one name's word sequence
    is a prefix of the other's."""
    a_words, b_words = a.split(), b.split()
    shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    return bool(shorter) and longer[:len(shorter)] == shorter


def _canonicalize_company_name(name: str, coverage: dict) -> str:
    """Returns the lowercased key `name` should be recorded under in
    `coverage`: the exact existing key if there's already an exact match;
    otherwise an existing key that's a corporate-suffix variant of it, if
    one exists (reusing that row instead of creating a disconnected new
    one); otherwise `name`'s own lowercased form (a genuinely new
    company)."""
    key = name.strip().lower()
    if key in coverage:
        return key
    for existing_key in coverage:
        if _is_name_variant(key, existing_key):
            return existing_key
    return key


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
        canonical_name = _canonicalize_company_name(result.company, coverage)
        record_evidence(
            coverage, canonical_name, "get_supplier_info", hop=hop,
            staleness_days=result.staleness_days, tag_mismatch=result.tag_mismatch,
        )
    # no_manifest_entry / no_predating_document / chunks_missing: nothing to
    # absorb -- these ARE the "no evidence" case, which coverage already
    # detects by a company simply having zero evidence_sources. No separate
    # bookkeeping needed for the negative cases.


def _absorb_search_corpus_result(result, coverage: dict, hop: int) -> None:
    if getattr(result, "status", None) == "found":
        for company in result.distinct_companies:
            canonical_name = _canonicalize_company_name(company, coverage)
            record_evidence(coverage, canonical_name, "search_corpus", hop=hop)


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

            # Confirmed on a real trace (4_dow_2017): the batching guidance
            # in the system prompt ("check multiple companies in one turn")
            # can be misapplied as "pass a LIST of company names as the
            # value of one call's company_name/company argument" instead of
            # the correct mechanism (several separate tool_calls entries in
            # the same turn). get_supplier_info/search_corpus expect a
            # single string; passing a list previously crashed the whole
            # call (caught, but wasted the entire hop -- on that real trace,
            # 9 companies got zero evidence-gathering because of it).
            # Real-world evidence throughout this project shows prompt
            # wording alone doesn't reliably prevent every misinterpretation
            # -- so instead of only relying on the prompt, tolerate this
            # shape directly: run the tool once per name in the list, and
            # combine the results into ONE ToolMessage (a single
            # tool_call_id can only receive one reply), rather than losing
            # the whole hop to an error. This converts a previously wasted
            # hop into the same real evidence multiple separate tool_calls
            # would have produced.
            name_arg_key = "company_name" if "company_name" in args else (
                "company" if "company" in args else None
            )
            if name_arg_key and isinstance(args.get(name_arg_key), list):
                names = args[name_arg_key]
                sub_observations = []
                for single_name in names:
                    sub_args = {**args, name_arg_key: single_name}
                    try:
                        sub_result = dispatch_fn(**sub_args)
                    except Exception as exc:
                        sub_observations.append(f"[{single_name}] error: {exc}")
                        evidence_log.append({
                            "hop": hop, "tool": name, "status": "error",
                            "company_name": single_name, "summary": f"error: {exc}",
                        })
                        continue
                    sub_obs_text = sub_result.as_observation()
                    sub_observations.append(f"[{single_name}]\n{sub_obs_text}")
                    absorber = _ABSORBERS.get(name)
                    if absorber is not None:
                        absorber(sub_result, coverage, hop)
                    evidence_log.append({
                        "hop": hop, "tool": name,
                        "status": getattr(sub_result, "status", "unknown"),
                        "company_name": single_name,
                        "summary": sub_obs_text.splitlines()[0] if sub_obs_text else "",
                    })
                combined = "\n\n---\n\n".join(sub_observations)
                new_messages.append(ToolMessage(content=combined, tool_call_id=call_id))
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
    return "summarize" if should_stop else "agent"


# --------------------------------------------------------------------------- #
# Node: summarize -- see module docstring for why this exists. Only reached
# when the graph is stopping right after a tools-node execution.
# --------------------------------------------------------------------------- #

def make_summarize_node(llm):
    """llm: the SAME raw chat model passed to build_agent_graph, used here
    WITHOUT .bind_tools() -- deliberately, so this turn cannot produce
    another tool-call attempt, only text."""

    def summarize_node(state: AgentState) -> dict:
        system = SystemMessage(content=render_system_prompt(state["max_hops"]))
        wrapup = HumanMessage(content=(
            "You have reached the end of this investigation -- either the "
            "tool-call budget is used up, or you have covered every "
            "significant company you found. Do not attempt any further "
            "tool calls; none are available. Write your final analyst "
            "summary now, based only on what you have already gathered: "
            "which companies are affected, what evidence backs each one, "
            "and what you were unable to confirm."
        ))
        response = llm.invoke([system, *state["messages"], wrapup])
        return {"messages": [response], **_accumulate_tokens(state, "summarize", response)}

    return summarize_node


# --------------------------------------------------------------------------- #
# Node: finalize -- ALWAYS re-runs the stopping-condition check itself, even
# when the model believed it was done. This is what makes gap-flagging a
# property of the graph's OUTPUT rather than of the model's own diligence:
# route_after_agent's "no tool calls -> finalize" path means a model can end
# the loop early, but it cannot make finalize() skip the coverage check.
# --------------------------------------------------------------------------- #

def _looks_like_malformed_tool_call(content: str) -> bool:
    """Detects the exact failure mode seen on a real Aurizon trace: the
    model tried to request a tool call but emitted it as plain JSON text
    in its message content instead of a properly structured tool_calls
    entry (which route_after_agent checks for). When that happens, content
    looks like a hand-written {"name": ..., "arguments": ...} blob rather
    than an actual analyst summary. Deliberately a loose heuristic -- false
    positives just mean a real final answer gets an extra honest check;
    false negatives just mean this stays mislabeled as agent_ended_early,
    same as before this fix. Either way is safe, so this doesn't need to
    be exact."""
    if not content:
        return False
    stripped = content.strip()
    return stripped.startswith("{") and '"name"' in stripped and '"arguments"' in stripped


def finalize_node(state: AgentState) -> dict:
    should_stop, reason, gaps = check_stopping_condition(state)
    # `should_stop` can come back False here (e.g. the model stopped calling
    # tools before coverage was ever met, or before hop 1 ran at all) --
    # finalize still runs, because the GRAPH is ending regardless (that's
    # why we're in this node). In that case `reason` would otherwise say
    # "continue", which is not an honest description of what actually
    # happened -- relabel it so stop_reason always reflects reality.
    last = state["messages"][-1]
    if reason == "continue":
        if isinstance(last, AIMessage) and _looks_like_malformed_tool_call(last.content):
            # The model was NOT trying to stop -- it was trying to call
            # another tool and failed to format the request correctly.
            # Labeling this "agent_ended_early" (implying a deliberate
            # choice) would be actively misleading when reading the trace
            # later; this is a distinct, real failure mode, not a decision.
            reason = "malformed_tool_call_output"
        else:
            reason = "agent_ended_early"

    model_answer = (
        last.content
        if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None)
        and reason != "malformed_tool_call_output"
        else None
    )

    lines = [f"[stop_reason: {reason} | hops used: {state['hop_count']}/{state['max_hops']}]"]
    if reason == "malformed_tool_call_output":
        lines.append(
            "\nNOTE: the model attempted another tool call but did not emit it in the "
            "expected structured format, so this run ended before the investigation "
            "was actually finished (not because the model judged it complete)."
        )
    if model_answer:
        lines.append("\nAnalyst summary:\n" + model_answer)

    n_required = sum(
        1 for row in state["coverage"].values()
        if row["tier"] is not None and row["tier"] <= SIGNIFICANT_TIER_MAX
    )
    # Found on a real trace: when the seed itself never resolves (see
    # graph_tool.py's not_resolved status), zero significant companies
    # are ever identified, so `gaps` is empty -- but for a reason that has
    # nothing to do with success. The old unconditional "no gaps" message
    # read as a clean pass either way, which is actively misleading on a
    # real event (a genuine PLN/Aurizon-seed-name-shaped failure) where the
    # investigation never got past identifying the disrupted company
    # itself. Distinguish "nothing required was ever found" from
    # "everything required was found and covered" explicitly.
    if gaps:
        lines.append(f"\nCOVERAGE GAPS ({len(gaps)}) -- reported explicitly, not omitted:")
        for g in gaps:
            lines.append(f"  - {g}")
    elif n_required == 0:
        lines.append(
            "\nNO SIGNIFICANT COMPANIES WERE EVER IDENTIFIED -- this is NOT a clean "
            "pass. The investigation could not get past identifying the disrupted "
            "company itself (or its downstream connections) in the graph, so there "
            "was nothing to gather evidence about. Treat this as an incomplete "
            "result, not a success."
        )
    else:
        lines.append(f"\nNo coverage gaps: all {n_required} tier-1/tier-2 companies found "
                     "had usable, sufficiently recent evidence.")

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
                   inside make_agent_node for the tool-calling turns, and
                   is deliberately NOT applied for make_summarize_node's
                   final wrap-up turn)
    llm_tools    : agent_tools.build_tool_bindings(...)[0]
    raw_dispatch : agent_tools.build_tool_bindings(...)[1]
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm, llm_tools))
    graph.add_node("tools", make_tools_node(raw_dispatch))
    graph.add_node("summarize", make_summarize_node(llm))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "summarize": "summarize"})
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()