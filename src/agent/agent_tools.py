"""
agent_tools.py -- Day 8: binds the three existing Layer 3 tools
(graph_tool.py, supplier_info_tool.py, search_corpus_tool.py) into the two
shapes the agent loop needs:

  - raw_dispatch: {tool_name -> callable(**kwargs) returning the tool's own
    dataclass result} (GraphTraversalResult / SupplierInfoResult /
    CorpusSearchResult). agent_graph.py's tools node calls THESE, because it
    needs the STRUCTURED result -- status, staleness_days, distinct_companies,
    n_dropped_unenriched -- to feed the stopping condition. Re-deriving those
    facts by parsing the .as_observation() TEXT back apart would be fragile
    and would duplicate logic that already exists as typed dataclass fields.

  - llm_tools: the same three tools wrapped with @tool, for schema purposes
    ONLY -- what llm.bind_tools() needs so the model knows these three tools
    exist and how to call them (name, description, argument types). Each
    one's docstring here is that tool's LLM-facing spec, distinct from (a)
    the developer-facing design rationale in graph_tool.py /
    supplier_info_tool.py / search_corpus_tool.py themselves, and (b) the
    cross-tool STRATEGY prompt in system_prompt.py. See graph_tool.py's
    revision note #3 for why these are deliberately kept separate.

Deliberately NOT using LangGraph's prebuilt ToolNode: ToolNode invokes a
tool and can only place its return value into a ToolMessage's `content` (a
string), discarding the structured object the tool actually computed.
agent_graph.py's tools node calls raw_dispatch directly, builds the
ToolMessage text itself via result.as_observation() (so the two paths never
disagree about what text the model sees), and keeps the structured result
alongside it for stopping_condition.py to read.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool


def build_tool_bindings(graph_store, supplier_store, corpus_store):
    """
    graph_store    : graph_tool.GraphStore
    supplier_store : supplier_info_tool.SupplierInfoStore
    corpus_store   : search_corpus_tool.CorpusSearchStore

    Returns (llm_tools, raw_dispatch) -- see module docstring.
    """

    # -- raw dispatch: thin, argument-forwarding wrappers around each store's
    #    own method. No logic lives here -- this is purely "give the agent
    #    loop a uniform call('**kwargs) -> dataclass' shape per tool name."
    def _traverse_raw(company_name: str, mode: str = "downstream", max_tier: int = 2,
                       component: Optional[str] = None, min_distance_km: float = 500):
        return graph_store.traverse_supply_graph(
            company_name, mode=mode, max_tier=max_tier,
            component=component, min_distance_km=min_distance_km,
        )

    def _supplier_info_raw(company_name: str, event_date: str,
                            event_id: Optional[str] = None, max_chunks: int = 8):
        return supplier_store.get_supplier_info(
            company_name, event_date, event_id=event_id, max_chunks=max_chunks,
        )

    def _search_corpus_raw(query: str, event_date: str,
                            company: Optional[str] = None, top_k: int = 5):
        return corpus_store.search_corpus(query, event_date, company=company, top_k=top_k)

    # -- LLM-visible tools: schema only. Bodies delegate to the same raw
    #    functions above and return .as_observation() -- these are never
    #    actually invoked by agent_graph.py (which calls raw_dispatch
    #    directly), but keeping them real, callable, and logic-sharing
    #    (not stub bodies) means they stay correct if ever used standalone,
    #    e.g. bound into a plain LangChain agent outside this project's
    #    custom graph.
    @tool
    def traverse_supply_graph(
        company_name: str,
        mode: str = "downstream",
        max_tier: int = 2,
        component: Optional[str] = None,
        min_distance_km: float = 500,
    ) -> str:
        """Find companies structurally connected to a given company in the supply
        graph. Call this FIRST when investigating a disruption, before checking
        any specific company's filings or searching the corpus by topic -- this
        answers "who is even relevant here?"

        mode="downstream" (default): companies at risk of being affected BY a
        disruption at company_name -- its customers, and their customers, up to
        max_tier steps away. Use this once, early, to identify who might be
        impacted downstream of the disrupted company.

        mode="substitutes": other companies making a similar component to
        company_name, far enough away (min_distance_km) to plausibly be
        unaffected by the same event. Use when checking whether an affected
        company's customers have an alternative source.
        """
        return _traverse_raw(
            company_name, mode=mode, max_tier=max_tier,
            component=component, min_distance_km=min_distance_km,
        ).as_observation()

    @tool
    def get_supplier_info(
        company_name: str,
        event_date: str,
        event_id: Optional[str] = None,
        max_chunks: int = 8,
    ) -> str:
        """Look up what a SPECIFIC, already-named company says about itself in
        its own SEC filings, as of event_date. Use this AFTER a company has
        already been named (e.g. by traverse_supply_graph or search_corpus) to
        ground a claim about that one company in primary-source text. This
        tool cannot discover a company you haven't already named, and does
        not search by topic -- use search_corpus for that instead.
        """
        return _supplier_info_raw(
            company_name, event_date, event_id=event_id, max_chunks=max_chunks,
        ).as_observation()

    @tool
    def search_corpus(
        query: str,
        event_date: str,
        company: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Free-text search over the SEC filing corpus, as of event_date. Can
        surface a company you never named (unlike get_supplier_info). Use
        this to find topical evidence (a specific risk, tariff, commodity)
        or when you don't yet know which company is relevant.
        """
        return _search_corpus_raw(
            query, event_date, company=company, top_k=top_k,
        ).as_observation()

    llm_tools = [traverse_supply_graph, get_supplier_info, search_corpus]
    raw_dispatch = {
        "traverse_supply_graph": _traverse_raw,
        "get_supplier_info": _supplier_info_raw,
        "search_corpus": _search_corpus_raw,
    }
    return llm_tools, raw_dispatch
