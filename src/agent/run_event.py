"""
run_event.py -- Day 9: the real runner.

This is the file Day 8 deliberately left unwritten. Everything up to now
(agent_state.py, stopping_condition.py, agent_tools.py, system_prompt.py,
agent_graph.py) only ever ran against fake stores and a scripted fake LLM
(test_agent_mocked.py) -- proving the LOOP's mechanics, not touching real
data or a real model. This file is the first thing that does both:
constructs the real GraphStore / SupplierInfoStore / CorpusSearchStore,
wires the shared Phonebook through all three (a one-line change flagged as
an open item at the end of day8_implementation_log.md), binds a real LLM
via llm_setup.py, builds the real compiled graph, and runs it on one real
event end to end.

[CONFIRM: the paths below are a guess at your real repo layout, same
convention already used in test_search_corpus_real.py / test_graph_tool.py
-- "the FIELD NAMES and DEFAULTS are confirmed correct by direct
inspection, the PATH is not." GRAPH_PATH matches GraphStore.__init__'s own
default exactly (confirmed by inspecting graph_tool.py directly); the
other four are educated guesses matching this project's established
data/<kind>/<file> layout -- adjust to wherever your real files live.]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from agent.tools.graph_tool import GraphStore
from agent.tools.supplier_info_tool import SupplierInfoStore
from agent.tools.search_corpus_tool import CorpusSearchStore
from phonebook import Phonebook

from agent_state import make_initial_state
from agent_tools import build_tool_bindings
from agent_graph import build_agent_graph
from llm_setup import build_llm


# --------------------------------------------------------------------------- #
# [CONFIRM: paths -- see module docstring]
# --------------------------------------------------------------------------- #
GRAPH_PATH = "data/processed/graph_enriched_corrected.pkl"   # confirmed: GraphStore's own default
MANIFEST_PATH = "data/corpus/manifest_fulltext_sentence.csv"
CHUNKS_PATH = "data/corpus/chunks_600_80_fulltext_sentence_tagged.jsonl"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "supply_chain_docs"
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
PHONEBOOK_PATH = "data/company_phonebook.csv"


def build_real_stores():
    """One Phonebook instance, shared across all three stores -- this is
    the wiring Day 8's log flagged as not yet done. Each store already
    accepts phonebook=None as its default (Day 7), so this is purely
    additive: passing a real Phonebook here doesn't change any store's
    behavior for names outside the 111-company/30-event set, only adds the
    pre-reviewed fast path for names inside it.
    """
    phonebook = Phonebook(PHONEBOOK_PATH)
    graph_store = GraphStore(GRAPH_PATH, phonebook=phonebook)
    supplier_store = SupplierInfoStore(MANIFEST_PATH, CHUNKS_PATH, phonebook=phonebook)
    corpus_store = CorpusSearchStore(
        chunks_path=CHUNKS_PATH,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        model_name=EMBED_MODEL_NAME,
        phonebook=phonebook,
    )
    return graph_store, supplier_store, corpus_store


def build_app(provider: str = "ollama", model: Optional[str] = None):
    """
    Builds everything ONCE: the real stores (expensive -- loads the graph
    pickle, the embedding model, connects to Chroma), the tool bindings,
    the LLM, and the compiled graph. Returns (app, graph_store) --
    graph_store is returned separately because run_all_events.py needs it
    to check whether a guessed seed company name actually resolves BEFORE
    spending a hop (and a real API call) on an event that can't work.

    Split out from run_one_event() specifically so a batch run across many
    events (run_all_events.py) can build this expensive state ONCE and
    reuse it, rather than reloading the graph/corpus/embedding model from
    scratch for every single event -- the same "load once, call many
    times" principle every Day 7 store already follows internally.
    """
    graph_store, supplier_store, corpus_store = build_real_stores()
    llm_tools, raw_dispatch = build_tool_bindings(graph_store, supplier_store, corpus_store)
    llm = build_llm(provider=provider, model=model)
    app = build_agent_graph(llm, llm_tools, raw_dispatch)
    return app, graph_store


def run_one_event(
    app,
    company_name: str,
    event_date: str,
    event_id: Optional[str] = None,
    max_hops: int = 4,
) -> dict:
    """
    Runs a single event through an ALREADY-BUILT app (see build_app()).

    company_name : the disrupted company (the traversal seed) -- e.g. "Aurizon"
    event_date   : 'YYYY-MM-DD', the leakage-guard cutoff for every tool call
    event_id     : optional -- your own event_id (e.g. '11_aurizon_2010'),
                   passed through to get_supplier_info's tag_mismatch
                   diagnostic if the model happens to name it; NOT used for
                   anything else, and never shown to the model as an
                   instruction to match against ground truth (see the Day 8
                   discussion on why the agent must stay blind to that).
    """
    query = (
        f"A disruption has occurred at {company_name}, first reported around "
        f"{event_date}. Investigate which companies are structurally at risk "
        f"and gather evidence on the significant ones."
    )
    return app.invoke(make_initial_state(query, max_hops=max_hops))


def save_trace(state: dict, out_path: str | Path) -> None:
    """
    Writes the hop-by-hop trace to disk -- the exact open item flagged at
    the end of day8_implementation_log.md ("the exact trace-logging format
    to results/traces/ isn't built yet"). Deliberately just the fields
    already tracked in AgentState -- no new bookkeeping invented here --
    since these are exactly what Day 9's by-hand trace reading and a
    future evaluation script both need: what was found, what evidence
    backs it, and what was honestly flagged as missing.
    """
    trace = {
        "hop_count": state["hop_count"],
        "max_hops": state["max_hops"],
        "stop_reason": state["stop_reason"],
        "gap_flags": state["gap_flags"],
        "final_report": state["final_report"],
        "coverage": state["coverage"],
        "evidence_log": state["evidence_log"],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2))
    print(f"Trace saved to {out_path}")


if __name__ == "__main__":
    # [CONFIRM: swap in a real event of your choosing -- this is a
    # placeholder call using the Aurizon event from the Day 8 discussion,
    # not a pre-selected "first" event]
    app, _graph_store = build_app()
    result = run_one_event(
        app,
        company_name="Aurizon",
        event_date="2010-12-25",
        event_id="11_aurizon_2010",
    )

    print(f"stop_reason: {result['stop_reason']}")
    print(f"hops used: {result['hop_count']}/{result['max_hops']}")
    print()
    print(result["final_report"])

    save_trace(result, "results/traces/11_aurizon_2010.json")