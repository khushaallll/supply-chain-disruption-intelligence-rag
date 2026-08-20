"""
llm_setup.py -- builds the LangChain chat-model object the agent
graph actually binds to.

Kept as its own tiny file, separate from agent_graph.py, for one reason:
agent_graph.build_agent_graph(llm, llm_tools, raw_dispatch) was deliberately
built during Day 8 to accept ANY LangChain-compatible chat model -- the
mocked tests prove this by passing in a fake object with a matching
.bind_tools()/.invoke() shape instead of a real one (see
test_agent_mocked.py's ScriptedLLM). Swapping providers or models should
mean changing ONLY this file. agent_graph.py, agent_tools.py,
stopping_condition.py, and agent_state.py never need to change for it.

TWO PROVIDERS SUPPORTED:

  "groq" (now the default) -- hosted, free-tier, open-weight models served
  fast. Picked specifically because local CPU-only inference (the original
  Day 9 plan, via Ollama) proved too slow to iterate on before the
  project's hard deadline. Requires a free API key: sign up at
  https://console.groq.com, generate a key, and set it as the
  GROQ_API_KEY environment variable before running anything that calls
  build_llm().

  "ollama" -- local, free, no rate limit, but CPU-bound on this machine.
  Kept available for whenever GPU access arrives, or for anything that
  shouldn't depend on an external service's uptime or quota.

IMPORTANT ON GROQ'S FREE TIER: free-tier daily token budgets are
model-specific and DO change over time -- check
https://console.groq.com/docs/rate-limits and
https://console.groq.com/docs/models for current numbers before
committing to a model for the full 30-event run. As a rough planning
number: this agent's own transcripts resend the full running conversation
to the model on every hop (see agent_graph.py's agent_node), so token use
compounds across a run's 4 hops rather than staying flat -- a single event
can plausibly use somewhere in the ballpark of 15-20K tokens once you
include get_supplier_info's real filing text. On a ~100K-tokens/day free
tier, that is realistically only a handful of events per day on the
largest hosted models, not all 30 in one sitting. If you hit 429 (rate
limit) errors partway through a batch run, that is expected given this
math, not a bug -- either spread the run across more than one day, or
switch DEFAULT_GROQ_MODEL to a smaller model with a larger free daily
quota (check the rate-limits page above first, since the right answer
here changes).
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_OLLAMA_MODEL = "gpt-oss:120b"

# Raised now that GPU VRAM is available -- the earlier 8192 was sized
# around an 8GB-RAM CPU-only machine's real constraints (see the "Revised"
# note in the old version of this file, and the Aurizon trace that showed
# exactly this failure: a get_supplier_info observation alone can be
# ~4,800+ tokens, and a too-small context window silently drops the OLDEST
# tokens first -- meaning the system prompt and the original task, not the
# newest tool output). On an H100 with ~90GB free, that tradeoff no longer
# applies -- 32768 gives comfortable headroom for a full 4-hop transcript
# (system prompt + growing history + several evidence-tool observations)
# without needing to trim what any tool returns.
DEFAULT_OLLAMA_NUM_CTX = 32768

# llama-3.3-70b-versatile is confirmed hosted on Groq, fast, and documented
# specifically for tool-use workloads -- a reasonable starting point given
# reliable tool-calling matters more here than raw chat quality. Swap this
# if its free-tier quota proves too tight in practice (see module docstring).
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def build_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
):
    """
    temperature=0.0 by default regardless of provider -- deliberate, not an
    oversight. This is an EVALUATION run feeding an ablation (RQ2), not a
    chat assistant: the same event fed through System C should behave as
    reproducibly as an LLM realistically can. Raise this only for
    exploratory, by-hand poking around -- never for a run whose trace ends
    up in a results table.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it as an "
                "environment variable before calling build_llm()."
            )
        return ChatGroq(model=model or DEFAULT_GROQ_MODEL, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or DEFAULT_OLLAMA_MODEL,
            num_ctx=num_ctx or DEFAULT_OLLAMA_NUM_CTX,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider '{provider}' -- use 'groq' or 'ollama'.")