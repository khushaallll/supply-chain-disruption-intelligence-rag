"""
llm_setup.py -- Day 9: builds the LangChain chat-model object the agent
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
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

DEFAULT_OLLAMA_MODEL = "qwen3:4b"

# RAISED from an earlier 4096 after a real trace showed the actual failure
# mode: get_supplier_info defaults to returning up to 8 chunks of real
# filing text, and this project's corpus was built with ~600-token chunks
# -- so a SINGLE tool observation can be ~4,800+ tokens, which alone
# already exceeded the old 4096 budget before the system prompt, the
# original question, or the hop-1 exchange were even counted. Ollama
# doesn't error on overflow -- it silently drops the OLDEST tokens, which
# means the system prompt and the original task are exactly what gets
# pushed out first. On a real Aurizon trace, this produced a model that
# had genuinely lost all memory of the investigation by hop 2 and just
# wrote a generic financial-analysis essay about whatever filing text was
# still in view. 8192 is not a proven-safe number either -- it's sized to
# comfortably fit one full get_supplier_info call plus overhead, not
# unlimited growth across all 4 hops. Watch actual RAM use (`ollama ps`)
# on a real run; if it's too tight on an 8GB machine, the next lever to
# pull is LOWERING max_chunks on get_supplier_info calls (agent_tools.py),
# not silently shrinking this back down -- a smaller context budget that
# quietly drops the task instructions again is a worse failure than a
# slower or more constrained run.
DEFAULT_OLLAMA_NUM_CTX = 8192

# llama-3.3-70b-versatile is confirmed hosted on Groq, fast, and documented
# specifically for tool-use workloads -- a reasonable starting point given
# reliable tool-calling matters more here than raw chat quality. Swap this
# if its free-tier quota proves too tight in practice (see module docstring).
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

def build_llm(
    provider: str = "groq",
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

        if not os.getenv("GROQ_API_KEY"):
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