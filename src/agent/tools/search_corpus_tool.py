"""
search_corpus(query, event_date, ...) -- Layer 3 agent tool.
(Renamed from search_news() -- see rationale below.)

Built directly on the hybrid retrieval logic already validated against 20
hand-written test queries in retrieval_test.py -- the BM25 + semantic +
reciprocal-rank-fusion mechanics here are UNCHANGED from that validated
version, same default parameters (n_candidates=50, k=60). This module adds
the agent-tool contract around that already-working core:

  - a structured, citable result (CorpusHit), not raw (chunk_id, score)
    tuples -- so Layer 5 can cite exactly which document backed a claim
  - an explicit `status`, so "nothing found" is a visible signal the
    agent's Day 8 stopping-condition logic can act on, not an empty list
    it has to interpret
  - the leakage guard exposed as a plain `event_date` string -- the caller
    never constructs a raw published_int_max filter by hand
  - resolve_company()'s exact-match safety net (ported from
    retrieval_test.py) applied AUTOMATICALLY whenever a company filter is
    used, with a recorded warning on any fuzzy resolution, and a dedicated
    `company_not_resolved` status distinct from a genuine empty search --
    the same "don't collapse different kinds of nothing into one kind of
    nothing" principle already applied in get_supplier_info()
  - load-once construction (CorpusSearchStore), mirroring
    SupplierInfoStore, instead of retrieval_test.py's module-level
    load-on-import side effects (which would fire expensive model loading
    every time this file is merely imported, e.g. by a test runner)
  - a loud, fatal check on Chroma/corpus count mismatch at construction
    time, rather than a printed warning that's easy to miss -- same
    "absence must be fatal and loud" principle as Day 5's Bug 5-A fix

Rename rationale: the corpus this tool searches is SEC annual reports plus
one synthetic trigger document per event (Day 5 log, Section 2.10) -- never
real news articles. "search_news" was inherited from the Project Overview's
more idealised description; "search_corpus" matches the vocabulary this
project already uses everywhere else (corpus construction, corpus coverage
problem, etc.) and doesn't imply a news archive that isn't actually there.

Heavy imports (chromadb, torch, sentence_transformers, rank_bm25) are
deliberately deferred to inside __init__, not placed at module level --
this lets the pure functions and the orchestration logic in this file be
imported and unit-tested (see test_search_corpus_logic.py) in an
environment that doesn't have those packages installed at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class CorpusHit:
    chunk_id: str
    doc_id: Optional[str]
    company: Optional[str]
    published_date: Optional[str]
    published_int: Optional[int]
    form: Optional[str]
    url: Optional[str]
    text: str
    score: float
    rank: int


@dataclass
class CorpusSearchResult:
    status: str                 # found | no_results | company_not_resolved
    query: str
    event_date: str
    hits: list = field(default_factory=list)              # list[CorpusHit]
    distinct_companies: list = field(default_factory=list)

    bm25_hit_count: int = 0
    semantic_hit_count: int = 0
    fused_hit_count: int = 0

    company_filter_requested: Optional[str] = None
    company_filter_resolved: Optional[str] = None
    warnings: list = field(default_factory=list)

    def as_observation(self) -> str:
        """Plain-text form to hand back to the LangGraph agent."""
        if self.status == "no_results":
            return (f"No corpus matches found for query '{self.query}' "
                    f"(as of {self.event_date}).")
        if self.status == "company_not_resolved":
            return (f"Requested company filter '{self.company_filter_requested}' "
                    f"does not match any company in the corpus -- treat as "
                    f"confirmed absent from the corpus, not as a failed search.")
        n = len(self.distinct_companies)
        lines = [f"[query: '{self.query}' | as of {self.event_date} | "
                 f"{len(self.hits)} result(s) across {n} distinct "
                 f"compan{'y' if n == 1 else 'ies'}]"]
        for h in self.hits:
            lines.append(f"\n({h.rank}) {h.company} | {h.published_date} | "
                         f"score={h.score:.4f} | doc_id={h.doc_id}")
            lines.append(h.text)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure functions -- no model/DB dependency, directly unit-testable
# ---------------------------------------------------------------------------

def date_to_published_int(event_date: str) -> int:
    """'2011-03-11' -> 20110311, matching the corpus's own encoding."""
    dt = datetime.strptime(event_date[:10], "%Y-%m-%d")
    return int(dt.strftime("%Y%m%d"))


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def matches_filter(chunk: dict, filters: dict) -> bool:
    """Unchanged from retrieval_test.py -- applies the leakage guard and
    any equality filters (e.g. company) to a single chunk dict."""
    for key, value in filters.items():
        if key == "published_int_max":
            if chunk.get("published_int", 0) > value:
                return False
        else:
            if chunk.get(key) != value:
                return False
    return True


def build_chroma_where(filters: Optional[dict]):
    """Unchanged from retrieval_test.py -- translates the same filter dict
    into a Chroma `where` clause."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if key == "published_int_max":
            conditions.append({"published_int": {"$lte": value}})
        else:
            conditions.append({key: value})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Unchanged from retrieval_test.py."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def resolve_company_name(keyword: str, companies: list[str]) -> Optional[str]:
    """Pulled out as a pure function (retrieval_test.py had this as a
    print-and-return method) so it's testable without a loaded corpus."""
    matches = [c for c in companies if keyword.lower() in c.lower()]
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# The store -- loads everything ONCE, exposes search_corpus() as the tool
# ---------------------------------------------------------------------------

class CorpusSearchStore:
    """
    Build one of these once per agent process (mirrors SupplierInfoStore in
    supplier_info.py). Loading the chunk file, building BM25, loading the
    embedding model, and connecting to Chroma are all expensive -- doing
    this once here, rather than on every call or on module import, keeps a
    multi-hop, multi-event agent run affordable.
    """

    def __init__(
        self,
        chunks_path: str | Path,
        chroma_path: str | Path,
        collection_name: str = "supply_chain_docs",
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    ):
        import json
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        import torch

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self._chunk_lookup = {c["chunk_id"]: c for c in self.chunks}
        self._companies = sorted(set(c["company"] for c in self.chunks))

        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(name=collection_name)

        if self.collection.count() != len(self.chunks):
            raise RuntimeError(
                f"Chroma has {self.collection.count()} chunks but the corpus "
                f"file has {len(self.chunks)} -- refusing to proceed. A silent "
                f"mismatch here would produce quietly-wrong retrieval on "
                f"every future call, not just this one. Same principle as "
                f"Day 5 Bug 5-A: absence/mismatch must be fatal and loud, "
                f"not a printed warning that's easy to miss."
            )

    # -- company-name safety net --------------------------------------------
    def resolve_company(self, keyword: str) -> Optional[str]:
        return resolve_company_name(keyword, self._companies)

    # -- the two ranking paths ------------------------------------------------
    def _bm25_rank(self, query: str, filters: Optional[dict]) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        candidates = list(zip(self.chunks, scores))
        if filters:
            candidates = [(c, s) for c, s in candidates if matches_filter(c, filters)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c["chunk_id"] for c, s in candidates]

    def _chroma_rank(self, query: str, filters: Optional[dict], n_results: int) -> list[str]:
        query_embedding = self.model.encode(f"search_query: {query}")
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=build_chroma_where(filters),
        )
        return results["ids"][0]

    # -- the tool ---------------------------------------------------------------
    def search_corpus(
        self,
        query: str,
        event_date: str,
        company: Optional[str] = None,
        top_k: int = 5,
        n_candidates: int = 50,
        k: int = 60,
    ) -> CorpusSearchResult:
        """
        query      : free text -- no company name required. This is the
                     tool's whole point: it can surface a company the agent
                     never named, unlike get_supplier_info().
        event_date : 'YYYY-MM-DD' -- leakage guard, translated internally
                     to published_int_max and applied to BOTH retrieval
                     paths. The caller never touches published_int directly.
        company    : optional -- narrows the search to one company.
                     Resolved via resolve_company() before any ranking is
                     attempted, so a misspelling is reported as
                     `company_not_resolved`, never silently indistinguishable
                     from a genuine zero-result search.
        """
        warnings: list[str] = []
        filters: dict = {"published_int_max": date_to_published_int(event_date)}

        resolved_company = None
        if company is not None:
            resolved_company = self.resolve_company(company)
            if resolved_company is None:
                return CorpusSearchResult(
                    status="company_not_resolved",
                    query=query, event_date=event_date,
                    company_filter_requested=company,
                )
            if resolved_company.lower() != company.strip().lower():
                warnings.append(f"'{company}' resolved to '{resolved_company}'")
            filters["company"] = resolved_company

        bm25_list = self._bm25_rank(query, filters)[:n_candidates]
        chroma_list = self._chroma_rank(query, filters, n_results=n_candidates)
        fused = reciprocal_rank_fusion([bm25_list, chroma_list], k=k)

        if not fused:
            return CorpusSearchResult(
                status="no_results",
                query=query, event_date=event_date,
                bm25_hit_count=len(bm25_list), semantic_hit_count=len(chroma_list),
                company_filter_requested=company, company_filter_resolved=resolved_company,
                warnings=warnings,
            )

        top = fused[:top_k]
        hits = []
        for rank, (chunk_id, score) in enumerate(top, start=1):
            c = self._chunk_lookup.get(chunk_id, {})
            hits.append(CorpusHit(
                chunk_id=chunk_id,
                doc_id=c.get("doc_id"),
                company=c.get("company"),
                published_date=c.get("published_date"),
                published_int=c.get("published_int"),
                form=c.get("form"),
                url=c.get("url"),
                text=c.get("text", ""),
                score=score,
                rank=rank,
            ))

        distinct_companies = sorted({h.company for h in hits if h.company})

        return CorpusSearchResult(
            status="found",
            query=query, event_date=event_date,
            hits=hits, distinct_companies=distinct_companies,
            bm25_hit_count=len(bm25_list), semantic_hit_count=len(chroma_list),
            fused_hit_count=len(fused),
            company_filter_requested=company, company_filter_resolved=resolved_company,
            warnings=warnings,
        )