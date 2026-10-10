"""Hybrid retrieval: dense + keyword search, fused with Reciprocal Rank Fusion, then re-ranked.

Why hybrid: dense vectors catch paraphrases ("time off" vs "leave"), keyword search
catches exact terms embeddings blur (policy codes, "Form HR-07", product names).
RRF merges the two ranked lists without having to calibrate their raw scores.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .embeddings import Embedder
from .models import ScoredChunk, SearchFilters
from .stores import Store
from .text import content_tokens


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum over lists of 1 / (k + rank(d)), rank starting at 1."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class LexicalReranker:
    """Offline re-ranker: IDF-weighted share of query terms present in the chunk,
    with a bonus when consecutive query terms appear together (phrase match).
    Returns scores in [0, 1]."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = content_tokens(query)
        if not q:
            return [0.0] * len(texts)
        docs = [set(content_tokens(t)) for t in texts]
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (1 + sum(t in d for d in docs))) for t in set(q)}
        total = sum(idf[t] for t in set(q))
        bigrams = list(zip(q, q[1:]))
        scores = []
        for text, toks in zip(texts, docs):
            coverage = sum(idf[t] for t in set(q) if t in toks) / total
            if bigrams:
                seq = content_tokens(text)
                pairs = set(zip(seq, seq[1:]))
                phrase = sum(b in pairs for b in bigrams) / len(bigrams)
            else:
                phrase = 0.0
            scores.append(0.8 * coverage + 0.2 * phrase)
        return scores


class CrossEncoderReranker:
    """Neural re-ranker (optional; needs `pip install sentence-transformers`)."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        raw = self.model.predict([(query, t) for t in texts])
        return [float(1 / (1 + math.exp(-s))) for s in raw]


class HybridRetriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or LexicalReranker()
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, filters: SearchFilters | None = None) -> list[ScoredChunk]:
        filters = filters or SearchFilters()
        qvec = self.embedder.embed([query])[0]
        dense = self.store.vector_search(np.asarray(qvec), self.candidate_k, filters)
        sparse = self.store.keyword_search(query, self.candidate_k, filters)

        by_id = {c.chunk_id: c for c, _ in dense + sparse}
        dense_ids = [c.chunk_id for c, _ in dense]
        sparse_ids = [c.chunk_id for c, _ in sparse]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.rrf_k)
        if not fused:
            return []

        candidates = sorted(fused, key=fused.get, reverse=True)[: self.candidate_k]
        rerank_scores = self.reranker.score(query, [f"{by_id[i].title}\n{by_id[i].text}" for i in candidates])

        # Final score: re-ranker relevance, with fused rank as a tie-breaker.
        max_fused = max(fused[i] for i in candidates)
        results = []
        for cid, rr in zip(candidates, rerank_scores):
            final = 0.85 * rr + 0.15 * (fused[cid] / max_fused)
            results.append(
                ScoredChunk(
                    chunk=by_id[cid],
                    score=round(final, 4),
                    vector_rank=dense_ids.index(cid) + 1 if cid in dense_ids else None,
                    keyword_rank=sparse_ids.index(cid) + 1 if cid in sparse_ids else None,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
