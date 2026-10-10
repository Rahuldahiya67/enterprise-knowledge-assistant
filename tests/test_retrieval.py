from datetime import date

import numpy as np
import pytest

from eka.embeddings import HashingEmbedder
from eka.models import SearchFilters
from eka.retrieval import LexicalReranker, reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused["b"] > fused["a"]
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == "b"


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder(dim=256)
    a = emb.embed(["annual leave policy", "annual leave policy"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    b = emb.embed(["annual leave days", "password encryption laptop"])
    assert float(a[0] @ b[0]) > float(a[0] @ b[1])


def test_reranker_prefers_relevant_text():
    rr = LexicalReranker()
    scores = rr.score("hotel limit per night", ["Hotel stays are capped at Rs 6,000 per night.", "Passwords expire every 90 days."])
    assert scores[0] > scores[1]
    assert 0 <= min(scores) and max(scores) <= 1


@pytest.mark.parametrize(
    "question, source",
    [
        ("How many days of paid annual leave do employees get?", "leave_policy.md"),
        ("What is the hotel cap per night in a metro city?", "travel_expense_policy.md"),
        ("Who approves purchase requisitions above 5 lakh?", "procurement_sop.md"),
        ("When is the dispatch plan frozen?", "dispatch_sop.md"),
        ("How quickly must a lost laptop be reported?", "information_security_policy.md"),
    ],
)
def test_hybrid_retrieval_finds_the_right_document(rag, question, source):
    results = rag.retrieve(question)
    assert results[0].chunk.source == source
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_results_record_which_retriever_found_them(rag):
    results = rag.retrieve("e-way bill lorry receipt dispatch documents")
    assert any(r.vector_rank for r in results)
    assert any(r.keyword_rank for r in results)


def test_department_filter_excludes_other_departments(rag):
    results = rag.retrieve("days", SearchFilters(department="Finance"))
    assert results and all(r.chunk.department == "Finance" for r in results)


def test_effective_date_filter(rag):
    results = rag.retrieve("policy", SearchFilters(effective_after=date(2025, 6, 1)), top_k=20)
    assert results
    assert all(r.chunk.effective_date >= date(2025, 6, 1) for r in results)
