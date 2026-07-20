"""Tests for retrieval over the seeded knowledge base."""

from rag.retriever import get_retriever


def test_similarity_search_returns_results():
    retriever = get_retriever()
    results = retriever.similarity_search("RAG 的优势是什么", top_k=5)
    assert results
    assert "text" in results[0] and "score" in results[0]


def test_retrieval_relevance():
    retriever = get_retriever()
    results = retriever.similarity_search("LangGraph 核心概念", top_k=5)
    joined = " ".join(r["text"] for r in results)
    assert "LangGraph" in joined or "State" in joined
