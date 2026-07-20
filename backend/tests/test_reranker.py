"""Tests for the reranker (before/after ordering)."""

from rag.reranker import LexicalReranker


def _docs():
    return [
        {"text": "今天天气很好,适合出门散步。", "score": 0.9, "metadata": {}},
        {"text": "RAG 使用 Cross-Encoder 重排序从候选中选出最相关的片段。", "score": 0.5, "metadata": {}},
        {"text": "猫是一种常见的宠物动物。", "score": 0.4, "metadata": {}},
    ]


def test_rerank_promotes_relevant_doc():
    reranker = LexicalReranker()
    ranked = reranker.rerank("Cross-Encoder 重排序是什么", _docs(), top_n=3)
    assert "重排序" in ranked[0]["text"]
    assert ranked[0]["rerank_score"] >= ranked[-1]["rerank_score"]


def test_rerank_compare_shape():
    reranker = LexicalReranker()
    cmp = reranker.rerank_compare("重排序", _docs(), top_n=2)
    assert "before" in cmp and "after" in cmp
    assert len(cmp["after"]) == 2
