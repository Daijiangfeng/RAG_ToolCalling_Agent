"""Tests for the shared tokenizer and evaluation regression comparison."""

from rag.evaluator import _overlap_ratio, compare_to_baseline
from rag.text import tokenize


def test_tokenize_splits_mixed_cjk_and_ascii():
    # A mixed run must not collapse into a single token; the ASCII word "rag"
    # is kept whole while CJK characters are split individually.
    assert tokenize("什么是RAG") == ["什", "么", "是", "rag"]


def test_overlap_ratio_no_longer_swallows_mixed_run():
    # Before the tokenizer fix "针对RAG" was one token that never matched the
    # document's "rag", forcing overlap to 0. It should now be positive.
    assert _overlap_ratio("针对RAG的优势", "RAG 的优势包括减少幻觉") > 0.0


def test_compare_to_baseline_no_baseline_is_advisory():
    summary = {"generation": {"faithfulness": 0.9}}
    # With an explicit empty baseline the comparison is unavailable, never raises.
    result = compare_to_baseline(summary, baseline={})
    assert result["available"] is False
    assert result["regressions"] == []


def test_compare_to_baseline_flags_regression_but_does_not_raise():
    baseline = {
        "generation": {"faithfulness": 0.9, "answer_relevance": 0.8, "context_relevance": 0.8},
        "retrieval": {"precision_at_k": 0.7, "recall_at_k": 0.7},
        "safety": {"hallucination_rate": 0.1},
    }
    # Faithfulness drops well beyond tolerance; hallucination rate spikes.
    current = {
        "generation": {"faithfulness": 0.5, "answer_relevance": 0.8, "context_relevance": 0.8},
        "retrieval": {"precision_at_k": 0.7, "recall_at_k": 0.7},
        "safety": {"hallucination_rate": 0.4},
    }
    result = compare_to_baseline(current, baseline=baseline)
    assert result["available"] is True
    flagged = {r["metric"] for r in result["regressions"]}
    assert "generation.faithfulness" in flagged
    assert "safety.hallucination_rate" in flagged
