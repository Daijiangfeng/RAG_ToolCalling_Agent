"""End-to-end tests for the LangGraph agent."""

from agent.graph import run_agent


def test_agent_rag_answer_with_sources():
    result = run_agent("RAG 的优势是什么?", session_id="t-rag")
    assert result["intent"] == "rag"
    assert result["sources"], "RAG answer should carry sources"
    assert result["answer"]
    steps = [s["step"] for s in result["trace"]]
    assert "retrieval" in steps and "rerank" in steps


def test_agent_calculator_tool():
    result = run_agent("12345*678", session_id="t-calc")
    assert result["tools"], "should invoke a tool"
    tool = result["tools"][0]
    assert tool["tool"] == "calculator"
    assert tool["output"]["result"] == 12345 * 678


def test_agent_web_search_tool():
    result = run_agent("最新的 AI 新闻有哪些?", session_id="t-web")
    assert result["tools"] and result["tools"][0]["tool"] == "web_search"


def test_agent_rejects_unknown_question():
    result = run_agent("请给出本产品在火星基地的部署步骤。", session_id="t-reject")
    assert "知识库中没有足够信息" in result["answer"]


def test_trace_only_has_summaries():
    result = run_agent("什么是 RAG?", session_id="t-trace")
    for step in result["trace"]:
        assert "summary" in step and isinstance(step["summary"], str)
