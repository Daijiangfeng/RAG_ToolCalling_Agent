"""Tests for the agent intent router."""

from agent.router import classify


def test_pure_math_routes_to_calculator():
    d = classify("12345*678")
    assert d["need_tool"] and d["tool_name"] == "calculator"


def test_calc_hint_routes_to_calculator():
    d = classify("帮我计算 (3+4)*25 等于多少?")
    assert d["tool_name"] == "calculator"


def test_realtime_routes_to_web_search():
    d = classify("最新的 AI 新闻有哪些?")
    assert d["tool_name"] == "web_search"


def test_datetime_routes_to_datetime_tool():
    d = classify("现在几点了?今天星期几?")
    assert d["tool_name"] == "datetime"


def test_knowledge_question_routes_to_rag():
    d = classify("RAG 的完整链路包含哪些阶段?")
    assert d["need_rag"] and not d["need_tool"]
