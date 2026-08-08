"""Intent router node.

Determines whether the question needs RAG (knowledge base) and/or a tool, and
which tool.  Uses fast deterministic rules first (math / realtime / datetime /
file query) and falls back to the LLM for the RAG-vs-tool decision.  Writes a
short reasoning *summary* to the trace (never the full chain-of-thought).
"""

from __future__ import annotations

import re

from agent.state import AgentState
from app.llm import get_llm
from app.logging import get_logger
from tools.calculator import normalize_math_expr

logger = get_logger(__name__)

# Rule patterns -----------------------------------------------------------
_MATH_RE = re.compile(r"^[\s\d\.\+\-\*/%\(\)×xX÷^]+$")
_MATH_HINT_RE = re.compile(r"(计算|等于|多少|算一下|求值|\d+\s*[\+\-\*/×÷]\s*\d+)")
_REALTIME_RE = re.compile(r"(最新|实时|今天|现在|近期|最近|新闻|天气|股价|热点|latest|news|today|current)")
_DATETIME_RE = re.compile(r"(几点|日期|星期|今天是|现在时间|多少天后|多少天前|date|time)")
_FILE_RE = re.compile(r"(文件|文档|上传的|这篇|该文档|附件)")

ROUTER_SYSTEM = (
    "ROUTER: 你是意图路由器。判断用户问题应使用哪种能力,只输出一个词:"
    "rag(需要查询知识库)/ calculator / web_search / datetime / file_query。"
)


def classify(question: str) -> dict:
    """Return the routing decision for a question."""
    q = question.strip()

    # 1) Pure arithmetic expression -> calculator.
    stripped = normalize_math_expr(q.rstrip("=?？"))
    if _MATH_RE.match(q) and re.search(r"\d", q) and re.search(r"[\+\-\*/%×xX÷^]", q):
        return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "calculator",
                "summary": "问题是纯算术表达式,选择 calculator 工具计算。"}

    # 2) Datetime intent.
    if _DATETIME_RE.search(q) and not _REALTIME_RE.search(q.replace("今天", "").replace("现在", "")):
        # "现在几点/今天日期" -> datetime; "最新新闻" handled below.
        if re.search(r"(几点|日期|星期|时间|多少天)", q):
            return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "datetime",
                    "summary": "问题涉及当前时间/日期计算,选择 datetime 工具。"}

    # 3) Realtime / news -> web_search.
    if _REALTIME_RE.search(q):
        return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "web_search",
                "summary": "问题涉及实时/最新信息,超出知识库范围,选择 web_search 工具。"}

    # 4) Explicit math hint with numbers.
    if _MATH_HINT_RE.search(q) and re.search(r"\d", q):
        expr = _extract_expression(q)
        if expr:
            return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "calculator",
                    "summary": "问题包含算术计算,选择 calculator 工具。"}

    # 5) File query.
    if _FILE_RE.search(q):
        return {"intent": "rag", "need_rag": True, "need_tool": False, "tool_name": "",
                "summary": "问题针对已上传文档内容,走知识库检索(RAG)。"}

    # 6) Default: ask the LLM, defaulting to RAG.
    decision = _llm_route(q)
    if decision == "calculator":
        return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "calculator",
                "summary": "LLM 判断需要计算,选择 calculator。"}
    if decision == "web_search":
        return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "web_search",
                "summary": "LLM 判断需要实时信息,选择 web_search。"}
    if decision == "datetime":
        return {"intent": "tool", "need_rag": False, "need_tool": True, "tool_name": "datetime",
                "summary": "LLM 判断需要时间信息,选择 datetime。"}
    return {"intent": "rag", "need_rag": True, "need_tool": False, "tool_name": "",
            "summary": "默认走知识库检索(RAG)以获取有依据的回答。"}


def _extract_expression(q: str) -> str:
    m = re.search(r"[\d\.\s\+\-\*/%×xX÷\(\)\^]{3,}", normalize_math_expr(q))
    return m.group(0).strip() if m else ""


def _llm_route(question: str) -> str:
    try:
        out = get_llm().complete(
            [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": f"意图: {question}"}]
        ).strip().lower()
    except Exception:  # pragma: no cover
        return "rag"
    for token in ("calculator", "web_search", "datetime", "file_query", "rag"):
        if token in out:
            return "rag" if token == "file_query" else token
    return "rag"


def intent_router(state: AgentState) -> AgentState:
    question = state["question"]
    decision = classify(question)
    state["intent"] = decision["intent"]
    state["need_rag"] = decision["need_rag"]
    state["need_tool"] = decision["need_tool"]
    state["tool_name"] = decision["tool_name"]
    state.setdefault("trace", []).append(
        {"step": "intent_router", "summary": decision["summary"], "tool": decision["tool_name"] or None,
         "data": {"intent": decision["intent"]}}
    )
    logger.info("Router: intent=%s tool=%s", decision["intent"], decision["tool_name"])
    return state
