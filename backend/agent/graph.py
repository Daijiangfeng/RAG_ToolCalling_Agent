"""LangGraph StateGraph assembling the agent workflow.

Flow::

    START
      -> intent_router
           -(need_tool)-> tool_node -----------------\
           -(need_rag)--> rewrite -> retrieve -> rerank
                                                   |
                                    (confidence<thr)-> reject -> END
                                    (else)-----------> generate
      tool_node -----------------------------------> generate
      generate -> critique
           -(pass)-> END
           -(fail, first time)-> generate  (single regeneration)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent import memory, nodes
from agent.router import intent_router
from agent.state import AgentState
from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)


# --- Conditional edge functions -------------------------------------------
def _route_after_router(state: AgentState) -> str:
    return "tool" if state.get("need_tool") else "rag"


def _route_after_rerank(state: AgentState) -> str:
    confidence = state.get("confidence", 0.0)
    has_ctx = bool(state.get("reranked_docs"))
    if not has_ctx or confidence < settings.confidence_threshold:
        return "reject"
    return "generate"


def _route_after_critique(state: AgentState) -> str:
    if state.get("critique_passed"):
        return "end"
    if state.get("regenerated"):
        # Already retried once; accept to avoid loops.
        return "end"
    state["regenerated"] = True
    return "regenerate"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("rewrite", nodes.rewrite_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("rerank", nodes.rerank_node)
    graph.add_node("tool", nodes.tool_node)
    graph.add_node("reject", nodes.reject_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("critique", nodes.critique_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router", _route_after_router, {"tool": "tool", "rag": "rewrite"}
    )

    # RAG branch
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank", _route_after_rerank, {"reject": "reject", "generate": "generate"}
    )
    graph.add_edge("reject", END)

    # Tool branch joins generation
    graph.add_edge("tool", "generate")

    # Generation -> self critique -> (retry once | end)
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique", _route_after_critique, {"regenerate": "generate", "end": END}
    )

    return graph.compile()


_compiled = None


def get_agent():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_agent(question: str, session_id: str = "default") -> dict[str, Any]:
    """Execute the agent for a single question and return a serialisable result."""
    agent = get_agent()
    initial: AgentState = {
        "question": question,
        "original_question": question,
        "session_id": session_id,
        "trace": [],
        "tool_results": [],
        "sources": [],
        "confidence": 0.0,
    }
    final: AgentState = agent.invoke(initial)

    answer = final.get("answer", "")
    memory.add_turn(session_id, question, answer)

    tools = [
        {"tool": r["tool"], "input": r["input"], "output": r["output"]}
        for r in final.get("tool_results", [])
    ]
    sources = [
        {
            "text": s.get("text", ""),
            "score": round(float(s.get("rerank_score", s.get("score", 0.0))), 4),
            "metadata": s.get("metadata", {}),
        }
        for s in final.get("sources", [])
    ]
    return {
        "answer": answer,
        "sources": sources,
        "confidence": final.get("confidence", 0.0),
        "tools": tools,
        "trace": final.get("trace", []),
        "intent": final.get("intent", ""),
    }
