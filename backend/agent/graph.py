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


def _make_initial_state(question: str, session_id: str) -> AgentState:
    """Build the initial agent state for a new question."""
    return {
        "question": question,
        "original_question": question,
        "session_id": session_id,
        "trace": [],
        "tool_results": [],
        "sources": [],
        "confidence": 0.0,
    }


def _format_result(final: AgentState) -> dict[str, Any]:
    """Format the final agent state into a serialisable response dict."""
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
        "answer": final.get("answer", ""),
        "sources": sources,
        "confidence": final.get("confidence", 0.0),
        "tools": tools,
        "trace": final.get("trace", []),
        "intent": final.get("intent", ""),
    }


def run_agent(question: str, session_id: str = "default") -> dict[str, Any]:
    """Execute the agent for a single question and return a serialisable result."""
    agent = get_agent()
    initial = _make_initial_state(question, session_id)
    final: AgentState = agent.invoke(initial)

    answer = final.get("answer", "")
    memory.add_turn(session_id, question, answer)

    result = _format_result(final)
    result["answer"] = answer  # ensure the post-memory answer is used
    return result


def run_agent_streaming(question: str, session_id: str = "default"):
    """Execute agent pre-generation steps, then yield answer tokens in real-time.

    Yields:
        Tuples of (event_type, data):
        - ("token", str): a piece of the generated answer
        - ("done", dict): final metadata (sources, confidence, tools, trace, intent)

    This provides true streaming: routing/retrieval/reranking run first, then
    the LLM generation streams tokens as they arrive rather than buffering the
    entire answer before emitting fake token-by-token output.
    """
    from collections.abc import Iterator
    from rag.generator import get_generator

    agent = get_agent()
    initial = _make_initial_state(question, session_id)
    final: AgentState = agent.invoke(initial)

    # For tool-based answers, the answer is already complete (no LLM streaming).
    # For RAG-based answers with sufficient context, we can re-generate with streaming.
    answer = final.get("answer", "")
    need_tool = final.get("need_tool", False)
    has_contexts = bool(final.get("reranked_docs"))

    if not need_tool and has_contexts and answer:
        # True streaming: re-generate using the LLM stream interface.
        # The answer was already generated in the agent run, but we re-stream it
        # for real-time output. In production, the generate_node could be skipped
        # and streaming done here directly.
        generator = get_generator()
        streamed_chunks: list[str] = []
        for token in generator.generate_stream(
            final.get("question", question), final.get("reranked_docs", [])
        ):
            streamed_chunks.append(token)
            yield ("token", token)
        # Use the streamed version as the authoritative answer.
        answer = "".join(streamed_chunks)
    else:
        # Non-streamable answer (tool results, rejections): emit as a single chunk.
        yield ("token", answer)

    memory.add_turn(session_id, question, answer)

    done_payload = _format_result(final)
    done_payload["answer"] = answer  # use the (possibly re-streamed) answer
    yield ("done", done_payload)
