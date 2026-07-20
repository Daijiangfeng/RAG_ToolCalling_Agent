"""Agent state definition for the LangGraph StateGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # Input
    question: str
    original_question: str
    session_id: str

    # Routing
    intent: str
    need_rag: bool
    need_tool: bool
    tool_name: str

    # RAG
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    confidence: float

    # Tools
    tool_results: list[dict[str, Any]]

    # Generation
    answer: str
    critique_passed: bool
    regenerated: bool

    # Observability
    trace: list[dict[str, Any]]
