"""LangGraph node functions.

Each node takes and returns an :class:`AgentState`, appending a concise reasoning
*summary* to ``state['trace']`` (we deliberately do NOT store full chain-of-
thought, only a short summary per the requirements).
"""

from __future__ import annotations

import re
import time

from agent import memory
from agent.state import AgentState
from app.config import settings
from app.llm import get_llm
from app.logging import get_logger
from rag.advanced import hyde, needs_rewrite, query_rewrite
from rag.generator import context_compression, get_generator
from rag.reranker import get_reranker
from rag.retriever import get_retriever
from tools import registry

logger = get_logger(__name__)

CRITIQUE_SYSTEM = (
    "CRITIQUE: 你是回答质量校验器。请判断答案是否:(1)有上下文事实依据;(2)确实回答了问题;"
    "(3)没有编造(幻觉)。若全部满足输出以 'PASS' 开头,否则以 'FAIL' 开头并简述原因。"
)


def _trace(state: AgentState, step: str, summary: str, tool: str | None = None, **data) -> None:
    state.setdefault("trace", []).append(
        {"step": step, "summary": summary, "tool": tool, "data": data}
    )


# ---------------------------------------------------------------------------
# RAG branch
# ---------------------------------------------------------------------------
def rewrite_node(state: AgentState) -> AgentState:
    question = state["question"]
    history = memory.get_history(state.get("session_id", "default"))
    new_q = question
    used = []
    if needs_rewrite(question, history):
        new_q = query_rewrite(question, history)
        used.append("query_rewrite")
    # HyDE: conditionally expand the query with a hypothetical answer.
    # Short, simple queries (< 15 chars) are unlikely to benefit from HyDE
    # and the extra LLM call adds latency. Only apply for longer/complex queries.
    if len(new_q) >= 15:
        retrieval_query = hyde(new_q)
        used.append("hyde")
    else:
        retrieval_query = new_q
        used.append("hyde_skipped(短查询)")
    state["question"] = new_q
    state["retrieval_query"] = retrieval_query  # type: ignore[typeddict-unknown-key]
    _trace(
        state,
        "query_rewrite",
        f"应用 Advanced RAG:{'、'.join(used)}；检索查询已优化。"
        + (f" 改写后问题:{new_q}" if new_q != question else ""),
    )
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents, with multi-step decomposition for complex queries."""
    from rag.decomposer import decompose_question

    retriever = get_retriever()
    query = state.get("retrieval_query") or state["question"]  # type: ignore[attr-defined]

    # Multi-step retrieval: decompose complex questions into sub-queries
    sub_queries = decompose_question(query)
    t0 = time.perf_counter()

    if len(sub_queries) > 1:
        # Multiple sub-queries: retrieve for each and fuse results via RRF
        all_docs = []
        seen_texts = set()
        for sq in sub_queries:
            docs = retriever.similarity_search(sq, top_k=settings.top_k)
            for doc in docs:
                text = doc.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_docs.append(doc)

        # Limit total to top_k * 1.5 (extra docs from multi-query)
        max_docs = int(settings.top_k * 1.5)
        docs = all_docs[:max_docs]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        state["retrieved_docs"] = docs
        _trace(
            state,
            "retrieval",
            f"多步检索: {len(sub_queries)} 个子查询 → {len(docs)} 去重候选片段，耗时 {retrieval_ms:.0f}ms。",
            data_count=len(docs),
            latency_ms=round(retrieval_ms, 1),
        )
    else:
        # Single query: original path
        docs = retriever.similarity_search(query, top_k=settings.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        state["retrieved_docs"] = docs
        _trace(
            state,
            "retrieval",
            f"向量检索返回 {len(docs)} 个候选片段(top_k={settings.top_k})，耗时 {retrieval_ms:.0f}ms。",
            data_count=len(docs),
            latency_ms=round(retrieval_ms, 1),
        )
    return state


def rerank_node(state: AgentState) -> AgentState:
    reranker = get_reranker()
    docs = state.get("retrieved_docs", [])
    t0 = time.perf_counter()
    reranked = reranker.rerank(state["question"], docs, top_n=settings.rerank_top_n)
    rerank_ms = (time.perf_counter() - t0) * 1000
    compressed = context_compression(reranked)
    confidence = float(reranked[0]["rerank_score"]) if reranked else 0.0
    state["reranked_docs"] = compressed
    state["sources"] = compressed
    state["confidence"] = round(confidence, 4)
    _trace(
        state,
        "rerank",
        f"Cross-Encoder 重排序保留 Top {len(compressed)} 上下文，最高分 {confidence:.3f}，耗时 {rerank_ms:.0f}ms。",
        top_score=confidence,
        kept=len(compressed),
        latency_ms=round(rerank_ms, 1),
    )
    return state


# ---------------------------------------------------------------------------
# Tool branch
# ---------------------------------------------------------------------------
def tool_node(state: AgentState) -> AgentState:
    tool_name = state.get("tool_name", "")
    args = _build_tool_args(tool_name, state["question"])
    output = registry.execute(tool_name, args)
    result = {"tool": tool_name, "input": args, "output": output}
    state.setdefault("tool_results", []).append(result)
    _trace(
        state,
        "tool_call",
        f"调用工具 {tool_name},参数 {args}。",
        tool=tool_name,
        output=output,
    )
    return state


def _build_tool_args(tool_name: str, question: str) -> dict:
    """Generate tool arguments from the question (mimics LLM arg generation)."""
    if tool_name == "calculator":
        expr = _extract_math(question)
        return {"expression": expr or question}
    if tool_name == "web_search":
        return {"query": question, "max_results": 3}
    if tool_name == "datetime":
        m = re.search(r"(-?\d+)\s*天(后|前)", question)
        if m:
            days = int(m.group(1)) * (1 if m.group(2) == "后" else -1)
            return {"offset_days": days}
        return {"offset_days": 0}
    if tool_name == "file_query":
        return {"query": question, "top_k": 5}
    return {}


def _extract_math(q: str) -> str:
    norm = q.replace("×", "*").replace("÷", "/").replace("x", "*").replace("X", "*").replace("^", "**")
    m = re.search(r"[\d\.\s\+\-\*/%\(\)]{3,}", norm)
    return m.group(0).strip().rstrip("=") if m else ""


# ---------------------------------------------------------------------------
# Generation + critique + rejection
# ---------------------------------------------------------------------------
def reject_node(state: AgentState) -> AgentState:
    state["answer"] = settings.rejection_message
    state["sources"] = []
    _trace(
        state,
        "reject",
        f"置信度 {state.get('confidence', 0.0):.3f} 低于阈值 {settings.confidence_threshold} 或无有效上下文,触发拒答机制。",
    )
    return state


def generate_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    if state.get("need_tool"):
        answer = _generate_from_tools(state)
        gen_ms = (time.perf_counter() - t0) * 1000
        _trace(state, "generation", f"基于工具执行结果生成回答，耗时 {gen_ms:.0f}ms。", latency_ms=round(gen_ms, 1))
    else:
        generator = get_generator()
        answer = generator.generate(state["question"], state.get("reranked_docs", []))
        gen_ms = (time.perf_counter() - t0) * 1000
        _trace(
            state, "generation",
            f"基于 {len(state.get('reranked_docs', []))} 条压缩上下文生成回答，耗时 {gen_ms:.0f}ms。",
            latency_ms=round(gen_ms, 1),
        )
    state["answer"] = answer
    return state


def _generate_from_tools(state: AgentState) -> str:
    parts = []
    for r in state.get("tool_results", []):
        tool = r["tool"]
        out = r["output"]
        if tool == "calculator":
            if "error" in out:
                parts.append(f"计算 `{out.get('expression')}` 出错:{out['error']}")
            else:
                parts.append(f"计算结果:`{out['expression']}` = **{out['result']}**")
        elif tool == "web_search":
            lines = [f"针对“{out.get('query')}”的搜索结果(provider={out.get('provider')}):"]
            for i, res in enumerate(out.get("results", []), 1):
                lines.append(f"{i}. [{res['title']}]({res['url']}) - {res['snippet']}")
            parts.append("\n".join(lines))
        elif tool == "datetime":
            parts.append(
                f"当前时间:{out.get('now')};目标日期:{out.get('target_date')}({out.get('weekday')})。"
            )
        elif tool == "file_query":
            lines = [f"在知识库中检索“{out.get('query')}”的结果:"]
            for i, h in enumerate(out.get("hits", []), 1):
                lines.append(f"{i}. (score={h['score']}) {h['text']}")
            parts.append("\n".join(lines))
        else:
            parts.append(str(out))
    return "\n\n".join(parts) if parts else settings.rejection_message


def critique_node(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    # Tool answers are grounded in deterministic tool output -> trust them.
    if state.get("need_tool"):
        state["critique_passed"] = True
        _trace(state, "self_critique", "工具型回答基于确定性工具输出,校验通过。")
        return state

    contexts = state.get("reranked_docs", [])
    ctx_text = "\n".join(c["text"] for c in contexts)
    llm = get_llm()
    verdict = llm.complete(
        [
            {"role": "system", "content": CRITIQUE_SYSTEM},
            {"role": "user", "content": f"问题:{state['question']}\n上下文:{ctx_text}\n答案:{answer}"},
        ]
    )
    passed = verdict.strip().upper().startswith("PASS")
    state["critique_passed"] = passed
    _trace(
        state,
        "self_critique",
        ("自我校验通过:回答有依据、切题、无明显幻觉。" if passed
         else f"自我校验未通过,将重新生成。原因:{verdict[:80]}"),
    )
    return state
