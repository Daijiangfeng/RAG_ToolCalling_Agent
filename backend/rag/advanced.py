"""Advanced RAG techniques: Query Rewrite and HyDE.

These are used by the Agentic RAG workflow so the agent can autonomously decide
to rewrite an ambiguous / pronoun-heavy question or generate a hypothetical
document (HyDE) before retrieval.
"""

from __future__ import annotations

import re

from app.llm import get_llm
from app.logging import get_logger

logger = get_logger(__name__)

_PRONOUN_RE = re.compile(r"(它|他|她|这个|那个|其|该|it|this|that|they)")

REWRITE_SYSTEM = (
    "ROUTER/REWRITE: 你是查询改写器。根据对话历史,将用户最新的、可能含指代的问题"
    "改写为独立、明确、可直接检索的问题。只输出改写后的问题本身。"
)

HYDE_SYSTEM = (
    "HYDE: 你是一个知识助手。请为下面的问题生成一段简短的假设性答案(2-3句),"
    "用于检索,不需要准确,只需覆盖相关概念与关键词。"
)


def needs_rewrite(question: str, history: list[dict] | None) -> bool:
    """Heuristic: rewrite when the question is short and contains pronouns and
    there is prior conversation to resolve them against."""
    if not history:
        return False
    return bool(_PRONOUN_RE.search(question)) and len(question) < 40


def query_rewrite(question: str, history: list[dict] | None = None) -> str:
    if not needs_rewrite(question, history):
        return question
    hist_text = ""
    for turn in (history or [])[-3:]:
        hist_text += f"用户:{turn.get('question','')}\n助手:{turn.get('answer','')}\n"
    llm = get_llm()
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"对话历史:\n{hist_text}\n最新问题:{question}"},
    ]
    rewritten = llm.complete(messages).strip()
    result = rewritten or question
    logger.info("Query rewrite: %r -> %r", question, result)
    return result


def hyde(question: str) -> str:
    """Generate a hypothetical answer for HyDE-style retrieval."""
    llm = get_llm()
    messages = [
        {"role": "system", "content": HYDE_SYSTEM},
        {"role": "user", "content": question},
    ]
    hypo = llm.complete(messages).strip()
    logger.info("HyDE hypothetical answer generated (%d chars)", len(hypo))
    # Concatenate original question so retrieval keeps its keywords too.
    return f"{question}\n{hypo}"
