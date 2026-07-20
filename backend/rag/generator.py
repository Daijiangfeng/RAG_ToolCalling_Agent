"""Answer generation with prompt optimisation and context compression."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.config import settings
from app.llm import get_llm
from app.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是企业知识助手。

规则:
1. 只能根据提供的上下文回答;
2. 不允许编造事实;
3. 必须在回答中引用来源(使用 [来源 N] 标注);
4. 如果上下文信息不足,必须明确拒绝回答,而不是猜测。
"""


def context_compression(
    docs: list[dict[str, Any]], max_chars: int = 2400, min_score: float = 0.0
) -> list[dict[str, Any]]:
    """Compress the reranked context.

    * drops documents scoring below ``min_score``;
    * removes near-duplicate chunks;
    * truncates the total context to ``max_chars``.
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for d in docs:
        if float(d.get("rerank_score", d.get("score", 0.0))) < min_score:
            continue
        fingerprint = d["text"][:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        text = d["text"]
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
        if not text:
            break
        item = dict(d)
        item["text"] = text
        kept.append(item)
        total += len(text)
    return kept


def _build_messages(question: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        src = meta.get("file_name", "unknown")
        page = meta.get("page_number", "?")
        blocks.append(f"[来源 {i}] (文件:{src} 第{page}页)\n{c['text']}")
    context_text = "\n\n".join(blocks)
    user = f"上下文:\n{context_text}\n\n问题:{question}\n\n请依据上下文回答并引用来源。"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    def __init__(self) -> None:
        self.llm = get_llm()

    def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return settings.rejection_message
        messages = _build_messages(question, contexts)
        return self.llm.complete(messages)

    def generate_stream(self, question: str, contexts: list[dict[str, Any]]) -> Iterator[str]:
        if not contexts:
            yield settings.rejection_message
            return
        messages = _build_messages(question, contexts)
        yield from self.llm.stream(messages)


_generator_singleton: Generator | None = None


def get_generator() -> Generator:
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = Generator()
    return _generator_singleton
