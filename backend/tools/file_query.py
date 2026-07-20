"""File query tool: search the ingested knowledge base by keyword / filename."""

from __future__ import annotations

from app.logging import get_logger
from rag.retriever import get_retriever

logger = get_logger(__name__)

SCHEMA = {
    "name": "file_query",
    "description": "在已上传的知识库文件中检索与关键词相关的片段。当用户想查询某个上传文档的具体内容时使用。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词或问题"},
            "file_name": {"type": "string", "description": "可选,限定文件名"},
            "top_k": {"type": "integer", "description": "返回片段数,默认 5"},
        },
        "required": ["query"],
    },
}


def query_files(query: str, file_name: str | None = None, top_k: int = 5) -> dict:
    retriever = get_retriever()
    candidates = retriever.similarity_search(query, top_k=max(top_k * 3, 10))
    if file_name:
        candidates = [
            c for c in candidates if file_name.lower() in str(c.get("metadata", {}).get("file_name", "")).lower()
        ]
    hits = candidates[:top_k]
    return {
        "query": query,
        "file_name": file_name,
        "hits": [
            {
                "text": h["text"][:300],
                "score": round(float(h.get("score", 0.0)), 4),
                "metadata": h.get("metadata", {}),
            }
            for h in hits
        ],
    }


def run(query: str, file_name: str | None = None, top_k: int = 5, **_: object) -> dict:
    return query_files(query, file_name, top_k)
