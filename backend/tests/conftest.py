"""Pytest configuration.

Forces the fully-offline backends (hash embedding, lexical reranker, in-memory
vector store, mock LLM) and an isolated temp database so the suite is fast and
deterministic without any API keys or heavy models.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# --- Environment must be set BEFORE importing app modules -------------------
_TMP = Path(tempfile.mkdtemp(prefix="rag_test_"))
os.environ.setdefault("AUTH_TOKEN", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RERANKER_BACKEND"] = "lexical"
os.environ["VECTOR_BACKEND"] = "memory"
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"

# Ensure the backend package root is importable.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _seed_kb():
    """Seed the in-memory knowledge base once for retrieval-dependent tests."""
    from app.db import init_db
    from rag.ingest import ingest_text

    init_db()
    ingest_text(
        "# RAG 概述\n\nRAG 检索增强生成结合检索与生成。RAG 的优势包括减少幻觉、可溯源、"
        "知识可更新、成本更低。完整链路:Loader、Embedding、Retriever、Reranker、LLM。",
        "rag_intro.md",
    )
    ingest_text(
        "# LangGraph\n\nLangGraph 的四个核心概念是 State、Node、Edge、Conditional Routing。"
        "Rerank 使用 Cross-Encoder 从候选中选出最相关的 Top 5。",
        "langgraph_intro.md",
    )
    yield
