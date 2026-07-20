"""Retriever: similarity search over the vector store."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.logging import get_logger
from rag.vectorstore import get_vector_store

logger = get_logger(__name__)


class Retriever:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.top_k
        self.store = get_vector_store()

    def similarity_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.top_k
        results = self.store.query(query, k)
        logger.info("Retrieved %d candidates for query (top_k=%d)", len(results), k)
        return results

    def index_chunks(self, chunks: list) -> int:
        """Add splitter ``Chunk`` objects to the store."""
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        self.store.add(ids, texts, metas)
        return len(chunks)


_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
