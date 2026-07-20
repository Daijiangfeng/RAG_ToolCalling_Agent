"""Vector store abstraction.

The default implementation is Chroma (persistent).  A dependency-free
:class:`InMemoryStore` is used as an automatic fallback so retrieval works even
when chromadb is not installed.  The :class:`VectorStore` interface is designed
so Milvus / Qdrant backends can be added later without touching callers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings
from app.logging import get_logger
from rag.embedding import Embedder, get_embedder

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryStore(VectorStore):
    """Simple cosine-similarity store; used when Chroma is unavailable."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vectors: list[list[float]] = []

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        vectors = self.embedder.embed(texts)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metadatas)
        self._vectors.extend(vectors)

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if not self._vectors:
            return []
        qv = self.embedder.embed_one(query_text)
        scored = []
        for i in range(len(self._ids)):
            sim = _cosine(qv, self._vectors[i])
            scored.append((sim, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, i in scored[:top_k]:
            results.append(
                {
                    "id": self._ids[i],
                    "text": self._texts[i],
                    "metadata": self._metas[i],
                    "score": float(sim),
                }
            )
        return results

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids.clear()
        self._texts.clear()
        self._metas.clear()
        self._vectors.clear()


class ChromaStore(VectorStore):
    def __init__(self, embedder: Embedder) -> None:  # pragma: no cover - optional dep
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, texts, metadatas):  # pragma: no cover - optional dep
        vectors = self.embedder.embed(texts)
        self._collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    def query(self, query_text, top_k):  # pragma: no cover - optional dep
        if self.count() == 0:
            return []
        qv = self.embedder.embed_one(query_text)
        res = self._collection.query(query_embeddings=[qv], n_results=min(top_k, self.count()))
        results = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i in range(len(ids)):
            results.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i] or {},
                    # Chroma returns cosine distance; convert to similarity.
                    "score": float(1.0 - dists[i]),
                }
            )
        return results

    def count(self):  # pragma: no cover - optional dep
        return self._collection.count()

    def reset(self):  # pragma: no cover - optional dep
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )


_store_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton
    embedder = get_embedder()
    if settings.vector_backend.lower() == "chroma":
        try:
            _store_singleton = ChromaStore(embedder)
            logger.info("Vector store: chroma (%s)", settings.chroma_dir)
            return _store_singleton
        except Exception as exc:  # pragma: no cover
            logger.warning("Chroma unavailable (%s) -> in-memory store fallback", exc)
    _store_singleton = InMemoryStore(embedder)
    logger.info("Vector store: in-memory (offline)")
    return _store_singleton
