"""Retriever: hybrid search (dense vector + sparse BM25) with RRF fusion.

Section 1.2 of the optimization plan requires fusing BM25 sparse retrieval with
vector dense retrieval using Reciprocal Rank Fusion (RRF) to improve recall for
queries where lexical overlap matters (e.g. technical terms, proper nouns).
"""

from __future__ import annotations

import math
from typing import Any

from app.config import settings
from app.logging import get_logger
from rag.text import tokenize
from rag.vectorstore import get_vector_store

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BM25 sparse scoring (operates over the indexed chunk texts held in memory)
# ---------------------------------------------------------------------------

class BM25Index:
    """Lightweight in-memory BM25 index over indexed chunks.

    Rebuilt each time the vector store changes. For the typical knowledge-base
    size (hundreds to low-thousands of chunks) this is fast enough at query time.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []  # [{id, text, metadata, tokens}]
        self._avg_dl: float = 0.0
        self._doc_count: int = 0
        self._df: dict[str, int] = {}  # document frequency per token
        self._k1 = 1.5
        self._b = 0.75

    def build(self, docs: list[dict[str, Any]]) -> None:
        """Build the index from a list of chunk dicts (must have 'text' key)."""
        self._docs = []
        self._df = {}
        total_tokens = 0

        for doc in docs:
            tokens = tokenize(doc.get("text", ""))
            entry = {**doc, "_tokens": tokens}
            self._docs.append(entry)
            total_tokens += len(tokens)
            seen: set[str] = set()
            for tok in tokens:
                if tok not in seen:
                    self._df[tok] = self._df.get(tok, 0) + 1
                    seen.add(tok)

        self._doc_count = len(self._docs)
        self._avg_dl = total_tokens / self._doc_count if self._doc_count else 1.0

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return top_k documents scored by BM25."""
        if not self._docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[float, int]] = []
        for idx, doc in enumerate(self._docs):
            score = self._score_doc(q_tokens, doc["_tokens"])
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            entry = {k: v for k, v in self._docs[idx].items() if k != "_tokens"}
            entry["bm25_score"] = score
            results.append(entry)
        return results

    def _score_doc(self, q_tokens: list[str], doc_tokens: list[str]) -> float:
        """Compute BM25 score for a single document."""
        dl = len(doc_tokens)
        score = 0.0
        tf_map: dict[str, int] = {}
        for tok in doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        for qt in set(q_tokens):
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            df = self._df.get(qt, 0)
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (self._k1 + 1)) / (
                tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
            )
            score += idf * tf_norm
        return score

    @property
    def is_empty(self) -> bool:
        return self._doc_count == 0


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *rankings: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists using RRF (k=60 default).

    Each document is identified by its 'id' key. The fused score is:
        sum(1 / (k + rank_i)) across all rankings where the doc appears.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc.get("id", doc.get("text", "")[:80])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = {k2: v for k2, v in doc.items()
                                   if k2 not in ("bm25_score",)}

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, rrf_score in fused:
        entry = doc_map.get(doc_id, {"id": doc_id})
        entry["score"] = rrf_score
        entry["rrf_score"] = rrf_score
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Retriever (hybrid: vector + BM25 + RRF)
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.top_k
        self.store = get_vector_store()
        self._bm25 = BM25Index()
        self._bm25_built_count = -1  # track when to rebuild

    def _ensure_bm25(self) -> None:
        """Rebuild BM25 index if the store has new documents."""
        current_count = self.store.count()
        if current_count != self._bm25_built_count and current_count > 0:
            # Fetch all documents from vector store for BM25 indexing.
            # For stores with a list_all method, use it; otherwise query broadly.
            try:
                all_docs = self.store.query(" ", top_k=max(current_count, 100))
            except Exception:
                all_docs = []
            if all_docs:
                self._bm25.build(all_docs)
                self._bm25_built_count = current_count
                logger.info("BM25 index rebuilt with %d documents", len(all_docs))

    def similarity_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Hybrid search: vector similarity + BM25, fused via RRF."""
        k = top_k or self.top_k

        # Dense retrieval (vector similarity)
        dense_results = self.store.query(query, k)

        # Sparse retrieval (BM25)
        self._ensure_bm25()
        if not self._bm25.is_empty:
            sparse_results = self._bm25.search(query, k)
            # Fuse using Reciprocal Rank Fusion
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            results = fused[:k]
            logger.info(
                "Hybrid search: %d dense + %d sparse -> %d fused (top_k=%d)",
                len(dense_results), len(sparse_results), len(results), k,
            )
        else:
            # Fallback to dense-only if BM25 index is empty
            results = dense_results
            logger.info("Retrieved %d candidates (dense-only, top_k=%d)", len(results), k)

        return results

    def index_chunks(self, chunks: list) -> int:
        """Add splitter ``Chunk`` objects to the store."""
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        self.store.add(ids, texts, metas)
        # Invalidate BM25 index so it rebuilds on next search.
        self._bm25_built_count = -1
        return len(chunks)

    def delete_file(self, file_name: str) -> int:
        """Remove all chunks belonging to ``file_name`` from the store.

        Used to make ingestion idempotent per file name: re-uploading a
        same-named document replaces its old vectors instead of accumulating
        duplicates. Returns the number of vectors removed.
        """
        removed = self.store.delete_by_file(file_name)
        if removed:
            # 向量发生变化，失效 BM25 索引以在下次检索时重建。
            self._bm25_built_count = -1
            logger.info("Deleted %d chunks for file '%s'", removed, file_name)
        return removed


_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
