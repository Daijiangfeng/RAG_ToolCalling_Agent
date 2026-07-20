"""Cross-Encoder reranker (with an offline lexical fallback).

Reranking is a core acceptance criterion: after retrieving the top-K candidates
we re-score every (query, document) pair and keep the best ``rerank_top_n``.

* :class:`CrossEncoderReranker` uses a sentence-transformers CrossEncoder such as
  ``BAAI/bge-reranker-large`` (heavy, optional).
* :class:`LexicalReranker` is a deterministic offline fallback based on token
  overlap / IDF-lite scoring.

``rerank_compare`` returns the ordering *before* and *after* reranking so the UI
and README can show the before/after effect.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]: ...

    def rerank(
        self, query: str, docs: list[dict[str, Any]], top_n: int | None = None
    ) -> list[dict[str, Any]]:
        if not docs:
            return []
        top_n = top_n or settings.rerank_top_n
        scores = self.score(query, docs)
        ranked = []
        for doc, s in zip(docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(s)
            item["score"] = float(s)
            ranked.append(item)
        ranked.sort(key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_n]

    def rerank_compare(
        self, query: str, docs: list[dict[str, Any]], top_n: int | None = None
    ) -> dict[str, Any]:
        """Return before/after orderings to demonstrate rerank effect."""
        top_n = top_n or settings.rerank_top_n
        before = [
            {"text": d["text"][:120], "score": round(float(d.get("score", 0.0)), 4)}
            for d in docs[:top_n]
        ]
        after_full = self.rerank(query, docs, top_n)
        after = [
            {"text": d["text"][:120], "score": round(float(d["rerank_score"]), 4)}
            for d in after_full
        ]
        return {"before": before, "after": after}


class LexicalReranker(Reranker):
    """Offline reranker: cosine of IDF-weighted token overlap vectors."""

    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return [float(d.get("score", 0.0)) for d in docs]
        doc_tokens = [_tokens(d["text"]) for d in docs]

        # IDF over the candidate set.
        df: dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        idf = {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}

        scores = []
        q_set = set(q_tokens)
        for toks in doc_tokens:
            if not toks:
                scores.append(0.0)
                continue
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            overlap = sum(idf.get(t, 1.0) * counts.get(t, 0) for t in q_set)
            length_norm = math.sqrt(len(toks))
            scores.append(overlap / length_norm if length_norm else 0.0)

        # Normalise to 0..1 so it doubles as a confidence signal.
        mx = max(scores) if scores else 0.0
        if mx > 0:
            scores = [s / mx for s in scores]
        return scores


class CrossEncoderReranker(Reranker):
    def __init__(self) -> None:  # pragma: no cover - heavy optional dep
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.reranker_model)

    def score(self, query, docs):  # pragma: no cover - heavy optional dep
        pairs = [(query, d["text"]) for d in docs]
        raw = self._model.predict(pairs)
        # sigmoid to 0..1
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


_reranker_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton
    if settings.reranker_backend.lower() in {"cross-encoder", "cross_encoder", "bge"}:
        try:
            _reranker_singleton = CrossEncoderReranker()
            logger.info("Reranker: cross-encoder (%s)", settings.reranker_model)
            return _reranker_singleton
        except Exception as exc:  # pragma: no cover
            logger.warning("CrossEncoder unavailable (%s) -> lexical reranker", exc)
    _reranker_singleton = LexicalReranker()
    logger.info("Reranker: lexical (offline)")
    return _reranker_singleton
