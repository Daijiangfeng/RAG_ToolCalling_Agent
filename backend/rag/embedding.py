"""Embedding backends.

An :class:`Embedder` maps text to a fixed-length float vector.  Three concrete
implementations are provided:

* :class:`OpenAIEmbedding`  - OpenAI-compatible embeddings API.
* :class:`BGEEmbedding`     - local sentence-transformers BGE model (optional).
* :class:`HashEmbedding`    - deterministic hashing-trick vectors (offline).

:func:`get_embedder` picks the configured backend and *automatically degrades*
to :class:`HashEmbedding` whenever the preferred backend is unavailable, so the
platform always has a working embedder.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.config import settings
from app.errors import ProviderError, describe_provider_error
from app.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedding(Embedder):
    """Deterministic, dependency-free embedding using the hashing trick.

    Tokens are hashed into a fixed number of buckets; the resulting bag-of-words
    vector is L2-normalised.  This is not semantically powerful but is stable and
    good enough for offline demos / tests (cosine similarity reflects lexical
    overlap).
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedding(Embedder):
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self._model = settings.openai_embedding_model
        self.dim = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        # Zhipu / OpenAI embeddings APIs cap the number of inputs per request,
        # so send in batches of 64 and concatenate the results.
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), 64):
                batch = texts[start:start + 64]
                resp = self._client.embeddings.create(model=self._model, input=batch)
                vectors.extend(d.embedding for d in resp.data)
        except Exception as exc:
            # Mirror the LLM client: a configured embedding provider that fails at
            # runtime (auth/quota/model) must raise a clear, user-facing error
            # instead of leaking the raw provider body as an HTTP 500.
            logger.error("OpenAI embedding call failed: %s", exc)
            raise ProviderError(describe_provider_error("向量模型", exc), detail=str(exc)) from exc
        if vectors:
            self.dim = len(vectors[0])
        return vectors


class BGEEmbedding(Embedder):
    def __init__(self) -> None:  # pragma: no cover - heavy optional dep
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.embedding_model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


_embedder_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton

    backend = settings.embedding_backend.lower()
    try:
        if backend == "openai" and settings.has_llm:
            _embedder_singleton = OpenAIEmbedding()
            logger.info("Embedding backend: openai")
        elif backend == "bge":
            _embedder_singleton = BGEEmbedding()
            logger.info("Embedding backend: bge (%s)", settings.embedding_model)
        else:
            raise RuntimeError("fallback")
    except Exception as exc:  # pragma: no cover - depends on env
        if backend not in {"hash", ""}:
            logger.warning("Embedding backend '%s' unavailable (%s) -> hash fallback", backend, exc)
        _embedder_singleton = HashEmbedding()
        logger.info("Embedding backend: hash (offline, dim=%d)", _embedder_singleton.dim)
    return _embedder_singleton
