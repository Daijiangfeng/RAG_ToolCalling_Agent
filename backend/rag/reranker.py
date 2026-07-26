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
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings
from app.errors import ProviderError, describe_provider_error
from app.logging import get_logger
from rag.text import tokenize

logger = get_logger(__name__)


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
    """Offline reranker: IDF-weighted *query coverage* (absolute 0..1).

    The score is the share of the query's IDF mass that the document actually
    covers.  It is deliberately **not** normalised to the candidate-set maximum:
    a max-normalised score would force the best candidate to 1.0 even for an
    off-topic question, defeating the downstream confidence threshold used to
    trigger rejection.  An absolute coverage score is high for relevant docs and
    low for unrelated ones, so it doubles as a meaningful confidence signal.
    """

    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return [float(d.get("score", 0.0)) for d in docs]
        doc_tokens = [tokenize(d["text"]) for d in docs]

        # IDF over the candidate set.
        df: dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        idf = {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}

        q_set = set(q_tokens)
        q_idf_total = sum(idf.get(t, 1.0) for t in q_set)

        scores = []
        for toks in doc_tokens:
            if not toks or q_idf_total <= 0:
                scores.append(0.0)
                continue
            present = q_set & set(toks)
            covered = sum(idf.get(t, 1.0) for t in present)
            scores.append(covered / q_idf_total)
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


class ZhipuReranker(Reranker):
    """智谱 rerank 模型（OpenAI 兼容端点，标准 HTTP Bearer 认证）。

    通过 ``POST {openai_base_url}/rerank`` 调用智谱重排序服务，请求头为
    ``Authorization: Bearer <auth_token>``（与 LLM / 向量 embedding 使用同一个令牌）。
    模型编码默认为 ``rerank``；若 ``RERANKER_MODEL`` 配置的是 HF 路径则回退到该编码。
    """

    def __init__(self) -> None:
        import httpx

        self._url = settings.openai_base_url.rstrip("/") + "/rerank"
        model = (settings.reranker_model or "").strip()
        # 智谱 rerank 模型编码为 "rerank"；若配置的是 HF 路径（含 "/"）或为空，回退。
        self._model = "rerank" if (not model or "/" in model) else model
        self._token = settings.auth_token
        self._client = httpx.Client(timeout=30.0)

    def score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        if not docs:
            return []
        texts = [d["text"] for d in docs]
        try:  # pragma: no cover - network
            resp = self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._token}"},
                json={
                    "model": self._model,
                    "query": query,
                    "documents": texts,
                    "top_n": len(texts),
                    "return_documents": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - network
            # 与 embedding / LLM 客户端一致：已配置凭证的提供方运行时失败
            # （鉴权/额度/模型/网络）必须抛出清晰的面向用户错误，而非静默降级。
            logger.error("Zhipu rerank call failed: %s", exc)
            raise ProviderError(
                describe_provider_error("重排序模型", exc), detail=str(exc)
            ) from exc
        # 按响应中的原始 ``index`` 回填得分到与 ``docs`` 等长的列表（未命中填 0.0），
        # 保证基类 ``rerank()`` 中 ``zip(docs, scores)`` 的顺序对齐。
        scores = [0.0] * len(docs)
        for item in data.get("results", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores


_reranker_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton
    backend = settings.reranker_backend.lower()
    if backend in {"cross-encoder", "cross_encoder", "bge"}:
        try:
            _reranker_singleton = CrossEncoderReranker()
            logger.info("Reranker: cross-encoder (%s)", settings.reranker_model)
            return _reranker_singleton
        except Exception as exc:  # pragma: no cover
            logger.warning("CrossEncoder unavailable (%s) -> lexical reranker", exc)
    elif backend == "zhipu":
        # 有令牌才在线调用智谱 rerank；缺凭证时回退到离线 lexical（合理的兜底降级）。
        if settings.has_llm:
            _reranker_singleton = ZhipuReranker()
            logger.info("Reranker: zhipu (%s)", _reranker_singleton._model)
            return _reranker_singleton
        logger.info("Reranker backend 'zhipu' 但未配置令牌 -> lexical 回退")
    _reranker_singleton = LexicalReranker()
    logger.info("Reranker: lexical (offline)")
    return _reranker_singleton
