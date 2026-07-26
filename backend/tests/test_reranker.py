"""Tests for the reranker (before/after ordering + Zhipu backend)."""

import rag.reranker as reranker_mod
from rag.reranker import LexicalReranker, ZhipuReranker, get_reranker


def _docs():
    return [
        {"text": "今天天气很好,适合出门散步。", "score": 0.9, "metadata": {}},
        {"text": "RAG 使用 Cross-Encoder 重排序从候选中选出最相关的片段。", "score": 0.5, "metadata": {}},
        {"text": "猫是一种常见的宠物动物。", "score": 0.4, "metadata": {}},
    ]


def test_rerank_promotes_relevant_doc():
    reranker = LexicalReranker()
    ranked = reranker.rerank("Cross-Encoder 重排序是什么", _docs(), top_n=3)
    assert "重排序" in ranked[0]["text"]
    assert ranked[0]["rerank_score"] >= ranked[-1]["rerank_score"]


def test_rerank_compare_shape():
    reranker = LexicalReranker()
    cmp = reranker.rerank_compare("重排序", _docs(), top_n=2)
    assert "before" in cmp and "after" in cmp
    assert len(cmp["after"]) == 2


# --- Zhipu rerank backend ---------------------------------------------------

class _FakeRerankResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHttpClient:
    """Records requests and returns a canned Zhipu rerank response."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        # 故意乱序且仅返回部分索引，校验 score() 能按 index 回填。
        return _FakeRerankResp(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }
        )


def test_zhipu_backend_without_token_falls_back_to_lexical(monkeypatch):
    # 无令牌时 backend=zhipu 应自动回退到离线 lexical。
    monkeypatch.setattr(reranker_mod.settings, "reranker_backend", "zhipu")
    monkeypatch.setattr(reranker_mod.settings, "auth_token", "")
    monkeypatch.setattr(reranker_mod, "_reranker_singleton", None)
    assert isinstance(get_reranker(), LexicalReranker)


def test_zhipu_reranker_reconstructs_scores_by_index():
    r = ZhipuReranker()
    fake = _FakeHttpClient()
    r._client = fake  # 注入假客户端，避免真实网络调用。
    docs = [
        {"text": "doc zero", "score": 0.0},
        {"text": "doc one", "score": 0.0},
        {"text": "doc two", "score": 0.0},
    ]

    scores = r.score("q", docs)
    # index 2 -> 0.9, index 0 -> 0.1, index 1 未命中 -> 0.0。
    assert scores == [0.1, 0.0, 0.9]

    # 请求使用标准 HTTP Bearer 认证与模型编码 rerank。
    call = fake.calls[0]
    assert call["headers"]["Authorization"].startswith("Bearer ")
    assert call["json"]["model"] == "rerank"
    assert call["json"]["documents"] == ["doc zero", "doc one", "doc two"]

    ranked = r.rerank("q", docs, top_n=2)
    assert [d["text"] for d in ranked] == ["doc two", "doc zero"]
    assert ranked[0]["rerank_score"] == 0.9
