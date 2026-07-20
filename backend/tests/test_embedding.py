"""Tests for embedding backends."""

from rag.embedding import HashEmbedding, get_embedder


def test_hash_embedding_dim_and_norm():
    emb = HashEmbedding(dim=64)
    vecs = emb.embed(["hello world", "another text"])
    assert len(vecs) == 2
    assert all(len(v) == 64 for v in vecs)
    # L2 norm ~ 1 for non-empty text
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedding_deterministic():
    emb = HashEmbedding(dim=32)
    assert emb.embed_one("RAG") == emb.embed_one("RAG")


def test_get_embedder_offline_fallback():
    emb = get_embedder()
    v = emb.embed_one("test")
    assert isinstance(v, list) and len(v) == emb.dim
