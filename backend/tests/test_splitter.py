"""Tests for the recursive character splitter."""

from rag.splitter import RecursiveCharacterSplitter


def test_chunk_size_respected():
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    text = "。".join(f"句子{i}" * 5 for i in range(30))
    chunks = splitter.split_text(text)
    assert chunks
    # allow a small slack because overlap is prepended
    assert all(len(c) <= 50 + 10 for c in chunks)


def test_overlap_must_be_smaller():
    import pytest

    with pytest.raises(ValueError):
        RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=100)


def test_split_segments_preserves_metadata():
    splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5)
    segments = [{"text": "内容" * 100, "page_number": 3, "heading": "第一章"}]
    chunks = splitter.split_segments(segments, file_name="doc.pdf")
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_number == 3
        assert c.heading == "第一章"
        assert c.metadata["file_name"] == "doc.pdf"
        assert c.metadata["chunk_id"] == c.chunk_id
