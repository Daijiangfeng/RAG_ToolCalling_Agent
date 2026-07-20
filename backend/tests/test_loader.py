"""Tests for document loaders."""

from rag.loader import parse_markdown


def test_parse_markdown_keeps_headings():
    raw = "# Title\n\nsome intro\n\n## Section A\n\ncontent A\n\n## Section B\n\ncontent B"
    segments = parse_markdown(raw)
    assert len(segments) >= 2
    headings = {s["heading"] for s in segments}
    assert "Section A" in headings
    assert "Section B" in headings
    assert all("text" in s and "page_number" in s for s in segments)


def test_parse_markdown_empty():
    assert parse_markdown("") == []
