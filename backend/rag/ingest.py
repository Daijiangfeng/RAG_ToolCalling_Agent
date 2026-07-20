"""Document ingestion pipeline shared by the upload API and seeding script."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.logging import get_logger
from rag.loader import load_file, parse_markdown
from rag.retriever import get_retriever
from rag.splitter import RecursiveCharacterSplitter

logger = get_logger(__name__)


def ingest_file(path: str | Path) -> dict:
    """Load -> split -> embed -> index a file on disk."""
    path = Path(path)
    segments = load_file(path)
    return _index_segments(segments, path.name)


def ingest_text(raw: str, file_name: str) -> dict:
    """Ingest raw markdown/plain text (used for seeding & tests)."""
    segments = parse_markdown(raw)
    return _index_segments(segments, file_name)


def _index_segments(segments: list[dict], file_name: str) -> dict:
    splitter = RecursiveCharacterSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_segments(segments, file_name=file_name)
    retriever = get_retriever()
    n = retriever.index_chunks(chunks)
    pages = len({s.get("page_number", 1) for s in segments}) or len(segments)
    logger.info("Ingested %s: %d pages, %d chunks", file_name, pages, n)
    return {"filename": file_name, "pages": pages, "chunks": n, "status": "processed"}
