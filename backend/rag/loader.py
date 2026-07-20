"""Document loaders for PDF and Markdown.

Every loader returns a normalised list of *segments*::

    {"text": str, "page_number": int, "heading": str}

PDF is parsed with PyMuPDF (``fitz``) page-by-page.  Markdown keeps track of the
current heading so downstream chunks can preserve their section title.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.logging import get_logger

logger = get_logger(__name__)

Segment = dict[str, object]


def load_pdf(path: str | Path) -> list[Segment]:
    """Load a PDF into per-page segments using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (pymupdf) is required to load PDF files") from exc

    path = Path(path)
    segments: list[Segment] = []
    with fitz.open(path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            if not text:
                continue
            heading = _guess_heading(text)
            segments.append(
                {"text": text, "page_number": page_index + 1, "heading": heading}
            )
    logger.info("Loaded PDF %s -> %d page segments", path.name, len(segments))
    return segments


def load_markdown(path: str | Path) -> list[Segment]:
    """Load a Markdown file, splitting on headings and preserving the title."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    return parse_markdown(raw)


def parse_markdown(raw: str) -> list[Segment]:
    segments: list[Segment] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                segments.append(
                    {"text": text, "page_number": 1, "heading": current_heading}
                )
            buffer.clear()

    for line in raw.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            current_heading = m.group(2).strip()
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    logger.info("Parsed Markdown -> %d heading segments", len(segments))
    return segments


def load_file(path: str | Path) -> list[Segment]:
    """Dispatch to the correct loader based on file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_markdown(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _guess_heading(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:80]
