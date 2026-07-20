"""Recursive character text splitter.

Splits loader *segments* into overlapping chunks while preserving the section
heading and page number.  ``chunk_size`` and ``chunk_overlap`` are configurable
(defaults 800 / 150).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "! ", "? ", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    chunk_id: str
    heading: str = ""
    page_number: int = 1
    metadata: dict = field(default_factory=dict)


class RecursiveCharacterSplitter:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        return self._merge(self._recursive_split(text, self.separators))

    def split_segments(self, segments: list[dict], file_name: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        for seg in segments:
            heading = str(seg.get("heading", ""))
            page = int(seg.get("page_number", 1))
            for piece in self.split_text(str(seg.get("text", ""))):
                cid = uuid.uuid4().hex[:12]
                chunks.append(
                    Chunk(
                        text=piece,
                        chunk_id=cid,
                        heading=heading,
                        page_number=page,
                        metadata={
                            "file_name": file_name,
                            "page_number": page,
                            "heading": heading,
                            "chunk_id": cid,
                        },
                    )
                )
        logger.info("Split %d segments -> %d chunks", len(segments), len(chunks))
        return chunks

    # ------------------------------------------------------------------
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[-1]
        remaining = separators
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining = separators[i + 1 :]
                break
            if sep in text:
                separator = sep
                remaining = separators[i + 1 :]
                break

        if separator == "":
            # Hard split by size.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        splits = text.split(separator)
        pieces: list[str] = []
        for part in splits:
            if not part:
                continue
            candidate = part + separator
            if len(candidate) > self.chunk_size and remaining:
                pieces.extend(self._recursive_split(candidate, remaining))
            else:
                pieces.append(candidate)
        return pieces

    def _merge(self, splits: list[str]) -> list[str]:
        """Greedily merge small splits up to ``chunk_size`` with overlap."""
        chunks: list[str] = []
        current = ""
        for piece in splits:
            if not piece.strip():
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                # start new chunk with tail overlap of the previous one
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = overlap + piece
        if current.strip():
            chunks.append(current.strip())
        return chunks
