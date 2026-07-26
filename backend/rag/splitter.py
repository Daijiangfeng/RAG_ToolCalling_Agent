"""Recursive character text splitter with parent-child chunk support.

Splits loader *segments* into overlapping chunks while preserving the section
heading and page number.  ``chunk_size`` and ``chunk_overlap`` are configurable
(defaults 800 / 150).

Section 1.2 enhancement: multi-granularity parent-child indexing. Child chunks
are fine-grained (sentence-level) for precise retrieval, while parent chunks
(paragraph-level) provide fuller context for the generator.
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


@dataclass
class ParentChildChunk:
    """A fine-grained child chunk linked to its broader parent context.

    When used with the retriever, the *child* text is embedded for precise
    matching, but the *parent* text is returned to the generator for richer
    context. This improves both retrieval precision and generation quality.
    """
    child: Chunk
    parent_text: str
    parent_id: str


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

    def split_segments_with_parents(
        self, segments: list[dict], file_name: str = "",
        child_chunk_size: int = 200,
    ) -> list[ParentChildChunk]:
        """Split into parent (paragraph) + child (sentence) chunks.

        Parent chunks are the normal-size chunks produced by split_segments().
        Child chunks are finer-grained sub-splits of each parent, used for
        embedding-level retrieval while the parent provides generation context.

        Args:
            segments: loader output segments.
            file_name: source file name for metadata.
            child_chunk_size: max characters per child chunk (default 200).

        Returns:
            List of ParentChildChunk linking each child to its parent.
        """
        parent_chunks = self.split_segments(segments, file_name)
        child_splitter = RecursiveCharacterSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=min(30, child_chunk_size // 4),
        )

        results: list[ParentChildChunk] = []
        for parent in parent_chunks:
            child_texts = child_splitter.split_text(parent.text)
            for ct in child_texts:
                if not ct.strip():
                    continue
                child_id = uuid.uuid4().hex[:12]
                child = Chunk(
                    text=ct,
                    chunk_id=child_id,
                    heading=parent.heading,
                    page_number=parent.page_number,
                    metadata={
                        **parent.metadata,
                        "chunk_id": child_id,
                        "parent_id": parent.chunk_id,
                        "is_child": True,
                    },
                )
                results.append(ParentChildChunk(
                    child=child,
                    parent_text=parent.text,
                    parent_id=parent.chunk_id,
                ))

        logger.info(
            "Parent-child split: %d parents -> %d children",
            len(parent_chunks), len(results),
        )
        return results

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
