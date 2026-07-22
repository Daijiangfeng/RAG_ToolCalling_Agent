"""Shared lexical tokenizer.

A single source of truth for token splitting used by both the reranker and the
evaluator.  Keeping one definition prevents the two from drifting apart: a
mismatch here silently distorts lexical-overlap metrics (see the CJK mixed-run
note below).

ASCII words are kept whole (e.g. ``rag``), while each CJK ideograph is its own
token.  Without the per-character CJK split, a mixed run like ``什么是RAG``
would collapse into a single token that never overlaps the document's ``rag``
token, making every lexical score 0 for Chinese queries.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Lower-case and split ``text`` into ASCII words + single CJK characters."""
    return _TOKEN_RE.findall((text or "").lower())
