"""Lightweight in-memory conversation memory.

Stores the last few QA turns per session so Query Rewrite can resolve pronouns.
This is intentionally simple (process-local); swap for Redis in production.
"""

from __future__ import annotations

from collections import defaultdict, deque

_MEMORY: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))


def get_history(session_id: str) -> list[dict]:
    return list(_MEMORY[session_id])


def add_turn(session_id: str, question: str, answer: str) -> None:
    _MEMORY[session_id].append({"question": question, "answer": answer})


def clear(session_id: str | None = None) -> None:
    if session_id is None:
        _MEMORY.clear()
    else:
        _MEMORY.pop(session_id, None)
