"""Conversation memory with abstract interface for backend swapping.

The default implementation is process-local (deque). A Redis-backed
implementation can be plugged in for distributed deployments by implementing
the :class:`ConversationMemory` protocol and calling :func:`set_backend`.

This abstraction enables Section 1.2 of the optimization plan: upgrading
session memory from in-process deque to Redis without changing call sites.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from app.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ConversationMemory(ABC):
    """Protocol for conversation memory backends."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]:
        ...

    @abstractmethod
    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        ...

    @abstractmethod
    def clear(self, session_id: str | None = None) -> None:
        ...


# ---------------------------------------------------------------------------
# Default in-memory implementation
# ---------------------------------------------------------------------------

class InMemoryConversationMemory(ConversationMemory):
    """Process-local conversation memory using a deque per session."""

    def __init__(self, max_turns: int = 5) -> None:
        self._store: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns))

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._store[session_id])

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        self._store[session_id].append({"question": question, "answer": answer})

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._store.clear()
        else:
            self._store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Redis-backed implementation (distributed, persistent)
# ---------------------------------------------------------------------------

class RedisConversationMemory(ConversationMemory):
    """Redis-backed conversation memory for distributed deployments.

    Each session's history is stored as a Redis List under key
    ``memory:{session_id}``. Turns are JSON-serialized dicts pushed with RPUSH
    and trimmed to ``max_turns`` entries.
    """

    def __init__(self, redis_url: str, max_turns: int = 5, key_prefix: str = "memory") -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisConversationMemory. "
                "Install it with: pip install 'redis[hiredis]>=5.0'"
            ) from exc

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._max_turns = max_turns
        self._prefix = key_prefix
        logger.info("RedisConversationMemory initialized (url=%s, max_turns=%d)", redis_url, max_turns)

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        raw_items = self._client.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw_items]

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        key = self._key(session_id)
        turn = json.dumps({"question": question, "answer": answer}, ensure_ascii=False)
        self._client.rpush(key, turn)
        # Keep only the last max_turns entries
        self._client.ltrim(key, -self._max_turns, -1)

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            # Clear all memory keys (use scan to avoid blocking)
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=f"{self._prefix}:*", count=100)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        else:
            self._client.delete(self._key(session_id))

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Global singleton with backend swapping support
# ---------------------------------------------------------------------------

_backend: ConversationMemory = InMemoryConversationMemory()


def set_backend(backend: ConversationMemory) -> None:
    """Replace the global memory backend (e.g. with Redis implementation)."""
    global _backend
    _backend = backend
    logger.info("Memory backend switched to %s", type(backend).__name__)


def get_backend() -> ConversationMemory:
    """Return the active memory backend instance."""
    return _backend


def init_memory_backend() -> None:
    """Auto-detect and initialize the best available memory backend.

    If REDIS_URL is configured and Redis is reachable, switches to
    RedisConversationMemory. Otherwise keeps the default InMemory backend.
    """
    from app.config import settings

    redis_url = getattr(settings, "redis_url", "")
    if not redis_url:
        logger.info("REDIS_URL not configured, using InMemoryConversationMemory")
        return

    try:
        backend = RedisConversationMemory(redis_url)
        if backend.ping():
            set_backend(backend)
        else:
            logger.warning("Redis not reachable at %s, falling back to InMemory", redis_url)
    except Exception as exc:
        logger.warning("Failed to init Redis memory (%s), falling back to InMemory", exc)


# ---------------------------------------------------------------------------
# Module-level convenience functions (maintain backwards compatibility)
# ---------------------------------------------------------------------------

def get_history(session_id: str) -> list[dict]:
    return _backend.get_history(session_id)


def add_turn(session_id: str, question: str, answer: str) -> None:
    _backend.add_turn(session_id, question, answer)


def clear(session_id: str | None = None) -> None:
    _backend.clear(session_id)
