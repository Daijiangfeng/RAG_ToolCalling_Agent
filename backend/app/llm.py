"""LLM client wrapper around any OpenAI-compatible API.

When no ``OPENAI_API_KEY`` is configured (or the SDK/network is unavailable) the
client transparently falls back to a deterministic *mock* implementation so that
the whole platform, its demos and its test-suite remain fully runnable offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)

Message = dict[str, str]


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        self._mode = "mock"
        if settings.has_llm:
            try:  # pragma: no cover - depends on optional network/SDK
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )
                self._mode = "openai"
                logger.info("LLMClient initialised in OpenAI mode (%s)", settings.model_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Falling back to mock LLM: %s", exc)
                self._client = None
                self._mode = "mock"
        else:
            logger.info("No OPENAI_API_KEY set -> LLMClient running in offline mock mode")

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(self, messages: list[Message], temperature: float = 0.2) -> str:
        if self._client is not None:
            try:  # pragma: no cover - network
                resp = self._client.chat.completions.create(
                    model=settings.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM call failed, using mock: %s", exc)
        return self._mock_complete(messages)

    def stream(self, messages: list[Message], temperature: float = 0.2) -> Iterator[str]:
        if self._client is not None:
            try:  # pragma: no cover - network
                stream = self._client.chat.completions.create(
                    model=settings.model_name,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM stream failed, using mock: %s", exc)
        # Mock streaming: yield token-by-token.
        text = self._mock_complete(messages)
        for token in re.findall(r"\S+\s*", text):
            yield token

    # ------------------------------------------------------------------
    # Deterministic offline mock
    # ------------------------------------------------------------------
    def _mock_complete(self, messages: list[Message]) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")

        # Self-critique node asks for a strict PASS/FAIL verdict.
        if "CRITIQUE" in system or "自我校验" in system:
            return "PASS: 回答基于所提供的上下文,直接回应了问题,未发现明显幻觉。"

        # Query rewrite node.
        if "REWRITE" in system or "改写" in system:
            # Return the last user line as-is if nothing to rewrite.
            return user.strip().splitlines()[-1] if user.strip() else user

        # HyDE hypothetical answer.
        if "HYDE" in system or "假设" in system:
            return f"关于“{_first_line(user)}”的一个可能答案是:该主题的核心概念、优势与常见用法。"

        # Router intent classification.
        if "ROUTER" in system or "意图" in system:
            return "rag"

        # Default generator behaviour: extractive answer from the provided context.
        context = _extract_context(user)
        if not context.strip():
            return settings.rejection_message
        question = _extract_question(user)
        snippet = context.strip().split("\n\n")[0][:400]
        return (
            f"根据知识库内容,针对“{question}”:\n\n{snippet}\n\n"
            f"(以上回答依据检索到的上下文,详见 Sources 引用。)"
        )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


def _extract_question(user: str) -> str:
    m = re.search(r"问题[:：]\s*(.+)", user)
    if m:
        return m.group(1).strip()
    return _first_line(user)


def _extract_context(user: str) -> str:
    m = re.search(r"上下文[:：]\s*(.+)", user, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


_llm_singleton: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton
