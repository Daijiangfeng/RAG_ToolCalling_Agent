"""LLM client wrapper around the Anthropic Messages API.

智谱 GLM 提供 Anthropic 兼容端点，因此本客户端直接用官方 ``anthropic`` SDK
调用。当未配置 ``ANTHROPIC_AUTH_TOKEN``（或 SDK/网络不可用）时，客户端透明地
回退到确定性的 *mock* 实现，以保证整个平台、演示与测试套件完全可离线运行。
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.config import settings
from app.errors import ProviderError, describe_provider_error
from app.logging import get_logger

logger = get_logger(__name__)

Message = dict[str, str]


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Convert OpenAI-style messages to the Anthropic layout.

    Anthropic 将 system 提示作为顶层参数（而非 messages 中的一条角色），
    因此这里把所有 ``system`` 消息合并为一段，其余消息原样保留。
    """
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    convo = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in messages
        if m.get("role") != "system"
    ]
    system = "\n".join(p for p in system_parts if p)
    if not convo:
        # Anthropic 要求至少一条消息；仅有 system 时降级为 user。
        convo = [{"role": "user", "content": system or ""}]
    return system, convo


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        self._mode = "mock"
        if settings.has_llm:
            try:  # pragma: no cover - depends on optional network/SDK
                import anthropic

                # 智谱 GLM Anthropic 兼容端点：使用 auth_token 令牌，SDK 会以
                # ``Authorization: Bearer <token>`` 发送（而非默认的 x-api-key）。
                self._client = anthropic.Anthropic(
                    auth_token=settings.auth_token,
                    base_url=settings.anthropic_base_url,
                )
                self._mode = "anthropic"
                logger.info("LLMClient initialised in Anthropic mode (%s)", settings.model_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Falling back to mock LLM: %s", exc)
                self._client = None
                self._mode = "mock"
        else:
            logger.info("No ANTHROPIC_AUTH_TOKEN set -> LLMClient running in offline mock mode")

    @property
    def mode(self) -> str:
        return self._mode

    def verify(self) -> tuple[bool, str]:
        """Best-effort credential/model check used at startup.

        Returns ``(ok, message)``.  In offline mock mode this is always OK.
        When a provider is configured, a minimal 1-token completion validates
        that the API key and model name are actually accepted, turning a silent
        runtime failure into an explicit, actionable startup log.
        """
        if self._client is None:
            return True, "离线 mock 模式（未配置 ANTHROPIC_AUTH_TOKEN）"
        try:  # pragma: no cover - network
            self._client.messages.create(
                model=settings.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
            )
            return True, f"智谱处理器可用（model={settings.model_name}）"
        except Exception as exc:  # pragma: no cover - network
            return False, describe_provider_error("LLM", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(self, messages: list[Message], temperature: float = 0.2) -> str:
        if self._client is not None:
            try:  # pragma: no cover - network
                system, convo = _split_system(messages)
                kwargs: dict = dict(
                    model=settings.model_name,
                    max_tokens=settings.max_tokens,
                    messages=convo,
                    temperature=temperature,
                )
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
            except Exception as exc:  # pragma: no cover
                # A configured provider that fails (auth/quota/model/network) is a
                # real misconfiguration -- surface it clearly instead of silently
                # returning mock answers that look like a working LLM.
                logger.error("LLM call failed: %s", exc)
                raise ProviderError(describe_provider_error("LLM", exc), detail=str(exc)) from exc
        return self._mock_complete(messages)

    def stream(self, messages: list[Message], temperature: float = 0.2) -> Iterator[str]:
        if self._client is not None:
            try:  # pragma: no cover - network
                system, convo = _split_system(messages)
                kwargs: dict = dict(
                    model=settings.model_name,
                    max_tokens=settings.max_tokens,
                    messages=convo,
                    temperature=temperature,
                )
                if system:
                    kwargs["system"] = system
                with self._client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        if text:
                            yield text
                return
            except Exception as exc:  # pragma: no cover
                logger.error("LLM stream failed: %s", exc)
                raise ProviderError(describe_provider_error("LLM", exc), detail=str(exc)) from exc
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
