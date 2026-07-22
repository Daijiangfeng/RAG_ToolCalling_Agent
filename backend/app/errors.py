"""Application-level exceptions.

``ProviderError`` represents a *user-facing* failure when talking to an external
LLM / embedding provider (auth, quota, network, invalid model...).  Unlike the
raw SDK exception it carries a clean Chinese message that the API layer turns
into a friendly HTTP 502 response instead of leaking the provider's raw error
body (e.g. ``{"error":{"code":"1001",...}}``) as a 500.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Raised when a configured LLM/embedding provider call fails."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


def describe_provider_error(scope: str, exc: Exception) -> str:
    """Build a friendly Chinese message from a raw provider exception.

    ``scope`` is a short label such as ``"LLM"`` or ``"向量模型"``.
    """
    raw = str(exc)
    low = raw.lower()
    # Auth failures: Zhipu 1001 (missing Authorization header) / 1000 (auth failed)
    # / HTTP 401 / generic authentication wording.
    auth_markers = ("1001", "1000", "401", "authorization", "authentication", "身份验证", "unauthorized")
    if any(m in low for m in auth_markers):
        return (
            f"{scope}鉴权失败：请求未通过智谱身份验证（如 code 1001/1000、HTTP 401）。"
            "请检查 backend/.env 中的 ANTHROPIC_AUTH_TOKEN（智谱 Bearer 令牌）是否正确、未过期，并重启服务。"
        )
    if "quota" in low or "1002" in raw or "1113" in raw or "额度" in raw:
        return f"{scope}调用受限：账户额度或权限不足，请检查智谱账户后重试。"
    if "model" in low and ("not" in low or "无效" in raw or "不存在" in raw):
        return f"{scope}模型不存在或不可用，请检查配置的模型名称。"
    return f"{scope}调用失败：{raw[:200]}"
