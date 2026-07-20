"""Web search tool.

Supports Tavily / SerpAPI when a key is configured; otherwise returns
deterministic mock results so the "real-time info" demo works offline.
"""

from __future__ import annotations

from app.config import settings
from app.logging import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "web_search",
    "description": "搜索互联网获取实时/最新信息,例如最新新闻、实时数据、当前事件。当问题涉及知识库之外的最新信息时使用。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回结果条数,默认 3"},
        },
        "required": ["query"],
    },
}


def _mock_results(query: str, max_results: int) -> list[dict]:
    return [
        {
            "title": f"[Mock] 关于“{query}”的结果 {i + 1}",
            "url": f"https://example.com/search?q={query}&r={i + 1}",
            "snippet": f"这是针对“{query}”的第 {i + 1} 条模拟搜索摘要(离线模式,未配置搜索 API Key)。",
        }
        for i in range(max_results)
    ]


def _tavily(query: str, max_results: int) -> list[dict]:  # pragma: no cover - network
    import httpx

    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.web_search_api_key,
            "query": query,
            "max_results": max_results,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def search(query: str, max_results: int = 3) -> dict:
    max_results = max(1, min(int(max_results or 3), 10))
    provider = settings.web_search_provider.lower()
    if provider == "tavily" and settings.web_search_api_key:
        try:  # pragma: no cover - network
            results = _tavily(query, max_results)
            return {"query": query, "provider": "tavily", "results": results}
        except Exception as exc:  # pragma: no cover
            logger.warning("tavily search failed (%s) -> mock", exc)
    return {"query": query, "provider": "mock", "results": _mock_results(query, max_results)}


def run(query: str, max_results: int = 3, **_: object) -> dict:
    return search(query, max_results)
