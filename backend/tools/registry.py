"""Tool registry.

Aggregates tool schemas and callables so the router / agent can:

* expose the JSON schemas to the LLM for tool selection + argument generation;
* dispatch a chosen tool by name.
"""

from __future__ import annotations

from typing import Any, Callable

from tools import calculator, datetime_tool, file_query, web_search

_TOOLS: dict[str, dict[str, Any]] = {
    calculator.SCHEMA["name"]: {"schema": calculator.SCHEMA, "run": calculator.run},
    web_search.SCHEMA["name"]: {"schema": web_search.SCHEMA, "run": web_search.run},
    file_query.SCHEMA["name"]: {"schema": file_query.SCHEMA, "run": file_query.run},
    datetime_tool.SCHEMA["name"]: {"schema": datetime_tool.SCHEMA, "run": datetime_tool.run},
}


def list_schemas() -> list[dict[str, Any]]:
    return [t["schema"] for t in _TOOLS.values()]


def get_tool(name: str) -> Callable[..., Any] | None:
    entry = _TOOLS.get(name)
    return entry["run"] if entry else None


def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    fn = get_tool(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"invalid arguments for {name}: {exc}"}
