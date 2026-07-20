"""Tests for tools and the tool registry."""

from tools import registry
from tools.calculator import calculate
from tools.datetime_tool import now


def test_calculator_basic():
    assert calculate("12345*678")["result"] == 12345 * 678
    assert calculate("(3+4)*25")["result"] == 175


def test_calculator_rejects_code():
    out = calculate("__import__('os').system('echo hi')")
    assert "error" in out


def test_datetime_offset():
    out = now(offset_days=1)
    assert "target_date" in out and out["offset_days"] == 1


def test_registry_dispatch():
    schemas = registry.list_schemas()
    names = {s["name"] for s in schemas}
    assert {"calculator", "web_search", "datetime", "file_query"} <= names
    result = registry.execute("calculator", {"expression": "2+2"})
    assert result["result"] == 4


def test_registry_unknown_tool():
    assert "error" in registry.execute("nope", {})
