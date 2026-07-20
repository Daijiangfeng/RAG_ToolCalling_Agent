"""Datetime tool: current time and simple date arithmetic."""

from __future__ import annotations

from datetime import datetime, timedelta

SCHEMA = {
    "name": "datetime",
    "description": "获取当前日期时间,或计算若干天之后/之前的日期。当用户询问现在几点、今天日期或日期加减时使用。",
    "parameters": {
        "type": "object",
        "properties": {
            "offset_days": {
                "type": "integer",
                "description": "相对今天的偏移天数,正数表示未来,负数表示过去。默认 0。",
            }
        },
        "required": [],
    },
}


def now(offset_days: int = 0) -> dict:
    target = datetime.now() + timedelta(days=int(offset_days or 0))
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "target_date": target.strftime("%Y-%m-%d"),
        "weekday": target.strftime("%A"),
        "offset_days": int(offset_days or 0),
    }


def run(offset_days: int = 0, **_: object) -> dict:
    return now(offset_days)
