"""Chat API: POST /api/chat and POST /api/chat/stream."""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agent.graph import run_agent
from app.db import get_db
from app.models import ChatTrace
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])

# CJK（中日韩）单字与非 CJK 词块分别成一个流式 token：
# 中文答案没有空格，原先的 \S+ 会把整段中文当成一个 token，导致中文场景没有
# 逐字流式效果。这里对 CJK 逐字切分，其余按“词 + 尾随空白”切分。
_STREAM_TOKEN = re.compile(
    r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
    r"|\S+\s*"
    r"|\s+"
)


def _persist(db: Session, req: ChatRequest, result: dict) -> None:
    db.add(
        ChatTrace(
            question=req.question,
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            trace_json=json.dumps(result.get("trace", []), ensure_ascii=False),
        )
    )
    db.commit()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    result = run_agent(req.question, session_id=req.session_id)
    _persist(db, req, result)
    return ChatResponse(**result)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """Server-Sent-Events stream: token deltas, then a final metadata event.

    The agent runs to completion first (so routing / retrieval / rerank / tools
    are resolved), then the final answer is streamed token-by-token, followed by
    a ``done`` event carrying sources / confidence / tools / trace.
    """
    result = run_agent(req.question, session_id=req.session_id)
    _persist(db, req, result)

    def event_gen():
        answer = result.get("answer", "")
        for token in _STREAM_TOKEN.findall(answer):
            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        done = {
            "type": "done",
            "answer": answer,
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "tools": result.get("tools", []),
            "trace": result.get("trace", []),
            "intent": result.get("intent", ""),
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
