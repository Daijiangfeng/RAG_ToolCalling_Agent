"""Chat API: POST /api/chat and POST /api/chat/stream."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agent.graph import run_agent, run_agent_streaming
from app.db import get_db
from app.models import ChatTrace
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


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
    """Server-Sent-Events stream with true LLM token streaming.

    The agent executes routing/retrieval/reranking first, then the generation
    phase streams tokens in real-time as they arrive from the LLM. Tool-based
    and rejection answers are emitted as a single chunk.
    """

    def event_gen():
        final_result = {}
        for event_type, data in run_agent_streaming(req.question, session_id=req.session_id):
            if event_type == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': data}, ensure_ascii=False)}\n\n"
            elif event_type == "done":
                final_result = data
                done_event = {"type": "done", **data}
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

        # Persist after streaming completes.
        _persist(db, req, final_result)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
