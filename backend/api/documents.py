"""Documents API: GET /api/documents -> list ingested KB files."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document
from app.schemas import DocumentInfo
from rag.vectorstore import get_vector_store

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentInfo]:
    rows = db.execute(select(Document).order_by(Document.created_time.desc())).scalars().all()
    return [DocumentInfo(**row.as_dict()) for row in rows]


@router.get("/documents/stats")
def stats(db: Session = Depends(get_db)) -> dict:
    rows = db.execute(select(Document)).scalars().all()
    return {
        "files": len(rows),
        "total_chunks": sum(r.chunks for r in rows),
        "vector_count": get_vector_store().count(),
    }
