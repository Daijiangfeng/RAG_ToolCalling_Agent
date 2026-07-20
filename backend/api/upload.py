"""Upload API: POST /api/upload -> ingest a PDF / Markdown into the KB."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Document
from app.schemas import UploadResponse
from rag.ingest import ingest_file

router = APIRouter(prefix="/api", tags=["upload"])

_ALLOWED = {".pdf", ".md", ".markdown", ".txt"}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = Path(settings.upload_dir) / (file.filename or "upload")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(dest)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}") from exc

    doc = Document(
        file_name=result["filename"],
        pages=result["pages"],
        chunks=result["chunks"],
        status=result["status"],
    )
    db.add(doc)
    db.commit()

    return UploadResponse(**result)
