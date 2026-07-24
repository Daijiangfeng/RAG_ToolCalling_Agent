"""Upload API: POST /api/upload -> ingest a PDF / Markdown into the KB."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.db import get_db
from app.models import Document
from app.schemas import UploadResponse
from rag.ingest import ingest_file

router = APIRouter(prefix="/api", tags=["upload"])

_ALLOWED = {".pdf", ".md", ".markdown", ".txt"}


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> UploadResponse:
    # 只取文件名本身，剔除任意目录分量，防止路径穿越 / 覆盖外部文件。
    safe_name = Path(file.filename or "upload").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = Path(settings.upload_dir) / safe_name

    def _write_and_ingest() -> dict:
        # 阻塞的文件写入 + 解析入库放到线程池，避免阻塞事件循环。
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return ingest_file(dest)

    try:
        result = await run_in_threadpool(_write_and_ingest)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}") from exc

    # 同名文档按文件名去重/更新，避免重复上传产生多条 Document 行。
    existing = db.query(Document).filter(Document.file_name == result["filename"]).first()
    if existing is not None:
        existing.pages = result["pages"]
        existing.chunks = result["chunks"]
        existing.status = result["status"]
    else:
        db.add(
            Document(
                file_name=result["filename"],
                pages=result["pages"],
                chunks=result["chunks"],
                status=result["status"],
            )
        )
    db.commit()

    return UploadResponse(**result)
