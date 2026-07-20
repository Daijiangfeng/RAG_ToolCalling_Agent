"""ORM models for documents and chat traces."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(512), index=True)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processed")
    created_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "pages": self.pages,
            "chunks": self.chunks,
            "status": self.status,
            "created_time": self.created_time.isoformat() if self.created_time else None,
        }


class ChatTrace(Base):
    __tablename__ = "chat_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    trace_json: Mapped[str] = mapped_column(Text, default="[]")
    created_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
