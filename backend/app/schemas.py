"""Pydantic request / response schemas for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    step: str
    summary: str
    tool: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    tools: list[ToolCall] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    intent: str = ""


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    status: str


class DocumentInfo(BaseModel):
    id: int
    file_name: str
    pages: int
    chunks: int
    status: str
    created_time: str | None = None


class RetrievalMetrics(BaseModel):
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0


class GenerationMetrics(BaseModel):
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    faithfulness: float = 0.0


class SafetyMetrics(BaseModel):
    hallucination_rate: float = 0.0


class EvaluationResponse(BaseModel):
    total: int = 0
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = Field(default_factory=GenerationMetrics)
    safety: SafetyMetrics = Field(default_factory=SafetyMetrics)
    per_type: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
