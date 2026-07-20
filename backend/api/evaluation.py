"""Evaluation API: GET /api/evaluation and POST /api/evaluation/run."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import EvaluationResponse
from rag.evaluator import run_evaluation

router = APIRouter(prefix="/api", tags=["evaluation"])

_cache: dict | None = None


def _to_response(summary: dict) -> EvaluationResponse:
    return EvaluationResponse(
        total=summary.get("total", 0),
        retrieval=summary.get("retrieval", {}),
        generation=summary.get("generation", {}),
        safety=summary.get("safety", {}),
        per_type=summary.get("per_type", {}),
        generated_at=summary.get("generated_at"),
    )


@router.get("/evaluation", response_model=EvaluationResponse)
def get_evaluation() -> EvaluationResponse:
    global _cache
    if _cache is None:
        _cache = run_evaluation(write_report=True)
    return _to_response(_cache)


@router.post("/evaluation/run", response_model=EvaluationResponse)
def run_eval() -> EvaluationResponse:
    global _cache
    _cache = run_evaluation(write_report=True)
    return _to_response(_cache)
