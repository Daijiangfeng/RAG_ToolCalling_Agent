"""FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import chat, documents, evaluation, upload
from app.config import BACKEND_DIR, settings
from app.db import init_db
from app.errors import ProviderError
from app.llm import get_llm
from app.logging import get_logger
from rag.ingest import ingest_file
from rag.vectorstore import get_vector_store

logger = get_logger(__name__)

app = FastAPI(
    title="Intelligent Knowledge Agent Platform",
    description="LangGraph + FastAPI RAG & Tool-Calling agent backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(evaluation.router)
app.include_router(documents.router)


@app.exception_handler(ProviderError)
def _provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    """Turn a configured-provider failure into a friendly HTTP 502.

    This keeps the raw upstream body (e.g. ``{"error":{"code":"1001"...}}``) out
    of the client response and gives an actionable Chinese message instead.
    """
    logger.error("ProviderError: %s | detail=%s", exc.message, exc.detail)
    return JSONResponse(status_code=502, content={"detail": exc.message})


def _store_is_compatible(store) -> bool:
    """Probe the store with the active embedder.

    Switching embedding models (e.g. hash 512-dim -> embedding-3 2048-dim)
    leaves a persisted collection whose vector dimensionality no longer matches
    the current embedder.  Querying surfaces that mismatch so we can rebuild the
    collection instead of failing every request.
    """
    try:
        store.query("healthcheck", top_k=1)
        return True
    except Exception as exc:  # pragma: no cover - depends on persisted state
        logger.warning("Vector store probe failed (%s) -> rebuilding collection", exc)
        return False


def seed_knowledge_base() -> None:
    """Ingest bundled sample docs so demos/evaluation work out of the box."""
    store = get_vector_store()
    if store.count() > 0:
        if _store_is_compatible(store):
            return
        # Dimension mismatch (embedding model changed): rebuild from scratch.
        store.reset()
    seed_dir = BACKEND_DIR / "data" / "seed_docs"
    if not seed_dir.exists():
        return
    for path in sorted(seed_dir.glob("*")):
        if path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            try:
                ingest_file(path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to seed %s: %s", path.name, exc)
    logger.info("Seeded knowledge base with %d vectors", store.count())


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    llm = get_llm()
    logger.info("LLM mode: %s | embedding: %s | reranker: %s",
                llm.mode, settings.embedding_backend, settings.reranker_backend)
    # Validate the configured provider up-front so an invalid/expired API key or
    # unavailable model surfaces as a clear startup log rather than a confusing
    # per-request failure later on.
    if settings.has_llm:
        ok, message = llm.verify()
        if ok:
            logger.info("LLM 凭证校验通过：%s", message)
        else:
            logger.error("LLM 凭证校验失败：%s", message)
    seed_knowledge_base()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": get_llm().mode,
        "vector_count": get_vector_store().count(),
    }


@app.get("/")
def root() -> dict:
    return {"name": "Intelligent Knowledge Agent Platform", "docs": "/docs"}
