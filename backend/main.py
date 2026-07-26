"""FastAPI application entry point."""

from __future__ import annotations

import os
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

# 解析允许的源；含通配符 "*" 时必须关闭 credentials（浏览器不允许
# allow_origins=["*"] 与 allow_credentials=True 共存，否则 CORS 完全失效）。
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
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
    # Initialize memory backend (Redis if available, else in-memory).
    from agent.memory import init_memory_backend
    init_memory_backend()

    # Initialize OpenTelemetry tracing (no-op if OTEL_ENABLED!=true).
    from app.telemetry import setup_telemetry
    setup_telemetry()

    # Initialize DI container with default factories.
    from app.container import setup_container
    setup_container()

    llm = get_llm()
    logger.info("LLM mode: %s | embedding: %s | reranker: %s",
                llm.mode, settings.embedding_backend, settings.reranker_backend)
    _log_token_diagnostics()
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


_TOKEN_ENV_VARS = ("ANTHROPIC_AUTH_TOKEN", "AUTH_TOKEN", "OPENAI_API_KEY")


def _mask_token(token: str) -> str:
    """Mask a token for logging: keep a short prefix/suffix, hide the middle."""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}***{token[-3:]}"


def _log_token_diagnostics() -> None:
    """Surface the effective auth-token state so a blank/shadowed token is obvious.

    智谱 code 1001（Header 中未收到 Authentication 参数）的根因通常是
    有效令牌为空/纯空白，或被 shell 中的空环境变量遮盖了 backend/.env。
    这里在启动日志中把有效令牌（掩码）与环境变量遮盖情况直接暴露出来。
    """
    shadowing = [name for name in _TOKEN_ENV_VARS if os.environ.get(name) is not None]
    if settings.has_llm:
        token = settings.auth_token
        logger.info(
            "LLM 凭证已加载（len=%d, mask=%s）%s",
            len(token),
            _mask_token(token),
            f"；环境变量 {shadowing} 已设置并优先于 backend/.env" if shadowing else "",
        )
        return
    # No usable token. Distinguish "nothing set" from "set but blank/whitespace",
    # since the latter is the exact trigger for an empty Bearer header -> 1001.
    blank_env = [
        name for name in _TOKEN_ENV_VARS
        if (raw := os.environ.get(name)) is not None and not raw.strip()
    ]
    if blank_env:
        logger.warning(
            "检测到空白的鉴权环境变量 %s，它会遮盖 backend/.env 中的令牌，"
            "导致以空 Bearer 头调用智谱（code 1001）。请在启动 uvicorn/PyCharm 的 "
            "shell 中清除这些变量，或赋予有效令牌。当前以离线 mock 模式运行。",
            blank_env,
        )
    else:
        logger.info("未配置 ANTHROPIC_AUTH_TOKEN -> 以离线 mock 模式运行。")


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
