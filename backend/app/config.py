"""Application configuration.

All runtime knobs are exposed via environment variables (with an optional
``.env`` file).  The platform is designed to run in a *hybrid* mode: it will use
real OpenAI-compatible APIs / BGE models when they are available, and it will
gracefully fall back to fully offline, deterministic implementations otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Central settings object, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenAI compatible) -----------------------------------------
    # 智谱 GLM 使用 AUTH_TOKEN 作为 Bearer 令牌（Authorization: Bearer <token>）。
    # 兼容旧的 OPENAI_API_KEY 变量名，便于平滑迁移。
    model_name: str = "glm-4.5-air"
    auth_token: str = Field(
        default="",
        validation_alias=AliasChoices("AUTH_TOKEN", "OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # --- Embedding --------------------------------------------------------
    # one of: "openai", "bge", "hash" (hash == offline fallback)
    embedding_backend: str = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "embedding-3"
    embedding_dim: int = 2048

    # --- Reranker ---------------------------------------------------------
    # one of: "cross-encoder", "lexical" (lexical == offline fallback)
    reranker_backend: str = "lexical"
    reranker_model: str = "BAAI/bge-reranker-large"

    # --- Vector store -----------------------------------------------------
    vector_backend: str = "chroma"
    chroma_dir: str = str(DATA_DIR / "chroma")
    collection_name: str = "knowledge_base"

    # --- RAG parameters ---------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 20
    rerank_top_n: int = 5
    confidence_threshold: float = 0.30

    # --- Tools ------------------------------------------------------------
    web_search_api_key: str = ""
    web_search_provider: str = "mock"  # "tavily" | "serpapi" | "mock"

    # --- Database ---------------------------------------------------------
    database_url: str = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"

    # --- Misc -------------------------------------------------------------
    upload_dir: str = str(DATA_DIR / "uploads")
    cors_origins: str = "*"

    @property
    def has_llm(self) -> bool:
        return bool(self.auth_token)

    @property
    def rejection_message(self) -> str:
        return "知识库中没有足够信息回答该问题。"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
