"""Application configuration.

All runtime knobs are exposed via environment variables (with an optional
``.env`` file).  The platform is designed to run in a *hybrid* mode: it will use
real OpenAI-compatible APIs / BGE models when they are available, and it will
gracefully fall back to fully offline, deterministic implementations otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
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

    # --- LLM (智谱 GLM，Anthropic 兼容端点) --------------------------------
    # 智谱 GLM 通过 Anthropic 兼容端点调用：使用 ANTHROPIC_AUTH_TOKEN 作为
    # Bearer 令牌，Anthropic SDK 会以 ``Authorization: Bearer <token>`` 发送
    # （而非默认的 x-api-key 头）。同一个令牌也用于 OpenAI 兼容的向量端点。
    # 兼容旧的 AUTH_TOKEN / OPENAI_API_KEY 变量名，便于平滑迁移。
    model_name: str = "glm-4.5-air"
    auth_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ANTHROPIC_AUTH_TOKEN", "AUTH_TOKEN", "OPENAI_API_KEY"
        ),
    )
    # Anthropic 兼容端点（对话/生成）。
    anthropic_base_url: str = "https://open.bigmodel.cn/api/anthropic"
    # Anthropic messages API 要求显式的 max_tokens。
    max_tokens: int = 1024
    # OpenAI 兼容端点（仅用于向量 embedding）。
    openai_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # --- Embedding --------------------------------------------------------
    # one of: "openai", "bge", "hash" (hash == offline fallback)
    embedding_backend: str = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "embedding-3"
    embedding_dim: int = 2048

    # --- Reranker ---------------------------------------------------------
    # one of: "cross-encoder", "zhipu", "lexical"
    #   zhipu   == 智谱在线 rerank（标准 HTTP Bearer，模型编码 "rerank"）
    #   lexical == 离线兜底回退
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
    web_search_provider: str = "mock"  # "tavily" | "mock"（仅实现 Tavily，无 key 回退 mock）

    # --- Database ---------------------------------------------------------
    database_url: str = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"

    # --- Redis (optional, for distributed session memory) -----------------
    redis_url: str = ""

    # --- Misc -------------------------------------------------------------
    upload_dir: str = str(DATA_DIR / "uploads")
    cors_origins: str = "*"

    @field_validator("auth_token", mode="before")
    @classmethod
    def _strip_auth_token(cls, value: object) -> str:
        # 去除令牌两端的空白/换行：粘贴 .env 或 export 环境变量时常混入尾随
        # 空格/换行，若原样发给 SDK 会得到空的 ``Authorization: Bearer ``，
        # 触发智谱 code 1001（未收到 Authentication 参数）。
        return str(value).strip() if value is not None else ""

    @property
    def has_llm(self) -> bool:
        # 仅当去空白后仍非空才视为已配置令牌；纯空白视为未配置 -> 离线 mock，
        # 避免发送空 Bearer 头导致的 1001 鉴权失败。
        return bool(self.auth_token and self.auth_token.strip())

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
