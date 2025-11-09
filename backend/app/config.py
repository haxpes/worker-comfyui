"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # OpenAI API
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/ai_sme",
        description="PostgreSQL connection string"
    )
    db_pool_size: int = Field(default=20, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=50)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_pool_recycle: int = Field(default=3600, ge=0)

    # Storage
    corpus_base_path: str = Field(
        default="/home/user/worker-comfyui/storage",
        description="Base path for corpus storage"
    )

    # Retrieval Parameters
    bm25_k: int = Field(default=20, ge=1, le=100, description="BM25 top-k results")
    dense_k: int = Field(default=20, ge=1, le=100, description="Dense retrieval top-k")
    fusion_k: int = Field(default=10, ge=1, le=50, description="After RRF fusion")
    mmr_k: int = Field(default=5, ge=1, le=20, description="After MMR diversity")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR diversity param")
    citations_count: int = Field(default=3, ge=1, le=10, description="Number of citations")

    # LLM Models
    default_model: str = Field(default="gpt-4o-mini", description="Default LLM model")
    complex_model: str = Field(default="gpt-4o", description="Complex reasoning model")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")

    # Reranking
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    rerank_top_k: int = Field(default=10, ge=1, le=50)

    # Validation
    validator_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum validator score"
    )

    # Memory
    conversation_summary_interval: int = Field(
        default=10,
        ge=1,
        description="Messages before creating summary"
    )
    max_conversation_history: int = Field(
        default=50,
        ge=1,
        description="Max messages to keep in memory"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Log file path")
    log_format: str = Field(default="json", description="Log format: json or text")

    # Observability
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="ai-sme-system", description="LangSmith project")
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")

    otel_exporter_otlp_endpoint: str = Field(default="", description="OTLP endpoint")
    otel_exporter_otlp_headers: str = Field(default="", description="OTLP headers")

    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, le=32, description="Uvicorn workers")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="CORS allowed origins"
    )

    # Performance
    enable_cache: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    max_concurrent_requests: int = Field(default=100, ge=1, description="Max concurrent requests")
    request_timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")

    # Feature Flags (for phased rollout)
    enable_raptor: bool = Field(default=False, description="Enable RAPTOR retrieval")
    enable_graphrag: bool = Field(default=False, description="Enable GraphRAG")
    enable_memory: bool = Field(default=False, description="Enable memory system")
    enable_router: bool = Field(default=False, description="Enable router MoE")
    enable_validator: bool = Field(default=False, description="Enable validator")


# Global settings instance
settings = Settings()
