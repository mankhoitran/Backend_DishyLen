"""Application configuration and environment loading."""

from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Food Agent API"
    app_version: str = "1.0.0"

    gemini_api_key: str = Field(default="AIzaSyBhK3yz4ZJmnDSUznI3aXQDWzZDxJcKkSw", alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    vllm_base_url: str = Field(default="http://192.168.20.150:8003/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="", alias="VLLM_API_KEY")
    vllm_model: str = Field(default="Qwen3-4B", alias="VLLM_MODEL")
    vllm_timeout_seconds: int = Field(default=60, alias="VLLM_TIMEOUT_SECONDS")
    duckduckgo_max_results: int = Field(default=5, alias="DUCKDUCKGO_MAX_RESULTS")

    sqlite_db_url: str = Field(default="sqlite:///./food_agent.db", alias="SQLITE_DB_URL")

    cors_origins: str = Field(
        default="http://localhost:8080,http://127.0.0.1:8080",
        alias="CORS_ORIGINS",
    )
    uploads_dir: str = Field(default="uploads", alias="UPLOADS_DIR")
    scan_max_items: int = Field(default=6, alias="SCAN_MAX_ITEMS")
    scan_fallback_items: str = Field(
        default="Pad Thai,Green Curry,Caesar Salad",
        alias="SCAN_FALLBACK_ITEMS",
    )

    max_agent_steps: int = 6
    gemini_timeout_seconds: int = 45
    gemini_call_pause_seconds: int = Field(default=1, alias="GEMINI_CALL_PAUSE_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
