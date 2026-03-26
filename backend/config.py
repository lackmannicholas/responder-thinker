"""
Application settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str

    # Realtime API (the Responder)
    # Use a versioned model name — the undated alias is sometimes rejected.
    # Valid options: gpt-4o-realtime-preview-2024-12-17, gpt-4o-mini-realtime-preview-2024-12-17
    realtime_model: str = "gpt-realtime"
    realtime_voice: str = "alloy"

    # Thinker models
    thinker_model: str = "gpt-4.1-mini"  # Weather, Stocks (fast)
    thinker_model_advanced: str = "gpt-4.1"  # News, Knowledge (smarter)

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LangSmith observability (optional)
    langsmith_api_key: str = ""
    langsmith_project: str = "responder-thinker"
    langsmith_tracing_enabled: bool = False


settings = Settings()
