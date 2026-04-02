"""
Application settings loaded from environment variables / .env file.
"""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so third-party libraries (openai-agents SDK, etc.)
# can read OPENAI_API_KEY and other vars directly from the environment.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_base_url: str = "https://us.api.openai.com/v1"

    # Realtime API (the Responder)
    # Use a versioned model name — the undated alias is sometimes rejected.
    # Valid options: gpt-4o-realtime-preview-2024-12-17, gpt-4o-mini-realtime-preview-2024-12-17
    realtime_model: str = "gpt-realtime"
    realtime_voice: str = "shimmer"

    # transcript model
    transcript_model: str = "gpt-4o-mini-transcribe"

    # Thinker models
    thinker_model: str = "gpt-4.1-mini"  # Weather, Stocks (fast)
    thinker_model_advanced: str = "gpt-4.1"  # News, Knowledge (smarter)

    # Redis
    redis_url: str = "redis://localhost:6379"

    # External API keys for thinker tools
    finnhub_api_key: str = ""  # https://finnhub.io/register (free: 60 req/min)
    newsapi_api_key: str = ""  # https://newsapi.org/register (free: 100 req/day)

    # LangSmith observability (optional)
    langsmith_api_key: str = ""
    langsmith_project: str = "responder-thinker"
    langsmith_tracing_enabled: bool = False


settings = Settings()

# Disable the Agents SDK's built-in tracing — we use LangSmith instead.
# os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


def make_openai_client() -> "AsyncOpenAI":
    """
    Create an AsyncOpenAI client, optionally wrapped with LangSmith tracing.

    Wrapping is only applied when tracing is enabled — avoids unnecessary
    overhead on every API call in non-traced environments.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    if settings.langsmith_tracing_enabled:
        from langsmith.wrappers import wrap_openai

        client = wrap_openai(client)

    return client
