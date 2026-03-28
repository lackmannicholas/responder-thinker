"""
LangSmith observability setup.

Every Thinker call is decorated with @traceable, so traces are emitted
automatically once the environment is configured here. The ThinkerRouter
is also traced, giving you a top-level span that shows routing decisions
and downstream Thinker execution in a single trace.
"""

import os

import structlog

log = structlog.get_logger()


def setup_tracing() -> None:
    """
    Configure LangSmith tracing via environment variables.

    Called once at application startup. If tracing is disabled or the
    API key is missing, this is a no-op — the app runs fine without it.
    """
    # Import here to avoid circular import at module load time
    from backend.config import settings

    log.info(f"tracing.setup", enabled=settings.langsmith_tracing_enabled, project=settings.langsmith_project, api_key_set=bool(settings.langsmith_api_key))

    if not settings.langsmith_tracing_enabled or not settings.langsmith_api_key:
        log.info("tracing.disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    log.info(
        "tracing.enabled",
        project=settings.langsmith_project,
    )
