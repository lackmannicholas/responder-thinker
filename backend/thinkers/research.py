"""
Research Thinker — simulates a long-running backend task.

Adds a configurable delay (default 10s) before returning a result so you
can test the Responder's ability to keep the conversation going while a
Thinker is still working.
"""

import asyncio

import structlog
from langsmith import traceable

from backend.thinkers.base import BaseThinker

log = structlog.get_logger()

# Simulated delay in seconds
SIMULATED_DELAY = 30


class ResearchThinker(BaseThinker):
    domain = "research"
    description = "Deep research tasks (slow — used for testing continued conversation)"
    model = "mock"

    @traceable(name="research_thinker.think")
    async def think(self, query: str, context: list[dict]) -> str:
        log.info(
            "research_thinker.started",
            query=query[:100],
            delay=SIMULATED_DELAY,
        )

        # Simulate a long-running backend operation
        await asyncio.sleep(SIMULATED_DELAY)

        log.info("research_thinker.complete", query=query[:100])

        return (
            f"After extensive research, here's what I found about your question: "
            f'"{query}". This is a simulated result from a long-running task that '
            f"took about {SIMULATED_DELAY} seconds to complete. In a real system "
            f"this could be a complex database query, a multi-step API pipeline, "
            f"or an agentic research workflow."
        )
