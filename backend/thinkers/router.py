"""
Thinker Router.

Routes classified intents to the appropriate Thinker agent.
This is intentionally simple — a dict lookup, not an ML model.
The Responder already did the hard work of intent classification.

In production, you might add:
  - Fallback chains (if primary thinker fails, try a generalist)
  - Concurrent thinker calls for compound intents
  - Circuit breakers for thinker health
  - Priority queues for latency-sensitive vs. background thinkers
"""

import structlog
from langsmith import traceable

from backend.thinkers.base import BaseThinker
from backend.thinkers.weather import WeatherThinker
from backend.thinkers.stocks import StocksThinker
from backend.thinkers.news import NewsThinker
from backend.thinkers.knowledge import KnowledgeThinker

log = structlog.get_logger()


class ThinkerRouter:
    """Routes queries to specialized Thinker agents by domain."""

    def __init__(self):
        self._thinkers: dict[str, BaseThinker] = {}
        self._register_thinkers()

    def _register_thinkers(self):
        """Register all available Thinker agents."""
        thinkers = [
            WeatherThinker(),
            StocksThinker(),
            NewsThinker(),
            KnowledgeThinker(),
        ]
        for thinker in thinkers:
            self._thinkers[thinker.domain] = thinker
            log.info("router.registered", domain=thinker.domain)

    @traceable(name="thinker_router.think")
    async def think(
        self,
        domain: str,
        query: str,
        context: list[dict],
        session_id: str,
    ) -> str:
        """
        Route a query to the appropriate Thinker.

        Returns the Thinker's response as a string, ready for the
        Responder to deliver via voice.
        """
        thinker = self._thinkers.get(domain)

        if thinker is None:
            log.warning("router.unknown_domain", domain=domain)
            # Fallback to knowledge thinker for unknown domains
            thinker = self._thinkers.get("knowledge")
            if thinker is None:
                return "I wasn't able to find that information right now."

        log.info(
            "router.routing",
            session_id=session_id,
            domain=domain,
            query=query[:100],
        )

        try:
            result = await thinker.think(query=query, context=context)
            log.info(
                "router.success",
                session_id=session_id,
                domain=domain,
                result_length=len(result),
            )
            return result
        except Exception as e:
            log.error(
                "router.thinker_error",
                session_id=session_id,
                domain=domain,
                error=str(e),
            )
            return "I ran into an issue looking that up. Could you try asking again?"
