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
from langsmith.run_helpers import get_current_run_tree

from backend.thinkers.base import BaseThinker
from backend.thinkers.weather import WeatherThinker
from backend.thinkers.stocks import StocksThinker
from backend.thinkers.news import NewsThinker
from backend.thinkers.knowledge import KnowledgeThinker
from backend.thinkers.research import ResearchThinker
from backend.state.session_store import SessionStore

log = structlog.get_logger()


class ThinkerRouter:
    """Routes queries to specialized Thinker agents by domain."""

    def __init__(self, session_store: SessionStore):
        self._thinkers: dict[str, BaseThinker] = {}
        self._session_store = session_store
        self._register_thinkers()

    def _register_thinkers(self):
        """Register all available Thinker agents."""
        thinkers = [
            WeatherThinker(),
            StocksThinker(),
            NewsThinker(),
            KnowledgeThinker(),
            ResearchThinker(),
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
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.session_id = session_id
            run_tree.metadata["session_id"] = session_id

        # Check global cache before calling the thinker
        cached = await self._session_store.get_cached_result(domain=domain, query=query)
        if cached is not None:
            log.info("router.cache_hit", domain=domain, session_id=session_id)
            return cached

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

            # Cache the result for future identical queries
            await self._session_store.cache_thinker_result(
                domain=domain,
                query=query,
                result=result,
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
