"""
BaseThinker: abstract base class for all Thinker agents.

Each Thinker owns a domain, has a focused system prompt and tools,
and exposes a single async `think()` method.
"""

from abc import ABC, abstractmethod

from backend.state.user_context import ThinkResult, UserContext


class BaseThinker(ABC):
    """Abstract base for all Thinker agents."""

    domain: str  # Unique identifier used for routing, e.g. "weather"
    description: str  # Human-readable purpose
    model: str  # OpenAI model ID

    @abstractmethod
    async def think(self, query: str, context: list[dict], user_context: UserContext) -> ThinkResult:
        """
        Process a query and return a spoken-word response with optional context updates.

        Args:
            query:        The user's question, rephrased by the Responder.
            context:      Recent conversation turns from Redis for grounding.
            user_context: Persistent user preferences, memory, and signals.

        Returns:
            ThinkResult with a concise, naturally-spoken response and optional
            ContextUpdate describing any user preferences/facts to persist.
        """
        ...
