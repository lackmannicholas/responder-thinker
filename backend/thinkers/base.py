"""
BaseThinker: abstract base class for all Thinker agents.

Each Thinker owns a domain, has a focused system prompt and tools,
and exposes a single async `think()` method.
"""

from abc import ABC, abstractmethod


class BaseThinker(ABC):
    """Abstract base for all Thinker agents."""

    domain: str          # Unique identifier used for routing, e.g. "weather"
    description: str     # Human-readable purpose
    model: str           # OpenAI model ID

    @abstractmethod
    async def think(self, query: str, context: list[dict]) -> str:
        """
        Process a query and return a spoken-word response.

        Args:
            query:   The user's question, rephrased by the Responder.
            context: Recent conversation turns from Redis for grounding.

        Returns:
            A concise, naturally-spoken response (no markdown, no bullet points).
        """
        ...
