"""
Redis-backed session store.

Manages per-session state including:
  - Conversation history (for Thinker context)
  - Tool call result caching (avoid redundant Thinker calls)
  - Session metadata

Redis is the right choice here because:
  - Thinkers are stateless — state lives outside them
  - Multiple Thinkers can read the same conversation context
  - Tool results can be cached with TTLs (weather doesn't change every second)
  - In production, this scales horizontally across backend instances
"""

import json
import time

import redis.asyncio as redis
import structlog

log = structlog.get_logger()

# Key patterns
CONVERSATION_KEY = "session:{session_id}:conversation"
CACHE_KEY = "session:{session_id}:cache:{domain}:{query_hash}"
METADATA_KEY = "session:{session_id}:meta"

# TTLs
SESSION_TTL = 3600  # 1 hour
CACHE_TTL_WEATHER = 600  # 10 minutes
CACHE_TTL_STOCKS = 60  # 1 minute
CACHE_TTL_NEWS = 300  # 5 minutes
CACHE_TTL_DEFAULT = 120  # 2 minutes


class SessionStore:
    """Redis-backed session state management."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis: redis.Redis | None = None

    async def connect(self):
        self._redis = redis.from_url(self._redis_url, decode_responses=True)
        await self._redis.ping()
        log.info("redis.connected", url=self._redis_url)

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()

    async def append_turn(self, session_id: str, role: str, content: str):
        """Append a conversation turn to the session history."""
        key = CONVERSATION_KEY.format(session_id=session_id)
        turn = json.dumps(
            {
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
        )
        await self._redis.rpush(key, turn)
        await self._redis.expire(key, SESSION_TTL)

    async def get_conversation_context(
        self, session_id: str, max_turns: int = 10
    ) -> list[dict]:
        """Get recent conversation history for Thinker context."""
        key = CONVERSATION_KEY.format(session_id=session_id)
        turns = await self._redis.lrange(key, -max_turns, -1)
        return [json.loads(turn) for turn in turns]

    async def cache_thinker_result(
        self,
        session_id: str,
        domain: str,
        query: str,
        result: str,
    ):
        """Cache a Thinker result to avoid redundant calls."""
        query_hash = str(hash(query))[:12]
        key = CACHE_KEY.format(
            session_id=session_id,
            domain=domain,
            query_hash=query_hash,
        )

        ttl = {
            "weather": CACHE_TTL_WEATHER,
            "stocks": CACHE_TTL_STOCKS,
            "news": CACHE_TTL_NEWS,
        }.get(domain, CACHE_TTL_DEFAULT)

        await self._redis.setex(key, ttl, result)

    async def get_cached_result(
        self, session_id: str, domain: str, query: str
    ) -> str | None:
        """Check for a cached Thinker result."""
        query_hash = str(hash(query))[:12]
        key = CACHE_KEY.format(
            session_id=session_id,
            domain=domain,
            query_hash=query_hash,
        )
        return await self._redis.get(key)
