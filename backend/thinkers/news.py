"""
News Thinker — handles news and current events queries.

Uses the OpenAI Agents SDK with local function tools that call the NewsAPI.org API.
Requires NEWSAPI_API_KEY env var (free tier: 100 req/day, https://newsapi.org/register).
Uses gpt-4.1 (advanced model) because news summarization requires
more reasoning than simple data lookups.
"""

import json
import structlog
from datetime import datetime, timedelta, timezone

import httpx
from agents import Agent, Runner, function_tool
from langsmith import traceable

from backend.config import settings
from backend.thinkers.base import BaseThinker
from backend.state.user_context import ThinkResult, UserContext

log = structlog.get_logger()

_NEWSAPI_BASE = "https://newsapi.org/v2"


_MOCK_HEADLINES: dict[str, list[dict]] = {
    "ai": [
        {"title": "OpenAI Releases New Real-Time Voice Model With Tool-Use Support", "source": "Reuters", "published_at": "2 hours ago"},
        {"title": "EU AI Act Enforcement Begins With First Round of Compliance Audits", "source": "Financial Times", "published_at": "5 hours ago"},
        {"title": "Google DeepMind Publishes Breakthrough in Long-Context Reasoning", "source": "Ars Technica", "published_at": "8 hours ago"},
    ],
    "economy": [
        {"title": "Federal Reserve Holds Rates Steady, Signals Potential Cut in September", "source": "Wall Street Journal", "published_at": "1 hour ago"},
        {"title": "US Jobs Report Shows Stronger-Than-Expected Hiring in Services Sector", "source": "Bloomberg", "published_at": "4 hours ago"},
    ],
    "sports": [
        {"title": "NBA Finals Game 5: Celtics Take 3-2 Series Lead in Overtime Thriller", "source": "ESPN", "published_at": "3 hours ago"},
        {"title": "FIFA Announces Expanded Club World Cup Format for 2027", "source": "BBC Sport", "published_at": "5 hours ago"},
    ],
}


@traceable(name="news_thinker.think._mock_headlines")
def _mock_headlines(topic: str, count: int = 3) -> str:
    normalized = topic.lower().strip()
    headlines = _MOCK_HEADLINES.get(
        normalized,
        [
            {"title": f"Major Developments Expected in {topic.capitalize()} This Quarter", "source": "Reuters", "published_at": "2 hours ago"},
            {"title": f"New Research Challenges Conventional Thinking on {topic.capitalize()}", "source": "AP", "published_at": "4 hours ago"},
        ],
    )
    return json.dumps(
        {
            "topic": topic,
            "headlines": headlines[:count],
            "note": "mock data — set NEWSAPI_API_KEY for live headlines",
        }
    )


@traceable(name="news_thinker.think._mock_category")
def _mock_category(category: str, count: int = 3) -> str:
    return json.dumps(
        {
            "category": category,
            "country": "us",
            "headlines": _MOCK_HEADLINES.get(category.lower(), _MOCK_HEADLINES.get("economy", []))[:count],
            "note": "mock data — set NEWSAPI_API_KEY for live headlines",
        }
    )


@traceable(name="news_thinker.think.get_top_headlines")
async def _get_top_headlines(topic: str, count: int = 3) -> str:
    api_key = settings.newsapi_api_key
    if not api_key:
        log.info("news_thinker.think._get_top_headlines: No News API Key")
        return _mock_headlines(topic, count)

    count = max(1, min(count, 10))
    from_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_NEWSAPI_BASE}/everything",
                params={
                    "q": topic,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "pageSize": count,
                    "language": "en",
                    "apiKey": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                return json.dumps({"error": data.get("message", "NewsAPI request failed")})

            headlines = []
            for article in data.get("articles", [])[:count]:
                headlines.append(
                    {
                        "title": article.get("title", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "published_at": article.get("publishedAt", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                    }
                )

            return json.dumps({"topic": topic, "headlines": headlines})
    except Exception:
        log.exception("news_thinker.think._get_top_headlines error")
        return _mock_headlines(topic, count)


@function_tool
async def get_top_headlines(topic: str, count: int = 3) -> str:
    """Get recent top headlines for a topic or keyword.

    Args:
        topic: The topic or search query, e.g. 'AI', 'economy', 'sports'
        count: Number of headlines to return (1-10)
    """
    return await _get_top_headlines(topic, count)


@traceable(name="news_thinker.think.get_headlines_by_category")
async def _get_headlines_by_category(category: str, country: str = "us", count: int = 3) -> str:
    api_key = settings.newsapi_api_key
    if not api_key:
        log.info("news_thinker.think._get_headlines_by_category: No News API Key")
        return _mock_category(category, count)

    count = max(1, min(count, 10))
    valid_categories = {"business", "entertainment", "general", "health", "science", "sports", "technology"}
    if category.lower() not in valid_categories:
        return json.dumps({"error": f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}"})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_NEWSAPI_BASE}/top-headlines",
                params={
                    "category": category.lower(),
                    "country": country.lower(),
                    "pageSize": count,
                    "apiKey": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                return json.dumps({"error": data.get("message", "NewsAPI request failed")})

            headlines = []
            for article in data.get("articles", [])[:count]:
                headlines.append(
                    {
                        "title": article.get("title", ""),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "published_at": article.get("publishedAt", ""),
                        "description": article.get("description", ""),
                    }
                )

            return json.dumps({"category": category, "country": country, "headlines": headlines})
    except Exception:
        log.exception("news_thinker.think._get_headlines_by_category error")
        return _mock_category(category, count)


@function_tool
async def get_headlines_by_category(category: str, country: str = "us", count: int = 3) -> str:
    """Get top headlines by news category.

    Args:
        category: One of: business, entertainment, general, health, science, sports, technology
        country: ISO 3166-1 alpha-2 country code, e.g. 'us', 'gb', 'de'
        count: Number of headlines to return (1-10)
    """
    return await _get_headlines_by_category(category, country, count)


NEWS_SYSTEM_PROMPT = """\
You are a news briefing specialist. Given a user's question about current events \
or news, use the available tools to fetch recent headlines, then provide \
a brief spoken summary of the key story or stories.

Use get_top_headlines for topic-based searches. Use get_headlines_by_category \
when the user asks about a broad category like "business news" or "sports news".

Keep your response to 3-4 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
Focus on the most relevant and recent headline.
"""


class NewsThinker(BaseThinker):
    domain = "news"
    description = "Recent headlines and current events"
    model = settings.thinker_model_advanced

    @traceable(name="news_thinker.think")
    async def think(self, query: str, context: list[dict], user_context: UserContext) -> ThinkResult:
        agent = Agent(
            name="News Specialist",
            instructions=NEWS_SYSTEM_PROMPT,
            model=self.model,
            tools=[get_top_headlines, get_headlines_by_category],
        )
        result = await Runner.run(agent, query)
        return ThinkResult(response=result.final_output)
