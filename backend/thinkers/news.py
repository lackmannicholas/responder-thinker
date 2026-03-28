"""
News Thinker — handles news and current events queries.

Uses gpt-4.1 (advanced model) because news summarization requires
more reasoning than simple data lookups.
"""

import json

from openai import AsyncOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from backend.config import settings
from backend.thinkers.base import BaseThinker

client = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

NEWS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news_headlines",
            "description": "Get recent news headlines for a topic or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or search query, e.g. 'AI', 'economy', 'sports'",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of headlines to retrieve (1-5)",
                        "default": 3,
                    },
                },
                "required": ["topic"],
            },
        },
    },
]

NEWS_SYSTEM_PROMPT = """\
You are a news briefing specialist. Given a user's question about current events \
or news, use the get_news_headlines tool to fetch recent headlines, then provide \
a brief spoken summary of the key story or stories.

Keep your response to 3-4 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
Focus on the most relevant and recent headline.
"""


class NewsThinker(BaseThinker):
    domain = "news"
    description = "Recent headlines and current events"
    model = settings.thinker_model_advanced

    @traceable(name="news_thinker.think")
    async def think(self, query: str, context: list[dict]) -> str:
        messages = [
            {"role": "system", "content": NEWS_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=NEWS_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_news_headlines":
                    args = json.loads(tool_call.function.arguments)
                    # TODO: Replace with real API call (NewsAPI, GNews, etc.)
                    news_data = await self._mock_headlines(args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(news_data),
                        }
                    )

            final = await client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return final.choices[0].message.content

        return message.content or "I couldn't find recent news on that topic."

    async def _mock_headlines(self, args: dict) -> dict:
        """
        Mock news data. Replace with a real API call.

        TODO: Integrate with NewsAPI, GNews, or similar.
        Cache results in Redis with a 5-minute TTL.
        """
        topic = args.get("topic", "general")
        count = min(args.get("count", 3), 5)
        return {
            "topic": topic,
            "headlines": [
                {
                    "title": f"Breaking: Major development in {topic} sector",
                    "source": "Reuters",
                    "published": "2 hours ago",
                    "summary": (f"Experts are closely watching developments in {topic} " "as new information emerges."),
                },
                {
                    "title": f"Analysis: What recent {topic} trends mean for the future",
                    "source": "AP News",
                    "published": "4 hours ago",
                    "summary": (f"Analysts weigh in on the implications of the latest " f"{topic} developments."),
                },
                {
                    "title": f"{topic.capitalize()} update: Key stakeholders respond",
                    "source": "BBC",
                    "published": "6 hours ago",
                    "summary": (f"Reactions continue to pour in following the latest " f"news on {topic}."),
                },
            ][:count],
        }
