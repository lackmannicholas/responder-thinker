"""
News Thinker — handles news and current events queries.

Uses gpt-4.1 (advanced model) because news summarization requires
more reasoning than simple data lookups.
"""

import json

from langsmith import traceable

from backend.config import settings, make_openai_client
from backend.thinkers.base import BaseThinker

client = make_openai_client()

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

        # Pre-built mock headlines by topic for a more convincing demo.
        # Generic fallback for unknown topics.
        topic_headlines = {
            "ai": [
                {
                    "title": "OpenAI Releases New Real-Time Voice Model With Tool-Use Support",
                    "source": "Reuters",
                    "published": "2 hours ago",
                    "summary": "The latest model supports function calling during live audio sessions, enabling more complex voice agent workflows.",
                },
                {
                    "title": "EU AI Act Enforcement Begins With First Round of Compliance Audits",
                    "source": "Financial Times",
                    "published": "5 hours ago",
                    "summary": "European regulators have started auditing major AI providers under the new framework.",
                },
                {
                    "title": "Google DeepMind Publishes Breakthrough in Long-Context Reasoning",
                    "source": "Ars Technica",
                    "published": "8 hours ago",
                    "summary": "A new architecture enables reliable reasoning across million-token context windows.",
                },
            ],
            "economy": [
                {
                    "title": "Federal Reserve Holds Rates Steady, Signals Potential Cut in September",
                    "source": "Wall Street Journal",
                    "published": "1 hour ago",
                    "summary": "The Fed kept the benchmark rate unchanged but indicated easing could begin later this year.",
                },
                {
                    "title": "US Jobs Report Shows Stronger-Than-Expected Hiring in Services Sector",
                    "source": "Bloomberg",
                    "published": "4 hours ago",
                    "summary": "Nonfarm payrolls beat estimates with particular strength in healthcare and hospitality.",
                },
                {
                    "title": "Housing Starts Rise for Third Consecutive Month Amid Easing Lumber Costs",
                    "source": "CNBC",
                    "published": "6 hours ago",
                    "summary": "Builders are responding to lower material costs, though affordability remains strained for buyers.",
                },
            ],
            "sports": [
                {
                    "title": "NBA Finals Game 5: Celtics Take 3-2 Series Lead in Overtime Thriller",
                    "source": "ESPN",
                    "published": "3 hours ago",
                    "summary": "A buzzer-beating three-pointer sent the game to overtime where Boston pulled away late.",
                },
                {
                    "title": "FIFA Announces Expanded Club World Cup Format for 2027",
                    "source": "BBC Sport",
                    "published": "5 hours ago",
                    "summary": "The tournament will feature 48 clubs across six confederations in the new format.",
                },
                {
                    "title": "MLB Trade Deadline: Yankees Acquire All-Star Pitcher in Blockbuster Deal",
                    "source": "The Athletic",
                    "published": "7 hours ago",
                    "summary": "New York sent a package of top prospects to secure a front-line starter for the playoff push.",
                },
            ],
        }

        # Normalize topic for lookup, fall back to generic headlines
        normalized = topic.lower().strip()
        headlines = topic_headlines.get(normalized, [
            {
                "title": f"Major Policy Shift Expected in {topic.capitalize()} Sector This Quarter",
                "source": "Reuters",
                "published": "2 hours ago",
                "summary": f"Industry leaders are responding to new regulatory signals affecting {topic}.",
            },
            {
                "title": f"New Research Challenges Conventional Thinking on {topic.capitalize()}",
                "source": "Associated Press",
                "published": "4 hours ago",
                "summary": f"A peer-reviewed study offers findings that could reshape how experts approach {topic}.",
            },
            {
                "title": f"Global Summit on {topic.capitalize()} Draws Record Attendance",
                "source": "BBC",
                "published": "6 hours ago",
                "summary": f"Delegates from over 40 countries gathered to discuss the future of {topic}.",
            },
        ])

        return {"topic": topic, "headlines": headlines[:count]}
