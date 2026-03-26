"""
Weather Thinker — handles weather queries.

Uses OpenAI function calling to fetch weather data.
Demonstrates how a Thinker agent has its own focused prompt and tools,
independent of the Responder and other Thinkers.
"""

import json

from openai import AsyncOpenAI
from langsmith import traceable

from backend.config import settings
from backend.thinkers.base import BaseThinker

client = AsyncOpenAI(api_key=settings.openai_api_key)

# Thinker-specific tools
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g. 'Seattle, WA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "default": "fahrenheit",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

WEATHER_SYSTEM_PROMPT = """\
You are a weather specialist. Given a user's question about weather, \
use the get_current_weather tool to fetch data, then provide a concise, \
conversational summary suitable for spoken delivery.

Keep your response to 2-3 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
"""


class WeatherThinker(BaseThinker):
    domain = "weather"
    description = "Current weather conditions and forecasts"
    model = settings.thinker_model

    @traceable(name="weather_thinker.think")
    async def think(self, query: str, context: list[dict]) -> str:
        messages = [
            {"role": "system", "content": WEATHER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=WEATHER_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_current_weather":
                    args = json.loads(tool_call.function.arguments)
                    # TODO: Replace with real weather API call (OpenWeatherMap, etc.)
                    weather_data = await self._mock_weather(args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(weather_data),
                        }
                    )

            final = await client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return final.choices[0].message.content

        return message.content or "I couldn't find weather information for that location."

    async def _mock_weather(self, args: dict) -> dict:
        """
        Mock weather data. Replace with a real API call.

        TODO: Integrate with OpenWeatherMap, WeatherAPI, or similar.
        Cache results in Redis with a 10-minute TTL.
        """
        location = args.get("location", "Unknown")
        return {
            "location": location,
            "temperature": 72,
            "unit": args.get("unit", "fahrenheit"),
            "conditions": "partly cloudy",
            "humidity": 45,
            "wind_speed": 8,
            "wind_direction": "NW",
        }
