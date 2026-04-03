"""
Weather Thinker — handles weather queries.

Uses the OpenAI Agents SDK with a local function tool that calls the Open-Meteo API
(free, no API key required).
"""

import json

import httpx
from agents import Agent, Runner, function_tool
from langsmith import traceable

from backend.config import settings
from backend.thinkers.base import BaseThinker
from backend.state.user_context import ThinkResult, UserContext, ContextUpdate

# WMO Weather interpretation codes → human-readable descriptions
# https://open-meteo.com/en/docs
_WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _mock_weather(location: str, unit: str = "fahrenheit") -> str:
    return json.dumps(
        {
            "location": location,
            "temperature": 72 if unit == "fahrenheit" else 22,
            "feels_like": 70 if unit == "fahrenheit" else 21,
            "unit": unit,
            "conditions": "partly cloudy",
            "humidity_percent": 45,
            "wind_speed_mph": 8,
            "wind_gusts_mph": 14,
            "note": "mock data — Open-Meteo API unavailable",
        }
    )


@function_tool
async def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get current weather conditions for a location.

    Args:
        location: City and optional state/country, e.g. 'Seattle, WA' or 'London, UK'
        unit: Temperature unit — 'fahrenheit' or 'celsius'
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Geocode location → lat/lon
            geo_resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location.split(",")[0].strip(), "count": 1, "language": "en"},
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return json.dumps({"error": f"Could not find location: {location}"})

            place = geo_data["results"][0]
            lat, lon = place["latitude"], place["longitude"]
            resolved_name = place.get("name", location)
            admin = place.get("admin1", "")
            country = place.get("country", "")

            # Fetch current weather
            temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
            weather_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": ("temperature_2m,relative_humidity_2m,apparent_temperature," "wind_speed_10m,wind_gusts_10m,weather_code"),
                    "temperature_unit": temp_unit,
                    "wind_speed_unit": "mph",
                },
            )
            weather_resp.raise_for_status()
            current = weather_resp.json()["current"]

            display_location = ", ".join(filter(None, [resolved_name, admin, country]))

            return json.dumps(
                {
                    "location": display_location,
                    "temperature": current["temperature_2m"],
                    "feels_like": current["apparent_temperature"],
                    "unit": unit,
                    "conditions": _WMO_CODES.get(current["weather_code"], "unknown"),
                    "humidity_percent": current["relative_humidity_2m"],
                    "wind_speed_mph": current["wind_speed_10m"],
                    "wind_gusts_mph": current["wind_gusts_10m"],
                }
            )
    except Exception:
        return _mock_weather(location, unit)


WEATHER_SYSTEM_PROMPT = """\
You are a weather specialist. Given a user's question about weather, \
use the get_current_weather tool to fetch live data, then provide a concise, \
conversational summary suitable for spoken delivery.

Keep your response to 2-3 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
"""


class WeatherThinker(BaseThinker):
    domain = "weather"
    description = "Current weather conditions and forecasts"
    model = settings.thinker_model

    @traceable(name="weather_thinker.think")
    async def think(self, query: str, context: list[dict], user_context: UserContext) -> ThinkResult:
        # Apply user preferences: default location and temperature unit
        effective_query = query
        prefs = user_context.preferences
        hints: list[str] = []

        if prefs.default_location:
            hints.append(f"The user's default location is {prefs.default_location}.")
        if prefs.temperature_unit:
            hints.append(f"The user prefers {prefs.temperature_unit} temperatures.")

        if hints:
            effective_query = f"{query}\n\nUser preferences: {' '.join(hints)}"

        agent = Agent(
            name="Weather Specialist",
            instructions=WEATHER_SYSTEM_PROMPT,
            model=self.model,
            tools=[get_current_weather],
        )
        result = await Runner.run(agent, effective_query)
        response = result.final_output

        # Extract location from the query for context updates
        update = ContextUpdate()
        # If the user explicitly named a location, persist it as their default
        query_lower = query.lower()
        # Simple heuristic: if query contains "in <location>" pattern, extract it
        if " in " in query_lower:
            location_part = query.split(" in ", 1)[-1].strip().rstrip("?.,!")
            if location_part and len(location_part) < 60:
                update.set_default_location = location_part
                update.new_facts.append(f"Asked about weather in {location_part}")

        return ThinkResult(
            response=response,
            context_update=update if not update.is_empty() else None,
        )
