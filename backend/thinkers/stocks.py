"""
Stocks Thinker — handles stock price and market data queries.

Uses gpt-4.1-mini with a get_stock_price tool.
"""

import json

from openai import AsyncOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from backend.config import settings
from backend.thinkers.base import BaseThinker

client = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

STOCKS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price and basic market data for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'SPY'",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
]

STOCKS_SYSTEM_PROMPT = """\
You are a financial data specialist. Given a user's question about stocks or \
market data, use the get_stock_price tool to fetch current data, then provide \
a concise, conversational summary suitable for spoken delivery.

Keep your response to 2-3 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
Do not provide financial advice — just report the data.
"""


class StocksThinker(BaseThinker):
    domain = "stocks"
    description = "Stock prices and market data"
    model = settings.thinker_model

    @traceable(name="stocks_thinker.think")
    async def think(self, query: str, context: list[dict]) -> str:
        messages = [
            {"role": "system", "content": STOCKS_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=STOCKS_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_stock_price":
                    args = json.loads(tool_call.function.arguments)
                    # TODO: Replace with real API call (Alpha Vantage, Polygon.io, etc.)
                    stock_data = await self._mock_stock_price(args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(stock_data),
                        }
                    )

            final = await client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return final.choices[0].message.content

        return message.content or "I couldn't find stock data for that symbol."

    async def _mock_stock_price(self, args: dict) -> dict:
        """
        Mock stock data. Replace with a real API call.

        TODO: Integrate with Alpha Vantage, Polygon.io, or Yahoo Finance.
        Cache results in Redis with a 1-minute TTL.
        """
        symbol = args.get("symbol", "UNKNOWN").upper()
        mock_prices = {
            "AAPL": {"price": 189.50, "change": 1.25, "change_pct": 0.66, "volume": 52_000_000},
            "TSLA": {"price": 248.73, "change": -3.42, "change_pct": -1.36, "volume": 98_000_000},
            "SPY": {"price": 522.15, "change": 2.10, "change_pct": 0.40, "volume": 75_000_000},
            "MSFT": {"price": 415.32, "change": 0.87, "change_pct": 0.21, "volume": 21_000_000},
            "GOOGL": {"price": 176.44, "change": -0.55, "change_pct": -0.31, "volume": 18_000_000},
            "NVDA": {"price": 875.40, "change": 12.30, "change_pct": 1.43, "volume": 44_000_000},
        }
        data = mock_prices.get(
            symbol,
            {"price": 100.00, "change": 0.50, "change_pct": 0.50, "volume": 10_000_000},
        )
        return {"symbol": symbol, **data, "currency": "USD"}
