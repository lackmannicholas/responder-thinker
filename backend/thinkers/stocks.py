"""
Stocks Thinker — handles stock price and market data queries.

Uses the OpenAI Agents SDK with local function tools that call the Finnhub API.
Requires FINNHUB_API_KEY env var (free tier: 60 req/min, https://finnhub.io/register).
"""

import json

import httpx
import structlog
from agents import Agent, Runner, function_tool
from langsmith import traceable

from backend.config import settings
from backend.thinkers.base import BaseThinker
from backend.state.user_context import ThinkResult, UserContext, ContextUpdate

log = structlog.get_logger()

_FINNHUB_BASE = "https://finnhub.io/api/v1"


_MOCK_STOCKS = {
    "AAPL": {"name": "Apple Inc", "price": 189.50, "change": 1.25, "pct": 0.66},
    "TSLA": {"name": "Tesla Inc", "price": 248.73, "change": -3.42, "pct": -1.36},
    "MSFT": {"name": "Microsoft Corp", "price": 415.32, "change": 0.87, "pct": 0.21},
    "GOOGL": {"name": "Alphabet Inc", "price": 176.44, "change": -0.55, "pct": -0.31},
    "NVDA": {"name": "NVIDIA Corp", "price": 875.40, "change": 12.30, "pct": 1.43},
    "SPY": {"name": "SPDR S&P 500 ETF", "price": 522.15, "change": 2.10, "pct": 0.40},
}


@traceable(name="stocks_thinker.think._mock_quote")
def _mock_quote(symbol: str) -> str:
    symbol = symbol.upper().strip()
    m = _MOCK_STOCKS.get(symbol, {"name": symbol, "price": 100.00, "change": 0.50, "pct": 0.50})
    return json.dumps(
        {
            "symbol": symbol,
            "name": m["name"],
            "current_price": m["price"],
            "change": m["change"],
            "change_percent": m["pct"],
            "currency": "USD",
            "note": "mock data — set FINNHUB_API_KEY for live quotes",
        }
    )


@traceable(name="stocks_thinker.think.get_stock_quote")
async def _get_stock_quote(symbol: str) -> str:
    token = settings.finnhub_api_key
    if not token:
        log.info("stocks_thinker.think.get_stock_quote: No Finnhub API key")
        return _mock_quote(symbol)

    symbol = symbol.upper().strip()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            quote_resp = await client.get(
                f"{_FINNHUB_BASE}/quote",
                params={"symbol": symbol, "token": token},
            )
            quote_resp.raise_for_status()
            q = quote_resp.json()

            if q.get("c", 0) == 0 and q.get("pc", 0) == 0:
                return json.dumps({"error": f"No data found for symbol: {symbol}"})

            profile_resp = await client.get(
                f"{_FINNHUB_BASE}/stock/profile2",
                params={"symbol": symbol, "token": token},
            )
            profile_resp.raise_for_status()
            profile = profile_resp.json()

            return json.dumps(
                {
                    "symbol": symbol,
                    "name": profile.get("name", symbol),
                    "current_price": q["c"],
                    "change": round(q["d"], 2) if q["d"] else 0,
                    "change_percent": round(q["dp"], 2) if q["dp"] else 0,
                    "high": q["h"],
                    "low": q["l"],
                    "open": q["o"],
                    "previous_close": q["pc"],
                    "currency": profile.get("currency", "USD"),
                    "exchange": profile.get("exchange", ""),
                }
            )
    except Exception:
        log.exception("stocks_thinker.think.get_stock_quote error")
        return _mock_quote(symbol)


@function_tool
async def get_stock_quote(symbol: str) -> str:
    """Get the current price, daily change, and volume for a stock ticker.

    Args:
        symbol: Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT'
    """
    return await _get_stock_quote(symbol)


def _mock_search(query: str) -> str:
    q = query.lower()
    results = [s for sym, s in _MOCK_STOCKS.items() if q in s["name"].lower() or q in sym.lower()]
    return json.dumps(
        {
            "query": query,
            "results": [{"symbol": k, "description": v["name"]} for k, v in _MOCK_STOCKS.items() if q in v["name"].lower() or q in k.lower()][:5],
            "note": "mock data — set FINNHUB_API_KEY for live search",
        }
    )


@traceable(name="stocks_thinker.think.search_stock_symbol")
async def _search_stock_symbol(query: str) -> str:
    token = settings.finnhub_api_key
    if not token:
        log.info("stocks_thinker.think.search_stock_symbol: No Finnhub API key")
        return _mock_search(query)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_FINNHUB_BASE}/search",
                params={"q": query, "token": token},
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("result", [])[:5]:
                results.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "description": item.get("description", ""),
                        "type": item.get("type", ""),
                    }
                )

            return json.dumps({"query": query, "results": results})
    except Exception:
        log.exception("stocks_thinker.think.search_stock_symbol error")
        return _mock_search(query)


@function_tool
async def search_stock_symbol(query: str) -> str:
    """Search for a stock ticker symbol by company name.

    Args:
        query: Company name or partial name, e.g. 'Apple', 'Tesla'
    """
    return await _search_stock_symbol(query)


STOCKS_SYSTEM_PROMPT = """\
You are a financial data specialist. Given a user's question about stocks or \
market data, use the available tools to fetch current data, then provide \
a concise, conversational summary suitable for spoken delivery.

If the user mentions a company name instead of a ticker, use search_stock_symbol \
first to find the correct ticker, then use get_stock_quote.

Keep your response to 2-3 sentences max. This will be read aloud by a voice agent.
Do NOT use bullet points, lists, or formatting. Just natural speech.
Do not provide financial advice — just report the data.
"""


class StocksThinker(BaseThinker):
    domain = "stocks"
    description = "Stock prices and market data"
    model = settings.thinker_model

    @traceable(name="stocks_thinker.think")
    async def think(self, query: str, context: list[dict], user_context: UserContext) -> ThinkResult:
        effective_query = query
        prefs = user_context.preferences

        if prefs.watched_tickers:
            effective_query = f"{query}\n\nContext: The user's watched tickers are " f"{', '.join(prefs.watched_tickers)}."

        agent = Agent(
            name="Stocks Specialist",
            instructions=STOCKS_SYSTEM_PROMPT,
            model=self.model,
            tools=[get_stock_quote, search_stock_symbol],
        )
        result = await Runner.run(agent, effective_query)
        response = result.final_output

        # Extract ticker symbols mentioned in the query to add to watched list
        import re

        update = ContextUpdate()
        tickers_in_query = re.findall(r'\b[A-Z]{1,5}\b', query)
        known_tickers = {t for t in tickers_in_query if t in _MOCK_STOCKS or len(t) >= 2}
        for ticker in known_tickers:
            update.add_watched_tickers.append(ticker)
            update.new_facts.append(f"Asked about {ticker} stock")

        return ThinkResult(
            response=response,
            context_update=update if not update.is_empty() else None,
        )
