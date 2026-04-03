# Responder-Thinker

A production-grade reference implementation of the **multi-thinker Responder-Thinker pattern** for real-time voice AI, built on OpenAI's Realtime API with a Python backend.

> **"Single-thinker is a monolith. Multi-thinker is microservices. Voice AI is learning the same lessons backend engineering learned 15 years ago."**

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [The Responder-Thinker Pattern](#the-responder-thinker-pattern)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Audio Pipeline](#audio-pipeline)
  - [Thinker Routing Flow](#thinker-routing-flow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker Deployment](#docker-deployment)
- [Thinker Agents](#thinker-agents)
  - [Overview](#overview)
  - [Triggering Each Thinker](#triggering-each-thinker)
  - [Adding a New Thinker](#adding-a-new-thinker)
- [Project Structure](#project-structure)
- [User Context System](#user-context-system)
  - [Browser Fingerprinting](#browser-fingerprinting)
  - [Context Model](#context-model)
  - [Bidirectional Context Flow](#bidirectional-context-flow)
  - [Conversation Summaries](#conversation-summaries)
- [How It Works (Deep Dive)](#how-it-works-deep-dive)
  - [WebRTC Signaling](#webrtc-signaling)
  - [Realtime Bridge](#realtime-bridge)
  - [Tool Call Interception](#tool-call-interception)
  - [Stalling & Conversation Flow](#stalling--conversation-flow)
  - [Barge-In / Interruption Handling](#barge-in--interruption-handling)
  - [Idle Detection](#idle-detection)
  - [State Management (Redis)](#state-management-redis)
  - [Observability (LangSmith)](#observability-langsmith)
- [Configuration Reference](#configuration-reference)
- [Design Decisions](#design-decisions)
- [License](#license)

---

## Why This Exists

Every WebRTC voice demo connects the browser directly to OpenAI. That's fine for a demo. It's not how production voice systems work.

In production — telephony, SIP trunks, Twilio — your backend is **always** in the middle. It controls the audio pipeline, manages state, runs business logic, and orchestrates agents. This repo is the **backend-mediated architecture** that bridges that gap:

```
Browser ←—WebRTC—→ Python Backend ←—WebSocket—→ OpenAI Realtime API
                        │
                   Thinker Agents
                   (text models)
```

What you get by putting your backend in the middle:

- **Interception**: See and modify every event between user and model
- **Agent orchestration**: Tool calls route to backend agents, not browser JavaScript
- **State management**: Redis-backed conversation history, cross-session caching
- **Observability**: LangSmith traces on every Thinker call
- **Security**: API keys never touch the browser
- **Transport flexibility**: Same backend works for WebRTC browsers *and* telephony SIP trunks

---

## The Responder-Thinker Pattern

The fundamental tension in voice AI: **speed and intelligence are at odds.** OpenAI's Realtime API is fast enough for natural conversation but too limited for complex tasks. The Responder-Thinker pattern resolves this by splitting responsibilities:

### Responder (OpenAI Realtime API)
- Always on the line — never leaves the user in silence
- Handles conversation flow, greetings, acknowledgments
- Performs intent classification ("what kind of question is this?")
- Stalls naturally while Thinkers work ("Let me look that up...")
- Delivers Thinker results conversationally

### Thinkers (text-based models via OpenAI Chat Completions API)
- Specialized agents that each own a domain
- Focused system prompts — no prompt bloat
- Domain-specific tools (weather API, stock lookup, etc.)
- Can use different model tiers per domain (fast vs. smart)
- Independently testable and optimizable

### Why Multi-Thinker?

A single-thinker architecture is a monolith: one agent responsible for data lookup, FAQ resolution, complex reasoning — everything. Its system prompt grows to accommodate every domain, degrading quality across all of them. You can't optimize one domain without risking regressions in others.

Multi-thinker is microservices for voice AI:
- Each Thinker has a concise, domain-specific prompt that doesn't compete with other domains
- Simple lookups use `gpt-4.1-mini` (~100ms); complex reasoning uses `gpt-4.1`
- Per-domain caching: weather caches for 10 minutes, stocks for 1 minute
- Swap or add domains without touching existing Thinkers

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      Browser (WebRTC)                        │
│  Mic → getUserMedia() → RTCPeerConnection                    │
│  Speaker ← Audio playback ← Remote track                    │
│  Events ← Server-Sent Events (SSE) ← /api/events/:session   │
└────────────────────────┬─────────────────────────────────────┘
                         │ SDP Offer/Answer + Audio (PCM16)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Python Backend (FastAPI)                      │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  WebRTC Server (aiortc)                               │   │
│  │  - Receives browser audio (48kHz stereo)              │   │
│  │  - Resamples to 24kHz mono for Realtime API           │   │
│  │  - Sends Realtime API audio back (24kHz → 48kHz)      │   │
│  │  - Wall-clock paced output (20ms frames)              │   │
│  └───────────────────────┬───────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐   │
│  │  Realtime Bridge (core orchestration)                 │   │
│  │  - WebSocket connection to OpenAI Realtime API        │   │
│  │  - Forwards audio bidirectionally                     │   │
│  │  - Intercepts tool calls → routes to Thinkers         │   │
│  │  - Manages turn lifecycle and stale result detection   │   │
│  │  - Idle detection (15s nudge, 60s disconnect)         │   │
│  └──────┬──────────┬──────────┬──────────┬───────────────┘   │
│         ▼          ▼          ▼          ▼                    │
│  ┌──────────┐ ┌─────────┐ ┌────────┐ ┌───────────┐          │
│  │ Weather  │ │ Stocks  │ │  News  │ │ Knowledge │          │
│  │ Thinker  │ │ Thinker │ │Thinker │ │  Thinker  │          │
│  │ gpt-4.1  │ │ gpt-4.1 │ │gpt-4.1 │ │  gpt-4.1  │          │
│  │  -mini   │ │  -mini  │ │        │ │           │          │
│  │Open-Meteo│ │ Finnhub │ │NewsAPI │ │ Parametric│          │
│  └──────────┘ └─────────┘ └────────┘ └───────────┘          │
│         │          │          │          │                    │
│         └──────────┴──────────┴──────────┘                    │
│                 ContextUpdate (bidirectional)                 │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐    │
│  │  Redis                                               │    │
│  │  - Conv. history (session:{id}:conversation, 1h TTL) │    │
│  │  - Thinker cache (cache:{domain}:{hash}, per-domain) │    │
│  │  - User context  (user:{fingerprint}:context, no TTL)│    │
│  └──────────────────────────────────────────────────────-┘    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  LangSmith                                           │     │
│  │  - Session trace → Turn spans → Thinker spans        │     │
│  └──────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Audio Pipeline

```
Browser Mic (48kHz stereo)
    │
    ▼ WebRTC audio track
aiortc receives AudioFrame
    │
    ▼ aiortc_frame_to_realtime_b64()
    │  - Resample 48kHz → 24kHz via libswresample
    │  - Mix stereo → mono
    │  - Encode as base64 PCM16
    │
    ▼ input_audio_buffer.append (WebSocket)
OpenAI Realtime API processes speech
    │
    ▼ response.output_audio.delta (base64 PCM16 24kHz)
    │
    ▼ realtime_b64_to_aiortc_frame()
    │  - Decode base64 → PCM16
    │  - Resample 24kHz → 48kHz via libswresample
    │
    ▼ AudioOutputStream.push_frame()
    │  - Re-chunk into 960-sample (20ms) frames
    │  - Wall-clock paced via monotonic timer
    │
    ▼ WebRTC audio track → Browser speaker
```

### Thinker Routing Flow

```
User says: "What's the weather in Seattle?"
    │
    ▼ Realtime API transcribes speech
    ▼ Responder classifies intent
    ▼ Responder says "Let me check on that..."
    ▼ Responder calls route_to_thinker(domain="weather", query="...")
    │
    ▼ Bridge intercepts tool call
    ▼ ThinkerRouter checks Redis cache
    │    ├─ Cache hit → return cached result
    │    └─ Cache miss ↓
    ▼ WeatherThinker.think()
    │    ├─ Calls get_current_weather tool
    │    ├─ Processes result
    │    └─ Returns spoken-word response
    │
    ▼ Bridge waits for active response to finish (stall guard)
    ▼ Bridge checks turn hasn't been interrupted (stale guard)
    ▼ Bridge submits function_call_output to Realtime API
    ▼ Bridge triggers response.create
    │
    ▼ Responder delivers result conversationally
    ▼ Audio streams back to browser
```

---

## Getting Started

### Prerequisites

- **Python 3.14+** (uses latest features; 3.11+ may work with minor adjustments)
- **Redis** (local install or Docker)
- **OpenAI API key** with access to the Realtime API
- **LangSmith API key** (optional — for tracing and observability)

### Local Development

```bash
# Clone the repo
git clone https://github.com/lackmannicholas/responder-thinker.git
cd responder-thinker

# Install dependencies (using uv recommended, or pip)
pip install -e ".[dev]"

# Create your configuration
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here

# Optional: Real API data (mock data used when unset)
# FINNHUB_API_KEY=your-finnhub-key     # https://finnhub.io/register (free)
# NEWSAPI_API_KEY=your-newsapi-key     # https://newsapi.org/register (free)

# Optional: LangSmith tracing
# LANGSMITH_TRACING_ENABLED=true
# LANGSMITH_API_KEY=lsv2-your-key-here
EOF

# Start Redis
docker compose up -d redis

# Run the backend (serves both API and frontend)
uvicorn backend.main:app --reload --port 8000

# Open in your browser
open http://localhost:8000
```

The app serves the frontend at `/` and the API at `/api/*`. Click **Connect**, grant microphone access, and start talking.

### Docker Deployment

Run the entire stack (Redis + backend + Nginx frontend) with Docker Compose:

```bash
# Create .env with your API keys first (see above)

# Build and start everything
docker compose up --build

# The app is available at http://localhost
```

The Docker setup includes:
- **Redis** on port 6379 with persistent storage
- **Backend** on port 8000 with UDP ports 10000-10100 for WebRTC
- **Nginx** on port 80 as a reverse proxy — routes `/api/*` to the backend, serves static assets, and handles SSE/WebSocket upgrades

---

## Thinker Agents

### Overview

| Thinker | Domain | Model | Tools | API | Purpose |
|---------|--------|-------|-------|-----|---------|
| **Weather** | `weather` | `gpt-4.1-mini` | `get_current_weather` | [Open-Meteo](https://open-meteo.com/) (free) | Current conditions and forecasts |
| **Stocks** | `stocks` | `gpt-4.1-mini` | `get_stock_price` | [Finnhub](https://finnhub.io/) (free tier) | Stock prices and market data |
| **News** | `news` | `gpt-4.1` | `get_news_headlines` | [NewsAPI](https://newsapi.org/) (free tier) | Recent headlines and current events |
| **Knowledge** | `knowledge` | `gpt-4.1` | None (parametric) | — | General Q&A with summary grounding |
| **Research** | `research` | Mock (30s delay) | None | — | Simulates long-running tasks for stalling tests |

All Thinkers with external APIs include **mock fallbacks** — when an API key is missing or the service is unreachable, they return realistic static data. This means the system works out of the box with just `OPENAI_API_KEY`.

### Triggering Each Thinker

The Responder (Realtime API) classifies your intent and routes to the appropriate Thinker automatically. Here's how to trigger each one:

#### Weather Thinker
Ask about weather, temperature, forecasts, or conditions for any location.

> *"What's the weather like in Seattle?"*
> *"Is it going to rain in New York tomorrow?"*
> *"What's the temperature in Tokyo right now?"*

The Responder routes to `domain: "weather"` → `WeatherThinker` calls `get_current_weather(location)` → returns a spoken summary.

**Tool — `get_current_weather`**:
- **Input**: `location` (e.g., "Seattle, WA"), optional `unit` ("fahrenheit" or "celsius")
- **Output**: Temperature, feels-like, conditions, humidity, wind speed/gusts
- **Cache TTL**: 10 minutes
- **API**: [Open-Meteo](https://open-meteo.com/) — free, no API key required. Uses geocoding API for location resolution and WMO weather codes for human-readable conditions. Falls back to mock data if the API is unreachable.
- **User Context**: Respects `preferences.default_location` and `preferences.temperature_unit`. Writes queried locations back as memory facts.

#### Stocks Thinker
Ask about stock prices, market data, or specific tickers.

> *"What's Apple's stock price?"*
> *"How is Tesla doing today?"*
> *"What's the price of SPY?"*

The Responder routes to `domain: "stocks"` → `StocksThinker` calls `get_stock_price(symbol)` → returns a spoken summary.

**Tool — `get_stock_price`**:
- **Input**: `symbol` (e.g., "AAPL", "TSLA", "SPY")
- **Output**: Current price, daily change, percentage change, volume, company name
- **Cache TTL**: 1 minute
- **API**: [Finnhub](https://finnhub.io/) — free tier (60 req/min). Requires `FINNHUB_API_KEY`. Uses `/quote` for prices, `/stock/profile2` for company info, and `/search` for ticker lookup by name. Falls back to mock data with pre-defined prices for popular tickers (AAPL, TSLA, MSFT, GOOGL, NVDA, SPY) when the API key is missing or the API is unreachable.
- **User Context**: Extracts ticker symbols from queries and adds them to `preferences.watched_tickers`. Records facts like "Asked about AAPL stock".

#### News Thinker
Ask about current events, headlines, or news on any topic.

> *"What's happening in the news today?"*
> *"Any news about AI?"*
> *"What are the latest headlines in sports?"*

The Responder routes to `domain: "news"` → `NewsThinker` calls `get_news_headlines(topic)` → returns a spoken briefing.

**Tool — `get_news_headlines`**:
- **Input**: `topic` (e.g., "AI", "economy", "sports"), optional `count` (1-5)
- **Output**: Headline, source, summary for each story
- **Cache TTL**: 5 minutes
- **API**: [NewsAPI](https://newsapi.org/) — free tier (100 req/day). Requires `NEWSAPI_API_KEY`. Uses `/everything` for topic searches with a 3-day rolling window, and `/top-headlines` for category queries (business, entertainment, health, science, sports, technology). Falls back to mock data with pre-defined headlines for popular topics when the API key is missing or the API is unreachable.

#### Knowledge Thinker
Ask general knowledge questions, facts, explanations — anything that doesn't fit a specific domain. This is also the fallback when routing is ambiguous.

> *"What is quantum computing?"*
> *"Explain how photosynthesis works."*
> *"Who won the 1969 World Series?"*

The Responder routes to `domain: "knowledge"` → `KnowledgeThinker` uses `gpt-4.1` parametric knowledge + recent conversation context → returns a conversational answer.

**No external tools** — relies on the model's built-in knowledge grounded by the last 4 conversation turns from Redis. When available, the user's rolling conversation summary is injected into the system prompt for cross-session context.

#### Research Thinker
Triggers a simulated 30-second delay. Use this to test how the Responder handles long-running backend tasks.

> *"Do some research on renewable energy trends."*
> *"Research the history of spaceflight."*

The Responder routes to `domain: "research"` → `ResearchThinker` sleeps for 30 seconds → returns a mock result. **The real value here is observing how the Responder keeps the conversation alive** — it should fill time naturally with acknowledgments and small talk.

### Adding a New Thinker

1. **Create the Thinker** — add `backend/thinkers/your_domain.py`:

```python
from langsmith import traceable
from openai import AsyncOpenAI
from backend.config import settings
from backend.thinkers.base import BaseThinker
from backend.state.user_context import UserContext, ThinkResult, ContextUpdate

client = AsyncOpenAI(api_key=settings.openai_api_key)

class YourDomainThinker(BaseThinker):
    domain = "your_domain"
    description = "What this thinker does"
    model = settings.thinker_model  # or thinker_model_advanced

    @traceable(name="your_domain_thinker.think")
    async def think(
        self, query: str, context: list[dict],
        user_context: UserContext | None = None,
    ) -> ThinkResult:
        # Your domain logic here — call APIs, use tools, etc.
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Your focused system prompt..."},
                {"role": "user", "content": query},
            ],
        )
        return ThinkResult(
            response=response.choices[0].message.content,
            context_update=ContextUpdate(new_facts=["User asked about your domain"]),
        )
```

2. **Register it** — in `backend/thinkers/router.py`, import and add to `_register_thinkers()`:

```python
from backend.thinkers.your_domain import YourDomainThinker

# Inside _register_thinkers():
thinkers = [
    WeatherThinker(),
    StocksThinker(),
    NewsThinker(),
    KnowledgeThinker(),
    ResearchThinker(),
    YourDomainThinker(),  # Add here
]
```

3. **Add the routing enum** — in `backend/transport/realtime_bridge.py`, add your domain to the `ROUTE_TO_THINKER_TOOL` definition:

```python
"enum": ["weather", "stocks", "news", "knowledge", "research", "your_domain"],
```

And add a description for the Responder:

```python
"description": (
    "'your_domain' for your-domain-specific questions, "
    # ...existing domains...
),
```

---

## Project Structure

```
responder-thinker/
├── backend/
│   ├── main.py                      # FastAPI app — endpoints, lifespan, session mgmt
│   ├── config.py                    # Pydantic settings, make_openai_client()
│   ├── audio_convert.py             # PCM16 resampling (48kHz↔24kHz) via libswresample
│   ├── transport/
│   │   ├── realtime_bridge.py       # Core orchestration — bridges WebRTC ↔ Realtime API
│   │   └── webrtc_server.py         # aiortc peer connections, AudioOutputStream
│   ├── thinkers/
│   │   ├── base.py                  # BaseThinker ABC: think(query, context, user_context) → ThinkResult
│   │   ├── router.py                # ThinkerRouter — domain lookup, caching, context updates
│   │   ├── weather.py               # Weather domain — Open-Meteo API + mock fallback
│   │   ├── stocks.py                # Stocks domain — Finnhub API + mock fallback
│   │   ├── news.py                  # News domain — NewsAPI + mock fallback
│   │   ├── knowledge.py             # Knowledge domain — parametric + summary grounding
│   │   └── research.py              # Research domain — 30s delay for stalling tests
│   ├── state/
│   │   ├── session_store.py         # Redis session state, conversation history, caching
│   │   └── user_context.py          # Pydantic models — UserContext, Preferences, ContextUpdate
│   └── observability/
│       └── tracing.py               # LangSmith setup
├── frontend/
│   └── static/
│       ├── index.html               # Single-page UI — dark theme, transcript + event log
│       └── app.js                   # WebRTC client + SSE event stream
├── docker-compose.yml               # Redis + backend + Nginx (full stack)
├── Dockerfile                       # Python 3.14-slim with uv package manager
├── nginx.conf                       # Reverse proxy — API, SSE, WebSocket routing
├── pyproject.toml                   # Dependencies and project metadata
└── test_*.py                        # Integration tests (WebRTC echo, Realtime API, pipeline)
```

---

## User Context System

The system maintains persistent, cross-session memory for each user — keyed by a browser fingerprint so it "recognizes" returning users without requiring login.

### Browser Fingerprinting

When the browser connects, it generates a SHA-256 hash from:
- Canvas rendering fingerprint
- WebGL renderer string
- Platform, timezone, screen resolution
- Language and hardware concurrency

This fingerprint is sent with the SDP offer and used as the `user_id` key for all context lookups. No cookies, no accounts — the system recognizes the same browser silently.

### Context Model

All persistent user state lives in a single `UserContext` object stored in Redis with **no TTL**:

```
UserContext
├── Preferences              (overwrite semantics)
│   ├── name                 # Extracted from "My name is Nick" patterns
│   ├── default_location     # Set by weather queries ("Seattle")
│   ├── temperature_unit     # "fahrenheit" or "celsius"
│   └── watched_tickers      # Accumulated from stock queries
├── MemoryStore              (append semantics, capped at 20 facts)
│   └── facts[]              # Inferred observations: "User asked about AAPL stock"
├── Summary                  (rolling merge)
│   ├── text                 # 3-5 sentence summary of all conversations
│   ├── turn_count_at_summary
│   └── updated_at
└── Signals                  (analytics)
    ├── topic_counts         # {"weather": 5, "stocks": 3, ...}
    ├── last_active
    └── session_count        # "This is session #4 with this user"
```

### Bidirectional Context Flow

Context flows in both directions — into Thinkers and back out:

**Into Thinkers** (read path):
- Every Thinker receives the full `UserContext` alongside the query
- Weather Thinker uses `default_location` and `temperature_unit`
- Stocks Thinker checks `watched_tickers` for patterns
- Knowledge Thinker uses `summary.text` for cross-session grounding

**Out of Thinkers** (write path):
- Thinkers return a `ThinkResult` containing a `response` and an optional `ContextUpdate`
- `ContextUpdate` supports: `set_name`, `set_default_location`, `set_temperature_unit`, `add_watched_tickers`, `new_facts`
- The `ThinkerRouter` applies updates to Redis immediately
- After any context update, the Responder's system prompt is refreshed mid-session via `session.update`

**Name extraction**: The bridge also watches user transcripts for introduction patterns ("My name is Nick", "I'm Nick", "Call me Nick") and writes the name directly to `preferences.name`.

### Conversation Summaries

A rolling summary is generated to carry context across sessions:

- **On disconnect**: When the Realtime API WebSocket closes, the bridge generates a summary using `gpt-4.1-mini` from the last 30 conversation turns
- **Mid-session**: Every 10 turns, the summary is regenerated to keep it current
- **Merge, not overwrite**: The summarization prompt explicitly instructs the model to *integrate* new information with the existing summary, preserving prior context
- **Cross-session grounding**: The Knowledge Thinker injects the summary into its system prompt, so the model naturally references prior conversations

---

## How It Works (Deep Dive)

### WebRTC Signaling

The browser initiates a connection by sending an SDP offer to `POST /api/rtc/offer`. The backend:

1. Creates an `RTCPeerConnection` via `aiortc` (server-side WebRTC)
2. Attaches an `AudioOutputStream` — a synthetic track that the Realtime Bridge writes to
3. Returns the SDP answer with ICE candidates

Once ICE negotiation completes, audio flows bidirectionally over WebRTC. The browser also opens an SSE connection to `GET /api/events/{session_id}` for real-time transcript and thinker events.

**Docker note**: aiortc gathers ICE candidates using the container's internal IP, which browsers can't reach. The `webrtc_server.py` includes a monkey-patch for `aioice` that binds UDP sockets to a fixed port range and advertises `127.0.0.1` when `RTC_FORCE_HOST` and `RTC_PORT_RANGE` environment variables are set.

### Realtime Bridge

`RealtimeBridge` is the core of the system. For each session, it:

1. Opens a WebSocket to OpenAI's Realtime API (`wss://us.api.openai.com/v1/realtime`)
2. Loads persistent user context from Redis (via browser fingerprint) and enriches the system prompt
3. Configures the session: voice, audio format (24kHz PCM16), semantic VAD, tools, and personalized instructions
4. Runs four concurrent async loops:
   - **Audio input loop**: Reads WebRTC frames → resamples → forwards to Realtime API
   - **Event handler loop**: Reads Realtime API events → dispatches audio/tool calls/transcripts
   - **Idle monitor loop**: Tracks user activity → nudges at 15s → disconnects at 60s
   - **Audio drain monitor loop**: Detects when audio finishes playing → resets idle timer

When the Realtime API WebSocket closes (browser disconnect), the bridge generates a final conversation summary before tearing down.

### Tool Call Interception

When the Responder decides a question needs a specialist, it calls the `route_to_thinker` function. The bridge intercepts this:

```
Realtime API → response.function_call_arguments.done
    │
    ▼ Bridge parses {domain, query}
    ▼ Snapshots current turn_id (for stale detection)
    ▼ Dispatches thinker call concurrently (asyncio.create_task)
    │
    │  Meanwhile, the Responder is still talking ("let me check...")
    │
    ▼ Thinker returns result
    ▼ Guard 1: Is turn_id still the same? (user may have interrupted)
    ▼ Guard 2: Wait for active response to finish (can't overlap response.create)
    ▼ Guard 3: Re-check turn_id (user may have interrupted during wait)
    ▼ Submit function_call_output + trigger response.create
    │
    ▼ Responder delivers the result as natural speech
```

The three guards are critical for production reliability:
- **Stale result detection**: If the user asked a new question while the Thinker was working, the result is no longer relevant
- **Response overlap prevention**: The Realtime API silently drops `response.create` while already generating — this guard prevents the "thinker came back but nothing happened" bug
- **Post-wait staleness**: The user could interrupt during the wait for the active response to finish

### Stalling & Conversation Flow

The Responder's system prompt is engineered for natural stalling. When it calls `route_to_thinker`, the Realtime API naturally acknowledges the request *before* the tool call executes. The instructions tell it to:

- Acknowledge what the user asked
- Use natural fillers like "Let me look that up" or "One moment while I check"
- Fill time with related context it already knows
- Never leave the user in silence

The Research Thinker (30-second delay) exists specifically to stress-test this behavior.

### Barge-In / Interruption Handling

When the user starts speaking while the Responder is outputting audio:

1. `input_audio_buffer.speech_started` event fires
2. Bridge cancels the in-flight response (`response.cancel`)
3. Bridge increments `turn_id` — invalidating any in-flight thinker tasks
4. Bridge flushes the audio output queue (so the speaker stops immediately)

Any Thinker results that return after an interruption are still submitted to the API (it requires tool call responses) but won't trigger a new `response.create`.

### Idle Detection

The bridge monitors user activity:
- **15 seconds of silence**: Sends a `response.create` asking the Responder to gently check in ("Still there? Anything else I can help with?")
- **60 seconds of silence**: Sends a goodbye message, waits 5 seconds for the audio to play, then disconnects the session

### State Management (Redis)

Thinkers are stateless by design — all shared state lives in Redis:

**Conversation History** (session-scoped, ephemeral)
- Key: `session:{id}:conversation`
- Each turn stored as `{role, content, timestamp}`
- Thinkers receive the last 10 turns for context grounding
- TTL: 1 hour

**Thinker Result Cache** (shared across sessions)
- Key: `cache:{domain}:{query_hash}`
- Shared across all sessions — if two users ask the same question, the second gets a cache hit
- Per-domain TTLs:

| Domain | Cache TTL |
|--------|-----------|
| Weather | 10 minutes |
| News | 5 minutes |
| Stocks | 1 minute |
| Default | 2 minutes |

The `ThinkerRouter` checks the cache before calling any Thinker. Cache hits are logged and traced.

**User Context** (permanent, cross-session)
- Key: `user:{fingerprint}:context`
- Stores preferences, memory facts, conversation summary, and behavioral signals
- **No TTL** — persists forever so the system truly "remembers" returning users
- See [User Context System](#user-context-system) for full details

### Observability (LangSmith)

When enabled, every session produces a hierarchical trace in LangSmith:

```
voice_session (root)
├── conversation_turn
│   ├── thinker_call (tool span)
│   │   └── thinker_router.think
│   │       └── weather_thinker.think (or stocks, news, etc.)
│   └── ...
├── conversation_turn
│   └── ...
└── ...
```

Each span includes:
- `session_id` for filtering/grouping
- Input queries and conversation context
- Thinker results and timing
- Cache hit/miss indicators

Enable tracing with:
```bash
LANGSMITH_TRACING_ENABLED=true
LANGSMITH_API_KEY=lsv2-your-key-here
LANGSMITH_PROJECT=responder-thinker
```

---

## Configuration Reference

All configuration is via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `OPENAI_BASE_URL` | `https://us.api.openai.com/v1` | OpenAI API base URL (regional endpoint support) |
| `REALTIME_MODEL` | `gpt-realtime` | Model for the Responder (Realtime API) |
| `REALTIME_VOICE` | `shimmer` | Voice for audio output |
| `TRANSCRIPT_MODEL` | `gpt-4o-mini-transcribe` | Model for input audio transcription |
| `THINKER_MODEL` | `gpt-4.1-mini` | Model for fast Thinkers (Weather, Stocks) |
| `THINKER_MODEL_ADVANCED` | `gpt-4.1` | Model for complex Thinkers (News, Knowledge) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `FINNHUB_API_KEY` | *(empty)* | [Finnhub](https://finnhub.io/register) API key for live stock data. Mock data used if unset. |
| `NEWSAPI_API_KEY` | *(empty)* | [NewsAPI](https://newsapi.org/register) API key for live news. Mock data used if unset. |
| `LANGSMITH_TRACING_ENABLED` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | *(empty)* | LangSmith API key |
| `LANGSMITH_PROJECT` | `responder-thinker` | LangSmith project name |
| `RTC_FORCE_HOST` | *(unset)* | Docker only: IP to advertise in ICE candidates |
| `RTC_PORT_RANGE` | *(unset)* | Docker only: UDP port range for WebRTC (e.g., `10000-10100`) |

---

## Design Decisions

### Why backend-mediated, not direct browser-to-OpenAI?

Every tutorial shows `Browser ↔ OpenAI Realtime API`. That's a toy architecture. In production telephony (Twilio, SIP), audio always flows through your backend. Backend mediation gives you interception of every event, server-side agent orchestration, Redis-backed state, and keeps API keys off the client. The same backend works for WebRTC browsers and telephony SIP trunks.

### Why multi-thinker instead of single-thinker?

A single generalist thinker becomes a god-object: one prompt responsible for weather, stocks, news, FAQ — everything. The prompt grows, quality degrades across all domains, and you can't improve one without risking regressions in the others. Multi-thinker gives you focused prompts per domain, independent model selection, per-domain caching TTLs, and isolated testing.

### Why the Responder does intent classification

The dumbest model makes the most important decision — and that's the right architecture. Routing needs to be fast (~100ms). The Responder already has full conversational context. "What kind of question is this?" is a dramatically simpler task than "what's the answer?" Constraining routing to a fixed enum of domains makes misclassification rare and fallback trivial (unknown → Knowledge Thinker).

### Why semantic VAD instead of server VAD?

Server VAD triggers on raw silence thresholds, which produces false positives during natural pauses. Semantic VAD understands conversational turn-taking — it knows the difference between "thinking about what to say next" and "done talking." This reduces unnecessary interruptions and produces more natural conversation flow.

### Why Redis for state?

Thinkers are stateless by design — they receive a query and context and return a response. Shared state (conversation history, cached results, user context) lives outside them in Redis. This means multiple Thinkers can read the same context, results cache globally across sessions, user preferences persist across sessions with no TTL, and the architecture scales horizontally across backend instances.

### Why browser fingerprinting instead of accounts?

This is a demo/reference architecture, not a production auth system. Browser fingerprinting gives us persistent user identity with zero friction — no login, no cookies, no middleware. The same fingerprint maps to the same `UserContext` across sessions. In production, you'd swap the fingerprint for a real user ID from your auth system — the `user_id` parameter flows through the entire stack already.

### Why mock fallbacks on real APIs?

Every external API Thinker (weather, stocks, news) works without API keys. When keys are missing or the service is unreachable, Thinkers return realistic mock data. This means the system is functional out of the box with just `OPENAI_API_KEY`, which lowers the barrier to trying it. It also means Thinkers are independently testable without external dependencies.

---

## License

MIT
