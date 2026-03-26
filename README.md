# responder-thinker

A production-grade reference implementation of the **multi-thinker Responder-Thinker pattern** for real-time voice AI, built on OpenAI's Realtime API with a Python backend.

## Why This Exists

Every WebRTC voice demo connects the browser directly to OpenAI. That's fine for a demo. It's not how production voice systems work.

In production — telephony, SIP trunks, Twilio — your backend is always in the middle. It controls the audio pipeline, manages state, runs business logic, and orchestrates agents. This repo is the **backend-mediated architecture** that nobody has documented:

```
Browser ←—WebRTC—→ Python Backend ←—WebSocket—→ OpenAI Realtime API
                        │
                   Thinker Agents
                   (text models)
```

## The Pattern

**Responder** (OpenAI Realtime API): Always on the line. Handles conversation flow, intent classification, and natural stalling while Thinkers work.

**Thinkers** (text-based models via OpenAI API): Specialized agents that handle specific domains — weather, stocks, news, knowledge. Each has a focused prompt, its own tools, and can be optimized independently.

**The insight**: Single-thinker is a monolith — one agent responsible for everything, with a bloated prompt that degrades across all domains. Multi-thinker is microservices — each agent owns a domain, scales independently, and can be swapped without touching the others.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Browser (WebRTC)                       │
│  Mic → getUserMedia() → RTCPeerConnection                │
│  Speaker ← Audio playback ← Remote track                │
└────────────────────────┬─────────────────────────────────┘
                         │ SDP Offer/Answer
                         │ Audio tracks (PCM16)
                         ▼
┌──────────────────────────────────────────────────────────┐
│               Python Backend (FastAPI)                    │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  WebRTC Server (aiortc)                             │ │
│  │  - Receives browser audio                           │ │
│  │  - Sends Realtime API audio back                    │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │  Realtime Bridge                                    │ │
│  │  - WebSocket connection to OpenAI Realtime API      │ │
│  │  - Forwards audio bidirectionally                   │ │
│  │  - Intercepts tool calls → routes to Thinkers       │ │
│  │  - Injects Thinker results back into conversation   │ │
│  └──────┬─────────┬──────────┬──────────┬──────────────┘ │
│         ▼         ▼          ▼          ▼                │
│  ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐       │
│  │ Weather  │ │ Stocks │ │  News  │ │Knowledge │       │
│  │ Thinker  │ │Thinker │ │Thinker │ │ Thinker  │       │
│  └──────────┘ └────────┘ └────────┘ └──────────┘       │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Redis           │  │  LangSmith      │               │
│  │  - Conv history  │  │  - Traces       │               │
│  │  - Tool cache    │  │  - Thinker perf │               │
│  │  - Session state │  │  - Routing viz  │               │
│  └─────────────────┘  └─────────────────┘               │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (local or Docker)
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/responder-thinker.git
cd responder-thinker

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Start Redis
docker compose up -d redis

# Run
uvicorn backend.main:app --reload --port 8000

# Open browser
open http://localhost:8000
```

## Thinker Agents

| Thinker | Domain | Model | Description |
|---------|--------|-------|-------------|
| Weather | `weather` | gpt-4.1-mini | Current conditions and forecasts |
| Stocks | `stocks` | gpt-4.1-mini | Stock prices and market data |
| News | `news` | gpt-4.1 | Recent headlines and current events |
| Knowledge | `knowledge` | gpt-4.1 | General Q&A (fallback) |

### Adding a New Thinker

1. Create `backend/thinkers/your_domain.py` extending `BaseThinker`
2. Implement the `think()` method with your domain-specific logic
3. Register it in `backend/thinkers/router.py`
4. Add the domain to the `route_to_thinker` tool enum in `realtime_bridge.py`

## Key Design Decisions

### Why backend-mediated, not direct browser-to-OpenAI?

- **Full control**: Intercept every event, modify the audio pipeline, run server-side logic
- **Production-ready**: Mirrors how telephony (Twilio, SIP) actually works
- **Thinker orchestration**: Tool calls route to your backend agents, not browser JS
- **State management**: Redis-backed conversation history, caching, session management
- **Observability**: LangSmith traces for every Thinker call

### Why multi-thinker instead of single-thinker?

- **Focused prompts**: Each Thinker has a concise, domain-specific system prompt
- **Independent optimization**: Tune weather accuracy without risking stock data quality
- **Latency control**: Simple FAQ lookups use `gpt-4.1-mini`; complex reasoning uses `gpt-4.1`
- **Caching**: Weather data caches for 10 minutes; stock data for 1 minute. Per-domain TTLs.
- **Testability**: Test each Thinker in isolation with domain-specific evals

### Why the Responder does intent classification

The dumbest model makes the most important decision — and that's the right architecture. Routing needs to be fast (~100ms), the Responder already has full conversational context, and "what kind of question is this?" is a simpler task than "what's the answer?"

## Blog Post

This repo accompanies the blog post: **"The Responder-Thinker Pattern: Why Single-Thinker Breaks and How to Build Multi-Thinker"** — [link TBD]

## License

MIT
