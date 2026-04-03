"""
Responder-Thinker: Multi-agent real-time voice AI.

FastAPI application that bridges:
  Browser (WebRTC) <-> Backend (Python) <-> OpenAI Realtime API (WebSocket)

The backend has full control over the audio stream and intercepts
tool calls to route them to specialized Thinker agents.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from backend.config import settings, make_openai_client
from backend.transport.webrtc_server import WebRTCServer
from backend.transport.realtime_bridge import RealtimeBridge
from backend.thinkers.router import ThinkerRouter
from backend.state.session_store import SessionStore
from backend.observability.tracing import setup_tracing

log = structlog.get_logger()

# Initialize shared components (one instance per process)
webrtc_server = WebRTCServer()
session_store = SessionStore(settings.redis_url)
thinker_router = ThinkerRouter(session_store=session_store)

# Active bridge instances keyed by session_id (for SSE access)
_bridges: dict[str, RealtimeBridge] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_tracing()

    # Set the Agents SDK's default OpenAI client so all thinkers
    # use the correct regional endpoint and LangSmith wrapping.
    from agents import set_default_openai_client

    set_default_openai_client(make_openai_client())

    await session_store.connect()
    log.info("responder_thinker.started", model=settings.realtime_model)
    yield
    await session_store.disconnect()


app = FastAPI(title="Responder-Thinker", version="0.1.0", lifespan=lifespan)

# Serve the frontend static assets (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/static/index.html")


@app.post("/api/rtc/offer")
async def rtc_offer(request: dict):
    """
    WebRTC signaling endpoint.
    Browser sends SDP offer + browser fingerprint, backend returns SDP answer.
    Once connected, audio flows: Browser <--WebRTC--> Backend.
    """
    session_id = str(uuid.uuid4())
    offer_sdp = request.get("sdp")
    fingerprint = request.get("fingerprint")  # Browser fingerprint as user_id

    log.info("rtc.offer_received", session_id=session_id, has_fingerprint=bool(fingerprint))

    # Create the WebRTC peer connection and get the answer SDP
    answer_sdp, session_tracks = await webrtc_server.create_session(
        session_id=session_id,
        offer_sdp=offer_sdp,
    )

    # Spin up the Realtime API bridge for this session
    bridge = RealtimeBridge(
        session_id=session_id,
        audio_track=session_tracks,
        thinker_router=thinker_router,
        session_store=session_store,
        user_id=fingerprint,
    )
    _bridges[session_id] = bridge

    async def _run_and_cleanup():
        try:
            await bridge.run()
        finally:
            _bridges.pop(session_id, None)
            await webrtc_server.close_session(session_id)

    asyncio.create_task(_run_and_cleanup())

    return {"sdp": answer_sdp, "session_id": session_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket fallback for environments where WebRTC isn't available.
    Same architecture, different transport on the browser side.
    """
    await websocket.accept()
    log.info("ws.connected", session_id=session_id)

    bridge = RealtimeBridge(
        session_id=session_id,
        websocket=websocket,
        thinker_router=thinker_router,
        session_store=session_store,
    )

    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("ws.disconnected", session_id=session_id)
    finally:
        await bridge.cleanup()


@app.get("/api/events/{session_id}")
async def session_events(session_id: str, request: Request):
    """
    SSE endpoint for streaming transcript and thinker events to the frontend.
    The browser connects after WebRTC is established.
    """
    bridge = _bridges.get(session_id)
    if not bridge:
        return StreamingResponse(
            iter(["data: {\"type\": \"error\", \"message\": \"session not found\"}\n\n"]),
            media_type="text/event-stream",
        )

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(bridge.event_queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    # Session ended
                    yield f"data: {json.dumps({'type': 'session_ended'})}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
