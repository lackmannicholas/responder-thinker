"""
RealtimeBridge: The core orchestration layer.

Connects to OpenAI's Realtime API via WebSocket and bridges audio
from the browser's WebRTC connection. This is where the Responder-Thinker
pattern actually lives:

  1. Audio flows in from the browser → forwarded to OpenAI Realtime API
  2. OpenAI Realtime API sends back events (audio, tool calls, transcripts)
  3. Audio responses → forwarded back to the browser
  4. Tool calls for `route_to_thinker` → intercepted and routed to Thinker agents
  5. Thinker results → injected back into the Realtime API conversation

The Responder (Realtime API) never stops — it's always present on the call.
Thinker calls happen concurrently in the background.
"""

import asyncio
import base64
import json
import time

import structlog
import websockets

from backend.config import settings
from backend.thinkers.router import ThinkerRouter
from backend.state.session_store import SessionStore

log = structlog.get_logger()

# OpenAI Realtime API WebSocket endpoint
REALTIME_API_URL = "wss://api.openai.com/v1/realtime"

# The tool definition that the Responder uses to route to Thinkers
ROUTE_TO_THINKER_TOOL = {
    "type": "function",
    "name": "route_to_thinker",
    "description": (
        "Route a user question to a specialized backend agent for processing. "
        "Use this whenever the user asks something that requires data lookup, "
        "real-time information, or complex reasoning. The Thinker will process "
        "the request and return a response for you to deliver to the user. "
        "While waiting, keep the conversation natural — acknowledge the request, "
        'say something like "let me check on that" or "one moment".'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["weather", "stocks", "news", "knowledge"],
                "description": (
                    "Which specialist to route to: "
                    "'weather' for current conditions and forecasts, "
                    "'stocks' for stock prices and market data, "
                    "'news' for recent headlines and current events, "
                    "'knowledge' for general questions and facts."
                ),
            },
            "query": {
                "type": "string",
                "description": "The user's question, rephrased clearly for the specialist.",
            },
        },
        "required": ["domain", "query"],
    },
}

# Responder system instructions
RESPONDER_INSTRUCTIONS = """\
You are a friendly, conversational voice assistant. You are the Responder — \
your job is to keep the conversation flowing naturally while specialized \
backend agents (Thinkers) handle complex questions.

## Your role:
- Greet users warmly and maintain natural conversation
- When the user asks something that needs real data (weather, stocks, news, \
  or complex knowledge), call the `route_to_thinker` tool with the right domain
- While waiting for the Thinker, keep talking naturally. Acknowledge what they \
  asked. Say "let me look that up" or "one moment while I check." Don't go silent.
- When the Thinker result arrives, deliver it conversationally — don't read it \
  like a report. Weave it into the conversation.

## Routing guide:
- Weather questions → domain: "weather"
- Stock prices, market data → domain: "stocks"
- News, current events, headlines → domain: "news"
- General knowledge, facts, explanations → domain: "knowledge"

## Important:
- You are ALWAYS present. Never leave the user in silence.
- Keep responses concise — this is voice, not text.
- If you're unsure which domain, use "knowledge" as the default.
- If the Thinker takes a while, it's okay to fill time with small talk or \
  related context you already know.
"""


class RealtimeBridge:
    """
    Bridges browser audio (WebRTC or WebSocket) to OpenAI Realtime API.

    Intercepts tool calls to route them to Thinker agents.
    """

    def __init__(
        self,
        session_id: str,
        thinker_router: ThinkerRouter,
        session_store: SessionStore,
        audio_track=None,  # WebRTC session tracks
        websocket=None,    # WebSocket fallback
    ):
        self.session_id = session_id
        self.thinker_router = thinker_router
        self.session_store = session_store
        self.audio_track = audio_track
        self.websocket = websocket
        self._realtime_ws = None
        self._running = False

    async def run(self):
        """Main loop: connect to Realtime API and start bidirectional streaming."""
        self._running = True

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{REALTIME_API_URL}?model={settings.realtime_model}"

        log.info("bridge.connecting", session_id=self.session_id)

        async with websockets.connect(url, extra_headers=headers) as ws:
            self._realtime_ws = ws

            # Configure the session
            await self._configure_session()

            # Run three concurrent loops:
            #   1. Read audio from browser → send to OpenAI
            #   2. Read events from OpenAI → handle them
            #   3. (Future) Periodic state sync
            await asyncio.gather(
                self._audio_input_loop(),
                self._event_handler_loop(),
                return_exceptions=True,
            )

    async def _configure_session(self):
        """Send initial session configuration to the Realtime API."""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": RESPONDER_INSTRUCTIONS,
                "voice": settings.realtime_voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    # TODO: Consider using your local VAD here instead.
                    # Cross-reference: your vad-rs project showed 500ms+ improvement.
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": [ROUTE_TO_THINKER_TOOL],
            },
        }
        await self._realtime_ws.send(json.dumps(config))
        log.info("bridge.session_configured", session_id=self.session_id)

    async def _audio_input_loop(self):
        """
        Read audio frames from the browser (WebRTC track) and forward
        to the OpenAI Realtime API as input_audio_buffer.append events.

        This is the hot path — every frame matters for latency.
        """
        if self.audio_track is None:
            log.warning("bridge.no_audio_track", session_id=self.session_id)
            return

        # Wait for the input track to be available (WebRTC negotiation)
        tracks = self.audio_track
        while tracks.input_track is None and self._running:
            await asyncio.sleep(0.01)

        input_track = tracks.input_track
        log.info("bridge.audio_input_started", session_id=self.session_id)

        try:
            while self._running:
                frame = await input_track.recv()

                # Convert aiortc AudioFrame to raw PCM16 bytes
                # aiortc delivers audio as AudioFrame with .to_ndarray()
                pcm_data = frame.to_ndarray().tobytes()
                audio_b64 = base64.b64encode(pcm_data).decode("utf-8")

                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }
                await self._realtime_ws.send(json.dumps(event))
        except Exception as e:
            log.error("bridge.audio_input_error", error=str(e))

    async def _event_handler_loop(self):
        """
        Read events from the OpenAI Realtime API and handle them.

        This is where the Responder-Thinker interception happens:
        tool calls for `route_to_thinker` get routed to Thinker agents
        instead of being passed back to the browser.
        """
        log.info("bridge.event_loop_started", session_id=self.session_id)

        try:
            async for message in self._realtime_ws:
                event = json.loads(message)
                event_type = event.get("type", "")

                match event_type:
                    # --- Audio output: send back to browser ---
                    case "response.audio.delta":
                        await self._handle_audio_output(event)

                    # --- Tool calls: intercept for Thinker routing ---
                    case "response.function_call_arguments.done":
                        await self._handle_tool_call(event)

                    # --- Transcription: store for context ---
                    case "conversation.item.input_audio_transcription.completed":
                        await self._handle_transcription(event, role="user")

                    case "response.audio_transcript.done":
                        await self._handle_transcription(event, role="assistant")

                    # --- Session lifecycle ---
                    case "session.created":
                        log.info("bridge.session_ready", session_id=self.session_id)

                    case "error":
                        log.error(
                            "bridge.realtime_error",
                            session_id=self.session_id,
                            error=event.get("error"),
                        )

                    # --- Everything else: log at debug level ---
                    case _:
                        log.debug("bridge.event", type=event_type)

        except websockets.exceptions.ConnectionClosed:
            log.info("bridge.realtime_disconnected", session_id=self.session_id)

    async def _handle_audio_output(self, event: dict):
        """
        Forward audio from Realtime API back to the browser.

        The audio comes as base64-encoded PCM16. We decode it and push
        it to the WebRTC output track.
        """
        audio_b64 = event.get("delta", "")
        if not audio_b64:
            return

        audio_bytes = base64.b64decode(audio_b64)

        if self.audio_track:
            # TODO: Convert raw PCM16 bytes back to an aiortc AudioFrame
            # and push to self.audio_track.output_track.push_frame(frame)
            #
            # This requires building an av.AudioFrame from the raw bytes.
            # Sample rate: 24000 (Realtime API default)
            # Channels: 1 (mono)
            # Format: s16 (signed 16-bit PCM)
            pass

    async def _handle_tool_call(self, event: dict):
        """
        Intercept tool calls from the Responder.

        If the tool is `route_to_thinker`, we:
          1. Parse the domain and query
          2. Route to the appropriate Thinker agent
          3. Return the Thinker's response to the Realtime API
          4. Trigger the Responder to deliver the answer

        This is the heart of the Responder-Thinker pattern.
        """
        call_id = event.get("call_id", "")
        name = event.get("name", "")
        arguments = event.get("arguments", "{}")

        if name != "route_to_thinker":
            log.warning("bridge.unknown_tool", name=name)
            return

        try:
            args = json.loads(arguments)
            domain = args.get("domain", "knowledge")
            query = args.get("query", "")
        except json.JSONDecodeError:
            log.error("bridge.tool_parse_error", arguments=arguments)
            return

        log.info(
            "bridge.thinker_routed",
            session_id=self.session_id,
            domain=domain,
            query=query[:100],
        )

        # Get conversation context from Redis for the Thinker
        context = await self.session_store.get_conversation_context(self.session_id)

        # Route to the Thinker — this runs concurrently while the Responder
        # continues talking to the user
        start_time = time.monotonic()
        result = await self.thinker_router.think(
            domain=domain,
            query=query,
            context=context,
            session_id=self.session_id,
        )
        elapsed_ms = (time.monotonic() - start_time) * 1000

        log.info(
            "bridge.thinker_complete",
            session_id=self.session_id,
            domain=domain,
            elapsed_ms=round(elapsed_ms, 1),
        )

        # Return the Thinker result to the Realtime API as a tool response
        tool_response = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self._realtime_ws.send(json.dumps(tool_response))

        # Trigger the Responder to generate a response based on the Thinker output
        await self._realtime_ws.send(json.dumps({"type": "response.create"}))

    async def _handle_transcription(self, event: dict, role: str):
        """Store transcriptions in Redis for conversation context."""
        transcript = event.get("transcript", "")
        if transcript:
            await self.session_store.append_turn(
                session_id=self.session_id,
                role=role,
                content=transcript,
            )

    async def cleanup(self):
        """Clean up resources."""
        self._running = False
        if self._realtime_ws:
            await self._realtime_ws.close()
        log.info("bridge.cleaned_up", session_id=self.session_id)
