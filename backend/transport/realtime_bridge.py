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
import json
import time

import structlog
import websockets
from langsmith.run_helpers import trace

from backend.config import settings
from backend.audio_convert import aiortc_frame_to_realtime_b64, realtime_b64_to_aiortc_frame
from backend.thinkers.router import ThinkerRouter
from backend.state.session_store import SessionStore

log = structlog.get_logger()

# OpenAI Realtime API WebSocket endpoint
REALTIME_API_URL = "wss://us.api.openai.com/v1/realtime"

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

## Language:
- The User will always speak English.
- Speak the language of the user — if they ask in Spanish, respond in Spanish.
- Default to speaking English if you can't detect the language.
- Keep your responses concise and conversational — this is a voice interaction, not text.

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
        audio_track=None,  # SessionTracks from WebRTCServer
        websocket=None,  # WebSocket fallback
    ):
        self.session_id = session_id
        self.thinker_router = thinker_router
        self.session_store = session_store
        self.audio_track = audio_track
        self.websocket = websocket
        self._realtime_ws = None
        self._running = False
        self._response_active = False  # True while the API is generating a response

        # Event queue for pushing transcripts/events to the frontend via SSE
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Tracing state: session (root) → turn → thinker
        self._session_ctx = None  # trace() for the whole session
        self._session_run = None  # RunTree for the session (root run)
        self._turn_ctx = None  # trace() for the current conversation turn
        self._turn_run = None  # RunTree for the current turn

    # ── Tracing lifecycle ───────────────────────────────────────────────
    # One session = one LangSmith run. Each turn and thinker call nests inside.

    async def _start_session_trace(self) -> None:
        """Open the root session trace. Called once when the bridge starts."""
        self._session_ctx = trace(
            "voice_session",
            run_type="chain",
            inputs={"session_id": self.session_id},
            metadata={"session_id": self.session_id},
        )
        self._session_run = await self._session_ctx.__aenter__()
        self._session_run.session_id = self.session_id

    async def _end_session_trace(self) -> None:
        """Close the root session trace. Called on cleanup."""
        await self._end_turn("[session ended]")
        if self._session_run:
            self._session_run.end(outputs={"session_id": self.session_id, "status": "completed"})
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        self._session_ctx = None
        self._session_run = None

    async def _start_turn(self, user_message: str, conversation_context: list[dict]) -> None:
        """Open a conversation turn trace, nested under the session."""
        await self._end_turn("[superseded]")
        self._turn_ctx = trace(
            "conversation_turn",
            run_type="chain",
            inputs={"user_message": user_message, "conversation_context": conversation_context},
            metadata={"session_id": self.session_id},
            parent=self._session_run,
        )
        self._turn_run = await self._turn_ctx.__aenter__()

    async def _end_turn(self, response: str) -> None:
        """Close the current turn trace with the assistant's response."""
        if self._turn_run:
            self._turn_run.end(outputs={"assistant_response": response})
        if self._turn_ctx:
            await self._turn_ctx.__aexit__(None, None, None)
        self._turn_ctx = None
        self._turn_run = None

    async def run(self):
        """Main loop: connect to Realtime API and start bidirectional streaming."""
        self._running = True
        await self._start_session_trace()

        try:
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
            }
            url = f"{REALTIME_API_URL}?model={settings.realtime_model}"

            log.info("bridge.connecting", session_id=self.session_id, url=url)

            async with websockets.connect(url, additional_headers=headers) as ws:
                self._realtime_ws = ws

                # Configure the session
                await self._configure_session()

                # Run two concurrent loops:
                #   1. Read audio from browser → send to OpenAI
                #   2. Read events from OpenAI → handle them
                await asyncio.gather(
                    self._audio_input_loop(),
                    self._event_handler_loop(),
                    return_exceptions=True,
                )
        finally:
            await self._end_session_trace()
            self.event_queue.put_nowait(None)  # Signal SSE stream to close

    async def _configure_session(self):
        """Send initial session configuration to the Realtime API."""
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": RESPONDER_INSTRUCTIONS,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        # "noise_reduction": {"type": "far_field"},
                        "transcription": {
                            "model": "gpt-4o-mini-transcribe",
                        },
                        "turn_detection": {"type": "semantic_vad"},  # {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500},
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "voice": settings.realtime_voice,
                    },
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
        session = self.audio_track
        while session.input_track is None and self._running:
            await asyncio.sleep(0.01)

        input_track = session.input_track
        log.info("bridge.audio_input_started", session_id=self.session_id)

        try:
            while self._running:
                frame = await input_track.recv()

                # Convert aiortc AudioFrame → 24kHz mono PCM16 base64
                # Handles sample rate conversion (48kHz → 24kHz) and channel mixing
                audio_b64 = aiortc_frame_to_realtime_b64(frame)
                await self._realtime_ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }
                    )
                )
        except Exception as e:
            log.error("bridge.audio_input_error", error=str(e), session_id=self.session_id)
        finally:
            # Browser disconnected — close the Realtime API WebSocket so
            # the event handler loop exits and run() can reach its finally block.
            self._running = False
            if self._realtime_ws:
                await self._realtime_ws.close()

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
                    # --- Response lifecycle ---
                    case "response.created":
                        self._response_active = True

                    case "response.done":
                        self._response_active = False

                    # --- Audio output: send back to browser ---
                    case "response.output_audio.delta":
                        await self._handle_audio_output(event)

                    # --- Tool calls: intercept for Thinker routing ---
                    case "response.function_call_arguments.done":
                        # Run thinker concurrently so the Responder keeps talking
                        asyncio.create_task(self._handle_tool_call(event))

                    # --- User interruption (barge-in) ---
                    case "input_audio_buffer.speech_started":
                        await self._end_turn("[interrupted by user]")
                        await self._handle_interrupt()

                    # --- Transcription: turn lifecycle ---
                    case "conversation.item.input_audio_transcription.completed":
                        await self._handle_user_transcription(event)

                    case "response.output_audio_transcript.done":
                        await self._handle_assistant_transcription(event)

                    # --- Session lifecycle ---
                    case "session.created":
                        log.info("bridge.session_ready", session_id=self.session_id)

                    case "error":
                        log.error(
                            "bridge.realtime_error",
                            session_id=self.session_id,
                            error=event.get("error"),
                        )

                    case _:
                        pass
                        # log.debug("bridge.event", type=event_type)

        except websockets.exceptions.ConnectionClosed:
            log.info("bridge.realtime_disconnected", session_id=self.session_id)

    async def _handle_audio_output(self, event: dict):
        """
        Forward audio from Realtime API back to the browser.

        Converts base64 PCM16 at 24kHz to an aiortc AudioFrame at 48kHz
        and pushes it to the WebRTC output track.
        """
        audio_b64 = event.get("delta", "")
        if not audio_b64 or not self.audio_track:
            return

        # Convert 24kHz PCM16 base64 → 48kHz aiortc AudioFrame
        frame = realtime_b64_to_aiortc_frame(audio_b64)
        await self.audio_track.output_track.push_frame(frame)

    async def _handle_interrupt(self):
        """
        Handle user barge-in: the user started speaking while the model
        was still outputting audio.

        1. Tell the Realtime API to stop generating the current response.
        2. Flush any queued audio so the user doesn't hear stale output.
        """
        log.info("bridge.user_interrupt", session_id=self.session_id)

        # Cancel the in-flight response so the API stops sending audio deltas
        if self._response_active:
            await self._realtime_ws.send(json.dumps({"type": "response.cancel"}))
            self._response_active = False

        # Flush queued audio frames so the browser stops hearing old output
        if self.audio_track:
            self.audio_track.output_track.clear()

    async def _handle_user_transcription(self, event: dict):
        """User finished speaking — start a new conversation turn trace."""
        transcript = event.get("transcript", "")
        if not transcript:
            return

        context = await self.session_store.get_conversation_context(self.session_id)

        await self.session_store.append_turn(
            session_id=self.session_id,
            role="user",
            content=transcript,
        )

        await self._start_turn(user_message=transcript, conversation_context=context)

        self.event_queue.put_nowait({"type": "transcript", "role": "user", "content": transcript})

        log.info(
            "bridge.transcription",
            session_id=self.session_id,
            role="user",
            transcript=transcript,
        )

    async def _handle_assistant_transcription(self, event: dict):
        """Assistant finished speaking — close the conversation turn trace."""
        transcript = event.get("transcript", "")
        if not transcript:
            return

        await self.session_store.append_turn(
            session_id=self.session_id,
            role="assistant",
            content=transcript,
        )

        await self._end_turn(transcript)

        self.event_queue.put_nowait({"type": "transcript", "role": "assistant", "content": transcript})

        log.info(
            "bridge.transcription",
            session_id=self.session_id,
            role="assistant",
            transcript=transcript,
        )

    async def _handle_tool_call(self, event: dict):
        """
        Intercept tool calls from the Responder.

        Parses the event, runs the Thinker as a child span of the current
        conversation turn, and returns the result to the Realtime API.
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

        self.event_queue.put_nowait({"type": "thinker", "event": "routed", "domain": domain, "query": query})

        context = await self.session_store.get_conversation_context(self.session_id)

        start_time = time.monotonic()

        # Nest the thinker trace explicitly under the current conversation turn
        parent = self._turn_run or self._session_run
        async with trace(
            "thinker_call",
            run_type="tool",
            inputs={"domain": domain, "query": query},
            metadata={"session_id": self.session_id, "domain": domain},
            parent=parent,
        ) as thinker_span:
            result = await self.thinker_router.think(
                domain=domain,
                query=query,
                context=context,
                session_id=self.session_id,
            )
            thinker_span.end(outputs={"result": result[:200] if result else ""})
        elapsed_ms = (time.monotonic() - start_time) * 1000

        log.info(
            "bridge.thinker_complete",
            session_id=self.session_id,
            domain=domain,
            elapsed_ms=round(elapsed_ms, 1),
        )

        self.event_queue.put_nowait({"type": "thinker", "event": "complete", "domain": domain, "elapsed_ms": round(elapsed_ms, 1)})

        # Return the Thinker result to the Realtime API as a tool response
        await self._realtime_ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    },
                }
            )
        )

        # Trigger the Responder to generate a response based on the Thinker output
        await self._realtime_ws.send(json.dumps({"type": "response.create"}))

    async def cleanup(self):
        """Clean up resources."""
        self._running = False
        self.event_queue.put_nowait(None)  # Signal SSE to close
        if self._realtime_ws:
            await self._realtime_ws.close()
        log.info("bridge.cleaned_up", session_id=self.session_id)
