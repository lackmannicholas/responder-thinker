"""
Phase 1, Step 3: Full pipeline integration test.

WebRTC (browser) → aiortc → Realtime API WebSocket → aiortc → WebRTC (browser)

This is a stripped-down version of the full bridge that ONLY handles audio.
No thinkers, no Redis, no routing — just prove the audio round-trip works.

If you can talk to the Realtime API through your backend and hear responses,
everything else is just layering features on top.

Usage:
    OPENAI_API_KEY=sk-... uvicorn scripts.test_full_pipeline:app --port 8002

Then open http://localhost:8002 in your browser and talk.
"""

import asyncio
import json
import os

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from backend.transport.audio_convert import (
    aiortc_frame_to_realtime_b64,
    realtime_b64_to_aiortc_frame,
)

app = FastAPI()
relay = MediaRelay()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_URL = "wss://api.openai.com/v1/realtime"
MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview")


class RealtimeOutputTrack(MediaStreamTrack):
    """Audio track that plays Realtime API responses back to the browser."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def push(self, frame):
        await self._queue.put(frame)

    async def recv(self):
        return await self._queue.get()


async def run_bridge(input_track: MediaStreamTrack, output_track: RealtimeOutputTrack):
    """Bridge audio between WebRTC and Realtime API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(
        f"{REALTIME_URL}?model={MODEL}", extra_headers=headers
    ) as ws:
        # Configure session — minimal, just audio
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "You are a friendly assistant. Keep responses very short. "
                    "This is a test of the audio pipeline."
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }))

        print("✓ Realtime API session configured")

        async def send_audio():
            """Read audio from WebRTC → send to Realtime API."""
            try:
                while True:
                    frame = await input_track.recv()
                    audio_b64 = aiortc_frame_to_realtime_b64(frame)
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }))
            except Exception as e:
                print(f"Send audio error: {e}")

        async def recv_events():
            """Read events from Realtime API → handle audio and transcripts."""
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    event_type = event.get("type", "")

                    if event_type == "response.audio.delta":
                        delta = event.get("delta", "")
                        if delta:
                            frame = realtime_b64_to_aiortc_frame(delta)
                            await output_track.push(frame)

                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "")
                        print(f"  Assistant: {transcript}")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        print(f"  User: {transcript}")

                    elif event_type == "error":
                        print(f"  ERROR: {event.get('error')}")

                    elif event_type in ("session.created", "session.updated"):
                        print(f"  ← {event_type}")

            except websockets.exceptions.ConnectionClosed:
                print("Realtime API disconnected")

        await asyncio.gather(send_audio(), recv_events())


@app.post("/offer")
async def offer(request: dict):
    pc = RTCPeerConnection()
    output_track = RealtimeOutputTrack()
    pc.addTrack(output_track)

    input_track_holder = [None]

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            input_track_holder[0] = relay.subscribe(track)
            print(f"✓ Got browser audio track")
            # Start the bridge once we have the audio track
            asyncio.ensure_future(
                run_bridge(input_track_holder[0], output_track)
            )

    @pc.on("connectionstatechange")
    async def on_state():
        state = pc.connectionState
        print(f"WebRTC state: {state}")
        if state == "failed":
            await pc.close()

    offer_desc = RTCSessionDescription(sdp=request["sdp"], type="offer")
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp}


@app.get("/")
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>Full Pipeline Test</title></head>
<body style="font-family: monospace; background: #0a0a0a; color: #e5e5e5; padding: 40px;">
    <h2>Full Pipeline Test: Browser → Backend → OpenAI → Backend → Browser</h2>
    <p>Click Connect, then talk. You should hear the AI respond through your backend.</p>
    <button id="btn" onclick="start()" style="padding: 10px 20px; font-size: 16px;">Connect</button>
    <pre id="log" style="margin-top: 20px; color: #737373;"></pre>

    <script>
    function log(msg) {
        document.getElementById('log').textContent += msg + '\\n';
    }

    async function start() {
        document.getElementById('btn').disabled = true;
        log('Requesting microphone...');

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 24000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });
        log('Mic access granted');

        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        stream.getTracks().forEach(t => pc.addTrack(t, stream));

        pc.ontrack = (e) => {
            log('Got remote audio track — AI responses will play here');
            const audio = new Audio();
            audio.srcObject = e.streams[0];
            audio.play().catch(() => log('Click page to enable audio playback'));
        };

        pc.onconnectionstatechange = () => {
            const state = pc.connectionState;
            log('State: ' + state);
            if (state === 'connected') {
                log('\\n✓ Connected! Start talking...');
            }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        log('Sending offer to backend...');

        const resp = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: offer.sdp }),
        });
        const { sdp } = await resp.json();
        await pc.setRemoteDescription({ type: 'answer', sdp });
        log('Answer received, establishing connection...');
    }
    </script>
</body>
</html>
    """)
