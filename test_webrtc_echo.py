"""
Phase 1, Step 2: Verify WebRTC audio handling with aiortc.

Standalone FastAPI server that:
  - Accepts WebRTC offer from browser
  - Receives mic audio
  - Echoes it back (loopback)

If you can hear yourself with ~100ms delay, your WebRTC pipeline works.
Then you just need to swap the echo with the Realtime API bridge.

Usage:
    uvicorn scripts.test_webrtc_echo:app --port 8001

Then open http://localhost:8001 in your browser.
"""

import asyncio
import json

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
relay = MediaRelay()


class AudioEcho(MediaStreamTrack):
    """Receives audio frames and echoes them back."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack):
        super().__init__()
        self.source = source

    async def recv(self):
        frame = await self.source.recv()
        return frame


@app.post("/offer")
async def offer(request: dict):
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            # Echo the audio back
            echo = AudioEcho(relay.subscribe(track))
            pc.addTrack(echo)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
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
<head><title>WebRTC Echo Test</title></head>
<body style="font-family: monospace; background: #0a0a0a; color: #e5e5e5; padding: 40px;">
    <h2>WebRTC Echo Test</h2>
    <p>Click Connect. If you hear yourself back, aiortc is working.</p>
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
            audio: { sampleRate: 24000, channelCount: 1, echoCancellation: false },
        });
        log('Mic access granted');

        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        stream.getTracks().forEach(t => pc.addTrack(t, stream));

        pc.ontrack = (e) => {
            log('Got remote track — playing audio');
            const audio = new Audio();
            audio.srcObject = e.streams[0];
            audio.play();
        };

        pc.onconnectionstatechange = () => log('State: ' + pc.connectionState);

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        log('Sending offer...');

        const resp = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: offer.sdp }),
        });
        const { sdp } = await resp.json();
        await pc.setRemoteDescription({ type: 'answer', sdp });
        log('Connected! You should hear yourself.');
    }
    </script>
</body>
</html>
    """)
