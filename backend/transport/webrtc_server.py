"""
Server-side WebRTC handling via aiortc.

Manages peer connections with browsers. For each session:
  - Receives audio from the browser's microphone track
  - Provides an audio track for sending Realtime API responses back

This is the piece that most demos skip by connecting directly to OpenAI.
Having it here gives you full control over the audio pipeline.
"""

import asyncio
import os
import socket as _socket
import threading
from dataclasses import dataclass, field
from fractions import Fraction

import av
import numpy as np
import structlog
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Docker ICE patch
#
# Problem: aiortc (via aioice) gathers ICE host candidates using the
# container's internal IP (172.x) on random ephemeral ports. The browser
# can't reach either the IP or the port.
#
# Fix: when RTC_FORCE_HOST and RTC_PORT_RANGE are set, we monkey-patch
# aioice.Connection.get_component_candidates to:
#   1. Bind UDP sockets to 0.0.0.0:<port_from_range> (so Docker port
#      forwarding reaches them).
#   2. Advertise RTC_FORCE_HOST (127.0.0.1) in ICE candidates so the
#      browser connects to localhost, which Docker forwards into the
#      container.
# ---------------------------------------------------------------------------
_RTC_FORCE_HOST = os.environ.get("RTC_FORCE_HOST")
_RTC_PORT_RANGE = os.environ.get("RTC_PORT_RANGE", "")

if _RTC_FORCE_HOST and _RTC_PORT_RANGE:
    import aioice.ice as _aioice
    from aioice import Candidate as _Candidate, turn as _turn

    _pmin, _pmax = (int(x) for x in _RTC_PORT_RANGE.split("-"))
    _port_lock = threading.Lock()
    _next_port = _pmin

    def _alloc_port() -> int:
        global _next_port
        with _port_lock:
            p = _next_port
            _next_port = _pmin if _next_port >= _pmax else _next_port + 1
            return p

    async def _docker_get_component_candidates(self, component: int, addresses: list[str], timeout: int = 5) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        loop = asyncio.get_event_loop()
        port = _alloc_port()

        try:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _aioice.StunProtocol(self),
                local_addr=("0.0.0.0", port),
            )
            sock = transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(
                    _socket.SOL_SOCKET,
                    _socket.SO_RCVBUF,
                    _turn.UDP_SOCKET_BUFFER_SIZE,
                )
        except OSError:
            log.warning("rtc.port_bind_failed", port=port)
            return candidates

        protocol.local_candidate = _Candidate(
            foundation=_aioice.candidate_foundation("host", "udp", _RTC_FORCE_HOST),
            component=component,
            transport="udp",
            priority=_aioice.candidate_priority(component, "host"),
            host=_RTC_FORCE_HOST,
            port=port,
            type="host",
        )
        candidates.append(protocol.local_candidate)
        self._protocols.append(protocol)
        return candidates

    _aioice.Connection.get_component_candidates = _docker_get_component_candidates
    log.info(
        "rtc.docker_ice_patched",
        host=_RTC_FORCE_HOST,
        ports=_RTC_PORT_RANGE,
    )


class AudioOutputStream(MediaStreamTrack):
    """
    Synthetic audio track that we push PCM16 frames into.
    The Realtime API bridge writes to this; WebRTC sends it to the browser.

    aiortc calls recv() in a loop to pull frames. We must always return
    a frame promptly (silence if nothing is queued) so the RTP stream
    stays alive and timestamps advance correctly.

    Incoming frames from the Realtime API are variable-sized. We re-chunk
    them into fixed 960-sample (20 ms) frames that aiortc expects.
    """

    kind = "audio"

    _SAMPLE_RATE = 48000
    _SAMPLES_PER_FRAME = 960  # 20 ms at 48 kHz
    _PTIME = 0.02  # 20 ms

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pts = 0  # running presentation timestamp in samples
        self._buffer = np.empty(0, dtype=np.int16)  # leftover samples
        self._start: float | None = None  # monotonic clock anchor
        self._frame_count = 0  # number of frames emitted (for clock calc)

    # Pre-allocate a reusable silence array
    _SILENCE = np.zeros(_SAMPLES_PER_FRAME, dtype=np.int16)

    def _make_frame(self, samples: np.ndarray) -> av.AudioFrame:
        """Build a 48 kHz mono s16 frame from a 1-D int16 array."""
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._SAMPLE_RATE
        return frame

    def clear(self):
        """Flush all queued audio (used on user interruption / barge-in)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer = np.empty(0, dtype=np.int16)

    async def push_frame(self, frame):
        """
        Accept a variable-sized frame, chunk it into 960-sample pieces,
        and enqueue them for recv().
        """
        audio = frame.to_ndarray().flatten().astype(np.int16)
        self._buffer = np.concatenate([self._buffer, audio])

        while len(self._buffer) >= self._SAMPLES_PER_FRAME:
            chunk = self._buffer[: self._SAMPLES_PER_FRAME]
            self._buffer = self._buffer[self._SAMPLES_PER_FRAME :]
            await self._queue.put(self._make_frame(chunk))

    async def recv(self):
        """
        Called by aiortc to get the next frame to send.

        Uses a wall-clock pacer: each call sleeps until the next 20 ms
        tick relative to a fixed start time. This keeps RTP packets
        evenly spaced regardless of event-loop jitter or queue timing.
        """
        loop = asyncio.get_event_loop()

        # Anchor the clock on the very first recv()
        if self._start is None:
            self._start = loop.time()

        # Calculate when the *next* frame should be emitted
        target = self._start + (self._frame_count + 1) * self._PTIME
        now = loop.time()
        delay = target - now

        if delay > 0:
            # Sleep until our next tick, but grab a frame from the queue
            # while we wait so we're ready the instant the tick fires.
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=delay)
                # We got a frame early — still wait for the tick
                remaining = target - loop.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
            except asyncio.TimeoutError:
                # Tick fired with no data — emit silence
                frame = self._make_frame(self._SILENCE)
        else:
            # We're behind schedule — drain a queued frame immediately
            # (no sleep) to catch up.
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                frame = self._make_frame(self._SILENCE)

        self._frame_count += 1

        # Stamp pts so aiortc/RTP can pace the stream
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._SAMPLE_RATE)
        self._pts += frame.samples
        return frame


@dataclass
class SessionTracks:
    """Audio tracks for a single WebRTC session."""

    peer_connection: RTCPeerConnection
    input_track: MediaStreamTrack | None = None  # Audio FROM browser
    output_track: AudioOutputStream = field(default_factory=AudioOutputStream)  # Audio TO browser


class WebRTCServer:
    """Manages WebRTC peer connections for all active sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionTracks] = {}
        self._relay = MediaRelay()

    async def create_session(self, session_id: str, offer_sdp: str) -> tuple[str, SessionTracks]:
        """
        Handle a WebRTC offer from the browser.

        Returns the SDP answer and the session tracks for the bridge to use.
        """
        pc = RTCPeerConnection()
        session = SessionTracks(peer_connection=pc)
        self._sessions[session_id] = session

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            log.info("rtc.track_received", session_id=session_id, kind=track.kind)
            if track.kind == "audio":
                # Store the browser's audio track — the bridge will read from this
                session.input_track = track

            @track.on("ended")
            async def on_ended():
                log.info("rtc.track_ended", session_id=session_id, kind=track.kind)

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            log.info(
                "rtc.state_change",
                session_id=session_id,
                state=pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed"):
                await self.close_session(session_id)

        # Add our output track (audio TO browser) to the peer connection
        pc.addTrack(session.output_track)

        # Set the remote offer and create our answer
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        log.info("rtc.session_created", session_id=session_id)
        return pc.localDescription.sdp, session

    async def close_session(self, session_id: str):
        """Tear down a WebRTC session."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.peer_connection.close()
            log.info("rtc.session_closed", session_id=session_id)
