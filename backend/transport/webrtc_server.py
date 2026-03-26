"""
Server-side WebRTC handling via aiortc.

Manages peer connections with browsers. For each session:
  - Receives audio from the browser's microphone track
  - Provides an audio track for sending Realtime API responses back

This is the piece that most demos skip by connecting directly to OpenAI.
Having it here gives you full control over the audio pipeline.
"""

import asyncio
from dataclasses import dataclass, field

import structlog
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

log = structlog.get_logger()


class AudioOutputStream(MediaStreamTrack):
    """
    Synthetic audio track that we push PCM16 frames into.
    The Realtime API bridge writes to this; WebRTC sends it to the browser.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def push_frame(self, frame):
        """Push an audio frame to be sent to the browser."""
        await self._queue.put(frame)

    async def recv(self):
        """Called by aiortc to get the next frame to send."""
        return await self._queue.get()


@dataclass
class SessionTracks:
    """Audio tracks for a single WebRTC session."""

    peer_connection: RTCPeerConnection
    input_track: MediaStreamTrack | None = None       # Audio FROM browser
    output_track: AudioOutputStream = field(default_factory=AudioOutputStream)  # Audio TO browser


class WebRTCServer:
    """Manages WebRTC peer connections for all active sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionTracks] = {}
        self._relay = MediaRelay()

    async def create_session(
        self, session_id: str, offer_sdp: str
    ) -> tuple[str, SessionTracks]:
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
