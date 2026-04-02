/**
 * Responder-Thinker WebRTC Client
 *
 * Connects to the Python backend via WebRTC for audio streaming.
 * The backend bridges to OpenAI's Realtime API via WebSocket.
 *
 * Flow:
 *   1. getUserMedia() for microphone access
 *   2. Create RTCPeerConnection
 *   3. Add mic track to peer connection
 *   4. Create offer, send to backend /api/rtc/offer
 *   5. Backend creates answer + starts Realtime API bridge
 *   6. Set remote description with answer
 *   7. Audio flows bidirectionally
 */

let peerConnection = null;
let localStream = null;
let sessionId = null;
let eventSource = null;
let streamingTurn = null; // Current assistant bubble being streamed into

// --- UI Helpers ---

function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const label = document.getElementById('statusText');
    dot.className = `status-dot ${state}`;
    label.textContent = text;
}

function addTranscript(role, content, domain) {
    const container = document.getElementById('transcript');
    const turn = document.createElement('div');
    turn.className = `turn ${role}`;

    if (domain) {
        const tag = document.createElement('div');
        tag.className = 'domain-tag';
        tag.textContent = `${domain} thinker`;
        turn.appendChild(tag);
    }

    const text = document.createElement('span');
    text.textContent = content;
    turn.appendChild(text);

    container.appendChild(turn);
    container.scrollTop = container.scrollHeight;
}

function addEvent(type, detail, className) {
    const log = document.getElementById('eventLog');
    const event = document.createElement('div');
    event.className = `event ${className || ''}`;

    const time = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });

    event.textContent = `${time} ${type}${detail ? ': ' + detail : ''}`;
    log.appendChild(event);
    log.scrollTop = log.scrollHeight;

    // Keep event log manageable
    while (log.children.length > 200) {
        log.removeChild(log.children[1]); // Keep the h2
    }
}

// --- WebRTC Connection ---

async function connect() {
    try {
        setStatus('connecting', 'Requesting microphone...');
        document.getElementById('connectBtn').disabled = true;

        // Get microphone access
        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 24000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
            video: false,
        });

        addEvent('mic', 'Microphone access granted');
        setStatus('connecting', 'Establishing connection...');

        // Create peer connection
        peerConnection = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        });

        // Add microphone track
        localStream.getTracks().forEach((track) => {
            peerConnection.addTrack(track, localStream);
        });

        // Handle incoming audio from backend
        peerConnection.ontrack = (event) => {
            addEvent('track', `Received ${event.track.kind} track`);

            const audio = new Audio();
            audio.srcObject = event.streams[0];
            audio.play().catch((e) => {
                addEvent('audio', 'Autoplay blocked — click to unmute', 'error');
            });
        };

        // Monitor connection state
        peerConnection.onconnectionstatechange = () => {
            const state = peerConnection.connectionState;
            addEvent('rtc', `Connection state: ${state}`);

            switch (state) {
                case 'connected':
                    setStatus('connected', 'Connected — speak!');
                    document.getElementById('disconnectBtn').disabled = false;
                    addTranscript('system', 'Connected. Start talking!');
                    break;
                case 'disconnected':
                case 'failed':
                    setStatus('error', 'Connection lost');
                    cleanup();
                    break;
            }
        };

        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                addEvent('ice', 'ICE candidate gathered');
            }
        };

        // Create offer and wait for ICE gathering to complete so the SDP
        // sent to the backend contains all candidates (aiortc needs them).
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        await new Promise((resolve) => {
            if (peerConnection.iceGatheringState === 'complete') {
                resolve();
            } else {
                peerConnection.addEventListener('icegatheringstatechange', () => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        resolve();
                    }
                });
            }
        });

        addEvent('rtc', 'Sending SDP offer to backend');

        const response = await fetch('/api/rtc/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sdp: peerConnection.localDescription.sdp }),
        });

        if (!response.ok) {
            throw new Error(`Backend returned ${response.status}`);
        }

        const { sdp, session_id } = await response.json();
        sessionId = session_id;

        addEvent('rtc', `Session: ${session_id.slice(0, 8)}...`);

        // Set the backend's answer
        const answer = new RTCSessionDescription({ type: 'answer', sdp });
        await peerConnection.setRemoteDescription(answer);

        addEvent('rtc', 'SDP answer set, waiting for ICE...');

        // Connect to SSE for transcripts and thinker events
        connectEventStream(session_id);

    } catch (error) {
        console.error('Connection failed:', error);
        setStatus('error', 'Connection failed');
        addEvent('error', error.message, 'error');
        document.getElementById('connectBtn').disabled = false;
        cleanup();
    }
}

function disconnect() {
    addEvent('rtc', 'Disconnecting...');
    addTranscript('system', 'Disconnected.');
    cleanup();
}

function cleanup() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    streamingTurn = null;

    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
        localStream = null;
    }

    sessionId = null;
    setStatus('', 'Disconnected');
    document.getElementById('connectBtn').disabled = false;
    document.getElementById('disconnectBtn').disabled = true;
}

// --- Server-Sent Events (transcripts & thinker events) ---

function connectEventStream(sid) {
    eventSource = new EventSource(`/api/events/${sid}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'transcript':
                if (data.role === 'user' && streamingTurn) {
                    // User transcription arrived after assistant started streaming —
                    // insert it before the in-progress assistant bubble to keep order correct.
                    const container = document.getElementById('transcript');
                    const turn = document.createElement('div');
                    turn.className = 'turn user';
                    const text = document.createElement('span');
                    text.textContent = data.content;
                    turn.appendChild(text);
                    container.insertBefore(turn, streamingTurn);
                    container.scrollTop = container.scrollHeight;
                } else {
                    addTranscript(data.role, data.content);
                }
                addEvent('transcript', `${data.role}: ${data.content.slice(0, 60)}...`);
                break;

            case 'transcript_interrupted':
                streamingTurn = null;
                break;

            case 'transcript_delta': {
                const container = document.getElementById('transcript');
                if (!streamingTurn) {
                    streamingTurn = document.createElement('div');
                    streamingTurn.className = 'turn assistant';
                    const text = document.createElement('span');
                    streamingTurn.appendChild(text);
                    container.appendChild(streamingTurn);
                }
                const span = streamingTurn.querySelector('span');
                span.textContent += data.delta;
                container.scrollTop = container.scrollHeight;
                break;
            }

            case 'transcript_done':
                if (streamingTurn) {
                    const span = streamingTurn.querySelector('span');
                    span.textContent = data.content;
                }
                streamingTurn = null;
                addEvent('transcript', `assistant: ${data.content.slice(0, 60)}...`);
                break;

            case 'thinker':
                if (data.event === 'routed') {
                    addEvent('thinker', `→ ${data.domain}: "${data.query.slice(0, 50)}"`, 'thinker');
                } else if (data.event === 'complete') {
                    addEvent('thinker', `← ${data.domain} (${data.elapsed_ms}ms)`, 'thinker');
                }
                break;

            case 'session_ended':
                addTranscript('system', 'Session ended.');
                addEvent('session', 'Session ended');
                cleanup();
                break;

            case 'session_idle_nudge':
                addEvent('idle', 'Nudging user (15s idle)');
                break;

            case 'audio_playback_finished':
                addEvent('idle', 'Audio finished — idle timer started');
                break;

            case 'session_idle_disconnect':
                addTranscript('system', 'Disconnected due to inactivity.');
                addEvent('idle', 'Disconnected (60s idle)');
                cleanup();
                break;
        }
    };

    eventSource.onerror = () => {
        addEvent('sse', 'Event stream disconnected', 'error');
    };
}
