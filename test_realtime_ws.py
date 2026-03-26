"""
Phase 1, Step 1: Verify WebSocket connection to OpenAI Realtime API.

Run this standalone to confirm:
  - Auth works with your API key
  - Session configuration is accepted
  - You can send text input and receive audio/text events back

Usage:
    python scripts/test_realtime_ws.py

This should print a stream of events including session.created,
session.updated, and response events. If it works, your bridge
will work too.
"""

import asyncio
import json
import os

import websockets

REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview")
API_KEY = os.environ.get("OPENAI_API_KEY", "")


async def main():
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }
    url = f"{REALTIME_API_URL}?model={MODEL}"

    print(f"Connecting to {MODEL}...")

    async with websockets.connect(url, extra_headers=headers) as ws:
        # Step 1: Configure session
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful assistant. Respond briefly.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "tools": [
                    {
                        "type": "function",
                        "name": "route_to_thinker",
                        "description": "Route a question to a specialist agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "enum": ["weather", "stocks", "news", "knowledge"],
                                },
                                "query": {"type": "string"},
                            },
                            "required": ["domain", "query"],
                        },
                    }
                ],
            },
        }
        await ws.send(json.dumps(config))
        print("✓ Session config sent")

        # Step 2: Wait for session.created and session.updated
        for _ in range(10):
            msg = json.loads(await ws.recv())
            event_type = msg.get("type", "")
            print(f"  ← {event_type}")
            if event_type == "session.updated":
                break

        # Step 3: Send a text message that should trigger route_to_thinker
        create_msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What's the weather like in Seattle right now?",
                    }
                ],
            },
        }
        await ws.send(json.dumps(create_msg))
        print("✓ User message sent: 'What's the weather like in Seattle right now?'")

        # Trigger response
        await ws.send(json.dumps({"type": "response.create"}))
        print("✓ response.create sent")

        # Step 4: Read events until response.done
        print("\nEvents:")
        audio_bytes_received = 0
        tool_call_detected = False

        try:
            async for raw in ws:
                msg = json.loads(raw)
                event_type = msg.get("type", "")

                if event_type == "response.audio.delta":
                    # Don't spam — just count audio bytes
                    delta = msg.get("delta", "")
                    audio_bytes_received += len(delta)
                elif event_type == "response.function_call_arguments.done":
                    tool_call_detected = True
                    print(f"  ← {event_type}")
                    print(f"    name: {msg.get('name')}")
                    print(f"    args: {msg.get('arguments')}")
                elif event_type == "response.done":
                    print(f"  ← {event_type}")
                    break
                elif event_type == "response.audio_transcript.delta":
                    # Print transcript chunks inline
                    print(msg.get("delta", ""), end="", flush=True)
                elif event_type == "response.audio_transcript.done":
                    print()  # Newline after transcript
                    print(f"  ← {event_type}")
                else:
                    print(f"  ← {event_type}")

        except asyncio.TimeoutError:
            print("Timed out waiting for events")

        # Summary
        print(f"\n--- Summary ---")
        print(f"Audio bytes received: {audio_bytes_received}")
        print(f"Tool call detected: {tool_call_detected}")

        if tool_call_detected:
            print("\n✓ route_to_thinker was called! The Responder is routing correctly.")
            print("  Next step: wire up the ThinkerRouter to handle this call.")
        elif audio_bytes_received > 0:
            print("\n⚠ Got audio but no tool call. The model responded directly")
            print("  instead of routing. May need to adjust the system prompt.")
        else:
            print("\n✗ No audio and no tool call. Something went wrong.")


if __name__ == "__main__":
    asyncio.run(main())
