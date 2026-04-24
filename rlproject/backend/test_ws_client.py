"""WebSocket client test."""
import asyncio
import websockets

async def test():
    try:
        uri = "ws://localhost:8000/ws/eval"
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri, ping_interval=None) as ws:
            print("Connected! Sending ping...")
            await ws.send("ping")
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"Received: {msg}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
