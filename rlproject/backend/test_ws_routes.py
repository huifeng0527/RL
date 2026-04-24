"""Check FastAPI WebSocket routes."""
from fastapi import FastAPI
from fastapi.websockets import WebSocket

app = FastAPI()

@app.websocket('/ws/eval')
async def ws_test(websocket):
    await websocket.accept()
    await websocket.send_text('connected')

# Check routes
for route in app.routes:
    print(route.path, type(route).__name__)
