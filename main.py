"""
main.py — Speech-to-Text WebSocket server

Architecture
────────────
Browser sends 4-second audio chunks over WebSocket (binary).
Browser also sends JSON control messages (text) for language switching.

                 ┌─────────────┐
  4 s webm  ───► │ chunk queue │ len ≥ WINDOW → merge → transcribe → send text
  JSON ctrl ───► │  set_lang   │ → update active language in transcriber
                 └─────────────┘

WINDOW = 1  → ~4 s latency, good for demo
WINDOW = 2  → ~8 s latency, slightly better accuracy
"""

import json
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from transcriber import transcribe_audio, WHISPER_OPTIONS
import uvicorn
import os

app = FastAPI()

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

WINDOW = 1  # number of 4-second chunks to buffer before transcribing


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/config")
async def config():
    """Expose server config to frontend so WINDOW never desyncs."""
    return JSONResponse({"window": WINDOW})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    buf: deque[bytes] = deque(maxlen=WINDOW)

    try:
        while True:
            message = await websocket.receive()

            # ── Text message = control command (e.g. language switch) ──────────
            if "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == "set_lang":
                        lang = ctrl.get("lang", "hi")
                        WHISPER_OPTIONS["language"] = None if lang == "auto" else lang
                        print(f"  🌐 Language switched to: {lang}")
                except Exception as e:
                    print(f"  ⚠ Control message error: {e}")
                continue

            # ── Binary message = audio chunk ──────────────────────────────────
            raw = message.get("bytes")
            if not raw:
                continue

            print(f"← {len(raw):,} B")
            buf.append(raw)

            if len(buf) < WINDOW:
                print(f"  buffering {len(buf)}/{WINDOW}…")
                continue

            combined = b"".join(buf)

            try:
                text = transcribe_audio(combined)
            except Exception as e:
                print(f"  transcription error: {e}")
                await websocket.send_text("")
                continue

            await websocket.send_text(text if text else "")

    except WebSocketDisconnect:
        print("✂ Client disconnected")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)