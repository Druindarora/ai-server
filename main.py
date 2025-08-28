# === FICHIER : main.py (début, patch) ===
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from datetime import datetime, timezone
import asyncio
import time
import io
import faulthandler
from contextlib import asynccontextmanager
from typing import AsyncIterator

from status_hub import StatusEventHub
from api import whisper, ollama
from logging_setup import setup_logging
from lifespan import lifespan as base_lifespan  # ✅ on chaîne le lifespan existant
from models.status_models import StatusResponse, WhisperStatus, OllamaStatus, ServicesStatus

logger = setup_logging()

@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ⚙️ Startup: exécute d'abord le lifespan existant, puis injecte loop + snapshot supplier
    async with base_lifespan(app):
        whisper.set_event_loop(asyncio.get_running_loop())
        whisper.set_status_supplier(lambda: status().model_dump())  # ✅ Pydantic v2
        yield
    # (Shutdown géré par base_lifespan)

app = FastAPI(title="AI Backend", lifespan=app_lifespan)

# Hub WS accessible aux routers
app.state.statusHub = StatusEventHub()
whisper.set_status_hub(app.state.statusHub)

@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await app.state.statusHub.connect(websocket)
    try:
        # 🔹 Snapshot initial complet (Pydantic v2)
        snapshot = status()
        await websocket.send_json(snapshot.model_dump())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        app.state.statusHub.disconnect(websocket)


# -------------------- Middleware --------------------
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    clen = request.headers.get("content-length")
    try:
        response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{method} {path} -> {response.status_code} in {dur_ms:.1f}ms (len={clen})")
        return response
    except Exception as e:
        dur_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"{method} {path} crashed after {dur_ms:.1f}ms: {e}")
        raise

# -------------------- Routes --------------------
app.include_router(whisper.router, prefix="/whisper", tags=["Whisper"])
app.include_router(ollama.router,  prefix="/ollama",  tags=["Ollama"])

@app.get("/status", response_model=StatusResponse)
def status():
    w_raw = whisper.get_status()
    o_raw = ollama.get_status()

    w = WhisperStatus(**w_raw)
    o = OllamaStatus(**o_raw)

    if w.available and o.available:
        overall = "ready"
    elif w.available or o.available:
        overall = "degraded"
    else:
        overall = "down"

    return StatusResponse(
        overall=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=ServicesStatus(whisper=w, ollama=o),
    )

@app.get("/debug/threads")
def debug_threads():
    buf = io.StringIO()
    faulthandler.dump_traceback(file=buf, all_threads=True)
    return {"threads": buf.getvalue()}
