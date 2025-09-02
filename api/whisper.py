# === FICHIER : api/whisper.py (CPU-only) ===
import os
import shutil
import tempfile
import mimetypes
import threading
import time
import logging
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, status
from faster_whisper import WhisperModel

from status_hub import StatusEventHub
from models.whisper_models import (
    WhisperPingResponse,
    WhisperModelsFull,
    WhisperSelectRequest,
    WhisperSelectResponse,
    WhisperTranscribeResponse,
)

router: APIRouter = APIRouter()
log = logging.getLogger("ai-server.whisper")

ALLOWED_MODELS: List[str] = ["tiny", "base", "small", "medium"]

_lock = threading.Lock()
_loading: bool = False
_requested_model: Optional[str] = None

# Hub & deps injectés par main.py
statusHub: Optional[StatusEventHub] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_status_supplier: Optional[Callable[[], Dict[str, Any]]] = None  # snapshot complet dict

def set_status_hub(hub: StatusEventHub) -> None:
    global statusHub
    statusHub = hub

def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _event_loop
    _event_loop = loop

def set_status_supplier(fn: Callable[[], Dict[str, Any]]) -> None:
    global _status_supplier
    _status_supplier = fn

# -------------------- Config Whisper : CPU only --------------------
# ⚠️ Forçage CPU uniquement (on ne tente pas CUDA). Compute par défaut "int8" pour rapidité/stabilité.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
DEVICE_EFF = "cpu"
COMPUTE_EFF = "int8"
log.info(f"Whisper forced CPU-only: device={DEVICE_EFF}, compute={COMPUTE_EFF}, default_model={WHISPER_MODEL}")

# -------------------- Chargement initial --------------------
t0 = time.perf_counter()
_model: WhisperModel = WhisperModel(WHISPER_MODEL, device=DEVICE_EFF, compute_type=COMPUTE_EFF)
log.info(f"Whisper model '{WHISPER_MODEL}' loaded in {(time.perf_counter()-t0):.2f}s (CPU)")

# -------------------- Helpers --------------------
def _is_model_cached(name: str) -> bool:
    try:
        WhisperModel(name, device="cpu", compute_type="int8", local_files_only=True)
        return True
    except Exception:
        return False

def _list_cached_models() -> WhisperModelsFull:
    downloaded = [m for m in ALLOWED_MODELS if _is_model_cached(m)]
    return WhisperModelsFull(downloaded=downloaded, all=ALLOWED_MODELS, current=WHISPER_MODEL)

def _broadcast_threadsafe(payload: Dict[str, Any]) -> None:
    if statusHub is None or _event_loop is None:
        log.warning("⚠️ StatusHub ou event loop manquant, broadcast ignoré")
        return
    fut = asyncio.run_coroutine_threadsafe(statusHub.broadcast(payload), _event_loop)
    try:
        fut.result(timeout=0)
    except Exception:
        pass

def _broadcast_snapshot() -> None:
    if _status_supplier is None:
        log.warning("⚠️ Aucun status supplier défini, snapshot impossible")
        return
    try:
        snapshot = _status_supplier()
        _broadcast_threadsafe(snapshot)
    except Exception as e:
        log.warning(f"Snapshot supplier failed: {e}")

def _load_model_async(name: str) -> None:
    """Thread: charge le modèle (CPU) puis broadcast un snapshot."""
    global _model, WHISPER_MODEL, _loading, _requested_model
    start = time.perf_counter()
    log.info(f"Loading Whisper model '{name}' (device=cpu, compute=int8)...")
    try:
        new_model = WhisperModel(name, device="cpu", compute_type="int8")
        with _lock:
            _model = new_model
            WHISPER_MODEL = name
        log.info(f"Model '{name}' ready in {(time.perf_counter()-start):.2f}s (CPU)")
    except Exception as e:
        log.exception(f"Model load failed for '{name}': {e}")
        raise
    finally:
        _loading = False
        _requested_model = None
        _broadcast_snapshot()

def _start_async_load(name: str) -> None:
    """Déclenche le chargement asynchrone (CPU) puis broadcast snapshot complet."""
    global _loading, _requested_model
    if name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu: {name}. Autorisés: {ALLOWED_MODELS}",
        )
    if _loading:
        return
    _loading = True
    _requested_model = name
    _broadcast_snapshot()  # montrera state='loading'
    threading.Thread(target=_load_model_async, args=(name,), daemon=True).start()

# -------------------- Routes --------------------
@router.get("/ping", response_model=WhisperPingResponse)
def ping():
    return {"ok": True, "service": "whisper"}

@router.get("/models", response_model=WhisperModelsFull)
def list_cached_models():
    return _list_cached_models()

@router.post("/select", response_model=WhisperSelectResponse)
async def whisper_select(req: WhisperSelectRequest, request: Request):
    """
    200: ready|loading, 404: modèle inconnu, 423: déjà en cours de chargement.
    Diffuse toujours un snapshot complet via WS.
    """
    log.info(f"select requested='{req.name}' current='{WHISPER_MODEL}' loading={_loading} (CPU)")
    if req.name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu: {req.name}. Autorisés: {ALLOWED_MODELS}",
        )

    if WHISPER_MODEL == req.name and not _loading:
        _broadcast_snapshot()
        return {"service": "whisper", "requested": req.name, "current": WHISPER_MODEL, "state": "ready"}

    if _loading:
        detail = {"requested": _requested_model, "current": WHISPER_MODEL, "state": "loading"}
        raise HTTPException(status_code=status.HTTP_423_LOCKED, detail=detail)

    _start_async_load(req.name)
    return {"service": "whisper", "requested": req.name, "current": WHISPER_MODEL, "state": "loading"}

@router.post("/transcribe", response_model=WhisperTranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    beam_size: int = Form(5),
    vad: bool = Form(True),
):
    if _loading:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail={"state": "loading", "current": WHISPER_MODEL, "requested": _requested_model},
        )

    if not file or not (file.filename or file.content_type):
        raise HTTPException(status_code=400, detail="Fichier manquant ou invalide.")
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"Content-Type invalide: {file.content_type}")

    t0 = time.perf_counter()
    log.info(f"[Transcribe] start: filename='{file.filename}' ctype='{file.content_type}' (CPU)")

    suffix = Path(file.filename or "").suffix or (mimetypes.guess_extension(file.content_type or "") or ".tmp")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        size = os.path.getsize(tmp_path)
        log.info(f"[Transcribe] temp file saved: {tmp_path} ({size} bytes)")

        # CPU-only transcription
        segments, info = _model.transcribe(
            tmp_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        text = "".join(s.text for s in segments).strip()
        log.info(
            f"[Transcribe] success in {(time.perf_counter()-t0):.2f}s, "
            f"duration={getattr(info,'duration',0):.2f}s, lang={getattr(info,'language','?')} (CPU)"
        )

        return {"text": text, "language": info.language, "duration": info.duration}

    except Exception as e:
        log.exception(f"[Transcribe] failed: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur transcription: {e}")

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# -------------------- Statut (/status global) --------------------
def get_status() -> Dict[str, Any]:
    base = {"device": "cpu", "compute_type": "int8"}
    if _loading:
        return {
            "available": False,
            "state": "loading",
            "current": WHISPER_MODEL,
            "requested": _requested_model,
            "models": _list_cached_models().dict(),
            **base,
        }
    try:
        _ = _model
        return {
            "available": True,
            "state": "ready",
            "models": {
                "downloaded": _list_cached_models().downloaded,
                "current": WHISPER_MODEL,
            },
            **base,
        }
    except Exception as e:
        return {
            "available": False,
            "state": "error",
            "error": str(e),
            "models": _list_cached_models().dict(),
            **base,
        }
