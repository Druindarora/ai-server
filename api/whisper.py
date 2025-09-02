# === FICHIER : api/whisper.py (CPU-only + WS streaming) ===


"""
=== Whisper API (CPU-only + streaming) ===

📌 Endpoints disponibles :

HTTP:
  - GET  /ping          → Vérifie la disponibilité du service
  - GET  /models        → Liste les modèles disponibles localement
  - POST /select        → Sélectionne un modèle (tiny, base, small, medium)
  - POST /transcribe    → Transcription d’un fichier audio complet

WebSocket:
  - WS   /ws/transcribe → Transcription en streaming
      Client → {"type":"start","language":"en","beam_size":5,"vad":true,"sample_rate":16000}
      Client → {"type":"audio_chunk","data":"<base64 PCM16 mono 16k>"}
      Client → {"type":"flush"} | {"type":"end"}
      Serveur → {"type":"ready"} | {"type":"partial"} | {"type":"final"} | {"type":"error"}

Statut interne:
  - get_status()        → Utilisé par /status global
"""





import os
import shutil
import tempfile
import mimetypes
import threading
import time
import logging
import asyncio
import base64
import json
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, status, WebSocket, WebSocketDisconnect
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
# ⚠️ Forçage CPU uniquement.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
DEVICE_EFF = "cpu"
COMPUTE_EFF = "int8"
DEFAULT_SR = 16000  # attendu côté client pour PCM16 mono
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

# -------------------- Routes HTTP --------------------
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

# -------------------- WebSocket streaming /ws/transcribe --------------------
# Protocole côté client :
#   → {"type":"start","language":"en","beam_size":5,"vad":true,"sample_rate":16000}
#   → {"type":"audio_chunk","data":"<base64 pcm16 mono 16k>"}
#   → {"type":"audio_chunk", ...} (répété)
#   → {"type":"end"}
# Serveur émet :
#   ← {"type":"ready"}
#   ← {"type":"partial","text":"...","t_end":12.34}
#   ← {"type":"final","text":"...","language":"en","duration":23.10}

class _StreamSession:
    # État par-connexion WS
    def __init__(self) -> None:
        self.language: Optional[str] = "en"
        self.beam_size: int = 5
        self.vad: bool = True
        self.sample_rate: int = DEFAULT_SR
        self.buffer: bytearray = bytearray()
        self.last_sent_end: float = 0.0
        self.infer_lock: asyncio.Lock = asyncio.Lock()
        self.last_infer_ts: float = 0.0
        self.min_interval_s: float = 0.6  # anti-spam d'inférence
        self.min_buffer_ms: int = 500     # attendre ≥500ms audio

def _pcm16_to_wav_file(pcm: bytes, sample_rate: int) -> str:
    # Écrit un wav mono 16-bit à partir de PCM16 brut
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return tmp_path

def _transcribe_file(path: str, language: Optional[str], beam_size: int, vad: bool) -> Tuple[List[Any], Any]:
    seg_iter, info = _model.transcribe(
        path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segs = list(seg_iter)
    return segs, info

async def _maybe_infer(session: _StreamSession, ws: WebSocket) -> None:
    # Déclenche une inférence si assez de données et si délai respecté
    now = time.monotonic()
    if session.infer_lock.locked():
        return
    if (now - session.last_infer_ts) < session.min_interval_s:
        return
    # Taille min en samples (mono 16k, 2 bytes/sample)
    if len(session.buffer) < int(session.sample_rate * 2 * (session.min_buffer_ms / 1000)):
        return
    asyncio.create_task(_run_infer(session, ws, final=False))

async def _run_infer(session: _StreamSession, ws: WebSocket, final: bool) -> None:
    # Lance une inférence (protégée par un lock)
    async with session.infer_lock:
        session.last_infer_ts = time.monotonic()
        pcm = bytes(session.buffer)  # copie instantanée
        tmp_wav = _pcm16_to_wav_file(pcm, session.sample_rate)
        try:
            segs, info = await asyncio.to_thread(
                _transcribe_file, tmp_wav, session.language, session.beam_size, session.vad
            )
            # Sélection des nouveaux segments (basé sur end time)
            new_segs = [s for s in segs if float(getattr(s, "end", 0.0)) > session.last_sent_end + 1e-3]
            if new_segs:
                new_text = "".join(getattr(s, "text", "") for s in new_segs).strip()
                session.last_sent_end = max(float(getattr(s, "end", 0.0)) for s in new_segs)
                if new_text:
                    await ws.send_json({"type": "partial", "text": new_text, "t_end": session.last_sent_end})
            if final:
                full_text = "".join(getattr(s, "text", "") for s in segs).strip()
                await ws.send_json(
                    {
                        "type": "final",
                        "text": full_text,
                        "language": getattr(info, "language", None),
                        "duration": getattr(info, "duration", None),
                    }
                )
        except Exception as e:
            # En cas d'erreur, on prévient le client sans fermer brutalement
            log.exception(f"[WS Transcribe] inference failed: {e}")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    session = _StreamSession()
    try:
        await ws.send_json({"type": "ready"})
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await ws.send_json({"type": "error", "message": "invalid_json"})
                continue

            mtype = data.get("type")
            if mtype == "start":
                # Paramètres optionnels de session
                lang = data.get("language")
                if isinstance(lang, str) and lang:
                    session.language = lang
                bs = data.get("beam_size")
                if isinstance(bs, int) and 1 <= bs <= 10:
                    session.beam_size = bs
                vad = data.get("vad")
                if isinstance(vad, bool):
                    session.vad = vad
                sr = data.get("sample_rate")
                if isinstance(sr, int) and sr > 0:
                    session.sample_rate = sr
                await ws.send_json({"type": "ack", "message": "started"})
            elif mtype == "audio_chunk":
                # Attendu: base64 de PCM16 mono @ sample_rate
                b64 = data.get("data")
                if not isinstance(b64, str):
                    await ws.send_json({"type": "error", "message": "chunk_missing_data"})
                    continue
                try:
                    chunk = base64.b64decode(b64, validate=True)
                except Exception:
                    await ws.send_json({"type": "error", "message": "invalid_base64"})
                    continue
                # Ajout à la mémoire tampon
                session.buffer.extend(chunk)
                # Inférence opportuniste
                await _maybe_infer(session, ws)
            elif mtype == "flush":
                await _run_infer(session, ws, final=False)
            elif mtype == "end":
                # Dernière inférence + final
                await _run_infer(session, ws, final=True)
                break
            else:
                await ws.send_json({"type": "error", "message": "unknown_type"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception(f"[WS Transcribe] connection error: {e}")
    finally:
        try:
            await ws.close()
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
