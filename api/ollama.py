import os
import time
import logging
from typing import List

import requests
from fastapi import APIRouter, HTTPException

from models.ollama_models import (
    OllamaPingResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
)

router: APIRouter = APIRouter()
log = logging.getLogger("ai-server.ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

@router.get("/ping", response_model=OllamaPingResponse)
def ping():
    return {"ok": True, "service": "ollama"}

# --- Helpers ---
def _list_installed() -> List[str]:
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        dur = (time.perf_counter() - t0) * 1000
        if r.ok and r.content:
            data = r.json()
            models = [str(m["name"]) for m in data.get("models", []) if isinstance(m, dict) and "name" in m]
            log.info(f"/api/tags -> {len(models)} models in {dur:.1f}ms")
            return models
        log.warning(f"/api/tags HTTP {r.status_code} in {dur:.1f}ms")
    except Exception as e:
        log.warning(f"/api/tags failed: {e}")
    return []

def get_status():
    try:
        models = _list_installed()
        state = "ready" if models else "waiting"
        return {"available": True, "state": state, "models": models}
    except Exception as e:
        log.exception(f"status failed: {e}")
        return {"available": False, "state": "down", "error": str(e)}

# --- Endpoint ---
@router.post("/generate", response_model=OllamaGenerateResponse)
def generate(req: OllamaGenerateRequest):
    t0 = time.perf_counter()

    model = req.model
    if not model:
        raise HTTPException(status_code=400, detail="Champ 'model' manquant (aucun modèle par défaut défini).")

    installed = _list_installed()
    if model not in installed:
        raise HTTPException(status_code=404, detail=f"Modèle inconnu: {model}")

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": req.prompt, "stream": False},
            timeout=120,
        )
        dur = (time.perf_counter() - t0) * 1000
        if not r.ok:
            log.warning(f"/api/generate HTTP {r.status_code} in {dur:.1f}ms")
            raise HTTPException(status_code=500, detail=f"Ollama erreur HTTP {r.status_code}")
        data = r.json() if r.content else {}
        text = data.get("response", "")
        log.info(f"generate model={model} len(prompt)={len(req.prompt)} -> {len(text)} chars in {dur:.1f}ms")
        return {"response": text}
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"generate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération: {e}")
