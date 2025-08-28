from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from api import whisper, ollama

logger = logging.getLogger("ai-server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("== AI Backend starting ==")
    try:
        w = whisper.get_status()
        o = ollama.get_status()
        logger.info(
            f"startup status: whisper={w.get('state')} "
            f"device={w.get('device')} compute={w.get('compute_type')}; "
            f"ollama={o.get('state')} models={len(o.get('models', []))}"
        )
    except Exception:
        logger.exception("Failed to fetch startup status")

    # yield => lance le serveur
    yield

    logger.info("== AI Backend stopping ==")
