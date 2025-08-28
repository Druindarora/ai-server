from pydantic import BaseModel
from typing import List

class WhisperModels(BaseModel):
    downloaded: List[str]
    current: str

class WhisperStatus(BaseModel):
    available: bool
    state: str
    models: WhisperModels
    device: str
    compute_type: str

class OllamaStatus(BaseModel):
    available: bool
    state: str
    models: List[str]

class ServicesStatus(BaseModel):
    whisper: WhisperStatus
    ollama: OllamaStatus

class StatusResponse(BaseModel):
    overall: str
    timestamp: str
    services: ServicesStatus
