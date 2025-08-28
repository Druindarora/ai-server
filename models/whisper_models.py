from typing import List, Literal
from pydantic import BaseModel

# GET /whisper/ping
class WhisperPingResponse(BaseModel):
    ok: bool
    service: Literal["whisper"]

# GET /whisper/models
class WhisperModelsFull(BaseModel):
    downloaded: List[str]
    all: List[str]
    current: str

# POST /whisper/select
AllowedWhisperName = Literal["tiny", "base", "small", "medium"]

class WhisperSelectRequest(BaseModel):
    name: AllowedWhisperName

class WhisperSelectResponse(BaseModel):
    requested: AllowedWhisperName
    current: AllowedWhisperName
    state: Literal["ready", "loading"]

# POST /whisper/transcribe
class WhisperTranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
