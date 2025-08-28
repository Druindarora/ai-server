from typing import Optional, List, Literal
from pydantic import BaseModel

# GET /ollama/ping
class OllamaPingResponse(BaseModel):
    ok: bool
    service: Literal["ollama"]

# POST /ollama/generate
class OllamaGenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None

class OllamaGenerateResponse(BaseModel):
    response: str

# (facultatif) pour un éventuel endpoint /ollama/status
class OllamaServiceStatus(BaseModel):
    available: bool
    state: str
    models: List[str]
