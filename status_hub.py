# === FICHIER : status_hub.py ===
from __future__ import annotations

from typing import Any, Set
from fastapi import WebSocket

class StatusEventHub:
    """Hub WS simple pour diffuser l'état aux clients."""
    def __init__(self) -> None:
        self.activeConnections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.activeConnections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.activeConnections.discard(websocket)

    async def broadcast(self, status: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in tuple(self.activeConnections):
            try:
                await ws.send_json(status)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.activeConnections.discard(ws)
