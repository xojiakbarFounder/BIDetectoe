"""
events.py
---------
REST + WebSocket endpoints for crossing events.

GET  /events/latest              – most recent N crossing events
WS   /events/ws                  – WebSocket stream of live events
POST /events/internal/push       – called by the pipeline to push a new event
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from loguru import logger

from api.schemas import CrossingEventOut
from api.state import live_state
from core.database import get_latest_events, save_crossing_event

router = APIRouter(prefix="/events", tags=["events"])


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info(f"WS client connected (total={len(self._active)})")

    def disconnect(self, ws: WebSocket) -> None:
        self._active.remove(ws)
        logger.info(f"WS client disconnected (total={len(self._active)})")

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._active.remove(ws)


manager = ConnectionManager()


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.get("/latest", response_model=list[CrossingEventOut])
def latest_events(limit: int = Query(default=20, ge=1, le=200)):
    """Return the most recent crossing events from the database."""
    rows = get_latest_events(limit=limit)
    return [CrossingEventOut(**r) for r in rows]


@router.post("/internal/push", include_in_schema=False)
async def push_event(payload: dict):
    """Pipeline dan kelgan crossing eventni saqlaydi va broadcast qiladi."""
    event = save_crossing_event(
        tracker_id=payload["tracker_id"],
        direction=payload["direction"],
        bbox=payload.get("bbox"),
    )
    # live_state ni shu jarayonda yangilaymiz
    direction = payload["direction"]
    with live_state._lock:
        if direction == "in":
            live_state.in_count += 1
        else:
            live_state.out_count += 1
        live_state.total_count = live_state.in_count + live_state.out_count
    await manager.broadcast(payload)
    return {"status": "ok"}


@router.post("/internal/stats", include_in_schema=False)
async def update_stats(payload: dict):
    """Pipeline dan FPS va active_tracks qiymatlarini qabul qiladi."""
    live_state.update(
        in_count=live_state.in_count,
        out_count=live_state.out_count,
        active_tracks=payload.get("active_tracks", 0),
        fps=payload.get("fps", 0.0),
    )
    return {"status": "ok"}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws")
async def events_websocket(websocket: WebSocket):
    """
    Clients subscribe here to receive live crossing events as JSON objects:

        { "tracker_id": 42, "direction": "in", "timestamp": "…" }
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; actual data is pushed via broadcast()
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
