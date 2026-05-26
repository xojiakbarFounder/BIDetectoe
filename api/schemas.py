"""
schemas.py
----------
Pydantic v2 request / response models for the FastAPI routes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Crossing events ───────────────────────────────────────────────────────────

class CrossingEventOut(BaseModel):
    id: int
    tracker_id: int
    direction: str
    timestamp: datetime

    model_config = {"from_attributes": True}


# ── Daily counts ──────────────────────────────────────────────────────────────

class DailyCountsOut(BaseModel):
    in_count: int
    out_count: int
    total: int


# ── Hourly stats ──────────────────────────────────────────────────────────────

class HourlyStatOut(BaseModel):
    hour: str
    in_count: int
    out_count: int
    total_count: int


# ── Peak hour ─────────────────────────────────────────────────────────────────

class PeakHourOut(BaseModel):
    hour: str
    in_count: int
    out_count: int
    total_count: int


# ── Live state (polled by dashboard) ─────────────────────────────────────────

class LiveStateOut(BaseModel):
    in_count: int
    out_count: int
    total_count: int
    active_tracks: int
    fps: float
    timestamp: str


# ── Chatbot ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    context_used: str
