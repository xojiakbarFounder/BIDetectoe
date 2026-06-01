"""
utils.py
--------
Thin wrapper around the FastAPI backend for the Streamlit dashboard.
"""

from __future__ import annotations

import os
import requests
from loguru import logger

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def _get(path: str, **params) -> dict | list | None:
    try:
        r = requests.get(f"{_API_BASE}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"API GET {path} failed: {exc}")
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{_API_BASE}{path}", json=payload, timeout=95)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"API POST {path} failed: {exc}")
        return None


# ── Analytics helpers ─────────────────────────────────────────────────────────

def fetch_live_state() -> dict:
    data = _get("/analytics/live")
    return data or {"in_count": 0, "out_count": 0, "total_count": 0, "active_tracks": 0, "fps": 0.0, "timestamp": "—"}


def fetch_today_counts(date: str | None = None) -> dict:
    params = {"date": date} if date else {}
    data = _get("/analytics/today", **params)
    return data or {"in_count": 0, "out_count": 0, "total": 0}


def fetch_hourly_stats(date: str | None = None) -> list[dict]:
    params = {"date": date} if date else {}
    data = _get("/analytics/hourly", **params)
    return data or []


def fetch_peak_hour(days: int = 7) -> dict | None:
    return _get("/analytics/peak-hour", days=days)


def fetch_latest_events(limit: int = 20) -> list[dict]:
    data = _get("/events/latest", limit=limit)
    return data or []


# ── Chatbot helper ────────────────────────────────────────────────────────────

def send_chat(message: str, history: list[dict]) -> str:
    data = _post("/chat", {"message": message, "history": history})
    if data:
        return data.get("reply", "No reply received.")
    return "Error contacting the AI backend. Is the API running?"
