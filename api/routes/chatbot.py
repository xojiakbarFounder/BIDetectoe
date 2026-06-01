"""
chatbot.py
----------
AI analytics chatbot powered by Gemini.

POST /chat - accepts a question + optional history, returns an AI reply
             grounded in live database context.

Example questions handled
    "How many people crossed today?"
    "What was the peak traffic hour?"
    "Show me hourly analytics for the last 2 days"
    "Compare in vs out counts"
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re

import httpx
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import ChatRequest, ChatResponse
from config import settings
from core.database import (
    get_day_counts,
    get_events_around_timestamp,
    get_events_for_day,
    get_hourly_stats,
    get_latest_events,
    get_latest_event_date_by_day,
    get_peak_hour,
    get_today_counts,
)

router = APIRouter(prefix="/chat", tags=["chatbot"])

_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_GEMINI_TIMEOUT_SECONDS = 90.0
_GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DAY_RE = re.compile(r"\b([1-9]|[12]\d|3[01])\b")
_REPORT_WORDS = ("malumot", "ma'lumot", "maʼlumot", "hisobot", "to'liq", "toliq", "to‘liq", "haqida")


def _get_api_key() -> str:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key not configured. Set GEMINI_API_KEY in .env",
        )
    return settings.gemini_api_key


def _extract_timestamps(text: str) -> list[dt.datetime]:
    timestamps = []
    for match in _TIMESTAMP_RE.findall(text):
        normalized = match.replace("T", " ")
        try:
            timestamps.append(dt.datetime.fromisoformat(normalized))
        except ValueError:
            continue
    return timestamps


def _extract_report_date(text: str) -> dt.date | None:
    lowered = text.lower()
    if not any(word in lowered for word in _REPORT_WORDS):
        return None

    date_match = _DATE_RE.search(text)
    if date_match:
        try:
            return dt.date.fromisoformat(date_match.group(0))
        except ValueError:
            return None

    day_match = _DAY_RE.search(text)
    if not day_match:
        return None

    day = int(day_match.group(1))
    return get_latest_event_date_by_day(day)


def _build_timestamp_searches(user_message: str) -> list[dict]:
    timestamp_searches = []
    for timestamp in _extract_timestamps(user_message):
        timestamp_searches.append(
            {
                "requested_time": timestamp.isoformat(sep=" "),
                "window_seconds": 180,
                "matching_events": get_events_around_timestamp(
                    timestamp,
                    window_seconds=180,
                    limit=20,
                ),
            }
        )
    return timestamp_searches


def _direction_label(direction: str) -> str:
    return "kirish" if direction == "in" else "chiqish"


def _answer_timestamp_search(timestamp_searches: list[dict]) -> str | None:
    if not timestamp_searches:
        return None

    lines = []
    for search in timestamp_searches:
        requested_time = search["requested_time"]
        events = search["matching_events"]
        if not events:
            lines.append(f"{requested_time} atrofida bazadan kirish/chiqish topilmadi.")
            continue

        exact_events = [
            event for event in events
            if abs(event["seconds_from_requested_time"]) < 1
        ]
        selected_events = exact_events or events[:3]

        if exact_events:
            lines.append(f"{requested_time} vaqtida bazada {len(exact_events)} ta hodisa topildi:")
        else:
            closest = events[0]
            seconds = abs(closest["seconds_from_requested_time"])
            lines.append(
                f"{requested_time} aniq sekundida hodisa topilmadi. "
                f"Eng yaqini {seconds:.1f} soniya farq bilan:"
            )

        for event in selected_events:
            seconds = event["seconds_from_requested_time"]
            sign = "+" if seconds >= 0 else ""
            lines.append(
                "- "
                f"{event['timestamp']} - {_direction_label(event['direction'])} "
                f"(tracker_id={event['tracker_id']}, {sign}{seconds:.3f}s)"
            )

    return "\n".join(lines)


def _answer_day_report(date: dt.date | None) -> str | None:
    if date is None:
        return None

    counts = get_day_counts(date)
    hourly = get_hourly_stats(date)
    events = get_events_for_day(date, limit=12)

    if counts["total"] == 0 and not hourly and not events:
        return f"{date.isoformat()} kuni uchun database’da kirish/chiqish ma’lumoti topilmadi."

    lines = [
        f"{date.isoformat()} kuni bo‘yicha ma’lumot:",
        f"- Kirish: {counts['in_count']} ta",
        f"- Chiqish: {counts['out_count']} ta",
        f"- Jami: {counts['total']} ta",
    ]

    if hourly:
        peak = max(hourly, key=lambda row: row["total_count"])
        lines.extend([
            "",
            "Soatlik statistika:",
            *[
                f"- {row['hour']}: kirish {row['in_count']}, chiqish {row['out_count']}, jami {row['total_count']}"
                for row in hourly
            ],
            "",
            (
                "Eng gavjum soat: "
                f"{peak['hour']} - jami {peak['total_count']} "
                f"(kirish {peak['in_count']}, chiqish {peak['out_count']})"
            ),
        ])

    if events:
        lines.extend([
            "",
            "Dastlabki hodisalar:",
            *[
                f"- {event['timestamp']} - {_direction_label(event['direction'])} "
                f"(tracker_id={event['tracker_id']})"
                for event in events
            ],
        ])

    return "\n".join(lines)


def _build_context(user_message: str = "", timestamp_searches: list[dict] | None = None) -> str:
    """Fetch fresh analytics from the DB and format as a compact JSON string."""
    today = get_today_counts()
    peak = get_peak_hour(days=7)
    hourly = get_hourly_stats()
    latest = get_latest_events(limit=20)

    ctx = {
        "today_counts": today,
        "peak_hour_last_7_days": peak,
        "hourly_stats_today": hourly,
        "latest_20_events": latest,
        "timestamp_searches": timestamp_searches if timestamp_searches is not None else _build_timestamp_searches(user_message),
    }
    return json.dumps(ctx, default=str, indent=2)


async def _call_gemini(api_key: str, payload: dict, model: str) -> dict:
    async with httpx.AsyncClient(timeout=_GEMINI_TIMEOUT_SECONDS, trust_env=False) as client:
        response = await client.post(
            _GEMINI_API_URL.format(model=model),
            params={"key": api_key},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def _generate_with_retry(api_key: str, payload: dict) -> tuple[dict, str]:
    models = [settings.gemini_model]
    if settings.gemini_model != _GEMINI_FALLBACK_MODEL:
        models.append(_GEMINI_FALLBACK_MODEL)

    last_exc: Exception | None = None
    for model in models:
        for attempt in range(3):
            try:
                return await _call_gemini(api_key, payload, model), model
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code
                if status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                logger.warning(
                    f"Gemini model {model} returned {status_code}; retry {attempt + 1}/3"
                )
                await asyncio.sleep(1.5 * (attempt + 1))
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    f"Gemini model {model} timed out; retry {attempt + 1}/3"
                )
                await asyncio.sleep(1.5 * (attempt + 1))

    if last_exc:
        raise last_exc
    raise HTTPException(status_code=502, detail="Gemini did not return a response.")


_SYSTEM_PROMPT = """\
You are an AI surveillance analytics assistant. Your job is to answer
questions about pedestrian traffic data collected from a live camera feed.

You are given LIVE DATA from the database in the context block below.
Use this data to answer the user's question accurately and concisely.
If data is insufficient, say so honestly.
If timestamp_searches contains matching_events, use those results to answer
questions about the requested time. Direction "in" means kirish and "out"
means chiqish. If there is no exact second match, report the closest event
and how many seconds away it is.

Always include numbers when available. Format counts clearly.
If the user asks for a chart or table, describe the data in text form
(the dashboard renders charts separately).

LIVE DATA CONTEXT:
{context}
"""


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a natural-language analytics question and return an AI answer."""
    timestamp_searches = _build_timestamp_searches(request.message)
    direct_reply = _answer_timestamp_search(timestamp_searches)
    if direct_reply:
        return ChatResponse(reply=direct_reply, context_used=json.dumps(timestamp_searches, default=str)[:200] + "...")

    report_date = _extract_report_date(request.message)
    direct_reply = _answer_day_report(report_date)
    if direct_reply:
        return ChatResponse(reply=direct_reply, context_used=f"date_report={report_date}")

    api_key = _get_api_key()
    context = _build_context(request.message, timestamp_searches)
    system_instruction = _SYSTEM_PROMPT.format(context=context)

    prompt_parts = []
    for turn in request.history[-10:]:
        if turn.get("role") in ("user", "assistant"):
            role = "Assistant" if turn["role"] == "assistant" else "User"
            prompt_parts.append(f"{role}: {turn['content']}")

    prompt_parts.append(f"User: {request.message}")
    prompt_parts.append("Assistant:")
    prompt = "\n\n".join(prompt_parts)

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600,
        },
    }

    try:
        data, model_used = await _generate_with_retry(api_key, payload)
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        reply = "".join(part.get("text", "") for part in parts).strip()
        logger.info(f"Chatbot replied via Gemini model={model_used}")
        return ChatResponse(reply=reply, context_used=context[:200] + "...")

    except httpx.HTTPStatusError as exc:
        logger.error(f"Gemini API error: {exc.response.text}")
        raise HTTPException(status_code=502, detail=f"Gemini error: {exc.response.text}")
    except httpx.TimeoutException as exc:
        logger.error(f"Gemini API timeout: {type(exc).__name__}: {exc!r}")
        raise HTTPException(status_code=504, detail="Gemini request timed out. Please try again.")
    except httpx.HTTPError as exc:
        logger.error(f"Gemini API error: {type(exc).__name__}: {exc!r}")
        raise HTTPException(status_code=502, detail=f"Gemini error: {type(exc).__name__}: {exc}")
