"""
chatbot.py
----------
AI analytics chatbot powered by OpenAI.

POST /chat   – accepts a question + optional history, returns an AI reply
              grounded in live database context.

Example questions handled
    "How many people crossed today?"
    "What was the peak traffic hour?"
    "Show me hourly analytics for the last 2 days"
    "Compare in vs out counts"
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from openai import OpenAI, APIError

from api.schemas import ChatRequest, ChatResponse
from config import settings
from core.database import get_hourly_stats, get_peak_hour, get_today_counts, get_latest_events

router = APIRouter(prefix="/chat", tags=["chatbot"])

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail="OpenAI API key not configured. Set OPENAI_API_KEY in .env",
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _build_context() -> str:
    """Fetch fresh analytics from the DB and format as a compact JSON string."""
    today = get_today_counts()
    peak = get_peak_hour(days=7)
    hourly = get_hourly_stats(days=1)
    latest = get_latest_events(limit=5)

    ctx = {
        "today_counts": today,
        "peak_hour_last_7_days": peak,
        "hourly_stats_today": hourly,
        "latest_5_events": latest,
    }
    return json.dumps(ctx, default=str, indent=2)


_SYSTEM_PROMPT = """\
You are an AI surveillance analytics assistant. Your job is to answer
questions about pedestrian traffic data collected from a live camera feed.

You are given LIVE DATA from the database in the context block below.
Use this data to answer the user's question accurately and concisely.
If data is insufficient, say so honestly.

Always include numbers when available. Format counts clearly.
If the user asks for a chart or table, describe the data in text form
(the dashboard renders charts separately).

LIVE DATA CONTEXT:
{context}
"""


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a natural-language analytics question and return an AI answer."""
    client = _get_client()

    context = _build_context()
    system_msg = _SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system_msg}]

    # Replay conversation history (last 10 turns max)
    for turn in request.history[-10:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        reply = response.choices[0].message.content or ""
        logger.info(f"Chatbot replied (tokens={response.usage.total_tokens})")
        return ChatResponse(reply=reply, context_used=context[:200] + "…")

    except APIError as exc:
        logger.error(f"OpenAI API error: {exc}")
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc.message}")
