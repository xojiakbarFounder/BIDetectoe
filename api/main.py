"""
main.py
-------
FastAPI application entry point.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.database import init_db
from api.routes import analytics, chatbot, events, video


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Surveillance Analytics API …")
    init_db()
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Surveillance Analytics API",
    description=(
        "Real-time pedestrian counting system powered by YOLOv8 + ByteTrack.\n\n"
        "Endpoints:\n"
        "- `/analytics/*`  – pre-aggregated stats & live counters\n"
        "- `/events/*`     – crossing events + WebSocket live feed\n"
        "- `/chat`         – AI chatbot analytics (OpenAI)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Streamlit dashboard on any origin in dev) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analytics.router)
app.include_router(events.router)
app.include_router(chatbot.router)
app.include_router(video.router)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "surveillance-analytics-api"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}
