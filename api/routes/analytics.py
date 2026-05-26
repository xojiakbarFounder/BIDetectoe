from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Query
from api.schemas import DailyCountsOut, HourlyStatOut, LiveStateOut, PeakHourOut
from api.state import live_state
from core.database import get_day_counts, get_hourly_stats, get_peak_hour

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/today", response_model=DailyCountsOut)
def today_counts(date: str | None = Query(default=None, description="YYYY-MM-DD")):
    d = dt.date.fromisoformat(date) if date else None
    data = get_day_counts(d)
    return DailyCountsOut(**data)


@router.get("/hourly", response_model=list[HourlyStatOut])
def hourly_stats(date: str | None = Query(default=None, description="YYYY-MM-DD")):
    d = dt.date.fromisoformat(date) if date else None
    rows = get_hourly_stats(d)
    return [HourlyStatOut(**r) for r in rows]


@router.get("/peak-hour", response_model=PeakHourOut | None)
def peak_hour(days: int = Query(default=7, ge=1, le=30)):
    data = get_peak_hour(days=days)
    return PeakHourOut(**data) if data else None


@router.get("/live", response_model=LiveStateOut)
def live_counts():
    return LiveStateOut(**live_state.snapshot())
