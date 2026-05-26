"""
database.py
-----------
SQLAlchemy ORM models, engine factory, and helper functions.

Tables
------
crossing_events  – one row per person-crossing (tracker_id, direction, bbox …)
hourly_stats     – pre-aggregated counts per hour (maintained by a trigger or
                   by the pipeline itself)
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


# ── Engine / session factory ──────────────────────────────────────────────────

def _make_engine():
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context-manager that yields a DB session and handles commit/rollback."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── ORM models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class CrossingEvent(Base):
    """One record per person crossing the virtual line."""

    __tablename__ = "crossing_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tracker_id = Column(Integer, nullable=False, index=True)
    direction = Column(String(4), nullable=False)          # "in" | "out"
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        index=True,
    )
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<CrossingEvent id={self.id} tracker={self.tracker_id} "
            f"dir={self.direction} ts={self.timestamp}>"
        )


class HourlyStats(Base):
    """
    Pre-aggregated counts per UTC hour.
    Updated incrementally by `upsert_hourly_stats`.
    """

    __tablename__ = "hourly_stats"
    __table_args__ = (UniqueConstraint("hour_bucket", name="uq_hour_bucket"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    hour_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
    in_count = Column(Integer, nullable=False, default=0)
    out_count = Column(Integer, nullable=False, default=0)
    total_count = Column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"<HourlyStats bucket={self.hour_bucket} "
            f"in={self.in_count} out={self.out_count}>"
        )


# ── Init / migrations ─────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist yet."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def save_crossing_event(
    tracker_id: int,
    direction: str,
    timestamp: dt.datetime | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> CrossingEvent:
    """Persist a crossing event and update the hourly stats bucket."""
    if timestamp is None:
        timestamp = dt.datetime.now(dt.timezone.utc)

    with get_db() as db:
        event = CrossingEvent(
            tracker_id=tracker_id,
            direction=direction,
            timestamp=timestamp,
            bbox_x1=bbox[0] if bbox else None,
            bbox_y1=bbox[1] if bbox else None,
            bbox_x2=bbox[2] if bbox else None,
            bbox_y2=bbox[3] if bbox else None,
        )
        db.add(event)
        db.flush()  # get the id before commit

        _upsert_hourly_stats(db, timestamp, direction)
        logger.debug(f"Saved crossing event id={event.id}")
        return event


def _upsert_hourly_stats(
    db: Session, timestamp: dt.datetime, direction: str
) -> None:
    """Increment the appropriate hourly bucket (upsert pattern)."""
    bucket = timestamp.replace(minute=0, second=0, microsecond=0)
    row = db.query(HourlyStats).filter(HourlyStats.hour_bucket == bucket).first()
    if row is None:
        row = HourlyStats(hour_bucket=bucket, in_count=0, out_count=0, total_count=0)
        db.add(row)

    if direction == "in":
        row.in_count += 1
    else:
        row.out_count += 1
    row.total_count = row.in_count + row.out_count


# ── Timezone helpers ──────────────────────────────────────────────────────────

def _local_tz() -> dt.timezone:
    from config import settings
    return dt.timezone(dt.timedelta(hours=settings.utc_offset_hours))


def _to_local(ts: dt.datetime) -> dt.datetime:
    return ts.astimezone(_local_tz())


def _local_now() -> dt.datetime:
    return dt.datetime.now(_local_tz())


# ── Query helpers used by the API / dashboard ─────────────────────────────────

def get_day_counts(date: dt.date | None = None) -> dict:
    """Return in/out/total for a specific local date (default: today)."""
    tz = _local_tz()
    if date is None:
        date = _local_now().date()
    day_start = dt.datetime(date.year, date.month, date.day, tzinfo=tz)
    day_end   = day_start + dt.timedelta(days=1)
    with get_db() as db:
        rows = (
            db.query(
                CrossingEvent.direction,
                func.count(CrossingEvent.id).label("cnt"),
            )
            .filter(CrossingEvent.timestamp >= day_start,
                    CrossingEvent.timestamp <  day_end)
            .group_by(CrossingEvent.direction)
            .all()
        )
    counts = {"in_count": 0, "out_count": 0, "total": 0}
    for direction, cnt in rows:
        if direction == "in":
            counts["in_count"] = cnt
        else:
            counts["out_count"] = cnt
        counts["total"] += cnt
    return counts


# kept for backward-compat
def get_today_counts() -> dict:
    return get_day_counts()


def get_hourly_stats(date: dt.date | None = None) -> list[dict]:
    """Return hourly stats for a specific local date (default: today)."""
    tz = _local_tz()
    if date is None:
        date = _local_now().date()
    day_start = dt.datetime(date.year, date.month, date.day, tzinfo=tz)
    day_end   = day_start + dt.timedelta(days=1)
    with get_db() as db:
        rows = (
            db.query(HourlyStats)
            .filter(HourlyStats.hour_bucket >= day_start,
                    HourlyStats.hour_bucket <  day_end)
            .order_by(HourlyStats.hour_bucket)
            .all()
        )
        return [
            {
                "hour": _to_local(r.hour_bucket).strftime("%Y-%m-%d %H:%M:%S"),
                "in_count": r.in_count,
                "out_count": r.out_count,
                "total_count": r.total_count,
            }
            for r in rows
        ]


def get_peak_hour(days: int = 7) -> dict | None:
    """Return the hour bucket with the highest total crossings in last N days."""
    since = _local_now() - dt.timedelta(days=days)
    with get_db() as db:
        row = (
            db.query(HourlyStats)
            .filter(HourlyStats.hour_bucket >= since)
            .order_by(HourlyStats.total_count.desc())
            .first()
        )
        if row is None:
            return None
        return {
            "hour": _to_local(row.hour_bucket).strftime("%Y-%m-%d %H:%M:%S"),
            "in_count": row.in_count,
            "out_count": row.out_count,
            "total_count": row.total_count,
        }


def get_latest_events(limit: int = 20) -> list[dict]:
    """Return the most recent crossing events (local time)."""
    with get_db() as db:
        rows = (
            db.query(CrossingEvent)
            .order_by(CrossingEvent.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "tracker_id": r.tracker_id,
                "direction": r.direction,
                "timestamp": _to_local(r.timestamp).isoformat(),
            }
            for r in rows
        ]
