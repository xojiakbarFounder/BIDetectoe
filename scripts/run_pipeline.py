#!/usr/bin/env python
"""
run_pipeline.py
---------------
Main entry point for the detection pipeline.

Architecture
    YouTube Stream
        ↓  (yt-dlp + OpenCV)
    StreamHandler
        ↓  (BGR frames @ target_fps)
    PersonDetector   (YOLOv8)
        ↓  (sv.Detections with bboxes + confidence)
    PersonTracker    (ByteTrack)
        ↓  (sv.Detections with tracker_id)
    LineCounter
        ↓  (CrossingEvent on each sign-flip)
    PostgreSQL       (save_crossing_event)
    +
    LiveState        (in-process KPI store for /analytics/live)
    +
    FastAPI push     (POST /events/internal/push → WebSocket broadcast)

Run
---
    python scripts/run_pipeline.py

Environment variables are read from .env (see .env.example).
"""

from __future__ import annotations

import sys
import os
import time
import threading
from collections import deque

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import httpx
from loguru import logger

from config import settings
from core.database import init_db
from core.detector import PersonDetector
from core.line_counter import CrossingEvent, LineCounter
from core.stream_handler import StreamHandler
from core.tracker import PersonTracker
from api.state import live_state
# ── FPS estimator ─────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.monotonic())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ── Async event pusher (fire-and-forget) ──────────────────────────────────────

_http_client = httpx.Client(base_url=f"http://localhost:{settings.api_port}", timeout=2.0)


class FramePusher:
    def __init__(self, client: httpx.Client, max_fps: float = 6.0) -> None:
        self.client = client
        self.interval = 1.0 / max(max_fps, 1.0)
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._latest = jpeg_bytes

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _pop_latest(self) -> bytes | None:
        with self._lock:
            data = self._latest
            self._latest = None
            return data

    def _run(self) -> None:
        while not self._stop.is_set():
            jpeg_bytes = self._pop_latest()
            if jpeg_bytes:
                try:
                    self.client.post(
                        "/video/internal/frame",
                        content=jpeg_bytes,
                        headers={"Content-Type": "image/jpeg"},
                    )
                except Exception as exc:
                    logger.debug(f"Frame push xatosi (muhim emas): {exc}")
            time.sleep(self.interval)


def _push_event(event: CrossingEvent) -> None:
    """API ga crossing event yuboradi (DB ga saqlash + broadcast)."""
    def _do():
        try:
            _http_client.post(
                "/events/internal/push",
                json={
                    "tracker_id": event.tracker_id,
                    "direction": event.direction,
                    "object_class": event.object_class,
                    "timestamp": event.timestamp,
                    "bbox": list(event.bbox) if event.bbox else None,
                },
            )
        except Exception as exc:
            logger.debug(f"API push xatosi (muhim emas): {exc}")
    threading.Thread(target=_do, daemon=True).start()


def _push_stats(active_tracks: int, fps: float) -> None:
    """API ga FPS va hozirgi track sonini yuboradi."""
    def _do():
        try:
            _http_client.post(
                "/events/internal/stats",
                json={"active_tracks": active_tracks, "fps": fps},
            )
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


def _push_frame(jpeg_bytes: bytes) -> None:
    """API processiga eng yangi annotated JPEG kadrni yuboradi."""
    try:
        _http_client.post(
            "/video/internal/frame",
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
    except Exception as exc:
        logger.debug(f"Frame push xatosi (muhim emas): {exc}")


# ── Main pipeline loop ────────────────────────────────────────────────────────

def run() -> None:
    logger.info("Initialising database …")
    init_db()

    logger.info("Loading detector …")
    detector = PersonDetector()

    logger.info("Loading tracker …")
    tracker = PersonTracker(frame_rate=settings.target_fps)

    fps_counter = FPSCounter()
    line_counter: LineCounter | None = None
    display_window = os.getenv("DISPLAY_WINDOW", "false").lower() in {"1", "true", "yes"}
    frame_pusher = FramePusher(_http_client, max_fps=6.0)
    frame_pusher.start()

    logger.info(f"Opening stream: {settings.youtube_stream_url}")

    with StreamHandler(
        url=settings.youtube_stream_url,
        target_fps=settings.target_fps,
    ) as stream:
        for frame_idx, frame in enumerate(stream):
            h, w = frame.shape[:2]

            # Initialise line counter once we know the frame size
            if line_counter is None:
                line_counter = LineCounter(frame_width=w, frame_height=h)

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            tracked = tracker.update(detections)

            # 3. Count crossings
            events = line_counter.update(tracked)

            # 4. Crossingni API ga yuborish (API DB ga saqlaydi)
            for ev in events:
                _push_event(ev)

            # 5. Har 30 kadrda FPS + active tracks ni API ga yuborish
            fps = fps_counter.tick()
            if frame_idx % 30 == 0:
                _push_stats(active_tracks=len(tracked), fps=fps)

            # 6. Annotate frame, show window + push to web stream
            labels = PersonTracker.build_labels(tracked)
            annotated = PersonDetector.annotate(frame, tracked, labels=labels)
            annotated = line_counter.draw(annotated)

            if display_window:
                cv2.imshow("Surveillance Pipeline [Q - quit]", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Q bosildi - to'xtatildi.")
                    break

            ok, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                jpeg_bytes = jpeg_buf.tobytes()
                live_state.set_frame(jpeg_bytes)
                frame_pusher.submit(jpeg_bytes)

            if frame_idx % 100 == 0:
                logger.info(
                    f"[frame {frame_idx}] FPS={fps:.1f} | "
                    f"detections={len(detections)} | tracks={len(tracked)} | "
                    f"IN={line_counter.in_count} OUT={line_counter.out_count}"
                )

    cv2.destroyAllWindows()
    frame_pusher.stop()
    logger.info("Pipeline stopped.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        _http_client.close()
