"""
state.py
--------
In-process live state store shared between the pipeline and the API.

When the pipeline and API run in the *same* process (default for development),
this module holds the up-to-the-second counters in memory.

In a multi-process deployment (e.g. Docker), the pipeline writes to the DB
and the /analytics/live endpoint reads from the DB instead.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import asdict, dataclass, field


@dataclass
class LiveState:
    in_count: int = 0
    out_count: int = 0
    total_count: int = 0
    category_counts: dict[str, int] = field(
        default_factory=lambda: {"person": 0, "car": 0, "motorcycle": 0}
    )
    active_tracks: int = 0
    fps: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _frame_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _latest_frame: bytes | None = field(default=None, repr=False, compare=False)

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self._frame_lock:
            self._latest_frame = jpeg_bytes

    def get_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame

    def update(
        self,
        in_count: int,
        out_count: int,
        active_tracks: int,
        fps: float,
        category_counts: dict[str, int] | None = None,
    ) -> None:
        with self._lock:
            self.in_count = in_count
            self.out_count = out_count
            self.total_count = in_count + out_count
            if category_counts is not None:
                self.category_counts = category_counts
            self.active_tracks = active_tracks
            self.fps = fps

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "in_count": self.in_count,
                "out_count": self.out_count,
                "total_count": self.total_count,
                "category_counts": dict(self.category_counts),
                "active_tracks": self.active_tracks,
                "fps": round(self.fps, 1),
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }


# Module-level singleton
live_state = LiveState()
