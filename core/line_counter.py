"""
line_counter.py
---------------
Virtual counting line that records each person crossing exactly once.

Logic
-----
* The line is defined by two points (normalised 0–1 relative to frame size).
* For every tracked detection we track which side of the line the centroid
  is on (sign of the cross-product of the line vector and the centroid vector).
* A crossing is registered when the sign flips AND it has not been registered
  for that tracker_id before.

Events
------
Each crossing yields a `CrossingEvent` dataclass that the pipeline can
persist to the database and broadcast via WebSocket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from config import settings


@dataclass
class CrossingEvent:
    tracker_id: int
    direction: str          # "in" | "out"  (positive→"in", negative→"out")
    object_class: str = "person"
    timestamp: float = field(default_factory=time.time)
    bbox: tuple[float, float, float, float] | None = None   # x1 y1 x2 y2


class LineCounter:
    """
    Counts persons crossing a virtual line drawn across the video frame.

    Parameters
    ----------
    frame_width, frame_height : int
        Dimensions of the video frames (needed to de-normalise coordinates).
    on_crossing : callable, optional
        Called with a `CrossingEvent` for each detected crossing.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        on_crossing: Callable[[CrossingEvent], None] | None = None,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.on_crossing = on_crossing

        # Pixel coordinates of the counting line
        self.pt1 = (
            int(settings.line_start_x * frame_width),
            int(settings.line_start_y * frame_height),
        )
        self.pt2 = (
            int(settings.line_end_x * frame_width),
            int(settings.line_end_y * frame_height),
        )

        # tracker_id → last known side (-1 / 0 / 1)
        self._last_side: dict[int, int] = {}
        # tracker_ids that have already been counted
        self._counted_ids: set[int] = set()

        self.in_count = 0
        self.out_count = 0

        logger.info(
            f"LineCounter initialised: {self.pt1} → {self.pt2} "
            f"(frame {frame_width}×{frame_height})"
        )

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, detections: sv.Detections) -> list[CrossingEvent]:
        """
        Process tracked detections and return any new crossing events.

        Parameters
        ----------
        detections : sv.Detections
            Must have `.tracker_id` populated (run through PersonTracker first).
        """
        if detections.tracker_id is None or len(detections) == 0:
            return []

        events: list[CrossingEvent] = []

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            side = self._side_of_line(cx, cy)

            prev_side = self._last_side.get(tid)
            self._last_side[tid] = side

            # Crossing = sign flip and not yet counted
            if prev_side is not None and prev_side != side and tid not in self._counted_ids:
                self._counted_ids.add(tid)
                direction = "in" if side > 0 else "out"
                if direction == "in":
                    self.in_count += 1
                else:
                    self.out_count += 1

                event = CrossingEvent(
                    tracker_id=tid,
                    direction=direction,
                    object_class=self._class_name(detections, i),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
                events.append(event)
                logger.info(
                    f"Crossing #{tid} → {direction} "
                    f"(total in={self.in_count} out={self.out_count})"
                )
                if self.on_crossing:
                    self.on_crossing(event)

        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the counting line and live counters on *frame* (copy)."""
        out = frame.copy()
        # Line
        cv2.line(out, self.pt1, self.pt2, (0, 255, 255), 3)
        # Arrow to indicate direction
        mid = (
            (self.pt1[0] + self.pt2[0]) // 2,
            (self.pt1[1] + self.pt2[1]) // 2,
        )
        cv2.arrowedLine(
            out, mid, (mid[0], mid[1] - 30), (0, 255, 255), 2, tipLength=0.4
        )
        # Counters
        cv2.rectangle(out, (10, 10), (200, 70), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"IN : {self.in_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"OUT: {self.out_count}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out

    def reset_counts(self) -> None:
        """Reset counters (e.g. at midnight for daily stats)."""
        self.in_count = 0
        self.out_count = 0
        self._counted_ids.clear()
        self._last_side.clear()
        logger.info("LineCounter counts reset.")

    # ── private helpers ───────────────────────────────────────────────────────

    def _side_of_line(self, px: int, py: int) -> int:
        """
        Cross-product sign: which side of the line is point (px, py) on?
        Returns +1 or -1 (never 0 because we use sign only).
        """
        ax, ay = self.pt1
        bx, by = self.pt2
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        return 1 if cross >= 0 else -1

    @staticmethod
    def _class_name(detections: sv.Detections, index: int) -> str:
        names = {0: "person", 2: "car", 3: "motorcycle"}
        if detections.class_id is None:
            return "person"
        return names.get(int(detections.class_id[index]), "object")
