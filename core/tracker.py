"""
tracker.py
----------
Wraps supervision's ByteTrack implementation.

ByteTrack assigns a persistent integer `tracker_id` to each detection
across frames so the line-counter can recognise when the *same* person
crosses the virtual line.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from loguru import logger

_CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
}


class PersonTracker:
    """
    Multi-object tracker using ByteTrack.

    ByteTrack is available directly inside the `supervision` package
    (sv.ByteTracker) so no separate C++ build is required.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 10,
    ) -> None:
        logger.info("Initialising ByteTrack …")
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        logger.info("ByteTrack ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Feed detections for the current frame and get back detections
        enriched with `tracker_id` integers.

        Parameters
        ----------
        detections : sv.Detections
            Raw detections from PersonDetector (no tracker_id yet).

        Returns
        -------
        sv.Detections
            Same detections with `.tracker_id` populated.
        """
        if len(detections) == 0:
            return detections

        tracked = self._tracker.update_with_detections(detections)
        return tracked

    def reset(self) -> None:
        """Re-initialise the tracker (e.g. on stream reconnection)."""
        self._tracker.reset()
        logger.info("ByteTrack state reset.")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def build_labels(detections: sv.Detections) -> list[str]:
        """Return display labels like 'person #42 (0.87)' for each detection."""
        labels = []
        for i in range(len(detections)):
            tid = (
                detections.tracker_id[i]
                if detections.tracker_id is not None
                else "?"
            )
            class_name = "object"
            if detections.class_id is not None:
                class_name = _CLASS_NAMES.get(int(detections.class_id[i]), "object")
            conf = (
                f"{detections.confidence[i]:.2f}"
                if detections.confidence is not None
                else ""
            )
            labels.append(f"{class_name} #{tid} ({conf})")
        return labels
