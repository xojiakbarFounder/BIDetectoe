"""
detector.py
-----------
Wraps YOLOv8 (ultralytics) for object detection.

Returns a `supervision.Detections` object so the rest of the pipeline
(tracker, line-counter) stays library-agnostic.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from loguru import logger

from config import settings
from inference.model_loader import load_yolo_model

_CLASS_IDS = settings.detection_class_ids


class PersonDetector:
    """YOLOv8-based detector returning supervision.Detections."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        model_path = model_path or settings.yolo_model
        self.confidence = confidence if confidence is not None else settings.yolo_confidence
        self.device = device or settings.resolved_yolo_device

        logger.info(f"Loading YOLO model '{model_path}' on device='{self.device}' ...")
        self._model = load_yolo_model(
            model_path,
            fallback_model=settings.yolo_pretrained_model,
        )
        logger.info("YOLO model loaded.")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on *frame* (BGR uint8 H×W×3).

        Returns
        -------
        sv.Detections
            Bounding boxes, confidence scores, and class IDs, filtered to the
            configured classes.
        """
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            classes=_CLASS_IDS,
            device=self.device,
            imgsz=settings.yolo_imgsz,
            verbose=False,
        )
        detections = sv.Detections.from_ultralytics(results[0])
        class_mask = np.isin(detections.class_id, _CLASS_IDS)
        return detections[class_mask]

    # ── convenience annotator ─────────────────────────────────────────────────

    @staticmethod
    def annotate(
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes + labels onto *frame* (in-place copy)."""
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        annotated = frame.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        if labels:
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
        return annotated
