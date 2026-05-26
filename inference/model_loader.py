"""Production model loading helpers for YOLOv8 inference."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from ultralytics import YOLO


def resolve_model_path(model_path: str, fallback_model: str = "yolov8n.pt") -> str:
    """
    Return a usable YOLO model path.

    Local custom models are preferred. If a configured local file is missing,
    the fallback pretrained model keeps the realtime pipeline alive.
    """
    path = Path(model_path)
    if path.exists() or not path.suffix:
        return model_path

    logger.warning(
        "Configured YOLO model '{}' was not found. Falling back to '{}'.",
        model_path,
        fallback_model,
    )
    return fallback_model


def load_yolo_model(model_path: str, fallback_model: str = "yolov8n.pt") -> YOLO:
    """Load a YOLO model from pretrained weights or a fine-tuned best.pt file."""
    resolved_path = resolve_model_path(model_path, fallback_model=fallback_model)
    logger.info("Loading YOLO model from '{}'.", resolved_path)
    return YOLO(resolved_path)

