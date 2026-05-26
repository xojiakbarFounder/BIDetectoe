"""
check_frame.py
--------------
Streamdan bitta frame oladi, detection + chiziqni chizadi,
debug_frame.jpg ga saqlaydi. VS Code da ochib tekshirish uchun.

    python scripts/check_frame.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from loguru import logger

from config import settings
from core.stream_handler import StreamHandler
from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.line_counter import LineCounter

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "debug_frame.jpg")

logger.info(f"Stream: {settings.youtube_stream_url}")
logger.info("Bitta frame kutilmoqda...")

detector = PersonDetector()
tracker  = PersonTracker(frame_rate=settings.target_fps)

with StreamHandler(url=settings.youtube_stream_url, target_fps=settings.target_fps) as stream:
    for frame in stream:
        h, w = frame.shape[:2]
        logger.info(f"Frame olindi: {w}x{h}")

        detections = detector.detect(frame)
        tracked    = tracker.update(detections)
        line       = LineCounter(frame_width=w, frame_height=h)

        labels    = PersonTracker.build_labels(tracked)
        annotated = PersonDetector.annotate(frame, tracked, labels=labels)
        annotated = line.draw(annotated)

        # Chiziq koordinatalarini ekranga chiqar
        logger.info(f"Chiziq: {line.pt1} → {line.pt2}")
        logger.info(f"Aniqlangan odamlar soni: {len(detections)}")

        cv2.imwrite(OUTPUT_PATH, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.success(f"Saqlandi: {os.path.abspath(OUTPUT_PATH)}")
        break
