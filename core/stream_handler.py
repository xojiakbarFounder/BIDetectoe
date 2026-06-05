"""
stream_handler.py
-----------------
Resolves a YouTube live-stream (or any URL) to a direct video URL via yt-dlp,
then yields decoded BGR frames at a capped frame-rate using OpenCV.

Usage
-----
    from core.stream_handler import StreamHandler

    with StreamHandler("https://www.youtube.com/watch?v=...") as stream:
        for frame in stream:
            # frame is a numpy uint8 array (H, W, 3) BGR
            process(frame)
"""

from __future__ import annotations

import time
from typing import Generator

import cv2
import numpy as np
from loguru import logger

# yt-dlp is optional at import time so the rest of the app loads even if the
# stream worker is not running.
try:
    import yt_dlp  # type: ignore
    _YT_DLP_AVAILABLE = True
except ImportError:
    _YT_DLP_AVAILABLE = False


class StreamHandler:
    """Iterate over frames from a live stream (YouTube, RTSP, HLS, file …)."""

    def __init__(
        self,
        url: str,
        target_fps: int = 10,
        reconnect_delay: float = 5.0,
        max_reconnects: int = 10,
    ) -> None:
        self.url = url
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

        self._cap: cv2.VideoCapture | None = None
        self._direct_url: str = url
        self._frame_interval: float = 1.0 / max(target_fps, 1)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "StreamHandler":
        self._direct_url = self._resolve_url(self.url)
        self._open_capture()
        return self

    def __exit__(self, *_):
        self._release()

    # ── public iterator ───────────────────────────────────────────────────────

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        reconnects = 0
        last_emit = 0.0

        while True:
            if self._cap is None or not self._cap.isOpened():
                if reconnects >= self.max_reconnects:
                    logger.error("Max reconnect attempts reached. Stopping stream.")
                    break
                logger.warning(
                    f"Stream not available. Reconnecting in {self.reconnect_delay}s "
                    f"(attempt {reconnects + 1}/{self.max_reconnects}) …"
                )
                time.sleep(self.reconnect_delay)
                self._direct_url = self._resolve_url(self.url)
                self._open_capture()
                reconnects += 1
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._release()
                continue

            # Frame-rate throttle
            now = time.monotonic()
            if now - last_emit < self._frame_interval:
                continue
            last_emit = now
            reconnects = 0  # reset on successful read
            yield frame

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_url(url: str) -> str:
        """Use yt-dlp to get a direct streamable URL from a YouTube link."""
        if not _YT_DLP_AVAILABLE:
            logger.debug("yt-dlp not installed — using URL as-is.")
            return url

        # Only attempt resolution for youtube / youtu.be URLs
        if "youtube.com" not in url and "youtu.be" not in url:
            return url

        logger.info(f"Resolving stream URL via yt-dlp: {url}")
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
            "extractor_args": {
                "youtube": {
                    "player_client": ["android"],
                },
            },
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                direct = info.get("url") or info.get("manifest_url") or url
                logger.info(f"Resolved to: {direct[:80]}…")
                return direct
        except Exception as exc:
            logger.warning(f"yt-dlp resolution failed ({exc}). Using URL as-is.")
            return url

    def _open_capture(self) -> None:
        """Open an OpenCV VideoCapture from the resolved URL."""
        self._release()
        logger.info(f"Opening capture: {self._direct_url[:80]}…")
        cap = cv2.VideoCapture(self._direct_url)
        if not cap.isOpened():
            logger.error("VideoCapture failed to open the stream.")
            self._cap = None
        else:
            # Reduce internal buffer to keep latency low
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self._cap = cap
            src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            logger.info(
                f"Stream opened. Source FPS={src_fps:.1f}, "
                f"throttled to {self.target_fps} FPS."
            )

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
