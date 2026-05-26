"""
preview_server.py
-----------------
Detection pipeline ni ishlatadi va annotated stream ni
http://localhost:5050 da MJPEG sifatida tarqatadi.

VS Code da ko'rish:
    1. python scripts/preview_server.py
    2. Ctrl+Shift+P  →  "Simple Browser: Show"
    3. URL:  http://localhost:5050
"""

import sys, os, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from loguru import logger

from config import settings
from core.stream_handler import StreamHandler
from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.line_counter import LineCounter

# ── Shared latest frame ───────────────────────────────────────────────────────

_lock  = threading.Lock()
_frame: bytes | None = None

def set_frame(data: bytes):
    global _frame
    with _lock:
        _frame = data

def get_frame() -> bytes | None:
    with _lock:
        return _frame

# ── MJPEG HTTP server ─────────────────────────────────────────────────────────

HTML_PAGE = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Detection Preview</title>
  <style>
    body { margin:0; background:#111; display:flex; justify-content:center;
           align-items:center; min-height:100vh; flex-direction:column; }
    img  { max-width:100%; border-radius:8px; }
    p    { color:#9ca3af; font-family:monospace; margin-top:10px; font-size:13px; }
  </style>
</head>
<body>
  <img src="/feed" alt="stream">
  <p>YOLOv8 + ByteTrack &mdash; Live Detection</p>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # suppress access logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE)

        elif self.path == "/feed":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = get_frame()
                    if frame:
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame + b"\r\n"
                        )
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def _serve():
    server = HTTPServer(("localhost", 5050), Handler)
    logger.info("Preview server: http://localhost:5050")
    server.serve_forever()


# ── Detection pipeline ────────────────────────────────────────────────────────

def run():
    threading.Thread(target=_serve, daemon=True).start()

    logger.info("YOLO modeli yuklanmoqda...")
    detector = PersonDetector()
    tracker  = PersonTracker(frame_rate=settings.target_fps)
    line_counter = None

    logger.info(f"Stream: {settings.youtube_stream_url}")
    logger.info("VS Code da: Ctrl+Shift+P → 'Simple Browser: Show' → http://localhost:5050")

    with StreamHandler(url=settings.youtube_stream_url, target_fps=settings.target_fps) as stream:
        for frame_idx, frame in enumerate(stream):
            h, w = frame.shape[:2]

            if line_counter is None:
                line_counter = LineCounter(frame_width=w, frame_height=h)
                logger.info(f"Chiziq: {line_counter.pt1} → {line_counter.pt2}")

            detections = detector.detect(frame)
            tracked    = tracker.update(detections)
            events     = line_counter.update(tracked)

            for ev in events:
                logger.info(f"CROSSING #{ev.tracker_id} → {ev.direction} "
                            f"(IN={line_counter.in_count} OUT={line_counter.out_count})")

            labels    = PersonTracker.build_labels(tracked)
            annotated = PersonDetector.annotate(frame, tracked, labels=labels)
            annotated = line_counter.draw(annotated)

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                set_frame(buf.tobytes())

            if frame_idx % 30 == 0:
                logger.info(f"[{frame_idx}] odamlar={len(detections)} | "
                            f"IN={line_counter.in_count} OUT={line_counter.out_count}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("To'xtatildi.")
