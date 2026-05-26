"""
video.py
--------
MJPEG streaming endpoint — serves the annotated pipeline frames live.

    GET /video/feed   →  multipart/x-mixed-replace MJPEG stream
    GET /video/frame  →  single latest JPEG (for polling clients)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from api.state import live_state

router = APIRouter(prefix="/video", tags=["video"])

_BOUNDARY = b"frame"
_NO_FRAME = b""  # placeholder when pipeline hasn't sent anything yet


async def _mjpeg_generator():
    while True:
        frame = live_state.get_frame()
        if frame:
            yield (
                b"--" + _BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(0.08)  # ~12 fps ceiling


@router.get("/feed")
async def video_feed():
    """Live MJPEG stream of the annotated pipeline output."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
    )


@router.get("/frame")
async def latest_frame():
    """Single latest JPEG frame — for polling clients like Streamlit."""
    frame = live_state.get_frame()
    if frame is None:
        return Response(status_code=204)
    return Response(content=frame, media_type="image/jpeg")
