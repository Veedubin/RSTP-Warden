"""rtsp_warden.frame_tap

A lightweight, dependency-free seam for "frame taps".

Design goals:
- stdlib-only (no OpenCV/Pillow)
- best-effort dispatch that never crashes ingest/proxy by default
- stable API contract for later wiring (e.g., MJPEG path / frame hub)

Each camera has its own dispatcher (``CameraRuntime.dispatcher`` in app.py). The camera's
proxy-stream ``StreamIngestor`` (recorder.py) reads ffmpeg's tap output from an ``os.pipe``
whose write end is handed to ffmpeg as ``pipe:<fd>`` and dispatches every JPEG here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


class FrameConsumer(Protocol):
    """Consumer contract for per-frame callbacks.

    Implementations MUST be cheap and MUST NOT block ingest threads.
    If you need heavier work, push to an internal queue/thread.
    """

    name: str

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        """Called when a new JPEG frame is available."""


@dataclass(slots=True)
class FrameTapDispatcher:
    """Dispatch JPEG frames to a set of consumers.

    The dispatcher is intentionally "safe" by default: consumer exceptions are swallowed
    (and logged at debug level) to avoid destabilizing ingest/record/proxy.
    """

    consumers: Sequence[FrameConsumer] = field(default_factory=tuple)
    swallow_exceptions: bool = True

    def dispatch(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        """Send a frame to all consumers.

        Args:
            camera: Camera name
            stream: Stream name (e.g., "main" | "sub")
            jpeg_bytes: JPEG-encoded bytes
            ts_unix: UNIX timestamp (float seconds)
        """

        for consumer in self.consumers:
            try:
                consumer.on_frame(
                    camera=camera, stream=stream, jpeg_bytes=jpeg_bytes, ts_unix=ts_unix
                )
            except Exception:
                if not self.swallow_exceptions:
                    raise
                # Debug-only: avoid noisy logs in hot paths.
                cname = getattr(consumer, "name", consumer.__class__.__name__)
                logger.debug(
                    "FrameConsumer failure (swallowed): %s camera=%s stream=%s",
                    cname,
                    camera,
                    stream,
                    exc_info=True,
                )


__all__ = ["FrameConsumer", "FrameTapDispatcher"]
