"""rtsp_warden.consumers.motion_demo

Example "motion-ish" heuristic consumer.

This intentionally avoids image decoding and third-party dependencies.
It uses the JPEG byte-length delta as a crude proxy signal.

This is not real motion detection, but it provides a working example
for wiring the frame-tap seam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..frame_tap import FrameConsumer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionHeuristicConsumer(FrameConsumer):
    """Log a "motion-ish" message when JPEG size changes sharply.

    Args:
        threshold_ratio: Trigger when abs(delta)/last_len >= threshold_ratio.

    Notes:
        - The first frame for a (camera, stream) key establishes baseline.
        - This is designed to be fast and safe in an ingest thread.
    """

    threshold_ratio: float = 0.20
    name: str = "motion_heuristic"
    _last_len: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        key = (camera, stream)
        n = len(jpeg_bytes)
        last = self._last_len.get(key)

        # Initialize baseline.
        if last is None:
            self._last_len[key] = n
            return

        denom = last if last > 0 else 1
        ratio = abs(n - last) / denom

        if ratio >= self.threshold_ratio:
            logger.info(
                "motion-ish camera=%s stream=%s last=%d now=%d ratio=%.3f ts=%.3f",
                camera,
                stream,
                last,
                n,
                ratio,
                ts_unix,
            )

        self._last_len[key] = n


if __name__ == "__main__":
    # Tiny self-test snippet (stdlib-only).
    import time

    from ..frame_tap import FrameTapDispatcher

    logging.basicConfig(level=logging.INFO)

    consumer = MotionHeuristicConsumer(threshold_ratio=0.20)
    dispatcher = FrameTapDispatcher(consumers=[consumer])

    now = time.time()
    dispatcher.dispatch("cam1", "sub", b"x" * 1000, now)
    dispatcher.dispatch("cam1", "sub", b"x" * 1020, now + 0.1)
    dispatcher.dispatch("cam1", "sub", b"x" * 1400, now + 0.2)  # should log
