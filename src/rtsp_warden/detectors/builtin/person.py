"""HOG + linear SVM person detector.

Uses OpenCV's ``cv2.HOGDescriptor`` with the default people detector
to find pedestrian bounding boxes in BGR frames. Designed for real-time
use inside the rtsp-warden ``DetectorRunner`` worker threads.

Algorithm
---------
1. Initialise ``cv2.HOGDescriptor`` and set the SVM detector to the
   built-in default people detector on ``setup()``.
2. For each frame:
   a. Downscale to a maximum width of 320 pixels (preserving aspect
      ratio) to keep HOG latency acceptable (~50-200ms per frame).
   b. Run ``hog.detectMultiScale`` with configurable ``winStride``,
      ``padding``, and ``scale`` parameters.
   c. Filter detections by ``min_confidence`` (the weight returned by
      HOG; detections below threshold are discarded).
   d. Scale bounding boxes back to the original frame coordinates.
3. Return a list of ``Detection`` objects with ``kind="person"``.

Performance note
----------------
HOG is SLOW. Expect ~50-200ms per frame at 320px width. This is
acceptable for low-framerate feeds (1-2 FPS detection interval) but
will not keep up with full-rate video.

Edge cases
----------
- Frames smaller than 8x8, empty (0x0), or None-like return ``[]``.
- If HOG setup fails (very rare), logs an error and ``process()``
  returns ``[]`` without crashing the worker thread.
- If ``detectMultiScale`` raises a ``cv2.error``, the error is logged
  and an empty list is returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..base import Detection

logger = logging.getLogger(__name__)

# Minimum frame dimension to attempt detection (HOG needs reasonable input)
_MIN_FRAME_DIM = 8
# Maximum processing width for HOG (trade-off between speed and accuracy)
_MAX_PROCESS_WIDTH = 320


@dataclass(slots=True)
class PersonDetector:
    """Real-time person detector using HOG + linear SVM.

    Args:
        min_confidence: Minimum detection confidence (HOG weight) to
            accept. Detections with weight below this are discarded.
        scale_factor: Scale factor for ``detectMultiScale`` image
            pyramid. Lower values increase detection quality but
            slow down processing.
        min_neighbors: Not used by HOG (kept for API parity with
            Haar cascades). Reserved for future grouping logic.
        name: Detector name (for logging / protocol compliance).
        kind: Detection kind string (default ``"person"``).
    """

    min_confidence: float = 0.5
    scale_factor: float = 1.1
    min_neighbors: int = 3
    name: str = "person"
    kind: str = "person"
    _hog: cv2.HOGDescriptor | None = None
    _process_count: int = 0
    _last_process_at: float = 0.0

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """Initialise the HOG descriptor with the default people detector.

        Must be called once before :meth:`process`.  Keeps the heavy
        OpenCV allocation out of import-time.
        """
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
            logger.debug(
                "PersonDetector setup: min_confidence=%.2f scale_factor=%.2f",
                self.min_confidence,
                self.scale_factor,
            )
        except Exception:
            logger.error("PersonDetector setup failed", exc_info=True)
            self._hog = None

    def teardown(self) -> None:
        """Release the HOG descriptor."""
        self._hog = None

    # -- frame processing ------------------------------------------------

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process one BGR frame and return person detections.

        Args:
            frame_bgr: Decoded BGR frame (HxWx3, uint8).
            ts_unix: Unix timestamp of the frame.

        Returns:
            List of :class:`Detection` instances for each person
            detection above ``min_confidence``.
        """
        if self._hog is None:
            logger.error("PersonDetector.process called before setup() or setup failed")
            return []

        # Guard: invalid / degenerate frames
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return []
            h, w = frame_bgr.shape[:2]
            if h < _MIN_FRAME_DIM or w < _MIN_FRAME_DIM:
                return []
        except Exception:
            return []

        # Downscale for speed: limit the longer edge to _MAX_PROCESS_WIDTH
        scale = 1.0
        try:
            max_dim = max(h, w)
            if max_dim > _MAX_PROCESS_WIDTH:
                scale = _MAX_PROCESS_WIDTH / max_dim
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                small_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                small_frame = frame_bgr
        except cv2.error:
            logger.debug("PersonDetector resize failed", exc_info=True)
            return []

        # Run HOG detection
        try:
            locations, weights = self._hog.detectMultiScale(
                small_frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
            )
        except cv2.error:
            logger.debug("PersonDetector detectMultiScale failed", exc_info=True)
            return []

        # Filter by min_confidence and convert to Detection list
        detections: list[Detection] = []
        for i, (x, y, w, h) in enumerate(locations):
            weight = float(weights[i]) if i < len(weights) else 0.0
            if weight < self.min_confidence:
                continue

            # Scale bbox back to original frame coordinates
            if scale != 1.0:
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                orig_w = int(w / scale)
                orig_h = int(h / scale)
            else:
                orig_x, orig_y, orig_w, orig_h = int(x), int(y), int(w), int(h)

            detections.append(
                Detection(
                    kind="person",
                    confidence=min(1.0, weight),
                    bbox=(orig_x, orig_y, orig_w, orig_h),
                    metadata={"weight": weight},
                    ts_unix=ts_unix,
                )
            )

        self._process_count += 1
        self._last_process_at = ts_unix

        return detections


__all__ = ["PersonDetector"]
