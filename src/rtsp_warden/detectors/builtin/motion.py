"""MOG2 background-subtraction motion detector.

Uses OpenCV's ``cv2.createBackgroundSubtractorMOG2`` to detect moving
foreground regions in a BGR video frame stream. Designed for real-time
use inside the rtsp-warden ``DetectorRunner`` worker threads.

Algorithm
---------
1. Apply MOG2 background subtractor to get a foreground mask.
2. Threshold the mask at 200 to remove shadow pixels (127).
3. Morphological opening (3x3 kernel) to remove noise.
4. Find external contours and filter by minimum area.
5. Return one ``Detection`` per significant contour.

Edge cases
----------
- Frames smaller than 4x4, empty (0x0), or None-like return ``[]``.
- All-black / all-white frames produce no foreground after MOG2
  learning and are safely ignored.
- A ``cv2`` import failure at runtime logs an error and returns ``[]``
  without crashing the worker thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..base import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------

_MIN_FRAME_DIM = 4  # MOG2 needs at least 4x4 to operate
_SHADOW_THRESH = 200  # pixels >= 200 are foreground (not shadow)
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _sensitivity_to_var_threshold(sensitivity: float) -> int:
    """Convert a 0-1 *sensitivity* to a MOG2 ``varThreshold`` value.

    Higher sensitivity -> lower threshold -> more responsive.
    The result is clamped to ``[8, 100]``.
    """
    vt = int((1.0 - sensitivity) * 50)
    return max(8, min(100, vt))


# ---------------------------------------------------------------------------
# MotionDetector
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotionDetector:
    """Real-time motion detector using MOG2 background subtraction.

    Args:
        min_area: Minimum contour area in pixels to be considered motion.
        sensitivity: 0.0 (least sensitive) to 1.0 (most sensitive).
        name: Detector name (for logging / protocol compliance).
        kind: Detection kind string (default ``"motion"``).
    """

    min_area: int = 500
    sensitivity: float = 0.5
    var_threshold: int | None = None  # Override computed varThreshold when set
    name: str = "motion"
    kind: str = "motion"
    _subtractor: cv2.BackgroundSubtractor | None = None

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """Create the MOG2 background subtractor.

        Must be called once before :meth:`process`.  Keeps the heavy
        OpenCV allocation out of import-time.

        If ``var_threshold`` is set (not None), it overrides the value
        computed from ``sensitivity``.  This allows camera-level sensitivity
        scaling to specify an exact varThreshold.
        """
        var_threshold = (
            self.var_threshold
            if self.var_threshold is not None
            else _sensitivity_to_var_threshold(self.sensitivity)
        )
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        logger.debug(
            "MotionDetector setup: min_area=%d sensitivity=%.2f varThreshold=%d",
            self.min_area,
            self.sensitivity,
            var_threshold,
        )

    def teardown(self) -> None:
        """Release the background subtractor."""
        self._subtractor = None

    # -- frame processing ------------------------------------------------

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process one BGR frame and return motion detections.

        Args:
            frame_bgr: Decoded BGR frame (HxWx3, uint8).
            ts_unix: Unix timestamp of the frame.

        Returns:
            List of :class:`Detection` instances for each significant
            motion contour found in the frame.
        """
        # Guard: subtractor must be initialised via setup()
        if self._subtractor is None:
            logger.error("MotionDetector.process called before setup()")
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

        # Apply MOG2 background subtractor
        try:
            fgmask = self._subtractor.apply(frame_bgr)
        except cv2.error:
            logger.debug("MOG2 apply failed on frame shape=%s", frame_bgr.shape, exc_info=True)
            return []

        # Remove shadows: MOG2 labels shadows as 127, foreground as 255.
        _, foreground = cv2.threshold(fgmask, _SHADOW_THRESH, 255, cv2.THRESH_BINARY)

        # Morphological opening to remove small noise
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, _OPEN_KERNEL)

        # Find external contours
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            confidence = min(1.0, area / (self.min_area * 4))
            detections.append(
                Detection(
                    kind="motion",
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    metadata={"area": area},
                    ts_unix=ts_unix,
                )
            )

        return detections


__all__ = ["MotionDetector"]
