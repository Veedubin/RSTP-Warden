"""Haar cascade vehicle detector.

Uses OpenCV's ``CascadeClassifier`` with a Haar cascade to detect
vehicles (cars) in BGR frames. Designed for real-time use inside the
rtsp-warden ``DetectorRunner`` worker threads.

Algorithm
---------
1. On ``setup()``, load a Haar cascade XML file. The search order is:
   a. ``haarcascade_car.xml`` from the OpenCV data directory (for
      OpenCV builds that include it).
   b. A bundled ``haarcascade_vehicle.xml`` fallback (derived from
      OpenCV's ``haarcascade_fullbody.xml``), shipped with rtsp-warden.
2. For each frame:
   a. Convert to grayscale.
   b. Optionally equalize the histogram to improve detection in
      varying lighting conditions.
   c. Run ``classifier.detectMultiScale`` with configurable
      ``scaleFactor``, ``minNeighbors``, and ``minSize``.
   d. Each detection becomes a ``Detection(kind="vehicle", ...)`` with
      confidence=1.0 (Haar cascades do not return confidence scores).
3. Return a list of ``Detection`` objects.

Limitations
-----------
- Haar cascades are noisy compared to modern deep-learning detectors
  (YOLO, SSD, etc.). Expect false positives and missed detections.
- No confidence score is available; all detections default to 1.0.
- ``min_confidence`` is accepted for API parity but not used for
  filtering since Haar cascades do not produce confidence values.
- The bundled ``haarcascade_vehicle.xml`` is based on
  ``haarcascade_fullbody.xml`` and detects full-body objects, not
  specifically cars. For production vehicle detection, consider using
  YOLO or a similar deep-learning model (future work).

Edge cases
----------
- Frames smaller than 8x8, empty (0x0), or None-like return ``[]``.
- If no cascade file is found (both OpenCV data and bundled fallback),
  logs an error and ``process()`` returns ``[]`` without crashing.
- If ``detectMultiScale`` raises a ``cv2.error``, the error is logged
  and an empty list is returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..base import Detection

logger = logging.getLogger(__name__)

# Minimum frame dimension to attempt detection
_MIN_FRAME_DIM = 8

# Cascade search order: OpenCV data dir first, then bundled fallback
_CASCADE_SEARCH_ORDER = [
    ("OpenCV data", "haarcascade_car.xml", True),
    ("bundled fallback", "haarcascade_vehicle.xml", False),
]


def _find_cascade() -> Path | None:
    """Search for a usable Haar cascade file.

    Looks for ``haarcascade_car.xml`` in the OpenCV data directory
    first, then falls back to the bundled ``haarcascade_vehicle.xml``.
    Returns the path if found, None otherwise.
    """
    # 1. OpenCV data directory
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_car.xml"
    if cascade_path.exists():
        return cascade_path

    # 2. Bundled fallback
    bundled_path = Path(__file__).parent / "cascades" / "haarcascade_vehicle.xml"
    if bundled_path.exists():
        return bundled_path

    return None


@dataclass(slots=True)
class VehicleDetector:
    """Real-time vehicle detector using Haar cascade classification.

    Args:
        min_confidence: Accepted for API parity but Haar cascades do
            not return confidence values; all detections default to 1.0.
        min_size: Minimum object size in pixels for detection. Smaller
            objects are ignored. Default 40 is a sensible starting point.
        name: Detector name (for logging / protocol compliance).
        kind: Detection kind string (default ``"vehicle"``).
    """

    min_confidence: float = 0.5  # accepted but Haar doesn't return confidence
    min_size: int = 40
    name: str = "vehicle"
    kind: str = "vehicle"
    _classifier: cv2.CascadeClassifier | None = None
    _loaded: bool = False

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """Load the Haar cascade classifier for vehicle detection.

        Must be called once before :meth:`process`.  Searches for the
        cascade file in the OpenCV data directory first, then falls back
        to the bundled ``haarcascade_vehicle.xml``.  If no cascade file
        is found, logs an error and the detector will return empty
        results.
        """
        cascade_path = _find_cascade()
        if cascade_path is None:
            logger.error(
                "VehicleDetector: no cascade file found. "
                "Searched OpenCV data dir and bundled fallback."
            )
            self._classifier = None
            self._loaded = False
            return

        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            logger.error(
                "VehicleDetector: failed to load cascade from %s",
                cascade_path,
            )
            self._classifier = None
            self._loaded = False
            return

        self._classifier = classifier
        self._loaded = True
        logger.debug(
            "VehicleDetector setup: cascade=%s min_size=%d min_confidence=%.2f",
            cascade_path,
            self.min_size,
            self.min_confidence,
        )

    def teardown(self) -> None:
        """Release the cascade classifier."""
        self._classifier = None
        self._loaded = False

    # -- frame processing ------------------------------------------------

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process one BGR frame and return vehicle detections.

        Args:
            frame_bgr: Decoded BGR frame (HxWx3, uint8).
            ts_unix: Unix timestamp of the frame.

        Returns:
            List of :class:`Detection` instances for each vehicle
            detection found in the frame.
        """
        if not self._loaded or self._classifier is None:
            logger.error("VehicleDetector.process called before setup() or setup failed")
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

        # Convert to grayscale
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            logger.debug("VehicleDetector: cvtColor failed", exc_info=True)
            return []

        # Equalize histogram to improve detection in varying lighting
        try:
            gray = cv2.equalizeHist(gray)
        except cv2.error:
            logger.debug("VehicleDetector: equalizeHist failed", exc_info=True)
            # Continue with non-equalized grayscale

        # Run cascade detection
        try:
            detections_raw = self._classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(self.min_size, self.min_size),
            )
        except cv2.error:
            logger.debug("VehicleDetector: detectMultiScale failed", exc_info=True)
            return []

        # Convert to Detection list
        detections: list[Detection] = []
        for x, y, w, h in detections_raw:
            detections.append(
                Detection(
                    kind="vehicle",
                    confidence=1.0,  # Haar cascades do not return confidence
                    bbox=(int(x), int(y), int(w), int(h)),
                    metadata={"size": (int(w), int(h))},
                    ts_unix=ts_unix,
                )
            )

        return detections


__all__ = ["VehicleDetector"]
