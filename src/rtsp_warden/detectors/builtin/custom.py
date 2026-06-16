"""Template custom detector for rtsp-warden.

Provides ``DemoCustomDetector`` -- a minimal working custom detector
that serves as both documentation and a test target for the custom
detector machinery in the registry.

Users can subclass or copy this pattern to write their own detectors.
Place the detector in a Python module on ``PYTHONPATH``, then reference
it in ``config.yaml``::

    detectors:
      - type: custom
        import_path: my_pkg.my_detectors:MyDetector
        config:
          threshold: 0.3

Custom detector contract
------------------------
A custom detector must satisfy the ``Detector`` protocol defined in
``rtsp_warden.detectors.base``:

- ``name: str`` -- detector name for logging
- ``kind: str`` -- detection kind string (e.g. ``"person"``, ``"vehicle"``)
- ``setup() -> None`` -- called once before any ``process()`` calls
- ``process(frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]``
  -- process a BGR frame and return detections
- ``teardown() -> None`` -- called once at shutdown

The detector must be importable via ``import_path`` in one of two formats:

1. Colon-separated: ``my_pkg.my_detectors:MyDetector``
2. Dot-separated: ``my_pkg.my_detectors.MyDetector``

The ``config`` dict is passed as keyword arguments to the detector
constructor (``__init__``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Detection


@dataclass(slots=True)
class DemoCustomDetector:
    """Template custom detector. Counts frames where >threshold% of pixels are "red".

    A pixel is considered "red" when the red channel (BGR[2]) is above
    150 AND both the green channel (BGR[1]) and blue channel (BGR[0])
    are below 80.

    If the ratio of red pixels exceeds ``threshold``, a single Detection
    spanning the whole frame is returned.

    Args:
        threshold: Minimum fraction of red pixels (0.0-1.0) to trigger
            a detection. Default 0.3 (30%).
        name: Detector name (for logging / protocol compliance).
        kind: Detection kind string (default ``"red_detector"``).
    """

    threshold: float = 0.3
    name: str = "demo_custom"
    kind: str = "red_detector"

    def setup(self) -> None:
        """No-op setup. Included for Detector protocol compliance."""
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process one BGR frame and return a detection if enough red pixels.

        Args:
            frame_bgr: Decoded BGR frame (HxWx3, uint8).
            ts_unix: Unix timestamp of the frame.

        Returns:
            A list containing a single Detection if the red pixel ratio
            exceeds ``threshold``, otherwise an empty list.
        """
        # Guard: invalid / degenerate frames
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return []
            h, w = frame_bgr.shape[:2]
            if h == 0 or w == 0:
                return []
        except Exception:
            return []

        # Count red pixels: BGR[..., 2] > 150 AND BGR[..., 1] < 80 AND BGR[..., 0] < 80
        red_mask = (frame_bgr[..., 2] > 150) & (frame_bgr[..., 1] < 80) & (frame_bgr[..., 0] < 80)
        total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
        red_ratio = float(np.count_nonzero(red_mask)) / total_pixels

        if red_ratio < self.threshold:
            return []

        # Detection spans the whole frame
        return [
            Detection(
                kind=self.kind,
                confidence=min(1.0, red_ratio),
                bbox=(0, 0, w, h),
                metadata={"red_ratio": red_ratio},
                ts_unix=ts_unix,
            )
        ]

    def teardown(self) -> None:
        """No-op teardown. Included for Detector protocol compliance."""
        pass


__all__ = ["DemoCustomDetector"]
