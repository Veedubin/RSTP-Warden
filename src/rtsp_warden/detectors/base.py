"""Base types for detectors.

A Detector processes a decoded BGR frame (np.ndarray, OpenCV convention) and
returns a list of Detection results. Detectors MUST be cheap, MUST NOT block,
and MUST be thread-safe (they will be called from worker threads).

The runner (DetectorRunner) is responsible for:
- Decoding JPEG bytes (we receive JPEG from the FrameConsumer seam)
- Applying privacy masks before passing to detectors
- Calling detectors on a worker thread pool
- Collecting results and forwarding to sinks
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class Detection:
    """A single detection result from a detector."""

    kind: str  # "motion" | "person" | "vehicle" | custom
    confidence: float = 1.0  # 0.0-1.0 (rough)
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h) in original frame coords
    metadata: dict[str, Any] = field(default_factory=dict)
    ts_unix: float = 0.0


class Detector(Protocol):
    """Protocol for all detector implementations.

    Implementations must be thread-safe and non-blocking.
    """

    name: str
    kind: str  # e.g. "motion", "person", "vehicle"

    def setup(self) -> None:
        """Lazy init (load models etc.); called once before any process calls."""
        ...

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        """Process a single BGR frame and return detections."""
        ...

    def teardown(self) -> None:
        """Cleanup (release models etc.); called once at shutdown."""
        ...


class NullDetector:
    """A no-op detector that always returns an empty list.

    Used as a fallback when a config references a detector type that
    is not yet implemented or fails to import.
    """

    name: str = "null"
    kind: str = "null"

    def setup(self) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        return []

    def teardown(self) -> None:
        pass


def apply_masks(frame: np.ndarray, masks: Sequence[Any]) -> np.ndarray:
    """Black out polygon regions defined by masks.

    Delegates to the real implementation in roi.py. Kept here for
    backward compatibility -- existing imports still work.

    Args:
        frame: BGR frame (np.ndarray, OpenCV convention).
        masks: Sequence of Mask objects (each has a .polygon attribute).

    Returns:
        The frame with masked regions zeroed out, or unchanged if
        no masks are provided.
    """
    from .roi import Mask
    from .roi import apply_masks as _real_apply_masks

    if not masks:
        return frame

    # Coerce any sequence of Mask-like objects to actual Mask instances
    real_masks: list[Mask] = []
    for m in masks:
        if isinstance(m, Mask):
            real_masks.append(m)
        else:
            # Dict-like fallback for forward compatibility
            real_masks.append(Mask(polygon=m["polygon"], name=m.get("name", "")))

    return _real_apply_masks(frame, real_masks)


__all__ = ["Detection", "Detector", "NullDetector", "apply_masks"]
