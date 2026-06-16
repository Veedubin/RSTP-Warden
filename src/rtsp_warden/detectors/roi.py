"""ROI and privacy mask types for detectors.

A Mask is a polygon region that gets blacked out in the frame BEFORE
the detector sees it. Use cases: hide neighbor's window, blur out
license plates, exclude timestamps/UI overlays.

An ROI (Region of Interest) is a polygon that detector results are
filtered against. Detections outside the ROI are discarded. Use cases:
"only detect motion in the driveway", "only alert on people in the
front yard".

Coordinate system:
  - Polygons are lists of (x, y) integer tuples
  - Coordinates are in the actual frame's pixel space (no scaling)
  - At least 3 points required for a valid polygon
  - cv2.fillPoly / cv2.pointPolygonTest handle the rest
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import Detection


@dataclass(slots=True)
class Mask:
    """Privacy mask -- polygon region to be blacked out before detection."""

    polygon: list[tuple[int, int]]
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Mask polygon must have >= 3 points, got {len(self.polygon)}")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of frame with the polygon region blacked out.

        Args:
            frame: BGR frame (H, W, 3) uint8.

        Returns:
            New frame with the mask region set to zero (black).
        """
        result = frame.copy()
        pts = np.array(self.polygon, dtype=np.int32)
        cv2.fillPoly(result, [pts], color=(0, 0, 0))
        return result


@dataclass(slots=True)
class ROI:
    """Region of interest -- detector results outside are discarded."""

    polygon: list[tuple[int, int]]
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"ROI polygon must have >= 3 points, got {len(self.polygon)}")

    def contains(self, bbox: tuple[int, int, int, int]) -> bool:
        """Return True if the bbox center is inside the ROI polygon.

        Args:
            bbox: (x, y, w, h) detection bounding box.

        Returns:
            True if the center point of the bbox lies within the polygon.
            Returns False for empty bboxes (w==0 or h==0).
        """
        x, y, w, h = bbox
        if w == 0 or h == 0:
            return False
        cx = x + w // 2
        cy = y + h // 2
        return self.contains_point(cx, cy)

    def contains_point(self, x: int, y: int) -> bool:
        """Return True if the point (x, y) is inside the ROI polygon.

        Args:
            x: X coordinate in frame pixel space.
            y: Y coordinate in frame pixel space.

        Returns:
            True if the point lies within or on the edge of the polygon.
        """
        pts = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def apply_masks(frame: np.ndarray, masks: list[Mask]) -> np.ndarray:
    """Apply all masks to the frame.

    Args:
        frame: BGR frame (H, W, 3) uint8.
        masks: List of Mask objects to apply.

    Returns:
        New frame with masks blacked out, or the original frame
        (no copy) if masks is empty.
    """
    if not masks:
        return frame

    result = frame.copy()
    for m in masks:
        pts = np.array(m.polygon, dtype=np.int32)
        cv2.fillPoly(result, [pts], color=(0, 0, 0))
    return result


def filter_by_roi(detections: list[Detection], roi: ROI | None) -> list[Detection]:
    """Filter detections to those inside the ROI.

    Args:
        detections: List of Detection objects with optional bbox.
        roi: ROI polygon to filter against. None means pass through
            all detections unchanged.

    Returns:
        List of detections whose bbox center lies inside the ROI.
        Detections without a bbox are included only when roi is None.
    """
    if roi is None:
        return detections
    return [d for d in detections if d.bbox is not None and roi.contains(d.bbox)]


__all__ = ["Mask", "ROI", "apply_masks", "filter_by_roi"]
