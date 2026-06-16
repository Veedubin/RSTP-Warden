"""Tests for ROI and privacy mask support (detectors/roi.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_warden.detectors.base import Detection, apply_masks
from rtsp_warden.detectors.roi import ROI, Mask, filter_by_roi

# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def test_mask_with_3_points_is_valid() -> None:
    """Mask with exactly 3 points is valid."""
    m = Mask(polygon=[(0, 0), (100, 0), (50, 100)], name="tri")
    assert len(m.polygon) == 3
    assert m.name == "tri"


def test_mask_with_2_points_raises() -> None:
    """Mask with fewer than 3 points raises ValueError."""
    with pytest.raises(ValueError, match=">= 3 points"):
        Mask(polygon=[(0, 0), (100, 0)])


def test_mask_with_0_points_raises() -> None:
    """Mask with empty polygon raises ValueError."""
    with pytest.raises(ValueError, match=">= 3 points"):
        Mask(polygon=[])


# ---------------------------------------------------------------------------
# Mask.apply
# ---------------------------------------------------------------------------


def test_mask_apply_blacks_out_region() -> None:
    """Mask.apply blacks out the polygon region on a white frame."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    mask = Mask(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    result = mask.apply(frame)
    # Center of the masked region should be black
    assert result[150, 150, 0] == 0
    assert result[150, 150, 1] == 0
    assert result[150, 150, 2] == 0
    # Outside the mask should remain white
    assert result[50, 50, 0] == 255


def test_mask_apply_returns_new_frame() -> None:
    """Mask.apply does not mutate the original frame."""
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = Mask(polygon=[(10, 10), (90, 10), (90, 90), (10, 90)])
    result = mask.apply(frame)
    # Original should still be all white
    assert frame[50, 50, 0] == 255
    # Result should have black in the center
    assert result[50, 50, 0] == 0


# ---------------------------------------------------------------------------
# apply_masks function
# ---------------------------------------------------------------------------


def test_apply_masks_no_masks_returns_unchanged() -> None:
    """apply_masks with empty list returns the same frame object (no copy)."""
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = apply_masks(frame, [])
    assert result is frame


def test_apply_masks_with_multiple_masks() -> None:
    """apply_masks with two masks blacks out both regions."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    mask1 = Mask(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    mask2 = Mask(polygon=[(200, 200), (300, 200), (300, 300), (200, 300)])
    result = apply_masks(frame, [mask1, mask2])
    # First mask region
    assert result[50, 50, 0] == 0
    # Second mask region
    assert result[250, 250, 0] == 0
    # Unmasked region
    assert result[400, 400, 0] == 255


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------


def test_roi_with_3_points_is_valid() -> None:
    """ROI with exactly 3 points is valid."""
    roi = ROI(polygon=[(0, 0), (100, 0), (50, 100)])
    assert len(roi.polygon) == 3


def test_roi_contains_point_inside() -> None:
    """ROI.contains returns True for a point inside the polygon."""
    # Square ROI from (100,100) to (200,200)
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    assert roi.contains_point(150, 150) is True


def test_roi_contains_point_outside() -> None:
    """ROI.contains_point returns False for a point outside the polygon."""
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    assert roi.contains_point(50, 50) is False


def test_roi_contains_bbox_inside() -> None:
    """ROI.contains returns True for a bbox whose center is inside."""
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    # bbox centered at (150, 150) -> inside
    assert roi.contains((140, 140, 20, 20)) is True


def test_roi_contains_empty_bbox_returns_false() -> None:
    """ROI.contains returns False for a bbox with zero width or height."""
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    assert roi.contains((150, 150, 0, 20)) is False
    assert roi.contains((150, 150, 20, 0)) is False


# ---------------------------------------------------------------------------
# filter_by_roi
# ---------------------------------------------------------------------------


def test_filter_by_roi_keeps_inside() -> None:
    """filter_by_roi keeps only detections inside the ROI."""
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    d1 = Detection(kind="motion", bbox=(140, 140, 20, 20))  # inside
    d2 = Detection(kind="motion", bbox=(150, 150, 10, 10))  # inside
    d3 = Detection(kind="motion", bbox=(50, 50, 20, 20))  # outside
    result = filter_by_roi([d1, d2, d3], roi)
    assert len(result) == 2
    assert d1 in result
    assert d2 in result


def test_filter_by_roi_none_passes_through() -> None:
    """filter_by_roi with roi=None returns all detections."""
    d1 = Detection(kind="motion", bbox=(10, 10, 20, 20))
    d2 = Detection(kind="person", bbox=(500, 500, 30, 30))
    result = filter_by_roi([d1, d2], None)
    assert result == [d1, d2]


def test_filter_by_roi_drops_none_bbox() -> None:
    """filter_by_roi drops detections with bbox=None when roi is set."""
    roi = ROI(polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])
    d_no_bbox = Detection(kind="motion")
    d_inside = Detection(kind="motion", bbox=(140, 140, 20, 20))
    result = filter_by_roi([d_no_bbox, d_inside], roi)
    assert len(result) == 1
    assert d_inside in result
