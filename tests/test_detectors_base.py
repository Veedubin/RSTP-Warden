"""Tests for detectors/base.py: Detection, NullDetector, apply_masks."""

from __future__ import annotations

import numpy as np

from rtsp_warden.detectors.base import Detection, NullDetector, apply_masks

# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


def test_detection_defaults() -> None:
    """Detection with only kind has sensible defaults."""
    d = Detection(kind="motion")
    assert d.kind == "motion"
    assert d.confidence == 1.0
    assert d.bbox is None
    assert d.metadata == {}
    assert d.ts_unix == 0.0


def test_detection_full_construction() -> None:
    """Detection with all fields set."""
    d = Detection(
        kind="person",
        confidence=0.85,
        bbox=(10, 20, 30, 40),
        metadata={"model": "hog"},
        ts_unix=1700000000.0,
    )
    assert d.kind == "person"
    assert d.confidence == 0.85
    assert d.bbox == (10, 20, 30, 40)
    assert d.metadata == {"model": "hog"}
    assert d.ts_unix == 1700000000.0


def test_detection_metadata_is_independent() -> None:
    """Two Detection instances have independent metadata dicts."""
    d1 = Detection(kind="motion")
    d2 = Detection(kind="vehicle")
    d1.metadata["foo"] = "bar"
    assert "foo" not in d2.metadata


# ---------------------------------------------------------------------------
# NullDetector
# ---------------------------------------------------------------------------


def test_null_detector_setup_is_noop() -> None:
    """NullDetector.setup() does not raise."""
    nd = NullDetector()
    nd.setup()


def test_null_detector_process_returns_empty() -> None:
    """NullDetector.process() always returns an empty list."""
    nd = NullDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = nd.process(frame, 1700000000.0)
    assert result == []


def test_null_detector_teardown_is_noop() -> None:
    """NullDetector.teardown() does not raise."""
    nd = NullDetector()
    nd.teardown()


def test_null_detector_satisfies_detector_protocol() -> None:
    """NullDetector has the required attributes of the Detector protocol."""
    nd = NullDetector()
    assert hasattr(nd, "name")
    assert hasattr(nd, "kind")
    assert hasattr(nd, "setup")
    assert hasattr(nd, "process")
    assert hasattr(nd, "teardown")
    assert nd.name == "null"
    assert nd.kind == "null"


# ---------------------------------------------------------------------------
# apply_masks
# ---------------------------------------------------------------------------


def test_apply_masks_no_masks_returns_frame_unchanged() -> None:
    """apply_masks with empty masks list returns the same frame."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = apply_masks(frame, [])
    assert result is frame


def test_apply_masks_with_masks_blacks_out_region() -> None:
    """apply_masks with Mask objects blacks out the polygon region."""
    from rtsp_warden.detectors.roi import Mask

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = Mask(polygon=[(10, 10), (50, 10), (50, 50), (10, 50)])
    result = apply_masks(frame, [mask])
    # The masked region should be black (0)
    assert result[30, 30, 0] == 0
    # Outside the mask should still be white (255)
    assert result[80, 80, 0] == 255
    # Original frame should not be mutated
    assert frame[30, 30, 0] == 255
