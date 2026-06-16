"""Tests for builtin/custom.py: DemoCustomDetector and registry custom loading."""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_warden.detectors.base import NullDetector
from rtsp_warden.detectors.builtin.custom import DemoCustomDetector
from rtsp_warden.detectors.registry import DetectorSpec, build_detector

# ---------------------------------------------------------------------------
# 1-3. DemoCustomDetector behaviour
# ---------------------------------------------------------------------------


def test_demo_custom_detector_red_frame() -> None:
    """A frame that is mostly red produces one detection."""
    det = DemoCustomDetector(threshold=0.3)
    det.setup()
    try:
        # Build a 200x200 frame that is mostly red (BGR: [0, 0, 200])
        red_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        red_frame[..., 2] = 200  # R channel high
        red_frame[..., 1] = 40  # G channel low
        red_frame[..., 0] = 40  # B channel low
        results = det.process(red_frame, ts_unix=1.0)
        assert len(results) == 1
        d = results[0]
        assert d.kind == "red_detector"
        assert d.bbox == (0, 0, 200, 200)
        assert d.confidence > 0.3
    finally:
        det.teardown()


def test_demo_custom_detector_non_red_frame() -> None:
    """A frame with no red pixels produces no detections."""
    det = DemoCustomDetector(threshold=0.3)
    det.setup()
    try:
        # All-black frame: no red pixels at all
        black_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        results = det.process(black_frame, ts_unix=1.0)
        assert results == []
    finally:
        det.teardown()


def test_threshold_changes_sensitivity() -> None:
    """High threshold suppresses detection; low threshold allows it."""
    # Build a partially red frame (~40% red pixels)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Make left 40% "red" (R>150, G<80, B<80)
    frame[:40, :, 2] = 180  # R channel
    frame[:40, :, 1] = 50  # G channel
    frame[:40, :, 0] = 50  # B channel
    # Rest stays all black (not red)

    # Low threshold (0.1): should detect
    det_low = DemoCustomDetector(threshold=0.1)
    det_low.setup()
    try:
        results_low = det_low.process(frame, ts_unix=1.0)
    finally:
        det_low.teardown()

    # High threshold (0.6): should not detect (~40% < 0.6)
    det_high = DemoCustomDetector(threshold=0.6)
    det_high.setup()
    try:
        results_high = det_high.process(frame, ts_unix=1.0)
    finally:
        det_high.teardown()

    assert len(results_low) > 0, "Low threshold should produce detection"
    assert len(results_high) == 0, "High threshold should suppress detection"


# ---------------------------------------------------------------------------
# 4-8. Registry custom detector loading
# ---------------------------------------------------------------------------


def test_registry_loads_custom_detector() -> None:
    """build_detector with type=custom loads DemoCustomDetector via import_path."""
    spec = DetectorSpec(
        type="custom",
        import_path="rtsp_warden.detectors.builtin.custom:DemoCustomDetector",
        config={"threshold": 0.5},
    )
    detector = build_detector(spec, "test_camera")
    assert isinstance(detector, DemoCustomDetector)
    assert detector.threshold == 0.5


def test_registry_loads_custom_with_colon_separator() -> None:
    """import_path with colon separator (module:Class) works correctly."""
    spec = DetectorSpec(
        type="custom",
        import_path="rtsp_warden.detectors.builtin.custom:DemoCustomDetector",
        config={"threshold": 0.3},
    )
    detector = build_detector(spec, "test_camera")
    assert isinstance(detector, DemoCustomDetector)


def test_registry_loads_custom_with_dot_separator() -> None:
    """import_path with dot separator (module.Class) works correctly."""
    spec = DetectorSpec(
        type="custom",
        import_path="rtsp_warden.detectors.builtin.custom.DemoCustomDetector",
        config={"threshold": 0.3},
    )
    detector = build_detector(spec, "test_camera")
    assert isinstance(detector, DemoCustomDetector)


def test_custom_detector_missing_import_path_raises() -> None:
    """type=custom with import_path=None raises ValueError."""
    spec = DetectorSpec(
        type="custom",
        import_path=None,
    )
    with pytest.raises(ValueError, match="import_path"):
        build_detector(spec, "test_camera")


def test_custom_detector_invalid_path_falls_back_to_null() -> None:
    """Bad import_path falls back to NullDetector with a warning logged."""
    spec = DetectorSpec(
        type="custom",
        import_path="nonexistent.module:BadClass",
        config={},
    )
    detector = build_detector(spec, "test_camera")
    assert isinstance(detector, NullDetector)
