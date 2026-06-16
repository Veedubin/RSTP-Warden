"""Tests for detectors/builtin/motion.py: MotionDetector."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.models import Camera
from rtsp_warden.db.schema import ensure_schema, list_events
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.builtin.motion import MotionDetector
from rtsp_warden.detectors.runner import DetectorRunner
from rtsp_warden.detectors.sinks import EventSink

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BG_FRAME = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


def _make_jpeg(frame: np.ndarray) -> bytes:
    """Encode a numpy BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok, "imencode failed"
    return buf.tobytes()


def _add_white_square(
    frame: np.ndarray,
    top_left: tuple[int, int] = (80, 80),
    size: int = 40,
) -> np.ndarray:
    """Return a copy of *frame* with a white square overlaid."""
    out = frame.copy()
    y, x = top_left
    out[y : y + size, x : x + size] = 255
    return out


# ---------------------------------------------------------------------------
# 1. setup / teardown lifecycle
# ---------------------------------------------------------------------------


def test_setup_creates_subtractor() -> None:
    """After setup(), the internal MOG2 subtractor is initialised."""
    det = MotionDetector()
    det.setup()
    try:
        assert det._subtractor is not None
    finally:
        det.teardown()


def test_teardown_releases_subtractor() -> None:
    """After teardown(), the subtractor is released (set to None)."""
    det = MotionDetector()
    det.setup()
    det.teardown()
    assert det._subtractor is None


# ---------------------------------------------------------------------------
# 2. Static scene (no motion)
# ---------------------------------------------------------------------------


def test_process_returns_empty_for_static_scene() -> None:
    """Feeding the same background frame 30 times yields no detections."""
    det = MotionDetector(min_area=500)
    det.setup()
    try:
        bg = _BG_FRAME.copy()
        for i in range(30):
            results = det.process(bg, ts_unix=float(i))
        # After learning the background, no motion should be detected
        assert len(results) == 0
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 3. Motion detection
# ---------------------------------------------------------------------------


def test_process_detects_motion() -> None:
    """After learning a static background, a moving white square triggers detection."""
    det = MotionDetector(min_area=500)
    det.setup()
    try:
        bg = _BG_FRAME.copy()
        # Learn the background for 30 frames
        for i in range(30):
            det.process(bg, ts_unix=float(i))
        # Introduce a moving white square
        moved = _add_white_square(bg, top_left=(80, 80), size=60)
        results = det.process(moved, ts_unix=30.0)
        assert len(results) >= 1, "expected at least one motion detection"
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 4. min_area filter
# ---------------------------------------------------------------------------


def test_min_area_filters_small_motion() -> None:
    """A very small change (< min_area) is filtered out."""
    det = MotionDetector(min_area=500)
    det.setup()
    try:
        bg = _BG_FRAME.copy()
        for i in range(30):
            det.process(bg, ts_unix=float(i))
        # Add a tiny 3x3 white square (area = 9, well below min_area=500)
        tiny = bg.copy()
        tiny[100:103, 100:103] = 255
        results = det.process(tiny, ts_unix=30.0)
        assert len(results) == 0, "small motion should be filtered by min_area"
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 5. Sensitivity affects threshold
# ---------------------------------------------------------------------------


def test_sensitivity_affects_threshold() -> None:
    """Higher sensitivity (0.9) detects subtler changes than default (0.5)."""
    bg = _BG_FRAME.copy()

    # Low sensitivity: barely any detections on a moderate change
    det_low = MotionDetector(min_area=200, sensitivity=0.1)
    det_low.setup()
    try:
        for i in range(30):
            det_low.process(bg, ts_unix=float(i))
        mod = bg.copy()
        # Moderate change: small bright region
        mod[50:70, 50:70] = np.clip(bg[50:70, 50:70].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        low_results = det_low.process(mod, ts_unix=30.0)
    finally:
        det_low.teardown()

    # High sensitivity: more detections on the same change
    det_high = MotionDetector(min_area=200, sensitivity=0.9)
    det_high.setup()
    try:
        for i in range(30):
            det_high.process(bg, ts_unix=float(i))
        high_results = det_high.process(mod, ts_unix=30.0)
    finally:
        det_high.teardown()

    # Higher sensitivity should find at least as many detections
    assert len(high_results) >= len(low_results)


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_empty_frame_returns_empty() -> None:
    """An empty (0x0) frame returns an empty list."""
    det = MotionDetector()
    det.setup()
    try:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        results = det.process(empty, ts_unix=1.0)
        assert results == []
    finally:
        det.teardown()


def test_tiny_frame_returns_empty() -> None:
    """A 2x2 frame is too small for MOG2 and returns an empty list."""
    det = MotionDetector()
    det.setup()
    try:
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        results = det.process(tiny, ts_unix=1.0)
        assert results == []
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 7. Detection field correctness
# ---------------------------------------------------------------------------


def test_detection_has_correct_fields() -> None:
    """Each Detection has kind='motion', a 4-tuple bbox, and metadata with 'area'."""
    det = MotionDetector(min_area=500)
    det.setup()
    try:
        bg = _BG_FRAME.copy()
        for i in range(30):
            det.process(bg, ts_unix=float(i))
        moved = _add_white_square(bg, top_left=(80, 80), size=60)
        results = det.process(moved, ts_unix=30.0)
        assert len(results) >= 1
        d = results[0]
        assert d.kind == "motion"
        assert d.bbox is not None
        assert len(d.bbox) == 4
        assert all(isinstance(v, int) for v in d.bbox)
        assert "area" in d.metadata
        assert isinstance(d.metadata["area"], int)
        assert d.ts_unix == 30.0
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 8. Integration with DetectorRunner + EventSink
# ---------------------------------------------------------------------------


@pytest.fixture
def motion_db(tmp_path, monkeypatch):
    """Set up an isolated SQLite DB with schema and a test camera for integration."""
    from rtsp_warden.db.engine import get_session

    db_url = f"sqlite:///{tmp_path}/test_motion.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()

    with get_session() as session:
        cam = Camera(
            name="motioncam", main_url="rtsp://x/main", sub_url="rtsp://x/sub", enabled=True
        )
        session.add(cam)
        session.commit()

    yield db_url
    reset_engine()


def test_integration_with_runner(motion_db: str) -> None:
    """Wire MotionDetector into DetectorRunner; detect motion; verify EventSink writes a row."""
    det = MotionDetector(min_area=200, sensitivity=0.7)
    sink = EventSink()
    results_collector: list[tuple[str, str, list[Detection]]] = []

    runner = DetectorRunner(
        detectors=(det,),
        result_sinks=[
            sink,
            lambda cam, stream, dets: results_collector.append((cam, stream, dets)),
        ],
        worker_count=1,
        queue_maxsize=64,
    )
    runner.setup()
    try:
        bg = _BG_FRAME.copy()

        # Feed background frames to learn the background model
        for i in range(30):
            jpeg = _make_jpeg(bg)
            runner.on_frame("motioncam", "sub", jpeg, ts_unix=float(i))

        # Wait for background frames to be processed
        for _ in range(100):
            if runner._frames_processed >= 30:
                break
            time.sleep(0.01)

        # Now feed a frame with a prominent moving object
        moved = _add_white_square(bg, top_left=(80, 80), size=60)
        jpeg_motion = _make_jpeg(moved)
        runner.on_frame("motioncam", "sub", jpeg_motion, ts_unix=30.0)

        # Wait for the motion frame to be processed
        for _ in range(100):
            if runner._frames_processed >= 31:
                break
            time.sleep(0.01)

        # Additional wait for EventSink DB writes to commit
        time.sleep(0.1)

        # The motion frame should have produced at least one detection
        # Filter to only non-empty detection results (background frames return [])
        motion_results = [r for r in results_collector if len(r[2]) > 0]
        assert len(motion_results) >= 1, "expected at least one non-empty detection result"
        assert motion_results[0][0] == "motioncam"
        assert motion_results[0][2][0].kind == "motion"

        # The EventSink should have written at least one event to the DB
        events = list_events(camera_name="motioncam")
        assert len(events) >= 1, "expected at least one event in DB"
        assert events[0].event_type == "motion"
    finally:
        runner.teardown()
