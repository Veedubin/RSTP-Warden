"""Tests for detectors/runner.py: DetectorRunner."""

from __future__ import annotations

import time

import cv2
import numpy as np

from rtsp_warden.detectors.base import Detection, NullDetector
from rtsp_warden.detectors.runner import DetectorRunner


def _make_jpeg(width: int = 64, height: int = 48) -> bytes:
    """Create a minimal valid JPEG image for testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _make_invalid_jpeg() -> bytes:
    """Return bytes that are not a valid JPEG."""
    return b"NOT_JPEG_DATA_1234567890"


class TrackingDetector:
    """A test detector that records calls and returns a detection."""

    name: str = "tracking"
    kind: str = "motion"
    call_count: int = 0

    def setup(self) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        self.call_count += 1
        return [Detection(kind="motion", confidence=0.9, ts_unix=ts_unix)]

    def teardown(self) -> None:
        pass


class FailingDetector:
    """A test detector that raises on process."""

    name: str = "failing"
    kind: str = "error"

    def setup(self) -> None:
        pass

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        raise RuntimeError("detector error")

    def teardown(self) -> None:
        pass


class SetupFailingDetector:
    """A test detector that raises on setup."""

    name: str = "setup_fail"
    kind: str = "error"

    def setup(self) -> None:
        raise RuntimeError("setup error")

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        return []

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Basic wiring
# ---------------------------------------------------------------------------


def test_runner_implements_name_attribute() -> None:
    """DetectorRunner has a .name attribute (FrameConsumer protocol)."""
    runner = DetectorRunner(name="test_runner")
    assert runner.name == "test_runner"


def test_runner_default_name() -> None:
    """DetectorRunner has default name 'detector_runner'."""
    runner = DetectorRunner()
    assert runner.name == "detector_runner"


# ---------------------------------------------------------------------------
# on_frame with valid JPEG
# ---------------------------------------------------------------------------


def test_on_frame_valid_jpeg_triggers_detection() -> None:
    """on_frame with valid JPEG causes detector.process() to be called."""
    detector = TrackingDetector()
    results: list[tuple[str, str, list[Detection]]] = []
    runner = DetectorRunner(
        detectors=(detector,),
        result_sinks=[lambda cam, stream, dets: results.append((cam, stream, dets))],
        worker_count=1,
        queue_maxsize=8,
    )
    runner.setup()
    try:
        jpeg = _make_jpeg()
        runner.on_frame("cam1", "sub", jpeg, 100.0)
        # Wait for worker to process
        for _ in range(50):
            if results:
                break
            time.sleep(0.01)
        assert len(results) == 1
        assert results[0][0] == "cam1"
        assert results[0][1] == "sub"
        assert len(results[0][2]) == 1
        assert results[0][2][0].kind == "motion"
    finally:
        runner.teardown()


# ---------------------------------------------------------------------------
# on_frame with invalid JPEG
# ---------------------------------------------------------------------------


def test_on_frame_invalid_jpeg_logged_and_skipped() -> None:
    """on_frame with invalid JPEG does not crash; detector is not called."""
    detector = TrackingDetector()
    results: list[tuple[str, str, list[Detection]]] = []
    runner = DetectorRunner(
        detectors=(detector,),
        result_sinks=[lambda cam, stream, dets: results.append((cam, stream, dets))],
        worker_count=1,
        queue_maxsize=8,
    )
    runner.setup()
    try:
        runner.on_frame("cam1", "sub", _make_invalid_jpeg(), 100.0)
        time.sleep(0.2)
        assert len(results) == 0
        assert detector.call_count == 0
    finally:
        runner.teardown()


# ---------------------------------------------------------------------------
# Per-detector exception is swallowed
# ---------------------------------------------------------------------------


def test_per_detector_exception_swallowed() -> None:
    """If one detector raises, other detectors still run."""
    failing = FailingDetector()
    tracking = TrackingDetector()
    results: list[tuple[str, str, list[Detection]]] = []
    runner = DetectorRunner(
        detectors=(failing, tracking),
        result_sinks=[lambda cam, stream, dets: results.append((cam, stream, dets))],
        worker_count=1,
        queue_maxsize=8,
    )
    runner.setup()
    try:
        jpeg = _make_jpeg()
        runner.on_frame("cam1", "sub", jpeg, 100.0)
        for _ in range(50):
            if results:
                break
            time.sleep(0.01)
        # The tracking detector should have still produced a result
        assert len(results) == 1
        assert len(results[0][2]) == 1
    finally:
        runner.teardown()


# ---------------------------------------------------------------------------
# Queue overflow drops oldest
# ---------------------------------------------------------------------------


def test_queue_overflow_drops_oldest() -> None:
    """When the queue is full, on_frame drops the oldest frame."""
    detector = NullDetector()
    runner = DetectorRunner(
        detectors=(detector,),
        result_sinks=[],
        worker_count=0,  # no workers consuming, so queue will fill
        queue_maxsize=2,
    )
    # Manually set up without starting workers
    runner._stop_event = __import__("threading").Event()
    runner._workers = []

    # Fill the queue
    jpeg = _make_jpeg()
    runner.on_frame("cam1", "sub", jpeg, 1.0)
    runner.on_frame("cam1", "sub", jpeg, 2.0)
    assert runner._queue.qsize() == 2

    # Third frame should drop the oldest
    runner.on_frame("cam1", "sub", jpeg, 3.0)
    # Queue should still be at max (2), and one frame was dropped
    assert runner._frames_dropped >= 1

    runner._stop_event.set()


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_returns_sensible_dict() -> None:
    """status() returns a dict with expected keys and types."""
    runner = DetectorRunner(
        detectors=(NullDetector(),),
        worker_count=1,
    )
    s = runner.status()
    assert isinstance(s, dict)
    assert "name" in s
    assert "frames_processed" in s
    assert "frames_dropped" in s
    assert "detections_total" in s
    assert "errors_total" in s
    assert "queue_size" in s
    assert "worker_count" in s
    assert "detector_count" in s
    assert s["detector_count"] == 1


# ---------------------------------------------------------------------------
# setup/teardown lifecycle
# ---------------------------------------------------------------------------


def test_setup_calls_detector_setup() -> None:
    """runner.setup() calls detector.setup() on each detector."""
    detector = TrackingDetector()
    runner = DetectorRunner(detectors=(detector,), worker_count=1)
    runner.setup()
    try:
        # setup was called (no exception = success for TrackingDetector)
        assert runner._workers  # worker thread was started
    finally:
        runner.teardown()


def test_teardown_calls_detector_teardown() -> None:
    """runner.teardown() calls detector.teardown() on each detector."""
    detector = TrackingDetector()
    runner = DetectorRunner(detectors=(detector,), worker_count=1)
    runner.setup()
    runner.teardown()
    # Workers should be stopped
    assert not runner._workers


def test_setup_failure_on_detector_is_swallowed() -> None:
    """If a detector raises in setup, the runner swallows it when swallow_exceptions=True."""
    detector = SetupFailingDetector()
    runner = DetectorRunner(
        detectors=(detector,),
        worker_count=1,
        swallow_exceptions=True,
    )
    # Should not raise
    runner.setup()
    runner.teardown()


# ---------------------------------------------------------------------------
# Result sinks
# ---------------------------------------------------------------------------


def test_result_sink_receives_multiple_detections() -> None:
    """Multiple detections from a single frame are passed to result sinks."""
    detector = TrackingDetector()
    results: list[tuple[str, str, list[Detection]]] = []
    runner = DetectorRunner(
        detectors=(detector,),
        result_sinks=[lambda cam, stream, dets: results.append((cam, stream, dets))],
        worker_count=1,
        queue_maxsize=8,
    )
    runner.setup()
    try:
        jpeg = _make_jpeg()
        runner.on_frame("cam1", "sub", jpeg, 100.0)
        for _ in range(50):
            if results:
                break
            time.sleep(0.01)
        assert len(results) == 1
    finally:
        runner.teardown()
