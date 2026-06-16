"""Tests for detectors/builtin/vehicle.py: VehicleDetector."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from rtsp_warden.detectors.builtin.vehicle import VehicleDetector

# ---------------------------------------------------------------------------
# 1. setup / teardown lifecycle
# ---------------------------------------------------------------------------


def test_setup_loads_classifier() -> None:
    """After setup(), the classifier is loaded and _loaded is True."""
    det = VehicleDetector()
    det.setup()
    try:
        assert det._loaded is True
        assert det._classifier is not None
    finally:
        det.teardown()


def test_teardown_unloads() -> None:
    """After teardown(), _loaded is False and classifier is None."""
    det = VehicleDetector()
    det.setup()
    det.teardown()
    assert det._loaded is False
    assert det._classifier is None


# ---------------------------------------------------------------------------
# 2. Edge cases: empty / tiny frames
# ---------------------------------------------------------------------------


def test_process_on_empty_frame_returns_empty() -> None:
    """An empty (0x0) frame returns an empty list."""
    det = VehicleDetector()
    det.setup()
    try:
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        results = det.process(empty, ts_unix=1.0)
        assert results == []
    finally:
        det.teardown()


def test_process_on_tiny_frame_returns_empty() -> None:
    """A 2x2 frame is too small for Haar cascade and returns an empty list."""
    det = VehicleDetector()
    det.setup()
    try:
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        results = det.process(tiny, ts_unix=1.0)
        assert results == []
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 3. Real frame (no cars expected)
# ---------------------------------------------------------------------------


def test_process_runs_without_error_on_real_frame() -> None:
    """Feeding a 640x480 random frame produces no exception; result is a list."""
    det = VehicleDetector()
    det.setup()
    try:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = det.process(frame, ts_unix=1.0)
        assert isinstance(results, list)
        # Random noise is unlikely to trigger car detection but we don't
        # assert empty since Haar cascades can produce false positives
    finally:
        det.teardown()


# ---------------------------------------------------------------------------
# 4. Missing cascade file handling
# ---------------------------------------------------------------------------


def test_missing_cascade_file_handled_gracefully() -> None:
    """If no cascade file is found, setup() fails gracefully and process() returns []."""
    det = VehicleDetector()
    # Patch _find_cascade to return None, simulating no cascade files available
    with patch("rtsp_warden.detectors.builtin.vehicle._find_cascade", return_value=None):
        det.setup()
    # _loaded should be False since no cascade found
    assert det._loaded is False
    # process() should return [] without crashing
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = det.process(frame, ts_unix=1.0)
    assert results == []
