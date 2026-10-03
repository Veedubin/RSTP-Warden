"""status_model.summarize_detection / camera_detection_summary: the JSON-safe detection block."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rtsp_warden import status_model
from rtsp_warden.status_model import camera_detection_summary, summarize_detection

FALLBACK = "CUDA requested but unavailable; running on CPUExecutionProvider"
MODEL_ERROR = "model yolox-s unavailable: download failed (offline)"


def _onnx_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": 1,
        "type": "onnx",
        "model": "yolox-s",
        "device": "auto",
        "provider": "CPUExecutionProvider",
        "fallback_warning": None,
        "fps": 2.0,
        "processed": 10,
        "skipped": 30,
        "errors": 0,
    }
    row.update(overrides)
    return row


def _motion_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": 0,
        "type": "motion",
        "model": None,
        "device": None,
        "provider": None,
        "fallback_warning": None,
        "fps": 5.0,
        "processed": 40,
        "skipped": 0,
        "errors": 0,
    }
    row.update(overrides)
    return row


def _raw(*rows: dict[str, Any], dropped: Any = 0, processed: Any = 40) -> dict[str, Any]:
    """Shape of AppRuntime.detection_status(name): runner.status() plus runtime keys."""
    return {
        "name": "detector_cam",
        "frames_processed": processed,
        "frames_dropped": dropped,
        "detections_total": 3,
        "errors_total": 0,
        "queue_size": 0,
        "worker_count": 1,
        "detector_count": len(rows),
        "detectors": list(rows),
        "restart_pending": False,
    }


@pytest.mark.parametrize("raw", [None, "x", 3, ["detectors"]])
def test_summarize_detection_is_none_for_non_mappings(raw: Any) -> None:
    assert summarize_detection(raw) is None


def test_summarize_detection_is_none_for_a_camera_without_a_runner() -> None:
    """AppRuntime.detection_status() of a camera with no enabled detector (Task 12 shape)."""
    idle = {
        "camera": "idle",
        "enabled": False,
        "tap_fps": 5.0,
        "tap_width": 320,
        "restart_pending": False,
    }
    assert summarize_detection(idle) is None
    assert summarize_detection({"frames_processed": 0}) is None  # no "detectors" list


def test_summarize_detection_casts_numpy_scalars_to_plain_numbers() -> None:
    raw = _raw(
        _onnx_row(fps=np.float32(2.0), processed=np.int64(10), skipped=np.int64(30)),
        dropped=np.int64(7),
        processed=np.int64(40),
    )
    out = summarize_detection(raw)
    assert out is not None
    assert type(out["dropped"]) is int and out["dropped"] == 7
    assert type(out["processed"]) is int and out["processed"] == 40
    det = out["detectors"][0]
    assert type(det["fps"]) is float and det["fps"] == 2.0
    assert type(det["processed"]) is int and type(det["skipped"]) is int
    json.dumps(out, allow_nan=False)  # what JSONResponse does


def test_summarize_detection_drops_unknown_keys_and_non_finite_fps() -> None:
    out = summarize_detection(_raw(_onnx_row(fps=float("nan"), secret="x")))
    assert out is not None
    det = out["detectors"][0]
    assert det["fps"] is None
    assert "secret" not in det
    assert set(out) == {
        "provider",
        "fallback_warning",
        "processed",
        "dropped",
        "errors",
        "warnings",
        "detectors",
    }


def test_summarize_detection_index_falls_back_to_list_position() -> None:
    row = _onnx_row()
    del row["index"]
    out = summarize_detection(_raw(_motion_row(), row))
    assert out is not None
    assert [d["index"] for d in out["detectors"]] == [0, 1]


def test_summarize_detection_collects_provider_and_fallback_warning() -> None:
    out = summarize_detection(
        _raw(_motion_row(), _onnx_row(device="cuda", fallback_warning=FALLBACK))
    )
    assert out is not None
    assert out["provider"] == "CPUExecutionProvider"
    assert out["fallback_warning"] == FALLBACK
    assert out["warnings"] == [f"detector 1 (onnx): {FALLBACK}"]


def test_summarize_detection_reports_a_model_load_error() -> None:
    """(review focus 1) An offline camera whose model cannot load says why, not just zeros."""
    out = summarize_detection(_raw(_onnx_row(provider=None, error=MODEL_ERROR)))
    assert out is not None
    assert out["provider"] is None
    assert out["detectors"][0]["error"] == MODEL_ERROR
    assert out["warnings"] == [f"detector 1 (onnx): {MODEL_ERROR}"]


def test_summarize_detection_reports_a_setup_exception() -> None:
    """A detector whose setup() raised (runner "setup_error") is reported like a load error."""
    out = summarize_detection(_raw(_onnx_row(provider=None, setup_error="RuntimeError: boom")))
    assert out is not None
    assert out["detectors"][0]["error"] == "RuntimeError: boom"
    assert out["warnings"] == ["detector 1 (onnx): RuntimeError: boom"]


def test_summarize_detection_cuts_long_text() -> None:
    out = summarize_detection(_raw(_onnx_row(error="x" * 1000)))
    assert out is not None
    assert len(out["detectors"][0]["error"]) == 300


def test_camera_detection_summary_without_the_method_is_none() -> None:
    assert camera_detection_summary(None, "cam") is None
    assert camera_detection_summary(SimpleNamespace(cameras=[]), "cam") is None


def test_camera_detection_summary_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(status_model.log, "debug", lambda *a, **k: calls.append(a))

    def boom(name: str) -> dict:
        raise RuntimeError("runner gone")

    assert camera_detection_summary(SimpleNamespace(detection_status=boom), "cam") is None
    assert calls and calls[0][1] == "cam"


def test_camera_detection_summary_returns_the_summary() -> None:
    rt = SimpleNamespace(detection_status=lambda name: _raw(_motion_row(), dropped=4))
    out = camera_detection_summary(rt, "cam")
    assert out is not None and out["dropped"] == 4
