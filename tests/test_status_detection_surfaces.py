"""The detection block in cli.build_status: /status.json, /health and `rtsp-warden status`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtsp_warden.cli import build_status
from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.detection import camera_badge
from tests.helpers_runtime import (
    ONNX,
    StubOnnxDetector,
    camera,
    install_stub_onnx,
    make_runtime,
    start_runtime,
    stub_processes,
)

FALLBACK = "CUDA requested but unavailable; running on CPUExecutionProvider"


def _row(index: int, type_: str, fps: float, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index,
        "type": type_,
        "model": "yolox-s" if type_ == "onnx" else None,
        "device": "cuda" if type_ == "onnx" else None,
        "provider": "CPUExecutionProvider" if type_ == "onnx" else None,
        "fallback_warning": None,
        "fps": fps,
        "processed": 10,
        "skipped": 0,
        "errors": 0,
    }
    row.update(extra)
    return row


def _raw(*rows: dict[str, Any], dropped: Any = 0) -> dict[str, Any]:
    return {
        "name": "detector_cam",
        "frames_processed": 40,
        "frames_dropped": dropped,
        "errors_total": 0,
        "detectors": list(rows),
    }


def _runtime(raw: Any) -> SimpleNamespace:
    """A fake AppRuntime with the attributes build_status reads, plus detection_status."""
    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    cam_rt = SimpleNamespace(
        camera=cam, hub=None, proxy=None, recorder=SimpleNamespace(processes=lambda: [])
    )
    return SimpleNamespace(
        cameras=[cam_rt], detection_status=lambda name: raw if name == "cam" else None
    )


def test_build_status_adds_detection_block_and_warnings_to_errors() -> None:
    raw = _raw(
        _row(0, "motion", 5.0),
        _row(1, "onnx", 2.0, fallback_warning=FALLBACK),
        dropped=np.int64(7),
    )
    rt = _runtime(raw)
    status = build_status(rt, AppConfig(cameras=[rt.cameras[0].camera]), version="t")
    cam = status["cameras"][0]
    assert cam["detection"]["provider"] == "CPUExecutionProvider"
    assert cam["detection"]["fallback_warning"] == FALLBACK
    assert cam["detection"]["dropped"] == 7
    assert cam["detection"]["processed"] == 40
    assert [d["fps"] for d in cam["detection"]["detectors"]] == [5.0, 2.0]
    assert status["errors"] == [f"cam: detector 1 (onnx): {FALLBACK}"]
    assert cam["ok"] is True and status["ok"] is True  # a CPU fallback is not an outage
    json.dumps(status, allow_nan=False)


def test_build_status_without_a_runner_has_detection_none() -> None:
    rt = _runtime(None)
    status = build_status(rt, AppConfig(cameras=[rt.cameras[0].camera]), version="t")
    assert status["cameras"][0]["detection"] is None
    assert status["errors"] == []


def test_build_status_with_a_runtime_lacking_detection_status() -> None:
    rt = _runtime(None)
    del rt.detection_status
    status = build_status(rt, AppConfig(cameras=[rt.cameras[0].camera]), version="t")
    assert status["cameras"][0]["detection"] is None


def _client(raw: Any) -> TestClient:
    rt = _runtime(raw)
    cfg = AppConfig(cameras=[rt.cameras[0].camera])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: rt, runtime=rt)
    return TestClient(app)


def test_status_json_includes_detection(clean_db: None) -> None:
    raw = _raw(_row(1, "onnx", 2.0, fallback_warning=FALLBACK), dropped=np.int64(7))
    r = _client(raw).get("/status.json")
    assert r.status_code == 200
    detection = r.json()["cameras"][0]["detection"]
    assert detection["fallback_warning"] == FALLBACK
    assert detection["dropped"] == 7


def test_health_page_lists_detection_warnings(clean_db: None) -> None:
    """(review focus 1) The model error reaches the existing Errors list on /health."""
    raw = _raw(_row(1, "onnx", 2.0, provider=None, error="model yolox-s unavailable"))
    r = _client(raw).get("/health", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert "cam: detector 1 (onnx): model yolox-s unavailable" in r.text


def test_real_runtime_reports_a_model_that_cannot_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review focus 1) An offline model: status.json, /health and the badge say why.

    A real AppRuntime (Task 12's helpers): the stub onnx detector's setup() fails the way
    Task 5's OnnxDetector does without its model file, the camera without detectors has no
    detection block, and the whole payload is JSON-safe.
    """
    install_stub_onnx(monkeypatch)
    monkeypatch.setattr(StubOnnxDetector, "fail_with", "yolox-s: download failed (offline)")
    calls = stub_processes(monkeypatch)
    cams = [camera("front", tmp_path, detectors=[ONNX]), camera("idle", tmp_path)]
    rt = make_runtime(tmp_path, cams, actions=())
    rt.build()
    try:
        start_runtime(rt, monkeypatch)
        status = build_status(rt, rt.cfg, version="t")
        badge = camera_badge(rt, "front")
    finally:
        rt.stop_all()

    assert ("rec.start", "front") in calls  # recording was started anyway
    by_name = {c["name"]: c for c in status["cameras"]}
    assert by_name["idle"]["detection"] is None
    front = by_name["front"]["detection"]
    assert front is not None
    assert front["detectors"][0]["error"] == "yolox-s: download failed (offline)"
    assert "front: detector 0 (onnx): yolox-s: download failed (offline)" in status["errors"]
    assert badge is not None and badge["text"] == "detection error"
    json.dumps(status, allow_nan=False)


def test_status_json_redacts_userinfo_in_stderr_tail() -> None:
    """/status.json needs no login, and ffmpeg prints the expanded camera URL when it cannot
    open its input: the user:password part of every URL in stderr_tail is masked."""

    class _Proc:
        args: list[str] = []

        def is_running(self) -> bool:
            return False

        def pid(self) -> None:
            return None

        def stderr_tail(self) -> list[str]:
            return [
                "[in#0] Error opening input rtsp://u:secretpw@h/m: Server returned 401",
                "retrying rtsps://admin:p@ss@cam.local:322/stream1",
                "no credentials here: rtsp://h/m",
            ]

    cam = CameraConfig(name="cam", main_url="rtsp://u:secretpw@h/m")
    ingest = SimpleNamespace(
        stream_name="main",
        upstream_url=cam.main_url,
        proc=_Proc(),
        record_cfg=None,
        record_output_dir=None,
        mjpeg_hub=None,
        rtsp_publish_url=None,
    )
    cam_rt = SimpleNamespace(
        camera=cam,
        hub=None,
        proxy=None,
        recorder=SimpleNamespace(processes=lambda: [ingest], main=ingest, sub=None),
    )
    rt = SimpleNamespace(cameras=[cam_rt], detection_status=lambda name: None)

    text = json.dumps(build_status(rt, AppConfig(cameras=[cam]), version="t"))

    assert "secretpw" not in text
    assert "p@ss" not in text and "ss@cam.local" not in text
    assert "rtsp://***:***@h/m" in text
    assert "rtsps://***:***@cam.local:322/stream1" in text
    assert "rtsp://h/m" in text


def test_console_status_table_masks_userinfo() -> None:
    """The supervisor's console table prints the last ffmpeg stderr line: no password there."""
    import io

    from rich.console import Console

    from rtsp_warden.app import AppRuntime

    proc = SimpleNamespace(
        is_running=lambda: False,
        poll=lambda: 1,
        stderr_tail=lambda: ["[in#0] Error opening input rtsp://u:secretpw@h/m"],
    )
    cam = CameraConfig(name="cam", main_url="rtsp://u:secretpw@h/m")
    out = io.StringIO()
    rt = AppRuntime(cfg=AppConfig(cameras=[cam]), console=Console(file=out, width=300))
    rt.cameras = [
        SimpleNamespace(
            camera=cam,
            proxy=None,
            hub=None,
            recorder=SimpleNamespace(main=SimpleNamespace(proc=proc), sub=None),
        )
    ]

    rt._status_table()

    assert "secretpw" not in out.getvalue()
    assert "***:***@h/m" in out.getvalue()
