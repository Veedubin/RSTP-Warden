"""Shared fakes for the runtime wiring tests (RW-3 Task 12). Not collected (no test_ prefix).

Nothing here spawns a process, opens a socket or loads a model: StubOnnxDetector replaces
OnnxDetector at the registry's call-time import, RecordingAction stands in for a configured
action, stub_processes() turns process start/stop into recorded calls, and feed() pushes one
synthetic tap frame through a camera's own dispatcher and drains that camera's runner on the
calling thread, so the frame's ts_unix is the only clock.
"""

from __future__ import annotations

import io
import queue
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
import pytest
from rich.console import Console

import rtsp_warden.detectors.builtin.onnx as onnx_module
from rtsp_warden.actions.base import ActionPayload, ActionResult
from rtsp_warden.actions.queue import ActionQueue, ClipScheduler
from rtsp_warden.app import AppRuntime
from rtsp_warden.config import AppConfig
from rtsp_warden.detectors.base import Detection
from rtsp_warden.recorder import CameraRecorder

T = TypeVar("T")

T0 = 1_790_000_000.0  # frame time of tap frame 0 (2026-09-21 UTC); nothing reads the wall clock
STEP = 0.2  # seconds between tap frames at detect_fps 5
VISIBLE_UNTIL = T0 + 1.0  # the stub's person is in view for frames 0-4, then gone

MOTION: dict[str, Any] = {"type": "motion"}
ONNX: dict[str, Any] = {"type": "onnx", "model": "yolox-s", "device": "cpu"}
HOOK: dict[str, Any] = {"name": "hook", "type": "webhook", "url": "http://example.invalid/hook"}
PHONE: dict[str, Any] = {"name": "phone", "type": "webhook", "url": "http://example.invalid/phone"}
PERSON_RULE: dict[str, Any] = {
    "name": "person-any-time",
    "labels": ["person"],
    "min_confidence": 0.6,
    "cooldown_seconds": 60,
    "clip": True,
    "actions": ["hook"],
}


class StubOnnxDetector:
    """OnnxDetector stand-in: no model file, no onnxruntime session.

    process() reports one person whose box moves 4 px right per frame while
    ts_unix < VISIBLE_UNTIL, then nothing; it works without setup(). Setting the class
    attribute ``fail_with`` makes setup() behave like Task 5's OnnxDetector when its model is
    missing and cannot be downloaded: it never raises, it sets ``error`` and process()
    returns no detections.
    """

    name = "onnx"
    kind = "onnx"
    input_width = 640
    fail_with: str | None = None
    instances: list[StubOnnxDetector] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.provider: str | None = None
        self.fallback_warning: str | None = None
        self.error: str | None = None
        self.labels = ["person"]
        StubOnnxDetector.instances.append(self)

    @property
    def loaded(self) -> bool:
        return self.provider is not None

    def setup(self) -> None:
        if self.fail_with is not None:
            self.error = self.fail_with
            return
        self.provider = "CPUExecutionProvider"

    def process(self, frame_bgr: np.ndarray, ts_unix: float) -> list[Detection]:
        if self.error is not None or ts_unix >= VISIBLE_UNTIL:
            return []
        step = round((ts_unix - T0) / STEP)
        bbox = (40 + 4 * step, 30, 60, 120)
        return [Detection(kind="person", confidence=0.9, bbox=bbox, ts_unix=ts_unix)]

    def teardown(self) -> None:
        self.provider = None


def install_stub_onnx(monkeypatch: pytest.MonkeyPatch) -> type[StubOnnxDetector]:
    """Make the registry build StubOnnxDetector for every enabled ``type: onnx`` spec.

    The registry's onnx builder imports OnnxDetector from its module at call time
    (Task 5, ``_build_onnx_detector``), so replacing the module attribute is enough.
    """
    monkeypatch.setattr(onnx_module, "OnnxDetector", StubOnnxDetector)
    monkeypatch.setattr(StubOnnxDetector, "instances", [])
    monkeypatch.setattr(StubOnnxDetector, "fail_with", None)
    return StubOnnxDetector


class RecordingAction:
    """Action stand-in: keeps every payload and attachment it is sent; no network."""

    def __init__(self, name: str = "hook") -> None:
        self.name = name
        self.type = "webhook"
        self.sent: list[tuple[ActionPayload, Path | None]] = []

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        self.sent.append((payload, attachment))
        return ActionResult(ok=True)

    def test(self) -> ActionResult:
        return ActionResult(ok=True)


def camera(name: str, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Raw camera config: main stream only, recording into tmp_path/rec, proxy off."""
    raw: dict[str, Any] = {
        "name": name,
        "main_url": f"rtsp://u:p@h/{name}",
        "record": {"enabled": True, "output_dir": str(tmp_path / "rec")},
        "proxy": {"enabled": False},
    }
    raw.update(overrides)
    return raw


def make_runtime(
    tmp_path: Path,
    cameras: list[dict[str, Any]],
    *,
    actions: tuple[dict[str, Any], ...] = (HOOK,),
    public_url: str = "http://warden.test:8080",
) -> AppRuntime:
    """An unbuilt AppRuntime over validated config; models_dir is an empty tmp directory."""
    cfg = AppConfig.model_validate(
        {
            "cameras": cameras,
            "actions": list(actions),
            "runtime": {"models_dir": str(tmp_path / "models")},
        }
    )
    return AppRuntime(cfg=cfg, console=Console(file=io.StringIO()), public_url=public_url)


def stub_processes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace recorder, action queue and clip scheduler start/stop with recorded calls."""
    calls: list[tuple[str, str]] = []

    def record(action: str) -> Callable[..., None]:
        def stub(self: Any, *_args: Any, **_kwargs: Any) -> None:
            camera_cfg = getattr(self, "camera", None)
            calls.append((action, camera_cfg.name if camera_cfg is not None else ""))

        return stub

    monkeypatch.setattr(CameraRecorder, "start", record("rec.start"))
    monkeypatch.setattr(CameraRecorder, "stop", record("rec.stop"))
    monkeypatch.setattr(ActionQueue, "start", record("queue.start"))
    monkeypatch.setattr(ActionQueue, "stop", record("queue.stop"))
    monkeypatch.setattr(ClipScheduler, "start", record("clips.start"))
    monkeypatch.setattr(ClipScheduler, "stop", record("clips.stop"))
    return calls


def start_runtime(rt: AppRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """rt.start() without replacing the test process's SIGINT/SIGTERM handlers."""
    monkeypatch.setattr(rt, "_install_signals", lambda: None)
    rt.start()


def jpeg_frame(step: int, width: int = 320, height: int = 180) -> bytes:
    """A tap-sized JPEG with a white rectangle where the stub reports the person."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = 40 + 4 * step
    cv2.rectangle(img, (x, 30), (x + 60, 150), (255, 255, 255), thickness=-1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def feed(rt: AppRuntime, camera_name: str, step: int) -> None:
    """Dispatch tap frame *step* on the camera's own dispatcher and process it now.

    The runner is never set up here (no worker thread): the job it queued is run on this
    thread through the runner's own _process_job, so frames are handled in order with
    ts_unix = T0 + step * STEP.
    """
    cam_rt = rt.find_camera(camera_name)
    assert cam_rt is not None
    cam_rt.dispatcher.dispatch(
        camera=camera_name,
        stream=cam_rt.camera.proxy.stream,
        jpeg_bytes=jpeg_frame(step),
        ts_unix=T0 + step * STEP,
    )
    runner = rt.find_runner(camera_name)
    if runner is None:
        return
    while True:
        try:
            job = runner._queue.get_nowait()
        except queue.Empty:
            return
        runner._process_job(job)


def wait_for(probe: Callable[[], T], timeout_s: float = 5.0) -> T:
    """Poll *probe* until it returns something truthy (a worker thread's result)."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = probe()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {probe}")
        time.sleep(0.01)
