"""Frame tap wiring: StreamIngestor, CameraRecorder, AppRuntime, DetectorRunner and serve.

Every camera has its own FrameTapDispatcher on ``CameraRuntime.dispatcher``, created before
its recorder. Only the proxy stream's ingestor carries the tap. The dispatcher holds that
camera's runner (one worker, short queue), then the ``--frame-consumer`` consumers. Nothing
here spawns ffmpeg: ``build()`` only constructs objects, and the lifecycle tests stub
``CameraRecorder.start`` / ``stop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

import rtsp_warden.cli as cli_mod
import rtsp_warden.db.bootstrap as bootstrap_mod
from rtsp_warden.app import AppRuntime, CameraRuntime
from rtsp_warden.config import (
    AppConfig,
    CameraConfig,
    ProxyConfig,
    RecordConfig,
    RuntimeConfig,
)
from rtsp_warden.consumers.motion_demo import MotionHeuristicConsumer
from rtsp_warden.detectors.registry import DetectorSpec
from rtsp_warden.detectors.runner import DetectorRunner
from rtsp_warden.frame_tap import FrameConsumer, FrameTapDispatcher
from rtsp_warden.recorder import CameraRecorder, StreamIngestor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubConsumer(FrameConsumer):
    name: str = "stub"

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        pass


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _cam(
    name: str,
    tmp_path: Path,
    *,
    sub: bool = False,
    record: bool = True,
    proxy: bool = False,
    port: int = 9001,
    motion: bool = False,
) -> CameraConfig:
    """Camera on rtsp://u:p@h/<name>[/sub]; MJPEG proxy only when ``proxy`` is True."""
    return CameraConfig(
        name=name,
        main_url=f"rtsp://u:p@h/{name}",
        sub_url=f"rtsp://u:p@h/{name}/sub" if sub else None,
        record=RecordConfig(enabled=record, output_dir=tmp_path / "rec"),
        proxy=ProxyConfig(enabled=proxy, port=port),
        detectors=[DetectorSpec(type="motion")] if motion else [],
    )


def _runtime(cams: list[CameraConfig], **kwargs: Any) -> AppRuntime:
    """A built AppRuntime (objects only; build() starts nothing)."""
    runtime = AppRuntime(cfg=AppConfig(cameras=cams, runtime=RuntimeConfig()), **kwargs)
    runtime.build()
    return runtime


def _rt(runtime: AppRuntime, name: str) -> CameraRuntime:
    rt = runtime.find_camera(name)
    assert rt is not None
    return rt


def _only_runner(rt: CameraRuntime) -> DetectorRunner:
    consumers = tuple(rt.dispatcher.consumers)
    assert len(consumers) == 1
    runner = consumers[0]
    assert isinstance(runner, DetectorRunner)
    return runner


# ---------------------------------------------------------------------------
# StreamIngestor and CameraRecorder fields
# ---------------------------------------------------------------------------


def test_stream_ingestor_has_frame_tap_fields() -> None:
    """StreamIngestor carries the tap flag, fps, width and dispatcher."""
    ingestor = StreamIngestor(
        camera_name="test",
        stream_name="main",
        upstream_url="rtsp://example.com/main",
        runtime=RuntimeConfig(),
    )
    assert ingestor.frame_tap_enabled is False
    assert ingestor.frame_tap_fps == 5.0
    assert ingestor.frame_tap_scale_width == 320
    assert ingestor.frame_tap_dispatcher is None
    assert ingestor.frame_tap_write_fd is None

    dispatcher = FrameTapDispatcher()
    ingestor2 = StreamIngestor(
        camera_name="test2",
        stream_name="sub",
        upstream_url="rtsp://example.com/sub",
        runtime=RuntimeConfig(),
        frame_tap_enabled=True,
        frame_tap_fps=10,
        frame_tap_scale_width=160,
        frame_tap_dispatcher=dispatcher,
    )
    assert ingestor2.frame_tap_enabled is True
    assert ingestor2.frame_tap_fps == 10
    assert ingestor2.frame_tap_scale_width == 160
    assert ingestor2.frame_tap_dispatcher is dispatcher


def test_camera_recorder_accepts_dispatcher(tmp_path: Path) -> None:
    dispatcher = FrameTapDispatcher(consumers=[StubConsumer()])
    recorder = CameraRecorder(
        camera=_cam("c", tmp_path, sub=True),
        runtime=RuntimeConfig(),
        frame_tap_dispatcher=dispatcher,
    )
    assert recorder.frame_tap_dispatcher is dispatcher
    assert recorder.frame_tap_required is False


# ---------------------------------------------------------------------------
# CameraRecorder: the tap rides on the proxy stream only
# ---------------------------------------------------------------------------


def test_tap_rides_on_the_sub_stream_when_the_proxy_reads_sub(tmp_path: Path) -> None:
    cam = _cam("c", tmp_path, sub=True)
    assert cam.proxy.stream == "sub"
    dispatcher = FrameTapDispatcher()

    rec = CameraRecorder(camera=cam, runtime=RuntimeConfig(), frame_tap_dispatcher=dispatcher)

    assert rec.main is not None and rec.sub is not None
    assert rec.sub.frame_tap_enabled is True
    assert rec.sub.frame_tap_dispatcher is dispatcher
    assert rec.main.frame_tap_enabled is False
    assert rec.main.frame_tap_dispatcher is None


def test_tap_follows_an_explicit_proxy_stream_main(tmp_path: Path) -> None:
    cam = _cam("c", tmp_path, sub=True)
    cam.proxy.stream = "main"
    dispatcher = FrameTapDispatcher()

    rec = CameraRecorder(camera=cam, runtime=RuntimeConfig(), frame_tap_dispatcher=dispatcher)

    assert rec.main is not None and rec.sub is not None
    assert rec.main.frame_tap_enabled is True
    assert rec.sub.frame_tap_enabled is False


def test_main_only_camera_taps_main(tmp_path: Path) -> None:
    cam = _cam("c", tmp_path)
    assert cam.proxy.stream == "main"

    rec = CameraRecorder(
        camera=cam, runtime=RuntimeConfig(), frame_tap_dispatcher=FrameTapDispatcher()
    )

    assert rec.sub is None
    assert rec.main is not None
    assert rec.main.frame_tap_enabled is True


def test_no_dispatcher_means_no_tap(tmp_path: Path) -> None:
    rec = CameraRecorder(camera=_cam("c", tmp_path, sub=True), runtime=RuntimeConfig())

    assert rec.main is not None and rec.sub is not None
    assert rec.main.frame_tap_enabled is False
    assert rec.sub.frame_tap_enabled is False


def test_the_tap_alone_creates_an_ingestor_only_when_required(tmp_path: Path) -> None:
    cam = _cam("c", tmp_path, record=False, proxy=False)

    idle = CameraRecorder(
        camera=cam, runtime=RuntimeConfig(), frame_tap_dispatcher=FrameTapDispatcher()
    )
    wanted = CameraRecorder(
        camera=cam,
        runtime=RuntimeConfig(),
        frame_tap_dispatcher=FrameTapDispatcher(),
        frame_tap_required=True,
    )

    assert idle.frame_tap_required is False
    assert idle.main is None
    assert not idle.has_any()  # a camera nothing consumes spawns no ffmpeg
    assert wanted.main is not None
    assert wanted.main.frame_tap_enabled is True
    assert wanted.main.mjpeg_hub is None
    assert wanted.main.rtsp_publish_url is None
    assert wanted.main.record_cfg is not None
    assert wanted.main.record_cfg.enabled is False


# ---------------------------------------------------------------------------
# AppRuntime.build: one dispatcher per camera, holding exactly its runner
# ---------------------------------------------------------------------------


def test_build_gives_each_camera_its_own_dispatcher(tmp_path: Path) -> None:
    runtime = _runtime(
        [
            _cam("a", tmp_path, sub=True, motion=True),
            _cam("b", tmp_path, motion=True),
            _cam("c", tmp_path),
        ]
    )
    a, b, c = runtime.cameras

    assert len({id(a.dispatcher), id(b.dispatcher), id(c.dispatcher)}) == 3
    for rt in (a, b, c):
        assert rt.recorder.frame_tap_dispatcher is rt.dispatcher
    runner_a, runner_b = _only_runner(a), _only_runner(b)
    assert runner_a.name == "detector_a" and runner_a.camera == "a"
    assert runner_b.name == "detector_b" and runner_b.camera == "b"
    assert tuple(c.dispatcher.consumers) == ()
    assert [r.name for r in runtime.detector_runners] == ["detector_a", "detector_b"]

    # The tap rides on the proxy stream only: sub for a, main for b and c.
    assert a.recorder.sub is not None and a.recorder.sub.frame_tap_enabled is True
    assert a.recorder.main is not None and a.recorder.main.frame_tap_enabled is False
    assert b.recorder.main is not None and b.recorder.main.frame_tap_enabled is True
    assert c.recorder.main is not None and c.recorder.main.frame_tap_enabled is True
    assert a.recorder.sub.frame_tap_dispatcher is a.dispatcher


def test_runtime_runners_use_one_worker_and_a_short_queue(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True)])

    runner = _only_runner(runtime.cameras[0])

    assert runner.worker_count == 1
    assert runner.queue_maxsize == 8


def test_a_frame_reaches_only_its_own_cameras_runner(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True), _cam("b", tmp_path, motion=True)])
    a, b = runtime.cameras
    runner_a, runner_b = _only_runner(a), _only_runner(b)

    # What b's tap reader thread does for every frame (runners not set up: no workers).
    b.dispatcher.dispatch(camera="b", stream="main", jpeg_bytes=_jpeg(), ts_unix=1.0)

    assert runner_a.status()["queue_size"] == 0
    assert runner_b.status()["queue_size"] == 1


def test_frame_consumers_join_every_cameras_dispatcher(tmp_path: Path) -> None:
    stub = StubConsumer()
    runtime = _runtime(
        [_cam("a", tmp_path, motion=True), _cam("b", tmp_path, record=False)],
        frame_consumers=(stub,),
    )
    a, b = runtime.cameras

    consumers_a = tuple(a.dispatcher.consumers)
    assert [getattr(c, "name", None) for c in consumers_a] == ["detector_a", "stub"]
    assert consumers_a[1] is stub
    assert tuple(b.dispatcher.consumers) == (stub,)
    # b records nothing and serves no proxy, but the consumer wants its frames.
    assert b.recorder.main is not None and b.recorder.main.frame_tap_enabled is True


def test_a_camera_nothing_consumes_spawns_no_ingestor(tmp_path: Path) -> None:
    runtime = _runtime([_cam("c", tmp_path, record=False, proxy=False)])

    rt = runtime.cameras[0]

    assert not rt.recorder.has_any()
    assert tuple(rt.dispatcher.consumers) == ()


def test_no_detectors_flag_builds_no_runner(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True)], detectors_enabled=False)

    rt = runtime.cameras[0]

    assert runtime.detector_runners == []
    assert tuple(rt.dispatcher.consumers) == ()


# ---------------------------------------------------------------------------
# rebuild_camera_detectors: swaps the runner on that camera's dispatcher only
# ---------------------------------------------------------------------------


def test_rebuild_without_detectors_unwires_the_runner(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True), _cam("b", tmp_path, motion=True)])
    a, b = runtime.cameras
    runner_b = _only_runner(b)
    a.camera.detectors[0].enabled = False

    runtime.rebuild_camera_detectors("a")

    assert tuple(a.dispatcher.consumers) == ()
    assert [r.name for r in runtime.detector_runners] == ["detector_b"]
    assert _only_runner(b) is runner_b


def test_rebuild_swaps_in_a_running_runner_and_keeps_frame_consumers(tmp_path: Path) -> None:
    stub = StubConsumer()
    runtime = _runtime([_cam("a", tmp_path, motion=True)], frame_consumers=(stub,))
    rt = runtime.cameras[0]
    old = tuple(rt.dispatcher.consumers)[0]

    runtime.rebuild_camera_detectors("a")
    try:
        consumers = tuple(rt.dispatcher.consumers)
        new = consumers[0]
        assert isinstance(new, DetectorRunner)
        assert new is not old
        assert consumers[1:] == (stub,)
        assert runtime.detector_runners == [new]
        assert new.queue_maxsize == 8
        assert new.status()["worker_count"] == 1  # set up before it was wired in
    finally:
        for runner in runtime.detector_runners:
            runner.teardown()


def test_rebuild_of_an_unknown_camera_raises(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path)])

    with pytest.raises(ValueError, match="camera 'nope' not found"):
        runtime.rebuild_camera_detectors("nope")


# ---------------------------------------------------------------------------
# Lifecycle requests (RW-0): restart builds a new dispatcher and rewires it
# ---------------------------------------------------------------------------


@pytest.fixture
def no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CameraRecorder, "start", lambda self: None)
    monkeypatch.setattr(CameraRecorder, "stop", lambda self: None)


@pytest.mark.usefixtures("no_processes")
def test_restart_rewires_the_new_cameras_dispatcher(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True)])
    old = _rt(runtime, "a")
    old_runner = _only_runner(old)

    fut = runtime.request_restart_camera("a")
    runtime._drain_requests()
    try:
        assert fut.result(timeout=0) is None
        new = _rt(runtime, "a")
        assert new is not old
        assert new.dispatcher is not old.dispatcher
        assert new.recorder.frame_tap_dispatcher is new.dispatcher
        assert new.recorder.main is not None
        assert new.recorder.main.frame_tap_dispatcher is new.dispatcher
        new_runner = _only_runner(new)
        assert new_runner is not old_runner
        assert new_runner.camera == "a"
        assert runtime.detector_runners == [new_runner]
    finally:
        for runner in runtime.detector_runners:
            runner.teardown()


@pytest.mark.usefixtures("no_processes")
def test_remove_unwires_the_cameras_dispatcher(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path, motion=True), _cam("b", tmp_path, motion=True)])
    b = _rt(runtime, "b")

    fut = runtime.request_remove_camera("b")
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert tuple(b.dispatcher.consumers) == ()
    assert [r.name for r in runtime.detector_runners] == ["detector_a"]
    assert _only_runner(_rt(runtime, "a")).camera == "a"


# ---------------------------------------------------------------------------
# DetectorRunner camera / stream guard
# ---------------------------------------------------------------------------


def test_runner_ignores_frames_for_other_cameras_and_streams() -> None:
    runner = DetectorRunner(worker_count=0, queue_maxsize=8, camera="a", stream="main")

    runner.on_frame("b", "main", _jpeg(), 1.0)
    runner.on_frame("a", "sub", _jpeg(), 1.0)
    assert runner.status()["queue_size"] == 0

    runner.on_frame("a", "main", _jpeg(), 1.0)
    assert runner.status()["queue_size"] == 1


def test_runner_without_camera_or_stream_accepts_every_frame() -> None:
    runner = DetectorRunner(worker_count=0, queue_maxsize=8)

    runner.on_frame("a", "main", _jpeg(), 1.0)
    runner.on_frame("b", "sub", _jpeg(), 1.0)

    assert runner.camera is None and runner.stream is None
    assert runner.status()["queue_size"] == 2


# ---------------------------------------------------------------------------
# serve --frame-consumer
# ---------------------------------------------------------------------------

SERVE_CONFIG = """cameras:
  - name: a
    main_url: rtsp://u:p@h/a
    record:
      enabled: false
    proxy:
      enabled: false
  - name: b
    main_url: rtsp://u:p@h/b
    record:
      enabled: false
    proxy:
      enabled: false
"""


def test_serve_adds_each_frame_consumer_to_every_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(SERVE_CONFIG)
    monkeypatch.chdir(tmp_path)
    started: list[AppRuntime] = []
    monkeypatch.setattr(cli_mod, "_require_binaries", lambda cfg: None)
    monkeypatch.setattr(bootstrap_mod, "bootstrap_database", lambda create_admin: None)
    monkeypatch.setattr(AppRuntime, "start", lambda self: started.append(self))
    monkeypatch.setattr(AppRuntime, "run_forever", lambda self: None)
    monkeypatch.setattr(AppRuntime, "stop_all", lambda self: None)

    result = CliRunner().invoke(
        cli_mod.app,
        ["serve", "-c", str(tmp_path / "config.yaml"), "--no-web", "--frame-consumer", "demo"],
    )

    assert result.exit_code == 0, result.output
    (runtime,) = started
    (consumer,) = runtime.frame_consumers
    assert isinstance(consumer, MotionHeuristicConsumer)
    assert [rt.camera.name for rt in runtime.cameras] == ["a", "b"]
    for rt in runtime.cameras:
        assert tuple(rt.dispatcher.consumers) == (consumer,)
        assert rt.dispatcher.consumers[0] is consumer
        assert rt.recorder.main is not None
        assert rt.recorder.main.frame_tap_enabled is True


def test_motion_demo_consumer_still_works() -> None:
    """Sanity check the FrameConsumer contract is preserved."""
    consumer = MotionHeuristicConsumer(threshold_ratio=0.20)
    dispatcher = FrameTapDispatcher(consumers=[consumer])

    # Should not raise
    dispatcher.dispatch("cam1", "sub", b"x" * 1000, 100.0)
    dispatcher.dispatch("cam1", "sub", b"x" * 1020, 100.1)
    dispatcher.dispatch("cam1", "sub", b"x" * 1400, 100.2)
