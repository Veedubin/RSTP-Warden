"""Per-camera lifecycle API on AppRuntime (RW-0 Task 0.2).

Covers the build/start/stop extraction, the request queue that run_forever drains on the
main thread, and supervisor robustness. Nothing here spawns ffmpeg, MediaMTX or an HTTP
server: process start/stop are recording stubs and run_forever runs on a fake clock.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import rtsp_warden.app as app_mod
from rtsp_warden.app import AppRuntime, CameraExistsError, CameraNotFoundError, CameraRuntime
from rtsp_warden.config import (
    AppConfig,
    CameraConfig,
    DetectorSpec,
    ProxyConfig,
    RecordConfig,
    RuntimeConfig,
)
from rtsp_warden.ffmpeg import ExponentialBackoff
from rtsp_warden.proxy.mjpeg import FrameHub, MjpegProxyServer
from rtsp_warden.proxy.rtsp_mediamtx import MediaMTXProxyServer
from rtsp_warden.recorder import CameraRecorder
from rtsp_warden.retention import RetentionManager
from rtsp_warden.web.services.preview import mjpeg_frames

JPEG = b"\xff\xd8\xff\xd9"


class _Proc:
    """Stand-in for ManagedProcess with only what the supervisor and status table read."""

    def __init__(self, running: bool, tail: list[str] | None = None) -> None:
        self._running = running
        self._tail = tail or []

    def poll(self) -> int | None:
        return None if self._running else 1

    def is_running(self) -> bool:
        return self._running

    def stderr_tail(self) -> list[str]:
        return self._tail


class _Log:
    """Replacement for rtsp_warden.app.log that keeps (level, formatted message) pairs."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str, args: tuple[object, ...]) -> None:
        self.records.append((level, msg % args if args else msg))

    def debug(self, msg: str, *args: object, **_kw: object) -> None:
        self._add("debug", msg, args)

    def info(self, msg: str, *args: object, **_kw: object) -> None:
        self._add("info", msg, args)

    def warning(self, msg: str, *args: object, **_kw: object) -> None:
        self._add("warning", msg, args)

    def error(self, msg: str, *args: object, **_kw: object) -> None:
        self._add("error", msg, args)


def _cam(
    name: str,
    tmp_path: Path,
    *,
    proxy: str | None = None,
    port: int = 9001,
    record: bool = False,
    detectors: list[DetectorSpec] | None = None,
) -> CameraConfig:
    """Camera on rtsp://u:p@h/<name>; proxy=None disables the proxy."""
    return CameraConfig(
        name=name,
        main_url=f"rtsp://u:p@h/{name}",
        record=RecordConfig(enabled=record, output_dir=tmp_path / "rec"),
        proxy=ProxyConfig(enabled=proxy is not None, mode=proxy or "mjpeg", port=port),
        detectors=detectors or [],
    )


def _runtime(cams: list[CameraConfig], *, auto_restart: bool = True) -> AppRuntime:
    cfg = AppConfig(cameras=cams, runtime=RuntimeConfig(auto_restart=auto_restart))
    return AppRuntime(cfg=cfg, console=Console(file=io.StringIO()))


def _start(runtime: AppRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime.start() without replacing the test process's SIGINT/SIGTERM handlers."""
    monkeypatch.setattr(runtime, "_install_signals", lambda: None)
    runtime.start()


def _run_ticks(runtime: AppRuntime, monkeypatch: pytest.MonkeyPatch, ticks: int) -> None:
    """Run runtime.run_forever() for *ticks* supervisor ticks on a fake clock (2 s per tick)."""
    clock = {"now": 1000.0, "ticks": 0}

    def fake_sleep(_seconds: float) -> None:
        clock["ticks"] += 1
        clock["now"] += 2.0
        if clock["ticks"] >= ticks:
            runtime.stop = True

    fake_time = SimpleNamespace(time=lambda: clock["now"], sleep=fake_sleep)
    monkeypatch.setattr(app_mod, "time", fake_time)
    runtime.run_forever()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace process start/stop with stubs that record (action, camera) in call order.

    The recorder stub marks every ingestor as a running process and clears it on stop.
    The MJPEG stub keeps a fake port table, so two cameras on one port clash the way
    ThreadingHTTPServer does (OSError 98) without opening a socket.
    """
    seen: list[tuple[str, str]] = []
    bound: dict[int, str] = {}

    def rec_start(self: CameraRecorder) -> None:
        seen.append(("rec.start", self.camera.name))
        for sp in self.processes():
            sp.proc = _Proc(running=True)  # type: ignore[assignment]

    def rec_stop(self: CameraRecorder) -> None:
        seen.append(("rec.stop", self.camera.name))
        for sp in self.processes():
            sp.proc = None

    def mjpeg_start(self: MjpegProxyServer) -> None:
        seen.append(("mjpeg.start", self.camera.name))
        port = int(self.camera.proxy.port)
        owner = bound.get(port)
        if owner is not None and owner != self.camera.name:
            raise OSError(98, "Address already in use")
        bound[port] = self.camera.name
        self._fake_running = True  # type: ignore[attr-defined]

    def mjpeg_stop(self: MjpegProxyServer) -> None:
        seen.append(("mjpeg.stop", self.camera.name))
        port = int(self.camera.proxy.port)
        if getattr(self, "_fake_running", False) and bound.get(port) == self.camera.name:
            del bound[port]
        self._fake_running = False  # type: ignore[attr-defined]

    def mjpeg_is_running(self: MjpegProxyServer) -> bool:
        return bool(getattr(self, "_fake_running", False))

    def mtx_start(self: MediaMTXProxyServer) -> None:
        seen.append(("mtx.start", self.camera.name))
        self._proc = _Proc(running=True)  # type: ignore[assignment]

    def mtx_stop(self: MediaMTXProxyServer) -> None:
        seen.append(("mtx.stop", self.camera.name))
        self._proc = None

    monkeypatch.setattr(CameraRecorder, "start", rec_start)
    monkeypatch.setattr(CameraRecorder, "stop", rec_stop)
    monkeypatch.setattr(MjpegProxyServer, "start", mjpeg_start)
    monkeypatch.setattr(MjpegProxyServer, "stop", mjpeg_stop)
    monkeypatch.setattr(MjpegProxyServer, "is_running", mjpeg_is_running)
    monkeypatch.setattr(MediaMTXProxyServer, "start", mtx_start)
    monkeypatch.setattr(MediaMTXProxyServer, "stop", mtx_stop)
    return seen


# ---------------------------------------------------------------------------
# Characterization: build/start/stop_all behave exactly as before the refactor
# ---------------------------------------------------------------------------


def test_build_keeps_camera_order_and_field_values(tmp_path: Path) -> None:
    """build() yields the same CameraRuntime objects and order as before (review focus)."""
    motion = [DetectorSpec(type="motion")]
    runtime = _runtime(
        [
            _cam("a", tmp_path, proxy="mjpeg", record=True, detectors=motion),
            _cam("b", tmp_path, proxy="rtsp", port=8554),
            _cam("c", tmp_path),
        ]
    )
    runtime.build()

    assert [rt.camera.name for rt in runtime.cameras] == ["a", "b", "c"]
    for rt, cam in zip(runtime.cameras, runtime.cfg.cameras, strict=True):
        assert rt.camera is cam
        assert rt.next_restart_at == 0.0
        assert rt.last_error == ""
        assert isinstance(rt.rec_backoff, ExponentialBackoff)
        assert isinstance(rt.proxy_backoff, ExponentialBackoff)
        assert rt.rec_backoff is not rt.proxy_backoff
        assert rt.rec_backoff.min_s == 1.0
        assert rt.rec_backoff.max_s == 60.0
        assert rt.rec_backoff.factor == 2.0
        assert rt.recorder.frame_tap_dispatcher is rt.dispatcher

    a, b, c = runtime.cameras
    assert isinstance(a.hub, FrameHub)
    assert isinstance(a.proxy, MjpegProxyServer)
    assert a.proxy.hub is a.hub
    assert a.recorder.proxy_hub is a.hub
    assert isinstance(a.retention, RetentionManager)
    assert a.retention.camera_root == tmp_path / "rec" / "a"
    assert isinstance(b.proxy, MediaMTXProxyServer)
    assert b.hub is None and b.retention is None
    assert c.proxy is None and c.hub is None and c.retention is None
    assert [r.name for r in runtime.detector_runners] == ["detector_a"]
    assert [getattr(x, "name", None) for x in a.dispatcher.consumers] == ["detector_a"]
    assert tuple(b.dispatcher.consumers) == ()
    assert tuple(c.dispatcher.consumers) == ()


def test_start_keeps_process_order(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MJPEG side-server after its ingest; MediaMTX before the ffmpeg that publishes to it."""
    runtime = _runtime(
        [
            _cam("a", tmp_path, proxy="mjpeg"),
            _cam("b", tmp_path, proxy="rtsp", port=8554),
            _cam("c", tmp_path),
        ]
    )
    runtime.build()

    _start(runtime, monkeypatch)

    assert calls == [
        ("rec.start", "a"),
        ("mjpeg.start", "a"),
        ("mtx.start", "b"),
        ("rec.start", "b"),
    ]


def test_stop_all_stops_every_camera_recorder_first(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(
        [
            _cam("a", tmp_path, proxy="mjpeg"),
            _cam("b", tmp_path, proxy="rtsp", port=8554),
            _cam("c", tmp_path),
        ]
    )
    runtime.build()
    _start(runtime, monkeypatch)
    calls.clear()

    runtime.stop_all()

    assert runtime.stop is True
    assert calls == [
        ("rec.stop", "a"),
        ("mjpeg.stop", "a"),
        ("rec.stop", "b"),
        ("mtx.stop", "b"),
        ("rec.stop", "c"),
    ]


# ---------------------------------------------------------------------------
# _build_camera_runtime and find_camera
# ---------------------------------------------------------------------------


def test_build_camera_runtime_builds_without_publishing_or_starting(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    runtime = _runtime([])
    runtime.build()
    cam = _cam("x", tmp_path, proxy="mjpeg")

    rt = runtime._build_camera_runtime(cam)

    assert isinstance(rt, CameraRuntime)
    assert rt.camera is cam
    assert isinstance(rt.hub, FrameHub)
    assert rt.proxy_restart_at == 0.0
    assert rt.proxy_error == ""
    assert runtime.cameras == []
    assert runtime.cfg.cameras == []
    assert calls == []


def test_build_camera_runtime_reuses_the_hub_only_for_mjpeg(tmp_path: Path) -> None:
    runtime = _runtime([])
    hub = FrameHub()

    mjpeg = runtime._build_camera_runtime(_cam("m", tmp_path, proxy="mjpeg"), hub=hub)
    rtsp = runtime._build_camera_runtime(_cam("r", tmp_path, proxy="rtsp"), hub=hub)
    off = runtime._build_camera_runtime(_cam("o", tmp_path), hub=hub)

    assert mjpeg.hub is hub
    assert isinstance(mjpeg.proxy, MjpegProxyServer)
    assert mjpeg.proxy.hub is hub
    assert mjpeg.recorder.proxy_hub is hub
    assert rtsp.hub is None
    assert off.hub is None


def test_find_camera_returns_the_runtime_or_none(tmp_path: Path) -> None:
    runtime = _runtime([_cam("a", tmp_path), _cam("b", tmp_path)])
    runtime.build()

    assert runtime.find_camera("b") is runtime.cameras[1]
    assert runtime.find_camera("nope") is None


# ---------------------------------------------------------------------------
# _start_camera / _stop_camera: one camera's failure does not stop the others
# ---------------------------------------------------------------------------


def test_start_survives_two_cameras_on_one_mjpeg_port(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two cameras left on the default port 9001 must not end serve at boot (review focus)."""
    runtime = _runtime(
        [
            _cam("a", tmp_path, proxy="mjpeg"),
            _cam("b", tmp_path, proxy="mjpeg"),
            _cam("c", tmp_path, proxy="rtsp", port=8554),
        ]
    )
    runtime.build()

    _start(runtime, monkeypatch)  # must not raise

    a, b, _c = runtime.cameras
    assert a.proxy.is_running()
    assert not b.proxy.is_running()
    assert a.last_error == ""
    assert b.last_error == "OSError: [Errno 98] Address already in use"
    assert b.proxy_error == "OSError: [Errno 98] Address already in use"
    assert ("rec.start", "b") in calls  # b still ingests; only its side-server failed
    assert calls[-2:] == [("mtx.start", "c"), ("rec.start", "c")]  # c still starts after b failed

    runtime._supervise_camera(b, 100.0)  # b's ingest runs, its side-server does not
    assert b.last_error == "OSError: [Errno 98] Address already in use"  # not wiped


def test_stop_camera_stops_the_proxy_even_when_the_recorder_stop_fails(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, proxy="mjpeg")])
    runtime.build()
    _start(runtime, monkeypatch)

    def broken_stop(self: CameraRecorder) -> None:
        raise RuntimeError("stop failed")

    monkeypatch.setattr(CameraRecorder, "stop", broken_stop)

    runtime._stop_camera(runtime.cameras[0])  # must not raise

    assert calls[-1] == ("mjpeg.stop", "a")
    assert not runtime.cameras[0].proxy.is_running()


def test_error_text_hides_url_credentials() -> None:
    """last_error reaches the web UI and the logs, so URL credentials are masked."""
    text = app_mod._error_text(OSError("cannot open rtsp://u:p@h/x"))

    assert text == "OSError: cannot open rtsp://***:***@h/x"
    assert "u:p@" not in text
    # A password with an "@" in it is masked whole, not just up to its first "@".
    assert app_mod._error_text(ValueError("rtsp://u:p@ss@h/x")) == "ValueError: rtsp://***:***@h/x"
    assert len(app_mod._error_text(RuntimeError("x" * 1000))) == 300


# ---------------------------------------------------------------------------
# request_* queue: add / remove / restart applied by _drain_requests
# ---------------------------------------------------------------------------


def test_request_add_camera_only_enqueues_and_the_drain_applies_it(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path)])
    runtime.build()
    cam = _cam("x", tmp_path, proxy="mjpeg", port=9002)
    published_at_start: list[bool] = []
    stub_start = CameraRecorder.start

    def rec_start(self: CameraRecorder) -> None:
        published_at_start.append(runtime.find_camera(self.camera.name) is not None)
        stub_start(self)

    monkeypatch.setattr(CameraRecorder, "start", rec_start)

    fut = runtime.request_add_camera(cam)

    assert isinstance(fut, Future)
    assert not fut.done()
    assert runtime.find_camera("x") is None

    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert [c.name for c in runtime.cfg.cameras] == ["a", "x"]
    assert [r.camera.name for r in runtime.cameras] == ["a", "x"]
    added = runtime.find_camera("x")
    assert added is not None and added.camera is cam
    assert calls == [("rec.start", "x"), ("mjpeg.start", "x")]
    # Published before its processes start, so stop_all() from a signal handler sees it.
    assert published_at_start == [True]


def test_add_keeps_the_cfg_entry_the_route_already_appended(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    runtime = _runtime([])
    runtime.build()
    cam = _cam("x", tmp_path)
    runtime.cfg.cameras.append(cam)  # the add-camera route appends before it asks

    fut = runtime.request_add_camera(cam)
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert len(runtime.cfg.cameras) == 1
    assert runtime.cfg.cameras[0] is cam
    added = runtime.find_camera("x")
    assert added is not None and added.camera is cam


def test_bad_requests_fail_their_future_and_the_next_request_still_runs(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    """Unknown or duplicate names fail their future; later requests still run (review focus)."""
    runtime = _runtime([_cam("a", tmp_path)])
    runtime.build()

    duplicate = runtime.request_add_camera(_cam("a", tmp_path))
    ghost_restart = runtime.request_restart_camera("ghost")
    ghost_remove = runtime.request_remove_camera("ghost")
    ok = runtime.request_add_camera(_cam("x", tmp_path))

    runtime._drain_requests()  # must not raise

    assert isinstance(duplicate.exception(timeout=0), CameraExistsError)
    assert isinstance(ghost_restart.exception(timeout=0), CameraNotFoundError)
    assert isinstance(ghost_remove.exception(timeout=0), CameraNotFoundError)
    assert str(duplicate.exception(timeout=0)) == "camera 'a' already exists"
    assert str(ghost_restart.exception(timeout=0)) == "camera 'ghost' not found"
    assert str(ghost_remove.exception(timeout=0)) == "camera 'ghost' not found"
    assert ok.result(timeout=0) is None
    assert [r.camera.name for r in runtime.cameras] == ["a", "x"]
    assert [c.name for c in runtime.cfg.cameras] == ["a", "x"]


def test_a_failed_request_logs_and_shows_no_credentials(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain's warning and last_error carry the start error with the URL masked."""
    fake_log = _Log()
    monkeypatch.setattr(app_mod, "log", fake_log)
    runtime = _runtime([])
    runtime.build()

    def unreachable(self: CameraRecorder) -> None:
        raise OSError(f"cannot open {self.camera.main_url}")

    monkeypatch.setattr(CameraRecorder, "start", unreachable)

    fut = runtime.request_add_camera(_cam("x", tmp_path, record=True))
    runtime._drain_requests()

    assert isinstance(fut.exception(timeout=0), OSError)
    x = runtime.find_camera("x")
    assert x is not None
    assert x.last_error == "OSError: cannot open rtsp://***:***@h/x"
    warning = "[supervisor] add x failed: OSError: cannot open rtsp://***:***@h/x"
    assert ("warning", warning) in fake_log.records
    assert not any("u:p@" in msg for _level, msg in fake_log.records)


def test_add_rejects_a_path_like_name(tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    """A hot-added name becomes a directory under the recordings root and the workspace."""
    runtime = _runtime([])
    runtime.build()
    names = ("..", "a/b", "new")

    futs = [runtime.request_add_camera(_cam(name, tmp_path)) for name in names]
    runtime._drain_requests()

    for fut, name in zip(futs, names, strict=True):
        exc = fut.exception(timeout=0)
        assert isinstance(exc, ValueError)
        assert str(exc) == f"invalid camera name {name!r}"
    assert runtime.cameras == []
    assert runtime.cfg.cameras == []
    assert calls == []


def test_add_with_an_unknown_proxy_mode_fails_its_future_not_serve(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    runtime = _runtime([])
    runtime.build()
    cam = _cam("x", tmp_path, proxy="mjpeg")
    cam.proxy.mode = "bogus"  # type: ignore[assignment]  # no validate_assignment

    fut = runtime.request_add_camera(cam)
    runtime._drain_requests()  # must not raise SystemExit

    assert isinstance(fut.exception(timeout=0), SystemExit)
    assert runtime.find_camera("x") is None and runtime.cfg.cameras == []


def test_a_web_thread_can_post_and_wait_while_the_main_thread_drains(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    runtime = _runtime([])
    runtime.build()
    results: list[object] = []

    def web_route() -> None:
        results.append(runtime.request_add_camera(_cam("x", tmp_path)).result(timeout=5))

    t = threading.Thread(target=web_route)
    t.start()
    for _ in range(500):  # the supervisor's ticks, bounded at about 5 s
        runtime._drain_requests()
        t.join(timeout=0.01)
        if not t.is_alive():
            break

    assert not t.is_alive()
    assert results == [None]
    assert runtime.find_camera("x") is not None


def test_an_async_caller_can_await_the_future(tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    """An async route awaits the future through wrap_future instead of blocking its loop."""
    runtime = _runtime([])
    runtime.build()
    results: list[object] = []

    async def async_route() -> None:
        fut = runtime.request_add_camera(_cam("x", tmp_path))
        results.append(await asyncio.wait_for(asyncio.wrap_future(fut), timeout=5))

    t = threading.Thread(target=lambda: asyncio.run(async_route()))
    t.start()
    for _ in range(500):  # the supervisor's ticks, bounded at about 5 s
        runtime._drain_requests()
        t.join(timeout=0.01)
        if not t.is_alive():
            break

    assert not t.is_alive()
    assert results == [None]
    assert runtime.find_camera("x") is not None


def test_a_cancelled_request_is_skipped(tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    runtime = _runtime([])
    runtime.build()

    fut = runtime.request_add_camera(_cam("x", tmp_path))
    assert fut.cancel()
    runtime._drain_requests()

    assert runtime.find_camera("x") is None
    assert runtime.cfg.cameras == []


def test_a_request_queued_during_a_drain_waits_for_the_next_tick(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that re-enqueues itself cannot keep the supervisor in one drain."""
    runtime = _runtime([_cam("a", tmp_path, record=True)])
    runtime.build()
    follow_ups: list[Future[None]] = []

    def rebuild_and_ask_again(name: str) -> None:
        # RW-3's rebuild may ask for a restart when the tap settings change (R18).
        if not follow_ups:
            follow_ups.append(runtime.request_restart_camera(name))

    monkeypatch.setattr(runtime, "rebuild_camera_detectors", rebuild_and_ask_again)

    first = runtime.request_restart_camera("a")
    runtime._drain_requests()

    assert first.result(timeout=0) is None
    assert len(follow_ups) == 1
    assert not follow_ups[0].done()
    assert runtime._requests.qsize() == 1
    assert calls == [("rec.stop", "a"), ("rec.start", "a")]

    runtime._drain_requests()  # the next tick applies it

    assert follow_ups[0].result(timeout=0) is None
    assert calls[2:] == [("rec.stop", "a"), ("rec.start", "a")]


def test_request_remove_camera_stops_it_and_drops_its_detector_runner(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    motion = [DetectorSpec(type="motion")]
    runtime = _runtime(
        [
            _cam("a", tmp_path, detectors=motion),
            _cam("b", tmp_path, proxy="mjpeg", detectors=motion),
        ]
    )
    runtime.build()
    removed = runtime.find_camera("b")
    assert removed is not None
    published_at_stop: list[bool] = []
    stub_stop = CameraRecorder.stop

    def rec_stop(self: CameraRecorder) -> None:
        published_at_stop.append(runtime.find_camera(self.camera.name) is not None)
        stub_stop(self)

    monkeypatch.setattr(CameraRecorder, "stop", rec_stop)

    fut = runtime.request_remove_camera("b")
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert [r.camera.name for r in runtime.cameras] == ["a"]
    assert [c.name for c in runtime.cfg.cameras] == ["a"]
    assert calls == [("rec.stop", "b"), ("mjpeg.stop", "b")]
    assert published_at_stop == [False]  # unpublished before it is stopped
    assert [r.name for r in runtime.detector_runners] == ["detector_a"]
    remaining = runtime.find_camera("a")
    assert remaining is not None
    assert [getattr(c, "name", None) for c in remaining.dispatcher.consumers] == ["detector_a"]
    assert tuple(removed.dispatcher.consumers) == ()


def test_request_restart_rebuilds_the_camera_and_keeps_its_hub(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An open /live.mjpeg viewer keeps receiving frames after a restart (review focus)."""
    runtime = _runtime([_cam("a", tmp_path, proxy="mjpeg")])
    runtime.build()
    _start(runtime, monkeypatch)
    old = runtime.find_camera("a")
    assert old is not None
    hub = old.hub
    assert hub is not None
    viewer = mjpeg_frames(hub, stop_after=2)  # what GET /cameras/a/live.mjpeg holds on to
    hub.update(b"\xff\xd8old\xff\xd9")
    assert b"old" in next(viewer)  # the viewer is open, mid-stream on frame 1
    old.next_restart_at = 999.0
    old.proxy_restart_at = 999.0
    old.last_error = "boom"
    old.proxy_error = "boom"
    runtime.cfg.cameras[0].main_url = "rtsp://u:p@h/edited"  # the edit route, in place
    calls.clear()

    fut = runtime.request_restart_camera("a")
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    new = runtime.find_camera("a")
    assert new is not None and new is not old
    assert len(runtime.cameras) == 1
    assert new.hub is hub
    assert isinstance(new.proxy, MjpegProxyServer) and new.proxy.hub is hub
    assert new.proxy is not old.proxy
    assert new.recorder is not old.recorder
    assert new.recorder.main is not None
    assert new.recorder.main.mjpeg_hub is hub
    assert new.recorder.main.upstream_url == "rtsp://u:p@h/edited"
    assert new.next_restart_at == 0.0
    assert new.proxy_restart_at == 0.0
    assert new.last_error == ""
    assert new.proxy_error == ""
    assert new.rec_backoff is not old.rec_backoff
    assert calls == [
        ("rec.stop", "a"),
        ("mjpeg.stop", "a"),
        ("rec.start", "a"),
        ("mjpeg.start", "a"),
    ]

    new.recorder.main.mjpeg_hub.update(JPEG)  # first frame from the restarted ffmpeg
    jpeg, frame_id, _ts = hub.wait_for_new(1, timeout=0)  # bounded: cannot hang
    assert (jpeg, frame_id) == (JPEG, 2)
    assert JPEG in next(viewer)  # the same open viewer gets it


def test_restart_uses_the_current_cfg_entry_and_starts_mediamtx_first(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, proxy="mjpeg")])
    runtime.build()
    _start(runtime, monkeypatch)
    replacement = _cam("a", tmp_path, proxy="rtsp", port=8554)
    runtime.cfg.cameras[0] = replacement  # a route that swapped the whole object
    calls.clear()

    fut = runtime.request_restart_camera("a")
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    new = runtime.find_camera("a")
    assert new is not None
    assert new.camera is replacement
    assert isinstance(new.proxy, MediaMTXProxyServer)
    assert new.hub is None
    assert calls == [
        ("rec.stop", "a"),
        ("mjpeg.stop", "a"),
        ("mtx.start", "a"),
        ("rec.start", "a"),
    ]


def test_restart_keeps_the_boot_tap_setting(tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    """A rebuilt camera gets the same frame tap as at boot, on its own new dispatcher."""
    motion = [DetectorSpec(type="motion")]
    runtime = _runtime([_cam("a", tmp_path, record=True, detectors=motion), _cam("b", tmp_path)])
    runtime.build()
    boot_a, boot_b = runtime.cameras
    assert boot_a.recorder.main is not None and boot_a.recorder.main.frame_tap_enabled
    assert boot_a.recorder.frame_tap_dispatcher is boot_a.dispatcher
    assert boot_b.recorder.main is None  # nothing reads b's stream at boot
    try:
        for name in ("a", "b"):
            fut = runtime.request_restart_camera(name)
            runtime._drain_requests()
            assert fut.result(timeout=0) is None

        a, b = runtime.find_camera("a"), runtime.find_camera("b")
        assert a is not None and a is not boot_a
        assert a.dispatcher is not boot_a.dispatcher
        assert a.recorder.frame_tap_dispatcher is a.dispatcher
        assert a.recorder.main is not None and a.recorder.main.frame_tap_enabled
        assert [getattr(c, "name", None) for c in a.dispatcher.consumers] == ["detector_a"]
        assert b is not None and b is not boot_b
        assert b.recorder.main is None
    finally:
        runtime.stop_all()  # joins the rebuilt detector runner's worker threads


def test_hot_add_wires_a_detector_runner_only_when_detectors_are_enabled(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    for enabled, expected in ((True, ["detector_x"]), (False, [])):
        runtime = _runtime([])
        runtime.detectors_enabled = enabled
        runtime.build()
        cam = _cam("x", tmp_path, detectors=[DetectorSpec(type="motion")])
        fut = runtime.request_add_camera(cam)
        try:
            runtime._drain_requests()
            assert fut.result(timeout=0) is None
            assert [r.name for r in runtime.detector_runners] == expected
            added = runtime.find_camera("x")
            assert added is not None
            assert [getattr(c, "name", None) for c in added.dispatcher.consumers] == expected
        finally:
            runtime.stop_all()  # joins the runner's worker threads


def test_a_detector_failure_does_not_stop_a_hot_add(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """As at boot, a detector that cannot be built never keeps the camera from recording."""
    runtime = _runtime([])
    runtime.build()

    def broken_rebuild(name: str) -> None:
        raise RuntimeError("model file missing")

    monkeypatch.setattr(runtime, "rebuild_camera_detectors", broken_rebuild)

    fut = runtime.request_add_camera(_cam("x", tmp_path, record=True))
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert ("rec.start", "x") in calls
    assert runtime.find_camera("x") is not None


def test_a_detector_failure_does_not_stop_a_restart(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, record=True)])
    runtime.build()
    _start(runtime, monkeypatch)
    calls.clear()

    def broken_rebuild(name: str) -> None:
        raise RuntimeError("model file missing")

    monkeypatch.setattr(runtime, "rebuild_camera_detectors", broken_rebuild)

    fut = runtime.request_restart_camera("a")
    runtime._drain_requests()

    assert fut.result(timeout=0) is None
    assert calls == [("rec.stop", "a"), ("rec.start", "a")]


def test_an_add_that_cannot_start_fails_its_future_and_stays_configured(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, proxy="mjpeg")])
    runtime.build()
    _start(runtime, monkeypatch)

    fut = runtime.request_add_camera(_cam("b", tmp_path, proxy="mjpeg"))  # port 9001 again
    runtime._drain_requests()  # must not raise

    assert isinstance(fut.exception(timeout=0), OSError)
    b = runtime.find_camera("b")
    assert b is not None
    assert b.last_error == "OSError: [Errno 98] Address already in use"
    assert b.proxy_error == "OSError: [Errno 98] Address already in use"
    assert [c.name for c in runtime.cfg.cameras] == ["a", "b"]  # the supervisor retries it


# ---------------------------------------------------------------------------
# Shutdown: no request is left waiting once stop_all() has run
# ---------------------------------------------------------------------------


def test_stop_all_fails_a_pending_request_and_unblocks_its_waiter(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    """stop_all() during a pending request: no deadlock, RuntimeError (review focus)."""
    runtime = _runtime([_cam("a", tmp_path)])
    runtime.build()
    errors: list[BaseException] = []
    posted = threading.Event()

    def web_route() -> None:
        fut = runtime.request_restart_camera("a")
        posted.set()
        try:
            fut.result(timeout=5)
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=web_route)
    t.start()
    assert posted.wait(timeout=5)

    runtime.stop_all()
    t.join(timeout=5)

    assert not t.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "runtime stopping"
    assert calls == [("rec.stop", "a")]  # stop_all ran; the restart itself never did


def test_a_request_after_stop_all_fails_at_once(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    runtime = _runtime([])
    runtime.build()
    runtime.stop_all()

    fut = runtime.request_add_camera(_cam("x", tmp_path))

    assert fut.done()
    assert isinstance(fut.exception(timeout=0), RuntimeError)
    runtime._drain_requests()
    assert runtime.find_camera("x") is None


def test_sigterm_inside_a_drained_request_fails_the_rest_without_hanging(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([])
    runtime.build()
    stub_start = CameraRecorder.start

    def start_then_sigterm(self: CameraRecorder) -> None:
        stub_start(self)
        runtime.stop_all()  # what the SIGTERM handler runs on the main thread

    monkeypatch.setattr(CameraRecorder, "start", start_then_sigterm)
    first = runtime.request_add_camera(_cam("x", tmp_path, proxy="mjpeg"))
    second = runtime.request_add_camera(_cam("y", tmp_path, proxy="mjpeg", port=9002))

    runtime._drain_requests()  # returns; does not run "y" after the stop

    assert first.result(timeout=0) is None
    assert isinstance(second.exception(timeout=0), RuntimeError)
    assert runtime.find_camera("y") is None
    # C-implemented queue: a signal handler cannot interrupt get()/put() halfway, so
    # stop_all() from the SIGTERM handler cannot deadlock on the queue's lock.
    assert isinstance(runtime._requests, queue.SimpleQueue)


# ---------------------------------------------------------------------------
# run_forever: drain first, per-camera schedules, one camera cannot end serve
# ---------------------------------------------------------------------------


def test_supervisor_restarts_a_dead_ingest_from_the_camera_schedule(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, record=True)])
    runtime.build()
    _start(runtime, monkeypatch)
    rt = runtime.cameras[0]
    assert rt.recorder.main is not None
    rt.recorder.main.proc = _Proc(running=False, tail=["boom"])  # type: ignore[assignment]
    calls.clear()

    runtime._supervise_camera(rt, 100.0)  # death noticed: schedule only
    assert rt.next_restart_at == 101.0
    assert rt.last_error == "boom"
    assert calls == []

    runtime._supervise_camera(rt, 100.5)  # not due yet
    assert calls == []

    runtime._supervise_camera(rt, 101.0)  # due: restart
    assert calls == [("rec.stop", "a"), ("rec.start", "a")]
    assert rt.next_restart_at == 0.0

    runtime._supervise_camera(rt, 101.5)  # running again
    assert rt.last_error == ""
    assert rt.next_restart_at == 0.0


def test_supervisor_restarts_a_dead_mjpeg_proxy_from_proxy_restart_at(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, proxy="mjpeg")])
    runtime.build()
    _start(runtime, monkeypatch)
    rt = runtime.cameras[0]
    assert rt.proxy is not None
    rt.proxy.stop()  # type: ignore[attr-defined]
    rt.proxy_error = "OSError: an earlier failure"
    calls.clear()

    runtime._supervise_camera(rt, 100.0)
    assert rt.proxy_restart_at == 101.0
    assert calls == []

    runtime._supervise_camera(rt, 101.0)
    assert calls == [("mjpeg.stop", "a"), ("mjpeg.start", "a")]
    assert rt.proxy_restart_at == 0.0
    assert rt.proxy_error == ""  # cleared once the proxy runs again

    rt.proxy_error = "OSError: an earlier failure"  # however it got there
    runtime._supervise_camera(rt, 101.5)  # the proxy runs
    assert rt.proxy_error == ""


def test_supervisor_restarts_dead_mediamtx_then_ffmpeg_in_order(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([_cam("a", tmp_path, proxy="rtsp", port=8554)])
    runtime.build()
    _start(runtime, monkeypatch)
    rt = runtime.cameras[0]
    rt.proxy._proc = _Proc(running=False)  # type: ignore[union-attr]
    rt.recorder.main.proc = _Proc(running=False)  # type: ignore[union-attr,assignment]
    calls.clear()

    runtime._supervise_camera(rt, 100.0)
    assert (rt.next_restart_at, rt.proxy_restart_at, calls) == (101.0, 101.0, [])

    runtime._supervise_camera(rt, 101.0)
    assert calls == [("rec.stop", "a"), ("mtx.start", "a"), ("rec.start", "a")]
    assert rt.proxy_restart_at == 0.0

    rt.proxy_error = "PermissionError: an earlier failure"  # however it got there
    runtime._supervise_camera(rt, 101.5)  # MediaMTX runs
    assert rt.proxy_error == ""


def test_run_forever_drains_requests_even_with_auto_restart_off(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime([], auto_restart=False)
    runtime.build()
    fut = runtime.request_add_camera(_cam("x", tmp_path, proxy="mjpeg"))

    _run_ticks(runtime, monkeypatch, ticks=1)

    assert fut.result(timeout=0) is None
    assert runtime.find_camera("x") is not None


def test_run_forever_survives_a_taken_mjpeg_port(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supervisor's proxy retry hitting OSError 98 logs and keeps running (review focus)."""
    fake_log = _Log()
    monkeypatch.setattr(app_mod, "log", fake_log)
    runtime = _runtime(
        [
            _cam("a", tmp_path, proxy="mjpeg"),
            _cam("b", tmp_path, proxy="mjpeg"),
            _cam("c", tmp_path, record=True),
        ]
    )
    runtime.build()
    _start(runtime, monkeypatch)  # b's side-server fails at boot: a holds port 9001
    a, b, c = runtime.cameras
    assert c.recorder.main is not None
    c.recorder.main.proc = _Proc(running=False)  # type: ignore[assignment]  # c's ffmpeg died
    c_started_at: list[float] = []
    stub_start = CameraRecorder.start

    def rec_start(self: CameraRecorder) -> None:
        if self.camera.name == "c":
            c_started_at.append(app_mod.time.time())  # the fake clock inside _run_ticks
        stub_start(self)

    monkeypatch.setattr(CameraRecorder, "start", rec_start)

    _run_ticks(runtime, monkeypatch, ticks=6)  # returns normally once the fake sleep stops it

    b_starts = [x for x in calls if x == ("mjpeg.start", "b")]
    assert len(b_starts) == 3  # boot, then retries at t=1002 and t=1006 (backoff 1 s, 2 s)
    assert a.proxy.is_running()
    # c is restarted on the tick where b's retry raised (t=1002), not one tick later.
    assert c_started_at == [1002.0]
    errors = [msg for level, msg in fake_log.records if level == "error"]
    assert "[supervisor] camera b: OSError: [Errno 98] Address already in use" in errors
    # Not wiped by the healthy ingest ticks while b's side-server stays down.
    assert b.last_error == "OSError: [Errno 98] Address already in use"
    assert b.proxy_error == "OSError: [Errno 98] Address already in use"


def test_an_ingest_start_error_outlives_the_next_tick(
    tmp_path: Path, calls: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ffmpeg that never started left no stderr: keep the start error, do not blank it."""

    def missing_ffmpeg(self: CameraRecorder) -> None:
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(CameraRecorder, "start", missing_ffmpeg)
    runtime = _runtime([_cam("a", tmp_path, record=True)])
    runtime.build()
    _start(runtime, monkeypatch)  # logged, not raised
    rt = runtime.cameras[0]
    error = "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"
    assert rt.last_error == error

    _run_ticks(runtime, monkeypatch, ticks=1)

    assert rt.next_restart_at == 1001.0  # the supervisor scheduled a retry
    assert rt.last_error == error


# ---------------------------------------------------------------------------
# Detector wiring: every rebind of the dispatcher's consumers holds _detector_lock
# ---------------------------------------------------------------------------


class _LockCheckingDispatcher:
    """Dispatcher stand-in that records whether _detector_lock was held at each rebind."""

    def __init__(self) -> None:
        self.lock: threading.Lock | None = None
        self.held: list[bool] = []
        self._consumers: tuple[object, ...] = ()

    @property
    def consumers(self) -> tuple[object, ...]:
        return self._consumers

    @consumers.setter
    def consumers(self, value: tuple[object, ...]) -> None:
        self.held.append(self.lock is not None and self.lock.locked())
        self._consumers = tuple(value)


def test_consumer_rebinds_hold_the_detector_lock(
    tmp_path: Path, calls: list[tuple[str, str]]
) -> None:
    """A web-thread rebuild and a main-thread remove cannot lose a consumer (gap-2 G-5)."""
    runtime = _runtime([_cam("a", tmp_path, detectors=[DetectorSpec(type="motion")])])
    runtime.build()  # wires the boot runners before any thread runs
    rt = runtime.find_camera("a")
    assert rt is not None
    dispatcher = _LockCheckingDispatcher()
    dispatcher.lock = runtime._detector_lock
    rt.dispatcher = dispatcher  # type: ignore[assignment]
    try:
        runtime.rebuild_camera_detectors("a")  # what the detector routes call
        fut = runtime.request_remove_camera("a")
        runtime._drain_requests()

        assert fut.result(timeout=0) is None
        assert dispatcher.held == [True, True]
        assert dispatcher.consumers == ()
    finally:
        runtime.stop_all()
