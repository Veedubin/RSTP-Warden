from __future__ import annotations

import logging
import queue
import re
import signal
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console
from rich.table import Table

from .config import AppConfig, CameraConfig
from .detectors.registry import build_detectors_for_camera
from .detectors.runner import DetectorRunner
from .detectors.sinks import EventSink
from .ffmpeg import ExponentialBackoff, redact_text
from .frame_tap import FrameTapDispatcher
from .proxy.mjpeg import FrameHub, MjpegProxyServer
from .proxy.rtsp_mediamtx import MediaMTXProxyServer
from .recorder import CameraRecorder
from .retention import RetentionManager
from .retention_resolver import resolve_retention

log = logging.getLogger(__name__)


@dataclass
class CameraRuntime:
    camera: CameraConfig
    recorder: CameraRecorder
    proxy: object | None
    hub: FrameHub | None
    retention: RetentionManager | None

    # backoff for recorder restart and proxy restart
    rec_backoff: ExponentialBackoff
    proxy_backoff: ExponentialBackoff

    # Set by the supervisor so the web UI can show a restart countdown and the last error.
    next_restart_at: float = 0.0
    last_error: str = ""
    # When the supervisor may next restart the proxy (0.0 = nothing scheduled).
    proxy_restart_at: float = 0.0
    # Why the proxy last failed to start ('' once it runs), e.g. a taken MJPEG port.
    proxy_error: str = ""

    def mark_healthy(self) -> None:
        """Clear restart state once every ingest process is running again."""
        self.next_restart_at = 0.0
        self.last_error = ""


def _last_stderr_line(procs: list) -> str:
    """Return the last non-empty stderr line across ingest processes, or ''.

    Credentials are masked before the 300-character cut, so a cut cannot leave part
    of a password behind.
    """
    for sp in procs:
        proc = getattr(sp, "proc", None)
        if proc is None:
            continue
        try:
            tail = [ln for ln in proc.stderr_tail() if ln.strip()]
        except Exception:
            continue
        if tail:
            return redact_text(tail[-1])[:300]
    return ""


def _error_text(exc: BaseException) -> str:
    """One-line description of *exc* for CameraRuntime.last_error (at most 300 chars).

    The text reaches the web UI and the logs, so credentials are masked with
    ``ffmpeg.redact_text`` (URL userinfo, even with a raw "@" or "/" in the password, and
    ``pwd=`` style query values). The message is capped at 1000 characters first.
    """
    text = f"{type(exc).__name__}: {exc}"[:1000]
    return redact_text(text)[:300]


# A hot-added camera's name becomes a directory (recordings, MediaMTX config), so it must
# never be a path; "new" is the add-camera page.
_CAMERA_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


class CameraNotFoundError(LookupError):
    """No camera of that name is in the runtime."""


class CameraExistsError(LookupError):
    """A camera of that name is already in the runtime."""


@dataclass(slots=True)
class _RuntimeRequest:
    """A queued per-camera lifecycle request; run_forever applies it on the main thread."""

    kind: Literal["add", "remove", "restart"]
    name: str
    camera: CameraConfig | None  # set for "add" only
    future: Future[None]


@dataclass
class AppRuntime:
    cfg: AppConfig
    console: Console = field(default_factory=Console)
    stop: bool = False
    cameras: list[CameraRuntime] = field(default_factory=list)
    frame_tap_dispatcher: FrameTapDispatcher | None = None
    detectors_enabled: bool = True
    detector_runners: list[DetectorRunner] = field(default_factory=list)
    _event_sink: EventSink | None = field(default=None, init=False, repr=False)
    _detector_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Lifecycle requests posted from any thread and applied by run_forever on the main thread.
    # SimpleQueue's get/put are C calls a signal handler cannot interrupt halfway, so
    # stop_all() run from the SIGTERM handler can drain it; queue.Queue's lock could deadlock.
    _requests: queue.SimpleQueue[_RuntimeRequest] = field(
        default_factory=queue.SimpleQueue, init=False, repr=False
    )
    # The dispatcher recorders are built with: the one passed in, never the one
    # _build_detectors() creates later, so a rebuilt camera keeps its boot argv.
    _ingest_dispatcher: FrameTapDispatcher | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ingest_dispatcher = self.frame_tap_dispatcher

    def build(self) -> None:
        self.cameras = [self._build_camera_runtime(cam) for cam in self.cfg.cameras]

        # Wire detector framework
        self._build_detectors()

    def _build_camera_runtime(
        self, cam: CameraConfig, *, hub: FrameHub | None = None
    ) -> CameraRuntime:
        """Build one camera's recorder, proxy, retention and backoffs. Starts nothing.

        ``hub`` is reused when the camera still serves MJPEG, so a restart keeps the
        FrameHub that open ``/live.mjpeg`` viewers hold. It is ignored otherwise.
        """
        proxy: object | None = None

        # In unified ingest mode, the hub (if any) is shared between ingest and proxy.
        if cam.proxy.enabled and cam.proxy.mode == "mjpeg":
            if hub is None:
                hub = FrameHub()
        else:
            hub = None

        recorder = CameraRecorder(
            camera=cam,
            runtime=self.cfg.runtime,
            proxy_hub=hub,
            frame_tap_dispatcher=self._ingest_dispatcher,
        )

        if cam.proxy.enabled:
            if cam.proxy.mode == "mjpeg":
                assert hub is not None
                proxy = MjpegProxyServer(camera=cam, runtime=self.cfg.runtime, hub=hub)
            elif cam.proxy.mode == "rtsp":
                proxy = MediaMTXProxyServer(camera=cam, runtime=self.cfg.runtime)
            else:
                raise SystemExit(f"Unsupported proxy mode: {cam.proxy.mode}")

        retention = None
        if cam.record.enabled:
            effective_retention = resolve_retention(cam, self.cfg.retention)
            retention = RetentionManager(
                camera_name=cam.name,
                camera_root=cam.record.output_dir / cam.name,
                cfg=effective_retention,
            )

        rec_backoff = ExponentialBackoff(
            min_s=self.cfg.runtime.restart_backoff_min_s,
            max_s=self.cfg.runtime.restart_backoff_max_s,
            factor=self.cfg.runtime.restart_backoff_factor,
        )
        proxy_backoff = ExponentialBackoff(
            min_s=self.cfg.runtime.restart_backoff_min_s,
            max_s=self.cfg.runtime.restart_backoff_max_s,
            factor=self.cfg.runtime.restart_backoff_factor,
        )

        return CameraRuntime(
            camera=cam,
            recorder=recorder,
            proxy=proxy,
            hub=hub,
            retention=retention,
            rec_backoff=rec_backoff,
            proxy_backoff=proxy_backoff,
        )

    def find_camera(self, name: str) -> CameraRuntime | None:
        """Return the CameraRuntime named *name*, or None. Safe to call from any thread."""
        for rt in self.cameras:
            if rt.camera.name == name:
                return rt
        return None

    def start(self) -> None:
        self._install_signals()

        # Start detector runners
        for runner in self.detector_runners:
            try:
                runner.setup()
            except Exception:
                log.warning("detector runner setup failed", exc_info=True)

        for rt in self.cameras:
            try:
                self._start_camera(rt)
            except Exception:
                # One camera that cannot start (a taken MJPEG port, say) must not stop the
                # others or end serve; the supervisor retries it with backoff.
                log.error("[supervisor] could not start %s: %s", rt.camera.name, rt.last_error)

        self._status_table()

    def _start_camera(self, rt: CameraRuntime) -> None:
        """Start one camera's processes in start()'s order; on failure set last_error, re-raise."""
        try:
            # Keep MJPEG health endpoint pointed at the current ingest process.
            self._sync_mjpeg_ingest_proc(rt)
            # Ordering matters for RTSP proxy: MediaMTX must be listening before FFmpeg publishes.
            if rt.proxy is not None and isinstance(rt.proxy, MediaMTXProxyServer):
                self._start_proxy(rt)

            if rt.recorder.has_any():
                rt.recorder.start()

            if rt.proxy is not None and isinstance(rt.proxy, MjpegProxyServer):
                # MJPEG proxy is serve-only; it consumes frames from the ingest-owned hub.
                self._sync_mjpeg_ingest_proc(rt)
                self._start_proxy(rt)
        except Exception as exc:
            rt.last_error = _error_text(exc)
            raise

    def _start_proxy(self, rt: CameraRuntime) -> None:
        """Start the camera's proxy; a failure is kept in rt.proxy_error and re-raised."""
        if not isinstance(rt.proxy, MjpegProxyServer | MediaMTXProxyServer):
            return
        try:
            rt.proxy.start()
        except Exception as exc:
            rt.proxy_error = _error_text(exc)
            raise
        rt.proxy_error = ""

    def _stop_camera(self, rt: CameraRuntime) -> None:
        """Stop one camera's ingest, then its proxy. Never raises."""
        try:
            rt.recorder.stop()
        except Exception:
            log.warning("[supervisor] stopping ingest for %s failed", rt.camera.name, exc_info=True)
        if rt.proxy is not None:
            try:
                rt.proxy.stop()  # type: ignore[attr-defined]
            except Exception:
                log.warning(
                    "[supervisor] stopping proxy for %s failed", rt.camera.name, exc_info=True
                )

    def _sync_mjpeg_ingest_proc(self, rt: CameraRuntime) -> None:
        """Keep MJPEG /healthz wired to the current ingest process."""
        if rt.proxy is None or not isinstance(rt.proxy, MjpegProxyServer):
            return
        stream = rt.camera.proxy.stream
        ing = rt.recorder.main if stream == "main" else rt.recorder.sub
        rt.proxy.ingest_proc = ing.proc if ing else None

    # ------------------------------------------------------------------
    # Per-camera lifecycle: request_* (any thread) -> _drain_requests (main thread)
    #
    # request_* only enqueue: they never block or raise, from any thread (the event loop
    # included) and even before start(). A future resolves once the next supervisor tick
    # has applied its request; stopping one ingest alone can take ~10 s. An async route
    # awaits asyncio.wrap_future(fut) under asyncio.wait_for, a threadpool route calls
    # fut.result(timeout=...), and a timeout does not cancel a request that is running.
    # Never wait on a future from the main thread (run_forever, _apply_*, start(), the
    # signal handler): only the main thread drains the queue. rebuild_camera_detectors,
    # called by _apply_add/_apply_restart right after the camera was rebuilt from its
    # current config, must not request another restart. Show a failure with
    # _error_text(fut.exception()) or the camera's last_error, never str(exc).
    # ------------------------------------------------------------------

    def request_add_camera(self, cam: CameraConfig) -> Future[None]:
        """Queue a hot-add of *cam*; returns at once, the future resolves on the next tick.

        The future fails with CameraExistsError when a camera of that name is already
        running, and with ValueError when the name is not a valid camera name. A camera
        that is added but cannot start (a taken port, say) stays configured, the
        supervisor keeps retrying it, and its future carries the start error.
        """
        return self._submit("add", cam.name, cam)

    def request_remove_camera(self, name: str) -> Future[None]:
        """Queue stopping camera *name* and dropping it from the runtime and cfg.cameras.

        Its recordings stay on disk. The future fails with CameraNotFoundError when no
        camera of that name is running.
        """
        return self._submit("remove", name, None)

    def request_restart_camera(self, name: str) -> Future[None]:
        """Queue a stop, rebuild from the current CameraConfig, and start of camera *name*.

        The rebuilt camera is a new CameraRuntime: look it up again with find_camera. The
        future fails with CameraNotFoundError when no camera of that name is running.
        """
        return self._submit("restart", name, None)

    def _submit(
        self, kind: Literal["add", "remove", "restart"], name: str, cam: CameraConfig | None
    ) -> Future[None]:
        fut: Future[None] = Future()
        self._requests.put(_RuntimeRequest(kind=kind, name=name, camera=cam, future=fut))
        if self.stop:
            # stop_all() has drained (or is draining) the queue: fail this request now.
            self._fail_pending_requests("runtime stopping")
        return fut

    def _fail_pending_requests(self, reason: str) -> None:
        """Fail every queued request with RuntimeError(reason). Safe from any thread."""
        while True:
            try:
                req = self._requests.get_nowait()
            except queue.Empty:
                return
            if req.future.set_running_or_notify_cancel():
                req.future.set_exception(RuntimeError(reason))

    def _drain_requests(self) -> None:
        """Apply the requests queued before this call, FIFO. Main thread only; never raises.

        A request enqueued while another is applied (a restart that rebuild_camera_detectors
        asks for, say) waits for the next tick, so a request that keeps re-enqueueing itself
        cannot keep the supervisor from supervising.
        """
        for _ in range(self._requests.qsize()):
            if self.stop:
                return
            try:
                req = self._requests.get_nowait()
            except queue.Empty:
                return
            if not req.future.set_running_or_notify_cancel():
                continue  # the caller cancelled it before it ran
            try:
                if req.kind == "add":
                    if req.camera is None:
                        raise ValueError("add request without a camera")
                    self._apply_add(req.camera)
                elif req.kind == "remove":
                    self._apply_remove(req.name)
                else:
                    self._apply_restart(req.name)
            except (Exception, SystemExit) as exc:
                # SystemExit too: _build_camera_runtime raises it for an unknown proxy mode,
                # and a bad hot-add must never end serve.
                log.warning("[supervisor] %s %s failed: %s", req.kind, req.name, _error_text(exc))
                req.future.set_exception(exc)
            else:
                log.info("[supervisor] %s %s done", req.kind, req.name)
                req.future.set_result(None)

    def _apply_add(self, cam: CameraConfig) -> None:
        """Build, publish, wire detectors for, and start a new camera."""
        if not _CAMERA_NAME_RE.fullmatch(cam.name) or cam.name == "new":
            raise ValueError(f"invalid camera name {cam.name!r}")
        if self.find_camera(cam.name) is not None:
            raise CameraExistsError(f"camera {cam.name!r} already exists")
        rt = self._build_camera_runtime(cam)
        # The add-camera route may already have appended this camera to cfg.cameras.
        if not any(c.name == cam.name for c in self.cfg.cameras):
            self.cfg.cameras.append(cam)
        # Publish before starting, so stop_all() (signal handler) always sees its processes.
        self.cameras = [*self.cameras, rt]
        if self.detectors_enabled:
            try:
                self.rebuild_camera_detectors(cam.name)
            except Exception:
                # As in start(): a detector problem never keeps the camera from recording.
                log.warning(
                    "[supervisor] detectors for %s failed to build", cam.name, exc_info=True
                )
        self._start_camera(rt)

    def _apply_remove(self, name: str) -> None:
        """Unpublish and stop a camera, drop its detector runner, forget its config."""
        rt = self.find_camera(name)
        if rt is None:
            raise CameraNotFoundError(f"camera {name!r} not found")
        # Unpublish first, so the supervisor stops supervising it, then stop it.
        self.cameras = [r for r in self.cameras if r is not rt]
        self._stop_camera(rt)
        runner_name = f"detector_{name}"
        with self._detector_lock:
            for runner in self.detector_runners:
                if runner.name == runner_name:
                    try:
                        runner.teardown()
                    except Exception:
                        log.warning("detector runner teardown failed", exc_info=True)
            self.detector_runners = [r for r in self.detector_runners if r.name != runner_name]
            if self.frame_tap_dispatcher is not None:
                self.frame_tap_dispatcher.consumers = tuple(
                    c
                    for c in self.frame_tap_dispatcher.consumers
                    if getattr(c, "name", None) != runner_name
                )
        self.cfg.cameras[:] = [c for c in self.cfg.cameras if c.name != name]

    def _apply_restart(self, name: str) -> None:
        """Stop a camera, rebuild it from its current config (same FrameHub), start it."""
        old = self.find_camera(name)
        if old is None:
            raise CameraNotFoundError(f"camera {name!r} not found")
        self._stop_camera(old)
        # Rebuild, do not just stop()/start(): the ingestors copy the URLs and settings when
        # they are built, so an edited CameraConfig only takes effect in a new recorder.
        cam = next((c for c in self.cfg.cameras if c.name == name), old.camera)
        new = self._build_camera_runtime(cam, hub=old.hub)
        self.cameras = [new if r is old else r for r in self.cameras]
        if self.detectors_enabled:
            try:
                self.rebuild_camera_detectors(name)
            except Exception:
                # As in start(): a detector problem never keeps the camera from recording.
                log.warning("[supervisor] detectors for %s failed to build", name, exc_info=True)
        self._start_camera(new)

    def run_forever(self) -> None:
        status_every = float(self.cfg.runtime.status_interval_s)
        next_status = 0.0

        while not self.stop:
            # Lifecycle requests first, and regardless of runtime.auto_restart.
            self._drain_requests()
            if self.stop:
                break
            now = time.time()

            # status rendering
            if status_every > 0 and now >= next_status:
                next_status = now + status_every
                self._status_table()

            for rt in self.cameras:
                try:
                    self._supervise_camera(rt, now)
                except Exception as exc:
                    # A taken port or a full disk on one camera must never end serve.
                    rt.last_error = _error_text(exc)
                    log.error(
                        "[supervisor] camera %s: %s", rt.camera.name, rt.last_error, exc_info=True
                    )

            time.sleep(0.5)

    def _supervise_camera(self, rt: CameraRuntime, now: float) -> None:
        """One supervisor pass over one camera: retention, then ingest and proxy restarts.

        The restart schedules live on the CameraRuntime (next_restart_at, proxy_restart_at),
        so a removed or rebuilt camera takes its schedule with it.
        """
        # retention cleanup
        if rt.retention:
            rt.retention.maybe_run()

        if not self.cfg.runtime.auto_restart:
            return
        key = rt.camera.name

        # supervise ingest processes (record + proxy fanout)
        if rt.recorder.has_any():
            procs = rt.recorder.processes()
            all_running = True
            any_dead = False

            for sp in procs:
                if sp.proc is None or sp.proc.poll() is not None:
                    any_dead = True
                    all_running = False
                    break
                if not sp.proc.is_running():
                    all_running = False

            if all_running:
                rt.rec_backoff.reset()
                rt.mark_healthy()
                if rt.proxy_error:
                    # The ingest runs but its proxy cannot start (a taken MJPEG port, say):
                    # keep that error visible instead of clearing it on every healthy tick.
                    rt.last_error = rt.proxy_error
            elif any_dead:
                if rt.next_restart_at <= 0.0:
                    delay = rt.rec_backoff.next_delay()
                    rt.next_restart_at = now + delay
                    # An ffmpeg that never started left no stderr: keep the start error.
                    rt.last_error = _last_stderr_line(procs) or rt.last_error
                    log.warning("[supervisor] ingest for %s died; restarting in %.1fs", key, delay)
                elif now >= rt.next_restart_at:
                    log.info("[supervisor] restarting ingest for %s", key)
                    rt.next_restart_at = 0.0
                    rt.recorder.stop()

                    # For RTSP proxy publish, ensure MediaMTX is up before restarting FFmpeg.
                    if isinstance(rt.proxy, MediaMTXProxyServer):
                        p = rt.proxy.process()
                        if p is None or not p.is_running():
                            self._start_proxy(rt)

                    rt.recorder.start()
                    self._sync_mjpeg_ingest_proc(rt)

        # supervise proxy (MJPEG http server or MediaMTX process)
        if rt.proxy is None:
            return
        if isinstance(rt.proxy, MjpegProxyServer):
            if rt.proxy.is_running():
                rt.proxy_backoff.reset()
                rt.proxy_restart_at = 0.0
                rt.proxy_error = ""
            elif rt.proxy_restart_at <= 0.0:
                delay = rt.proxy_backoff.next_delay()
                rt.proxy_restart_at = now + delay
                log.warning("[supervisor] mjpeg proxy for %s died; restarting in %.1fs", key, delay)
            elif now >= rt.proxy_restart_at:
                log.info("[supervisor] restarting mjpeg proxy for %s", key)
                # Clear the schedule first: if start() raises (port taken), the next tick
                # backs off again instead of retrying on every 0.5 s tick.
                rt.proxy_restart_at = 0.0
                rt.proxy.stop()
                self._sync_mjpeg_ingest_proc(rt)
                self._start_proxy(rt)
        else:
            proc = rt.proxy.process()  # type: ignore[attr-defined]
            if proc is not None and proc.is_running():
                rt.proxy_backoff.reset()
                rt.proxy_restart_at = 0.0
                rt.proxy_error = ""
            elif proc is None or proc.poll() is not None:
                if rt.proxy_restart_at <= 0.0:
                    delay = rt.proxy_backoff.next_delay()
                    rt.proxy_restart_at = now + delay
                    log.warning("[supervisor] proxy for %s died; restarting in %.1fs", key, delay)
                elif now >= rt.proxy_restart_at:
                    log.info("[supervisor] restarting proxy for %s", key)
                    rt.proxy_restart_at = 0.0
                    rt.proxy.stop()  # type: ignore[attr-defined]
                    self._start_proxy(rt)

    def stop_all(self) -> None:
        self.stop = True
        # Nothing drains the queue after this; fail what is queued so no caller waits forever.
        self._fail_pending_requests("runtime stopping")

        # Teardown detector runners
        for runner in self.detector_runners:
            try:
                runner.teardown()
            except Exception:
                pass

        for rt in self.cameras:
            self._stop_camera(rt)

    def _install_signals(self) -> None:
        def _handle(_signum, _frame) -> None:
            log.info("Stopping (SIGINT/SIGTERM)...")
            self.stop_all()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _build_detectors(self) -> None:
        """Build DetectorRunners for cameras that have detectors configured."""
        if not self.detectors_enabled:
            return

        # Create a shared EventSink
        self._event_sink = EventSink()

        for cam_rt in self.cameras:
            cam = cam_rt.camera
            enabled_specs = [s for s in cam.detectors if s.enabled]
            if not enabled_specs:
                continue

            bundle = build_detectors_for_camera(cam, cam.detectors)
            if not bundle.detectors:
                continue

            runner = DetectorRunner(
                name=f"detector_{cam.name}",
                detectors=tuple(bundle.detectors),
                result_sinks=[self._event_sink],
                masks=bundle.masks,
                roi=bundle.roi,
                grid_masks=bundle.grid_masks,
            )
            self.detector_runners.append(runner)

            # Wire into the frame tap dispatcher
            if self.frame_tap_dispatcher is not None:
                # Add the runner as a consumer
                existing = list(self.frame_tap_dispatcher.consumers)
                existing.append(runner)
                self.frame_tap_dispatcher.consumers = tuple(existing)
            else:
                # Create a dispatcher just for detectors
                self.frame_tap_dispatcher = FrameTapDispatcher(consumers=(runner,))

    def rebuild_camera_detectors(self, camera_name: str) -> None:
        """Rebuild detectors for a single camera and atomically swap the runner.

        Reads the current camera config (in-memory), rebuilds the detector
        runner for that camera, and swaps it into the active runtime under
        a lock so no frames are dropped during the transition.

        Args:
            camera_name: Name of the camera whose detectors to rebuild.

        Raises:
            ValueError: If the camera name is not found.
        """
        # Find the CameraRuntime for this camera.
        cam_rt = None
        for rt in self.cameras:
            if rt.camera.name == camera_name:
                cam_rt = rt
                break
        if cam_rt is None:
            raise ValueError(f"camera {camera_name!r} not found")

        cam = cam_rt.camera
        enabled_specs = [s for s in cam.detectors if s.enabled]
        if not enabled_specs:
            # No detectors for this camera -- remove any existing runner.
            with self._detector_lock:
                old_runners = [
                    r for r in self.detector_runners if r.name == f"detector_{camera_name}"
                ]
                for r in old_runners:
                    r.teardown()
                self.detector_runners = [
                    r for r in self.detector_runners if r.name != f"detector_{camera_name}"
                ]
            return

        bundle = build_detectors_for_camera(cam, cam.detectors)
        if not bundle.detectors:
            # No detectors built -- remove any existing runner.
            with self._detector_lock:
                old_runners = [
                    r for r in self.detector_runners if r.name == f"detector_{camera_name}"
                ]
                for r in old_runners:
                    r.teardown()
                self.detector_runners = [
                    r for r in self.detector_runners if r.name != f"detector_{camera_name}"
                ]
            return

        if self._event_sink is None:
            self._event_sink = EventSink()

        new_runner = DetectorRunner(
            name=f"detector_{cam.name}",
            detectors=tuple(bundle.detectors),
            result_sinks=[self._event_sink],
            masks=bundle.masks,
            roi=bundle.roi,
            grid_masks=bundle.grid_masks,
        )

        # Start the new runner's workers before taking the lock, so it is not held while
        # threads start.
        new_runner.setup()

        with self._detector_lock:
            # Teardown old runners for this camera.
            old_runners = [r for r in self.detector_runners if r.name == f"detector_{camera_name}"]
            for r in old_runners:
                r.teardown()

            # Remove old runners and add new one.
            self.detector_runners = [
                r for r in self.detector_runners if r.name != f"detector_{camera_name}"
            ]
            self.detector_runners.append(new_runner)

            # Wire into the dispatcher under the same lock, so a concurrent rebuild or
            # remove cannot lose a consumer (gap-2 G-5).
            if self.frame_tap_dispatcher is not None:
                existing = list(self.frame_tap_dispatcher.consumers)
                # Remove old runner consumers for this camera.
                existing = [
                    c
                    for c in existing
                    if not hasattr(c, "name") or c.name != f"detector_{camera_name}"
                ]
                existing.append(new_runner)
                self.frame_tap_dispatcher.consumers = tuple(existing)

    def _status_table(self) -> None:
        table = Table(title="rtsp-warden status", show_lines=False)
        table.add_column("camera", style="cyan", no_wrap=True)
        table.add_column("rec main", style="green")
        table.add_column("rec sub", style="green")
        table.add_column("proxy", style="magenta")
        table.add_column("last frame age", style="yellow")
        table.add_column("last error", style="red")

        for rt in self.cameras:
            # recorders
            main_status = "-"
            sub_status = "-"
            last_err = ""

            main_sr = rt.recorder.main
            sub_sr = rt.recorder.sub

            if main_sr and main_sr.proc:
                main_status = "RUN" if main_sr.proc.is_running() else f"EXIT({main_sr.proc.poll()})"
                tail = main_sr.proc.stderr_tail()
                if tail:
                    last_err = tail[-1]
            if sub_sr and sub_sr.proc:
                sub_status = "RUN" if sub_sr.proc.is_running() else f"EXIT({sub_sr.proc.poll()})"
                tail = sub_sr.proc.stderr_tail()
                if tail:
                    last_err = tail[-1]

            # proxy
            proxy_status = "-"
            frame_age = "-"
            if rt.proxy is not None:
                if isinstance(rt.proxy, MjpegProxyServer):
                    proxy_status = f"mjpeg :{rt.camera.proxy.port}/mjpeg"
                    if rt.hub:
                        frame, _fid, ts = rt.hub.snapshot()
                        if ts:
                            frame_age = f"{time.time() - ts:.1f}s"
                elif isinstance(rt.proxy, MediaMTXProxyServer):
                    proxy_status = f"rtsp :{rt.camera.proxy.port}/{rt.camera.proxy.path}"

                proc = rt.proxy.process()  # type: ignore[attr-defined]
                if proc:
                    tail = proc.stderr_tail()
                    if tail:
                        last_err = tail[-1]

            table.add_row(
                rt.camera.name, main_status, sub_status, proxy_status, frame_age, last_err[:120]
            )

        self.console.clear()
        self.console.print(table)
