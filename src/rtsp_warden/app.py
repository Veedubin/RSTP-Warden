from __future__ import annotations

import json
import logging
import queue
import re
import signal
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from .actions.base import Action, ActionPayload
from .actions.factory import build_actions
from .actions.queue import ActionJob, ActionQueue, ClipJob, ClipScheduler
from .actions.rules import RuleDecision, RuleEngine
from .clips import build_event_clip, clip_stream_and_chunk
from .config import AppConfig, CameraConfig, DetectorSpec, compute_tap_settings
from .db import schema as db_schema
from .detectors.event_builder import EventBuilder, EventInfo, MotionBurst
from .detectors.model_registry import load_descriptor
from .detectors.registry import DEFAULT_ONNX_MODEL, build_detectors_for_camera
from .detectors.runner import DetectorRunner
from .detectors.sinks import EventSink
from .detectors.tracking import Tracker
from .ffmpeg import ExponentialBackoff, redact_text
from .frame_tap import FrameConsumer, FrameTapDispatcher
from .proxy.mjpeg import FrameHub, MjpegProxyServer
from .proxy.rtsp_mediamtx import MediaMTXProxyServer
from .recorder import CameraRecorder, StreamIngestor
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

    # This camera's own frame tap fan-out: its detector runner, then any --frame-consumer
    # consumers. It is created before the recorder, whose proxy-stream ingestor feeds it.
    dispatcher: FrameTapDispatcher = field(default_factory=FrameTapDispatcher)
    # This camera's rules. Built with the runtime and kept across detector rebuilds, so a hot
    # reload does not reset rule cooldowns; None only in hand-built test runtimes.
    rule_engine: RuleEngine | None = None

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


# A clip job waits this long after ended_at + clips.post_seconds, so ffmpeg has closed the
# segment that holds the end of the clip window before it is cut (R8).
CLIP_SETTLE_SECONDS = 2.0


def _utc(dt: datetime) -> datetime:
    """*dt* as an aware UTC datetime; naive values are UTC (how the database stores them)."""
    return db_schema.as_utc(dt) or dt


def _iso_utc(dt: datetime) -> str:
    """ISO 8601 with an explicit +00:00 offset, as action payloads carry it."""
    return _utc(dt).isoformat()


@dataclass
class AppRuntime:
    cfg: AppConfig
    console: Console = field(default_factory=Console)
    stop: bool = False
    cameras: list[CameraRuntime] = field(default_factory=list)
    # --frame-consumer consumers: every camera's dispatcher gets each one after its runner.
    frame_consumers: tuple[FrameConsumer, ...] = ()
    detectors_enabled: bool = True
    detector_runners: list[DetectorRunner] = field(default_factory=list)
    # Base of the absolute links in action payloads; cli.serve sets it (R19).
    public_url: str = "http://localhost:8080"
    # Configured actions by name, the queue that runs them and the delayed clip jobs: build()
    # creates them, start() starts their worker threads, stop_all() stops them.
    actions: dict[str, Action] = field(default_factory=dict)
    action_queue: ActionQueue | None = None
    clip_scheduler: ClipScheduler | None = None
    # Events that a rule with clip: true matched, waiting for their close (under _clip_lock).
    _clip_events: set[int] = field(default_factory=set, init=False, repr=False)
    _clip_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # The ingest restart each camera last requested because its frame tap had to change.
    _tap_restarts: dict[str, Future[None]] = field(default_factory=dict, init=False, repr=False)
    # Threads tearing down runners a rebuild swapped out (see _retire_runners).
    _retiring: list[threading.Thread] = field(default_factory=list, init=False, repr=False)
    _event_sink: EventSink | None = field(default=None, init=False, repr=False)
    # Model input widths by model name: status polls ask for them every few seconds, so each
    # descriptor is read once; build() and rebuild_camera_detectors() forget them.
    _input_widths: dict[str, int | None] = field(default_factory=dict, init=False, repr=False)
    _detector_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Lifecycle requests posted from any thread and applied by run_forever on the main thread.
    # SimpleQueue's get/put are C calls a signal handler cannot interrupt halfway, so
    # stop_all() run from the SIGTERM handler can drain it; queue.Queue's lock could deadlock.
    _requests: queue.SimpleQueue[_RuntimeRequest] = field(
        default_factory=queue.SimpleQueue, init=False, repr=False
    )

    def build(self) -> None:
        self._input_widths.clear()
        # Process-wide action plumbing; nothing starts until start().
        self.actions = build_actions(self.cfg)
        self.action_queue = ActionQueue(self.actions)
        self.clip_scheduler = ClipScheduler(builder=self._build_clip)
        self.cameras = [self._build_camera_runtime(cam) for cam in self.cfg.cameras]

        # Wire detector framework
        self._build_detectors()

    def _build_camera_runtime(
        self,
        cam: CameraConfig,
        *,
        hub: FrameHub | None = None,
        rule_engine: RuleEngine | None = None,
    ) -> CameraRuntime:
        """Build one camera's recorder, proxy, retention and backoffs. Starts nothing.

        ``hub`` is reused when the camera still serves MJPEG, so a restart keeps the
        FrameHub that open ``/live.mjpeg`` viewers hold. It is ignored otherwise.
        ``rule_engine`` is reused by a restart whose rules did not change, so rule
        cooldowns survive an edit or a detect_fps change.
        """
        proxy: object | None = None

        # In unified ingest mode, the hub (if any) is shared between ingest and proxy.
        if cam.proxy.enabled and cam.proxy.mode == "mjpeg":
            if hub is None:
                hub = FrameHub()
        else:
            hub = None

        # The camera's own dispatcher exists before its recorder, so the proxy-stream ingestor
        # captures it; _build_detectors / rebuild_camera_detectors attach the runner later.
        dispatcher = FrameTapDispatcher(consumers=self.frame_consumers)
        recorder = CameraRecorder(
            camera=cam,
            runtime=self.cfg.runtime,
            proxy_hub=hub,
            frame_tap_dispatcher=dispatcher,
            frame_tap_required=self._wants_frames(cam),
            # detect_fps and the widest model input: the value rebuild_camera_detectors compares
            # the running ingest against, so a restart can never loop (R18).
            tap_settings=self.tap_settings_for(cam),
        )

        if cam.proxy.enabled:
            if cam.proxy.mode == "mjpeg":
                assert hub is not None
                proxy = MjpegProxyServer(camera=cam, runtime=self.cfg.runtime, hub=hub)
            elif cam.proxy.mode == "rtsp":
                proxy = MediaMTXProxyServer(camera=cam, runtime=self.cfg.runtime)
            else:
                raise SystemExit(f"Unsupported proxy mode: {cam.proxy.mode}")

        # Every camera gets a RetentionManager, recording or not: event thumbnails live
        # under <record.output_dir>/<camera>/thumbnails/ either way (ruling R20). With
        # recording off it sweeps only thumbnails/ and clips/, never old segments.
        retention = RetentionManager(
            camera_name=cam.name,
            camera_root=cam.record.output_dir / cam.name,
            cfg=resolve_retention(cam, self.cfg.retention),
            only_subdirs=None if cam.record.enabled else ("thumbnails", "clips"),
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
            dispatcher=dispatcher,
            rule_engine=rule_engine if rule_engine is not None else RuleEngine(cam.name, cam.rules),
        )

    def _wants_frames(self, cam: CameraConfig) -> bool:
        """True when something consumes this camera's tap: a --frame-consumer, or enabled
        detectors (unless serve runs with --no-detectors)."""
        if self.frame_consumers:
            return True
        return self.detectors_enabled and any(s.enabled for s in cam.detectors)

    def find_camera(self, name: str) -> CameraRuntime | None:
        """Return the CameraRuntime named *name*, or None. Safe to call from any thread."""
        for rt in self.cameras:
            if rt.camera.name == name:
                return rt
        return None

    def start(self) -> None:
        self._install_signals()

        # Action and clip workers first, so the first event already has somewhere to go.
        if self.action_queue is not None:
            self.action_queue.start()
        if self.clip_scheduler is not None:
            self.clip_scheduler.start()

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
                self.rebuild_camera_detectors(cam.name, from_lifecycle=True)
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
            rt.dispatcher.consumers = ()
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
        same_rules = old.rule_engine is not None and old.rule_engine.rules == list(cam.rules)
        new = self._build_camera_runtime(
            cam, hub=old.hub, rule_engine=old.rule_engine if same_rules else None
        )
        self.cameras = [new if r is old else r for r in self.cameras]
        if self.detectors_enabled:
            try:
                self.rebuild_camera_detectors(name, from_lifecycle=True)
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
        # ...and wait for the runners earlier rebuilds swapped out (they close their events).
        for thread in list(self._retiring):
            thread.join(timeout=5.0)

        for rt in self.cameras:
            self._stop_camera(rt)

        # The runners are down, so nothing enqueues any more. Queued action jobs get the
        # queue's own stop timeout; clip jobs not yet due are dropped (segments stay on disk).
        for worker in (self.action_queue, self.clip_scheduler):
            if worker is None:
                continue
            try:
                worker.stop()
            except Exception:
                log.warning("[supervisor] stopping %s failed", type(worker).__name__, exc_info=True)

    def _install_signals(self) -> None:
        def _handle(_signum, _frame) -> None:
            log.info("Stopping (SIGINT/SIGTERM)...")
            self.stop_all()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _build_detectors(self) -> None:
        """Build a DetectorRunner per camera with enabled detectors, on its own dispatcher.

        Runners are only built here; start() sets them up.
        """
        if not self.detectors_enabled:
            return

        # Create a shared EventSink
        self._event_sink = EventSink()

        for cam_rt in self.cameras:
            runner = self._make_runner(cam_rt.camera)
            if runner is None:
                continue
            self.detector_runners.append(runner)
            self._set_consumers(cam_rt, runner)

    def _make_runner(self, cam: CameraConfig) -> DetectorRunner | None:
        """Build (not set up) the camera's DetectorRunner; None when it has no detectors.

        Every runner gets fresh tracking state (Tracker, EventBuilder, MotionBurst), so a
        rebuild resets it (spec 7.1). The EventBuilder hands opened events to the camera's
        rules and closed ones to the clip scheduler. A detector that cannot be built (an
        unknown model, say) only costs that camera its detection, never its recording.
        """
        if not self.detectors_enabled or not any(s.enabled for s in cam.detectors):
            return None
        try:
            bundle = build_detectors_for_camera(
                cam, cam.detectors, models_dir=self.cfg.runtime.models_dir
            )
        except Exception:
            log.warning("[detect] building the detectors of %s failed", cam.name, exc_info=True)
            return None
        if not bundle.detectors:
            return None
        if self._event_sink is None:
            self._event_sink = EventSink()
        tap_fps, _tap_width = self.tap_settings_for(cam)
        event_builder = EventBuilder(
            camera=cam.name,
            output_dir=Path(cam.record.output_dir),
            area_masks=bundle.area_masks,
            on_open=self._on_event_open,
            on_close=self._on_event_close,
        )
        # One worker keeps each camera's frames in order (MOG2 and the tracker are stateful);
        # a short queue keeps drop-oldest meaning "freshest frame" when inference falls behind.
        return DetectorRunner(
            name=f"detector_{cam.name}",
            detectors=tuple(bundle.detectors),
            result_sinks=[self._event_sink],
            queue_maxsize=8,
            worker_count=1,
            masks=bundle.masks,
            roi=bundle.roi,
            grid_masks=bundle.grid_masks,
            camera=cam.name,
            slots=tuple(bundle.slots),
            tracker=Tracker(grace_seconds=cam.track_grace_seconds, min_frames=cam.min_track_frames),
            event_builder=event_builder,
            motion_burst=MotionBurst(
                min_frames=cam.min_track_frames, grace_seconds=cam.track_grace_seconds
            ),
            tap_fps=tap_fps,
        )

    def _set_consumers(self, cam_rt: CameraRuntime, runner: DetectorRunner | None) -> None:
        """Point the camera's dispatcher at its runner (if any), then the global consumers.

        Assigning a new tuple is safe while the tap reader thread iterates the old one.
        """
        head: tuple[FrameConsumer, ...] = (runner,) if runner is not None else ()
        cam_rt.dispatcher.consumers = (*head, *self.frame_consumers)

    def rebuild_camera_detectors(self, camera_name: str, *, from_lifecycle: bool = False) -> None:
        """Rebuild one camera's detectors from its in-memory CameraConfig and swap the runner.

        The new runner is set up first, then swapped into ``detector_runners`` and onto the
        camera's own dispatcher under ``_detector_lock``, so the tap never feeds a stopped
        runner; the old runner is torn down afterwards, on a background thread
        (``_retire_runners``) so a worker stuck in a model download never holds the
        caller. Without enabled detectors the
        dispatcher keeps only the --frame-consumer consumers. When the frame tap the camera
        needs changed (detect_fps, a wider model input, a first detector on a camera without
        an ingest), one ingest restart is requested; run_forever applies it (R18).

        Args:
            camera_name: Name of the camera whose detectors to rebuild.
            from_lifecycle: True when _apply_add / _apply_restart call it on the main thread
                inside a drain, right after building the camera from its current config. The
                fresh recorder already carries tap_settings_for(cam), and a drain must never
                enqueue a lifecycle request, so no restart is considered then.

        Raises:
            ValueError: If the camera name is not found.
        """
        cam_rt = self.find_camera(camera_name)
        if cam_rt is None:
            raise ValueError(f"camera {camera_name!r} not found")

        self._input_widths.clear()  # a config change may have added or edited a model
        runner_name = f"detector_{camera_name}"
        new_runner = self._make_runner(cam_rt.camera)
        if new_runner is not None:
            new_runner.setup()

        with self._detector_lock:
            old_runners = [r for r in self.detector_runners if r.name == runner_name]
            self.detector_runners = [r for r in self.detector_runners if r.name != runner_name]
            if new_runner is not None:
                self.detector_runners.append(new_runner)
            self._set_consumers(cam_rt, new_runner)

        if old_runners:
            self._retire_runners(old_runners, camera_name)

        if not from_lifecycle:
            self._request_restart_if_tap_changed(cam_rt)

    def _retire_runners(self, runners: list[DetectorRunner], camera_name: str) -> None:
        """Tear runners a rebuild swapped out down on a daemon thread.

        A runner's teardown joins its worker (up to 5 s) and then closes its open events.
        A worker stuck in a model download would otherwise hold the web request (or the
        supervisor tick) that rebuilt the camera for those seconds. The runners are off
        the camera's dispatcher already, so they get no new frames; stop_all() joins the
        teardowns that are still running.
        """

        def teardown_all() -> None:
            for runner in runners:
                try:
                    runner.teardown()
                except Exception:
                    log.warning(
                        "detector runner teardown failed for %s", camera_name, exc_info=True
                    )

        thread = threading.Thread(
            target=teardown_all, name=f"detector-teardown:{camera_name}", daemon=True
        )
        with self._detector_lock:
            self._retiring = [t for t in self._retiring if t.is_alive()]
            self._retiring.append(thread)
        thread.start()

    # ------------------------------------------------------------------
    # Detection wiring: frame tap, events -> rules -> actions and clips
    # ------------------------------------------------------------------

    def payload_for(self, event: EventInfo) -> ActionPayload:
        """The payload every action gets for *event*, with absolute links from public_url."""
        event_url = f"{self.public_url.rstrip('/')}/events/{event.id}"
        return ActionPayload(
            camera=event.camera,
            label=event.label,
            confidence=float(event.confidence),
            zone=event.zone,
            started_at=_iso_utc(event.started_at),
            ended_at=_iso_utc(event.ended_at) if event.ended_at is not None else None,
            thumbnail_url=f"{event_url}/thumbnail.jpg" if event.thumbnail_path else None,
            clip_url=f"{event_url}/clip" if event.clip_path else None,
            event_url=event_url,
            test=event.event_type == "test",
        )

    def dispatch_event(
        self, event: EventInfo, *, bypass_cooldown: bool = False, allow_clip: bool = True
    ) -> RuleDecision | None:
        """Run the camera's rules on an opened event and queue the matched actions (spec 8.3).

        The EventBuilder calls this (through _on_event_open) when an event opens, on the
        runner's worker thread. The "Fire test event" route calls it with
        ``bypass_cooldown=True, allow_clip=False`` (R13). Each matched action is queued once
        per event, with the thumbnail file attached when it exists. Rules held back by their
        cooldown are noted on the event as ``suppressed_by`` (spec 8.2). When a matched rule
        has ``clip: true`` and ``allow_clip`` is set, the clip is queued when the event
        closes. Returns None when the camera is unknown or has no rule engine.
        """
        rt = self.find_camera(event.camera)
        if rt is None or rt.rule_engine is None:
            return None
        decision = rt.rule_engine.evaluate(event, bypass_cooldown=bypass_cooldown)
        if decision.suppressed:
            self._note_suppressed(event.id, list(decision.suppressed))
        if decision.matched and self.action_queue is not None:
            payload = self.payload_for(event)
            attachment = self._thumbnail_file(event)
            queued: set[str] = set()
            for match in decision.matched:
                for action_name in match.actions:
                    if action_name in queued:
                        continue  # two rules naming one action send it once
                    queued.add(action_name)
                    if action_name not in self.actions:
                        log.warning("[actions] %s: no action named %r", event.camera, action_name)
                        continue
                    self.action_queue.enqueue(
                        ActionJob(
                            event_id=event.id,
                            action_name=action_name,
                            payload=payload,
                            attachment=attachment,
                        )
                    )
        if allow_clip and any(match.clip for match in decision.matched):
            with self._clip_lock:
                self._clip_events.add(event.id)
        return decision

    def _on_event_open(self, event: EventInfo) -> None:
        """EventBuilder hook: rules fire on open. Never raises into the runner's worker."""
        try:
            self.dispatch_event(event)
        except Exception:
            log.warning("[actions] rules for event %s failed", event.id, exc_info=True)

    def _on_event_close(self, event: EventInfo) -> None:
        """EventBuilder hook: queue the clip of a closed event that a clip rule matched (R8).

        The job may run at ended_at + clips.post_seconds + CLIP_SETTLE_SECONDS. Cameras that
        do not record (R20) and a disabled ``clips`` section get no clip.
        """
        with self._clip_lock:
            wanted = event.id in self._clip_events
            self._clip_events.discard(event.id)
        if not wanted or self.clip_scheduler is None or event.ended_at is None:
            return
        cam = self._camera_config(event.camera)
        if cam is None or not cam.record.enabled or not self.cfg.clips.enabled:
            return
        ended = _utc(event.ended_at)
        not_before = ended.timestamp() + float(self.cfg.clips.post_seconds) + CLIP_SETTLE_SECONDS
        try:
            self.clip_scheduler.schedule(
                ClipJob(
                    event_id=event.id,
                    camera=event.camera,
                    started_at=_utc(event.started_at),
                    ended_at=ended,
                    not_before=not_before,
                )
            )
        except Exception:
            log.warning("[clips] could not queue the clip of event %s", event.id, exc_info=True)

    def _build_clip(self, job: ClipJob) -> Path | None:
        """ClipScheduler builder: cut the event's clip from the camera's recorded segments.

        Returns the clip's path relative to record.output_dir (what events.clip_path stores,
        R20), or None when the camera is gone, records nothing, or ffmpeg produced nothing.
        """
        cam = self._camera_config(job.camera)
        if cam is None or not cam.record.enabled:
            return None
        stream, chunk_seconds = clip_stream_and_chunk(cam)
        path = build_event_clip(
            camera_root=Path(cam.record.output_dir) / cam.name,
            stream=stream,
            chunk_seconds=chunk_seconds,
            started_at=job.started_at,
            ended_at=job.ended_at,
            pre_seconds=float(self.cfg.clips.pre_seconds),
            post_seconds=float(self.cfg.clips.post_seconds),
            max_duration=float(self.cfg.clips.max_duration),
            event_id=job.event_id,
            ffmpeg_path=self.cfg.runtime.ffmpeg_path,
        )
        if path is None:
            return None
        try:
            return Path(path).relative_to(cam.record.output_dir)
        except ValueError:
            return Path(path)

    def _note_suppressed(self, event_id: int, rule_names: list[str]) -> None:
        """Record on the event which rules' cooldown held it back (metadata suppressed_by)."""
        try:
            row = db_schema.get_event(event_id)
            meta = json.loads(row.metadata_json or "{}") if row is not None else {}
            meta["suppressed_by"] = rule_names
            db_schema.update_event(event_id, metadata=meta)
        except Exception:
            log.warning(
                "[actions] could not note suppressed_by on event %s", event_id, exc_info=True
            )

    def _camera_config(self, name: str) -> CameraConfig | None:
        """The running camera's config, else the configured one, else None."""
        rt = self.find_camera(name)
        if rt is not None:
            return rt.camera
        return next((c for c in self.cfg.cameras if c.name == name), None)

    def _thumbnail_file(self, event: EventInfo) -> Path | None:
        """Absolute path of the event's thumbnail when the file exists (ntfy/Apprise attach it)."""
        cam = self._camera_config(event.camera)
        if cam is None or not event.thumbnail_path:
            return None
        path = Path(cam.record.output_dir) / event.thumbnail_path
        return path if path.is_file() else None

    def tap_settings_for(self, cam: CameraConfig) -> tuple[float, int]:
        """The frame tap this camera needs: (detect_fps, widest enabled model input, >= 320).

        The recorder is built with it and rebuild_camera_detectors compares the running
        ingest against it, so the two never disagree and a restart cannot loop (R18).
        """
        return compute_tap_settings(cam, self._model_input_width)

    def _model_input_width(self, spec: DetectorSpec) -> int | None:
        """input_size[0] of an onnx spec's model; None for other types and unknown models."""
        if spec.type != "onnx":
            return None
        model = spec.model or DEFAULT_ONNX_MODEL
        if model in self._input_widths:
            return self._input_widths[model]
        width: int | None
        try:
            desc = load_descriptor(model, self.cfg.runtime.models_dir)
        except Exception:
            width = None  # the registry skips a detector whose model it cannot resolve, too
        else:
            width = int(desc.input_size[0])
        self._input_widths[model] = width
        return width

    @staticmethod
    def _tap_ingestor(rt: CameraRuntime) -> StreamIngestor | None:
        """The ingestor of the camera's proxy stream: the one that carries its frame tap."""
        return rt.recorder.main if rt.camera.proxy.stream == "main" else rt.recorder.sub

    def _tap_restart_needed(self, rt: CameraRuntime) -> bool:
        """True when the ingest must restart to give the camera's frame consumers their tap.

        That is when something consumes frames and either no ingestor carries a tap (the
        first detector on a camera that records and proxies nothing) or the running tap's
        (fps, width) differ from tap_settings_for() (detect_fps edited, a wider model).
        """
        if not self._wants_frames(rt.camera):
            return False
        ing = self._tap_ingestor(rt)
        if ing is None or not ing.frame_tap_enabled:
            return True
        running = (float(ing.frame_tap_fps), int(ing.frame_tap_scale_width))
        return running != self.tap_settings_for(rt.camera)

    def _request_restart_if_tap_changed(self, rt: CameraRuntime) -> None:
        """Queue one ingest restart when the tap must change; run_forever applies it."""
        if not self._tap_restart_needed(rt):
            return
        name = rt.camera.name
        pending = self._tap_restarts.get(name)
        if pending is not None and not pending.done():
            return  # the queued restart rebuilds from the newest config anyway
        fps, width = self.tap_settings_for(rt.camera)
        log.info(
            "[detect] %s: frame tap now %g fps at %d px; restarting its ingest", name, fps, width
        )
        self._tap_restarts[name] = self.request_restart_camera(name)

    def find_runner(self, name: str) -> DetectorRunner | None:
        """The current DetectorRunner of camera *name*, or None. Safe from any thread."""
        runner_name = f"detector_{name}"
        for runner in list(self.detector_runners):
            if runner.name == runner_name:
                return runner
        return None

    def detection_status(self, name: str) -> dict[str, Any] | None:
        """JSON-safe detection status of camera *name*, or None when there is no such camera.

        runner.status() (counters and per-detector rows), plus ``camera``, ``enabled`` (a
        runner exists), the tap the camera needs (``tap_fps``, ``tap_width``) and
        ``restart_pending`` (its running ingest does not provide that tap yet). Each
        per-detector row carries ``error``: why that detector's model is not loaded, or None.
        """
        rt = self.find_camera(name)
        if rt is None:
            return None
        runner = self.find_runner(name)
        status: dict[str, Any] = dict(runner.status()) if runner is not None else {}
        if runner is not None:
            rows = status.get("detectors")
            if not isinstance(rows, list):
                rows = [{} for _ in runner.detectors]
                status["detectors"] = rows
            for row, det in zip(rows, runner.detectors, strict=False):
                if isinstance(row, dict):
                    error = getattr(det, "error", None)
                    row.setdefault("error", str(error) if error else None)
        tap_fps, tap_width = self.tap_settings_for(rt.camera)
        status.update(
            camera=name,
            enabled=runner is not None,
            tap_fps=float(tap_fps),
            tap_width=int(tap_width),
            restart_pending=bool(self._tap_restart_needed(rt)),
        )
        return status

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

            # ffmpeg prints the expanded camera URL when it cannot open it: mask the userinfo.
            shown_err = redact_text(last_err[:1000])[:120]
            table.add_row(
                rt.camera.name, main_status, sub_status, proxy_status, frame_age, shown_err
            )

        self.console.clear()
        self.console.print(table)
