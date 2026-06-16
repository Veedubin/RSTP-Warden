from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from .config import AppConfig, CameraConfig
from .detectors.registry import build_detector, build_masks, build_roi
from .detectors.roi import Mask
from .detectors.runner import DetectorRunner
from .detectors.sinks import EventSink
from .ffmpeg import ExponentialBackoff
from .frame_tap import FrameTapDispatcher
from .proxy.mjpeg import FrameHub, MjpegProxyServer
from .proxy.rtsp_mediamtx import MediaMTXProxyServer
from .recorder import CameraRecorder
from .retention import RetentionManager

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

    def build(self) -> None:
        self.cameras = []

        for cam in self.cfg.cameras:
            proxy: object | None = None
            hub: FrameHub | None = None

            # In unified ingest mode, the hub (if any) is shared between ingest and proxy.
            if cam.proxy.enabled and cam.proxy.mode == "mjpeg":
                hub = FrameHub()

            recorder = CameraRecorder(
                camera=cam,
                runtime=self.cfg.runtime,
                proxy_hub=hub,
                frame_tap_dispatcher=self.frame_tap_dispatcher,
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
                retention = RetentionManager(
                    camera_name=cam.name,
                    camera_root=cam.record.output_dir / cam.name,
                    cfg=cam.record.retention,
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

            self.cameras.append(
                CameraRuntime(
                    camera=cam,
                    recorder=recorder,
                    proxy=proxy,
                    hub=hub,
                    retention=retention,
                    rec_backoff=rec_backoff,
                    proxy_backoff=proxy_backoff,
                )
            )

        # Wire detector framework
        self._build_detectors()

    def start(self) -> None:
        self._install_signals()

        # Start detector runners
        for runner in self.detector_runners:
            try:
                runner.setup()
            except Exception:
                log.warning("detector runner setup failed", exc_info=True)

        for rt in self.cameras:
            # Keep MJPEG health endpoint pointed at the current ingest process.
            self._sync_mjpeg_ingest_proc(rt)
            # Ordering matters for RTSP proxy: MediaMTX must be listening before FFmpeg publishes.
            if rt.proxy is not None and isinstance(rt.proxy, MediaMTXProxyServer):
                rt.proxy.start()

            if rt.recorder.has_any():
                rt.recorder.start()

            if rt.proxy is not None and isinstance(rt.proxy, MjpegProxyServer):
                # MJPEG proxy is serve-only; it consumes frames from the ingest-owned hub.
                self._sync_mjpeg_ingest_proc(rt)
                rt.proxy.start()

        self._status_table()

    def _sync_mjpeg_ingest_proc(self, rt: CameraRuntime) -> None:
        """Keep MJPEG /healthz wired to the current ingest process."""
        if rt.proxy is None or not isinstance(rt.proxy, MjpegProxyServer):
            return
        stream = rt.camera.proxy.stream
        ing = rt.recorder.main if stream == "main" else rt.recorder.sub
        rt.proxy.ingest_proc = ing.proc if ing else None

    def run_forever(self) -> None:
        status_every = float(self.cfg.runtime.status_interval_s)
        next_status = 0.0

        # Track next allowed restart times (wall-clock)
        next_rec_restart: dict[str, float] = {}
        next_proxy_restart: dict[str, float] = {}

        while not self.stop:
            now = time.time()

            # status rendering
            if status_every > 0 and now >= next_status:
                next_status = now + status_every
                self._status_table()

            for rt in self.cameras:
                # retention cleanup
                if rt.retention:
                    rt.retention.maybe_run()

                # supervise ingest processes (record + proxy fanout)
                if rt.recorder.has_any() and self.cfg.runtime.auto_restart:
                    key = rt.camera.name
                    sched = next_rec_restart.get(key, 0.0)

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
                        next_rec_restart[key] = 0.0
                    elif any_dead:
                        if sched <= 0.0:
                            delay = rt.rec_backoff.next_delay()
                            next_rec_restart[key] = now + delay
                            log.warning(
                                f"[supervisor] ingest for {key} died; restarting in {delay:.1f}s"
                            )
                        elif now >= sched:
                            log.info(f"[supervisor] restarting ingest for {key}")
                            rt.recorder.stop()

                            # For RTSP proxy publish, ensure MediaMTX is up before restarting FFmpeg.
                            if rt.proxy is not None and isinstance(rt.proxy, MediaMTXProxyServer):
                                p = rt.proxy.process()
                                if p is None or not p.is_running():
                                    rt.proxy.start()

                            rt.recorder.start()
                            self._sync_mjpeg_ingest_proc(rt)
                            next_rec_restart[key] = 0.0

                # supervise proxy (MJPEG http server or MediaMTX process)
                if rt.proxy is not None and self.cfg.runtime.auto_restart:
                    key = rt.camera.name
                    sched = next_proxy_restart.get(key, 0.0)

                    if isinstance(rt.proxy, MjpegProxyServer):
                        if rt.proxy.is_running():
                            rt.proxy_backoff.reset()
                            next_proxy_restart[key] = 0.0
                        else:
                            if sched <= 0.0:
                                delay = rt.proxy_backoff.next_delay()
                                next_proxy_restart[key] = now + delay
                                log.warning(
                                    f"[supervisor] mjpeg proxy for {key} died; restarting in {delay:.1f}s"
                                )
                            elif now >= sched:
                                log.info(f"[supervisor] restarting mjpeg proxy for {key}")
                                rt.proxy.stop()
                                self._sync_mjpeg_ingest_proc(rt)
                                rt.proxy.start()
                                next_proxy_restart[key] = 0.0
                    else:
                        proc = rt.proxy.process()  # type: ignore[attr-defined]
                        if proc is not None and proc.is_running():
                            rt.proxy_backoff.reset()
                            next_proxy_restart[key] = 0.0
                        else:
                            dead = (proc is None) or (proc.poll() is not None)
                            if dead:
                                if sched <= 0.0:
                                    delay = rt.proxy_backoff.next_delay()
                                    next_proxy_restart[key] = now + delay
                                    log.warning(
                                        f"[supervisor] proxy for {key} died; restarting in {delay:.1f}s"
                                    )
                                elif now >= sched:
                                    log.info(f"[supervisor] restarting proxy for {key}")
                                    rt.proxy.stop()  # type: ignore[attr-defined]
                                    rt.proxy.start()  # type: ignore[attr-defined]
                                    next_proxy_restart[key] = 0.0

            time.sleep(0.5)

    def stop_all(self) -> None:
        self.stop = True

        # Teardown detector runners
        for runner in self.detector_runners:
            try:
                runner.teardown()
            except Exception:
                pass

        for rt in self.cameras:
            try:
                rt.recorder.stop()
            except Exception:
                pass
            if rt.proxy is not None:
                try:
                    rt.proxy.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass

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

            detectors = []
            for spec in enabled_specs:
                try:
                    det = build_detector(spec, cam.name)
                    detectors.append(det)
                except Exception:
                    log.warning(
                        "failed to build detector type=%s for camera=%s",
                        spec.type,
                        cam.name,
                        exc_info=True,
                    )

            if not detectors:
                continue

            # Build ROI and masks from the first spec that defines them
            # (all specs for a camera share the same runner, so ROI/masks
            # come from the first spec that has them)
            runner_roi = None
            runner_masks: list[Mask] = []
            for spec in enabled_specs:
                roi = build_roi(spec)
                masks = build_masks(spec)
                if roi is not None:
                    runner_roi = roi
                if masks:
                    runner_masks.extend(masks)

            runner = DetectorRunner(
                name=f"detector_{cam.name}",
                detectors=tuple(detectors),
                result_sinks=[self._event_sink],
                masks=runner_masks,
                roi=runner_roi,
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


def run_app(cfg: AppConfig) -> None:
    rt = AppRuntime(cfg=cfg)
    rt.build()
    rt.start()
    try:
        rt.run_forever()
    finally:
        rt.stop_all()
