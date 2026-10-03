from __future__ import annotations

import logging
import os
import subprocess
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import (
    CameraConfig,
    ProxyConfig,
    RuntimeConfig,
    StreamRecordConfig,
    compute_tap_settings,
)
from .ffmpeg import ManagedProcess, build_ffmpeg_ingest_cmd
from .frame_tap import FrameTapDispatcher

# NOTE: FrameHub lives in proxy/mjpeg.py, but is used by the unified ingest layer
# as a purely in-memory fanout target.
from .proxy.mjpeg import FrameHub

log = logging.getLogger(__name__)

StreamName = Literal["main", "sub"]


def _segment_out_pattern(
    *, out_dir: Path, camera_name: str, stream_name: StreamName, container: str
) -> str:
    """Canonical recording path contract.

    {output_dir}/{camera}/{stream}/{camera}_{stream}_%Y%m%d_%H%M%S.ts
    """
    base = out_dir / camera_name / stream_name
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{camera_name}_{stream_name}_%Y%m%d_%H%M%S.{container}")


@dataclass
class StreamIngestor:
    """Single upstream RTSP pull for a (camera, stream), with optional fanout outputs."""

    camera_name: str
    stream_name: StreamName
    upstream_url: str

    runtime: RuntimeConfig

    # optional outputs
    record_cfg: StreamRecordConfig | None = None
    record_output_dir: Path | None = None

    proxy_cfg: ProxyConfig | None = None
    mjpeg_hub: FrameHub | None = None
    rtsp_publish_url: str | None = None

    # Frame tap (new) — dedicated low-res MJPEG stream for CV consumers
    frame_tap_enabled: bool = False
    frame_tap_fps: float = 5.0
    frame_tap_scale_width: int = 320
    frame_tap_dispatcher: FrameTapDispatcher | None = None

    # Audio opt-in for recording (Sprint 4)
    audio: bool = False

    # runtime state
    proc: ManagedProcess | None = None
    _mjpeg_thread: threading.Thread | None = None

    # Frame tap pipe. Both ends exist only inside start(): the parent closes its write end
    # once ffmpeg holds a copy, and the reader thread owns (and closes) the read end.
    _frame_tap_read_fd: int | None = None
    _frame_tap_write_fd: int | None = None
    _frame_tap_thread: threading.Thread | None = None

    @property
    def frame_tap_write_fd(self) -> int | None:
        """The parent's copy of the tap pipe's write end; None once ffmpeg was spawned."""
        return self._frame_tap_write_fd

    def _frame_tap_on(self) -> bool:
        """A tap needs somebody to dispatch to; an unread pipe would fill up and stall ffmpeg."""
        return self.frame_tap_enabled and self.frame_tap_dispatcher is not None

    def required(self) -> bool:
        """True if anything requires this stream."""
        return bool(
            self.record_cfg is not None
            or self.mjpeg_hub is not None
            or self.rtsp_publish_url
            or self._frame_tap_on()
        )

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.is_running())

    def start(self) -> None:
        if not self.required():
            return
        if self.proc is not None and self.proc.is_running():
            return

        # Determine which outputs are enabled.
        record_enabled = self.record_cfg is not None and bool(self.record_cfg.enabled)
        mjpeg_enabled = self.mjpeg_hub is not None
        rtsp_publish_enabled = self.rtsp_publish_url is not None
        frame_tap_on = self._frame_tap_on()

        out_pattern = ""
        chunk_seconds = 300
        container = "mkv"
        rtsp_transport_in = "tcp"

        # Reuse per-stream rtsp_transport even if record is disabled (it is the only knob today).
        if self.record_cfg is not None:
            rtsp_transport_in = self.record_cfg.rtsp_transport
            if record_enabled:
                assert self.record_output_dir is not None
                out_pattern = _segment_out_pattern(
                    out_dir=self.record_output_dir,
                    camera_name=self.camera_name,
                    stream_name=self.stream_name,
                    container=self.record_cfg.container,
                )
                chunk_seconds = int(self.record_cfg.chunk_seconds)
                container = self.record_cfg.container

        mjpeg_fps = 7
        mjpeg_scale_width = 0
        if mjpeg_enabled and self.proxy_cfg is not None:
            mjpeg_fps = int(self.proxy_cfg.fps)
            mjpeg_scale_width = int(self.proxy_cfg.scale_width)

        rtsp_transport_out = "tcp"
        if self.proxy_cfg is not None:
            rtsp_transport_out = "tcp"  # keep deterministic for local publish

        # Frame tap: a dedicated pipe for the low-res MJPEG output. Popen's pass_fds hands the
        # write end to ffmpeg under the SAME fd number, so the output must name that number.
        tap_pipe = ""
        pass_fds: tuple[int, ...] = ()
        if frame_tap_on:
            r_fd, w_fd = os.pipe()
            self._frame_tap_read_fd = r_fd
            self._frame_tap_write_fd = w_fd
            tap_pipe = f"pipe:{w_fd}"
            pass_fds = (w_fd,)

        cmd = build_ffmpeg_ingest_cmd(
            ffmpeg_path=self.runtime.ffmpeg_path,
            rtsp_url=self.upstream_url,
            rtsp_transport_in=rtsp_transport_in,
            record_enabled=record_enabled,
            chunk_seconds=chunk_seconds,
            out_pattern=out_pattern,
            container=container,
            audio=self.audio,
            mjpeg_enabled=mjpeg_enabled,
            mjpeg_fps=mjpeg_fps,
            mjpeg_scale_width=mjpeg_scale_width,
            rtsp_publish_enabled=rtsp_publish_enabled,
            rtsp_publish_url=self.rtsp_publish_url or "",
            rtsp_transport_out=rtsp_transport_out,
            frame_tap_enabled=frame_tap_on,
            frame_tap_fps=self.frame_tap_fps,
            frame_tap_scale_width=self.frame_tap_scale_width,
            frame_tap_pipe=tap_pipe,
            loglevel=self.runtime.ffmpeg_loglevel,
        )

        stdout = subprocess.PIPE if mjpeg_enabled else subprocess.DEVNULL
        self.proc = ManagedProcess(
            name=f"ingest:{self.camera_name}:{self.stream_name}",
            args=cmd,
            stdout=stdout,
            stderr_tail_lines=int(self.runtime.stderr_tail_lines),
            pass_fds=pass_fds,
        )
        try:
            self.proc.start()
        except BaseException:
            self._close_frame_tap_read_fd()
            raise
        finally:
            # ffmpeg holds its own copy now (or never will). The parent's copy must go, or the
            # reader never sees EOF when ffmpeg exits and leaks one fd per restart.
            self._close_frame_tap_write_fd()

        if mjpeg_enabled:
            self._start_mjpeg_reader()
        if frame_tap_on:
            self._start_frame_tap_reader()

    def _close_frame_tap_write_fd(self) -> None:
        """Close the parent's tap write end once; the field is cleared before the close."""
        fd, self._frame_tap_write_fd = self._frame_tap_write_fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _close_frame_tap_read_fd(self) -> None:
        """Close a tap read end no reader thread took over; the field is cleared first."""
        fd, self._frame_tap_read_fd = self._frame_tap_read_fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _start_mjpeg_reader(self) -> None:
        if self.proc is None or self.proc.popen is None:
            return
        if self.proc.popen.stdout is None:
            return
        if self.mjpeg_hub is None:
            return

        # Only one reader per process.
        if self._mjpeg_thread and self._mjpeg_thread.is_alive():
            return

        t = threading.Thread(
            target=self._mjpeg_reader_loop,
            name=f"mjpeg-reader:{self.camera_name}:{self.stream_name}",
            daemon=True,
        )
        self._mjpeg_thread = t
        t.start()

    def _mjpeg_reader_loop(self) -> None:
        """Parse concatenated JPEG frames from ffmpeg stdout and publish to FrameHub."""
        assert self.proc is not None
        assert self.proc.popen is not None
        assert self.proc.popen.stdout is not None
        assert self.mjpeg_hub is not None

        stdout = self.proc.popen.stdout
        buf = bytearray()
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        try:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)

                # Extract as many complete JPEGs as possible.
                while True:
                    s = buf.find(SOI)
                    if s < 0:
                        # Keep buffer from growing without bound in pathological cases.
                        if len(buf) > 2_000_000:
                            del buf[:-2]
                        break

                    if s > 0:
                        del buf[:s]

                    e = buf.find(EOI, 2)
                    if e < 0:
                        break

                    frame = bytes(buf[: e + 2])
                    del buf[: e + 2]

                    # Publish.
                    self.mjpeg_hub.update(frame)
        except Exception as e:  # pragma: no cover
            log.warning(f"[mjpeg_reader] {self.camera_name}/{self.stream_name} error: {e!r}")

    def _start_frame_tap_reader(self) -> None:
        """Hand the tap read end to a new reader thread, which owns and closes it."""
        dispatcher = self.frame_tap_dispatcher
        if dispatcher is None:
            self._close_frame_tap_read_fd()
            return
        r_fd, self._frame_tap_read_fd = self._frame_tap_read_fd, None
        if r_fd is None:
            return

        t = threading.Thread(
            target=self._frame_tap_reader_loop,
            args=(r_fd, dispatcher),
            name=f"frame-tap-reader:{self.camera_name}:{self.stream_name}",
            daemon=True,
        )
        self._frame_tap_thread = t
        t.start()

    def _frame_tap_reader_loop(self, r_fd: int, dispatcher: FrameTapDispatcher) -> None:
        """Read JPEG frames from the tap pipe and dispatch them; closes r_fd at EOF."""
        buf = bytearray()
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        try:
            while True:
                try:
                    chunk = os.read(r_fd, 4096)
                except OSError:
                    break  # pipe closed
                if not chunk:
                    break
                buf.extend(chunk)

                while True:
                    s = buf.find(SOI)
                    if s < 0:
                        if len(buf) > 2_000_000:
                            del buf[:-2]
                        break
                    if s > 0:
                        del buf[:s]
                    e = buf.find(EOI, 2)
                    if e < 0:
                        break
                    frame = bytes(buf[: e + 2])
                    del buf[: e + 2]

                    dispatcher.dispatch(
                        camera=self.camera_name,
                        stream=self.stream_name,
                        jpeg_bytes=frame,
                        ts_unix=_time.time(),
                    )
        except Exception as e:  # pragma: no cover
            log.warning(f"[frame_tap_reader] {self.camera_name}/{self.stream_name} error: {e!r}")
        finally:
            try:
                os.close(r_fd)
            except OSError:
                pass

    def stop(self) -> None:
        # Normally both are None by now: start() closed the write end and the reader thread
        # owns the read end. This covers a start() that never got that far.
        self._close_frame_tap_write_fd()
        self._close_frame_tap_read_fd()

        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

        if self._mjpeg_thread and self._mjpeg_thread.is_alive():
            # Reader thread will exit once stdout closes; join briefly.
            self._mjpeg_thread.join(timeout=1.0)
        self._mjpeg_thread = None

        # The tap reader sees EOF once ffmpeg has exited, closes its fd and returns.
        if self._frame_tap_thread and self._frame_tap_thread.is_alive():
            self._frame_tap_thread.join(timeout=1.0)
        self._frame_tap_thread = None


@dataclass
class CameraRecorder:
    """Camera-level coordinator.

    In v0.2.0 this owned two independent segment recorders.
    In unified ingest mode, it owns up to two StreamIngestors (main/sub), where each
    StreamIngestor is the single upstream puller for its stream and can feed record + proxy.

    When RecordConfig.mode == "event", the recorder starts a polling thread that
    watches for detector events and starts/stops ffmpeg accordingly.  In "continuous"
    mode (default), ffmpeg runs non-stop as before.
    """

    camera: CameraConfig
    runtime: RuntimeConfig
    proxy_hub: FrameHub | None = None
    # This camera's own dispatcher. The tap rides on the proxy stream's ingestor only.
    frame_tap_dispatcher: FrameTapDispatcher | None = None
    # True when something consumes the tap (a detector runner or a --frame-consumer): the
    # proxy stream is then ingested even when nothing records or proxies it.
    frame_tap_required: bool = False
    # Frame tap (fps, width). None = compute_tap_settings(camera): detect_fps at 320 px.
    tap_settings: tuple[float, int] | None = None

    main: StreamIngestor | None = None
    sub: StreamIngestor | None = None

    # Event-mode state
    _event_poll_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _event_poll_stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _event_recording: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build ingestors lazily so app.py can control startup ordering (e.g., start MediaMTX first).
        self.main = self._build_ingestor("main")
        self.sub = self._build_ingestor("sub")

    def _build_ingestor(self, stream_name: StreamName) -> StreamIngestor | None:
        cam = self.camera
        if stream_name == "sub" and cam.sub_url is None:
            return None

        # Record requirements.
        record_cfg: StreamRecordConfig | None = None
        record_output_dir: Path | None = None
        if cam.record.enabled:
            cfg = cam.record.main if stream_name == "main" else cam.record.sub
            if cfg.enabled:
                record_cfg = cfg
                record_output_dir = cam.record.output_dir

        # Proxy requirements.
        mjpeg_hub: FrameHub | None = None
        publish_url: str | None = None

        if cam.proxy.enabled and cam.proxy.stream == stream_name:
            if cam.proxy.mode == "mjpeg":
                # hub is provided by app.py (single owner), but we can fall back if needed.
                mjpeg_hub = self.proxy_hub or FrameHub()
                self.proxy_hub = mjpeg_hub
            elif cam.proxy.mode == "rtsp":
                # We always publish locally into MediaMTX.
                publish_url = f"rtsp://127.0.0.1:{cam.proxy.port}/{cam.proxy.path}"

        # The frame tap always comes from the stream that feeds the proxy (main when there is
        # no sub stream; the config validator guarantees that stream exists), never both.
        tap_here = self.frame_tap_dispatcher is not None and stream_name == cam.proxy.stream

        # If nothing needs this stream, do not create it.
        if (
            record_cfg is None
            and mjpeg_hub is None
            and publish_url is None
            and not (tap_here and self.frame_tap_required)
        ):
            return None

        upstream = cam.main_url if stream_name == "main" else cam.sub_url
        assert upstream is not None

        tap_fps, tap_width = (
            self.tap_settings if self.tap_settings is not None else compute_tap_settings(cam)
        )

        # Even if record is disabled, re-use rtsp_transport setting from record cfg defaults.
        # This avoids introducing a new config knob in the parallel phase.
        if record_cfg is None and cam.record is not None:
            record_cfg = cam.record.main if stream_name == "main" else cam.record.sub
            # But treat it as record-disabled by clearing enabled.
            record_cfg = StreamRecordConfig(
                enabled=False,
                container=record_cfg.container,
                chunk_seconds=record_cfg.chunk_seconds,
                rtsp_transport=record_cfg.rtsp_transport,
            )

        return StreamIngestor(
            camera_name=cam.name,
            stream_name=stream_name,
            upstream_url=upstream,
            runtime=self.runtime,
            record_cfg=record_cfg,
            record_output_dir=record_output_dir,
            proxy_cfg=cam.proxy if cam.proxy.enabled else None,
            mjpeg_hub=mjpeg_hub,
            rtsp_publish_url=publish_url,
            frame_tap_enabled=tap_here,
            frame_tap_fps=tap_fps,
            frame_tap_scale_width=tap_width,
            frame_tap_dispatcher=self.frame_tap_dispatcher if tap_here else None,
            audio=cam.record.audio,
        )

    def has_any(self) -> bool:
        return bool(self.main or self.sub)

    def start(self) -> None:
        """Start ingestors.

        In "continuous" mode (default), ffmpeg starts immediately.
        In "event" mode, a polling thread starts instead of ffmpeg; it will
        start/stop ffmpeg based on recent detector events.
        """
        if self.camera.record.mode == "event" and self.camera.record.enabled:
            self._start_event_poll()
        else:
            if self.main:
                self.main.start()
            if self.sub:
                self.sub.start()

    def stop(self) -> None:
        """Stop ingestors and any event-mode polling thread."""
        self._stop_event_poll()
        if self.main:
            self.main.stop()
        if self.sub:
            self.sub.stop()

    # ------------------------------------------------------------------
    # Event-mode recording
    # ------------------------------------------------------------------

    def _should_be_recording(self) -> bool:
        """Return True if an event has occurred within the post_seconds window.

        This is the testable decision point for event-mode recording.
        Queries the database for recent events for this camera.
        """
        from .db.schema import get_latest_event_for_camera

        cfg = self.camera.record.event_record
        event = get_latest_event_for_camera(
            camera_name=self.camera.name,
            since_seconds=cfg.post_seconds,
        )
        return event is not None

    def _start_event_poll(self) -> None:
        """Start the event-mode polling thread.

        The thread wakes every 1 second and checks whether ffmpeg should be
        running.  If a recent event exists within post_seconds, ffmpeg starts;
        otherwise ffmpeg stops.
        """
        if self._event_poll_thread is not None and self._event_poll_thread.is_alive():
            return  # already running
        self._event_poll_stop.clear()
        t = threading.Thread(
            target=self._event_poll_loop,
            name=f"event-poll:{self.camera.name}",
            daemon=True,
        )
        self._event_poll_thread = t
        t.start()
        log.info(f"[event-poll] Started event-mode polling for camera {self.camera.name}")

    def _stop_event_poll(self) -> None:
        """Signal the event-mode polling thread to stop and wait for it."""
        if self._event_poll_thread is None or not self._event_poll_thread.is_alive():
            return
        self._event_poll_stop.set()
        self._event_poll_thread.join(timeout=5.0)
        self._event_poll_thread = None
        log.info(f"[event-poll] Stopped event-mode polling for camera {self.camera.name}")

    def _event_poll_loop(self) -> None:
        """Polling loop: start/stop ffmpeg based on recent detector events.

        Design notes:
        * pre_seconds is aspirational — we cannot buffer video that was never
          captured.  The segment always starts at "now" (the moment the event
          is detected).  Document this clearly for users.
        * When ffmpeg is stopped, the segment is finalized automatically by
          the ManagedProcess teardown (same as a normal graceful stop).
        """
        while not self._event_poll_stop.is_set():
            try:
                should_record = self._should_be_recording()
                currently_recording = self._is_ingestor_running()

                if should_record and not currently_recording:
                    log.info(
                        f"[event-poll] Event detected — starting recording for {self.camera.name}"
                    )
                    self._start_ingestors()
                    self._event_recording = True
                elif not should_record and currently_recording:
                    log.info(
                        f"[event-poll] No recent events — stopping recording for {self.camera.name}"
                    )
                    self._stop_ingestors()
                    self._event_recording = False
            except Exception:
                log.exception(f"[event-poll] Error in poll loop for {self.camera.name}")
            self._event_poll_stop.wait(timeout=1.0)

    def _is_ingestor_running(self) -> bool:
        """Check if any ingestor (ffmpeg process) is currently running."""
        if self.main and self.main.is_running():
            return True
        if self.sub and self.sub.is_running():
            return True
        return False

    def _start_ingestors(self) -> None:
        """Start all configured ingestors (ffmpeg processes)."""
        if self.main:
            self.main.start()
        if self.sub:
            self.sub.start()

    def _stop_ingestors(self) -> None:
        """Stop all running ingestors (ffmpeg processes)."""
        if self.main:
            self.main.stop()
        if self.sub:
            self.sub.stop()

    def processes(self) -> list[StreamIngestor]:
        procs: list[StreamIngestor] = []
        if self.main:
            procs.append(self.main)
        if self.sub:
            procs.append(self.sub)
        return procs

    def mjpeg_hub(self) -> FrameHub | None:
        return self.proxy_hub
