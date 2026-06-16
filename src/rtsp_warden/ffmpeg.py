from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

DEBUG_STDIO = os.getenv("RTSP_WARDEN_DEBUG_STDIO", "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_STDOUT = os.getenv("RTSP_WARDEN_DEBUG_STDOUT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# DEBUG_STDIO: mirror child stderr to parent terminal (disables stderr tail capture).
# DEBUG_STDOUT: also inherit child stdout when safe (disabled automatically for pipe:1 MJPEG).


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


# Escape hatch for broken RTSP sources:
# - Some cameras produce packets with missing/invalid timestamps and/or omit SPS/PPS until
#   the next IDR (keyframe). With "-c copy" remuxing, FFmpeg cannot fabricate missing
#   codec parameters or strict monotonic DTS required by MP4. In those cases, forcing a
#   decode+re-encode path is the only reliable fix.
RECORD_TRANSCODE = _env_bool("RTSP_WARDEN_RECORD_TRANSCODE", False)
RECORD_CODEC = os.getenv("RTSP_WARDEN_RECORD_CODEC", "libx264").strip() or "libx264"
RECORD_PRESET = os.getenv("RTSP_WARDEN_RECORD_PRESET", "veryfast").strip() or "veryfast"
RECORD_CRF = int(os.getenv("RTSP_WARDEN_RECORD_CRF", "23").strip() or "23")
RECORD_FPS = int(os.getenv("RTSP_WARDEN_RECORD_FPS", "0").strip() or "0")
RECORD_GOP = int(os.getenv("RTSP_WARDEN_RECORD_GOP", "0").strip() or "0")


def normalize_rtsp_url(url: str) -> str:
    """Normalize RTSP URLs for FFmpeg compatibility.

    Some cameras require an explicit empty password marker ("user:@host") to authenticate
    even when the password is blank. FFmpeg accepts both forms, but authentication behavior
    varies by camera. To avoid surprising failures, if a URL includes credentials with
    a username but no password delimiter ("user@host"), we rewrite it to "user:@host".

    This preserves any percent-encoding in the credential portion because we operate on
    the netloc rather than rebuilding from decoded components.
    """
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"rtsp", "rtsps"}:
            return url
        if "@" not in parts.netloc:
            return url
        creds, hostport = parts.netloc.rsplit("@", 1)
        if not creds:
            return url
        # If there is no ':' in the credential portion, there is no password delimiter.
        if ":" not in creds:
            netloc = f"{creds}:@{hostport}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return url
    except Exception:  # pylint: disable=broad-exception-caught
        return url


def redact_url(url: str) -> str:
    """Redact credentials from a URL for safe logging.

    Example:
      rtsp://user:pass@host/path -> rtsp://***:***@host/path
    """
    try:
        parts = urlsplit(url)
        if parts.username is None and parts.password is None:
            return url

        # Rebuild netloc preserving host/port, replacing credentials.
        host = parts.hostname or ""
        netloc = host
        if parts.port:
            netloc = f"{host}:{parts.port}"

        # Preserve explicit "user:pass@" marker if any credentials were present.
        if parts.username is not None and parts.password is None:
            netloc = f"***@{netloc}" if netloc else "***@"
        else:
            netloc = f"***:***@{netloc}" if netloc else "***:***@"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:  # pylint: disable=broad-exception-caught
        # Fall back to a minimal redaction in case of parsing edge cases.
        if "@" in url and "://" in url:
            try:
                scheme, rest = url.split("://", 1)
                before_at, after_at = rest.split("@", 1)
                if ":" in before_at:
                    return f"{scheme}://***:***@{after_at}"
                return f"{scheme}://***@{after_at}"
            except Exception:  # pylint: disable=broad-exception-caught
                return "***"
        return url


def redact_argv(argv: Iterable[str]) -> str:
    """Return a redacted, human-readable argv string suitable for logs."""
    out: list[str] = []
    for a in argv:
        if isinstance(a, str) and a.startswith("rtsp://"):
            out.append(redact_url(a))
        else:
            out.append(a)
    return " ".join(out)


def _sanitize_probe_values(analyzeduration_us: int, probesize: int) -> tuple[int, int]:
    """FFmpeg probing defaults.

    Some call sites set analyzeduration to 0 for low-latency startup, but that can prevent
    FFmpeg from reliably detecting stream parameters (common with RTSP cameras).
    Treat non-positive values as "use a sane default".
    """
    ad_min = 60_000_000
    ps_min = 100_000_000
    ad = analyzeduration_us if analyzeduration_us >= ad_min else ad_min
    ps = probesize if probesize >= ps_min else ps_min
    return ad, ps


def which_or_raise(exe: str) -> str:
    resolved = shutil.which(exe) or (exe if shutil.which(str(exe)) else None)
    if not resolved:
        raise FileNotFoundError(f"Required executable not found on PATH: {exe}")
    return resolved


@dataclass
class ExponentialBackoff:
    min_s: float = 1.0
    max_s: float = 60.0
    factor: float = 2.0
    _current: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._current = 0.0

    def next_delay(self) -> float:
        if self._current <= 0:
            self._current = self.min_s
        else:
            self._current = min(self.max_s, self._current * self.factor)
        return self._current


@dataclass
class ManagedProcess:
    """A small wrapper around subprocess with stderr tail capture.

    stdout can be:
      * None (inherit)
      * subprocess.DEVNULL
      * subprocess.PIPE (caller consumes)
    stderr defaults to subprocess.PIPE so we can capture tail lines for debugging.
    """

    name: str
    args: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None
    stdout: object = subprocess.DEVNULL
    stderr: object = subprocess.PIPE
    stderr_tail_lines: int = 200
    pass_fds: tuple[int, ...] = ()

    popen: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _stderr_thread: threading.Thread | None = field(default=None, init=False)
    _stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200), init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def start(self) -> None:
        if self.popen and self.popen.poll() is None:
            return

        # Refresh tail buffer maxlen if user configured it.
        self._stderr_tail = deque(self._stderr_tail, maxlen=self.stderr_tail_lines)

        # IMPORTANT: never leak credentials in RTSP URLs.
        log.info("[proc] starting %s: %s", self.name, redact_argv(self.args))
        # Optional debug: let child process stderr/stdout flow directly to our terminal.
        # This is intentionally noisy and should be used temporarily.
        inherit_stdout = False
        if DEBUG_STDOUT or DEBUG_STDIO:
            # Avoid dumping binary MJPEG (pipe:1) to terminal.
            if not any(a == "pipe:1" for a in self.args):
                inherit_stdout = True
        popen_stdout = None if inherit_stdout else self.stdout  # type: ignore[assignment]
        popen_stderr = None if DEBUG_STDIO else self.stderr  # type: ignore[assignment]

        self.popen = subprocess.Popen(
            self.args,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=popen_stdout,  # type: ignore[arg-type]
            stderr=popen_stderr,  # type: ignore[arg-type]
            pass_fds=self.pass_fds,
        )

        if self.popen.stderr is not None and self.stderr == subprocess.PIPE:
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader, name=f"{self.name}-stderr", daemon=True
            )
            self._stderr_thread.start()

    def _stderr_reader(self) -> None:
        assert self.popen is not None
        assert self.popen.stderr is not None
        try:
            for raw in iter(self.popen.stderr.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                with self._lock:
                    self._stderr_tail.append(line)
        except Exception as e:  # pragma: no cover  # pylint: disable=broad-exception-caught
            with self._lock:
                self._stderr_tail.append(f"[stderr_reader_error] {e!r}")

    def poll(self) -> int | None:
        if not self.popen:
            return None
        return self.popen.poll()

    def is_running(self) -> bool:
        return bool(self.popen and self.popen.poll() is None)

    def pid(self) -> int | None:
        return self.popen.pid if self.popen else None

    def stderr_tail(self) -> list[str]:
        with self._lock:
            return list(self._stderr_tail)

    def terminate(self, timeout_s: float = 5.0) -> None:
        if not self.popen:
            return
        if self.popen.poll() is not None:
            self.popen = None
            return

        # Prefer a graceful shutdown first so FFmpeg can flush trailers / finalize segments.
        log.info("[proc] terminating %s (pid=%s)", self.name, self.popen.pid)
        try:
            try:
                self.popen.send_signal(signal.SIGINT)
            except Exception:  # pragma: no cover
                self.popen.terminate()
            self.popen.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Escalate to SIGTERM, then SIGKILL if needed.
            log.warning("[proc] SIGTERM %s (timeout)", self.name)
            try:
                self.popen.terminate()
                self.popen.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                log.warning("[proc] killing %s (timeout)", self.name)
                self.popen.kill()
        finally:
            self.popen = None


def _audio_args(audio: bool) -> list[str]:
    """Return ffmpeg args for audio handling based on the audio opt-in flag.

    When *audio* is True the caller should replace the global ``-an`` with
    per-output mapping that includes the optional audio stream.  When False an
    empty list is returned (the caller keeps ``-an``).
    """
    if audio:
        return ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
    return []


def build_ffmpeg_segment_cmd(
    *,
    ffmpeg_path: str,
    rtsp_url: str,
    rtsp_transport: str,
    chunk_seconds: int,
    out_pattern: str,
    container: str,
    audio: bool = False,
    loglevel: str = "warning",
    analyzeduration_us: int = 30_000_000,
    probesize: int = 50_000_000,
) -> list[str]:
    """Build an ffmpeg command that remuxes RTSP into time-based segments."""
    rtsp_url = normalize_rtsp_url(rtsp_url)
    ad, ps = _sanitize_probe_values(analyzeduration_us, probesize)
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-analyzeduration",
        str(int(ad)),
        "-probesize",
        str(int(ps)),
        # Many RTSP cameras have missing/invalid RTP timestamps. Ask FFmpeg to synthesize
        # timestamps based on wallclock so segmenting does not rely on the source being sane.
        "-fflags",
        "+genpts+igndts",
        "-use_wallclock_as_timestamps",
        "1",
        "-rtsp_transport",
        rtsp_transport,
        "-i",
        rtsp_url,
    ]
    if not audio:
        cmd.append("-an")

    if RECORD_TRANSCODE:
        # Decode+re-encode path: required for some cameras that never present usable
        # SPS/PPS (dimensions) or have non-monotonic/missing DTS.
        if RECORD_FPS > 0:
            cmd += ["-fps_mode", "cfr", "-r", str(int(RECORD_FPS))]
        gop = RECORD_GOP if RECORD_GOP > 0 else (RECORD_FPS * 2 if RECORD_FPS > 0 else 60)
        cmd += [
            "-c:v",
            RECORD_CODEC,
            "-preset",
            RECORD_PRESET,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(int(gop)),
        ]
        # libx264 uses CRF; other encoders may ignore it harmlessly.
        cmd += ["-crf", str(int(RECORD_CRF))]
    else:
        cmd += ["-c:v", "copy"]

    cmd += _audio_args(audio)

    cmd += [
        "-avoid_negative_ts",
        "make_zero",
        "-f",
        "segment",
        "-segment_time",
        str(int(chunk_seconds)),
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
    ]

    # segment muxer sometimes benefits from explicit segment_format for mp4
    if container == "mp4":
        cmd += ["-segment_format", "mp4"]
    elif container == "ts":
        cmd += ["-segment_format", "mpegts"]
    # mkv uses default Matroska format, no explicit -segment_format needed

    cmd.append(out_pattern)
    return cmd


def build_ffmpeg_mjpeg_stdout_cmd(
    *,
    ffmpeg_path: str,
    rtsp_url: str,
    rtsp_transport: str,
    fps: int,
    scale_width: int,
    loglevel: str = "warning",
    analyzeduration_us: int = 30_000_000,
    probesize: int = 50_000_000,
) -> list[str]:
    """Build an ffmpeg command that outputs a MJPEG stream to stdout."""
    rtsp_url = normalize_rtsp_url(rtsp_url)
    ad, ps = _sanitize_probe_values(analyzeduration_us, probesize)

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-analyzeduration",
        str(int(ad)),
        "-probesize",
        str(int(ps)),
        "-fflags",
        "+genpts+igndts",
        "-use_wallclock_as_timestamps",
        "1",
        "-rtsp_transport",
        rtsp_transport,
        "-i",
        rtsp_url,
        "-an",
    ]
    if scale_width and scale_width > 0:
        cmd += ["-vf", f"scale={scale_width}:-1"]

    cmd += [
        "-fps_mode",
        "cfr",
        "-r",
        str(int(fps)),
        "-f",
        "mjpeg",
        "pipe:1",
    ]
    return cmd


def build_ffmpeg_ingest_cmd(
    *,
    ffmpeg_path: str,
    rtsp_url: str,
    rtsp_transport_in: str,
    # Record/segment output (optional)
    record_enabled: bool,
    chunk_seconds: int = 300,
    out_pattern: str = "",
    container: str = "mkv",
    audio: bool = False,
    # MJPEG stdout output (optional)
    mjpeg_enabled: bool = False,
    mjpeg_fps: int = 7,
    mjpeg_scale_width: int = 0,
    # RTSP publish output (optional)
    rtsp_publish_enabled: bool = False,
    rtsp_publish_url: str = "",
    rtsp_transport_out: str = "tcp",
    # Frame tap output (optional) — dedicated low-res MJPEG stream for CV consumers
    frame_tap_enabled: bool = False,
    frame_tap_fps: int = 5,
    frame_tap_scale_width: int = 320,
    frame_tap_pipe: str = "pipe:3",
    loglevel: str = "warning",
    analyzeduration_us: int = 30_000_000,
    probesize: int = 50_000_000,
) -> list[str]:
    """Build a single-FFmpeg multi-output ingest command.

    This is the core of ingest unification: one upstream pull (RTSP input) can feed:
      * segment recording (copy/remux)
      * MJPEG frames to stdout (decode+encode)
      * RTSP publish (copy) to a local RTSP server (e.g., MediaMTX)

    Notes:
      * Output options are applied per-output (ordering matters).
      * When mjpeg_enabled=True, the caller should run the process with stdout=PIPE.
    """
    rtsp_url = normalize_rtsp_url(rtsp_url)
    ad, ps = _sanitize_probe_values(analyzeduration_us, probesize)

    if not (record_enabled or mjpeg_enabled or rtsp_publish_enabled or frame_tap_enabled):
        raise ValueError("At least one output must be enabled")

    if frame_tap_enabled:
        if frame_tap_fps <= 0:
            raise ValueError("frame_tap_fps must be > 0")
        if frame_tap_scale_width < 0:
            raise ValueError("frame_tap_scale_width must be >= 0")

    if record_enabled:
        if not out_pattern:
            raise ValueError("out_pattern is required when record_enabled")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")

    if rtsp_publish_enabled and not rtsp_publish_url:
        raise ValueError("rtsp_publish_url is required when rtsp_publish_enabled")

    cmd: list[str] = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-analyzeduration",
        str(int(ad)),
        "-probesize",
        str(int(ps)),
        "-fflags",
        "+genpts+igndts",
        "-use_wallclock_as_timestamps",
        "1",
        # input options
        "-rtsp_transport",
        rtsp_transport_in,
        "-i",
        rtsp_url,
    ]
    if not audio:
        cmd.append("-an")

    # 1) record segments
    if record_enabled:
        cmd += ["-map", "0:v:0"]

        if RECORD_TRANSCODE:
            # Decode+re-encode path (escape hatch) for cameras with missing SPS/PPS or broken DTS.
            if RECORD_FPS > 0:
                cmd += ["-fps_mode", "cfr", "-r", str(int(RECORD_FPS))]
            gop = RECORD_GOP if RECORD_GOP > 0 else (RECORD_FPS * 2 if RECORD_FPS > 0 else 60)
            cmd += [
                "-c:v",
                RECORD_CODEC,
                "-preset",
                RECORD_PRESET,
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(int(gop)),
                "-crf",
                str(int(RECORD_CRF)),
            ]
        else:
            cmd += ["-c:v", "copy"]

        cmd += _audio_args(audio)

        cmd += [
            "-avoid_negative_ts",
            "make_zero",
            "-f",
            "segment",
            "-segment_time",
            str(int(chunk_seconds)),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
        ]
        if container == "mp4":
            cmd += ["-segment_format", "mp4"]
        elif container == "ts":
            cmd += ["-segment_format", "mpegts"]
        # mkv uses default Matroska format, no explicit -segment_format needed
        cmd.append(out_pattern)

    # 2) RTSP publish (copy)
    if rtsp_publish_enabled:
        cmd += [
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-f",
            "rtsp",
            "-rtsp_transport",
            rtsp_transport_out,
        ]
        cmd.append(rtsp_publish_url)

    # 3) MJPEG to stdout (decode + encode)
    if mjpeg_enabled:
        cmd += [
            "-map",
            "0:v:0",
        ]
        if mjpeg_scale_width and mjpeg_scale_width > 0:
            cmd += ["-vf", f"scale={int(mjpeg_scale_width)}:-1"]
        cmd += [
            "-r",
            str(int(mjpeg_fps)),
            "-c:v",
            "mjpeg",
            "-f",
            "mjpeg",
            "pipe:1",
        ]

    # 4) Frame tap MJPEG to dedicated pipe (low-res, dedicated for CV consumers)
    if frame_tap_enabled:
        cmd += ["-map", "0:v:0"]
        if frame_tap_scale_width and frame_tap_scale_width > 0:
            cmd += ["-vf", f"scale={int(frame_tap_scale_width)}:-1"]
        cmd += [
            "-r",
            str(int(frame_tap_fps)),
            "-c:v",
            "mjpeg",
            "-f",
            "mjpeg",
            frame_tap_pipe,
        ]

    return cmd
