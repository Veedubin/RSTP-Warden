"""Connection test for an RTSP URL: ffprobe for stream facts, one ffmpeg frame as a snapshot.

Used by the add-camera form (``web/routes/camera_edit.py``). Everything here blocks: up
to two subprocesses of ``timeout_s`` each, run one after the other. Call it from a plain
``def`` route (FastAPI runs those in its threadpool) or through ``asyncio.to_thread``,
never directly inside an ``async def``.

Subprocesses start through ``subprocess.run`` looked up at call time, so tests fake them
with ``unittest.mock.patch("subprocess.run", ...)``. Every error string in a
``ProbeResult`` has the URL's credentials removed: ffmpeg and ffprobe echo the full input
URL, password included, in their error lines.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .config import RuntimeConfig
from .ffmpeg import normalize_rtsp_url, redact_text

log = logging.getLogger(__name__)

SOCKET_TIMEOUT_US = "5000000"  # ffmpeg/ffprobe -timeout: RTSP connect and each socket read
SNAPSHOT_MAX_WIDTH = 640
MAX_PLAUSIBLE_FPS = 240.0
MAX_ERROR_CHARS = 300
NO_DIMENSIONS_ERROR = "stream has no video dimensions yet"
_SCHEMES = frozenset({"rtsp", "rtsps"})
_UNPARSABLE = (
    "The URL could not be parsed. Check the port, and percent-encode '/', '?', '#' "
    "and '@' in the username and password (for example %2F for '/')."
)


@dataclass(slots=True)
class ProbeResult:
    """Outcome of one connection test.

    ``ok`` False: the stream could not be probed; ``error`` says why and no snapshot was
    taken. ``ok`` True with ``error`` set: the stream was probed but the snapshot failed.
    ``error`` never contains credentials.
    """

    ok: bool
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    snapshot_jpeg: bytes | None = None
    error: str | None = None


def resolve_ffprobe_path(runtime: RuntimeConfig) -> str:
    """Return the ffprobe executable, resolved with ``shutil.which``.

    ``runtime.ffprobe_path`` when set; otherwise the file named ``ffprobe`` in the
    directory of ``runtime.ffmpeg_path``, or bare ``"ffprobe"`` (looked up on PATH) when
    ``ffmpeg_path`` has no directory part. Raises FileNotFoundError when it is missing.
    """
    if runtime.ffprobe_path:
        candidate = runtime.ffprobe_path
    else:
        ffmpeg_dir = os.path.dirname(runtime.ffmpeg_path)
        candidate = os.path.join(ffmpeg_dir, "ffprobe") if ffmpeg_dir else "ffprobe"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"ffprobe not found ({candidate}); install it next to ffmpeg "
            "or set runtime.ffprobe_path in config.yaml"
        )
    return resolved


def validate_rtsp_url(url: str) -> str:
    """Return ``url`` stripped, or raise ValueError with a message that is safe to show.

    Accepts only rtsp:// and rtsps:// URLs with a host. Rejects option-like input (a
    leading '-'), whitespace or control characters, a bad port, and credentials that
    break URL parsing (a raw '/', '?', '#' or '@' in the username or password). Messages
    never echo the URL, because it may contain a password.
    """
    text = (url or "").strip()
    if not text:
        raise ValueError("Enter an RTSP URL.")
    if text.startswith("-"):
        raise ValueError("The URL must not start with '-'.")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError("The URL must not contain spaces or control characters.")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        raise ValueError(_UNPARSABLE) from None
    if parts.scheme not in _SCHEMES:
        raise ValueError("Only rtsp:// and rtsps:// URLs can be tested.")
    if not parts.hostname:
        raise ValueError("The URL has no host.")
    if port == 0:
        raise ValueError(_UNPARSABLE)
    at_in_netloc = parts.netloc.count("@")
    if at_in_netloc > 1 or (at_in_netloc == 0 and "@" in text):
        raise ValueError(_UNPARSABLE)
    return text


def parse_frame_rate(rate: str | None) -> float | None:
    """Parse an ffprobe rate such as ``"15/1"`` or ``"30000/1001"``.

    Returns None for a missing, malformed, zero, negative or implausible (above
    MAX_PLAUSIBLE_FPS, e.g. the 90 kHz RTP clock ``"90000/1"``) rate.
    """
    if not rate:
        return None
    num_text, sep, den_text = str(rate).partition("/")
    try:
        num = float(num_text)
        den = float(den_text) if sep else 1.0
    except ValueError:
        return None
    if den == 0:
        return None
    fps = num / den
    if not math.isfinite(fps) or fps <= 0 or fps > MAX_PLAUSIBLE_FPS:
        return None
    return round(fps, 2)


def probe_stream(
    url: str,
    *,
    runtime: RuntimeConfig,
    timeout_s: float = 15.0,
    snapshot: bool = True,
) -> ProbeResult:
    """Probe ``url`` with ffprobe and, when that succeeds, grab one JPEG frame with ffmpeg.

    Never raises for bad input or a failing camera: every problem becomes
    ``ProbeResult(ok=False, error=...)``. Blocks for up to ``2 * timeout_s`` seconds.
    """
    try:
        target = normalize_rtsp_url(validate_rtsp_url(url))
    except ValueError as exc:
        return ProbeResult(ok=False, error=str(exc))
    try:
        ffprobe = resolve_ffprobe_path(runtime)
    except FileNotFoundError as exc:
        return ProbeResult(ok=False, error=str(exc))

    result = _probe(ffprobe, target, timeout_s)
    if result.ok and snapshot:
        result.snapshot_jpeg, result.error = _snapshot(runtime.ffmpeg_path, target, timeout_s)
    log.info(
        "[probe] %s: %s",
        _scrub(target, target),
        result.error or f"ok ({result.codec} {result.width}x{result.height})",
    )
    return result


def _probe_argv(ffprobe: str, url: str) -> list[str]:
    return [
        ffprobe,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        SOCKET_TIMEOUT_US,
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        url,
    ]


def _snapshot_argv(ffmpeg: str, url: str) -> list[str]:
    return [
        ffmpeg,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        SOCKET_TIMEOUT_US,
        "-i",
        url,
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({SNAPSHOT_MAX_WIDTH},iw)':-2",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "pipe:1",
    ]


def _run(argv: list[str], timeout_s: float) -> subprocess.CompletedProcess[bytes]:
    # subprocess.run is looked up on the module at call time so tests can patch it.
    return subprocess.run(
        argv,
        capture_output=True,
        timeout=timeout_s,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _probe(ffprobe: str, url: str, timeout_s: float) -> ProbeResult:
    try:
        proc = _run(_probe_argv(ffprobe, url), timeout_s)
    except subprocess.TimeoutExpired:
        return ProbeResult(ok=False, error=f"timed out after {timeout_s:g} s")
    except OSError as exc:
        return ProbeResult(ok=False, error=f"could not run ffprobe ({_os_reason(exc)})")
    if proc.returncode != 0:
        return ProbeResult(ok=False, error=_tool_error("ffprobe", proc, url))
    try:
        data = json.loads(proc.stdout or b"{}")
    except ValueError:
        return ProbeResult(ok=False, error="ffprobe returned output that is not JSON")
    video = _first_video_stream(data)
    if video is None:
        return ProbeResult(ok=False, error="no video stream found")
    codec_name = video.get("codec_name")
    codec = codec_name if isinstance(codec_name, str) and codec_name else None
    fps = parse_frame_rate(video.get("avg_frame_rate")) or parse_frame_rate(
        video.get("r_frame_rate")
    )
    width = _positive_int(video.get("width"))
    height = _positive_int(video.get("height"))
    if width is None or height is None:
        return ProbeResult(ok=False, codec=codec, fps=fps, error=NO_DIMENSIONS_ERROR)
    return ProbeResult(ok=True, codec=codec, width=width, height=height, fps=fps)


def _snapshot(ffmpeg: str, url: str, timeout_s: float) -> tuple[bytes | None, str | None]:
    try:
        proc = _run(_snapshot_argv(ffmpeg, url), timeout_s)
    except subprocess.TimeoutExpired:
        return None, f"snapshot timed out after {timeout_s:g} s"
    except OSError as exc:
        return None, f"snapshot failed: could not run ffmpeg ({_os_reason(exc)})"
    if proc.returncode != 0:
        return None, f"snapshot failed: {_tool_error('ffmpeg', proc, url)}"
    jpeg = proc.stdout or b""
    if not jpeg.startswith(b"\xff\xd8"):
        return None, "snapshot failed: ffmpeg returned no JPEG image"
    return jpeg, None


def _first_video_stream(data: Any) -> dict[str, Any] | None:
    streams = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type", "video") == "video":
            return stream
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _os_reason(exc: OSError) -> str:
    return exc.strerror or type(exc).__name__


def _tool_error(tool: str, proc: subprocess.CompletedProcess[bytes], url: str) -> str:
    """Last non-empty stderr line, credentials removed BEFORE truncation."""
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return f"{tool} exited with code {proc.returncode}"
    return _scrub(lines[-1], url)[:MAX_ERROR_CHARS]


def _scrub(text: str, url: str) -> str:
    """Remove ``url``'s own userinfo from ``text``, then apply ``ffmpeg.redact_text``.

    The literal replacement does not depend on redact_text's pattern, so the exact
    credentials this probe used are masked even if that pattern misses them.
    """
    netloc = urlsplit(url).netloc
    if "@" in netloc:
        userinfo = netloc.rsplit("@", 1)[0]
        if userinfo:
            text = text.replace(f"{userinfo}@", "***:***@")
    return redact_text(text)
