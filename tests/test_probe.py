"""Tests for rtsp_warden.probe, the add-camera "Test connection" service.

Offline: every ffprobe/ffmpeg call is faked by patching subprocess.run, and
shutil.which is monkeypatched. URLs use the RFC 5737 host 192.0.2.10 and fake
credentials only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from rtsp_warden import probe
from rtsp_warden.config import AppConfig, RuntimeConfig


def _which_everything(name: str) -> str:
    """Fake shutil.which: every binary exists; bare names resolve under /usr/bin."""
    return name if "/" in name else f"/usr/bin/{name}"


@pytest.fixture
def fake_which(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch shutil.which; returns the list of names it was asked for."""
    asked: list[str] = []

    def which(name: str) -> str:
        asked.append(name)
        return _which_everything(name)

    monkeypatch.setattr(shutil, "which", which)
    return asked


# --- RuntimeConfig.ffprobe_path ---------------------------------------------


def test_runtime_config_ffprobe_path_defaults_to_none() -> None:
    assert RuntimeConfig().ffprobe_path is None


def test_runtime_config_accepts_ffprobe_path() -> None:
    cfg = AppConfig.model_validate(
        {"cameras": [], "runtime": {"ffmpeg_path": "ffmpeg", "ffprobe_path": "/opt/ff/ffprobe"}}
    )
    assert cfg.runtime.ffprobe_path == "/opt/ff/ffprobe"


# --- resolve_ffprobe_path ---------------------------------------------------


@pytest.mark.parametrize(
    ("ffmpeg_path", "expected_lookup"),
    [
        ("ffmpeg", "ffprobe"),
        ("/usr/bin/ffmpeg", "/usr/bin/ffprobe"),
        ("/usr/lib/jellyfin-ffmpeg/ffmpeg", "/usr/lib/jellyfin-ffmpeg/ffprobe"),
        ("./bin/ffmpeg", "./bin/ffprobe"),
    ],
)
def test_resolve_ffprobe_path_uses_sibling_of_ffmpeg(
    fake_which: list[str], ffmpeg_path: str, expected_lookup: str
) -> None:
    resolved = probe.resolve_ffprobe_path(RuntimeConfig(ffmpeg_path=ffmpeg_path))
    assert fake_which == [expected_lookup]
    assert resolved == _which_everything(expected_lookup)


def test_resolve_ffprobe_path_prefers_explicit_setting(fake_which: list[str]) -> None:
    runtime = RuntimeConfig(ffmpeg_path="/usr/bin/ffmpeg", ffprobe_path="/opt/ff/ffprobe")
    assert probe.resolve_ffprobe_path(runtime) == "/opt/ff/ffprobe"
    assert fake_which == ["/opt/ff/ffprobe"]


def test_resolve_ffprobe_path_treats_empty_setting_as_unset(fake_which: list[str]) -> None:
    runtime = RuntimeConfig(ffmpeg_path="/usr/bin/ffmpeg", ffprobe_path="")
    assert probe.resolve_ffprobe_path(runtime) == "/usr/bin/ffprobe"
    assert fake_which == ["/usr/bin/ffprobe"]


def test_resolve_ffprobe_path_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError) as excinfo:
        probe.resolve_ffprobe_path(RuntimeConfig())
    assert "ffprobe not found (ffprobe)" in str(excinfo.value)
    assert "runtime.ffprobe_path" in str(excinfo.value)


# --- validate_rtsp_url ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://u:p@192.0.2.10:554/videoMain",
        "rtsps://u:p@192.0.2.10:322/stream1",
        "rtsp://192.0.2.10/videoMain",
        "RTSP://192.0.2.10/videoMain",
        "rtsp://u:p@[2001:db8::1]:554/videoMain",
        "rtsp://u:p%2Fw%23x%40y%25@192.0.2.10:554/videoMain",
        "rtsp://u:p@192.0.2.10/cam@1",
    ],
)
def test_validate_rtsp_url_accepts_and_strips(url: str) -> None:
    assert probe.validate_rtsp_url(f"  {url}\n") == url


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "Enter an RTSP URL."),
        ("   ", "Enter an RTSP URL."),
        ("-i", "must not start with '-'"),
        ("-rtsp://192.0.2.10/videoMain", "must not start with '-'"),
        ("http://192.0.2.10/videoMain", "Only rtsp:// and rtsps:// URLs can be tested."),
        ("file:/etc/passwd", "Only rtsp:// and rtsps:// URLs can be tested."),
        ("file:///etc/passwd", "Only rtsp:// and rtsps:// URLs can be tested."),
        ("not-a-url", "Only rtsp:// and rtsps:// URLs can be tested."),
        ("rtsp:///videoMain", "The URL has no host."),
        ("rtsp://192.0.2.10/video Main", "spaces or control characters"),
        ("rtsp://192.0.2.10/x\ny", "spaces or control characters"),
        ("rtsp://192.0.2.10:99999/videoMain", "could not be parsed"),
        ("rtsp://192.0.2.10:0/videoMain", "could not be parsed"),
        ("rtsp://[2001:db8::1/videoMain", "could not be parsed"),
        ("rtsp://u:p/w@192.0.2.10/videoMain", "could not be parsed"),
        ("rtsp://u:p#w@192.0.2.10/videoMain", "could not be parsed"),
        ("rtsp://u:p@w@192.0.2.10/videoMain", "could not be parsed"),
        ("rtsp://us/er:pw@192.0.2.10/videoMain", "could not be parsed"),
    ],
)
def test_validate_rtsp_url_rejects(url: str, message: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        probe.validate_rtsp_url(url)
    assert message in str(excinfo.value)


def test_validate_rtsp_url_messages_never_echo_credentials() -> None:
    for url in ("rtsp://u:pw9x/z@192.0.2.10/m", "http://u:pw9x@192.0.2.10/m"):
        with pytest.raises(ValueError) as excinfo:
            probe.validate_rtsp_url(url)
        assert "pw9x" not in str(excinfo.value)


# --- parse_frame_rate and ProbeResult ---------------------------------------


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("15/1", 15.0),
        ("30000/1001", 29.97),
        ("25", 25.0),
        ("0/0", None),
        ("15/0", None),
        ("-5/1", None),
        ("90000/1", None),
        ("nan/1", None),
        ("abc", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_frame_rate(rate: str | None, expected: float | None) -> None:
    assert probe.parse_frame_rate(rate) == expected


def test_probe_result_optional_fields_default_to_none() -> None:
    result = probe.ProbeResult(ok=False, error="boom")
    assert result.codec is None
    assert result.width is None
    assert result.height is None
    assert result.fps is None
    assert result.snapshot_jpeg is None
    assert result.error == "boom"


# --- probe_stream -----------------------------------------------------------

URL = "rtsp://u:p@192.0.2.10:554/videoMain"
REDACTED_URL = "rtsp://***:***@192.0.2.10:554/videoMain"
JPEG = b"\xff\xd8\xff\xd9"
H264_720P_JSON = (
    b'{"streams":[{"index":0,"codec_name":"h264","codec_type":"video",'
    b'"width":1280,"height":720,"r_frame_rate":"15/1","avg_frame_rate":"15/1"}]}'
)
# What ffprobe printed (rc 0) when the keyframe had not arrived within analyzeduration.
NO_DIMS_JSON = (
    b'{"streams":[{"index":0,"codec_name":"h264","codec_type":"video",'
    b'"width":0,"height":0,"r_frame_rate":"90000/1","avg_frame_rate":"15/1"}]}'
)
# What ffprobe printed (rc 1) on a failed RTSP DESCRIBE.
FAILED_PROBE_STDOUT = b"{\n\n}\n"
UNAUTHORIZED_STDERR = (
    b"[rtsp @ 0x55d0c0] method DESCRIBE failed: 401 Unauthorized\n"
    b"rtsp://u:p@192.0.2.10:554/videoMain: "
    b"Server returned 401 Unauthorized (authorization failed)\n"
)

Handler = Callable[[list[str]], subprocess.CompletedProcess[bytes]]


def _done(
    cmd: list[str], returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _raises(exc: BaseException) -> Handler:
    def handler(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
        raise exc

    return handler


class FakeRun:
    """Stands in for subprocess.run: answers ffprobe and ffmpeg separately, records calls."""

    def __init__(self, *, on_probe: Handler | None = None, on_snap: Handler | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.on_probe = on_probe or (lambda cmd: _done(cmd, stdout=H264_720P_JSON))
        self.on_snap = on_snap or (lambda cmd: _done(cmd, stdout=JPEG))

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(cmd), kwargs))
        if "ffprobe" in os.path.basename(cmd[0]):
            return self.on_probe(cmd)
        return self.on_snap(cmd)


def _run_probe(fake: FakeRun, url: str = URL, **kwargs: Any) -> probe.ProbeResult:
    runtime = kwargs.pop("runtime", RuntimeConfig())
    with patch("subprocess.run", side_effect=fake):
        return probe.probe_stream(url, runtime=runtime, **kwargs)


def test_probe_stream_reports_codec_resolution_fps_and_snapshot(fake_which: list[str]) -> None:
    fake = FakeRun()
    result = _run_probe(fake)
    assert result == probe.ProbeResult(
        ok=True, codec="h264", width=1280, height=720, fps=15.0, snapshot_jpeg=JPEG, error=None
    )
    assert len(fake.calls) == 2


def test_probe_stream_argv_and_subprocess_options(fake_which: list[str]) -> None:
    fake = FakeRun()
    _run_probe(fake, url=f"  {URL} ")
    (probe_cmd, probe_kwargs), (snap_cmd, snap_kwargs) = fake.calls
    assert probe_cmd == [
        "/usr/bin/ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        URL,
    ]
    assert snap_cmd == [
        "ffmpeg",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        "5000000",
        "-i",
        URL,
        "-frames:v",
        "1",
        "-vf",
        "scale='min(640,iw)':-2",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "pipe:1",
    ]
    expected_kwargs = {
        "capture_output": True,
        "timeout": 15.0,
        "stdin": subprocess.DEVNULL,
        "check": False,
    }
    assert probe_kwargs == expected_kwargs
    assert snap_kwargs == expected_kwargs


def test_probe_stream_uses_binaries_next_to_custom_ffmpeg(fake_which: list[str]) -> None:
    fake = FakeRun()
    runtime = RuntimeConfig(ffmpeg_path="/usr/lib/jellyfin-ffmpeg/ffmpeg")
    _run_probe(fake, runtime=runtime)
    assert fake.calls[0][0][0] == "/usr/lib/jellyfin-ffmpeg/ffprobe"
    assert fake.calls[1][0][0] == "/usr/lib/jellyfin-ffmpeg/ffmpeg"


def test_probe_stream_normalizes_username_only_url(fake_which: list[str]) -> None:
    fake = FakeRun()
    _run_probe(fake, url="rtsp://u@192.0.2.10/videoMain")
    assert fake.calls[0][0][-1] == "rtsp://u:@192.0.2.10/videoMain"
    assert fake.calls[1][0][fake.calls[1][0].index("-i") + 1] == "rtsp://u:@192.0.2.10/videoMain"


def test_probe_stream_custom_timeout_reaches_both_calls(fake_which: list[str]) -> None:
    fake = FakeRun()
    _run_probe(fake, timeout_s=4.5)
    assert [kwargs["timeout"] for _cmd, kwargs in fake.calls] == [4.5, 4.5]


def test_probe_stream_without_snapshot_runs_only_ffprobe(fake_which: list[str]) -> None:
    fake = FakeRun()
    result = _run_probe(fake, snapshot=False)
    assert result.ok is True
    assert result.snapshot_jpeg is None
    assert len(fake.calls) == 1


def test_probe_stream_failure_reports_last_stderr_line_redacted(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 1, FAILED_PROBE_STDOUT, UNAUTHORIZED_STDERR))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(
        ok=False,
        error=f"{REDACTED_URL}: Server returned 401 Unauthorized (authorization failed)",
    )
    assert len(fake.calls) == 1  # no snapshot after a failed probe


def test_probe_stream_failure_without_stderr_reports_exit_code(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 1, FAILED_PROBE_STDOUT, b"\n  \n"))
    result = _run_probe(fake)
    assert result.ok is False
    assert result.error == "ffprobe exited with code 1"


def test_probe_stream_error_is_redacted_before_truncation(fake_which: list[str]) -> None:
    line = "x" * 285 + "rtsp://u:pw9xLONGSECRET@192.0.2.10/videoMain: Connection timed out"
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 1, FAILED_PROBE_STDOUT, line.encode()))
    result = _run_probe(fake)
    assert result.error is not None
    assert len(result.error) == 300
    assert "pw9x" not in result.error
    assert result.error.endswith("rtsp://***:***@")


def test_probe_stream_timeout(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=_raises(subprocess.TimeoutExpired(cmd="ffprobe", timeout=15.0)))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(ok=False, error="timed out after 15 s")
    assert len(fake.calls) == 1


def test_probe_stream_ffprobe_cannot_start(fake_which: list[str]) -> None:
    missing = FileNotFoundError(2, "No such file or directory", "/usr/bin/ffprobe")
    fake = FakeRun(on_probe=_raises(missing))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(
        ok=False, error="could not run ffprobe (No such file or directory)"
    )


def test_probe_stream_missing_ffprobe_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with patch("subprocess.run") as run:
        result = probe.probe_stream(URL, runtime=RuntimeConfig())
    run.assert_not_called()
    assert result.ok is False
    assert result.error is not None
    assert result.error.startswith("ffprobe not found (ffprobe)")


@pytest.mark.parametrize("url", ["http://192.0.2.10/videoMain", "-i", "file:/etc/passwd"])
def test_probe_stream_invalid_url_runs_nothing(fake_which: list[str], url: str) -> None:
    with patch("subprocess.run") as run:
        result = probe.probe_stream(url, runtime=RuntimeConfig())
    run.assert_not_called()
    assert fake_which == []  # validated before ffprobe is even looked up
    assert result.ok is False
    assert result.error in {
        "Only rtsp:// and rtsps:// URLs can be tested.",
        "The URL must not start with '-'.",
    }


def test_probe_stream_unreadable_json(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 0, b"not json"))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(ok=False, error="ffprobe returned output that is not JSON")


def test_probe_stream_no_video_stream(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 0, b'{"streams": []}'))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(ok=False, error="no video stream found")


def test_probe_stream_zero_dimensions_is_not_ok(fake_which: list[str]) -> None:
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 0, NO_DIMS_JSON))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(
        ok=False, codec="h264", fps=15.0, error="stream has no video dimensions yet"
    )
    assert probe.NO_DIMENSIONS_ERROR == "stream has no video dimensions yet"
    assert len(fake.calls) == 1  # no snapshot


def test_probe_stream_fps_falls_back_to_r_frame_rate(fake_which: list[str]) -> None:
    stdout = (
        b'{"streams":[{"codec_name":"hevc","codec_type":"video","width":2560,'
        b'"height":1440,"r_frame_rate":"25/1","avg_frame_rate":"0/0"}]}'
    )
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 0, stdout))
    result = _run_probe(fake, snapshot=False)
    assert (result.ok, result.codec, result.width, result.height, result.fps) == (
        True,
        "hevc",
        2560,
        1440,
        25.0,
    )


def test_probe_stream_snapshot_timeout_keeps_probe_facts(fake_which: list[str]) -> None:
    fake = FakeRun(on_snap=_raises(subprocess.TimeoutExpired(cmd="ffmpeg", timeout=15.0)))
    result = _run_probe(fake)
    assert result == probe.ProbeResult(
        ok=True,
        codec="h264",
        width=1280,
        height=720,
        fps=15.0,
        snapshot_jpeg=None,
        error="snapshot timed out after 15 s",
    )


def test_probe_stream_snapshot_failure_is_redacted(fake_which: list[str]) -> None:
    stderr = f"Error opening input file {URL}.\n".encode()
    fake = FakeRun(on_snap=lambda cmd: _done(cmd, 8, b"", stderr))
    result = _run_probe(fake)
    assert result.ok is True
    assert result.snapshot_jpeg is None
    assert result.error == f"snapshot failed: Error opening input file {REDACTED_URL}."


def test_probe_stream_snapshot_not_a_jpeg(fake_which: list[str]) -> None:
    fake = FakeRun(on_snap=lambda cmd: _done(cmd, 0, b""))
    result = _run_probe(fake)
    assert result.ok is True
    assert result.error == "snapshot failed: ffmpeg returned no JPEG image"


def test_probe_stream_snapshot_ffmpeg_cannot_start(fake_which: list[str]) -> None:
    missing = FileNotFoundError(2, "No such file or directory", "ffmpeg")
    fake = FakeRun(on_snap=_raises(missing))
    result = _run_probe(fake)
    assert result.ok is True
    assert result.error == "snapshot failed: could not run ffmpeg (No such file or directory)"


# --- credentials never leave the probe (review focus) -----------------------

# Password "p/w#x@y%" percent-encoded the way the add-camera form stores it (quote(safe="")).
SPECIAL_URL = "rtsp://admin:p%2Fw%23x%40y%25@192.0.2.10:554/videoMain"


def test_percent_encoded_special_password_never_reaches_error(fake_which: list[str]) -> None:
    assert probe.validate_rtsp_url(SPECIAL_URL) == SPECIAL_URL  # the URL still parses
    stderr = f"{SPECIAL_URL}: Server returned 401 Unauthorized (authorization failed)\n"
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 1, FAILED_PROBE_STDOUT, stderr.encode()))
    result = _run_probe(fake, url=SPECIAL_URL)
    assert fake.calls[0][0][-1] == SPECIAL_URL  # passed on still encoded; ffmpeg decodes it
    assert result.error == (
        f"{REDACTED_URL}: Server returned 401 Unauthorized (authorization failed)"
    )
    for secret in ("admin:", "p%2Fw%23x%40y%25", "p/w#x@y%"):
        assert secret not in result.error


def test_own_credentials_masked_even_if_redact_text_misses_them(
    fake_which: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "redact_text", lambda text: text)
    stderr = f"Error opening input file {SPECIAL_URL}.\n".encode()
    fake = FakeRun(on_snap=lambda cmd: _done(cmd, 8, b"", stderr))
    result = _run_probe(fake, url=SPECIAL_URL)
    assert result.ok is True
    assert result.error == f"snapshot failed: Error opening input file {REDACTED_URL}."


def test_probe_logs_one_line_without_credentials(
    fake_which: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(probe.log, "info", lambda msg, *args: messages.append(msg % args))
    stderr = f"{SPECIAL_URL}: Server returned 401 Unauthorized (authorization failed)\n"
    fake = FakeRun(on_probe=lambda cmd: _done(cmd, 1, FAILED_PROBE_STDOUT, stderr.encode()))
    _run_probe(fake, url=SPECIAL_URL)
    assert len(messages) == 1
    assert messages[0].startswith(f"[probe] {REDACTED_URL}: ")
    assert "401 Unauthorized" in messages[0]
    assert "p%2Fw" not in messages[0]
    assert "admin:" not in messages[0]
