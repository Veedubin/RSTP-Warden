from types import SimpleNamespace

import pytest

from rtsp_warden.app import _last_stderr_line
from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.ffmpeg import redact_text
from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.services.cameras import list_cameras, live_status, status_label


class _Proc:
    def __init__(self, running: bool, tail=None):
        self._running = running
        self._tail = tail or []

    def poll(self):
        return None if self._running else 1

    def is_running(self):
        return self._running

    def stderr_tail(self):
        return self._tail


def _rt(procs, next_restart_at=0.0, last_error="", hub=None):
    recorder = SimpleNamespace(processes=lambda: [SimpleNamespace(proc=p) for p in procs])
    return SimpleNamespace(
        recorder=recorder, next_restart_at=next_restart_at, last_error=last_error, hub=hub
    )


def test_running():
    assert live_status(_rt([_Proc(True)]), now=100.0)["status"] == "running"


def test_idle_when_no_processes():
    assert live_status(_rt([]), now=100.0)["status"] == "idle"


def test_restarting_with_countdown_and_error():
    st = live_status(_rt([_Proc(False)], next_restart_at=107.4, last_error="boom"), now=100.0)
    assert st["status"] == "restarting"
    assert st["restart_in"] == 7
    assert st["last_error"] == "boom"


def test_failed_when_dead_and_no_restart_scheduled():
    assert live_status(_rt([_Proc(False)]), now=100.0)["status"] == "failed"


def test_last_frame_age_from_hub():
    hub = FrameHub()
    hub.update(b"\xff\xd8\xff\xd9")
    st = live_status(_rt([_Proc(True)], hub=hub))
    assert st["last_frame_age"] is not None and st["last_frame_age"] < 5


def test_mark_healthy_clears_restart_state():
    from rtsp_warden.app import CameraRuntime

    rt = CameraRuntime.__new__(CameraRuntime)
    rt.next_restart_at = 123.0
    rt.last_error = "boom"
    rt.mark_healthy()
    assert rt.next_restart_at == 0.0
    assert rt.last_error == ""


def _event_rt(recording: bool, procs):
    recorder = SimpleNamespace(
        processes=lambda: [SimpleNamespace(proc=p) for p in procs],
        camera=SimpleNamespace(record=SimpleNamespace(mode="event")),
        _event_recording=recording,
    )
    return SimpleNamespace(recorder=recorder, next_restart_at=0.0, last_error="", hub=None)


def test_event_mode_waiting_is_not_failed():
    assert live_status(_event_rt(False, [None]), now=100.0)["status"] == "waiting"


def test_event_mode_recording_is_running():
    assert live_status(_event_rt(True, [_Proc(True)]), now=100.0)["status"] == "running"


def _cfg_and_cam_rt():
    from rtsp_warden.config import AppConfig, CameraConfig

    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    cam_rt = SimpleNamespace(
        camera=cam,
        recorder=SimpleNamespace(processes=lambda: [SimpleNamespace(proc=_Proc(True))]),
        next_restart_at=0.0,
        last_error="",
        hub=None,
    )
    return AppConfig(cameras=[cam]), cam_rt


def test_list_cameras_without_runtime_has_no_detection_badge():
    from rtsp_warden.web.services.cameras import list_cameras

    cfg, _cam_rt = _cfg_and_cam_rt()
    assert list_cameras(cfg)[0]["detection"] is None


def test_list_cameras_runtime_without_detection_status_has_no_badge():
    """Older fakes (and runtimes without detection) lack detection_status: no badge, no error."""
    from rtsp_warden.web.services.cameras import list_cameras

    cfg, cam_rt = _cfg_and_cam_rt()
    row = list_cameras(cfg, SimpleNamespace(cameras=[cam_rt]))[0]
    assert row["status"] == "running"
    assert row["detection"] is None


def test_list_cameras_carries_the_detection_badge():
    from rtsp_warden.web.services.cameras import list_cameras

    cfg, cam_rt = _cfg_and_cam_rt()
    raw = {
        "frames_processed": 12,
        "frames_dropped": 3,
        "errors_total": 0,
        "detectors": [
            {
                "index": 0,
                "type": "onnx",
                "provider": "CUDAExecutionProvider",
                "fps": 2.0,
                "processed": 12,
                "skipped": 18,
                "errors": 0,
            }
        ],
    }
    rt = SimpleNamespace(cameras=[cam_rt], detection_status=lambda name: raw)
    badge = list_cameras(cfg, rt)[0]["detection"]
    assert badge["text"] == "GPU"
    assert badge["level"] == "ok"
    assert badge["dropped"] == 3


# --- redact_text: credentials never reach the UI or logs (review focus 1) ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Connection to rtsp://u:p@h/m failed: 401 Unauthorized",
            "Connection to rtsp://***:***@h/m failed: 401 Unauthorized",
        ),
        ("rtsp://u@h/m: 401", "rtsp://***:***@h/m: 401"),
        ("rtsp://u:p@q@h/m: 401", "rtsp://***:***@h/m: 401"),
        ("rtsp://u:p/q#r?s@h/m failed", "rtsp://***:***@h/m failed"),
        (
            "rtsp://u%40x:p%2F%23%3F%40%25q@cam.local:554/videoMain: Invalid data",
            "rtsp://***:***@cam.local:554/videoMain: Invalid data",
        ),
        (
            "a rtsp://u:p@h/m and rtsps://x:y@g/n",
            "a rtsp://***:***@h/m and rtsps://***:***@g/n",
        ),
        (
            "GET /cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=u&pwd=p",
            "GET /cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=u&pwd=***",
        ),
        ("POST /x?Password=p&next=1", "POST /x?Password=***&next=1"),
        (
            "[rtsp @ 0x5626] method DESCRIBE failed: 404 Not Found",
            "[rtsp @ 0x5626] method DESCRIBE failed: 404 Not Found",
        ),
        ("bypass=1 mypwd=2", "bypass=1 mypwd=2"),
        ("rtsp://***:***@h/m", "rtsp://***:***@h/m"),
        ("", ""),
    ],
    ids=[
        "userinfo",
        "user-only",
        "raw-at-in-password",
        "raw-reserved-chars-in-password",
        "percent-encoded-password",
        "two-urls",
        "cgi-pwd",
        "password-query-any-case",
        "ffmpeg-log-tag-untouched",
        "no-word-boundary-untouched",
        "idempotent",
        "empty",
    ],
)
def test_redact_text(text: str, expected: str) -> None:
    assert redact_text(text) == expected


# --- status labels and redacted errors on the card data ---


@pytest.mark.parametrize(
    ("status", "restart_in", "expected"),
    [
        ("running", None, "Running"),
        ("restarting", 7, "Restarting in 7s"),
        ("restarting", 0, "Restarting"),
        ("restarting", None, "Restarting"),
        ("failed", None, "Failed"),
        ("idle", None, "Idle"),
        ("waiting", None, "Waiting for event"),
        ("unknown", None, "Unknown"),
        ("stopped", None, "Stopped"),
    ],
)
def test_status_label(status: str, restart_in: int | None, expected: str) -> None:
    assert status_label(status, restart_in) == expected


def test_live_status_redacts_last_error():
    st = live_status(_rt([_Proc(False)], last_error="open rtsp://u:p@h/m failed"), now=100.0)
    assert st["last_error"] == "open rtsp://***:***@h/m failed"


def test_list_cameras_without_runtime_is_unknown():
    cfg = AppConfig(cameras=[CameraConfig(name="cam", main_url="rtsp://u:p@h/m")])
    row = list_cameras(cfg)[0]
    assert row["status"] == "unknown"
    assert row["status_label"] == "Unknown"
    assert row["last_error"] == ""


def test_list_cameras_label_and_redacted_error_from_runtime():
    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    cam_rt = _rt([_Proc(False)], last_error="Connection to rtsp://u:p@h/m failed")
    cam_rt.camera = cam
    row = list_cameras(AppConfig(cameras=[cam]), SimpleNamespace(cameras=[cam_rt]))[0]
    assert row["status"] == "failed"
    assert row["status_label"] == "Failed"
    assert row["last_error"] == "Connection to rtsp://***:***@h/m failed"
    assert row["main_url_redacted"] == "rtsp://***:***@h/m"


def test_last_stderr_line_is_redacted_before_the_300_char_cut():
    line = "x" * 285 + " rtsp://u:pwpwpwpw@h/m"
    out = _last_stderr_line([SimpleNamespace(proc=_Proc(False, tail=["", line]))])
    assert "pw" not in out
    assert out == ("x" * 285 + " rtsp://***:***@h/m")[:300]


# --- a running camera whose proxy cannot start is "degraded", not "running" ---


def test_status_label_degraded():
    assert status_label("degraded") == "Proxy down"


def test_live_status_degraded_when_the_proxy_cannot_start():
    cam_rt = _rt([_Proc(True)], last_error="OSError: [Errno 98] Address already in use")
    cam_rt.proxy_error = "OSError: [Errno 98] Address already in use"
    st = live_status(cam_rt, now=100.0)
    assert st["status"] == "degraded"
    assert st["last_error"] == "OSError: [Errno 98] Address already in use"


def test_live_status_proxy_error_is_redacted():
    cam_rt = _rt([_Proc(True)])
    cam_rt.proxy_error = "RuntimeError: publish to rtsp://u:p@h/m refused"
    st = live_status(cam_rt, now=100.0)
    assert st["status"] == "degraded"
    assert st["last_error"] == "RuntimeError: publish to rtsp://***:***@h/m refused"


def test_dead_ingest_wins_over_proxy_error():
    cam_rt = _rt([_Proc(False)], last_error="ffmpeg died")
    cam_rt.proxy_error = "OSError: port taken"
    st = live_status(cam_rt, now=100.0)
    assert st["status"] == "failed"
    assert st["last_error"] == "ffmpeg died"


def test_running_without_proxy_error_stays_running():
    cam_rt = _rt([_Proc(True)])
    cam_rt.proxy_error = ""
    assert live_status(cam_rt, now=100.0)["status"] == "running"


def test_error_text_masks_a_password_with_a_raw_slash():
    from rtsp_warden.app import _error_text

    text = _error_text(OSError("open rtsp://u:p/q#r@h/m failed"))
    assert text == "OSError: open rtsp://***:***@h/m failed"


# --- /status.json and /health need no login, so they must never carry a password ---


class _StatusProc(_Proc):
    """Fake ManagedProcess with the extra fields cli._proc_status reads."""

    args = ["ffmpeg", "-i", "rtsp://u:p@h/m"]

    def pid(self):
        return 4242


def _status_client(tail: list[str]):
    from fastapi.testclient import TestClient

    from rtsp_warden.web.app import create_app
    from rtsp_warden.web.config import WebSettings

    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    ingest = SimpleNamespace(
        stream_name="main",
        upstream_url=cam.main_url,
        proc=_StatusProc(False, tail=tail),
        record_cfg=None,
        record_output_dir=None,
        mjpeg_hub=None,
        rtsp_publish_url=None,
    )
    cam_rt = SimpleNamespace(
        camera=cam, proxy=None, recorder=SimpleNamespace(processes=lambda: [ingest])
    )
    runtime = SimpleNamespace(cameras=[cam_rt])
    app = create_app(WebSettings(), cfg=AppConfig(cameras=[cam]), runtime_provider=lambda: runtime)
    return TestClient(app)


def test_status_json_and_health_json_redact_stderr_tail(db_with_user):
    client = _status_client(
        ["Error opening input file rtsp://u:p@h/m.", "GET /snap?usr=u&pwd=p&x=1 failed"]
    )
    for r in (
        client.get("/status.json"),
        client.get("/health", headers={"Accept": "application/json"}),
    ):
        assert r.status_code == 200
        assert "u:p@" not in r.text
        assert "pwd=p" not in r.text
        assert "Error opening input file rtsp://***:***@h/m." in r.text
        assert "pwd=***" in r.text


def test_health_page_and_metrics_never_show_the_password(db_with_user):
    client = _status_client(["Error opening input file rtsp://u:p@h/m."])
    for path in ("/health", "/health/partial", "/metrics"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "u:p@" not in r.text, path
