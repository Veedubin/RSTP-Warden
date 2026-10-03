from types import SimpleNamespace

from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.services.cameras import live_status


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
