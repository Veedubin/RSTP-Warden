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
