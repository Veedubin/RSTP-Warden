from rtsp_warden import retention_resolver
from rtsp_warden.config import CameraConfig, RetentionConfig
from rtsp_warden.retention_resolver import resolve_retention


def _cam(**kw) -> CameraConfig:
    return CameraConfig(name="c", main_url="rtsp://h/m", sub_url="rtsp://h/s", **kw)


def test_camera_override_wins():
    cam = _cam(retention=RetentionConfig(max_days=3), record={"retention": {"max_days": 9}})
    assert resolve_retention(cam, RetentionConfig(max_days=30)).max_days == 3


def test_legacy_record_retention_is_honored_with_warning(monkeypatch):
    # Capture the module logger directly: the app's logging setup disables
    # propagation, so caplog on the root logger is order-dependent.
    messages: list[str] = []
    monkeypatch.setattr(
        retention_resolver.log, "warning", lambda msg, *args: messages.append(msg % args)
    )
    cam = _cam(record={"retention": {"max_days": 9}})
    out = resolve_retention(cam, RetentionConfig(max_days=30))
    assert out.max_days == 9
    assert len(messages) == 1
    assert "record.retention" in messages[0]
    assert "cameras[].retention" in messages[0]


def test_global_used_when_nothing_set():
    cam = _cam()
    assert resolve_retention(cam, RetentionConfig(max_days=30)).max_days == 30
