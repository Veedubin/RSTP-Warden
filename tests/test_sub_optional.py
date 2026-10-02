from rtsp_warden.config import AppConfig, CameraConfig, RuntimeConfig
from rtsp_warden.recorder import CameraRecorder
from rtsp_warden.web.services.cameras import list_cameras


def test_sub_url_defaults_to_none_and_proxy_falls_back_to_main():
    cam = CameraConfig(name="c", main_url="rtsp://h/m", proxy={"stream": "sub"})
    assert cam.sub_url is None
    assert cam.proxy.stream == "main"


def test_recorder_builds_only_main_when_no_sub(tmp_path):
    cam = CameraConfig(
        name="c",
        main_url="rtsp://h/m",
        record={"enabled": True, "output_dir": str(tmp_path)},
    )
    rec = CameraRecorder(camera=cam, runtime=RuntimeConfig())
    assert rec.main is not None
    assert rec.sub is None
    assert rec.has_any()


def test_list_cameras_handles_missing_sub():
    cfg = AppConfig(cameras=[CameraConfig(name="c", main_url="rtsp://u:p@h/m")])
    row = list_cameras(cfg)[0]
    assert row["sub_url_redacted"] is None
    assert row["main_url_redacted"].startswith("rtsp://***")


def test_recorder_builds_main_for_proxy_only_camera():
    cam = CameraConfig(
        name="c", main_url="rtsp://h/m", record={"enabled": False}, proxy={"enabled": True}
    )
    rec = CameraRecorder(camera=cam, runtime=RuntimeConfig())
    assert rec.main is not None
    assert rec.has_any()
