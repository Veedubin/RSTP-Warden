from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.preview import mjpeg_frames

JPEG = b"\xff\xd8\xff\xd9"


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post(
        "/login", data={"username": "admin", "password": "testpass123", "csrf_token": token}
    )


@pytest.fixture
def hub() -> FrameHub:
    h = FrameHub()
    h.update(JPEG)
    return h


@pytest.fixture
def client(db_with_user, hub) -> TestClient:
    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    cfg = AppConfig(cameras=[cam])
    # The dashboard's status builder reads recorder and proxy, so the fake has both.
    cam_rt = SimpleNamespace(
        camera=cam, hub=hub, proxy=None, recorder=SimpleNamespace(processes=lambda: [])
    )
    runtime = SimpleNamespace(cameras=[cam_rt])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: runtime, runtime=runtime)
    c = TestClient(app)
    _login(c)
    return c


def test_snapshot_route_serves_latest_frame(client):
    r = client.get("/cameras/cam/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG


def test_snapshot_503_without_runtime(db_with_user):
    cfg = AppConfig(cameras=[CameraConfig(name="cam", main_url="rtsp://h/m")])
    c = TestClient(create_app(WebSettings(), cfg=cfg))
    _login(c)
    assert c.get("/cameras/cam/snapshot.jpg").status_code == 503


def test_mjpeg_frames_generator_emits_multipart_parts(hub):
    parts = list(mjpeg_frames(hub, stop_after=1))
    assert parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n")
    assert parts[0].endswith(JPEG + b"\r\n")


def test_detail_page_uses_same_origin_urls(client):
    html = client.get("/cameras/cam").text
    assert "/cameras/cam/live.mjpeg" in html
    assert "127.0.0.1:9001" not in html
