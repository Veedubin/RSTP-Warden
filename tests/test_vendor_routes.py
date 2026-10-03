"""Camera settings page for Foscam cameras (RW-4): /cameras/{name}/vendor and its fragments.

The CGI client is replaced by a fake; config write-back goes to a real temp config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import rtsp_warden.web.routes.vendor as vendor_routes
from rtsp_warden.auth import hash_password
from rtsp_warden.config import AppConfig, CameraConfig, VendorConfig, load_config
from rtsp_warden.db.schema import create_user
from rtsp_warden.vendors.foscam import (
    DeviceInfo,
    FoscamError,
    ImageSettings,
    StreamProfile,
    VideoSettings,
)
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings

LEAK = "pw-must-not-leak"

RAW_CONFIG = """\
cameras:
  - name: front_door
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.72:554/videoMain
    record:
      enabled: false
    proxy:
      enabled: false
    vendor:
      type: foscam
      port: 88
  - name: garage
    main_url: rtsp://u:p@192.0.2.11/m
    record:
      enabled: false
    proxy:
      enabled: false
  - name: anon
    main_url: rtsp://192.0.2.12/m
    record:
      enabled: false
    proxy:
      enabled: false
    vendor:
      type: foscam
"""


def _profiles() -> list[StreamProfile]:
    return [
        StreamProfile(index=0, resolution=0, bit_rate=2097152, frame_rate=25, gop=50, vbr=True),
        StreamProfile(index=1, resolution=0, bit_rate=1048576, frame_rate=15, gop=30, vbr=False),
        StreamProfile(index=2, resolution=3, bit_rate=524288, frame_rate=15, gop=60, vbr=True),
        StreamProfile(index=3, resolution=0, bit_rate=2097152, frame_rate=30, gop=60, vbr=True),
    ]


class FakeFoscam:
    """Stands in for FoscamClient: records calls, answers from attributes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail: FoscamError | None = None
        self.types = {"main": 1, "sub": 0}
        self.profiles = {"main": _profiles(), "sub": _profiles()}
        self.image = ImageSettings(
            brightness=60, contrast=48, hue=50, saturation=56, sharpness=48, denoise=50
        )
        self.video = VideoSettings(
            mirror=False,
            flip=True,
            infrared_mode=1,
            osd_timestamp=True,
            osd_name=True,
            osd_position=0,
        )
        self.info = DeviceInfo(
            product="C1+V3", firmware="2.82.2.35", hardware="1.12.5.4", name="Cam"
        )

    def _rec(self, call: str, /, *args: Any, **kwargs: Any) -> None:
        if self.fail is not None:
            raise self.fail
        self.calls.append((call, args, kwargs))

    def device_info(self) -> DeviceInfo:
        self._rec("device_info")
        return self.info

    def stream_type(self, stream: str) -> int:
        self._rec("stream_type", stream)
        return self.types[stream]

    def set_stream_type(self, stream: str, index: int) -> None:
        self._rec("set_stream_type", stream, index)
        self.types[stream] = index

    def stream_profiles(self, stream: str) -> list[StreamProfile]:
        self._rec("stream_profiles", stream)
        return self.profiles[stream]

    def set_stream_profile(self, stream: str, profile: StreamProfile) -> None:
        self._rec("set_stream_profile", stream, profile)
        self.profiles[stream][profile.index] = profile

    def image_settings(self) -> ImageSettings:
        self._rec("image_settings")
        return self.image

    def set_image_setting(self, name: str, value: int) -> None:
        self._rec("set_image_setting", name, value)

    def video_settings(self) -> VideoSettings:
        self._rec("video_settings")
        return self.video

    def set_mirror(self, on: bool) -> None:
        self._rec("set_mirror", on)

    def set_flip(self, on: bool) -> None:
        self._rec("set_flip", on)

    def set_infrared_mode(self, mode: int) -> None:
        self._rec("set_infrared_mode", mode)

    def set_infrared(self, on: bool) -> None:
        self._rec("set_infrared", on)

    def set_osd(self, **kwargs: Any) -> None:
        self._rec("set_osd", **kwargs)

    def snapshot(self) -> bytes:
        self._rec("snapshot")
        return b"\xff\xd8jpeg\xff\xd9"

    def reboot(self) -> None:
        self._rec("reboot")

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CAM_USER", "jc")
    monkeypatch.setenv("CAM_PASS", LEAK)
    path = tmp_path / "config.yaml"
    path.write_text(RAW_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeFoscam:
    client = FakeFoscam()
    real = vendor_routes._client

    def factory(cfg: AppConfig, camera: CameraConfig) -> Any:
        real(cfg, camera)  # the real factory still validates host, vendor and credentials
        return client

    monkeypatch.setattr(vendor_routes, "_client", factory)
    return client


@pytest.fixture
def cfg(config_file: Path) -> AppConfig:
    return load_config(config_file)


@pytest.fixture
def client(db_with_user: str, config_file: Path, cfg: AppConfig, fake: FakeFoscam) -> TestClient:
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None, config_path=config_file)
    c = TestClient(app)
    _login(c)
    return c


def _login(client: TestClient, username: str = "admin", password: str = "testpass123") -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _post(
    client: TestClient, url: str, data: dict[str, str] | None = None, *, htmx: bool = True
) -> httpx.Response:
    token = client.cookies.get("warden_csrf", "")
    headers = {"X-CSRF-Token": token}
    if htmx:
        headers["HX-Request"] = "true"
    return client.post(
        url, data={**(data or {}), "csrf_token": token}, headers=headers, follow_redirects=False
    )


def _raw(path: Path, name: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return next(c for c in data["cameras"] if c["name"] == name)


# --- config ----------------------------------------------------------------------------------


def test_vendor_config_defaults_port_88_and_validates() -> None:
    cam = CameraConfig(name="c", main_url="rtsp://u:p@h/m", vendor={"type": "foscam"})
    assert cam.vendor == VendorConfig(type="foscam", port=88)
    assert CameraConfig(name="c", main_url="rtsp://u:p@h/m").vendor is None
    with pytest.raises(ValidationError):
        CameraConfig(name="c", main_url="rtsp://u:p@h/m", vendor={"type": "hikvision"})
    with pytest.raises(ValidationError, match="port"):
        CameraConfig(name="c", main_url="rtsp://u:p@h/m", vendor={"type": "foscam", "port": 0})


# --- the real client factory --------------------------------------------------------------


def test_client_factory_uses_main_url_host_and_credentials_and_vendor_port(cfg: AppConfig) -> None:
    cam = cfg.cameras[0]
    client = vendor_routes._client(cfg, cam)
    assert client.base_url == "http://192.0.2.72:88/cgi-bin/CGIProxy.fcgi"
    assert (client.username, client.password) == ("jc", LEAK)
    with pytest.raises(FoscamError, match="vendor"):
        vendor_routes._client(cfg, cfg.cameras[1])  # garage: no vendor block
    with pytest.raises(FoscamError, match="credentials"):
        vendor_routes._client(cfg, cfg.cameras[2])  # anon: no user info in main_url


# --- page ----------------------------------------------------------------------------------


def test_detail_page_links_to_camera_settings_for_admins(client: TestClient) -> None:
    r = client.get("/cameras/front_door")
    assert r.status_code == 200
    assert 'href="/cameras/front_door/vendor"' in r.text


def test_page_shows_device_streams_image_and_video_state(
    client: TestClient, fake: FakeFoscam
) -> None:
    r = client.get("/cameras/front_door/vendor")
    assert r.status_code == 200
    text = r.text
    assert LEAK not in text
    assert "C1+V3" in text and "2.82.2.35" in text
    # main stream: four profiles, the active one marked, resolution labelled
    assert "1280x720" in text and "640x360" in text
    assert 'name="profile" value="1" checked' in text
    assert "2048 kbit/s" in text or "2.0 Mbit/s" in text
    # image sliders carry the camera's values
    assert 'name="brightness"' in text and 'value="60"' in text
    assert 'name="denoise"' in text
    # video state
    assert 'name="flip" checked' in text
    assert 'name="mirror"' in text and 'name="mirror" checked' not in text
    assert 'value="1" selected' in text  # infrared mode manual
    assert 'src="/cameras/front_door/vendor/snapshot.jpg"' in text
    assert "device_info" in fake.names() and "video_settings" in fake.names()


def test_page_without_vendor_offers_the_enable_form(client: TestClient) -> None:
    r = client.get("/cameras/garage/vendor")
    assert r.status_code == 200
    assert 'action="/cameras/garage/vendor/enable"' in r.text
    assert 'name="port"' in r.text and 'value="88"' in r.text
    assert "C1+V3" not in r.text


def test_page_reports_camera_errors_without_a_500(client: TestClient, fake: FakeFoscam) -> None:
    fake.fail = FoscamError("getDevInfo: the camera rejected the user name or password", code=-2)
    r = client.get("/cameras/front_door/vendor")
    assert r.status_code == 200
    assert "rejected the user name or password" in r.text
    assert LEAK not in r.text


def test_page_without_credentials_says_so(client: TestClient) -> None:
    r = client.get("/cameras/anon/vendor")
    assert r.status_code == 200
    assert "credentials" in r.text


def test_unknown_camera_is_404_and_viewers_are_403(
    client: TestClient, db_with_user: str, cfg: AppConfig, config_file: Path
) -> None:
    assert client.get("/cameras/nope/vendor").status_code == 404
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None, config_path=config_file)
    viewer = TestClient(app)
    _login(viewer, "viewer", "viewerpass123")
    assert viewer.get("/cameras/front_door/vendor").status_code == 403
    assert _post(viewer, "/cameras/front_door/vendor/reboot").status_code == 403


# --- enable / disable ------------------------------------------------------------------------


def test_enable_writes_the_vendor_block_and_keeps_env_references(
    client: TestClient, cfg: AppConfig, config_file: Path
) -> None:
    r = _post(
        client,
        "/cameras/garage/vendor/enable",
        {"vendor_type": "foscam", "port": "8088"},
        htmx=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/garage/vendor"
    assert _raw(config_file, "garage")["vendor"] == {"type": "foscam", "port": 8088}
    assert cfg.cameras[1].vendor == VendorConfig(type="foscam", port=8088)
    assert "${CAM_USER}" in config_file.read_text(encoding="utf-8")
    assert (
        _post(
            client, "/cameras/garage/vendor/enable", {"vendor_type": "foscam", "port": "x"}
        ).status_code
        == 422
    )


def test_disable_removes_the_vendor_block(
    client: TestClient, cfg: AppConfig, config_file: Path
) -> None:
    r = _post(client, "/cameras/front_door/vendor/disable", htmx=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front_door"
    assert "vendor" not in _raw(config_file, "front_door")
    assert cfg.cameras[0].vendor is None


# --- stream ----------------------------------------------------------------------------------


def test_use_profile_switches_the_stream_type(client: TestClient, fake: FakeFoscam) -> None:
    r = _post(
        client,
        "/cameras/front_door/vendor/stream",
        {"stream": "main", "action": "use", "profile": "0"},
    )
    assert r.status_code == 200
    assert ("set_stream_type", ("main", 0), {}) in fake.calls
    assert "now plays profile 0" in r.text
    assert 'name="profile" value="0" checked' in r.text


def test_save_profile_sends_every_field(client: TestClient, fake: FakeFoscam) -> None:
    r = _post(
        client,
        "/cameras/front_door/vendor/stream",
        {
            "stream": "sub",
            "action": "save",
            "profile": "2",
            "resolution": "3",
            "bit_rate": "524288",
            "frame_rate": "10",
            "gop": "20",
            "vbr": "on",
        },
    )
    assert r.status_code == 200
    saved = next(c for c in fake.calls if c[0] == "set_stream_profile")
    assert saved[1] == (
        "sub",
        StreamProfile(index=2, resolution=3, bit_rate=524288, frame_rate=10, gop=20, vbr=True),
    )
    assert "Profile 2 saved" in r.text


@pytest.mark.parametrize(
    "data",
    [
        {"stream": "main", "action": "use", "profile": "7"},
        {"stream": "third", "action": "use", "profile": "0"},
        {
            "stream": "main",
            "action": "save",
            "profile": "0",
            "resolution": "0",
            "bit_rate": "x",
            "frame_rate": "10",
            "gop": "20",
        },
        {
            "stream": "main",
            "action": "save",
            "profile": "0",
            "resolution": "0",
            "bit_rate": "1000",
            "frame_rate": "99",
            "gop": "20",
        },
        {"stream": "main", "action": "dance", "profile": "0"},
    ],
)
def test_stream_rejects_bad_input(
    client: TestClient, fake: FakeFoscam, data: dict[str, str]
) -> None:
    assert _post(client, "/cameras/front_door/vendor/stream", data).status_code == 422
    assert not any(c[0].startswith("set_") for c in fake.calls)


def test_stream_plain_post_redirects_and_camera_error_is_shown(
    client: TestClient, fake: FakeFoscam
) -> None:
    r = _post(
        client,
        "/cameras/front_door/vendor/stream",
        {"stream": "main", "action": "use", "profile": "0"},
        htmx=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front_door/vendor"
    fake.fail = FoscamError(
        "setMainVideoStreamType: the camera could not execute the command", code=-4
    )
    r = _post(
        client,
        "/cameras/front_door/vendor/stream",
        {"stream": "main", "action": "use", "profile": "0"},
    )
    assert r.status_code == 200
    assert "could not execute" in r.text


# --- image -----------------------------------------------------------------------------------


def test_image_sends_only_the_changed_values(client: TestClient, fake: FakeFoscam) -> None:
    r = _post(
        client,
        "/cameras/front_door/vendor/image",
        {
            "brightness": "70",
            "contrast": "48",
            "hue": "50",
            "saturation": "56",
            "sharpness": "48",
            "denoise": "30",
        },
    )
    assert r.status_code == 200
    sets = [c for c in fake.calls if c[0] == "set_image_setting"]
    assert sets == [
        ("set_image_setting", ("brightness", 70), {}),
        ("set_image_setting", ("denoise", 30), {}),
    ]
    assert "Image settings saved" in r.text
    assert (
        _post(client, "/cameras/front_door/vendor/image", {"brightness": "101"}).status_code == 422
    )


# --- video (mirror, flip, infrared, OSD) -------------------------------------------------------


def test_video_applies_mirror_flip_infrared_and_osd(client: TestClient, fake: FakeFoscam) -> None:
    r = _post(
        client,
        "/cameras/front_door/vendor/video",
        {
            "mirror": "on",
            "infrared_mode": "1",
            "infrared_on": "on",
            "osd_timestamp": "on",
            "osd_position": "2",
        },
    )
    assert r.status_code == 200
    names = fake.names()
    assert ("set_mirror", (True,), {}) in fake.calls
    assert ("set_flip", (False,), {}) in fake.calls  # the box was unticked
    assert ("set_infrared", (True,), {}) in fake.calls
    assert "set_infrared_mode" not in names  # unchanged (already manual)
    osd = next(c for c in fake.calls if c[0] == "set_osd")
    assert osd[2] == {
        "timestamp": True,
        "name": False,
        "position": 2,
        "temp_humid": False,
        "mask": False,
    }
    assert "Video settings saved" in r.text


def test_video_auto_infrared_never_switches_the_led_by_hand(
    client: TestClient, fake: FakeFoscam
) -> None:
    _post(
        client,
        "/cameras/front_door/vendor/video",
        {
            "flip": "on",
            "infrared_mode": "0",
            "infrared_on": "on",
            "osd_timestamp": "on",
            "osd_name": "on",
            "osd_position": "0",
        },
    )
    names = fake.names()
    assert "set_infrared_mode" in names and "set_infrared" not in names
    assert "set_mirror" not in names and "set_flip" not in names and "set_osd" not in names


# --- snapshot and reboot ------------------------------------------------------------------------


def test_snapshot_route_serves_the_camera_jpeg(client: TestClient, fake: FakeFoscam) -> None:
    r = client.get("/cameras/front_door/vendor/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8jpeg\xff\xd9"
    fake.fail = FoscamError("snapPicture2: the camera timed out", code=-5)
    assert client.get("/cameras/front_door/vendor/snapshot.jpg").status_code == 502


def test_reboot_sends_the_command_and_redirects_with_a_flash(
    client: TestClient, fake: FakeFoscam
) -> None:
    r = _post(client, "/cameras/front_door/vendor/reboot", htmx=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front_door/vendor"
    assert fake.names() == ["reboot"]
    assert "warden_flash" in r.headers.get("set-cookie", "")
