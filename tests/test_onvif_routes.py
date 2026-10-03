"""ONVIF page routes: form posts, htmx fragments, per-camera ONVIF port, locked preset writes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rtsp_warden.web.routes.onvif as onvif_routes
from rtsp_warden.config import AppConfig, CameraConfig, load_config
from rtsp_warden.onvif import events as onvif_events
from rtsp_warden.onvif.discovery import DiscoveredCamera, OnvifError
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.paths import STATIC_DIR

RAW_CONFIG = """\
cameras:
  - name: front_door
    main_url: rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.10:554/videoMain
    onvif_port: 888
    record:
      enabled: false
    proxy:
      enabled: false
    presets:
      - name: gate
        pan: 0.25
        tilt: 0.3
        zoom: 0.0
  - name: garage
    main_url: rtsp://u:p@192.0.2.11/m
    record:
      enabled: false
    proxy:
      enabled: false
onvif:
  discovery_enabled: true
  ptz_enabled: true
  events_enabled: true
"""

LEAK = "pw-must-not-leak"

CAPABILITIES = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <soap:Body>
    <tds:GetCapabilitiesResponse>
      <tds:Capabilities>
        <tt:PTZ xmlns:tt="http://www.onvif.org/ver10/schema">
          <tt:XAddr>http://192.0.2.10:888/onvif/ptz_service</tt:XAddr>
        </tt:PTZ>
      </tds:Capabilities>
    </tds:GetCapabilitiesResponse>
  </soap:Body>
</soap:Envelope>"""

PROFILES = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <soap:Body>
    <tptz:GetProfilesResponse>
      <tptz:Profiles token="profile1"/>
    </tptz:GetProfilesResponse>
  </soap:Body>
</soap:Envelope>"""

OK = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body/>
</soap:Envelope>"""

SOAP_OPS = ("GetCapabilities", "GetProfiles", "ContinuousMove", "AbsoluteMove", "Stop")


def _soap_transport(captured: list[httpx.Request], *, fail: bool = False) -> httpx.MockTransport:
    """Answer the PTZ SOAP calls by operation name and record every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if fail:
            return httpx.Response(500, text="camera error")
        body = request.content.decode()
        if ":GetCapabilities" in body:
            return httpx.Response(200, text=CAPABILITIES)
        if ":GetProfiles" in body:
            return httpx.Response(200, text=PROFILES)
        return httpx.Response(200, text=OK)

    return httpx.MockTransport(handler)


def _ops(captured: list[httpx.Request]) -> list[str]:
    """SOAP operation of each captured request, in order."""
    ops = []
    for request in captured:
        body = request.content.decode()
        ops.append(next(op for op in SOAP_OPS if f":{op}" in body))
    return ops


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _post(
    client: TestClient, url: str, data: dict[str, str] | None = None, *, htmx: bool = True
) -> httpx.Response:
    """Form POST with the CSRF token; as htmx unless htmx=False."""
    token = client.cookies.get("warden_csrf", "")
    headers = {"X-CSRF-Token": token}
    if htmx:
        headers["HX-Request"] = "true"
    return client.post(
        url,
        data={**(data or {}), "csrf_token": token},
        headers=headers,
        follow_redirects=False,
    )


class FakeSubscriber:
    """Stands in for OnvifEventSubscriber: no network, no poll task."""

    def __init__(self, camera_name: str) -> None:
        self.camera_name = camera_name
        self.subscription_ref = "http://192.0.2.10:888/onvif/subscription?id=1"
        self.is_running = False
        self.last_event_time = None
        self.stopped = False

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False
        self.stopped = True


@pytest.fixture(autouse=True)
def _clear_subscriptions() -> Iterator[None]:
    """The subscriber registry is process-global; keep tests independent."""
    onvif_events._active_subscribers.clear()
    yield
    onvif_events._active_subscribers.clear()


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", LEAK)
    path = tmp_path / "config.yaml"
    path.write_text(RAW_CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def app_and_cfg(db_with_user: str, config_file: Path) -> tuple[FastAPI, AppConfig]:
    cfg = load_config(config_file)
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None, config_path=config_file)
    return app, cfg


@pytest.fixture
def client(app_and_cfg: tuple[FastAPI, AppConfig]) -> TestClient:
    app, _ = app_and_cfg
    c = TestClient(app)
    _login(c)
    return c


@pytest.fixture
def ptz_requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Route PTZ calls through a MockTransport while keeping the real client factory."""
    captured: list[httpx.Request] = []
    real_factory = onvif_routes._ptz_client

    def factory(cfg: AppConfig, camera: CameraConfig):
        ptz = real_factory(cfg, camera)
        ptz._transport = _soap_transport(captured)
        return ptz

    monkeypatch.setattr(onvif_routes, "_ptz_client", factory)
    return captured


# --- device service URL -------------------------------------------------------


@pytest.mark.parametrize(
    ("main_url", "onvif_port", "expected"),
    [
        (
            "rtsp://u:p@192.0.2.10:554/videoMain",
            888,
            "http://192.0.2.10:888/onvif/device_service",
        ),
        ("rtsp://u:p@192.0.2.10:554/videoMain", None, "http://192.0.2.10/onvif/device_service"),
        ("rtsp://u:p@192.0.2.10/m", 80, "http://192.0.2.10/onvif/device_service"),
        (
            "rtsp://u:p@[2001:db8::5]:554/m",
            8080,
            "http://[2001:db8::5]:8080/onvif/device_service",
        ),
        ("rtsp://u:p@ss@192.0.2.12/m", None, "http://192.0.2.12/onvif/device_service"),
    ],
)
def test_derive_onvif_xaddr_uses_camera_onvif_port(
    main_url: str, onvif_port: int | None, expected: str
) -> None:
    cam = CameraConfig(name="c", main_url=main_url, onvif_port=onvif_port)
    assert onvif_routes._derive_onvif_xaddr(cam) == expected


def test_derive_onvif_xaddr_without_host_raises() -> None:
    cam = CameraConfig(name="c", main_url="rtsp:///nohost")
    with pytest.raises(OnvifError, match="no host"):
        onvif_routes._derive_onvif_xaddr(cam)


def test_client_factories_use_camera_port_and_global_credentials() -> None:
    cam = CameraConfig(name="front_door", main_url="rtsp://u:p@192.0.2.10:554/m", onvif_port=888)
    cfg = AppConfig(
        cameras=[cam], onvif={"username": "onvif-user", "password": "x", "ptz_timeout_seconds": 4}
    )
    ptz = onvif_routes._ptz_client(cfg, cam)
    assert ptz.device_xaddr == "http://192.0.2.10:888/onvif/device_service"
    assert ptz.username == "onvif-user"
    assert ptz.timeout_seconds == 4
    sub = onvif_routes._event_client(cfg, cam)
    assert sub.camera_name == "front_door"
    assert sub._client.ptz.device_xaddr == "http://192.0.2.10:888/onvif/device_service"
    assert sub._topics == [
        "tns1:VideoSource/MotionAlarm",
        "tns1:VideoSource/ImagingAlarm",
        "tns1:RuleEngine",
    ]


# --- page ----------------------------------------------------------------------


def _onvif_content(html: str) -> str:
    """The ONVIF page's own markup (base.html chrome excluded)."""
    return html[html.index('<article id="onvif-discovery"') : html.rindex("</article>")]


def test_page_renders_server_side_controls_without_alpine(client: TestClient) -> None:
    r = client.get("/onvif?camera=front_door")
    assert r.status_code == 200
    body = _onvif_content(r.text)
    assert 'hx-post="/onvif/cameras/front_door/ptz"' in body
    assert 'name="direction" value="up"' in body
    assert 'hx-post="/onvif/cameras/front_door/ptz/goto"' in body
    assert 'hx-post="/onvif/cameras/front_door/ptz/delete"' in body
    assert 'hx-post="/onvif/cameras/front_door/ptz/save"' in body
    assert 'hx-post="/onvif/cameras/garage/events/subscribe"' in body
    assert "http://192.0.2.10:888/onvif/device_service" in body
    assert "<td>gate</td>" in body
    assert '<option value="front_door" selected>' in body
    for leftover in (":hx-post", "x-for", "x-data", "hx-vals", "placeholder/ptz"):
        assert leftover not in body, leftover
    assert "routed into the alert system" not in body


def test_page_without_camera_hides_ptz_and_lists_subscriptions(client: TestClient) -> None:
    r = client.get("/onvif")
    assert r.status_code == 200
    assert 'id="onvif-ptz"' not in r.text
    assert 'id="onvif-subscriptions"' in r.text
    assert "<td>garage</td>" in r.text


def test_page_with_unknown_camera_shows_notice(client: TestClient) -> None:
    r = client.get("/onvif?camera=nope")
    assert r.status_code == 200
    assert "nope" in r.text
    assert "is not in config.yaml" in r.text
    assert 'id="onvif-ptz"' not in r.text


def test_old_per_camera_ptz_page_redirects(client: TestClient) -> None:
    r = client.get("/onvif/cameras/front_door/ptz", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/onvif?camera=front_door"


# --- PTZ moves -----------------------------------------------------------------


def test_ptz_move_posts_form_and_returns_fragment(
    client: TestClient, ptz_requests: list[httpx.Request]
) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz", {"direction": "left", "duration_ms": "10"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="onvif-ptz"' in r.text
    assert "Moved left for 10 ms." in r.text
    assert "<html" not in r.text
    assert str(ptz_requests[0].url) == "http://192.0.2.10:888/onvif/device_service"
    assert _ops(ptz_requests) == ["GetCapabilities", "GetProfiles", "ContinuousMove", "Stop"]
    assert 'x="-1.0000"' in ptz_requests[2].content.decode()


@pytest.mark.parametrize(
    ("direction", "duration_ms", "expected_ops", "expected_text"),
    [
        ("stop", "500", ["GetCapabilities", "GetProfiles", "Stop"], "Stopped."),
        (
            "zoom_in",
            "0",
            ["GetCapabilities", "GetProfiles", "ContinuousMove"],
            "Moving zoom in. Press Stop to halt.",
        ),
    ],
)
def test_ptz_stop_and_move_until_stop(
    client: TestClient,
    ptz_requests: list[httpx.Request],
    direction: str,
    duration_ms: str,
    expected_ops: list[str],
    expected_text: str,
) -> None:
    r = _post(
        client,
        "/onvif/cameras/front_door/ptz",
        {"direction": direction, "duration_ms": duration_ms},
    )
    assert r.status_code == 200
    assert _ops(ptz_requests) == expected_ops
    assert expected_text in r.text


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"direction": "sideways"}, "Unknown PTZ direction"),
        ({"direction": "up", "duration_ms": "abc"}, "Duration must be a whole number"),
        ({"direction": "up", "duration_ms": "20000"}, "Duration must be a whole number"),
    ],
)
def test_ptz_bad_input_is_422_fragment(
    client: TestClient, ptz_requests: list[httpx.Request], data: dict[str, str], message: str
) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz", data)
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="onvif-ptz"' in r.text
    assert message in r.text
    assert ptz_requests == []


def test_ptz_disabled_shows_notice_without_camera_calls(
    client: TestClient,
    app_and_cfg: tuple[FastAPI, AppConfig],
    ptz_requests: list[httpx.Request],
) -> None:
    _, cfg = app_and_cfg
    cfg.onvif.ptz_enabled = False
    r = _post(client, "/onvif/cameras/front_door/ptz", {"direction": "up"})
    assert r.status_code == 200
    assert "PTZ is off" in r.text
    assert ptz_requests == []


def test_ptz_camera_failure_is_shown_in_fragment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[httpx.Request] = []
    real_factory = onvif_routes._ptz_client

    def failing_factory(cfg: AppConfig, camera: CameraConfig):
        ptz = real_factory(cfg, camera)
        ptz._transport = _soap_transport(captured, fail=True)
        return ptz

    monkeypatch.setattr(onvif_routes, "_ptz_client", failing_factory)
    r = _post(client, "/onvif/cameras/front_door/ptz", {"direction": "up", "duration_ms": "10"})
    assert r.status_code == 200
    assert "PTZ failed: GetCapabilities HTTP error" in r.text


def test_ptz_unknown_camera_is_404_html(client: TestClient) -> None:
    r = _post(client, "/onvif/cameras/nope/ptz", {"direction": "up"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "is not in config.yaml" in r.text


def test_plain_form_post_redirects_back_with_flash(
    client: TestClient, ptz_requests: list[httpx.Request]
) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz", {"direction": "stop"}, htmx=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/onvif?camera=front_door"
    assert "warden_flash" in r.headers.get("set-cookie", "")


# --- presets -------------------------------------------------------------------


def test_presets_fragment_get(client: TestClient) -> None:
    r = client.get("/onvif/cameras/front_door/presets", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert 'id="onvif-presets"' in r.text
    assert "<td>gate</td>" in r.text


def test_goto_preset_moves_camera(client: TestClient, ptz_requests: list[httpx.Request]) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz/goto", {"preset_name": "gate"})
    assert r.status_code == 200
    assert 'id="onvif-presets"' in r.text
    assert "Moved to preset &#39;gate&#39;." in r.text
    assert _ops(ptz_requests) == ["GetCapabilities", "GetProfiles", "AbsoluteMove"]
    assert 'x="0.2500"' in ptz_requests[2].content.decode()


def test_goto_unknown_preset_is_404_fragment(
    client: TestClient, ptz_requests: list[httpx.Request]
) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz/goto", {"preset_name": "nope"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "not found" in r.text
    assert ptz_requests == []


def test_save_preset_writes_raw_yaml_under_lock(
    client: TestClient, app_and_cfg: tuple[FastAPI, AppConfig], config_file: Path
) -> None:
    _, cfg = app_and_cfg
    r = _post(
        client,
        "/onvif/cameras/front_door/ptz/save",
        {"preset_name": "driveway", "pan": "0.5", "tilt": "-0.25", "zoom": "1"},
    )
    assert r.status_code == 200
    assert "Saved preset" in r.text
    assert "<td>driveway</td>" in r.text
    text = config_file.read_text(encoding="utf-8")
    assert "${CAM_USER}:${CAM_PASS}" in text
    assert LEAK not in text
    assert (config_file.parent / ".config.yaml.lock").exists()
    raw = yaml.safe_load(text)
    assert raw["cameras"][0]["presets"][-1] == {
        "name": "driveway",
        "pan": 0.5,
        "tilt": -0.25,
        "zoom": 1.0,
    }
    assert [p["name"] for p in raw["cameras"][0]["presets"]] == ["gate", "driveway"]
    assert "presets" not in raw["cameras"][1]
    assert "detectors" not in raw["cameras"][0]
    assert [p.name for p in cfg.cameras[0].presets] == ["gate", "driveway"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"preset_name": "x", "pan": "abc"}, "Pan must be a number."),
        ({"preset_name": "x", "pan": "5"}, "Pan must be between -1 and 1."),
        ({"preset_name": "x", "tilt": "nan"}, "Tilt must be between -1 and 1."),
        ({"preset_name": "x", "zoom": "-0.5"}, "Zoom must be between 0 and 1."),
        ({"preset_name": "   "}, "Preset name must be non-empty"),
    ],
)
def test_save_preset_bad_input_is_422_and_writes_nothing(
    client: TestClient,
    app_and_cfg: tuple[FastAPI, AppConfig],
    config_file: Path,
    data: dict[str, str],
    message: str,
) -> None:
    _, cfg = app_and_cfg
    r = _post(client, "/onvif/cameras/front_door/ptz/save", data)
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("text/html")
    assert 'id="onvif-presets"' in r.text
    assert message in r.text
    assert "<details open>" in r.text
    assert config_file.read_text(encoding="utf-8") == RAW_CONFIG
    assert [p.name for p in cfg.cameras[0].presets] == ["gate"]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (PermissionError(13, "Permission denied"), "Permission denied"),
        (ValueError("cameras is not a list"), "cameras is not a list"),
    ],
)
def test_save_preset_reports_unwritable_config(
    client: TestClient,
    app_and_cfg: tuple[FastAPI, AppConfig],
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
) -> None:
    _, cfg = app_and_cfg

    def refuse(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr("rtsp_warden.web.services.camera_config.patch_camera", refuse)
    r = _post(
        client,
        "/onvif/cameras/front_door/ptz/save",
        {"preset_name": "driveway", "pan": "0", "tilt": "0", "zoom": "0"},
    )
    assert r.status_code == 200
    assert f"Could not write {config_file}: {reason}." in r.text
    assert "The preset was not saved." in r.text
    assert [p.name for p in cfg.cameras[0].presets] == ["gate"]


def test_delete_preset_by_form_field(
    client: TestClient, app_and_cfg: tuple[FastAPI, AppConfig], config_file: Path
) -> None:
    _, cfg = app_and_cfg
    r = _post(client, "/onvif/cameras/front_door/ptz/delete", {"preset_name": "gate"})
    assert r.status_code == 200
    assert "Deleted preset" in r.text
    assert "No presets saved for this camera." in r.text
    text = config_file.read_text(encoding="utf-8")
    assert "${CAM_PASS}" in text
    assert LEAK not in text
    assert "presets" not in yaml.safe_load(text)["cameras"][0]
    assert cfg.cameras[0].presets == []


def test_preset_name_with_slash_can_be_saved_and_deleted(
    client: TestClient, app_and_cfg: tuple[FastAPI, AppConfig]
) -> None:
    _, cfg = app_and_cfg
    r = _post(
        client,
        "/onvif/cameras/front_door/ptz/save",
        {"preset_name": "gate/left", "pan": "0", "tilt": "0", "zoom": "0"},
    )
    assert r.status_code == 200
    r = _post(client, "/onvif/cameras/front_door/ptz/delete", {"preset_name": "gate/left"})
    assert r.status_code == 200
    assert "Deleted preset" in r.text
    assert [p.name for p in cfg.cameras[0].presets] == ["gate"]


def test_delete_unknown_preset_is_404_fragment(client: TestClient) -> None:
    r = _post(client, "/onvif/cameras/front_door/ptz/delete", {"preset_name": "nope"})
    assert r.status_code == 404
    assert 'id="onvif-presets"' in r.text
    assert "not found" in r.text


# --- discovery -----------------------------------------------------------------


class FakeDiscovery:
    """Stands in for OnvifDiscovery: no UDP socket."""

    result: list[DiscoveredCamera] = []
    error: Exception | None = None

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    def discover(self) -> list[DiscoveredCamera]:
        if FakeDiscovery.error is not None:
            raise FakeDiscovery.error
        return FakeDiscovery.result


@pytest.fixture
def fake_discovery(monkeypatch: pytest.MonkeyPatch) -> type[FakeDiscovery]:
    FakeDiscovery.result = [
        DiscoveredCamera(
            xaddr="http://192.0.2.20:888/onvif/device_service",
            address="192.0.2.20",
            name="Porch",
            manufacturer="Acme",
            model="Cam1",
            types=["dn:NetworkVideoTransmitter"],
        )
    ]
    FakeDiscovery.error = None
    monkeypatch.setattr(onvif_routes, "OnvifDiscovery", FakeDiscovery)
    return FakeDiscovery


def test_discover_returns_html_fragment(
    client: TestClient, fake_discovery: type[FakeDiscovery]
) -> None:
    r = _post(client, "/onvif/discover")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert not r.text.lstrip().startswith("{")
    assert "<table>" in r.text
    assert "<td>192.0.2.20</td>" in r.text
    assert "http://192.0.2.20:888/onvif/device_service" in r.text


def test_discover_disabled_and_failure_messages(
    client: TestClient,
    app_and_cfg: tuple[FastAPI, AppConfig],
    fake_discovery: type[FakeDiscovery],
) -> None:
    _, cfg = app_and_cfg
    fake_discovery.error = OnvifError("Failed to bind discovery socket: denied")
    r = _post(client, "/onvif/discover")
    assert r.status_code == 200
    assert "Discovery failed: Failed to bind discovery socket: denied" in r.text
    cfg.onvif.discovery_enabled = False
    r = _post(client, "/onvif/discover")
    assert "Discovery is off" in r.text


# --- event subscriptions -------------------------------------------------------


def test_subscribe_and_unsubscribe_return_fragments(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[FakeSubscriber] = []

    def factory(cfg: AppConfig, camera: CameraConfig) -> FakeSubscriber:
        sub = FakeSubscriber(camera.name)
        created.append(sub)
        return sub

    monkeypatch.setattr(onvif_routes, "_event_client", factory)

    r = _post(client, "/onvif/cameras/front_door/events/subscribe")
    assert r.status_code == 200
    assert 'id="onvif-subscriptions"' in r.text
    assert "Subscribed to events from" in r.text
    assert 'hx-post="/onvif/cameras/front_door/events/unsubscribe"' in r.text
    assert "front_door" in onvif_events.get_active_subscribers()

    r = _post(client, "/onvif/cameras/front_door/events/subscribe")
    assert "Already subscribed" in r.text
    assert len(created) == 1

    r = _post(client, "/onvif/cameras/front_door/events/unsubscribe")
    assert r.status_code == 200
    assert "Unsubscribed" in r.text
    assert created[0].stopped is True
    assert onvif_events.get_active_subscribers() == {}
    assert 'hx-post="/onvif/cameras/front_door/events/subscribe"' in r.text


def test_unsubscribe_without_subscription_is_404_fragment(client: TestClient) -> None:
    r = _post(client, "/onvif/cameras/front_door/events/unsubscribe")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "No active subscription" in r.text


def test_events_disabled_notice_does_not_build_subscriber(
    client: TestClient,
    app_and_cfg: tuple[FastAPI, AppConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cfg = app_and_cfg
    cfg.onvif.events_enabled = False

    def factory(cfg: AppConfig, camera: CameraConfig) -> FakeSubscriber:
        raise AssertionError("subscriber must not be built while events are off")

    monkeypatch.setattr(onvif_routes, "_event_client", factory)
    r = _post(client, "/onvif/cameras/front_door/events/subscribe")
    assert r.status_code == 200
    assert "ONVIF events are off" in r.text


def test_events_status_is_html_fragment(client: TestClient) -> None:
    r = client.get("/onvif/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<td>garage</td>" in r.text
    assert "not subscribed" in r.text


# --- client-side contract ------------------------------------------------------


def test_warden_js_swaps_html_error_fragments() -> None:
    js = (Path(STATIC_DIR) / "js" / "warden.js").read_text(encoding="utf-8")
    assert "htmx:beforeSwap" in js
    assert 'contentType.indexOf("text/html") === 0' in js
    assert "evt.detail.shouldSwap = true;" in js
