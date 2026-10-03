"""Add-camera flow: form page, connection test, ONVIF URL fill, save + hot-add (RW-2 Task 7)."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote, urlsplit

import pytest
import yaml
from fastapi.testclient import TestClient
from markupsafe import escape

from rtsp_warden import probe
from rtsp_warden.auth import hash_password
from rtsp_warden.config import AppConfig, expand_env
from rtsp_warden.db.schema import create_user
from rtsp_warden.onvif import media as onvif_media
from rtsp_warden.onvif.discovery import OnvifError
from rtsp_warden.onvif.media import StreamUris
from rtsp_warden.probe import ProbeResult
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes import camera_edit
from rtsp_warden.web.services.camera_config import validate_name

JPEG = b"\xff\xd8\xff\xd9"
HOST = "192.0.2.10"
RAW_CONFIG = {
    "cameras": [
        {
            "name": "porch",
            "main_url": "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.5:554/live",
            "proxy": {"enabled": True, "mode": "mjpeg", "stream": "main", "port": 9001},
        }
    ]
}


def _login(client: TestClient, username: str = "admin", password: str = "testpass123") -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post("/login", data={"username": username, "password": password, "csrf_token": token})


def _post(client: TestClient, url: str, data: dict[str, str]):
    token = client.cookies.get("warden_csrf", "")
    return client.post(url, data=data, headers={"X-CSRF-Token": token}, follow_redirects=False)


def _form(**overrides: str) -> dict[str, str]:
    data = {
        "name": "front",
        "host": HOST,
        "username": "u",
        "password": "s3cret",
        "main_url": f"rtsp://{HOST}:554/videoMain",
        "sub_url": "",
        "onvif_port": "",
        "record_enabled": "on",
    }
    data.update(overrides)
    return data


def _ok_result() -> ProbeResult:
    return ProbeResult(
        ok=True, codec="h264", width=1280, height=720, fps=15.0, snapshot_jpeg=JPEG, error=None
    )


def _must_not_run(*args, **kwargs):
    raise AssertionError("must not be called for invalid input")


@pytest.fixture(autouse=True)
def _forget_camera_env():
    """The save route sets CAM_<SLUG>_* in os.environ; drop them after each test."""
    before = {key for key in os.environ if key.startswith("CAM_")}
    yield
    for key in [k for k in os.environ if k.startswith("CAM_") and k not in before]:
        del os.environ[key]


@pytest.fixture(autouse=True)
def _all_ports_free(monkeypatch):
    """Proxy port allocation must not depend on what is bound on the test machine."""
    monkeypatch.setattr("rtsp_warden.ports.port_is_free", lambda host, port: True)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "conf" / "config.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(RAW_CONFIG, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def cfg() -> AppConfig:
    # Like load_config: the in-memory config is expanded, the file keeps ${VAR} text.
    return AppConfig.model_validate(expand_env(RAW_CONFIG, {"CAM_USER": "u", "CAM_PASS": "p"}))


@pytest.fixture
def runtime() -> MagicMock:
    rt = MagicMock()
    done: concurrent.futures.Future = concurrent.futures.Future()
    done.set_result(None)
    rt.request_add_camera.return_value = done
    return rt


@pytest.fixture
def app(db_with_user, cfg, config_path, runtime):
    return create_app(
        WebSettings(),
        cfg=cfg,
        runtime_provider=lambda: None,
        config_path=config_path,
        runtime=runtime,
    )


@pytest.fixture
def client(app) -> TestClient:
    c = TestClient(app)
    _login(c)
    return c


@pytest.fixture
def viewer(app) -> TestClient:
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    c = TestClient(app)
    _login(c, "viewer", "viewerpass123")
    return c


# --- form page ---------------------------------------------------------------------------


def test_new_form_renders_every_field(client):
    r = client.get("/cameras/new")
    assert r.status_code == 200
    assert "Add camera" in r.text
    for needle in (
        'action="/cameras"',
        'name="csrf_token"',
        'name="name"',
        'name="host"',
        'name="username"',
        'name="password"',
        'type="password"',
        'id="main_url"',
        'id="sub_url"',
        'id="onvif_port"',
        'name="record_enabled"',
        'hx-post="/cameras/new/test"',
        'hx-post="/cameras/new/onvif"',
        'id="probe-result"',
        'id="onvif-result"',
    ):
        assert needle in r.text, needle


def test_new_form_is_not_shadowed_by_camera_detail(client):
    # GET /cameras/{name} would answer 404 "Camera 'new' not found" if it matched first.
    r = client.get("/cameras/new")
    assert r.status_code == 200
    assert 'id="camera-form"' in r.text
    assert "not found" not in r.text


def test_new_form_requires_admin(viewer):
    assert viewer.get("/cameras/new").status_code == 403


# --- test connection ---------------------------------------------------------------------


def test_connection_test_shows_codec_resolution_fps_and_snapshot(client, monkeypatch):
    calls: list[str] = []

    def fake_probe(url, *, runtime, timeout_s=15.0, snapshot=True):
        calls.append(url)
        return _ok_result()

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    r = _post(client, "/cameras/new/test", {**_form(main_url="/videoMain"), "stream": "main"})
    assert r.status_code == 200
    assert "<html" not in r.text
    assert "h264" in r.text
    assert "1280x720" in r.text
    assert "15 fps" in r.text
    assert f'src="data:image/jpeg;base64,{base64.b64encode(JPEG).decode()}"' in r.text
    assert calls == [f"rtsp://u:s3cret@{HOST}:554/videoMain"]
    assert f"rtsp://***:***@{HOST}:554/videoMain" in r.text
    assert "s3cret" not in r.text


def test_connection_test_runs_in_the_threadpool(client, monkeypatch):
    seen: dict[str, bool] = {}

    def fake_probe(url, *, runtime, timeout_s=15.0, snapshot=True):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return _ok_result()

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    r = _post(client, "/cameras/new/test", {**_form(), "stream": "main"})
    assert r.status_code == 200
    assert seen == {"on_loop": False}


def test_connection_test_of_the_sub_stream(client, monkeypatch):
    calls: list[str] = []

    def fake_probe(url, *, runtime, timeout_s=15.0, snapshot=True):
        calls.append(url)
        return _ok_result()

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    r = _post(client, "/cameras/new/test", {**_form(sub_url="/videoSub"), "stream": "sub"})
    assert r.status_code == 200
    assert calls == [f"rtsp://u:s3cret@{HOST}:554/videoSub"]

    monkeypatch.setattr(probe, "probe_stream", _must_not_run)
    r = _post(client, "/cameras/new/test", {**_form(sub_url=""), "stream": "sub"})
    assert r.status_code == 200
    assert "Enter a sub stream path or URL first." in r.text


def test_connection_test_without_credentials_sends_no_userinfo(client, monkeypatch):
    calls: list[str] = []

    def fake_probe(url, *, runtime, timeout_s=15.0, snapshot=True):
        calls.append(url)
        return _ok_result()

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    form = {**_form(username="", password=""), "stream": "main"}
    assert _post(client, "/cameras/new/test", form).status_code == 200
    assert calls == [f"rtsp://{HOST}:554/videoMain"]


def test_connection_test_shows_a_snapshot_failure_note(client, monkeypatch):
    no_snapshot = ProbeResult(
        ok=True,
        codec="h264",
        width=1280,
        height=720,
        fps=15.0,
        snapshot_jpeg=None,
        error="snapshot failed: timed out after 15 s",
    )
    monkeypatch.setattr(probe, "probe_stream", lambda url, **kwargs: no_snapshot)
    r = _post(client, "/cameras/new/test", {**_form(), "stream": "main"})
    assert r.status_code == 200
    assert "1280x720" in r.text
    assert "snapshot failed: timed out after 15 s" in r.text
    assert "data:image/jpeg" not in r.text


def test_connection_test_percent_encodes_reserved_password_characters(client, monkeypatch):
    """(review focus) '/', '#', '?', '@' and '%' in a password must not break the URL."""
    password = "p/w#?@%x"
    calls: list[str] = []

    def fake_probe(url, *, runtime, timeout_s=15.0, snapshot=True):
        calls.append(url)
        return _ok_result()

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    r = _post(client, "/cameras/new/test", {**_form(password=password), "stream": "main"})
    assert r.status_code == 200
    parts = urlsplit(calls[0])
    assert parts.hostname == HOST
    assert parts.port == 554
    assert parts.path == "/videoMain"
    assert unquote(parts.username) == "u"
    assert unquote(parts.password) == password
    assert password not in r.text
    assert parts.password not in r.text


@pytest.mark.parametrize(
    "main_url",
    [
        "http://192.0.2.10/x",
        "file:///etc/passwd",
        "-i",
        "rtsp://a:b@192.0.2.10/x",
    ],
)
def test_connection_test_rejects_unsafe_urls_without_running_ffprobe(client, monkeypatch, main_url):
    monkeypatch.setattr(probe, "probe_stream", _must_not_run)
    r = _post(client, "/cameras/new/test", {**_form(main_url=main_url), "stream": "main"})
    assert r.status_code == 200
    assert "probe-failed" in r.text


def test_connection_test_shows_the_probe_error(client, monkeypatch):
    failed = ProbeResult(
        ok=False,
        codec=None,
        width=None,
        height=None,
        fps=None,
        snapshot_jpeg=None,
        error="Server returned 401 Unauthorized (authorization failed)",
    )
    monkeypatch.setattr(probe, "probe_stream", lambda url, **kwargs: failed)
    r = _post(client, "/cameras/new/test", {**_form(), "stream": "main"})
    assert r.status_code == 200
    assert "401 Unauthorized" in r.text
    assert "data:image/jpeg" not in r.text


def test_connection_test_reports_a_missing_ffprobe(client, monkeypatch):
    def fake_probe(url, **kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(probe, "probe_stream", fake_probe)
    r = _post(client, "/cameras/new/test", {**_form(), "stream": "main"})
    assert r.status_code == 200
    assert "ffprobe was not found" in r.text


# --- ONVIF fill --------------------------------------------------------------------------


def _fake_discover(calls: list, *, sub: bool = True):
    async def fake(
        host, username, password, *, ports=(80, 8080, 888, 2020), timeout_s=3.0, transport=None
    ):
        calls.append((host, username, password, tuple(ports), timeout_s))
        return StreamUris(
            host=host,
            port=888,
            main=f"rtsp://{host}:554/videoMain",
            sub=f"rtsp://{host}:554/videoSub" if sub else None,
            profiles=["p0", "p1"] if sub else ["p0"],
        )

    return fake


def test_onvif_fill_returns_out_of_band_inputs(client, monkeypatch):
    calls: list = []
    monkeypatch.setattr(onvif_media, "discover_stream_uris", _fake_discover(calls))
    r = _post(client, "/cameras/new/onvif", _form())
    assert r.status_code == 200
    assert calls == [(HOST, "u", "s3cret", (80, 8080, 888, 2020), 3.0)]
    assert r.text.count('hx-swap-oob="true"') == 3
    assert 'id="main_url"' in r.text
    assert f'value="rtsp://{HOST}:554/videoMain"' in r.text
    assert 'id="sub_url"' in r.text
    assert f'value="rtsp://{HOST}:554/videoSub"' in r.text
    assert 'id="onvif_port"' in r.text
    assert 'value="888"' in r.text
    assert "s3cret" not in r.text


def test_onvif_fill_with_one_profile_leaves_the_sub_stream_alone(client, monkeypatch):
    calls: list = []
    monkeypatch.setattr(onvif_media, "discover_stream_uris", _fake_discover(calls, sub=False))
    r = _post(client, "/cameras/new/onvif", _form())
    assert r.status_code == 200
    assert r.text.count('hx-swap-oob="true"') == 2
    assert 'id="sub_url"' not in r.text
    assert "only one stream" in r.text


def test_onvif_fill_uses_only_the_typed_port(client, monkeypatch):
    calls: list = []
    monkeypatch.setattr(onvif_media, "discover_stream_uris", _fake_discover(calls))
    _post(client, "/cameras/new/onvif", _form(onvif_port="888"))
    assert calls[0][3] == (888,)


def test_onvif_fill_failure_names_the_ports_tried(client, monkeypatch):
    """(review focus) bounded: 4 ports x 3 s, and the fragment lists the ports tried."""
    calls: list = []

    async def fake(
        host, username, password, *, ports=(80, 8080, 888, 2020), timeout_s=3.0, transport=None
    ):
        calls.append((tuple(ports), timeout_s))
        raise OnvifError("no ONVIF service answered")

    monkeypatch.setattr(onvif_media, "discover_stream_uris", fake)
    r = _post(client, "/cameras/new/onvif", _form())
    assert r.status_code == 200
    assert calls == [((80, 8080, 888, 2020), 3.0)]
    assert len(camera_edit.ONVIF_PORTS) * camera_edit.ONVIF_TIMEOUT_S <= 12.0
    assert "no ONVIF service answered" in r.text
    assert "80, 8080, 888, 2020" in r.text
    assert "hx-swap-oob" not in r.text


def test_onvif_fill_rejects_a_bad_host_without_network(client, monkeypatch):
    monkeypatch.setattr(onvif_media, "discover_stream_uris", _must_not_run)
    r = _post(client, "/cameras/new/onvif", _form(host="rtsp://192.0.2.10"))
    assert r.status_code == 200
    assert "IP address or host name" in r.text


# --- save --------------------------------------------------------------------------------


def test_save_writes_config_and_env_and_hot_adds(client, cfg, config_path, runtime):
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"

    text = config_path.read_text(encoding="utf-8")
    assert "s3cret" not in text
    data = yaml.safe_load(text)
    assert [c["name"] for c in data["cameras"]] == ["porch", "front"]
    assert data["cameras"][0]["main_url"] == "rtsp://${CAM_USER}:${CAM_PASS}@192.0.2.5:554/live"
    entry = data["cameras"][1]
    assert (
        entry["main_url"] == f"rtsp://${{CAM_FRONT_USER}}:${{CAM_FRONT_PASS}}@{HOST}:554/videoMain"
    )
    assert "sub_url" not in entry
    assert entry["proxy"]["port"] == 9002
    assert entry["record"]["enabled"] is True

    env_path = config_path.parent / ".env"
    env_text = env_path.read_text(encoding="utf-8")
    assert 'CAM_FRONT_USER="u"' in env_text
    assert 'CAM_FRONT_PASS="s3cret"' in env_text
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    assert [c.name for c in cfg.cameras] == ["porch", "front"]
    added = cfg.cameras[1]
    assert added.main_url == f"rtsp://u:s3cret@{HOST}:554/videoMain"
    runtime.request_add_camera.assert_called_once_with(added)
    assert os.environ["CAM_FRONT_PASS"] == "s3cret"


def test_save_flashes_success_once(client):
    _post(client, "/cameras", _form())
    assert "Camera front added and started." in client.get("/cameras/new").text
    assert "Camera front added and started." not in client.get("/cameras/new").text


def test_save_with_sub_stream_and_onvif_port(client, cfg, config_path):
    r = _post(client, "/cameras", _form(sub_url="/videoSub", onvif_port="888"))
    assert r.status_code == 303
    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cameras"][1]
    assert entry["sub_url"] == f"rtsp://${{CAM_FRONT_USER}}:${{CAM_FRONT_PASS}}@{HOST}:554/videoSub"
    assert entry["onvif_port"] == 888
    assert cfg.cameras[1].sub_url == f"rtsp://u:s3cret@{HOST}:554/videoSub"
    assert cfg.cameras[1].onvif_port == 888


def test_save_with_empty_main_stream_uses_the_host_root(client, config_path):
    r = _post(client, "/cameras", _form(main_url=""))
    assert r.status_code == 303
    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cameras"][1]
    assert entry["main_url"] == f"rtsp://${{CAM_FRONT_USER}}:${{CAM_FRONT_PASS}}@{HOST}:554/"


def test_save_without_credentials_writes_no_env_references(client, cfg, config_path):
    r = _post(client, "/cameras", _form(username="", password=""))
    assert r.status_code == 303
    entry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["cameras"][1]
    assert entry["main_url"] == f"rtsp://{HOST}:554/videoMain"
    assert cfg.cameras[1].main_url == f"rtsp://{HOST}:554/videoMain"
    env_path = config_path.parent / ".env"
    assert not env_path.exists() or "CAM_FRONT" not in env_path.read_text()


def test_save_stores_a_percent_encoded_password(client, cfg, config_path):
    """(review focus) a password with '/', '#', '?', '@', '%' survives save and expansion."""
    password = "p/w#?@%x"
    encoded = "p%2Fw%23%3F%40%25x"
    r = _post(client, "/cameras", _form(password=password))
    assert r.status_code == 303
    text = config_path.read_text(encoding="utf-8")
    assert password not in text
    assert encoded not in text
    assert f'CAM_FRONT_PASS="{encoded}"' in (config_path.parent / ".env").read_text()
    parts = urlsplit(cfg.cameras[1].main_url)
    assert parts.hostname == HOST
    assert parts.port == 554
    assert parts.path == "/videoMain"
    assert unquote(parts.password) == password


def test_save_without_runtime_writes_files_and_says_restart(client, app, cfg, config_path):
    """(review focus) runtime None: config.yaml and .env are written, the flash says restart."""
    app.state.runtime = None
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert yaml.safe_load(config_path.read_text())["cameras"][1]["name"] == "front"
    assert 'CAM_FRONT_PASS="s3cret"' in (config_path.parent / ".env").read_text()
    assert [c.name for c in cfg.cameras] == ["porch", "front"]
    assert "takes effect when rtsp-warden restarts" in client.get("/cameras/new").text


def test_save_rejects_an_existing_name_in_another_case(client, cfg, config_path, runtime):
    """(review focus) 'PORCH' collides with 'porch'."""
    before = config_path.read_text()
    r = _post(client, "/cameras", _form(name="PORCH"))
    assert r.status_code == 422
    with pytest.raises(ValueError) as exc_info:
        validate_name("PORCH", ["porch"])
    assert str(escape(str(exc_info.value))) in r.text
    assert 'value="PORCH"' in r.text
    assert "s3cret" not in r.text
    assert config_path.read_text() == before
    assert not (config_path.parent / ".env").exists()
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()


@pytest.mark.parametrize("name", ["new", "bad name", "../up", ""])
def test_save_rejects_invalid_names(client, config_path, name):
    before = config_path.read_text()
    r = _post(client, "/cameras", _form(name=name))
    assert r.status_code == 422
    assert 'id="name-error"' in r.text
    assert config_path.read_text() == before


@pytest.mark.parametrize(
    ("field", "value", "error_id"),
    [
        ("host", "", "host-error"),
        ("host", "rtsp://192.0.2.10", "host-error"),
        ("host", "192.0.2.10:554", "host-error"),
        ("main_url", "rtsp://a:b@192.0.2.10/x", "main_url-error"),
        ("main_url", "http://192.0.2.10/x", "main_url-error"),
        ("sub_url", "-i", "sub_url-error"),
        ("onvif_port", "70000", "onvif_port-error"),
    ],
)
def test_save_rejects_bad_fields(client, config_path, field, value, error_id):
    before = config_path.read_text()
    r = _post(client, "/cameras", _form(**{field: value}))
    assert r.status_code == 422
    assert f'id="{error_id}"' in r.text
    assert config_path.read_text() == before


def test_save_without_a_config_file_is_refused(client, app, cfg, runtime):
    app.state.config_path = None
    r = _post(client, "/cameras", _form())
    assert r.status_code == 503
    assert "without a config file" in r.text
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()


def _make_read_only(directory: Path) -> None:
    """chmod 0500, or skip when this user can still write there (root, CAP_DAC_OVERRIDE)."""
    directory.chmod(0o500)
    probe_file = directory / ".probe"
    try:
        probe_file.touch()
    except PermissionError:
        return
    probe_file.unlink()
    directory.chmod(0o700)
    pytest.skip("directory is still writable for this user")


def test_save_into_a_read_only_config_dir_flashes_an_error(client, cfg, config_path, runtime):
    before = config_path.read_text()
    _make_read_only(config_path.parent)
    try:
        r = _post(client, "/cameras", _form())
    finally:
        config_path.parent.chmod(0o700)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/new"
    assert config_path.read_text() == before
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()
    page = client.get("/cameras/new").text
    assert "Could not save camera front" in page
    assert str(config_path.parent) in page


def test_save_reports_a_start_failure_without_credentials(client, runtime, config_path):
    failed: concurrent.futures.Future = concurrent.futures.Future()
    failed.set_exception(OSError(f"cannot open rtsp://u:s3cret@{HOST}:554/videoMain"))
    runtime.request_add_camera.return_value = failed
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"
    assert yaml.safe_load(config_path.read_text())["cameras"][1]["name"] == "front"
    page = client.get("/cameras/new").text
    assert "Camera front was saved but did not start" in page
    assert "s3cret" not in page


def test_save_does_not_wait_forever_for_the_runtime(client, runtime, monkeypatch):
    runtime.request_add_camera.return_value = concurrent.futures.Future()  # never resolves
    monkeypatch.setattr(camera_edit, "ADD_TIMEOUT_S", 0.05)
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert "still starting it" in client.get("/cameras/new").text


def test_add_post_routes_require_admin(viewer):
    assert _post(viewer, "/cameras/new/test", _form()).status_code == 403
    assert _post(viewer, "/cameras/new/onvif", _form()).status_code == 403
    assert _post(viewer, "/cameras", _form()).status_code == 403


def _refuse(*args, **kwargs):
    raise PermissionError(13, "Permission denied", "/etc/rtsp-warden/.env")


def test_save_write_error_flashes_and_changes_nothing(
    client, cfg, config_path, runtime, monkeypatch
):
    """R9 for every user (root too): a failing .env write is a flash, never a 500."""
    monkeypatch.setattr(camera_edit, "upsert_env_vars", _refuse)
    before = config_path.read_text()
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/new"
    assert config_path.read_text() == before
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()
    assert "Could not save camera front" in client.get("/cameras/new").text


def test_save_config_write_error_flashes_and_changes_nothing(
    client, cfg, config_path, runtime, monkeypatch
):
    monkeypatch.setattr(camera_edit.camera_config, "append_camera", _refuse)
    before = config_path.read_text()
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/new"
    assert config_path.read_text() == before
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()
    assert "CAM_FRONT_PASS" not in os.environ


def test_save_refuses_to_overwrite_existing_env_keys(client, cfg, config_path, runtime):
    """A hand-written camera may already use CAM_FRONT_* in the shared .env."""
    env_path = config_path.parent / ".env"
    env_path.write_text('CAM_FRONT_USER="x"\nCAM_FRONT_PASS="y"\n', encoding="utf-8")
    before = config_path.read_text()
    r = _post(client, "/cameras", _form())
    assert r.status_code == 422
    assert "already defines CAM_FRONT_USER" in r.text
    assert env_path.read_text(encoding="utf-8") == 'CAM_FRONT_USER="x"\nCAM_FRONT_PASS="y"\n'
    assert config_path.read_text() == before
    assert [c.name for c in cfg.cameras] == ["porch"]
    runtime.request_add_camera.assert_not_called()


def test_save_allows_env_keys_that_already_hold_the_same_values(client, config_path):
    """A retry after a failed save finds its own values in .env; that is not a conflict."""
    env_path = config_path.parent / ".env"
    env_path.write_text('CAM_FRONT_USER="u"\nCAM_FRONT_PASS="s3cret"\n', encoding="utf-8")
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"


def test_save_refuses_keys_set_in_the_process_environment(client, config_path, monkeypatch):
    monkeypatch.setenv("CAM_FRONT_PASS", "from-the-service-environment")
    r = _post(client, "/cameras", _form())
    assert r.status_code == 422
    assert "already defines CAM_FRONT_PASS" in r.text
    assert not (config_path.parent / ".env").exists()


def test_save_refuses_a_camera_added_to_the_file_by_hand(client, cfg, config_path, runtime):
    """config.yaml may hold a camera added by hand since startup (not in cfg yet)."""
    raw = yaml.safe_load(config_path.read_text())
    raw["cameras"].append({"name": "Front", "main_url": "rtsp://192.0.2.9/x"})
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    before = config_path.read_text()
    r = _post(client, "/cameras", _form())
    assert r.status_code == 422
    assert 'id="name-error"' in r.text
    assert config_path.read_text() == before
    assert not (config_path.parent / ".env").exists()
    runtime.request_add_camera.assert_not_called()


def test_save_reports_a_config_file_that_is_not_valid_yaml(client, config_path, runtime):
    config_path.write_text("cameras: [\n", encoding="utf-8")
    r = _post(client, "/cameras", _form())
    assert r.status_code == 422
    assert "could not be read" in r.text
    assert not (config_path.parent / ".env").exists()
    runtime.request_add_camera.assert_not_called()


def test_save_reports_a_start_that_exits(client, runtime):
    """RW-0 fails the future with SystemExit for an unknown proxy mode; never a 500."""
    failed: concurrent.futures.Future = concurrent.futures.Future()
    failed.set_exception(SystemExit("unknown proxy mode"))
    runtime.request_add_camera.return_value = failed
    r = _post(client, "/cameras", _form())
    assert r.status_code == 303
    assert "Camera front was saved but did not start" in client.get("/cameras/new").text


# --- camera list -------------------------------------------------------------------------


def test_camera_list_shows_add_button_to_admins_only(client, viewer):
    assert 'href="/cameras/new"' in client.get("/cameras").text
    r = viewer.get("/cameras")
    assert r.status_code == 200
    assert 'href="/cameras/new"' not in r.text
