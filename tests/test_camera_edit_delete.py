"""Edit and delete a camera from its detail page (RW-2 Task 8).

Offline: config.yaml and .env live in a temp directory, the runtime is a MagicMock
whose request_* methods return concurrent.futures.Future objects, and nothing starts
ffmpeg or talks to a camera.
"""

from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote, urlsplit

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.app import CameraNotFoundError
from rtsp_warden.auth import hash_password
from rtsp_warden.config import AppConfig, load_config
from rtsp_warden.db.schema import create_user
from rtsp_warden.onvif.events import (
    get_active_subscribers,
    register_subscriber,
    unregister_subscriber,
)
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes import camera_edit

FRONT_MAIN = "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain"
FRONT_SUB = "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoSub"
ENV_TEXT = 'CAM_FRONT_USER="admin"\nCAM_FRONT_PASS="old%2Fpass"\n'


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


def _refuse(*args, **kwargs):
    raise PermissionError(13, "Permission denied", "/etc/rtsp-warden/config.yaml")


def _flash(html: str) -> str:
    """The one-shot flash line of a page ('' when there is none)."""
    match = re.search(r'<div id="flash" class="flash flash-(\w+)"[^>]*>(.*?)</div>', html, re.S)
    return f"{match.group(1)}: {match.group(2)}" if match else ""


def _done() -> Future:
    fut: Future = Future()
    fut.set_result(None)
    return fut


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """config.yaml and .env in their own directory (the test DB stays in tmp_path)."""
    monkeypatch.setenv("CAM_FRONT_USER", "admin")
    monkeypatch.setenv("CAM_FRONT_PASS", "old%2Fpass")
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    recordings = tmp_path / "recordings"
    data = {
        "cameras": [
            {
                "name": "front",
                "main_url": FRONT_MAIN,
                "sub_url": FRONT_SUB,
                "record": {"enabled": True, "output_dir": str(recordings)},
                "proxy": {"enabled": False, "mode": "mjpeg", "port": 9001},
                "sensitivity": 65,
            },
            {
                "name": "back",
                "main_url": "rtsp://u:p@192.0.2.11/m",
                "record": {"enabled": False, "output_dir": str(recordings)},
                "proxy": {"enabled": False, "mode": "mjpeg", "port": 9002},
            },
        ],
        "runtime": {"ffmpeg_path": "ffmpeg", "workspace_dir": str(tmp_path / "workspace")},
    }
    path = conf_dir / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (conf_dir / ".env").write_text(ENV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def runtime() -> MagicMock:
    rt = MagicMock()
    rt.request_restart_camera.return_value = _done()
    rt.request_remove_camera.return_value = _done()
    return rt


def _login(client: TestClient, username: str = "admin", password: str = "testpass123") -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post("/login", data={"username": username, "password": password, "csrf_token": token})


def _client(
    config_file: Path,
    runtime: object | None,
    username: str = "admin",
    password: str = "testpass123",
) -> TestClient:
    cfg = load_config(config_file)
    app = create_app(
        WebSettings(),
        cfg=cfg,
        runtime_provider=lambda: None,
        config_path=config_file,
        runtime=runtime,
    )
    client = TestClient(app)
    _login(client, username, password)
    return client


@pytest.fixture
def client(db_with_user: str, config_file: Path, runtime: MagicMock) -> TestClient:
    return _client(config_file, runtime)


@pytest.fixture
def viewer_client(db_with_user: str, config_file: Path, runtime: MagicMock) -> TestClient:
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    return _client(config_file, runtime, "viewer", "viewerpass123")


def _cfg(client: TestClient) -> AppConfig:
    return client.app.state.cfg


def _post(client: TestClient, url: str, data: dict[str, str] | None = None):
    form = dict(data or {})
    form["csrf_token"] = client.cookies.get("warden_csrf", "")
    return client.post(url, data=form, follow_redirects=False)


def _form(**overrides: str | None) -> dict[str, str]:
    """The edit form as the browser submits it unchanged; None drops a field (unchecked box)."""
    data: dict[str, str | None] = {
        "main_url": "rtsp://192.0.2.10:554/videoMain",
        "sub_url": "rtsp://192.0.2.10:554/videoSub",
        "username": "admin",
        "password": "",
        "onvif_port": "",
        "record_enabled": "on",
    }
    data.update(overrides)
    return {key: value for key, value in data.items() if value is not None}


def _raw(config_file: Path, name: str) -> dict:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    return next(c for c in data["cameras"] if c["name"] == name)


# --- edit form -------------------------------------------------------------------------------


def test_edit_form_is_prefilled_from_raw_yaml_without_secrets(client: TestClient) -> None:
    r = client.get("/cameras/front/edit")
    assert r.status_code == 200
    html = r.text
    assert 'action="/cameras/front/edit"' in html
    assert 'value="rtsp://192.0.2.10:554/videoMain"' in html
    assert 'value="rtsp://192.0.2.10:554/videoSub"' in html
    assert 'id="username" name="username" type="text" autocomplete="off" value="admin"' in html
    assert 'id="password" name="password" type="password" autocomplete="new-password"' in html
    assert 'autocomplete="new-password" value=""' in html
    assert "Preview port: 9001" in html
    assert "Save changes" in html
    assert "CAM_FRONT" not in html
    assert "old%2Fpass" not in html
    assert "old/pass" not in html
    # The name is read-only, and the add-only controls are not on the edit page.
    assert 'name="name"' not in html
    assert 'name="host"' not in html
    assert 'hx-post="/cameras/new/test"' not in html
    assert 'hx-post="/cameras/new/onvif"' not in html


def test_add_form_is_unchanged_by_the_edit_mode(client: TestClient) -> None:
    html = client.get("/cameras/new").text
    assert 'name="name"' in html
    assert 'name="host"' in html
    assert 'hx-post="/cameras/new/onvif"' in html
    assert '<button type="submit">Save camera</button>' in html
    assert '<a href="/cameras" role="button" class="secondary outline">Cancel</a>' in html


def test_edit_unknown_camera_is_404(client: TestClient) -> None:
    assert client.get("/cameras/nope/edit").status_code == 404
    assert _post(client, "/cameras/nope/edit", _form()).status_code == 404


def test_edit_requires_admin(viewer_client: TestClient, config_file: Path) -> None:
    before = config_file.read_text(encoding="utf-8")
    assert viewer_client.get("/cameras/front/edit").status_code == 403
    r = _post(viewer_client, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 403
    assert config_file.read_text(encoding="utf-8") == before


def test_settings_page_redirects_to_the_edit_form(client: TestClient) -> None:
    r = client.get("/cameras/front/settings")
    assert r.status_code == 200
    assert 'action="/cameras/front/edit"' in r.text


# --- edit save -------------------------------------------------------------------------------


def test_edit_main_url_patches_only_that_key_and_restarts(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    env_before = (config_file.parent / ".env").read_text(encoding="utf-8")
    back_before = _raw(config_file, "back")
    front_before = _raw(config_file, "front")

    r = _post(client, "/cameras/front/edit", _form(main_url="rtsp://192.0.2.10:554/videoMain2"))

    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"
    front = _raw(config_file, "front")
    assert front["main_url"] == (
        "rtsp://${CAM_FRONT_USER}:${CAM_FRONT_PASS}@192.0.2.10:554/videoMain2"
    )
    assert {k: v for k, v in front.items() if k != "main_url"} == {
        k: v for k, v in front_before.items() if k != "main_url"
    }
    assert _raw(config_file, "back") == back_before
    assert (config_file.parent / ".env").read_text(encoding="utf-8") == env_before
    runtime.request_restart_camera.assert_called_once_with("front")
    assert _cfg(client).cameras[0].main_url == "rtsp://admin:old%2Fpass@192.0.2.10:554/videoMain2"
    assert "Camera front saved. Its stream was restarted." in client.get("/cameras/front").text


def test_unchanged_form_writes_nothing(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    before = config_file.read_text(encoding="utf-8")
    r = _post(client, "/cameras/front/edit", _form())
    assert r.status_code == 303
    assert config_file.read_text(encoding="utf-8") == before
    runtime.request_restart_camera.assert_not_called()
    assert "Nothing changed." in client.get("/cameras/front").text


def test_blank_password_keeps_env_file_and_record_toggle_restarts(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    env_before = (config_file.parent / ".env").read_text(encoding="utf-8")

    r = _post(client, "/cameras/front/edit", _form(record_enabled=None))

    assert r.status_code == 303
    record = _raw(config_file, "front")["record"]
    assert record["enabled"] is False
    assert record["output_dir"] == str(config_file.parent.parent / "recordings")
    assert (config_file.parent / ".env").read_text(encoding="utf-8") == env_before
    assert os.environ["CAM_FRONT_PASS"] == "old%2Fpass"
    assert _cfg(client).cameras[0].record.enabled is False
    runtime.request_restart_camera.assert_called_once_with("front")


def test_new_password_with_reserved_characters_goes_to_env_file_only(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    """(review focus) '/', '#', '?', '@' and '%' survive, encoded, and never reach config.yaml."""
    secret = "n3w/p#ss?@%"

    r = _post(client, "/cameras/front/edit", _form(password=secret))

    assert r.status_code == 303
    assert "n3w" not in config_file.read_text(encoding="utf-8")
    assert _raw(config_file, "front")["main_url"] == FRONT_MAIN
    env_text = (config_file.parent / ".env").read_text(encoding="utf-8")
    assert 'CAM_FRONT_PASS="n3w%2Fp%23ss%3F%40%25"' in env_text
    assert 'CAM_FRONT_USER="admin"' in env_text
    assert "old%2Fpass" not in env_text
    assert os.environ["CAM_FRONT_PASS"] == "n3w%2Fp%23ss%3F%40%25"
    parts = urlsplit(_cfg(client).cameras[0].main_url)
    assert (parts.hostname, parts.port, unquote(parts.password or "")) == (
        "192.0.2.10",
        554,
        secret,
    )
    runtime.request_restart_camera.assert_called_once_with("front")
    page = client.get("/cameras/front").text
    assert "Camera front saved." in page
    assert "n3w" not in page


def test_clearing_sub_url_removes_the_key(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    r = _post(client, "/cameras/front/edit", _form(sub_url=""))
    assert r.status_code == 303
    assert "sub_url" not in _raw(config_file, "front")
    cam = _cfg(client).cameras[0]
    assert cam.sub_url is None
    assert cam.proxy.stream == "main"
    runtime.request_restart_camera.assert_called_once_with("front")


def test_onvif_port_change_saves_without_restart(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    r = _post(client, "/cameras/front/edit", _form(onvif_port="888"))
    assert r.status_code == 303
    assert _raw(config_file, "front")["onvif_port"] == 888
    assert _cfg(client).cameras[0].onvif_port == 888
    runtime.request_restart_camera.assert_not_called()


def test_credentials_typed_into_the_url_are_rejected(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    before = config_file.read_text(encoding="utf-8")
    r = _post(client, "/cameras/front/edit", _form(main_url="rtsp://admin:hunter2@192.0.2.10/x"))
    assert r.status_code == 422
    assert 'id="main_url-error"' in r.text
    assert "own fields" in r.text
    assert "hunter2" not in r.text
    assert config_file.read_text(encoding="utf-8") == before
    runtime.request_restart_camera.assert_not_called()


def test_invalid_onvif_port_is_rejected(client: TestClient, config_file: Path) -> None:
    before = config_file.read_text(encoding="utf-8")
    r = _post(client, "/cameras/front/edit", _form(onvif_port="70000", password="typed"))
    assert r.status_code == 422
    assert 'id="onvif_port-error"' in r.text
    assert "Enter the password again." in r.text
    assert "typed" not in r.text
    assert config_file.read_text(encoding="utf-8") == before


def test_edit_without_runtime_saves_and_says_restart_needed(
    db_with_user: str, config_file: Path
) -> None:
    """(review focus) No runtime attached: config.yaml is still written and the page says so."""
    c = _client(config_file, None)
    r = _post(c, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 303
    assert _raw(config_file, "front")["record"]["enabled"] is False
    assert "takes effect when rtsp-warden restarts" in c.get("/cameras/front").text


def test_edit_without_config_file_is_refused(db_with_user: str, runtime: MagicMock) -> None:
    cfg = AppConfig.model_validate({"cameras": [{"name": "front", "main_url": "rtsp://h/m"}]})
    c = TestClient(create_app(WebSettings(), cfg=cfg, runtime=runtime))
    _login(c)
    r = c.get("/cameras/front/edit", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"
    assert _post(c, "/cameras/front/edit", _form()).status_code == 303
    assert "without a config file" in c.get("/cameras/front").text
    runtime.request_restart_camera.assert_not_called()


def test_edit_with_read_only_config_dir_shows_error(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    before = config_file.read_text(encoding="utf-8")
    _make_read_only(config_file.parent)
    try:
        r = _post(client, "/cameras/front/edit", _form(record_enabled=None))
    finally:
        config_file.parent.chmod(0o700)
    assert r.status_code == 303
    assert config_file.read_text(encoding="utf-8") == before
    page = client.get("/cameras/front").text
    assert "Could not save camera front" in page
    assert str(config_file.parent) in page
    assert _cfg(client).cameras[0].record.enabled is True
    runtime.request_restart_camera.assert_not_called()


def test_edit_write_error_flashes_and_changes_nothing(
    client: TestClient, config_file: Path, runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9 for every user (root too): a failing config.yaml write is a flash, never a 500."""
    monkeypatch.setattr("rtsp_warden.web.services.camera_config.patch_camera", _refuse)
    before = config_file.read_text(encoding="utf-8")
    r = _post(client, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 303
    assert config_file.read_text(encoding="utf-8") == before
    assert _cfg(client).cameras[0].record.enabled is True
    runtime.request_restart_camera.assert_not_called()
    assert "Could not save camera front" in _flash(client.get("/cameras/front").text)


def test_edit_restart_that_times_out_says_still_restarting(
    client: TestClient, runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending: Future = Future()  # the supervisor has not applied it yet
    runtime.request_restart_camera.return_value = pending
    monkeypatch.setattr(camera_edit, "RUNTIME_REQUEST_TIMEOUT_S", 0.05)
    r = _post(client, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 303
    flash = _flash(client.get("/cameras/front").text)
    assert flash.startswith("info: ")
    assert "still restarting" in flash
    assert "fail" not in flash.lower()
    assert not pending.cancelled()  # a timeout never cancels the queued restart


def test_edit_restart_failure_is_reported_without_credentials(
    client: TestClient, runtime: MagicMock
) -> None:
    failed: Future = Future()
    failed.set_exception(SystemExit("unknown proxy mode for rtsp://u:p@192.0.2.10/m"))
    runtime.request_restart_camera.return_value = failed
    r = _post(client, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 303
    flash = _flash(client.get("/cameras/front").text)
    assert flash.startswith("error: ")
    assert "could not be restarted" in flash
    assert "u:p@" not in flash


def test_edit_of_a_camera_the_runtime_does_not_run_says_restart_needed(
    client: TestClient, runtime: MagicMock
) -> None:
    not_running: Future = Future()
    not_running.set_exception(CameraNotFoundError("camera 'front' not found"))
    runtime.request_restart_camera.return_value = not_running
    r = _post(client, "/cameras/front/edit", _form(record_enabled=None))
    assert r.status_code == 303
    assert "takes effect when rtsp-warden restarts" in _flash(client.get("/cameras/front").text)


# --- delete ----------------------------------------------------------------------------------


def test_detail_page_shows_edit_and_delete_to_admins_only(
    client: TestClient, viewer_client: TestClient
) -> None:
    html = client.get("/cameras/front").text
    assert 'href="/cameras/front/edit"' in html
    assert 'action="/cameras/front/delete"' in html
    assert 'data-camera="front"' in html
    assert "stay on disk" in html
    viewer_html = viewer_client.get("/cameras/front").text
    assert 'action="/cameras/front/delete"' not in viewer_html


def test_delete_removes_camera_and_keeps_recordings(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    segment = config_file.parent.parent / "recordings" / "front" / "main" / "front_main_x.ts"
    segment.parent.mkdir(parents=True)
    segment.write_bytes(b"\x47" * 188)

    r = _post(client, "/cameras/front/delete")

    assert r.status_code == 303
    assert r.headers["location"] == "/cameras"
    names = [c["name"] for c in yaml.safe_load(config_file.read_text(encoding="utf-8"))["cameras"]]
    assert names == ["back"]
    assert [c.name for c in _cfg(client).cameras] == ["back"]
    runtime.request_remove_camera.assert_called_once_with("front")
    assert segment.exists()
    page = client.get("/cameras").text
    assert "Camera front removed." in page
    assert str(segment.parent.parent) in page


def test_delete_of_a_camera_the_runtime_does_not_run_still_succeeds(
    client: TestClient, runtime: MagicMock
) -> None:
    not_running: Future = Future()
    not_running.set_exception(CameraNotFoundError("camera 'front' not found"))
    runtime.request_remove_camera.return_value = not_running

    r = _post(client, "/cameras/front/delete")

    assert r.status_code == 303
    assert [c.name for c in _cfg(client).cameras] == ["back"]
    assert "Camera front removed." in client.get("/cameras").text


def test_delete_unknown_camera_is_404(client: TestClient, runtime: MagicMock) -> None:
    assert _post(client, "/cameras/nope/delete").status_code == 404
    runtime.request_remove_camera.assert_not_called()


def test_delete_requires_admin(viewer_client: TestClient, config_file: Path) -> None:
    before = config_file.read_text(encoding="utf-8")
    assert _post(viewer_client, "/cameras/front/delete").status_code == 403
    assert config_file.read_text(encoding="utf-8") == before


def test_delete_without_runtime_still_updates_config(db_with_user: str, config_file: Path) -> None:
    c = _client(config_file, None)
    r = _post(c, "/cameras/front/delete")
    assert r.status_code == 303
    names = [cam["name"] for cam in yaml.safe_load(config_file.read_text())["cameras"]]
    assert names == ["back"]
    assert "Camera front removed." in c.get("/cameras").text


def test_delete_stops_the_onvif_event_subscription(client: TestClient) -> None:
    class FakeSubscriber:
        stopped = False

        async def stop(self) -> None:
            await asyncio.sleep(0)
            FakeSubscriber.stopped = True

    register_subscriber("front", FakeSubscriber())  # type: ignore[arg-type]
    try:
        r = _post(client, "/cameras/front/delete")
        still_registered = "front" in get_active_subscribers()
    finally:
        unregister_subscriber("front")
    assert r.status_code == 303
    assert FakeSubscriber.stopped is True
    assert still_registered is False


def test_delete_with_wedged_ffmpeg_times_out_but_config_change_lands(
    client: TestClient, config_file: Path, runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review focus) A remove that never finishes is reported, not a hung request."""
    pending: Future = Future()  # never resolved
    runtime.request_remove_camera.return_value = pending
    monkeypatch.setattr(camera_edit, "RUNTIME_REQUEST_TIMEOUT_S", 0.05)

    r = _post(client, "/cameras/front/delete")

    assert r.status_code == 303
    assert r.headers["location"] == "/cameras"
    names = [c["name"] for c in yaml.safe_load(config_file.read_text(encoding="utf-8"))["cameras"]]
    assert names == ["back"]
    assert [c.name for c in _cfg(client).cameras] == ["back"]
    flash = _flash(client.get("/cameras").text)
    assert "still stopping it" in flash
    assert "fail" not in flash.lower()
    assert not pending.cancelled()  # a timeout never cancels the queued removal


def test_delete_with_read_only_config_dir_keeps_the_camera(
    client: TestClient, config_file: Path, runtime: MagicMock
) -> None:
    before = config_file.read_text(encoding="utf-8")
    _make_read_only(config_file.parent)
    try:
        r = _post(client, "/cameras/front/delete")
    finally:
        config_file.parent.chmod(0o700)
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"
    assert config_file.read_text(encoding="utf-8") == before
    assert [c.name for c in _cfg(client).cameras] == ["front", "back"]
    runtime.request_remove_camera.assert_not_called()
    assert "Could not delete camera front" in client.get("/cameras/front").text


def test_delete_write_error_flashes_and_keeps_the_camera(
    client: TestClient, config_file: Path, runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9 for every user (root too): a failing config.yaml write is a flash, never a 500."""
    monkeypatch.setattr("rtsp_warden.web.services.camera_config.remove_camera", _refuse)
    before = config_file.read_text(encoding="utf-8")
    r = _post(client, "/cameras/front/delete")
    assert r.status_code == 303
    assert r.headers["location"] == "/cameras/front"
    assert config_file.read_text(encoding="utf-8") == before
    assert [c.name for c in _cfg(client).cameras] == ["front", "back"]
    runtime.request_remove_camera.assert_not_called()
    assert "Could not delete camera front" in _flash(client.get("/cameras/front").text)


def test_delete_reports_a_runtime_that_could_not_stop_the_camera(
    client: TestClient, runtime: MagicMock
) -> None:
    failed: Future = Future()
    failed.set_exception(RuntimeError("runtime stopping"))
    runtime.request_remove_camera.return_value = failed
    r = _post(client, "/cameras/front/delete")
    assert r.status_code == 303
    flash = _flash(client.get("/cameras").text)
    assert flash.startswith("error: ")
    assert "runtime stopping" in flash
    assert [c.name for c in _cfg(client).cameras] == ["back"]


def test_delete_removes_the_cameras_login_from_the_env_file(
    client: TestClient, config_file: Path
) -> None:
    env_path = config_file.parent / ".env"
    env_path.write_text(ENV_TEXT + 'OTHER="kept"\n', encoding="utf-8")
    r = _post(client, "/cameras/front/delete")
    assert r.status_code == 303
    assert env_path.read_text(encoding="utf-8") == 'OTHER="kept"\n'
    assert "CAM_FRONT_USER" not in os.environ
    assert "CAM_FRONT_PASS" not in os.environ


def test_delete_keeps_a_login_another_camera_still_uses(
    client: TestClient, config_file: Path
) -> None:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    data["cameras"][1]["main_url"] = FRONT_MAIN.replace("192.0.2.10", "192.0.2.11")
    config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    r = _post(client, "/cameras/front/delete")
    assert r.status_code == 303
    assert (config_file.parent / ".env").read_text(encoding="utf-8") == ENV_TEXT
    assert os.environ["CAM_FRONT_PASS"] == "old%2Fpass"
