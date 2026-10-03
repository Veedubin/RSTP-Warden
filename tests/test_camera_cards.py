"""Camera cards, list and dashboard first paint, the detail page layout and its status row."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.auth import hash_password
from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.db import create_user
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.paths import TEMPLATES_DIR
from rtsp_warden.web.routes._common import templates
from rtsp_warden.web.services.cameras import list_cameras, status_label

CAM_URL = "rtsp://u:p@h/m"
ERROR_LINE = "Connection to rtsp://u:p@h/m failed: 401 Unauthorized"
REDACTED_ERROR = "Connection to rtsp://***:***@h/m failed: 401 Unauthorized"
FAILED_BADGE = '<span class="status-badge status-failed">Failed</span>'

# RW-3 replaces exactly this block with the detection panel include; it then sets
# DETECTORS_ARTICLE to that include. The viewer test below matches either version.
DETECTORS_ARTICLE = (
    "{# Detectors #}\n"
    '<article hx-get="/cameras/{{ camera.name }}/detectors"\n'
    '         hx-trigger="load, every 10s"\n'
    '         hx-swap="innerHTML">\n'
    "  <p>Loading detectors...</p>\n"
    "</article>\n"
)

ADMIN_ONLY_MARKERS = (
    'href="/cameras/cam/edit"',
    'href="/cameras/cam/zones"',
    'href="/cameras/cam/sensitivity"',
    'href="/cameras/cam/detection-classes"',
    'action="/cameras/cam/retention"',
)


class _Proc:
    """Fake ManagedProcess with what live_status and cli.build_status read."""

    def __init__(self, running: bool) -> None:
        self._running = running

    def poll(self) -> int | None:
        return None if self._running else 1

    def is_running(self) -> bool:
        return self._running

    def pid(self) -> int | None:
        return None

    def stderr_tail(self) -> list[str]:
        return []


def _runtime(cam: CameraConfig, *, running: bool, last_error: str = "") -> SimpleNamespace:
    """One camera whose single ingest process is running or dead (no restart scheduled)."""
    ingest = SimpleNamespace(
        stream_name="main",
        upstream_url=cam.main_url,
        proc=_Proc(running),
        record_cfg=None,
        record_output_dir=None,
        mjpeg_hub=None,
        rtsp_publish_url=None,
    )
    cam_rt = SimpleNamespace(
        camera=cam,
        hub=None,
        proxy=None,
        recorder=SimpleNamespace(processes=lambda: [ingest]),
        next_restart_at=0.0,
        last_error=last_error,
    )
    return SimpleNamespace(cameras=[cam_rt])


def _login(client: TestClient, username: str, password: str) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _card_row(status: str, restart_in: int | None = None, last_error: str = "") -> dict:
    cfg = AppConfig(cameras=[CameraConfig(name="cam", main_url=CAM_URL)])
    row = list_cameras(cfg)[0]
    row.update(
        status=status,
        restart_in=restart_in,
        status_label=status_label(status, restart_in),
        last_error=last_error,
    )
    return row


def _render_card(row: dict) -> str:
    return templates.env.get_template("partials/camera_card.html").render(camera=row)


@pytest.fixture
def failed_app(db_with_user):
    cam = CameraConfig(name="cam", main_url=CAM_URL)
    runtime = _runtime(cam, running=False, last_error=ERROR_LINE)
    return create_app(
        WebSettings(),
        cfg=AppConfig(cameras=[cam]),
        runtime_provider=lambda: runtime,
        runtime=runtime,
    )


@pytest.fixture
def admin(failed_app) -> TestClient:
    client = TestClient(failed_app)
    _login(client, "admin", "testpass123")
    return client


@pytest.fixture
def viewer(failed_app) -> TestClient:
    create_user("viewer", hash_password("viewerpass123"), is_admin=False)
    client = TestClient(failed_app)
    _login(client, "viewer", "viewerpass123")
    return client


# --- the card partial ---


@pytest.mark.parametrize(
    ("status", "restart_in", "label"),
    [
        ("running", None, "Running"),
        ("restarting", 12, "Restarting in 12s"),
        ("failed", None, "Failed"),
        ("idle", None, "Idle"),
        ("waiting", None, "Waiting for event"),
        ("degraded", None, "Proxy down"),
    ],
)
def test_card_partial_has_badge_for_each_status(status, restart_in, label):
    html = _render_card(_card_row(status, restart_in))
    assert f'<span class="status-badge status-{status}">{label}</span>' in html
    assert 'hx-get="/cameras/cam/status"' in html
    assert 'hx-swap="outerHTML"' in html
    assert "u:p@" not in html


def test_card_shows_error_line_only_when_not_running():
    assert REDACTED_ERROR in _render_card(_card_row("failed", last_error=REDACTED_ERROR))
    assert REDACTED_ERROR not in _render_card(_card_row("running", last_error=REDACTED_ERROR))


# --- first paint and polling use the same partial ---


def test_list_page_renders_card_partial(admin):
    html = admin.get("/cameras").text
    assert FAILED_BADGE in html
    assert 'hx-get="/cameras/cam/status"' in html
    assert REDACTED_ERROR in html
    assert "u:p@" not in html


def test_dashboard_renders_card_partial(admin):
    html = admin.get("/").text
    assert FAILED_BADGE in html
    assert 'hx-get="/cameras/cam/status"' in html
    assert "u:p@" not in html


def test_status_poll_returns_card_with_redacted_error(admin):
    r = admin.get("/cameras/cam/status", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert r.text.lstrip().startswith('<article class="camera-card" id="camera-card-cam"')
    assert FAILED_BADGE in r.text
    assert REDACTED_ERROR in r.text
    assert "u:p@" not in r.text


# --- detail page ---


def test_detail_polls_status_row(admin):
    html = admin.get("/cameras/cam").text
    assert 'hx-get="/cameras/cam/status-row"' in html
    assert '<tr id="camera-status-row"' in html
    assert FAILED_BADGE in html
    assert REDACTED_ERROR in html
    assert "u:p@" not in html


def test_status_row_is_a_bare_tr_fragment(admin):
    r = admin.get("/cameras/cam/status-row", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert r.text.lstrip().startswith('<tr id="camera-status-row"')
    assert "<html" not in r.text
    assert FAILED_BADGE in r.text
    assert REDACTED_ERROR in r.text


@pytest.mark.parametrize("suffix", ["status", "status-row"])
def test_poll_for_removed_camera_stops_polling(admin, suffix):
    r = admin.get(f"/cameras/gone/{suffix}", headers={"HX-Request": "true"})
    assert r.status_code == 286
    assert r.text == ""


@pytest.mark.parametrize("suffix", ["status", "status-row"])
def test_plain_request_for_unknown_camera_is_404(admin, suffix):
    assert admin.get(f"/cameras/gone/{suffix}").status_code == 404


def test_detail_shows_admin_controls_to_admin(admin):
    html = admin.get("/cameras/cam").text
    for marker in ADMIN_ONLY_MARKERS:
        assert marker in html
    assert "Changes apply after the next restart." in html


def test_detail_hides_admin_controls_from_viewer(viewer):
    r = viewer.get("/cameras/cam")
    assert r.status_code == 200
    for marker in ADMIN_ONLY_MARKERS:
        assert marker not in r.text
    # The detectors panel (RW-3: the detection panel) still loads for viewers.
    assert 'hx-get="/cameras/cam/detect' in r.text


def test_detail_drops_the_always_empty_recordings_table(admin):
    assert "Recent recordings" not in admin.get("/cameras/cam").text


def test_detectors_article_is_byte_for_byte_unchanged():
    text = (TEMPLATES_DIR / "cameras" / "detail.html").read_text(encoding="utf-8")
    assert text.count(DETECTORS_ARTICLE) == 1
