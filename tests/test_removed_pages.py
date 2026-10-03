"""Routes and pages removed with the never-written recordings and clips tables (migration 0003).

The HLS player those pages carried survives as ``partials/hls_player.html`` for ``.ts`` clips.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.db.engine import get_session
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes._common import templates


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


def _insert_event_row(event_id: int) -> None:
    """A bare events row; this SQL works on both the 0002 and the 0003 schema."""
    with get_session() as session:
        session.execute(
            text("INSERT INTO events (id, event_type, message) VALUES (:id, 'motion', 'm')"),
            {"id": event_id},
        )
        session.commit()


@pytest.fixture
def client(db_with_user: str) -> TestClient:
    cfg = AppConfig(cameras=[CameraConfig(name="cam", main_url="rtsp://u:p@h/m")])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None)
    c = TestClient(app)
    _login(c)
    return c


@pytest.mark.parametrize(
    "path",
    [
        "/recordings",
        "/recordings/1",
        "/api/recordings/1/timeline",
        "/clips/1",
        "/clips/1/download",
        "/static/js/timeline.js",
    ],
)
def test_removed_get_routes_return_404(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_manual_clip_generation_route_is_gone(client: TestClient) -> None:
    _insert_event_row(7)
    token = client.cookies.get("warden_csrf", "")
    r = client.post("/events/7/clip", headers={"X-CSRF-Token": token}, follow_redirects=False)
    assert r.status_code == 404


def test_nav_and_dashboard_have_no_recordings(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert 'href="/recordings"' not in r.text
    assert "Recent recordings" not in r.text


def test_event_detail_has_no_generate_clip_button(client: TestClient) -> None:
    _insert_event_row(7)
    r = client.get("/events/7")
    assert r.status_code == 200
    assert "Generate Clip" not in r.text
    assert "/events/7/clip" not in r.text


def test_hls_player_partial_renders_the_playlist() -> None:
    html = templates.get_template("partials/hls_player.html").render(htl_src="/events/7/clip.m3u8")
    assert '<video id="player"' in html
    assert '<script src="/static/js/hls.min.js"></script>' in html
    assert 'var src = "/events/7/clip.m3u8";' in html


def test_hls_player_partial_escapes_the_source_for_javascript() -> None:
    html = templates.get_template("partials/hls_player.html").render(htl_src="/x</script>")
    assert "/x</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_hls_player_partial_without_a_source() -> None:
    html = templates.get_template("partials/hls_player.html").render(htl_src=None)
    assert "<video" not in html
    assert "No video available for playback." in html
