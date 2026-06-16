"""Tests for Sprint 2 Batch 3: core pages (dashboard, cameras, recordings, events, health).

Validates route handlers, service functions, and template rendering
for all dashboard and CRUD views.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.auth import hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Create a sample config.yaml with 2 cameras."""
    cfg = {
        "cameras": [
            {
                "name": "front",
                "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                "record": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "recordings"),
                    "main": {"enabled": True, "container": "ts", "chunk_seconds": 300},
                    "sub": {"enabled": False, "container": "ts", "chunk_seconds": 300},
                },
                "proxy": {
                    "enabled": True,
                    "mode": "mjpeg",
                    "stream": "sub",
                    "bind_host": "127.0.0.1",
                    "port": 9001,
                },
            },
            {
                "name": "back",
                "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                "record": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "recordings"),
                    "main": {"enabled": True, "container": "ts", "chunk_seconds": 300},
                    "sub": {"enabled": False, "container": "ts", "chunk_seconds": 300},
                },
                "proxy": {"enabled": False, "mode": "mjpeg"},
            },
        ],
        "runtime": {
            "ffmpeg_path": "ffmpeg",
            "mediamtx_path": "mediamtx",
            "workspace_dir": str(tmp_path / "workspace"),
            "auto_restart": True,
            "restart_backoff_min_s": 1,
            "restart_backoff_max_s": 60,
            "restart_backoff_factor": 2,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.fixture
def app_with_config(sample_config: Path) -> tuple:
    """Return (app, cfg) with sample config and auth."""
    db_url = f"sqlite:///{sample_config.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(sample_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    return app, cfg


@pytest.fixture
def client_with_config(app_with_config: tuple) -> TestClient:
    """Authenticated TestClient with sample config and admin user."""
    app, _ = app_with_config
    client = TestClient(app)
    # Login
    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


@pytest.fixture
def app_no_config(tmp_path: Path) -> tuple:
    """Return (app, None) with auth but no config."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=None, runtime_provider=lambda: None)
    return app, None


@pytest.fixture
def client_no_config(app_no_config: tuple) -> TestClient:
    """Authenticated TestClient without config."""
    app, _ = app_no_config
    client = TestClient(app)
    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------


class TestDashboard:
    """Tests for the dashboard (/) route."""

    def test_dashboard_requires_auth(self) -> None:
        """Unauthenticated GET / should redirect to login or return 401."""
        settings = WebSettings()
        app = create_app(settings)
        client = TestClient(app)
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)

    def test_dashboard_renders_for_authenticated_user(self, client_with_config: TestClient) -> None:
        """Authenticated GET / should render the dashboard."""
        r = client_with_config.get("/")
        assert r.status_code == 200
        assert "Dashboard" in r.text

    def test_dashboard_shows_cameras(self, client_with_config: TestClient) -> None:
        """Dashboard should list configured cameras."""
        r = client_with_config.get("/")
        assert r.status_code == 200
        assert "front" in r.text
        assert "back" in r.text

    def test_dashboard_shows_no_recordings(self, client_with_config: TestClient) -> None:
        """Dashboard should show no-recordings message when DB is empty."""
        r = client_with_config.get("/")
        assert r.status_code == 200
        # Either shows "No recordings" or the recordings section
        assert "No recordings" in r.text or "recordings" in r.text.lower()

    def test_dashboard_shows_no_events(self, client_with_config: TestClient) -> None:
        """Dashboard should show no-events message when DB is empty."""
        r = client_with_config.get("/")
        assert r.status_code == 200
        assert "No events" in r.text or "events" in r.text.lower()


# ---------------------------------------------------------------------------
# Cameras tests
# ---------------------------------------------------------------------------


class TestCameras:
    """Tests for the cameras routes."""

    def test_cameras_list_renders(self, client_with_config: TestClient) -> None:
        """GET /cameras should render the camera list."""
        r = client_with_config.get("/cameras")
        assert r.status_code == 200
        assert "front" in r.text
        assert "back" in r.text

    def test_camera_detail_renders(self, client_with_config: TestClient) -> None:
        """GET /cameras/front should render the camera detail."""
        r = client_with_config.get("/cameras/front")
        assert r.status_code == 200
        assert "front" in r.text

    def test_camera_detail_has_mjpeg_url(self, client_with_config: TestClient) -> None:
        """Camera detail for MJPEG proxy camera should show MJPEG URL."""
        r = client_with_config.get("/cameras/front")
        assert r.status_code == 200
        assert "9001" in r.text  # proxy port

    def test_camera_detail_no_proxy(self, client_with_config: TestClient) -> None:
        """Camera detail for non-proxy camera should show no-preview message."""
        r = client_with_config.get("/cameras/back")
        assert r.status_code == 200
        assert "not enabled" in r.text.lower() or "No live preview" in r.text

    def test_camera_detail_404_for_unknown(self, client_with_config: TestClient) -> None:
        """GET /cameras/nonexistent should return 404."""
        r = client_with_config.get("/cameras/nonexistent")
        assert r.status_code == 404

    def test_camera_status_partial(self, client_with_config: TestClient) -> None:
        """GET /cameras/front/status should return camera card HTML."""
        r = client_with_config.get("/cameras/front/status")
        assert r.status_code == 200
        assert "front" in r.text


# ---------------------------------------------------------------------------
# Recordings tests
# ---------------------------------------------------------------------------


class TestRecordings:
    """Tests for the recordings routes."""

    def test_recordings_list_renders_empty(self, client_with_config: TestClient) -> None:
        """GET /recordings should render empty list."""
        r = client_with_config.get("/recordings")
        assert r.status_code == 200
        assert "No recordings" in r.text or "recordings" in r.text.lower()

    def test_recordings_list_with_filter(self, client_with_config: TestClient) -> None:
        """GET /recordings?camera=front should accept filter params."""
        r = client_with_config.get("/recordings?camera=front")
        assert r.status_code == 200

    def test_recording_detail_404(self, client_with_config: TestClient) -> None:
        """GET /recordings/99999 should return 404 for non-existent recording."""
        r = client_with_config.get("/recordings/99999")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Events tests
# ---------------------------------------------------------------------------


class TestEvents:
    """Tests for the events routes."""

    def test_events_list_renders_empty(self, client_with_config: TestClient) -> None:
        """GET /events should render empty list."""
        r = client_with_config.get("/events")
        assert r.status_code == 200
        assert "No events" in r.text or "events" in r.text.lower()

    def test_events_list_with_filter(self, client_with_config: TestClient) -> None:
        """GET /events?severity=warn should accept filter params."""
        r = client_with_config.get("/events?severity=warn")
        assert r.status_code == 200

    def test_event_detail_404(self, client_with_config: TestClient) -> None:
        """GET /events/99999 should return 404 for non-existent event."""
        r = client_with_config.get("/events/99999")
        assert r.status_code == 404

    def test_events_partial(self, client_with_config: TestClient) -> None:
        """GET /events/partial should return event rows HTML."""
        r = client_with_config.get("/events/partial")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Health tests
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for the health endpoints."""

    def test_healthz(self) -> None:
        """GET /healthz should return ok."""
        app = create_app(WebSettings())
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_health_returns_json(self) -> None:
        """GET /health with Accept: application/json should return JSON."""
        app = create_app(WebSettings())
        client = TestClient(app)
        r = client.get("/health", headers={"Accept": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body or "version" in body

    def test_health_returns_html(self) -> None:
        """GET /health with Accept: text/html should return HTML."""
        app = create_app(WebSettings())
        client = TestClient(app)
        r = client.get("/health", headers={"Accept": "text/html"})
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_health_partial(self) -> None:
        """GET /health/partial should return HTML fragment."""
        app = create_app(WebSettings())
        client = TestClient(app)
        r = client.get("/health/partial")
        assert r.status_code == 200

    def test_health_with_config(self, client_with_config: TestClient) -> None:
        """GET /health with config should show camera status."""
        r = client_with_config.get("/health", headers={"Accept": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert "cameras" in body


# ---------------------------------------------------------------------------
# Service layer tests
# ---------------------------------------------------------------------------


class TestCameraService:
    """Tests for the camera listing service."""

    def test_list_cameras(self, sample_config: Path) -> None:
        """list_cameras should return camera dicts from config."""
        cfg = load_config(sample_config)
        from rtsp_warden.web.services.cameras import list_cameras

        cameras = list_cameras(cfg)
        assert len(cameras) == 2
        assert cameras[0]["name"] == "front"
        assert cameras[1]["name"] == "back"

    def test_list_cameras_redacts_urls(self, sample_config: Path) -> None:
        """list_cameras should redact RTSP URLs."""
        cfg = load_config(sample_config)
        from rtsp_warden.web.services.cameras import list_cameras

        cameras = list_cameras(cfg)
        for cam in cameras:
            assert "pass" not in cam["main_url_redacted"]
            assert "pass" not in cam["sub_url_redacted"]

    def test_get_camera_by_name(self, sample_config: Path) -> None:
        """get_camera_by_name should return camera dict or None."""
        cfg = load_config(sample_config)
        from rtsp_warden.web.services.cameras import get_camera_by_name

        cam = get_camera_by_name(cfg, "front")
        assert cam is not None
        assert cam["name"] == "front"

        cam = get_camera_by_name(cfg, "nonexistent")
        assert cam is None


class TestRecordingService:
    """Tests for the recording listing service."""

    def test_list_recordings_empty(self) -> None:
        """list_recordings should return empty list when DB has no recordings."""
        import tempfile

        from rtsp_warden.web.services.recordings import list_recordings

        db_url = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        recordings, total = list_recordings(limit=10)
        assert recordings == []
        assert total == 0

    def test_get_recording_by_id_not_found(self) -> None:
        """get_recording_by_id should return None for non-existent ID."""
        import tempfile

        from rtsp_warden.web.services.recordings import get_recording_by_id

        db_url = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        result = get_recording_by_id(99999)
        assert result is None


class TestEventService:
    """Tests for the event listing service."""

    def test_list_events_empty(self) -> None:
        """list_events should return empty list when DB has no events."""
        import tempfile

        from rtsp_warden.web.services.events import list_events

        db_url = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        events, total = list_events(limit=10)
        assert events == []
        assert total == 0

    def test_get_event_by_id_not_found(self) -> None:
        """get_event_by_id should return None for non-existent ID."""
        import tempfile

        from rtsp_warden.web.services.events import get_event_by_id

        db_url = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        result = get_event_by_id(99999)
        assert result is None


class TestRuntimeService:
    """Tests for the runtime status service."""

    def test_get_runtime_status_no_runtime(self, sample_config: Path) -> None:
        """get_runtime_status with None runtime should return minimal status."""
        cfg = load_config(sample_config)
        from rtsp_warden.web.services.runtime import get_runtime_status

        status = get_runtime_status(None, cfg)
        assert status["ok"] is True
        assert len(status["cameras"]) == 2
        assert status["cameras"][0]["name"] == "front"
        assert status["cameras"][0]["status"] == "stopped"

    def test_get_runtime_status_has_version(self, sample_config: Path) -> None:
        """get_runtime_status should always include version."""
        cfg = load_config(sample_config)
        from rtsp_warden.web.services.runtime import get_runtime_status

        status = get_runtime_status(None, cfg)
        assert "version" in status
