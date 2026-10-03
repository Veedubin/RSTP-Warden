"""Web UI tests for the detector integration.

Tests the dashboard detections stat, camera detector partial,
and detector-related web endpoints.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.auth import hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema, insert_event
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


@pytest.fixture
def config_with_detectors(tmp_path: Path) -> Path:
    """Create a config.yaml with cameras that have detectors."""
    cfg = {
        "cameras": [
            {
                "name": "driveway",
                "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                "record": {"enabled": True, "output_dir": str(tmp_path / "recordings")},
                "proxy": {
                    "enabled": True,
                    "mode": "mjpeg",
                    "stream": "sub",
                    "bind_host": "127.0.0.1",
                    "port": 9001,
                },
                "detectors": [
                    {
                        "type": "motion",
                        "enabled": True,
                        "interval_seconds": 1.0,
                        "min_area": 500,
                        "sensitivity": 0.5,
                    },
                ],
            },
            {
                "name": "backyard",
                "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                "record": {"enabled": True, "output_dir": str(tmp_path / "recordings")},
                "proxy": {"enabled": False, "mode": "mjpeg"},
                "detectors": [],
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
def app_with_detectors(config_with_detectors: Path) -> tuple:
    """Return (app, cfg) with detectors config and auth."""
    db_url = f"sqlite:///{config_with_detectors.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(config_with_detectors)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    return app, cfg


@pytest.fixture
def client_with_detectors(app_with_detectors: tuple) -> TestClient:
    """Authenticated TestClient with detector-enabled config."""
    app, _ = app_with_detectors
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


class TestDashboardDetections:
    """Tests for dashboard detections today stat."""

    def test_dashboard_shows_detections_today(self, client_with_detectors: TestClient) -> None:
        """Write 5 events, GET /, verify '5' appears in detections_today stat."""
        for _ in range(5):
            insert_event(
                camera_name="driveway",
                event_type="motion",
                label="motion",
                confidence=1.0,
                zone="",
                track_id=None,
                message="test detection",
                created_at=datetime.now(timezone.utc),
            )

        r = client_with_detectors.get("/")
        assert r.status_code == 200
        assert "Detections today" in r.text
        # The number 5 should appear in the stat card
        assert ">5<" in r.text


class TestCameraDetectors:
    """Tests for camera detector display and partials."""

    def test_camera_detail_loads_detectors(self, client_with_detectors: TestClient) -> None:
        """Camera with 1 motion detector: detail page loads the Detection panel."""
        r = client_with_detectors.get("/cameras/driveway")
        assert r.status_code == 200
        # The htmx container of the Detection panel (RW-3 Task 15) should be present
        assert 'hx-get="/cameras/driveway/detection"' in r.text

    def test_detection_panel_contains_the_detector_list(
        self, client_with_detectors: TestClient
    ) -> None:
        """The panel renders the detector table and polls the detector partial."""
        r = client_with_detectors.get("/cameras/driveway/detection")
        assert r.status_code == 200
        assert 'hx-get="/cameras/driveway/detectors"' in r.text
        assert "min_area=500" in r.text
        assert "/cameras/driveway/detectors/0/enabled" in r.text

    def test_camera_detectors_partial(self, client_with_detectors: TestClient) -> None:
        """GET /cameras/{name}/detectors (htmx) returns detector info."""
        r = client_with_detectors.get("/cameras/driveway/detectors")
        assert r.status_code == 200
        assert "motion" in r.text
        assert "min_area=500" in r.text

    def test_camera_with_no_detectors_shows_help(self, client_with_detectors: TestClient) -> None:
        """Camera with no detectors: partial shows 'No detectors configured'."""
        r = client_with_detectors.get("/cameras/backyard/detectors")
        assert r.status_code == 200
        assert "No detectors configured" in r.text
        assert "config.yaml" in r.text

    def test_detector_partial_404_for_unknown_camera(
        self, client_with_detectors: TestClient
    ) -> None:
        """GET /cameras/{name}/detectors for a camera that is not configured -> 404."""
        r = client_with_detectors.get("/cameras/nonexistent/detectors")
        assert r.status_code == 404

    def test_detector_partial_requires_auth(self, app_with_detectors: tuple) -> None:
        """GET /cameras/{name}/detectors without auth -> 401/302."""
        app, _ = app_with_detectors
        client = TestClient(app)
        r = client.get("/cameras/driveway/detectors", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)
