"""Web UI tests for camera sensitivity adjustment.

Tests the sensitivity page render, save, config persistence,
auth enforcement, and detector rebuild on save-and-reload.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

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


@pytest.fixture
def config_with_sensitivity(tmp_path: Path) -> Path:
    """Create a config.yaml with a camera that has detectors and sensitivity."""
    cfg = {
        "cameras": [
            {
                "name": "front_door",
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
                "sensitivity": 50,
                "detectors": [
                    {
                        "type": "motion",
                        "enabled": True,
                        "interval_seconds": 1.0,
                        "min_area": 500,
                    },
                    {
                        "type": "person",
                        "enabled": True,
                        "interval_seconds": 2.0,
                    },
                ],
            },
            {
                "name": "backyard",
                "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                "record": {"enabled": True, "output_dir": str(tmp_path / "recordings")},
                "proxy": {"enabled": False, "mode": "mjpeg"},
                "sensitivity": 75,
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
def app_with_sensitivity(config_with_sensitivity: Path) -> tuple:
    """Return (app, cfg) with sensitivity config and auth."""
    db_url = f"sqlite:///{config_with_sensitivity.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(config_with_sensitivity)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    app.state.config_path = str(config_with_sensitivity)
    return app, cfg


@pytest.fixture
def client_with_sensitivity(app_with_sensitivity: tuple) -> TestClient:
    """Authenticated TestClient with sensitivity-enabled config."""
    app, _ = app_with_sensitivity
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


class TestSensitivityPage:
    """Tests for GET /cameras/{name}/sensitivity."""

    def test_sensitivity_page_returns_200_for_admin(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Admin user can access the sensitivity page."""
        r = client_with_sensitivity.get("/cameras/front_door/sensitivity")
        assert r.status_code == 200
        assert "Sensitivity" in r.text
        assert "front_door" in r.text

    def test_sensitivity_page_shows_current_value(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Sensitivity page shows the current value."""
        r = client_with_sensitivity.get("/cameras/front_door/sensitivity")
        assert r.status_code == 200
        assert "50" in r.text

    def test_sensitivity_page_shows_detector_mappings(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Sensitivity page shows per-detector mapping preview."""
        r = client_with_sensitivity.get("/cameras/front_door/sensitivity")
        assert r.status_code == 200
        assert "motion" in r.text
        assert "varThreshold" in r.text

    def test_sensitivity_page_returns_404_for_unknown_camera(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Sensitivity page returns 404 for non-existent camera."""
        r = client_with_sensitivity.get("/cameras/nonexistent/sensitivity")
        assert r.status_code == 404

    def test_sensitivity_page_requires_admin(self, app_with_sensitivity: tuple) -> None:
        """Sensitivity page returns 401/403 for non-admin users."""
        app, _ = app_with_sensitivity
        client = TestClient(app)
        r = client.get("/cameras/front_door/sensitivity", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)


class TestSaveSensitivity:
    """Tests for POST /cameras/{name}/sensitivity."""

    def test_save_sensitivity_redirects_on_success(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Saving a valid sensitivity value redirects (303) to camera detail."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "75", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "/cameras/front_door" in r.headers.get("location", "")

    def test_save_sensitivity_updates_camera_config(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Saving sensitivity updates the in-memory camera config."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "80", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the sensitivity page shows the new value
        r = client_with_sensitivity.get("/cameras/front_door/sensitivity")
        assert r.status_code == 200
        assert "80" in r.text

    def test_save_sensitivity_with_out_of_range_returns_422(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Saving sensitivity > 100 returns 422."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "150", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_sensitivity_with_negative_returns_422(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Saving sensitivity < 0 returns 422."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "-5", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_sensitivity_with_non_int_returns_422(
        self, client_with_sensitivity: TestClient
    ) -> None:
        """Saving non-integer sensitivity returns 422."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "abc", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_sensitivity_writes_to_config_yaml(
        self,
        client_with_sensitivity: TestClient,
        config_with_sensitivity: Path,
    ) -> None:
        """Saving sensitivity persists the change to config.yaml."""
        csrf = client_with_sensitivity.cookies.get("warden_csrf", "")
        r = client_with_sensitivity.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "90", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Reload config.yaml from disk
        raw = yaml.safe_load(config_with_sensitivity.read_text(encoding="utf-8"))
        front_door = None
        for cam in raw["cameras"]:
            if cam["name"] == "front_door":
                front_door = cam
                break
        assert front_door is not None
        assert front_door["sensitivity"] == 90

    def test_save_and_reload_calls_rebuild_camera_detectors(
        self, app_with_sensitivity: tuple
    ) -> None:
        """Save + Reload action calls rebuild_camera_detectors on the runtime."""
        app, _ = app_with_sensitivity

        mock_runtime = MagicMock()
        app.state.runtime = mock_runtime

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

        r = client.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "70", "action": "save_and_reload", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        mock_runtime.rebuild_camera_detectors.assert_called_once_with("front_door")


class TestSensitivityAuth:
    """Tests for auth enforcement on sensitivity routes."""

    def test_sensitivity_page_requires_auth(self, app_with_sensitivity: tuple) -> None:
        """Sensitivity page requires authentication."""
        app, _ = app_with_sensitivity
        client = TestClient(app)
        r = client.get("/cameras/front_door/sensitivity", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)

    def test_save_sensitivity_requires_admin(self, app_with_sensitivity: tuple) -> None:
        """Saving sensitivity requires admin auth."""
        app, _ = app_with_sensitivity
        client = TestClient(app)
        csrf = client.get("/login").cookies.get("warden_csrf", "")
        r = client.post(
            "/cameras/front_door/sensitivity",
            data={"sensitivity": "80", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)
