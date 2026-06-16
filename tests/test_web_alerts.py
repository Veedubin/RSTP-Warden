"""Tests for the alerts web UI routes."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.alerts.base import NotificationResult
from rtsp_warden.auth import hash_password
from rtsp_warden.config import load_config
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, create_user, ensure_schema
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes.alerts import router as alerts_router

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alerts_config(tmp_path: Path) -> Path:
    """Create a config.yaml with alerts notifiers."""
    cfg = {
        "cameras": [
            {
                "name": "front",
                "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                "record": {"enabled": True, "output_dir": str(tmp_path / "recordings")},
                "proxy": {"enabled": False, "mode": "mjpeg"},
            },
        ],
        "runtime": {
            "ffmpeg_path": "ffmpeg",
            "workspace_dir": str(tmp_path / "workspace"),
        },
        "alerts": {
            "enabled": True,
            "notifiers": [
                {
                    "name": "test-phone",
                    "type": "ntfy",
                    "url": "https://ntfy.sh",
                    "topic": "warden-test",
                    "enabled": True,
                    "severities": ["warn", "error"],
                },
                {
                    "name": "test-hook",
                    "type": "webhook",
                    "url": "https://example.com/webhook",
                    "enabled": True,
                    "severities": ["info", "warn", "error"],
                },
            ],
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.fixture
def admin_client(alerts_config: Path) -> TestClient:
    """Authenticated TestClient as admin user with alerts router included."""
    db_url = f"sqlite:///{alerts_config.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(alerts_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    # Include alerts router (Phase 2 will add this to app.py)
    app.include_router(alerts_router)
    client = TestClient(app)

    # Login as admin
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
def viewer_client(alerts_config: Path) -> TestClient:
    """Authenticated TestClient as viewer (non-admin) user."""
    db_url = f"sqlite:///{alerts_config.parent / 'test_viewer.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()

    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)
    viewer_pw = hash_password("viewerpass123")
    create_user("viewer", viewer_pw, is_admin=False)

    cfg = load_config(alerts_config)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    app.include_router(alerts_router)
    client = TestClient(app)

    r = client.get("/login")
    csrf = r.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "viewer", "password": "viewerpass123", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        cookies={"warden_csrf": csrf},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlertsWebUI:
    """Tests for the alerts web routes."""

    def test_alerts_page_renders(self, admin_client: TestClient) -> None:
        """Admin GET /alerts -> 200 with notifier listing."""
        r = admin_client.get("/alerts")
        assert r.status_code == 200
        assert "test-phone" in r.text
        assert "test-hook" in r.text
        assert "ntfy" in r.text
        assert "webhook" in r.text

    def test_alerts_test_endpoint(self, admin_client: TestClient) -> None:
        """GET /alerts/test-name/test returns JSON with success status (mocked manager)."""
        # Mock the AlertManager on app state
        mock_result = NotificationResult(
            notifier_name="test-phone",
            success=True,
            sent_at=datetime.now(tz=timezone.utc),
            http_status=200,
        )
        mock_manager = AsyncMock()
        mock_manager.test_notifier = AsyncMock(return_value=mock_result)
        admin_client.app.state.alert_manager = mock_manager

        r = admin_client.get("/alerts/test-phone/test")
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["notifier_name"] == "test-phone"

    def test_alerts_page_requires_admin(self, viewer_client: TestClient) -> None:
        """Non-admin GET /alerts -> 403."""
        r = viewer_client.get("/alerts")
        assert r.status_code == 403
