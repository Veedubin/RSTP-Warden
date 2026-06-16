"""Tests for per-camera data retention policy resolution (Feature 1, Sprint 6).

Covers:
  - resolve_retention: per-camera override vs. global fallback
  - Deep-copy isolation: mutating result does not affect inputs
  - RetentionManager integration: resolved config drives cleanup
  - Web route: POST /cameras/{name}/retention saves/clears override
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden.auth import hash_password
from rtsp_warden.config import (
    CameraConfig,
    RetentionConfig,
    load_config,
)
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import create_admin_user, ensure_schema
from rtsp_warden.retention import RetentionManager
from rtsp_warden.retention_resolver import resolve_retention

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Unit tests: resolve_retention
# ---------------------------------------------------------------------------


class TestResolveRetention:
    """Unit tests for resolve_retention()."""

    def test_no_override_returns_global_copy(self) -> None:
        """When camera has no per-camera override, returns a copy of global."""
        global_cfg = RetentionConfig(max_days=30, max_gb=50.0, keep_last_n=5)
        cam = CameraConfig(
            name="lobby",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
        )
        result = resolve_retention(cam, global_cfg)
        assert result.max_days == 30
        assert result.max_gb == 50.0
        assert result.keep_last_n == 5

    def test_override_returns_camera_retention(self) -> None:
        """When camera has a per-camera override, returns that."""
        global_cfg = RetentionConfig(max_days=30, max_gb=50.0)
        cam = CameraConfig(
            name="front_door",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
            retention=RetentionConfig(max_days=90, max_gb=100.0, keep_last_n=500),
        )
        result = resolve_retention(cam, global_cfg)
        assert result.max_days == 90
        assert result.max_gb == 100.0
        assert result.keep_last_n == 500

    def test_deep_copy_isolation_global(self) -> None:
        """Mutating the returned config does not affect the global input."""
        global_cfg = RetentionConfig(max_days=30)
        cam = CameraConfig(
            name="lobby",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
        )
        result = resolve_retention(cam, global_cfg)
        result.max_days = 0
        assert global_cfg.max_days == 30  # unchanged

    def test_deep_copy_isolation_camera(self) -> None:
        """Mutating the returned config does not affect the camera override."""
        cam = CameraConfig(
            name="front_door",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
            retention=RetentionConfig(max_days=90),
        )
        global_cfg = RetentionConfig(max_days=30)
        result = resolve_retention(cam, global_cfg)
        result.max_days = 0
        assert cam.retention.max_days == 90  # unchanged

    def test_deep_copy_isolation_between_cameras(self) -> None:
        """Two cameras resolving from the same global don't share state."""
        global_cfg = RetentionConfig(max_days=30)
        cam_a = CameraConfig(
            name="a",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
        )
        cam_b = CameraConfig(
            name="b",
            main_url="rtsp://y/main",
            sub_url="rtsp://y/sub",
        )
        ret_a = resolve_retention(cam_a, global_cfg)
        ret_b = resolve_retention(cam_b, global_cfg)
        ret_a.max_days = 999
        assert ret_b.max_days == 30  # independent

    def test_none_retention_means_no_override(self) -> None:
        """CameraConfig(retention=None) falls back to global."""
        global_cfg = RetentionConfig(max_days=14, cleanup_interval_seconds=600)
        cam = CameraConfig(
            name="driveway",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
            retention=None,
        )
        result = resolve_retention(cam, global_cfg)
        assert result.max_days == 14
        assert result.cleanup_interval_seconds == 600

    def test_global_defaults_used_when_no_override(self) -> None:
        """Default RetentionConfig is used when neither camera nor caller override."""
        global_cfg = RetentionConfig()
        cam = CameraConfig(
            name="backyard",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
        )
        result = resolve_retention(cam, global_cfg)
        assert result.max_days is None
        assert result.max_gb is None
        assert result.keep_last_n == 0
        assert result.cleanup_interval_seconds == 300


# ---------------------------------------------------------------------------
# Integration: RetentionManager uses resolved config
# ---------------------------------------------------------------------------


class TestRetentionManagerIntegration:
    """Integration tests verifying RetentionManager uses the resolved config."""

    def test_deletes_expired_files_with_per_camera_override(self, tmp_path: Path) -> None:
        """RetentionManager deletes old files when per-camera max_days=1."""
        cam = CameraConfig(
            name="testcam",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
            retention=RetentionConfig(max_days=1, cleanup_interval_seconds=1),
        )
        global_cfg = RetentionConfig(max_days=30)
        effective = resolve_retention(cam, global_cfg)
        assert effective.max_days == 1

        camera_root = tmp_path / "testcam"
        camera_root.mkdir()
        old_file = camera_root / "old_segment.ts"
        old_file.write_bytes(b"\x00" * 1024)
        # Set mtime to 2 days ago so it exceeds max_days=1
        old_time = time.time() - 2 * 86400
        os.utime(old_file, (old_time, old_time))

        mgr = RetentionManager(camera_name="testcam", camera_root=camera_root, cfg=effective)
        mgr.run()
        assert not old_file.exists()

    def test_keeps_files_within_max_days(self, tmp_path: Path) -> None:
        """RetentionManager keeps files newer than max_days."""
        cam = CameraConfig(
            name="testcam",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
            retention=RetentionConfig(max_days=7, cleanup_interval_seconds=1),
        )
        global_cfg = RetentionConfig(max_days=30)
        effective = resolve_retention(cam, global_cfg)
        assert effective.max_days == 7

        camera_root = tmp_path / "testcam"
        camera_root.mkdir()
        recent_file = camera_root / "recent_segment.ts"
        recent_file.write_bytes(b"\x00" * 1024)

        mgr = RetentionManager(camera_name="testcam", camera_root=camera_root, cfg=effective)
        mgr.run()
        assert recent_file.exists()

    def test_global_fallback_used_for_camera_without_override(self, tmp_path: Path) -> None:
        """Camera without override uses global retention config."""
        cam = CameraConfig(
            name="fallback_cam",
            main_url="rtsp://x/main",
            sub_url="rtsp://x/sub",
        )
        global_cfg = RetentionConfig(max_days=1, cleanup_interval_seconds=1)
        effective = resolve_retention(cam, global_cfg)
        assert effective.max_days == 1

        camera_root = tmp_path / "fallback_cam"
        camera_root.mkdir()
        old_file = camera_root / "old.ts"
        old_file.write_bytes(b"\x00" * 1024)
        old_time = time.time() - 2 * 86400
        os.utime(old_file, (old_time, old_time))

        mgr = RetentionManager(camera_name="fallback_cam", camera_root=camera_root, cfg=effective)
        mgr.run()
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# Config roundtrip
# ---------------------------------------------------------------------------


class TestRetentionConfigRoundtrip:
    """YAML roundtrip tests for per-camera retention."""

    def test_camera_retention_survives_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Per-camera retention is preserved through YAML serialize/deserialize."""
        cfg_data = {
            "cameras": [
                {
                    "name": "front_door",
                    "main_url": "rtsp://user:pass@192.168.1.50:554/main",
                    "sub_url": "rtsp://user:pass@192.168.1.50:554/sub",
                    "retention": {
                        "max_days": 90,
                        "max_gb": 100.0,
                        "keep_last_n": 500,
                    },
                },
            ],
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data))
        loaded = load_config(cfg_path)
        cam = loaded.cameras[0]
        assert cam.retention is not None
        assert cam.retention.max_days == 90
        assert cam.retention.max_gb == 100.0
        assert cam.retention.keep_last_n == 500

    def test_camera_without_retention_survives_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Camera without retention override has retention=None after roundtrip."""
        cfg_data = {
            "cameras": [
                {
                    "name": "driveway",
                    "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                    "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                },
            ],
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data))
        loaded = load_config(cfg_path)
        cam = loaded.cameras[0]
        assert cam.retention is None

    def test_global_retention_survives_yaml_roundtrip(self, tmp_path: Path) -> None:
        """AppConfig.retention (global) is preserved through roundtrip."""
        cfg_data = {
            "retention": {
                "max_days": 30,
                "max_gb": 50.0,
                "keep_last_n": 100,
                "cleanup_interval_seconds": 300,
            },
            "cameras": [
                {
                    "name": "hallway",
                    "main_url": "rtsp://x/main",
                    "sub_url": "rtsp://x/sub",
                },
            ],
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data))
        loaded = load_config(cfg_path)
        assert loaded.retention.max_days == 30
        assert loaded.retention.max_gb == 50.0
        assert loaded.retention.keep_last_n == 100


# ---------------------------------------------------------------------------
# Web route tests: POST /cameras/{name}/retention
# ---------------------------------------------------------------------------


def _make_admin_client(
    tmp_path: Path,
    cameras: list[dict],
    global_retention: dict | None = None,
) -> tuple[TestClient, Path]:
    """Build an authenticated TestClient as admin with a config.yaml on disk.

    Returns (client, config_path).
    """
    db_url = f"sqlite:///{tmp_path / 'test_retention.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()

    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg_data: dict = {"cameras": cameras}
    if global_retention is not None:
        cfg_data["retention"] = global_retention
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))
    app_cfg = load_config(cfg_path)

    from rtsp_warden.web.app import create_app
    from rtsp_warden.web.config import WebSettings

    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=app_cfg, runtime_provider=lambda: None)
    app.state.config_path = str(cfg_path)
    client = TestClient(app)

    # Login as admin with CSRF
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

    # Teardown DB
    yield client, cfg_path
    reset_engine()


@pytest.fixture
def admin_client_retention(tmp_path: Path) -> tuple[TestClient, Path]:
    """Fixture that yields an authenticated admin client with a basic config."""
    cameras = [
        {
            "name": "front_door",
            "main_url": "rtsp://x/main",
            "sub_url": "rtsp://x/sub",
        },
    ]
    yield from _make_admin_client(tmp_path, cameras)


@pytest.fixture
def admin_client_with_override(tmp_path: Path) -> tuple[TestClient, Path]:
    """Fixture with a camera that already has a per-camera retention override."""
    cameras = [
        {
            "name": "front_door",
            "main_url": "rtsp://x/main",
            "sub_url": "rtsp://x/sub",
            "retention": {"max_days": 90, "max_gb": 100.0},
        },
    ]
    yield from _make_admin_client(tmp_path, cameras)


def _get_csrf(client: TestClient) -> str:
    """Extract CSRF token from client cookies."""
    return client.cookies.get("warden_csrf", "")


class TestCameraRetentionWebRoute:
    """Tests for the POST /cameras/{name}/retention web route."""

    def test_save_per_camera_retention(
        self, admin_client_retention: tuple[TestClient, Path]
    ) -> None:
        """POST saves per-camera retention override to config.yaml."""
        client, cfg_path = admin_client_retention
        csrf = _get_csrf(client)

        resp = client.post(
            "/cameras/front_door/retention",
            data={
                "max_days": "7",
                "max_gb": "25.0",
                "keep_last_n": "10",
                "cleanup_interval_seconds": "600",
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert resp.status_code in (200, 303, 302)

        # Reload config from disk and verify
        reloaded = load_config(cfg_path)
        cam = reloaded.cameras[0]
        assert cam.retention is not None
        assert cam.retention.max_days == 7
        assert cam.retention.max_gb == 25.0
        assert cam.retention.keep_last_n == 10

    def test_reset_to_global_clears_override(
        self, admin_client_with_override: tuple[TestClient, Path]
    ) -> None:
        """POST with action=reset clears the per-camera retention override."""
        client, cfg_path = admin_client_with_override
        csrf = _get_csrf(client)

        resp = client.post(
            "/cameras/front_door/retention",
            data={"action": "reset"},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert resp.status_code in (200, 303, 302)

        # Reload config from disk and verify override is gone
        reloaded = load_config(cfg_path)
        cam = reloaded.cameras[0]
        assert cam.retention is None

    def test_retention_route_404_for_unknown_camera(
        self, admin_client_retention: tuple[TestClient, Path]
    ) -> None:
        """POST /cameras/{name}/retention returns 404 for unknown camera."""
        client, cfg_path = admin_client_retention
        csrf = _get_csrf(client)

        resp = client.post(
            "/cameras/nonexistent/retention",
            data={"max_days": "7"},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert resp.status_code == 404
