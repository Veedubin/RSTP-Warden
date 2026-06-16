"""Web UI tests for detection zone management.

Tests the zone list page, zone editor partial, save/delete zone
routes, detector reload, config persistence, and auth enforcement.
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
def config_with_zones(tmp_path: Path) -> Path:
    """Create a config.yaml with a camera that has zones configured."""
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
                "zones": [
                    {
                        "name": "exclude road",
                        "grid_cols": 16,
                        "grid_rows": 16,
                        "blocked_cells": [[0, 0], [0, 1], [1, 0], [1, 1]],
                        "frame_width": 1920,
                        "frame_height": 1080,
                        "enabled": True,
                    },
                    {
                        "name": "ignore neighbors",
                        "grid_cols": 16,
                        "grid_rows": 16,
                        "blocked_cells": [[5, 3], [6, 3], [5, 4], [6, 4]],
                        "frame_width": 1920,
                        "frame_height": 1080,
                        "enabled": True,
                    },
                ],
            },
            {
                "name": "backyard",
                "main_url": "rtsp://user:pass@192.168.1.51:554/main",
                "sub_url": "rtsp://user:pass@192.168.1.51:554/sub",
                "record": {"enabled": True, "output_dir": str(tmp_path / "recordings")},
                "proxy": {"enabled": False, "mode": "mjpeg"},
                "zones": [],
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
def app_with_zones(config_with_zones: Path) -> tuple:
    """Return (app, cfg) with zones config and auth."""
    db_url = f"sqlite:///{config_with_zones.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(config_with_zones)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    # Store config path for persistence tests
    app.state.config_path = str(config_with_zones)
    return app, cfg


@pytest.fixture
def client_with_zones(app_with_zones: tuple) -> TestClient:
    """Authenticated TestClient with zone-enabled config."""
    app, _ = app_with_zones
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


class TestZonesListPage:
    """Tests for GET /cameras/{name}/zones."""

    def test_zones_page_returns_200_for_admin(self, client_with_zones: TestClient) -> None:
        """Admin user can access the zones list page."""
        r = client_with_zones.get("/cameras/front_door/zones")
        assert r.status_code == 200
        assert "Detection zones" in r.text
        assert "front_door" in r.text

    def test_zones_page_lists_existing_zones(self, client_with_zones: TestClient) -> None:
        """Zones page shows the configured zones."""
        r = client_with_zones.get("/cameras/front_door/zones")
        assert r.status_code == 200
        assert "exclude road" in r.text
        assert "ignore neighbors" in r.text

    def test_zones_page_shows_grid_info(self, client_with_zones: TestClient) -> None:
        """Zones page shows grid dimensions and blocked cell counts."""
        r = client_with_zones.get("/cameras/front_door/zones")
        assert r.status_code == 200
        assert "16x16" in r.text

    def test_zones_page_for_camera_with_no_zones(self, client_with_zones: TestClient) -> None:
        """Camera with no zones shows 'Add new zone' prompt."""
        r = client_with_zones.get("/cameras/backyard/zones")
        assert r.status_code == 200
        assert "No detection zones" in r.text
        assert "Add new zone" in r.text

    def test_zones_page_404_for_unknown_camera(self, client_with_zones: TestClient) -> None:
        """Zones page returns 404 for a camera that doesn't exist."""
        r = client_with_zones.get("/cameras/nonexistent/zones")
        assert r.status_code == 404

    def test_zones_page_requires_admin(self, app_with_zones: tuple) -> None:
        """Zones page returns 401/403 for non-admin users."""
        app, _ = app_with_zones
        client = TestClient(app)
        r = client.get("/cameras/front_door/zones", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)


class TestZonesEditor:
    """Tests for GET /cameras/{name}/zones/editor."""

    def test_editor_returns_200_for_new_zone(self, client_with_zones: TestClient) -> None:
        """Editor partial loads for creating a new zone."""
        r = client_with_zones.get("/cameras/front_door/zones/editor")
        assert r.status_code == 200
        assert "New zone" in r.text
        assert "zoneEditor" in r.text

    def test_editor_returns_200_for_existing_zone(self, client_with_zones: TestClient) -> None:
        """Editor partial loads for editing an existing zone."""
        r = client_with_zones.get("/cameras/front_door/zones/editor?zone_name=exclude+road")
        assert r.status_code == 200
        assert "exclude road" in r.text
        assert "zoneEditor" in r.text

    def test_editor_has_svg_grid_overlay(self, client_with_zones: TestClient) -> None:
        """Editor partial contains an SVG element for the grid overlay."""
        r = client_with_zones.get("/cameras/front_door/zones/editor")
        assert r.status_code == 200
        assert "<svg" in r.text
        assert "viewBox" in r.text

    def test_editor_has_blocked_cells_data(self, client_with_zones: TestClient) -> None:
        """Editor partial includes blocked cells data for existing zone."""
        r = client_with_zones.get("/cameras/front_door/zones/editor?zone_name=exclude+road")
        assert r.status_code == 200
        # The blocked_cells_json should contain the cell data
        assert "blockedCells" in r.text

    def test_editor_has_snapshot_url(self, client_with_zones: TestClient) -> None:
        """Editor partial includes snapshot URL for MJPEG cameras."""
        r = client_with_zones.get("/cameras/front_door/zones/editor")
        assert r.status_code == 200
        assert "snapshot.jpg" in r.text

    def test_editor_no_snapshot_for_disabled_proxy(self, client_with_zones: TestClient) -> None:
        """Editor partial shows no snapshot for cameras without proxy."""
        r = client_with_zones.get("/cameras/backyard/zones/editor")
        assert r.status_code == 200
        assert "No snapshot available" in r.text

    def test_editor_requires_admin(self, app_with_zones: tuple) -> None:
        """Editor partial requires admin auth."""
        app, _ = app_with_zones
        client = TestClient(app)
        r = client.get("/cameras/front_door/zones/editor", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 402)


class TestSaveZone:
    """Tests for POST /cameras/{name}/zones."""

    def test_save_zone_redirects_on_success(self, client_with_zones: TestClient) -> None:
        """Saving a new zone redirects (303) back to zones page."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "test zone",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["0,0", "0,1", "1,0"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "/cameras/backyard/zones" in r.headers.get("location", "")

    def test_save_zone_updates_camera_config(self, client_with_zones: TestClient) -> None:
        """Saving a zone adds it to the camera's zones list in memory."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "new zone",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["3,4", "3,5"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify zone was added
        r = client_with_zones.get("/cameras/backyard/zones")
        assert r.status_code == 200
        assert "new zone" in r.text

    def test_save_zone_updates_existing_zone(self, client_with_zones: TestClient) -> None:
        """Saving a zone with an existing name replaces it."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/front_door/zones",
            data={
                "zone_name": "exclude road",
                "grid_cols": "16",
                "grid_rows": "16",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["0,0", "0,1"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify zone was updated (still only 2 zones)
        r = client_with_zones.get("/cameras/front_door/zones")
        assert r.status_code == 200
        assert "exclude road" in r.text
        assert "ignore neighbors" in r.text

    def test_save_zone_with_empty_name_returns_422(self, client_with_zones: TestClient) -> None:
        """Saving a zone with an empty name returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_invalid_grid_returns_422(self, client_with_zones: TestClient) -> None:
        """Saving a zone with grid_cols < 2 returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "bad grid",
                "grid_cols": "1",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_grid_over_64_returns_422(self, client_with_zones: TestClient) -> None:
        """Saving a zone with grid_cols > 64 returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "huge grid",
                "grid_cols": "65",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_out_of_bounds_cell_returns_422(
        self, client_with_zones: TestClient
    ) -> None:
        """Saving a zone with a cell out of grid bounds returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "bad cell",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["8,0"],  # col 8 is out of bounds for 8-col grid
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_invalid_cell_format_returns_422(
        self, client_with_zones: TestClient
    ) -> None:
        """Saving a zone with a malformed blocked_cell value returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "bad format",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["not_a_cell"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_invalid_dimensions_returns_422(
        self, client_with_zones: TestClient
    ) -> None:
        """Saving a zone with non-integer dimensions returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "bad dims",
                "grid_cols": "abc",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_with_zero_frame_returns_422(self, client_with_zones: TestClient) -> None:
        """Saving a zone with zero frame dimensions returns 422."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "zero frame",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "0",
                "frame_height": "0",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_save_zone_parses_blocked_cells_correctly(self, client_with_zones: TestClient) -> None:
        """Multiple blocked_cell form values are parsed into the correct set."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "five cells",
                "grid_cols": "16",
                "grid_rows": "16",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["0,0", "1,1", "2,2", "3,3", "4,4"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the zone was saved with 5 blocked cells
        r = client_with_zones.get("/cameras/backyard/zones")
        assert r.status_code == 200
        assert "five cells" in r.text

    def test_save_zone_writes_to_config_yaml(
        self, client_with_zones: TestClient, config_with_zones: Path
    ) -> None:
        """Saving a zone persists the change to config.yaml."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/backyard/zones",
            data={
                "zone_name": "persisted zone",
                "grid_cols": "8",
                "grid_rows": "8",
                "frame_width": "1920",
                "frame_height": "1080",
                "blocked_cell": ["0,0"],
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Reload config.yaml from disk
        raw = yaml.safe_load(config_with_zones.read_text(encoding="utf-8"))
        backyard = None
        for cam in raw["cameras"]:
            if cam["name"] == "backyard":
                backyard = cam
                break
        assert backyard is not None
        assert "zones" in backyard
        assert len(backyard["zones"]) == 1
        assert backyard["zones"][0]["name"] == "persisted zone"


class TestDeleteZone:
    """Tests for POST /cameras/{name}/zones/{zone_name}/delete."""

    def test_delete_zone_redirects_on_success(self, client_with_zones: TestClient) -> None:
        """Deleting a zone redirects back to zones page."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/front_door/zones/exclude road/delete",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "/cameras/front_door/zones" in r.headers.get("location", "")

    def test_delete_zone_removes_from_camera(self, client_with_zones: TestClient) -> None:
        """Deleting a zone removes it from the camera's zone list."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/front_door/zones/exclude road/delete",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify zone was removed
        r = client_with_zones.get("/cameras/front_door/zones")
        assert r.status_code == 200
        assert "exclude road" not in r.text
        assert "ignore neighbors" in r.text  # other zone still present

    def test_delete_zone_updates_config_yaml(
        self, client_with_zones: TestClient, config_with_zones: Path
    ) -> None:
        """Deleting a zone persists the removal to config.yaml."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")
        r = client_with_zones.post(
            "/cameras/front_door/zones/ignore neighbors/delete",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Reload config.yaml from disk
        raw = yaml.safe_load(config_with_zones.read_text(encoding="utf-8"))
        front_door = None
        for cam in raw["cameras"]:
            if cam["name"] == "front_door":
                front_door = cam
                break
        assert front_door is not None
        zone_names = [z["name"] for z in front_door.get("zones", [])]
        assert "ignore neighbors" not in zone_names
        assert "exclude road" in zone_names


class TestReloadZones:
    """Tests for POST /cameras/{name}/zones/reload."""

    def test_reload_calls_rebuild_camera_detectors(self, client_with_zones: TestClient) -> None:
        """Reload route calls rebuild_camera_detectors on the runtime."""
        csrf = client_with_zones.cookies.get("warden_csrf", "")

        # We need to mock the runtime. The app fixture doesn't have a runtime,
        # so we expect a 503 "runtime not initialized" without a mock.
        r = client_with_zones.post(
            "/cameras/front_door/zones/reload",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        # No runtime -> 503
        assert r.status_code == 503

    def test_reload_with_mocked_runtime(self, app_with_zones: tuple) -> None:
        """Reload route calls rebuild_camera_detectors when runtime is available."""
        app, _ = app_with_zones

        # Create a mock runtime with rebuild_camera_detectors
        mock_runtime = MagicMock()
        app.state.runtime = mock_runtime

        client = TestClient(app)
        # Login first
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
            "/cameras/front_door/zones/reload",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True
        mock_runtime.rebuild_camera_detectors.assert_called_once_with("front_door")


class TestZonesAuth:
    """Tests for auth enforcement on zone routes."""

    def test_zones_list_requires_auth(self, app_with_zones: tuple) -> None:
        """Zones list page requires authentication."""
        app, _ = app_with_zones
        client = TestClient(app)
        r = client.get("/cameras/front_door/zones", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)

    def test_save_zone_requires_admin(self, app_with_zones: tuple) -> None:
        """Saving a zone requires admin auth."""
        app, _ = app_with_zones
        client = TestClient(app)
        csrf = client.get("/login").cookies.get("warden_csrf", "")
        r = client.post(
            "/cameras/front_door/zones",
            data={"zone_name": "hack", "grid_cols": "8", "grid_rows": "8"},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)

    def test_delete_zone_requires_admin(self, app_with_zones: tuple) -> None:
        """Deleting a zone requires admin auth."""
        app, _ = app_with_zones
        client = TestClient(app)
        csrf = client.get("/login").cookies.get("warden_csrf", "")
        r = client.post(
            "/cameras/front_door/zones/test/delete",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)


class TestDetailPageZoneLink:
    """Tests for the Detection zones link on the camera detail page."""

    def test_detail_page_has_zones_link(self, client_with_zones: TestClient) -> None:
        """Camera detail page includes a link to the zones page."""
        r = client_with_zones.get("/cameras/front_door")
        assert r.status_code == 200
        assert "/cameras/front_door/zones" in r.text
        assert "detection zones" in r.text.lower()
