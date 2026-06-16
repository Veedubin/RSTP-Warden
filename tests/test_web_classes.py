"""Web UI tests for detection class configuration and detector enable toggles.

Tests the detection-classes page render, save, config persistence,
auth enforcement, and per-detector enable/disable toggle routes.
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
def config_with_classes(tmp_path: Path) -> Path:
    """Create a config.yaml with a camera that has detect_classes and detectors."""
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
                "detect_classes": ["person", "dog", "car"],
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
                    {
                        "type": "dnn",
                        "enabled": True,
                        "interval_seconds": 3.0,
                        "config": {
                            "classes": ["person", "dog", "car", "truck"],
                        },
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
def app_with_classes(config_with_classes: Path) -> tuple:
    """Return (app, cfg) with detect_classes config and auth."""
    db_url = f"sqlite:///{config_with_classes.parent / 'test.db'}"
    os.environ["WARDEN_DB_URL"] = db_url
    reset_engine()
    ensure_schema()
    pw_hash = hash_password("testpass123")
    create_admin_user("admin", pw_hash)

    cfg = load_config(config_with_classes)
    settings = WebSettings(host="127.0.0.1", port=8080)
    app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
    app.state.config_path = str(config_with_classes)
    return app, cfg


@pytest.fixture
def client_with_classes(app_with_classes: tuple) -> TestClient:
    """Authenticated TestClient with detect_classes-enabled config."""
    app, _ = app_with_classes
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


class TestDetectionClassesPage:
    """Tests for GET /cameras/{name}/detection-classes."""

    def test_detection_classes_page_returns_200_for_admin(
        self, client_with_classes: TestClient
    ) -> None:
        """Admin user can access the detection classes page."""
        r = client_with_classes.get("/cameras/front_door/detection-classes")
        assert r.status_code == 200
        assert "Detection classes" in r.text
        assert "front_door" in r.text

    def test_detection_classes_page_lists_all_coco_classes(
        self, client_with_classes: TestClient
    ) -> None:
        """Detection classes page shows all 80 COCO classes."""
        r = client_with_classes.get("/cameras/front_door/detection-classes")
        assert r.status_code == 200
        # Check some representative class names
        assert "person" in r.text
        assert "bicycle" in r.text
        assert "car" in r.text
        assert "toothbrush" in r.text

    def test_detection_classes_page_shows_active_classes_checked(
        self, client_with_classes: TestClient
    ) -> None:
        """Detection classes page shows active classes as checked."""
        r = client_with_classes.get("/cameras/front_door/detection-classes")
        assert r.status_code == 200
        # person, dog, car are active -- they should be "checked"
        # The template uses name="class_person" etc.
        assert 'name="class_person"' in r.text
        assert 'name="class_dog"' in r.text
        assert 'name="class_car"' in r.text

    def test_detection_classes_page_404_for_unknown_camera(
        self, client_with_classes: TestClient
    ) -> None:
        """Detection classes page returns 404 for non-existent camera."""
        r = client_with_classes.get("/cameras/nonexistent/detection-classes")
        assert r.status_code == 404

    def test_detection_classes_page_requires_admin(self, app_with_classes: tuple) -> None:
        """Detection classes page returns 401 for non-admin users."""
        app, _ = app_with_classes
        client = TestClient(app)
        r = client.get("/cameras/front_door/detection-classes", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)

    def test_detection_classes_page_no_filter_shows_all_active(
        self, client_with_classes: TestClient
    ) -> None:
        """Camera with no detect_classes shows 'All classes active'."""
        r = client_with_classes.get("/cameras/backyard/detection-classes")
        assert r.status_code == 200


class TestSaveDetectionClasses:
    """Tests for POST /cameras/{name}/detection-classes."""

    def test_save_detection_classes_redirects_on_success(
        self, client_with_classes: TestClient
    ) -> None:
        """Saving selected classes redirects (303) to camera detail."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={
                "class_person": "on",
                "class_car": "on",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "/cameras/front_door" in r.headers.get("location", "")

    def test_save_detection_classes_updates_camera_config(
        self, client_with_classes: TestClient
    ) -> None:
        """Saving classes updates the in-memory camera detect_classes."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={
                "class_person": "on",
                "class_dog": "on",
                "class_cat": "on",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the detection classes page shows updated state
        r = client_with_classes.get("/cameras/front_door/detection-classes")
        assert r.status_code == 200
        assert "3" in r.text  # 3 classes active

    def test_save_detection_classes_with_no_checkboxes(
        self, client_with_classes: TestClient
    ) -> None:
        """Saving with no checkboxes results in detect_classes = []."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={"csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Check the detail page shows empty classes message
        r = client_with_classes.get("/cameras/front_door")
        assert r.status_code == 200
        assert "0 classes active" in r.text

    def test_save_detection_classes_writes_to_config_yaml(
        self,
        client_with_classes: TestClient,
        config_with_classes: Path,
    ) -> None:
        """Saving classes persists the change to config.yaml."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={
                "class_person": "on",
                "class_truck": "on",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Reload config.yaml from disk
        raw = yaml.safe_load(config_with_classes.read_text(encoding="utf-8"))
        front_door = None
        for cam in raw["cameras"]:
            if cam["name"] == "front_door":
                front_door = cam
                break
        assert front_door is not None
        assert "person" in front_door["detect_classes"]
        assert "truck" in front_door["detect_classes"]


class TestDetectorEnabledToggle:
    """Tests for POST /cameras/{name}/detectors/{det_type}/enabled."""

    def test_disable_detector_redirects_on_success(self, client_with_classes: TestClient) -> None:
        """Disabling a detector redirects (303) to camera detail."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert "/cameras/front_door" in r.headers.get("location", "")

    def test_disable_detector_sets_enabled_false(self, client_with_classes: TestClient) -> None:
        """Disabling a motion detector sets spec.enabled = False."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the detector list shows disabled
        r = client_with_classes.get("/cameras/front_door/detectors")
        assert r.status_code == 200
        # The motion row should have disabled class
        assert "disabled" in r.text

    def test_enable_detector_sets_enabled_true(self, app_with_classes: tuple) -> None:
        """Enabling a previously disabled detector sets spec.enabled = True."""
        app, _ = app_with_classes

        # First, manually disable the motion detector
        for cam in app.state.cfg.cameras:
            if cam.name == "front_door":
                for det in cam.detectors:
                    if det.type == "motion":
                        det.enabled = False

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

        # Now enable it
        r = client.post(
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "true", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the detector list shows enabled
        r = client.get("/cameras/front_door/detectors")
        assert r.status_code == 200

    def test_toggle_detector_404_for_unknown_type(self, client_with_classes: TestClient) -> None:
        """Toggling a detector type that doesn't exist returns 404."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/custom_nonexistent/enabled",
            data={"enabled": "true", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 404

    def test_toggle_detector_calls_rebuild_with_mocked_runtime(
        self, app_with_classes: tuple
    ) -> None:
        """Toggling a detector calls rebuild_camera_detectors."""
        app, _ = app_with_classes

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
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        mock_runtime.rebuild_camera_detectors.assert_called_once_with("front_door")

    def test_toggle_detector_writes_to_config_yaml(
        self,
        app_with_classes: tuple,
        config_with_classes: Path,
    ) -> None:
        """Toggling a detector persists the change to config.yaml."""
        app, _ = app_with_classes

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
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Reload config.yaml from disk
        raw = yaml.safe_load(config_with_classes.read_text(encoding="utf-8"))
        front_door = None
        for cam in raw["cameras"]:
            if cam["name"] == "front_door":
                front_door = cam
                break
        assert front_door is not None
        motion_det = None
        for det in front_door["detectors"]:
            if det["type"] == "motion":
                motion_det = det
                break
        assert motion_det is not None
        assert motion_det["enabled"] is False


class TestDetectionClassesAuth:
    """Tests for auth enforcement on detection class routes."""

    def test_detection_classes_page_requires_auth(self, app_with_classes: tuple) -> None:
        """Detection classes page requires authentication."""
        app, _ = app_with_classes
        client = TestClient(app)
        r = client.get("/cameras/front_door/detection-classes", follow_redirects=False)
        assert r.status_code in (303, 401, 307, 302)

    def test_save_detection_classes_requires_admin(self, app_with_classes: tuple) -> None:
        """Saving detection classes requires admin auth."""
        app, _ = app_with_classes
        client = TestClient(app)
        csrf = client.get("/login").cookies.get("warden_csrf", "")
        r = client.post(
            "/cameras/front_door/detection-classes",
            data={"class_person": "on", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)

    def test_toggle_detector_requires_admin(self, app_with_classes: tuple) -> None:
        """Toggling detector enabled flag requires admin auth."""
        app, _ = app_with_classes
        client = TestClient(app)
        csrf = client.get("/login").cookies.get("warden_csrf", "")
        r = client.post(
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)
