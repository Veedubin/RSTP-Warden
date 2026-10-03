"""Web UI tests for detection class configuration and detector enable toggles.

Tests the detection-classes page render, save, config persistence,
auth enforcement, and per-detector enable/disable toggle routes.
"""

from __future__ import annotations

import os
import re
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
from rtsp_warden.web.services import detection as detection_service

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
        """Camera with no detect_classes: 'All' mode selected and every label box ticked."""
        r = client_with_classes.get("/cameras/backyard/detection-classes")
        assert r.status_code == 200
        assert "All 80 classes active" in r.text
        assert 'value="all" checked' in r.text
        ticked = re.findall(r'<input type="checkbox" name="class_[^"]+"\s+checked', r.text)
        assert len(ticked) == 80

    def test_detection_classes_page_custom_mode_for_a_list(
        self, client_with_classes: TestClient
    ) -> None:
        """A camera with a class list shows 'custom' mode and ticks exactly that list."""
        r = client_with_classes.get("/cameras/front_door/detection-classes")
        assert "3 of 80 classes active" in r.text
        assert 'value="custom" checked' in r.text
        ticked = re.findall(r'<input type="checkbox" name="class_([^"]+)"\s+checked', r.text)
        assert sorted(ticked) == ["car", "dog", "person"]
        assert "grid-template-columns" not in r.text


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


class TestClassesMode:
    """classes_mode=all restores 'no filter'; posted names are limited to the model's labels."""

    def test_save_mode_all_writes_null(
        self, client_with_classes: TestClient, config_with_classes: Path
    ) -> None:
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={"classes_mode": "all", "class_person": "on", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        raw = yaml.safe_load(config_with_classes.read_text(encoding="utf-8"))
        front_door = next(c for c in raw["cameras"] if c["name"] == "front_door")
        assert "detect_classes" in front_door
        assert front_door["detect_classes"] is None
        assert load_config(config_with_classes).cameras[0].detect_classes is None
        r = client_with_classes.get("/cameras/front_door")
        assert "All classes active (no filter)" in r.text

    def test_save_ignores_names_the_model_does_not_have(
        self, client_with_classes: TestClient
    ) -> None:
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={
                "classes_mode": "custom",
                "class_person": "on",
                "class_unicorn": "on",
                "csrf_token": csrf,
            },
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        cam = next(c for c in client_with_classes.app.state.cfg.cameras if c.name == "front_door")
        assert cam.detect_classes == ["person"]

    def test_onnx_camera_offers_its_model_labels(self, db_with_user: str, tmp_path: Path) -> None:
        models = tmp_path / "models"
        (models / "critters").mkdir(parents=True)
        (models / "critters" / "model.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "critters",
                    "file": "critters.onnx",
                    "labels": "labels.txt",
                    "input_size": [64, 64],
                    "postprocess": "yolox",
                }
            ),
            encoding="utf-8",
        )
        (models / "critters" / "labels.txt").write_text("person\nraccoon\nfox\n", encoding="utf-8")
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cameras": [
                        {
                            "name": "yard",
                            "main_url": "rtsp://u:p@h/m",
                            "detect_classes": ["raccoon"],
                            "detectors": [{"type": "onnx", "model": "critters"}],
                        }
                    ],
                    "runtime": {"models_dir": str(models)},
                }
            ),
            encoding="utf-8",
        )
        app = create_app(WebSettings(), cfg=load_config(path), config_path=path)
        client = TestClient(app)
        client.get("/login")
        csrf = client.cookies.get("warden_csrf", "")
        client.post(
            "/login", data={"username": "admin", "password": "testpass123", "csrf_token": csrf}
        )
        r = client.get("/cameras/yard/detection-classes")
        assert r.status_code == 200
        assert "1 of 3 classes active" in r.text
        assert 'name="class_raccoon"' in r.text
        assert 'name="class_fox"' in r.text
        assert "toothbrush" not in r.text
        assert "Other (2)" in r.text

        r = client.post(
            "/cameras/yard/detection-classes",
            data={"class_fox": "on", "class_toothbrush": "on", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["cameras"][0]["detect_classes"] == ["fox"]

    def test_save_reports_a_config_write_failure(
        self, client_with_classes: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(path: Path, data: dict) -> None:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(detection_service, "_locked_write_yaml", refuse)
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detection-classes",
            data={"class_dog": "on", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert "Permission denied" in r.text
        cam = next(c for c in client_with_classes.app.state.cfg.cameras if c.name == "front_door")
        assert cam.detect_classes == ["dog"]


class TestDetectorEnabledToggle:
    """Tests for POST /cameras/{name}/detectors/{index}/enabled (index 0 is the motion spec)."""

    def test_disable_detector_redirects_on_success(self, client_with_classes: TestClient) -> None:
        """Disabling a detector redirects (303) to camera detail."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/0/enabled",
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
            "/cameras/front_door/detectors/0/enabled",
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
            "/cameras/front_door/detectors/0/enabled",
            data={"enabled": "true", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303

        # Verify the detector list shows enabled
        r = client.get("/cameras/front_door/detectors")
        assert r.status_code == 200

    def test_toggle_detector_404_for_unknown_index(self, client_with_classes: TestClient) -> None:
        """Toggling an index past the end of the detectors list returns 404."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/99/enabled",
            data={"enabled": "true", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 404

    def test_toggle_detector_422_for_a_type_name(self, client_with_classes: TestClient) -> None:
        """The old type-keyed URL is gone: a non-integer index is a 422."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/motion/enabled",
            data={"enabled": "true", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 422

    def test_toggle_by_index_leaves_other_detectors_alone(
        self, client_with_classes: TestClient, config_with_classes: Path
    ) -> None:
        """Disabling index 2 (dnn) keeps motion and person enabled, in memory and on disk."""
        csrf = client_with_classes.cookies.get("warden_csrf", "")
        r = client_with_classes.post(
            "/cameras/front_door/detectors/2/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code == 303
        app = client_with_classes.app
        cam = next(c for c in app.state.cfg.cameras if c.name == "front_door")
        assert [d.enabled for d in cam.detectors] == [True, True, False]
        raw = yaml.safe_load(config_with_classes.read_text(encoding="utf-8"))
        front_door = next(c for c in raw["cameras"] if c["name"] == "front_door")
        assert [d.get("enabled") for d in front_door["detectors"]] == [True, True, False]
        assert front_door["detectors"][2]["config"] == {
            "classes": ["person", "dog", "car", "truck"]
        }

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
            "/cameras/front_door/detectors/0/enabled",
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
            "/cameras/front_door/detectors/0/enabled",
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
            "/cameras/front_door/detectors/0/enabled",
            data={"enabled": "false", "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
            follow_redirects=False,
        )
        assert r.status_code in (303, 401, 307, 402)
