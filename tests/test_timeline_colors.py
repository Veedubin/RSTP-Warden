"""Tests for Feature 2: Colored Timeline Markers by Object Type.

Validates:
- categorize_object() for all COCO labels and heuristic fallbacks
- color_for_object_type() and OBJECT_TYPE_COLORS
- TimelineEvent dataclass has object_type and color fields
- build_timeline() populates object_type and color correctly
- API endpoint returns object_type and color in event JSON
- Severity-based styling still works for motion-only events
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
from rtsp_warden.db.models import Camera, Event, Recording
from rtsp_warden.db.schema import ensure_schema
from rtsp_warden.detectors.categorize import categorize_object
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.timeline import (
    TimelineData,
    TimelineEvent,
    build_timeline,
)
from rtsp_warden.web.services.timeline_colors import (
    OBJECT_TYPE_COLORS,
    color_for_object_type,
)

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Unit tests: categorize_object
# ---------------------------------------------------------------------------


class TestCategorizeObject:
    """Tests for the categorize_object function."""

    # --- Exact label matches (COCO classes) ---

    def test_categorize_person(self) -> None:
        assert categorize_object("person") == "person"

    def test_categorize_pet_cat(self) -> None:
        assert categorize_object("cat") == "pet"

    def test_categorize_pet_dog(self) -> None:
        assert categorize_object("dog") == "pet"

    def test_categorize_critter_bird(self) -> None:
        assert categorize_object("bird") == "critter"

    def test_categorize_critter_deer(self) -> None:
        assert categorize_object("deer") == "critter"

    def test_categorize_critter_raccoon(self) -> None:
        assert categorize_object("raccoon") == "critter"

    def test_categorize_critter_squirrel(self) -> None:
        assert categorize_object("squirrel") == "critter"

    def test_categorize_critter_rabbit(self) -> None:
        assert categorize_object("rabbit") == "critter"

    def test_categorize_critter_fox(self) -> None:
        assert categorize_object("fox") == "critter"

    def test_categorize_critter_coyote(self) -> None:
        assert categorize_object("coyote") == "critter"

    def test_categorize_critter_skunk(self) -> None:
        assert categorize_object("skunk") == "critter"

    def test_categorize_critter_mouse(self) -> None:
        assert categorize_object("mouse") == "critter"

    def test_categorize_critter_bear(self) -> None:
        assert categorize_object("bear") == "critter"

    def test_categorize_critter_horse(self) -> None:
        assert categorize_object("horse") == "critter"

    def test_categorize_critter_sheep(self) -> None:
        assert categorize_object("sheep") == "critter"

    def test_categorize_critter_cow(self) -> None:
        assert categorize_object("cow") == "critter"

    def test_categorize_critter_butterfly(self) -> None:
        assert categorize_object("butterfly") == "critter"

    def test_categorize_vehicle_car(self) -> None:
        assert categorize_object("car") == "vehicle"

    def test_categorize_vehicle_truck(self) -> None:
        assert categorize_object("truck") == "vehicle"

    def test_categorize_vehicle_bus(self) -> None:
        assert categorize_object("bus") == "vehicle"

    def test_categorize_vehicle_motorcycle(self) -> None:
        assert categorize_object("motorcycle") == "vehicle"

    def test_categorize_vehicle_bicycle(self) -> None:
        assert categorize_object("bicycle") == "vehicle"

    def test_categorize_vehicle_airplane(self) -> None:
        assert categorize_object("airplane") == "vehicle"

    def test_categorize_vehicle_train(self) -> None:
        assert categorize_object("train") == "vehicle"

    def test_categorize_vehicle_boat(self) -> None:
        assert categorize_object("boat") == "vehicle"

    # --- Other / motion / unknown ---

    def test_categorize_other_motion(self) -> None:
        assert categorize_object("motion") == "other"

    def test_categorize_other_unknown(self) -> None:
        assert categorize_object("unknown_thing") == "other"

    def test_categorize_other_empty(self) -> None:
        assert categorize_object("") == "other"

    # --- Case insensitive ---

    def test_categorize_case_insensitive_person(self) -> None:
        assert categorize_object("PERSON") == "person"

    def test_categorize_case_insensitive_cat(self) -> None:
        assert categorize_object("Cat") == "pet"

    def test_categorize_case_insensitive_car(self) -> None:
        assert categorize_object("CAR") == "vehicle"

    # --- Heuristic fallback ---

    def test_categorize_heuristic_human(self) -> None:
        assert categorize_object("human_body") == "person"

    def test_categorize_heuristic_face(self) -> None:
        assert categorize_object("face_detector") == "person"

    def test_categorize_heuristic_pet(self) -> None:
        assert categorize_object("pet_detector") == "pet"

    def test_categorize_heuristic_vehicle(self) -> None:
        assert categorize_object("vehicle_alert") == "vehicle"

    def test_categorize_heuristic_critter(self) -> None:
        assert categorize_object("critter_alert") == "critter"

    # --- Whitespace handling ---

    def test_categorize_strips_whitespace(self) -> None:
        assert categorize_object("  person  ") == "person"


# ---------------------------------------------------------------------------
# Unit tests: color_for_object_type
# ---------------------------------------------------------------------------


class TestColorForObjectType:
    """Tests for the color_for_object_type function."""

    def test_color_for_person(self) -> None:
        assert color_for_object_type("person") == "#e74c3c"

    def test_color_for_pet(self) -> None:
        assert color_for_object_type("pet") == "#3498db"

    def test_color_for_critter(self) -> None:
        assert color_for_object_type("critter") == "#f1c40f"

    def test_color_for_vehicle(self) -> None:
        assert color_for_object_type("vehicle") == "#e67e22"

    def test_color_for_other(self) -> None:
        assert color_for_object_type("other") == "#7f8c8d"

    def test_color_for_unknown_falls_back_to_gray(self) -> None:
        assert color_for_object_type("nonexistent") == "#7f8c8d"

    def test_color_map_has_all_categories(self) -> None:
        expected = {"person", "pet", "critter", "vehicle", "other"}
        assert set(OBJECT_TYPE_COLORS.keys()) == expected

    def test_color_map_values_are_hex(self) -> None:
        for key, value in OBJECT_TYPE_COLORS.items():
            assert value.startswith("#"), f"{key} color {value} is not hex"
            assert len(value) == 7, f"{key} color {value} is not 7 chars"


# ---------------------------------------------------------------------------
# Unit tests: TimelineEvent dataclass
# ---------------------------------------------------------------------------


class TestTimelineEventFields:
    """Tests for the TimelineEvent dataclass having object_type and color."""

    def test_timeline_event_has_object_type(self) -> None:
        evt = TimelineEvent(
            id=1,
            event_type="person",
            severity="warn",
            ts_unix=100.0,
            object_type="person",
            color="#e74c3c",
        )
        assert evt.object_type == "person"

    def test_timeline_event_has_color(self) -> None:
        evt = TimelineEvent(
            id=1,
            event_type="car",
            severity="info",
            ts_unix=200.0,
            object_type="vehicle",
            color="#e67e22",
        )
        assert evt.color == "#e67e22"

    def test_timeline_event_defaults(self) -> None:
        evt = TimelineEvent(
            id=1,
            event_type="motion",
            severity="info",
            ts_unix=300.0,
        )
        assert evt.object_type == "other"
        assert evt.color == "#7f8c8d"

    def test_timeline_event_backward_compat(self) -> None:
        """Existing construction without new fields still works (defaults)."""
        evt = TimelineEvent(id=1, event_type="motion", severity="info", ts_unix=300.0)
        assert evt.object_type == "other"
        assert evt.color == "#7f8c8d"

    def test_timeline_data_with_events(self) -> None:
        td = TimelineData(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_ts=1000.0,
            end_ts=2000.0,
            events=[
                TimelineEvent(
                    id=1,
                    event_type="person",
                    severity="warn",
                    ts_unix=1002.0,
                    object_type="person",
                    color="#e74c3c",
                ),
            ],
        )
        assert len(td.events) == 1
        assert td.events[0].object_type == "person"
        assert td.events[0].color == "#e74c3c"


# ---------------------------------------------------------------------------
# Integration tests: build_timeline populates object_type and color
# ---------------------------------------------------------------------------


class TestBuildTimelineColors:
    """Tests that build_timeline correctly populates object_type and color."""

    def test_build_timeline_populates_object_type_for_person(self, tmp_path: Path) -> None:
        """Person events get object_type='person' and correct color."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        with get_session() as session:
            cam = Camera(
                name="front",
                main_url="rtsp://a:b@1.2.3.4/main",
                sub_url="rtsp://a:b@1.2.3.4/sub",
            )
            session.add(cam)
            session.commit()
            session.refresh(cam)
            camera_id = cam.id

            evt = Event(
                camera_id=camera_id,
                event_type="person",
                severity="warn",
                message="person detected",
                metadata_json="{}",
                created_at=datetime(2025, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
            )
            session.add(evt)
            session.commit()

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.events) == 1
        assert result.events[0].object_type == "person"
        assert result.events[0].color == "#e74c3c"

    def test_build_timeline_populates_object_type_for_vehicle(self, tmp_path: Path) -> None:
        """Vehicle events get object_type='vehicle' and correct color."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        with get_session() as session:
            cam = Camera(
                name="front",
                main_url="rtsp://a:b@1.2.3.4/main",
                sub_url="rtsp://a:b@1.2.3.4/sub",
            )
            session.add(cam)
            session.commit()
            session.refresh(cam)
            camera_id = cam.id

            evt = Event(
                camera_id=camera_id,
                event_type="car",
                severity="info",
                message="car detected",
                metadata_json="{}",
                created_at=datetime(2025, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
            )
            session.add(evt)
            session.commit()

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.events) == 1
        assert result.events[0].object_type == "vehicle"
        assert result.events[0].color == "#e67e22"

    def test_build_timeline_motion_gets_other(self, tmp_path: Path) -> None:
        """Motion events get object_type='other' and gray color."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        with get_session() as session:
            cam = Camera(
                name="front",
                main_url="rtsp://a:b@1.2.3.4/main",
                sub_url="rtsp://a:b@1.2.3.4/sub",
            )
            session.add(cam)
            session.commit()
            session.refresh(cam)
            camera_id = cam.id

            evt = Event(
                camera_id=camera_id,
                event_type="motion",
                severity="info",
                message="motion detected",
                metadata_json="{}",
                created_at=datetime(2025, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
            )
            session.add(evt)
            session.commit()

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.events) == 1
        assert result.events[0].object_type == "other"
        assert result.events[0].color == "#7f8c8d"

    def test_build_timeline_severity_still_works(self, tmp_path: Path) -> None:
        """Severity field is still populated correctly alongside new fields."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        with get_session() as session:
            cam = Camera(
                name="front",
                main_url="rtsp://a:b@1.2.3.4/main",
                sub_url="rtsp://a:b@1.2.3.4/sub",
            )
            session.add(cam)
            session.commit()
            session.refresh(cam)
            camera_id = cam.id

            _create_event(
                session,
                camera_id=camera_id,
                event_type="person",
                severity="warn",
                created_at=datetime(2025, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
            )

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.events) == 1
        # Severity is still there
        assert result.events[0].severity == "warn"
        # New fields also populated
        assert result.events[0].object_type == "person"
        assert result.events[0].color == "#e74c3c"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_event(
    session,
    camera_id: int,
    event_type: str = "motion",
    severity: str = "info",
    created_at: datetime | None = None,
) -> Event:
    """Insert an Event row and return it."""
    evt = Event(
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        message="test event",
        metadata_json="{}",
        created_at=created_at or datetime.now(timezone.utc),
    )
    session.add(evt)
    session.commit()
    session.refresh(evt)
    return evt


# ---------------------------------------------------------------------------
# Integration tests: API endpoint includes object_type and color
# ---------------------------------------------------------------------------


class TestTimelineAPIColors:
    """Tests for the /api/recordings/{id}/timeline endpoint returning color fields."""

    @pytest.fixture
    def sample_config_with_rec(self, tmp_path: Path) -> tuple[Path, int]:
        """Create config, DB, camera, and a recording; return (config_path, recording_id)."""
        rec_dir = tmp_path / "recordings" / "front" / "main"
        rec_dir.mkdir(parents=True)

        cfg_data = {
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
                    "proxy": {"enabled": False, "mode": "mjpeg"},
                },
            ],
            "runtime": {
                "ffmpeg_path": "ffmpeg",
                "workspace_dir": str(tmp_path / "workspace"),
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data))

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        with get_session() as session:
            cam = Camera(
                name="front",
                main_url="rtsp://user:pass@192.168.1.50:554/main",
                sub_url="rtsp://user:pass@192.168.1.50:554/sub",
            )
            session.add(cam)
            session.commit()
            session.refresh(cam)
            camera_id = cam.id

            rec = Recording(
                camera_id=camera_id,
                stream="main",
                path="/recordings/front/main/test.ts",
                size_bytes=2048,
                start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
                container="ts",
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            recording_id = rec.id

        return cfg_path, recording_id

    @pytest.fixture
    def admin_client_with_rec(self, sample_config_with_rec: tuple) -> TestClient:
        """Authenticated TestClient with a recording in the DB."""
        cfg_path, _ = sample_config_with_rec

        pw_hash = hash_password("testpass123")
        from rtsp_warden.db.engine import get_session as gs

        with gs() as session:
            from rtsp_warden.db.models import User

            user = User(username="admin", password_hash=pw_hash, role="admin", is_active=True)
            session.add(user)
            session.commit()

        cfg = load_config(cfg_path)
        settings = WebSettings(host="127.0.0.1", port=8080)
        app = create_app(settings, cfg=cfg, runtime_provider=lambda: None)
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

    def test_api_returns_object_type_and_color(
        self, sample_config_with_rec: tuple, admin_client_with_rec: TestClient
    ) -> None:
        """GET /api/recordings/{id}/timeline includes object_type and color."""
        _, rec_id = sample_config_with_rec
        r = admin_client_with_rec.get(f"/api/recordings/{rec_id}/timeline")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["events"], list)
        # If there are events, check they have the new fields
        for evt in data["events"]:
            assert "object_type" in evt, f"Event missing object_type: {evt}"
            assert "color" in evt, f"Event missing color: {evt}"
            assert evt["object_type"] in ("person", "pet", "critter", "vehicle", "other")
            assert evt["color"].startswith("#")

    def test_api_backward_compat_existing_fields(
        self, sample_config_with_rec: tuple, admin_client_with_rec: TestClient
    ) -> None:
        """Existing fields (id, type, severity, ts) are still present."""
        _, rec_id = sample_config_with_rec
        r = admin_client_with_rec.get(f"/api/recordings/{rec_id}/timeline")
        assert r.status_code == 200
        data = r.json()
        for evt in data["events"]:
            assert "id" in evt
            assert "type" in evt
            assert "severity" in evt
            assert "ts" in evt
