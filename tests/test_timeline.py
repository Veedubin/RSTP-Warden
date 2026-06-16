"""Tests for Sprint 4 Batch 6: Timeline canvas (recording scrubber UI).

Validates:
- TimelineData dataclass construction
- build_timeline service: segment scanning and event querying
- API endpoint: auth, 404, JSON structure
"""

from __future__ import annotations

import os
import time
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
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.timeline import (
    TimelineData,
    TimelineEvent,
    TimelineSegment,
    build_timeline,
)

# Ensure auth is enabled for tests
os.environ["WARDEN_AUTH_ENABLED"] = "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_recording(
    session, camera_id: int, start_time: datetime, end_time: datetime | None = None
) -> Recording:
    """Insert a Recording row and return it."""
    rec = Recording(
        camera_id=camera_id,
        stream="main",
        path=f"/recordings/front/main/{start_time.strftime('%Y%m%d_%H%M%S')}.ts",
        size_bytes=1024,
        start_time=start_time,
        end_time=end_time,
        container="ts",
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


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
# Unit tests: dataclass construction
# ---------------------------------------------------------------------------


class TestTimelineDataclass:
    """Tests for timeline dataclass construction."""

    def test_timeline_segment_construction(self) -> None:
        """TimelineSegment can be created with all fields."""
        seg = TimelineSegment(start_ts=1000.0, end_ts=1005.0, path="seg-001.ts", size_bytes=2048)
        assert seg.start_ts == 1000.0
        assert seg.end_ts == 1005.0
        assert seg.path == "seg-001.ts"
        assert seg.size_bytes == 2048

    def test_timeline_event_construction(self) -> None:
        """TimelineEvent can be created with all fields."""
        evt = TimelineEvent(id=1, event_type="motion", severity="info", ts_unix=1002.0)
        assert evt.id == 1
        assert evt.event_type == "motion"
        assert evt.severity == "info"
        assert evt.ts_unix == 1002.0

    def test_timeline_data_construction(self) -> None:
        """TimelineData can be created with all fields."""
        td = TimelineData(
            recording_id=42,
            camera_name="front",
            stream="main",
            start_ts=1000.0,
            end_ts=2000.0,
            segments=[TimelineSegment(start_ts=1000.0, end_ts=1005.0, path="s.ts", size_bytes=100)],
            events=[TimelineEvent(id=1, event_type="motion", severity="info", ts_unix=1002.0)],
        )
        assert td.recording_id == 42
        assert td.camera_name == "front"
        assert td.stream == "main"
        assert len(td.segments) == 1
        assert len(td.events) == 1

    def test_timeline_data_defaults_empty(self) -> None:
        """TimelineData defaults to empty segments and events."""
        td = TimelineData(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_ts=0.0,
            end_ts=100.0,
        )
        assert td.segments == []
        assert td.events == []


# ---------------------------------------------------------------------------
# Unit tests: build_timeline
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    """Tests for the build_timeline service function."""

    def test_build_timeline_with_no_segments(self, tmp_path: Path) -> None:
        """Empty recordings dir produces empty segments list."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # Need a DB for event queries; set up a temp DB.
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert isinstance(result, TimelineData)
        assert result.segments == []
        assert result.events == []

    def test_build_timeline_finds_matching_segments(self, tmp_path: Path) -> None:
        """build_timeline returns .ts segments whose mtime falls within the time range."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        # Write 3 .ts files with specific mtimes
        now = time.time()
        for i in range(3):
            p = rec_dir / f"seg-0000{i + 1}.ts"
            p.write_bytes(b"x" * 100)
            # Set mtime to now + i*5 seconds
            os.utime(p, (now + i * 5, now + i * 5))

        start = datetime.fromtimestamp(now, tz=timezone.utc)
        end = datetime.fromtimestamp(now + 600, tz=timezone.utc)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.segments) == 3
        # Segments should be sorted by start_ts
        for seg in result.segments:
            assert seg.path.endswith(".ts")
            assert seg.size_bytes == 100
            assert seg.start_ts >= start.timestamp() - 1  # allow float imprecision
            assert seg.end_ts > seg.start_ts

    def test_build_timeline_excludes_segments_outside_range(self, tmp_path: Path) -> None:
        """Segments outside the recording's time range are filtered."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        now = time.time()
        # Write a segment far in the future
        p = rec_dir / "seg-future.ts"
        p.write_bytes(b"x" * 50)
        future_time = now + 86400 * 30  # 30 days in future
        os.utime(p, (future_time, future_time))

        # The recording window is "now" to "now + 1 hour"
        start = datetime.fromtimestamp(now, tz=timezone.utc)
        end = datetime.fromtimestamp(now + 3600, tz=timezone.utc)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        assert len(result.segments) == 0

    def test_build_timeline_includes_events_in_range(self, tmp_path: Path) -> None:
        """Events from the camera within the time range are included."""
        rec_dir = tmp_path / "front" / "main"
        rec_dir.mkdir(parents=True)

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        os.environ["WARDEN_DB_URL"] = db_url
        reset_engine()
        ensure_schema()

        from rtsp_warden.db.engine import get_session

        # Create a camera row
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

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # Create events within and outside the range
        with get_session() as session:
            _create_event(
                session,
                camera_id=camera_id,
                severity="info",
                created_at=datetime(2025, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
            )
            _create_event(
                session,
                camera_id=camera_id,
                severity="warn",
                created_at=datetime(2025, 1, 1, 12, 45, 0, tzinfo=timezone.utc),
            )
            _create_event(
                session,
                camera_id=camera_id,
                severity="error",
                created_at=datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            )

        result = build_timeline(
            recording_id=1,
            camera_name="front",
            stream="main",
            start_time=start,
            end_time=end,
            recordings_dir=rec_dir,
        )
        # Only the first two events fall within the time range
        assert len(result.events) == 2
        assert result.events[0].severity == "info"
        assert result.events[1].severity == "warn"


# ---------------------------------------------------------------------------
# Integration tests: API endpoint
# ---------------------------------------------------------------------------


class TestTimelineAPI:
    """Tests for the /api/recordings/{id}/timeline endpoint."""

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

        # Set up DB
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
        cfg_path, rec_id = sample_config_with_rec

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

    def test_api_endpoint_returns_404_for_missing_recording(
        self, admin_client_with_rec: TestClient
    ) -> None:
        """GET /api/recordings/9999/timeline returns 404."""
        r = admin_client_with_rec.get("/api/recordings/9999/timeline")
        assert r.status_code == 404

    def test_api_endpoint_returns_timeline_json(
        self, sample_config_with_rec: tuple, admin_client_with_rec: TestClient
    ) -> None:
        """GET /api/recordings/{id}/timeline returns 200
        with JSON containing segments and events."""
        _, rec_id = sample_config_with_rec
        r = admin_client_with_rec.get(f"/api/recordings/{rec_id}/timeline")
        assert r.status_code == 200
        data = r.json()
        assert data["recording_id"] == rec_id
        assert data["camera_name"] == "front"
        assert data["stream"] == "main"
        assert "start_ts" in data
        assert "end_ts" in data
        assert isinstance(data["segments"], list)
        assert isinstance(data["events"], list)

    def test_api_endpoint_requires_auth(self) -> None:
        """GET /api/recordings/{id}/timeline without auth returns 401 or redirect."""
        settings = WebSettings()
        app = create_app(settings)
        client = TestClient(app)
        r = client.get("/api/recordings/1/timeline", follow_redirects=False)
        assert r.status_code in (401, 303, 307, 302)
