"""Tests for detectors/sinks.py: EventSink and db/schema event functions."""

from __future__ import annotations

import pytest

from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.models import Camera
from rtsp_warden.db.schema import (
    count_events,
    create_event,
    ensure_schema,
    get_event_by_id,
    list_events,
)
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.sinks import EventSink

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_db(tmp_path, monkeypatch):
    """Set up an isolated SQLite DB with schema and a test camera."""
    from rtsp_warden.db.engine import get_session

    db_url = f"sqlite:///{tmp_path}/test_events.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()

    # Create a test camera
    with get_session() as session:
        cam = Camera(name="testcam", main_url="rtsp://x/main", sub_url="rtsp://x/sub", enabled=True)
        session.add(cam)
        session.commit()
        session.expunge(cam)

    yield db_url
    reset_engine()


# ---------------------------------------------------------------------------
# EventSink
# ---------------------------------------------------------------------------


def test_event_sink_creates_event_per_detection(event_db: str) -> None:
    """EventSink.__call__ inserts one Event row per detection."""
    sink = EventSink()
    detections = [Detection(kind="motion", confidence=0.8, ts_unix=1700000000.0)]
    sink("testcam", "sub", detections)

    events = list_events(camera_name="testcam")
    assert len(events) == 1
    assert events[0].event_type == "motion"


def test_event_sink_multiple_detections(event_db: str) -> None:
    """EventSink creates multiple Event rows for multiple detections."""
    sink = EventSink()
    detections = [
        Detection(kind="motion", confidence=0.8, ts_unix=1700000000.0),
        Detection(kind="person", confidence=0.9, ts_unix=1700000001.0),
    ]
    sink("testcam", "sub", detections)

    events = list_events(camera_name="testcam")
    assert len(events) == 2


def test_event_sink_unknown_camera(event_db: str) -> None:
    """EventSink with unknown camera_name sets camera_id to None."""
    sink = EventSink()
    detections = [Detection(kind="motion", confidence=0.5, ts_unix=1700000000.0)]
    sink("unknown_cam", "sub", detections)

    events = list_events()
    assert len(events) == 1
    assert events[0].camera_id is None


def test_event_sink_severity_mapping(event_db: str) -> None:
    """EventSink maps detection kind to correct severity."""
    sink = EventSink()
    detections = [
        Detection(kind="motion", confidence=0.5, ts_unix=1700000000.0),
        Detection(kind="person", confidence=0.9, ts_unix=1700000001.0),
        Detection(kind="vehicle", confidence=0.7, ts_unix=1700000002.0),
    ]
    sink("testcam", "sub", detections)

    events = list_events(camera_name="testcam")
    # Order: newest first (desc)
    severities = {e.event_type: e.severity for e in events}
    assert severities["motion"] == "info"
    assert severities["person"] == "warn"
    assert severities["vehicle"] == "info"


# ---------------------------------------------------------------------------
# DB schema event functions
# ---------------------------------------------------------------------------


def test_create_event_with_camera(event_db: str) -> None:
    """create_event with camera_name sets camera_id."""
    event = create_event(
        camera_name="testcam", event_type="motion", severity="info", message="test"
    )
    assert event.camera_id is not None
    assert event.event_type == "motion"
    assert event.severity == "info"
    assert event.message == "test"


def test_list_events_and_count_events(event_db: str) -> None:
    """list_events and count_events return correct results after creating events."""
    create_event(camera_name="testcam", event_type="motion", severity="info")
    create_event(camera_name="testcam", event_type="person", severity="warn")
    create_event(event_type="motion", severity="info")  # no camera

    all_events = list_events()
    assert len(all_events) == 3

    cam_events = list_events(camera_name="testcam")
    assert len(cam_events) == 2

    assert count_events() == 3
    assert count_events(camera_name="testcam") == 2
    assert count_events(event_type="motion") == 2


def test_get_event_by_id(event_db: str) -> None:
    """get_event_by_id returns the correct event or None."""
    event = create_event(camera_name="testcam", event_type="motion", message="found me")
    fetched = get_event_by_id(event.id)
    assert fetched is not None
    assert fetched.message == "found me"

    assert get_event_by_id(999999) is None
