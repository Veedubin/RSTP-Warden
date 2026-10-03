"""Tests for detectors/sinks.py: EventSink writes one events row per detection."""

from __future__ import annotations

import json

import pytest

from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import ensure_schema, list_events
from rtsp_warden.detectors import sinks as sinks_mod
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.sinks import EventSink


@pytest.fixture
def event_db(tmp_path, monkeypatch):
    """An isolated SQLite DB at the current schema (no cameras table any more)."""
    db_url = f"sqlite:///{tmp_path}/test_events.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()
    yield db_url
    reset_engine()


def test_event_sink_creates_event_per_detection(event_db: str) -> None:
    sink = EventSink()
    detections = [Detection(kind="motion", confidence=0.8, bbox=(1, 2, 3, 4), ts_unix=1.0)]
    sink("testcam", "sub", detections)

    events = list_events(camera_name="testcam")
    assert len(events) == 1
    event = events[0]
    assert event.camera_name == "testcam"
    assert event.event_type == "motion"
    assert event.label == "motion"
    assert event.confidence == pytest.approx(0.8)
    assert event.zone == ""
    assert event.message == "motion detected on testcam/sub (confidence=0.80)"
    meta = json.loads(event.metadata_json)
    assert meta == {"bbox": [1, 2, 3, 4], "stream": "sub", "ts_unix": 1.0}


def test_event_sink_multiple_detections(event_db: str) -> None:
    sink = EventSink()
    detections = [
        Detection(kind="motion", confidence=0.8, ts_unix=1700000000.0),
        Detection(kind="person", confidence=0.9, ts_unix=1700000001.0),
    ]
    sink("testcam", "sub", detections)

    assert len(list_events(camera_name="testcam")) == 2


def test_event_sink_stores_any_camera_name(event_db: str) -> None:
    """No cameras table lookup any more: the name is stored as given."""
    sink = EventSink()
    sink("unknown_cam", "sub", [Detection(kind="motion", confidence=0.5, ts_unix=1.0)])

    events = list_events()
    assert len(events) == 1
    assert events[0].camera_name == "unknown_cam"


def test_event_sink_severity_mapping(event_db: str) -> None:
    sink = EventSink()
    detections = [
        Detection(kind="motion", confidence=0.5, ts_unix=1700000000.0),
        Detection(kind="person", confidence=0.9, ts_unix=1700000001.0),
        Detection(kind="vehicle", confidence=0.7, ts_unix=1700000002.0),
    ]
    sink("testcam", "sub", detections)

    severities = {e.event_type: e.severity for e in list_events(camera_name="testcam")}
    assert severities == {"motion": "info", "person": "warn", "vehicle": "info"}


def test_event_sink_logs_and_continues_when_the_insert_fails(
    event_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_insert(**kwargs: object) -> int:
        raise RuntimeError("database is locked")

    warnings: list[str] = []
    monkeypatch.setattr(sinks_mod._schema, "insert_event", broken_insert)
    monkeypatch.setattr(
        sinks_mod.logger, "warning", lambda msg, *args, **kw: warnings.append(msg % args)
    )

    EventSink()("testcam", "sub", [Detection(kind="motion", confidence=0.5, ts_unix=1.0)])

    assert warnings == ["failed to insert event for testcam/sub kind=motion"]
