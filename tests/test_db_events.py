"""Event and action-run helpers in db/schema.py (the 0003 schema).

Every DB datetime is written as naive UTC and read back through as_utc().
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rtsp_warden.db.schema import (
    action_stats,
    as_utc,
    close_event,
    count_events,
    get_event,
    get_latest_event_for_camera,
    insert_action_run,
    insert_event,
    list_action_runs,
    list_events,
    update_event,
)

T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


def _event(camera: str = "front", label: str = "person", at: datetime = T0, **kw: object) -> int:
    fields: dict = {
        "camera_name": camera,
        "event_type": label,
        "label": label,
        "confidence": 0.9,
        "zone": "",
        "track_id": None,
        "message": f"{label} on {camera}",
        "created_at": at,
    }
    fields.update(kw)
    return insert_event(**fields)


class TestAsUtc:
    def test_none(self) -> None:
        assert as_utc(None) is None

    def test_naive_is_taken_as_utc(self) -> None:
        assert as_utc(datetime(2026, 10, 2, 12, 0)) == T0
        assert as_utc(datetime(2026, 10, 2, 12, 0)).tzinfo is timezone.utc

    def test_aware_is_converted(self) -> None:
        plus2 = timezone(timedelta(hours=2))
        result = as_utc(datetime(2026, 10, 2, 14, 0, tzinfo=plus2))
        assert result == T0
        assert result.tzinfo is timezone.utc


class TestEvents:
    def test_insert_and_get_round_trip(self, clean_db: None) -> None:
        plus2 = timezone(timedelta(hours=2))
        event_id = _event(
            at=datetime(2026, 10, 2, 14, 0, 0, 250000, tzinfo=plus2),
            zone="porch",
            track_id=4,
            metadata={"bbox": [1, 2, 3, 4]},
            thumbnail_path="front/thumbnails/1.jpg",
            severity="warn",
        )
        event = get_event(event_id)
        assert event is not None
        assert event.camera_name == "front"
        assert event.label == "person"
        assert event.event_type == "person"
        assert event.confidence == pytest.approx(0.9)
        assert event.zone == "porch"
        assert event.track_id == 4
        assert event.severity == "warn"
        assert event.thumbnail_path == "front/thumbnails/1.jpg"
        assert event.clip_path is None
        assert event.ended_at is None
        assert json.loads(event.metadata_json) == {"bbox": [1, 2, 3, 4]}
        # stored as naive UTC with microseconds, read back through as_utc()
        assert event.created_at == datetime(2026, 10, 2, 12, 0, 0, 250000)
        assert as_utc(event.created_at) == T0 + timedelta(microseconds=250000)

    def test_insert_stores_numpy_confidence_as_float(self, clean_db: None) -> None:
        event = get_event(_event(confidence=np.float32(0.5)))
        assert event is not None
        assert event.confidence == 0.5
        assert type(event.confidence) is float

    def test_get_missing_event(self, clean_db: None) -> None:
        assert get_event(999) is None

    def test_update_event_fields(self, clean_db: None) -> None:
        event_id = _event()
        update_event(
            event_id,
            confidence=0.97,
            zone="driveway",
            thumbnail_path="front/thumbnails/9.jpg",
            metadata={"suppressed_by": "quiet-hours"},
        )
        event = get_event(event_id)
        assert event is not None
        assert event.confidence == pytest.approx(0.97)
        assert event.zone == "driveway"
        assert event.thumbnail_path == "front/thumbnails/9.jpg"
        assert json.loads(event.metadata_json) == {"suppressed_by": "quiet-hours"}

    def test_update_event_rejects_unknown_fields(self, clean_db: None) -> None:
        event_id = _event()
        with pytest.raises(ValueError, match="camera_id"):
            update_event(event_id, camera_id=3)

    def test_update_missing_event_is_ignored(self, clean_db: None) -> None:
        update_event(999, zone="x")

    def test_close_event_sets_ended_at_as_naive_utc(self, clean_db: None) -> None:
        event_id = _event()
        plus2 = timezone(timedelta(hours=2))
        close_event(event_id, datetime(2026, 10, 2, 14, 0, 30, tzinfo=plus2), clip_path="c.mp4")
        event = get_event(event_id)
        assert event is not None
        assert event.ended_at == datetime(2026, 10, 2, 12, 0, 30)
        assert as_utc(event.ended_at) == T0 + timedelta(seconds=30)
        assert event.clip_path == "c.mp4"

    def test_list_and_count_filters(self, clean_db: None) -> None:
        first = _event("front", "person", T0)
        second = _event("front", "car", T0 + timedelta(minutes=1))
        third = _event("back", "person", T0 + timedelta(minutes=2))

        assert [e.id for e in list_events()] == [third, second, first]
        assert [e.id for e in list_events(camera_name="front")] == [second, first]
        assert [e.id for e in list_events(label="person")] == [third, first]
        assert [e.id for e in list_events(since=T0 + timedelta(seconds=30))] == [third, second]
        assert [e.id for e in list_events(until=T0 + timedelta(seconds=90))] == [second, first]
        assert [e.id for e in list_events(limit=1, offset=1)] == [second]
        assert count_events() == 3
        assert count_events(camera_name="front") == 2
        assert count_events(label="person", since=T0 + timedelta(seconds=30)) == 1

    def test_list_breaks_created_at_ties_by_id(self, clean_db: None) -> None:
        first = _event(at=T0)
        second = _event(at=T0)
        assert [e.id for e in list_events()] == [second, first]

    def test_since_accepts_aware_non_utc(self, clean_db: None) -> None:
        _event(at=T0)
        plus2 = timezone(timedelta(hours=2))
        assert count_events(since=datetime(2026, 10, 2, 13, 59, tzinfo=plus2)) == 1
        assert count_events(since=datetime(2026, 10, 2, 14, 1, tzinfo=plus2)) == 0

    def test_latest_event_for_camera_uses_camera_name(self, clean_db: None) -> None:
        now = datetime.now(timezone.utc)
        _event("garage", at=now - timedelta(seconds=100), message="old")
        assert get_latest_event_for_camera("garage", since_seconds=60) is None
        _event("garage", at=now, message="new")
        latest = get_latest_event_for_camera("garage", since_seconds=60)
        assert latest is not None
        assert latest.message == "new"
        assert get_latest_event_for_camera("garage").message == "new"
        assert get_latest_event_for_camera("nope") is None


class TestActionRuns:
    def test_insert_and_list_in_order(self, clean_db: None) -> None:
        event_id = _event()
        first = insert_action_run(event_id=event_id, action_name="phone", status="ok", error=None)
        second = insert_action_run(
            event_id=event_id, action_name="hook", status="failed", error="HTTP 500"
        )
        runs = list_action_runs(event_id)
        assert [r.id for r in runs] == [first, second]
        assert runs[1].error == "HTTP 500"
        assert runs[0].created_at is not None
        assert list_action_runs(event_id + 1) == []

    def test_action_stats(self, clean_db: None) -> None:
        assert action_stats() == {}
        event_id = _event()
        insert_action_run(event_id=event_id, action_name="phone", status="failed", error="x")
        insert_action_run(event_id=event_id, action_name="phone", status="ok", error=None)
        insert_action_run(event_id=event_id, action_name="hook", status="failed", error="y")

        stats = action_stats()

        assert set(stats) == {"phone", "hook"}
        assert stats["phone"]["last_status"] == "ok"
        assert stats["phone"]["failures"] == 1
        assert stats["hook"]["last_status"] == "failed"
        assert stats["hook"]["failures"] == 1
        last_run = stats["phone"]["last_run"]
        assert last_run.tzinfo is timezone.utc
        assert abs((datetime.now(timezone.utc) - last_run).total_seconds()) < 60
