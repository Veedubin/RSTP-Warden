"""Tests for detectors/event_builder.py (EventBuilder) and the EventSink rewrite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from rtsp_warden.db import schema
from rtsp_warden.detectors import event_builder, sinks
from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.event_builder import EventBuilder, EventInfo
from rtsp_warden.detectors.grid_mask import GridMask
from rtsp_warden.detectors.sinks import EventSink
from rtsp_warden.detectors.tracking import Track, TrackerUpdate

FRAME_W, FRAME_H = 320, 180
SHAPE = (FRAME_H, FRAME_W)


class FakeDb:
    """Records insert/update/close calls the way db/schema.py's helpers receive them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, Any]]] = []
        self._next_id = 100

    def insert_event(self, **fields: Any) -> int:
        self._next_id += 1
        self.calls.append(("insert", self._next_id, fields))
        return self._next_id

    def update_event(self, event_id: int, **fields: Any) -> None:
        self.calls.append(("update", event_id, fields))

    def close_event(self, event_id: int, ended_at: datetime, **fields: Any) -> None:
        self.calls.append(("close", event_id, {"ended_at": ended_at, **fields}))

    def of(self, kind: str) -> list[tuple[str, int, dict[str, Any]]]:
        return [call for call in self.calls if call[0] == kind]


class FailingInsertDb(FakeDb):
    def insert_event(self, **fields: Any) -> int:
        raise RuntimeError("database is locked")


def _utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _track(
    track_id: int = 1,
    *,
    label: str = "person",
    first: float = 10.0,
    last: float = 10.2,
    conf: float = 0.8,
    bbox: tuple[int, int, int, int] = (100, 50, 40, 60),
) -> Track:
    return Track(
        id=track_id,
        label=label,
        first_seen=first,
        last_seen=last,
        frames=2,
        bbox=bbox,
        best_confidence=conf,
        best_bbox=bbox,
        best_frame=_frame(),
        best_ts=last,
    )


def _update(
    *,
    opened: tuple[Track, ...] = (),
    improved: tuple[Track, ...] = (),
    closed: tuple[Track, ...] = (),
) -> TrackerUpdate:
    return TrackerUpdate(
        opened=list(opened),
        improved=list(improved),
        closed=list(closed),
        active=list(opened) + list(improved),
    )


def _builder(tmp_path: Path, db: Any, **kwargs: Any) -> EventBuilder:
    return EventBuilder(camera="yard", output_dir=tmp_path, area_masks=[], db=db, **kwargs)


def _confidence_writes(db: FakeDb) -> list[tuple[str, int, dict[str, Any]]]:
    return [call for call in db.of("update") if "confidence" in call[2]]


# ---------------------------------------------------------------------------
# Object events
# ---------------------------------------------------------------------------


def test_open_inserts_row_writes_thumbnail_and_calls_on_open(tmp_path: Path) -> None:
    db = FakeDb()
    opened: list[EventInfo] = []
    builder = _builder(tmp_path, db, on_open=opened.append)
    track = _track()

    builder.on_tracks(_update(opened=(track,)), SHAPE)

    inserts = db.of("insert")
    assert len(inserts) == 1
    _, event_id, fields = inserts[0]
    assert fields["camera_name"] == "yard"
    assert fields["event_type"] == "object"
    assert fields["label"] == "person"
    assert fields["confidence"] == pytest.approx(0.8)
    assert fields["zone"] == ""
    assert fields["track_id"] == 1
    assert fields["created_at"] == _utc(10.0)
    assert fields["metadata"] == {"bbox": [100, 50, 40, 60], "frame_size": [320, 180]}
    rel = f"yard/thumbnails/{event_id}.jpg"
    assert db.of("update") == [("update", event_id, {"thumbnail_path": rel})]
    assert (tmp_path / rel).is_file()
    assert track.event_id == event_id
    assert len(opened) == 1
    info = opened[0]
    assert (info.id, info.camera, info.label, info.event_type) == (
        event_id,
        "yard",
        "person",
        "object",
    )
    assert info.started_at == _utc(10.0)
    assert info.ended_at is None
    assert info.thumbnail_path == rel
    assert info.clip_path is None
    assert info.track_id == 1
    assert builder.open_event_ids == [event_id]


def test_zone_is_looked_up_from_the_best_box_centre_and_frame_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_zone_for_point(area_masks: Any, x: float, y: float, frame_w: int, frame_h: int) -> str:
        calls.append((list(area_masks), x, y, frame_w, frame_h))
        return "driveway"

    monkeypatch.setattr(event_builder, "zone_for_point", fake_zone_for_point)
    db = FakeDb()
    mask = GridMask(grid_cols=2, grid_rows=2)
    builder = EventBuilder(
        camera="yard", output_dir=tmp_path, area_masks=[("driveway", mask)], db=db
    )
    track = _track(bbox=(100, 50, 40, 60))

    builder.on_tracks(_update(opened=(track,)), SHAPE)

    assert calls == [([("driveway", mask)], 120.0, 80.0, 320, 180)]
    assert db.of("insert")[0][2]["zone"] == "driveway"
    assert track.zone == "driveway"


def test_no_area_zones_means_empty_zone_without_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_be_called(*args: Any) -> str:
        raise AssertionError("zone_for_point called without area zones")

    monkeypatch.setattr(event_builder, "zone_for_point", must_not_be_called)
    db = FakeDb()
    _builder(tmp_path, db).on_tracks(_update(opened=(_track(),)), SHAPE)
    assert db.of("insert")[0][2]["zone"] == ""


def test_improved_updates_are_throttled_to_one_per_second_of_frame_time(tmp_path: Path) -> None:
    db = FakeDb()
    builder = _builder(tmp_path, db)
    track = _track(last=10.25, conf=0.6)
    builder.on_tracks(_update(opened=(track,)), SHAPE)  # written at 10.25

    def improve(last: float, conf: float) -> None:
        track.last_seen = last
        track.best_confidence = conf
        track.best_ts = last
        builder.on_tracks(_update(improved=(track,)), SHAPE)

    improve(10.5, 0.7)  # 0.25 s after the last write: dirty, not written
    improve(11.0, 0.75)  # 0.75 s: still not written
    assert _confidence_writes(db) == []

    track.last_seen = 11.25  # matched again, no improvement: 1.0 s since the write
    builder.on_tracks(_update(), SHAPE)
    writes = _confidence_writes(db)
    assert len(writes) == 1
    assert writes[0][2]["confidence"] == pytest.approx(0.75)
    assert writes[0][2]["zone"] == ""

    improve(11.5, 0.9)  # 0.25 s after that write: dirty again, not written yet
    assert len(_confidence_writes(db)) == 1


def test_close_flushes_the_dirty_best_and_sets_ended_at_to_last_seen(tmp_path: Path) -> None:
    db = FakeDb()
    closed: list[EventInfo] = []
    builder = _builder(tmp_path, db, on_close=closed.append)
    track = _track(last=10.2, conf=0.6)
    builder.on_tracks(_update(opened=(track,)), SHAPE)
    track.last_seen = 10.4
    track.best_confidence = 0.95
    track.best_ts = 10.4
    builder.on_tracks(_update(improved=(track,)), SHAPE)

    builder.on_tracks(_update(closed=(track,)), SHAPE)

    closes = db.of("close")
    assert len(closes) == 1
    _, event_id, fields = closes[0]
    assert fields["ended_at"] == _utc(10.4)
    assert fields["confidence"] == pytest.approx(0.95)
    assert fields["zone"] == ""
    assert [info.id for info in closed] == [event_id]
    assert closed[0].ended_at == _utc(10.4)
    assert closed[0].confidence == pytest.approx(0.95)
    assert builder.open_event_ids == []


def test_close_without_improvement_only_sets_ended_at(tmp_path: Path) -> None:
    db = FakeDb()
    builder = _builder(tmp_path, db)
    track = _track(last=10.2)
    builder.on_tracks(_update(opened=(track,)), SHAPE)
    builder.on_tracks(_update(closed=(track,)), SHAPE)
    event_id = db.of("insert")[0][1]
    assert db.of("close") == [("close", event_id, {"ended_at": _utc(10.2)})]


def test_closed_track_that_never_opened_writes_nothing(tmp_path: Path) -> None:
    db = FakeDb()
    _builder(tmp_path, db).on_tracks(_update(closed=(_track(),)), SHAPE)
    assert db.calls == []


def test_insert_failure_is_logged_and_skips_on_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        event_builder.log, "warning", lambda msg, *args, **kwargs: warnings.append(msg)
    )
    opened: list[EventInfo] = []
    builder = _builder(tmp_path, FailingInsertDb(), on_open=opened.append)

    builder.on_tracks(_update(opened=(_track(),)), SHAPE)

    assert opened == []
    assert warnings == ["failed to insert event for %s label=%s"]
    assert builder.open_event_ids == []


def test_thumbnail_failure_keeps_the_event_without_a_thumbnail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        event_builder.log, "warning", lambda msg, *args, **kwargs: warnings.append(msg)
    )
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    db = FakeDb()
    opened: list[EventInfo] = []
    builder = EventBuilder(
        camera="yard", output_dir=blocker, area_masks=[], db=db, on_open=opened.append
    )

    builder.on_tracks(_update(opened=(_track(),)), SHAPE)

    assert len(db.of("insert")) == 1
    assert db.of("update") == []
    assert opened[0].thumbnail_path is None
    assert warnings == ["failed to write thumbnail %s"]


def test_callback_exceptions_are_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        event_builder.log, "warning", lambda msg, *args, **kwargs: warnings.append(msg)
    )

    def boom(info: EventInfo) -> None:
        raise RuntimeError("rule engine down")

    db = FakeDb()
    _builder(tmp_path, db, on_open=boom).on_tracks(_update(opened=(_track(),)), SHAPE)

    assert len(db.of("insert")) == 1
    assert warnings == ["event callback failed for %s event %s"]


def test_close_all_closes_everything_once_and_ignores_later_calls(tmp_path: Path) -> None:
    db = FakeDb()
    closed: list[EventInfo] = []
    builder = _builder(tmp_path, db, on_close=closed.append)
    track = _track(last=10.2)
    builder.on_tracks(_update(opened=(track,)), SHAPE)
    builder.on_motion(True, False, 9.0)

    builder.close_all(12.0)

    assert [call[2]["ended_at"] for call in db.of("close")] == [_utc(10.2), _utc(12.0)]
    assert [info.event_type for info in closed] == ["object", "motion"]
    builder.close_all(13.0)
    builder.on_tracks(_update(opened=(_track(2),)), SHAPE)
    builder.on_motion(True, False, 14.0)
    assert len(db.of("close")) == 2
    assert len(db.of("insert")) == 2


# ---------------------------------------------------------------------------
# Motion events
# ---------------------------------------------------------------------------


def test_motion_burst_is_one_row_without_thumbnail(tmp_path: Path) -> None:
    db = FakeDb()
    opened: list[EventInfo] = []
    closed: list[EventInfo] = []
    builder = _builder(tmp_path, db, on_open=opened.append, on_close=closed.append)

    builder.on_motion(True, False, 100.0)
    builder.on_motion(True, False, 100.5)  # already open: ignored
    builder.on_motion(False, True, 104.0)

    inserts = db.of("insert")
    assert len(inserts) == 1
    _, event_id, fields = inserts[0]
    assert fields["event_type"] == "motion"
    assert fields["label"] == "motion"
    assert fields["confidence"] == 1.0
    assert fields["zone"] == ""
    assert fields["track_id"] is None
    assert fields["created_at"] == _utc(100.0)
    assert db.of("close") == [("close", event_id, {"ended_at": _utc(104.0)})]
    assert [info.event_type for info in opened] == ["motion"]
    assert opened[0].thumbnail_path is None
    assert closed[0].ended_at == _utc(104.0)
    assert list(tmp_path.rglob("*.jpg")) == []


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def test_thumbnail_rel_path() -> None:
    assert EventBuilder.thumbnail_rel_path("yard", 42) == "yard/thumbnails/42.jpg"


def test_write_thumbnail_is_atomic_valid_jpeg_with_the_box_drawn(tmp_path: Path) -> None:
    builder = _builder(tmp_path, FakeDb())
    frame = _frame()

    builder.write_thumbnail("yard/thumbnails/7.jpg", frame, (100, 50, 40, 60))

    path = tmp_path / "yard" / "thumbnails" / "7.jpg"
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8"
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (FRAME_H, FRAME_W, 3)
    blue, green, red = (int(v) for v in image[50, 120])  # top edge of the box
    assert green > 150 and blue < 100 and red < 100
    assert int(image[120, 20].max()) < 30  # far from the box: still black
    assert [p.name for p in path.parent.iterdir()] == ["7.jpg"]  # no temp file left
    assert int(frame.max()) == 0  # the caller's frame was not drawn on


# ---------------------------------------------------------------------------
# Real database round trip (db/schema.py helpers from Task 7)
# ---------------------------------------------------------------------------


def test_event_row_round_trip_with_real_schema(clean_db: None, tmp_path: Path) -> None:
    builder = EventBuilder(camera="yard", output_dir=tmp_path, area_masks=[])
    track = _track(first=1_700_000_000.0, last=1_700_000_000.4, conf=0.82)
    builder.on_tracks(_update(opened=(track,)), SHAPE)
    track.last_seen = 1_700_000_003.0
    builder.on_tracks(_update(closed=(track,)), SHAPE)

    assert track.event_id is not None
    row = schema.get_event(track.event_id)
    assert row is not None
    assert row.camera_name == "yard"
    assert row.label == "person"
    assert row.event_type == "object"
    assert row.track_id == 1
    assert row.confidence == pytest.approx(0.82)
    assert row.thumbnail_path == f"yard/thumbnails/{track.event_id}.jpg"
    assert (tmp_path / row.thumbnail_path).is_file()
    assert schema.as_utc(row.created_at) == _utc(1_700_000_000.0)
    assert schema.as_utc(row.ended_at) == _utc(1_700_000_003.0)
    assert [event.id for event in schema.list_events(camera_name="yard")] == [track.event_id]


# ---------------------------------------------------------------------------
# EventSink (legacy result sink) on insert_event
# ---------------------------------------------------------------------------


def test_event_sink_writes_one_row_per_detection_with_camera_name() -> None:
    db = FakeDb()
    sink = EventSink(db=db)

    sink(
        "yard",
        "main",
        [
            Detection(kind="motion", confidence=0.8, bbox=(1, 2, 3, 4), ts_unix=1_700_000_000.0),
            Detection(kind="person", confidence=0.9, ts_unix=1_700_000_001.0),
        ],
    )

    inserts = db.of("insert")
    assert [call[2]["label"] for call in inserts] == ["motion", "person"]
    first = inserts[0][2]
    assert first["camera_name"] == "yard"
    assert first["event_type"] == "motion"
    assert first["confidence"] == pytest.approx(0.8)
    assert first["zone"] == ""
    assert first["track_id"] is None
    assert first["created_at"] == _utc(1_700_000_000.0)
    assert first["message"] == "motion detected on yard/main (confidence=0.80)"
    assert first["metadata"] == {"bbox": [1, 2, 3, 4], "stream": "main", "ts_unix": 1_700_000_000.0}
    assert db.of("update") == [("update", inserts[1][1], {"severity": "warn"})]


def test_event_sink_ignores_empty_lists() -> None:
    db = FakeDb()
    EventSink(db=db)("yard", "main", [])
    assert db.calls == []


def test_event_sink_logs_and_continues_when_the_database_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(sinks.logger, "warning", lambda msg, *args, **kwargs: warnings.append(msg))
    detections = [
        Detection(kind="motion", confidence=0.5, ts_unix=1.0),
        Detection(kind="motion", confidence=0.6, ts_unix=2.0),
    ]

    EventSink(db=FailingInsertDb())("yard", "main", detections)

    assert warnings == ["failed to insert event for %s/%s kind=%s"] * 2


def test_event_sink_real_database_row(clean_db: None) -> None:
    EventSink()("yard", "main", [Detection(kind="person", confidence=0.9, ts_unix=1_700_000_000.0)])

    rows = schema.list_events(camera_name="yard")
    assert len(rows) == 1
    assert rows[0].camera_name == "yard"
    assert rows[0].label == "person"
    assert rows[0].event_type == "person"
    assert rows[0].severity == "warn"
    assert schema.as_utc(rows[0].created_at) == _utc(1_700_000_000.0)
