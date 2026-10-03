"""Tests for recording modes (continuous vs event-only) and related DB queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rtsp_warden.config import CameraConfig, EventRecordConfig, RecordConfig
from rtsp_warden.db.engine import reset_engine
from rtsp_warden.db.schema import ensure_schema, get_latest_event_for_camera, insert_event

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_db(tmp_path: pytest.TestPath, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up an isolated SQLite DB with schema for event tests."""
    db_url = f"sqlite:///{tmp_path}/event_test.db"
    monkeypatch.setenv("WARDEN_DB_URL", db_url)
    reset_engine()
    ensure_schema()
    yield
    reset_engine()


def _insert(camera: str, message: str, ago_seconds: float = 0.0) -> int:
    """Insert a motion event for *camera* created *ago_seconds* before now."""
    return insert_event(
        camera_name=camera,
        event_type="motion",
        label="motion",
        confidence=1.0,
        zone="",
        track_id=None,
        message=message,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=ago_seconds),
    )


# ---------------------------------------------------------------------------
# DB: get_latest_event_for_camera
# ---------------------------------------------------------------------------


def test_get_latest_event_for_camera_returns_none_when_no_events(
    event_db: None,
) -> None:
    """Returns None when no events exist for the camera."""
    result = get_latest_event_for_camera("front_door", since_seconds=60)
    assert result is None


def test_get_latest_event_for_camera_returns_recent(event_db: None) -> None:
    """Returns a recent event when one exists within the window."""
    event_id = _insert("garage", "movement")
    result = get_latest_event_for_camera("garage", since_seconds=60)
    assert result is not None
    assert result.id == event_id
    assert result.event_type == "motion"


def test_get_latest_event_for_camera_filters_old(event_db: None) -> None:
    """Returns None when the only event is older than since_seconds."""
    _insert("driveway", "old movement", ago_seconds=100)
    result = get_latest_event_for_camera("driveway", since_seconds=60)
    assert result is None


def test_get_latest_event_for_camera_returns_most_recent(event_db: None) -> None:
    """When multiple events exist, returns the most recent one."""
    for i, offset in enumerate([3, 2, 0]):
        _insert("patio", f"event_{i}", ago_seconds=offset)

    result = get_latest_event_for_camera("patio", since_seconds=0)
    assert result is not None
    # The most recent event (offset=0) should be returned
    assert result.message == "event_2"


def test_get_latest_event_for_camera_unknown_camera(event_db: None) -> None:
    """Returns None when no event carries that camera name."""
    _insert("patio", "elsewhere")
    result = get_latest_event_for_camera("nonexistent_cam", since_seconds=60)
    assert result is None


# ---------------------------------------------------------------------------
# Config: RecordConfig and EventRecordConfig
# ---------------------------------------------------------------------------


def test_record_config_default_is_continuous() -> None:
    """RecordConfig defaults to mode='continuous'."""
    cfg = RecordConfig()
    assert cfg.mode == "continuous"


def test_record_config_event_with_event_record() -> None:
    """RecordConfig can be created in event mode with nested EventRecordConfig."""
    cfg = RecordConfig(
        mode="event", event_record=EventRecordConfig(pre_seconds=10, post_seconds=30)
    )
    assert cfg.mode == "event"
    assert cfg.event_record.pre_seconds == 10
    assert cfg.event_record.post_seconds == 30
    assert cfg.event_record.min_segment_seconds == 10
    assert cfg.event_record.max_segment_seconds == 600


def test_event_record_validates_positive_seconds() -> None:
    """EventRecordConfig raises ValueError for non-positive values."""
    with pytest.raises(ValueError, match="value must be > 0"):
        EventRecordConfig(pre_seconds=0)

    with pytest.raises(ValueError, match="value must be > 0"):
        EventRecordConfig(post_seconds=-1)

    with pytest.raises(ValueError, match="value must be > 0"):
        EventRecordConfig(min_segment_seconds=0)

    with pytest.raises(ValueError, match="value must be > 0"):
        EventRecordConfig(max_segment_seconds=-5)


def test_event_record_default_values() -> None:
    """EventRecordConfig has sensible defaults."""
    cfg = EventRecordConfig()
    assert cfg.pre_seconds == 5
    assert cfg.post_seconds == 10
    assert cfg.min_segment_seconds == 10
    assert cfg.max_segment_seconds == 600


# ---------------------------------------------------------------------------
# CameraRecorder: event-mode _should_be_recording
# ---------------------------------------------------------------------------


def test_should_be_recording_returns_true_when_event_exists(event_db: None) -> None:
    """_should_be_recording returns True when a recent event exists."""
    from rtsp_warden.config import RuntimeConfig
    from rtsp_warden.recorder import CameraRecorder

    cam_cfg = CameraConfig(
        name="lobby",
        main_url="rtsp://example.com/main",
        sub_url="rtsp://example.com/sub",
        record=RecordConfig(mode="event", event_record=EventRecordConfig(post_seconds=60)),
    )
    recorder = CameraRecorder(camera=cam_cfg, runtime=RuntimeConfig())

    # No events yet — should not record
    assert recorder._should_be_recording() is False

    # Create a recent event
    _insert("lobby", "person detected")
    assert recorder._should_be_recording() is True
