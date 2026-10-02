from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rtsp_warden.clips import ClipError, ClipGenerator
from rtsp_warden.config import AppConfig, CameraConfig, ClipsConfig


def test_find_segments_matches_recorder_naming(tmp_path: Path):
    seg_dir = tmp_path / "foscam_c1" / "main"
    seg_dir.mkdir(parents=True)
    (seg_dir / "foscam_c1_main_20261002_025541.ts").write_bytes(b"")
    (seg_dir / "foscam_c1_main_20261002_025641.ts").write_bytes(b"")
    (seg_dir / "notes.txt").write_bytes(b"")

    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    # Segment names are local time (ffmpeg -strftime), so build the window the same way.
    start = datetime(2026, 10, 2, 2, 55, 0).astimezone()
    end = datetime(2026, 10, 2, 2, 56, 0).astimezone()
    found = gen.find_segments("foscam_c1", "main", start, end, segment_duration=60.0)
    assert [p.name for p in found] == ["foscam_c1_main_20261002_025541.ts"]


def test_find_segments_still_accepts_bare_timestamp_names(tmp_path: Path):
    seg_dir = tmp_path / "cam" / "main"
    seg_dir.mkdir(parents=True)
    (seg_dir / "20261002_025541.ts").write_bytes(b"")
    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    start = datetime(2026, 10, 2, 2, 55, 0).astimezone()
    end = datetime(2026, 10, 2, 2, 56, 0).astimezone()
    assert len(gen.find_segments("cam", "main", start, end, segment_duration=60.0)) == 1


def _local_named_segment(seg_dir: Path, local_start: datetime) -> Path:
    p = seg_dir / f"cam_main_{local_start.strftime('%Y%m%d_%H%M%S')}.ts"
    p.write_bytes(b"")
    return p


def test_find_segments_parses_names_as_local_time_and_uses_chunk_length(tmp_path: Path):
    seg_dir = tmp_path / "cam" / "main"
    seg_dir.mkdir(parents=True)
    local_start = datetime(2026, 3, 1, 12, 0, 0).astimezone()  # naive local -> aware local
    _local_named_segment(seg_dir, local_start)
    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    # An event 30s into a 60s segment, expressed in UTC as the events table stores it.
    start = (local_start + timedelta(seconds=20)).astimezone(timezone.utc)
    end = (local_start + timedelta(seconds=40)).astimezone(timezone.utc)
    assert gen.find_segments("cam", "main", start, end, segment_duration=60.0)
    assert not gen.find_segments("cam", "main", start, end, segment_duration=4.0)


def test_generate_passes_segment_duration_to_find_segments(tmp_path: Path, monkeypatch):
    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    seen: dict = {}

    def fake_find(camera, stream, start, end, segment_duration=4.0):
        seen["segment_duration"] = segment_duration
        return []

    monkeypatch.setattr(gen, "find_segments", fake_find)
    with pytest.raises(ClipError):
        gen.generate(
            camera_name="cam",
            stream="main",
            event_start=datetime.now(timezone.utc),
            event_id=1,
            segment_duration=300.0,
        )
    assert seen["segment_duration"] == 300.0


def test_clip_stream_prefers_main_when_no_sub():
    from rtsp_warden.web.routes.events import clip_stream_and_chunk

    cfg = AppConfig(
        cameras=[
            CameraConfig(name="a", main_url="rtsp://h/a", record={"main": {"chunk_seconds": 60}}),
            CameraConfig(name="b", main_url="rtsp://h/b", sub_url="rtsp://h/b2"),
        ]
    )
    assert clip_stream_and_chunk(cfg, "a") == ("main", 60.0)
    assert clip_stream_and_chunk(cfg, "b") == ("sub", 300.0)
    assert clip_stream_and_chunk(cfg, "zzz") == ("main", 300.0)
