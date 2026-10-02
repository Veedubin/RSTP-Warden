from datetime import datetime, timezone
from pathlib import Path

from rtsp_warden.clips import ClipGenerator
from rtsp_warden.config import ClipsConfig


def test_find_segments_matches_recorder_naming(tmp_path: Path):
    seg_dir = tmp_path / "foscam_c1" / "main"
    seg_dir.mkdir(parents=True)
    (seg_dir / "foscam_c1_main_20261002_025541.ts").write_bytes(b"")
    (seg_dir / "foscam_c1_main_20261002_025641.ts").write_bytes(b"")
    (seg_dir / "notes.txt").write_bytes(b"")

    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    start = datetime(2026, 10, 2, 2, 55, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 2, 2, 56, 0, tzinfo=timezone.utc)
    found = gen.find_segments("foscam_c1", "main", start, end, segment_duration=60.0)
    assert [p.name for p in found] == ["foscam_c1_main_20261002_025541.ts"]


def test_find_segments_still_accepts_bare_timestamp_names(tmp_path: Path):
    seg_dir = tmp_path / "cam" / "main"
    seg_dir.mkdir(parents=True)
    (seg_dir / "20261002_025541.ts").write_bytes(b"")
    gen = ClipGenerator(cfg=ClipsConfig(), recordings_dir=tmp_path)
    start = datetime(2026, 10, 2, 2, 55, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 2, 2, 56, 0, tzinfo=timezone.utc)
    assert len(gen.find_segments("cam", "main", start, end, segment_duration=60.0)) == 1
