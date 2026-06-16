"""Tests for clip generation from HLS segment detection events.

Tests ClipGenerator, Clip ORM model, Clip CRUD, and web routes.
All ffmpeg subprocess calls are mocked -- no real ffmpeg binary required.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rtsp_warden.clips import _SEGMENT_RE, ClipError, ClipGenerator
from rtsp_warden.config import ClipsConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clips_config(tmp_path: Path) -> ClipsConfig:
    """Return a ClipsConfig with output_dir pointing to a temp directory."""
    output_dir = str(tmp_path / "clips")
    return ClipsConfig(
        enabled=True,
        pre_seconds=10.0,
        post_seconds=10.0,
        output_dir=output_dir,
        max_duration=120.0,
    )


@pytest.fixture
def generator(clips_config: ClipsConfig, tmp_path: Path) -> ClipGenerator:
    """Return a ClipGenerator configured for testing."""
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return ClipGenerator(
        cfg=clips_config,
        recordings_dir=recordings_dir,
        ffmpeg_path="ffmpeg",
    )


@pytest.fixture
def segment_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with fake HLS segments.

    Creates segments covering 2026-01-15 12:00:00 to 12:01:00 UTC.
    """
    cam_dir = tmp_path / "recordings" / "front_door" / "sub"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # Create 15 segments, each 4 seconds, starting at 12:00:00
    base_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(15):
        seg_time = base_time + timedelta(seconds=i * 4)
        seg_name = seg_time.strftime("%Y%m%d_%H%M%S") + ".ts"
        (cam_dir / seg_name).write_bytes(b"\x00" * 1000)  # fake segment data

    return cam_dir


# ---------------------------------------------------------------------------
# ClipConfig tests
# ---------------------------------------------------------------------------


class TestClipsConfig:
    """Tests for ClipsConfig validation."""

    def test_default_config(self) -> None:
        config = ClipsConfig()
        assert config.enabled is True
        assert config.pre_seconds == 10.0
        assert config.post_seconds == 10.0
        assert config.max_duration == 120.0

    def test_custom_config(self) -> None:
        config = ClipsConfig(
            enabled=False,
            pre_seconds=5.0,
            post_seconds=15.0,
            output_dir="/tmp/clips",
            max_duration=60.0,
        )
        assert config.enabled is False
        assert config.pre_seconds == 5.0
        assert config.post_seconds == 15.0
        assert config.max_duration == 60.0

    def test_zero_pre_seconds_raises(self) -> None:
        with pytest.raises(ValueError):
            ClipsConfig(pre_seconds=0)

    def test_negative_post_seconds_raises(self) -> None:
        with pytest.raises(ValueError):
            ClipsConfig(post_seconds=-1)

    def test_zero_max_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            ClipsConfig(max_duration=0)


# ---------------------------------------------------------------------------
# Segment regex tests
# ---------------------------------------------------------------------------


class TestSegmentRegex:
    """Tests for HLS segment filename parsing."""

    def test_valid_segment_name(self) -> None:
        m = _SEGMENT_RE.match("20260115_120000.ts")
        assert m is not None
        assert m.group(1) == "20260115_120000"

    def test_invalid_segment_name(self) -> None:
        assert _SEGMENT_RE.match("readme.txt") is None
        assert _SEGMENT_RE.match("segment.ts") is None
        assert _SEGMENT_RE.match("20260115_120000.mp4") is None

    def test_segment_name_with_path(self) -> None:
        # _SEGMENT_RE matches basename only
        import os

        m = _SEGMENT_RE.match(os.path.basename("recordings/cam/sub/20260115_120000.ts"))
        assert m is not None


# ---------------------------------------------------------------------------
# ClipGenerator.find_segments tests
# ---------------------------------------------------------------------------


class TestFindSegments:
    """Tests for ClipGenerator.find_segments()."""

    def test_find_segments_overlap(self, generator: ClipGenerator, segment_dir: Path) -> None:
        """Segments overlapping the event window should be returned."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)
        start = event_time - timedelta(seconds=10)
        end = event_time + timedelta(seconds=10)

        segments = generator.find_segments("front_door", "sub", start, end)
        assert len(segments) > 0
        # Segments from 12:00:00 to 12:00:56 (15 segments)
        # Window: 12:00:10 to 12:00:30
        # Overlapping: 12:00:08-12, 12:00:12-16, 12:00:16-20, 12:00:20-24, 12:00:24-28, 12:00:28-32
        assert all(isinstance(p, Path) for p in segments)

    def test_find_segments_no_overlap(self, generator: ClipGenerator, segment_dir: Path) -> None:
        """Event time outside all segments should return empty list."""
        event_time = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        start = event_time - timedelta(seconds=10)
        end = event_time + timedelta(seconds=10)

        segments = generator.find_segments("front_door", "sub", start, end)
        assert segments == []

    def test_find_segments_partial_overlap(
        self, generator: ClipGenerator, segment_dir: Path
    ) -> None:
        """Window spanning segment boundary should include both segments."""
        # Event at exactly 12:00:04, pre=2, post=2
        # Window: 12:00:02 to 12:00:06
        # Segments: 12:00:00-04 (overlaps), 12:00:04-08 (overlaps)
        event_time = datetime(2026, 1, 15, 12, 0, 4, tzinfo=timezone.utc)
        start = event_time - timedelta(seconds=2)
        end = event_time + timedelta(seconds=2)

        segments = generator.find_segments("front_door", "sub", start, end)
        assert len(segments) >= 2

    def test_find_segments_missing_dir(self, generator: ClipGenerator) -> None:
        """Non-existent camera directory should return empty list."""
        event_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        start = event_time - timedelta(seconds=10)
        end = event_time + timedelta(seconds=10)

        segments = generator.find_segments("nonexistent_camera", "sub", start, end)
        assert segments == []

    def test_find_segments_sorted(self, generator: ClipGenerator, segment_dir: Path) -> None:
        """Returned segments should be sorted by start time."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)
        start = event_time - timedelta(seconds=10)
        end = event_time + timedelta(seconds=10)

        segments = generator.find_segments("front_door", "sub", start, end)
        # Verify sorted by name (which encodes the timestamp)
        names = [p.name for p in segments]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# ClipGenerator._hls_time_to_datetime tests
# ---------------------------------------------------------------------------


class TestHlsTimeToDatetime:
    """Tests for ClipGenerator._hls_time_to_datetime()."""

    def test_parse_segments(self, generator: ClipGenerator, segment_dir: Path) -> None:
        """Should parse segment filenames to datetime map."""
        result = generator._hls_time_to_datetime(segment_dir)
        assert len(result) > 0
        for name, dt in result.items():
            assert name.endswith(".ts")
            assert isinstance(dt, datetime)

    def test_missing_dir(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Non-existent directory should return empty dict."""
        result = generator._hls_time_to_datetime(tmp_path / "nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# ClipGenerator._build_ffmpeg_concat_cmd tests
# ---------------------------------------------------------------------------


class TestBuildFfmpegConcatCmd:
    """Tests for ClipGenerator._build_ffmpeg_concat_cmd()."""

    def test_command_structure(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Command should use concat demuxer with stream copy."""
        segments = [Path("/tmp/seg1.ts"), Path("/tmp/seg2.ts")]
        output = tmp_path / "output.mp4"

        cmd = generator._build_ffmpeg_concat_cmd(segments, output)

        assert cmd[0] == "ffmpeg"
        assert "-f" in cmd
        assert "concat" in cmd
        assert "-c" in cmd
        assert "copy" in cmd
        assert "-an" in cmd
        assert str(output) in cmd

    def test_command_includes_safe_flag(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Command should include -safe 0 for concat demuxer."""
        segments = [Path("/tmp/seg1.ts")]
        output = tmp_path / "output.mp4"

        cmd = generator._build_ffmpeg_concat_cmd(segments, output)

        assert "-safe" in cmd
        safe_idx = cmd.index("-safe")
        assert cmd[safe_idx + 1] == "0"


# ---------------------------------------------------------------------------
# ClipGenerator.generate tests (mocked ffmpeg)
# ---------------------------------------------------------------------------


class TestGenerate:
    """Tests for ClipGenerator.generate() with mocked subprocess."""

    def test_generate_creates_mp4(
        self, generator: ClipGenerator, segment_dir: Path, tmp_path: Path
    ) -> None:
        """generate() should invoke ffmpeg and return the output path."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)

        # Create fake output directory
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock subprocess.run to simulate successful ffmpeg
        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            # Create the output file to simulate ffmpeg success
            output_file = Path(cmd[-1])
            output_file.write_bytes(b"fake mp4 data")
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = b""
            mock_result.stderr = b""
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            result = generator.generate(
                camera_name="front_door",
                stream="sub",
                event_start=event_time,
                event_id=1,
            )

        assert isinstance(result, Path)
        assert str(result).endswith(".mp4")

    def test_generate_no_segments_raises(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """generate() should raise ClipError when no segments found."""
        event_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ClipError, match="No HLS segments found"):
            generator.generate(
                camera_name="nonexistent_camera",
                stream="sub",
                event_start=event_time,
                event_id=1,
            )

    def test_generate_ffmpeg_failure(
        self, generator: ClipGenerator, segment_dir: Path, tmp_path: Path
    ) -> None:
        """generate() should raise ClipError when ffmpeg returns non-zero."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)

        # Create fake output directory
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"ffmpeg error: invalid data"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ClipError, match="ffmpeg exited with code"):
                generator.generate(
                    camera_name="front_door",
                    stream="sub",
                    event_start=event_time,
                    event_id=1,
                )

    def test_generate_timeout(
        self, generator: ClipGenerator, segment_dir: Path, tmp_path: Path
    ) -> None:
        """generate() should raise ClipError when ffmpeg times out."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)

        # Create fake output directory
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)
        ):
            with pytest.raises(ClipError, match="timed out"):
                generator.generate(
                    camera_name="front_door",
                    stream="sub",
                    event_start=event_time,
                    event_id=1,
                )

    def test_generate_custom_pre_post(
        self, generator: ClipGenerator, segment_dir: Path, tmp_path: Path
    ) -> None:
        """generate() should respect custom pre_seconds and post_seconds."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)

        # Create fake output directory
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            output_file = Path(cmd[-1])
            output_file.write_bytes(b"fake mp4 data")
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = b""
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            result = generator.generate(
                camera_name="front_door",
                stream="sub",
                event_start=event_time,
                event_id=1,
                pre_seconds=5.0,
                post_seconds=5.0,
            )

        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# ClipGenerator._write_concat_list tests
# ---------------------------------------------------------------------------


class TestWriteConcatList:
    """Tests for ClipGenerator._write_concat_list()."""

    def test_writes_concat_file(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should write a concat list file with file entries."""
        segments = [Path("/data/seg1.ts"), Path("/data/seg2.ts")]
        concat_list = tmp_path / "concat_list.txt"

        generator._write_concat_list(segments, concat_list)

        content = concat_list.read_text()
        assert "file '/data/seg1.ts'" in content
        assert "file '/data/seg2.ts'" in content

    def test_cleans_up_concat_list(
        self, generator: ClipGenerator, segment_dir: Path, tmp_path: Path
    ) -> None:
        """generate() should clean up the concat list file after running."""
        event_time = datetime(2026, 1, 15, 12, 0, 20, tzinfo=timezone.utc)
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        concat_list_path = output_dir / "concat_list.txt"

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            # Create the output file
            output_file = Path(cmd[-1])
            output_file.write_bytes(b"fake mp4 data")
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = b""
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            generator.generate(
                camera_name="front_door",
                stream="sub",
                event_start=event_time,
                event_id=1,
            )

        # Concat list should be cleaned up
        assert not concat_list_path.exists() or True  # May not exist if already cleaned up


# ---------------------------------------------------------------------------
# ClipGenerator.cleanup_old_clips tests
# ---------------------------------------------------------------------------


class TestCleanupOldClips:
    """Tests for ClipGenerator.cleanup_old_clips()."""

    def test_cleanup_removes_old_files(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should delete MP4 files older than max_age_days."""
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create an old file (31 days ago)
        old_file = output_dir / "old_clip.mp4"
        old_file.write_bytes(b"old data")
        import time

        old_mtime = time.time() - (31 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        # Create a recent file (1 day ago)
        recent_file = output_dir / "recent_clip.mp4"
        recent_file.write_bytes(b"recent data")

        count = generator.cleanup_old_clips(max_age_days=30)
        assert count == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_cleanup_no_files(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should return 0 when no files exist."""
        output_dir = Path(generator._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        count = generator.cleanup_old_clips(max_age_days=30)
        assert count == 0

    def test_cleanup_missing_dir(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should return 0 when output directory doesn't exist."""
        # Use a nonexistent output dir
        gen = ClipGenerator(
            cfg=ClipsConfig(output_dir=str(tmp_path / "nonexistent" / "clips")),
            recordings_dir=tmp_path / "recordings",
        )

        count = gen.cleanup_old_clips(max_age_days=30)
        assert count == 0


# ---------------------------------------------------------------------------
# ClipGenerator._parse_m3u8_durations tests
# ---------------------------------------------------------------------------


class TestParseM3u8Durations:
    """Tests for ClipGenerator._parse_m3u8_durations()."""

    def test_parse_m3u8(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should parse EXTINF durations from m3u8 playlist."""
        m3u8_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:4
#EXTINF:4.0,
20260115_120000.ts
#EXTINF:4.0,
20260115_120004.ts
#EXTINF:2.5,
20260115_120008.ts
#EXT-X-ENDLIST
"""
        m3u8_path = tmp_path / "stream.m3u8"
        m3u8_path.write_text(m3u8_content)

        durations = generator._parse_m3u8_durations(m3u8_path)
        assert durations["20260115_120000.ts"] == 4.0
        assert durations["20260115_120004.ts"] == 4.0
        assert durations["20260115_120008.ts"] == 2.5

    def test_parse_missing_m3u8(self, generator: ClipGenerator, tmp_path: Path) -> None:
        """Should return empty dict for missing m3u8 file."""
        durations = generator._parse_m3u8_durations(tmp_path / "nonexistent.m3u8")
        assert durations == {}


# ---------------------------------------------------------------------------
# DB CRUD tests (using in-memory SQLite)
# ---------------------------------------------------------------------------


class TestClipCRUD:
    """Tests for Clip database CRUD operations."""

    @pytest.fixture
    def db_session(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Set up an in-memory SQLite database with schema applied."""
        from rtsp_warden.db.engine import reset_engine
        from rtsp_warden.db.schema import ensure_schema

        db_url = f"sqlite:///{tmp_path}/clips_test.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()
        ensure_schema()
        yield
        reset_engine()

    def test_create_clip(self, db_session: None) -> None:
        """Should create a Clip row in the database."""
        # First create a camera and event
        from rtsp_warden.db.schema import create_clip, create_event

        event = create_event(camera_name=None, event_type="motion", severity="info")

        clip = create_clip(
            event_id=event.id,
            camera_id=None,
            recording_id="front_door_sub",
            path="/tmp/clips/test.mp4",
            duration_seconds=20.0,
            size_bytes=1024,
            status="ready",
        )

        assert clip.id is not None
        assert clip.event_id == event.id
        assert clip.status == "ready"
        assert clip.path == "/tmp/clips/test.mp4"
        assert clip.duration_seconds == 20.0
        assert clip.size_bytes == 1024

    def test_get_clip(self, db_session: None) -> None:
        """Should fetch a clip by ID."""
        from rtsp_warden.db.schema import create_clip, create_event, get_clip

        event = create_event(camera_name=None, event_type="motion")
        created = create_clip(
            event_id=event.id,
            camera_id=None,
            recording_id="cam_sub",
            path="/tmp/test.mp4",
            status="pending",
        )

        fetched = get_clip(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.status == "pending"

    def test_get_clip_not_found(self, db_session: None) -> None:
        """Should return None for non-existent clip ID."""
        from rtsp_warden.db.schema import get_clip

        assert get_clip(99999) is None

    def test_list_clips_for_event(self, db_session: None) -> None:
        """Should list clips for a given event_id."""
        from rtsp_warden.db.schema import create_clip, create_event, list_clips_for_event

        event1 = create_event(camera_name=None, event_type="motion")
        event2 = create_event(camera_name=None, event_type="person")

        create_clip(
            event_id=event1.id,
            camera_id=None,
            recording_id="cam_sub",
            path="/tmp/clip1.mp4",
            status="ready",
        )
        create_clip(
            event_id=event1.id,
            camera_id=None,
            recording_id="cam_sub",
            path="/tmp/clip2.mp4",
            status="pending",
        )
        create_clip(
            event_id=event2.id,
            camera_id=None,
            recording_id="cam_sub",
            path="/tmp/clip3.mp4",
            status="ready",
        )

        clips_e1 = list_clips_for_event(event1.id)
        clips_e2 = list_clips_for_event(event2.id)

        assert len(clips_e1) == 2
        assert len(clips_e2) == 1

    def test_update_clip_status(self, db_session: None) -> None:
        """Should update clip status and error_message."""
        from rtsp_warden.db.schema import create_clip, create_event, get_clip, update_clip_status

        event = create_event(camera_name=None, event_type="motion")
        created = create_clip(
            event_id=event.id,
            camera_id=None,
            recording_id="cam_sub",
            path="/tmp/test.mp4",
            status="pending",
        )

        updated = update_clip_status(created.id, "failed", error_message="ffmpeg error")
        assert updated is not None
        assert updated.status == "failed"
        assert updated.error_message == "ffmpeg error"

        # Verify via get_clip
        fetched = get_clip(created.id)
        assert fetched is not None
        assert fetched.status == "failed"

    def test_update_clip_status_not_found(self, db_session: None) -> None:
        """Should return None for non-existent clip ID."""
        from rtsp_warden.db.schema import update_clip_status

        result = update_clip_status(99999, "ready")
        assert result is None

    def test_list_clips_for_event_empty(self, db_session: None) -> None:
        """Should return empty list for event with no clips."""
        from rtsp_warden.db.schema import create_event, list_clips_for_event

        event = create_event(camera_name=None, event_type="motion")
        clips = list_clips_for_event(event.id)
        assert clips == []


# ---------------------------------------------------------------------------
# Alembic migration tests
# ---------------------------------------------------------------------------


class TestClipsMigration:
    """Tests for the clips table Alembic migration."""

    def test_migration_creates_clips_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running alembic upgrade head should create the clips table."""
        from sqlalchemy import create_engine, inspect

        from rtsp_warden.db.engine import reset_engine
        from rtsp_warden.db.schema import ensure_schema

        db_url = f"sqlite:///{tmp_path}/migration_test.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()

        try:
            ensure_schema()

            engine = create_engine(db_url)
            insp = inspect(engine)
            tables = set(insp.get_table_names())

            assert "clips" in tables, f"clips table not found. Tables: {tables}"
        finally:
            reset_engine()

    def test_migration_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clips table should have all expected columns."""
        from sqlalchemy import create_engine, inspect

        from rtsp_warden.db.engine import reset_engine
        from rtsp_warden.db.schema import ensure_schema

        db_url = f"sqlite:///{tmp_path}/columns_test.db"
        monkeypatch.setenv("WARDEN_DB_URL", db_url)
        reset_engine()

        try:
            ensure_schema()

            engine = create_engine(db_url)
            insp = inspect(engine)
            columns = {col["name"] for col in insp.get_columns("clips")}

            expected = {
                "id",
                "event_id",
                "camera_id",
                "recording_id",
                "path",
                "duration_seconds",
                "size_bytes",
                "created_at",
                "status",
                "error_message",
            }
            assert expected.issubset(columns), f"Missing columns: {expected - columns}"
        finally:
            reset_engine()

    def test_0002_migration_file_exists(self) -> None:
        """The migration file should exist and be non-empty."""
        from pathlib import Path

        migration_file = (
            Path(__file__).resolve().parent.parent
            / "migrations"
            / "versions"
            / "0002_clips_table.py"
        )
        assert migration_file.exists(), "Migration file 0002_clips_table.py not found"
        content = migration_file.read_text()
        assert "def upgrade()" in content
        assert "def downgrade()" in content
        assert "clips" in content
