"""Event clips from recorded .ts segments (spec 8.4, rulings R8 and R20).

Segment names are local wall time (ffmpeg -strftime); event times are UTC. ffmpeg is never
run: subprocess.run is replaced by fakes that write ``Path(cmd[-1])``.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rtsp_warden import clips
from rtsp_warden.clips import (
    _SEGMENT_RE,
    CLIP_FFMPEG_TIMEOUT_S,
    build_event_clip,
    clip_rel_path,
    clip_stream_and_chunk,
    select_segments,
)
from rtsp_warden.config import CameraConfig, ClipsConfig

# Naive local wall time made aware in the local zone, the way the recorder names files.
LOCAL0 = datetime(2026, 3, 1, 12, 0, 0).astimezone()


def _seg(camera_root: Path, offset_s: int, stream: str = "main", camera: str = "yard") -> Path:
    """Write a recorder-named segment that starts offset_s seconds after LOCAL0."""
    seg_dir = camera_root / stream
    seg_dir.mkdir(parents=True, exist_ok=True)
    start = LOCAL0 + timedelta(seconds=offset_s)
    path = seg_dir / f"{camera}_{stream}_{start:%Y%m%d_%H%M%S}.ts"
    path.write_bytes(b"\x47" * 188)
    return path


def _utc(offset_s: float) -> datetime:
    """The instant offset_s seconds after LOCAL0, as an aware UTC datetime."""
    return (LOCAL0 + timedelta(seconds=offset_s)).astimezone(timezone.utc)


def _ok(cmd: list[str], **_kw: Any) -> MagicMock:
    Path(cmd[-1]).write_bytes(b"clip")
    return MagicMock(returncode=0, stdout=b"", stderr=b"")


def _build(camera_root: Path, *, runner: Any = None, **overrides: Any) -> Path | None:
    kwargs: dict[str, Any] = {
        "camera_root": camera_root,
        "stream": "main",
        "chunk_seconds": 60,
        "started_at": _utc(115),
        "ended_at": _utc(120),
        "pre_seconds": 10.0,
        "post_seconds": 10.0,
        "max_duration": 120.0,
        "event_id": 7,
        "ffmpeg_path": "ffmpeg",
        "runner": runner,
    }
    kwargs.update(overrides)
    return build_event_clip(**kwargs)


# ---------------------------------------------------------------------------
# Segment names and selection
# ---------------------------------------------------------------------------


def test_segment_regex_matches_recorder_and_bare_names() -> None:
    m = _SEGMENT_RE.match("foscam_c1_main_20261002_025541.ts")
    assert m is not None and m.group(1) == "20261002_025541"
    m = _SEGMENT_RE.match("20260115_120000.ts")
    assert m is not None and m.group(1) == "20260115_120000"
    for name in ("readme.txt", "segment.ts", "20260115_120000.mp4", "cam_main_20260115.ts"):
        assert _SEGMENT_RE.match(name) is None


def test_select_segments_ends_a_segment_where_the_next_starts(tmp_path: Path) -> None:
    root = tmp_path / "yard"
    s0, s60, s120 = _seg(root, 0), _seg(root, 60), _seg(root, 120)

    assert select_segments(root, "main", _utc(65), _utc(85), 60) == [(s60, _utc(60))]
    assert [p for p, _ in select_segments(root, "main", _utc(105), _utc(130), 60)] == [s60, s120]
    assert [p for p, _ in select_segments(root, "main", _utc(-5), _utc(1), 60)] == [s0]


def test_select_segments_long_segment_still_covers_the_window_start(tmp_path: Path) -> None:
    """The muxer cuts on a keyframe, so a 60 s segment can run to 68 s (G6)."""
    root = tmp_path / "yard"
    s0 = _seg(root, 0)
    _seg(root, 68)

    assert [p for p, _ in select_segments(root, "main", _utc(65), _utc(67), 60)] == [s0]


def test_select_segments_caps_a_segment_before_a_recording_gap(tmp_path: Path) -> None:
    root = tmp_path / "yard"
    _seg(root, 0)
    _seg(root, 600)

    assert select_segments(root, "main", _utc(200), _utc(210), 60) == []


def test_select_segments_reads_naive_bounds_as_utc(tmp_path: Path) -> None:
    root = tmp_path / "yard"
    s60 = _seg(root, 60)
    naive_start = _utc(65).replace(tzinfo=None)
    naive_end = _utc(85).replace(tzinfo=None)

    assert [p for p, _ in select_segments(root, "main", naive_start, naive_end, 60)] == [s60]


def test_select_segments_ignores_other_files_and_missing_dirs(tmp_path: Path) -> None:
    root = tmp_path / "yard"
    _seg(root, 0)
    (root / "main" / "notes.txt").write_text("x")
    (root / "main" / f"yard_main_{LOCAL0:%Y%m%d_%H%M%S}.mkv").write_bytes(b"")

    assert len(select_segments(root, "main", _utc(0), _utc(10), 60)) == 1
    assert select_segments(root, "sub", _utc(0), _utc(10), 60) == []
    assert select_segments(tmp_path / "nope", "main", _utc(0), _utc(10), 60) == []


# ---------------------------------------------------------------------------
# build_event_clip
# ---------------------------------------------------------------------------


def test_build_event_clip_trims_concats_and_remuxes(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 0)
    s60 = _seg(root, 60)
    s120 = _seg(root, 120)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    lists: list[str] = []

    def fake_run(cmd: list[str], **kw: Any) -> MagicMock:
        calls.append((cmd, kw))
        if "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat":
            lists.append(Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8"))
        return _ok(cmd)

    clips_dir = root / "clips"
    result = _build(root, runner=fake_run)

    assert result == clips_dir / "7.mp4"
    assert result.read_bytes() == b"clip"
    # window [105 s, 130 s]: starts 45 s into the 60 s segment, lasts 25 s
    assert calls[0][0] == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "45.000",
        "-t",
        "25.000",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(clips_dir / "7.concat.txt"),
        "-c",
        "copy",
        "-an",
        "-f",
        "mpegts",
        str(clips_dir / "7.ts"),
    ]
    assert calls[1][0] == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clips_dir / "7.ts"),
        "-c",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(clips_dir / "7.mp4"),
    ]
    assert all(kw == {"capture_output": True, "timeout": CLIP_FFMPEG_TIMEOUT_S} for _, kw in calls)
    assert lists == [f"file '{s60}'\nfile '{s120}'\n"]
    assert sorted(p.name for p in clips_dir.iterdir()) == ["7.mp4"]


def test_build_event_clip_keeps_the_ts_when_the_remux_fails(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)
    _seg(root, 120)

    def fake_run(cmd: list[str], **_kw: Any) -> MagicMock:
        out = Path(cmd[-1])
        if out.suffix == ".mp4":
            out.write_bytes(b"partial")
            return MagicMock(returncode=1, stdout=b"", stderr=b"moov atom failure")
        return _ok(cmd)

    result = _build(root, runner=fake_run)

    assert result == root / "clips" / "7.ts"
    assert result.read_bytes() == b"clip"
    assert sorted(p.name for p in (root / "clips").iterdir()) == ["7.ts"]


def test_build_event_clip_keeps_the_ts_when_the_remux_times_out(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)

    def fake_run(cmd: list[str], **_kw: Any) -> MagicMock:
        if cmd[-1].endswith(".mp4"):
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=CLIP_FFMPEG_TIMEOUT_S)
        return _ok(cmd)

    result = _build(root, runner=fake_run, started_at=_utc(80), ended_at=_utc(85))

    assert result == root / "clips" / "7.ts"


def test_build_event_clip_returns_none_when_the_concat_fails(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)
    failing = MagicMock(return_value=MagicMock(returncode=1, stdout=b"", stderr=b"bad data"))

    assert _build(root, runner=failing, started_at=_utc(80), ended_at=_utc(85)) is None
    assert failing.call_count == 1
    assert list((root / "clips").iterdir()) == []


def test_build_event_clip_returns_none_when_ffmpeg_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)
    missing = MagicMock(side_effect=FileNotFoundError(2, "No such file", "ffmpeg"))

    assert _build(root, runner=missing, started_at=_utc(80), ended_at=_utc(85)) is None
    assert list((root / "clips").iterdir()) == []


def test_build_event_clip_without_segments_runs_nothing(tmp_path: Path) -> None:
    never = MagicMock(side_effect=AssertionError("ffmpeg must not run"))

    assert _build(tmp_path / "rec" / "yard", runner=never) is None
    never.assert_not_called()


def test_build_event_clip_caps_the_window_at_max_duration(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)
    _seg(root, 120)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: Any) -> MagicMock:
        calls.append(cmd)
        return _ok(cmd)

    _build(root, runner=fake_run, max_duration=15.0)

    concat = calls[0]
    assert concat[concat.index("-ss") + 1] == "45.000"
    assert concat[concat.index("-t") + 1] == "15.000"


def test_build_event_clip_starts_at_the_first_segment_when_the_window_is_earlier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 100)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: Any) -> MagicMock:
        calls.append(cmd)
        return _ok(cmd)

    _build(root, runner=fake_run, started_at=_utc(102), ended_at=_utc(110), pre_seconds=10.0)

    concat = calls[0]
    assert concat[concat.index("-ss") + 1] == "0.000"
    assert concat[concat.index("-t") + 1] == "20.000"  # 100 s .. 120 s


def test_concat_list_is_absolute_and_quote_escaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("rec") / "o'brien"  # relative, like the default ./recordings
    seg = _seg(root, 60, camera="o'brien")
    lists: list[str] = []

    def fake_run(cmd: list[str], **_kw: Any) -> MagicMock:
        if "-safe" in cmd:
            lists.append(Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8"))
        return _ok(cmd)

    _build(root, runner=fake_run, started_at=_utc(80), ended_at=_utc(85))

    escaped = str(tmp_path / seg).replace("'", "'\\''")
    assert lists == [f"file '{escaped}'\n"]


def test_build_event_clip_looks_up_subprocess_run_at_call_time(tmp_path: Path) -> None:
    root = tmp_path / "rec" / "yard"
    _seg(root, 60)

    with patch("subprocess.run", side_effect=_ok) as fake:
        result = _build(root, started_at=_utc(80), ended_at=_utc(85))

    assert fake.call_count == 2
    assert result == root / "clips" / "7.mp4"


def test_build_event_clip_logs_when_no_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(clips.log, "warning", lambda msg, *a, **_k: seen.append(msg % a))

    assert _build(tmp_path / "rec" / "yard", runner=_ok) is None
    assert len(seen) == 1
    assert seen[0].startswith("[clips] event 7: no recorded main segments cover ")


# ---------------------------------------------------------------------------
# Config, stream choice and stored path
# ---------------------------------------------------------------------------


def test_clips_config_defaults() -> None:
    cfg = ClipsConfig()
    assert (cfg.enabled, cfg.pre_seconds, cfg.post_seconds, cfg.max_duration) == (
        True,
        10.0,
        10.0,
        120.0,
    )


@pytest.mark.parametrize("field", ["pre_seconds", "post_seconds", "max_duration"])
def test_clips_config_rejects_non_positive_seconds(field: str) -> None:
    with pytest.raises(ValueError):
        ClipsConfig(**{field: 0})


def test_clip_stream_and_chunk() -> None:
    main_only = CameraConfig(
        name="a", main_url="rtsp://u:p@h/m", record={"main": {"chunk_seconds": 60}}
    )
    with_sub = CameraConfig(name="b", main_url="rtsp://u:p@h/m", sub_url="rtsp://u:p@h/s")
    sub_off = CameraConfig(
        name="c",
        main_url="rtsp://u:p@h/m",
        sub_url="rtsp://u:p@h/s",
        record={"sub": {"enabled": False}},
    )
    sub_mkv = CameraConfig(
        name="d",
        main_url="rtsp://u:p@h/m",
        sub_url="rtsp://u:p@h/s",
        record={"sub": {"container": "mkv"}},
    )

    assert clip_stream_and_chunk(main_only) == ("main", 60)
    assert clip_stream_and_chunk(with_sub) == ("sub", 300)
    assert clip_stream_and_chunk(sub_off) == ("main", 300)
    assert clip_stream_and_chunk(sub_mkv) == ("main", 300)


def test_clip_rel_path_is_relative_to_output_dir(tmp_path: Path) -> None:
    assert clip_rel_path("yard", tmp_path / "rec" / "yard" / "clips" / "7.mp4") == (
        "yard/clips/7.mp4"
    )
