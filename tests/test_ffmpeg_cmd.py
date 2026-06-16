"""Tests for ffmpeg.py — FFmpeg command builder and ManagedProcess."""

from __future__ import annotations

import pytest

from rtsp_warden.ffmpeg import ManagedProcess, build_ffmpeg_ingest_cmd


def test_ts_container_adds_mpegts_segment_format() -> None:
    """build_ffmpeg_ingest_cmd with container='ts' adds -segment_format mpegts."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
    )
    assert "-segment_format" in cmd
    idx = cmd.index("-segment_format")
    assert cmd[idx + 1] == "mpegts"


def test_mp4_container_adds_mp4_segment_format() -> None:
    """container='mp4' produces -segment_format mp4."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.mp4",
        container="mp4",
    )
    assert "-segment_format" in cmd
    idx = cmd.index("-segment_format")
    assert cmd[idx + 1] == "mp4"


def test_mkv_container_no_explicit_segment_format() -> None:
    """container='mkv' does NOT have -segment_format."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.mkv",
        container="mkv",
    )
    assert "-segment_format" not in cmd


def test_frame_tap_enabled_adds_pipe3_output() -> None:
    """frame_tap_enabled=True adds pipe:3 and scale=320:-1."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
        frame_tap_enabled=True,
        frame_tap_fps=5,
        frame_tap_scale_width=320,
    )
    assert "pipe:3" in cmd
    assert "scale=320:-1" in cmd


def test_frame_tap_disabled_no_pipe3() -> None:
    """frame_tap_enabled=False (default) does NOT add pipe:3."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
    )
    assert "pipe:3" not in cmd


def test_frame_tap_validation_negative_fps() -> None:
    """frame_tap_enabled=True, frame_tap_fps=0 raises ValueError."""
    with pytest.raises(ValueError, match="frame_tap_fps must be > 0"):
        build_ffmpeg_ingest_cmd(
            ffmpeg_path="ffmpeg",
            rtsp_url="rtsp://example.com/stream",
            rtsp_transport_in="tcp",
            record_enabled=True,
            out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
            container="ts",
            frame_tap_enabled=True,
            frame_tap_fps=0,
        )


def test_managed_process_has_pass_fds() -> None:
    """ManagedProcess(pass_fds=(7,)) stores the tuple."""
    mp = ManagedProcess(name="test", args=["echo", "hi"], pass_fds=(7,))
    assert mp.pass_fds == (7,)
