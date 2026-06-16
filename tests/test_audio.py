"""Tests for Sprint 4 Batch 2: audio opt-in per camera."""

from __future__ import annotations

from rtsp_warden.config import AppConfig, RecordConfig
from rtsp_warden.ffmpeg import build_ffmpeg_ingest_cmd, build_ffmpeg_segment_cmd


def test_audio_false_omits_audio_args() -> None:
    """build_ffmpeg_ingest_cmd with audio=False does NOT contain -c:a."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
        audio=False,
    )
    assert "-c:a" not in cmd
    assert "-an" in cmd


def test_audio_true_includes_aac() -> None:
    """build_ffmpeg_ingest_cmd with audio=True contains -c:a aac."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
        audio=True,
    )
    assert "-c:a" in cmd
    idx = cmd.index("-c:a")
    assert cmd[idx + 1] == "aac"


def test_audio_true_uses_optional_audio_map() -> None:
    """audio=True adds -map 0:a? (the ? makes it optional if no audio in source)."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
        audio=True,
    )
    assert "0:a?" in cmd
    # Find the -map that maps audio (there may be multiple -map entries)
    audio_map_idx = None
    for i, arg in enumerate(cmd):
        if arg == "-map" and i + 1 < len(cmd) and cmd[i + 1] == "0:a?":
            audio_map_idx = i
            break
    assert audio_map_idx is not None, "Expected -map 0:a? in command"


def test_audio_true_uses_128k_bitrate() -> None:
    """audio=True adds -b:a 128k."""
    cmd = build_ffmpeg_ingest_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport_in="tcp",
        record_enabled=True,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.ts",
        container="ts",
        audio=True,
    )
    assert "-b:a" in cmd
    idx = cmd.index("-b:a")
    assert cmd[idx + 1] == "128k"


def test_config_audio_default_is_false() -> None:
    """RecordConfig() defaults audio to False (backward compatible)."""
    cfg = RecordConfig()
    assert cfg.audio is False


def test_config_audio_can_be_set_true() -> None:
    """RecordConfig(audio=True) sets audio to True."""
    cfg = RecordConfig(audio=True)
    assert cfg.audio is True


def test_app_config_passes_audio_through() -> None:
    """Full AppConfig with RecordConfig(audio=True) parses and round-trips."""
    raw = {
        "cameras": [
            {
                "name": "test-cam",
                "main_url": "rtsp://example.com/main",
                "sub_url": "rtsp://example.com/sub",
                "record": {"enabled": True, "audio": True},
            }
        ],
    }
    cfg = AppConfig.model_validate(raw)
    assert cfg.cameras[0].record.audio is True

    # Round-trip: serialize back and re-validate
    dumped = cfg.model_dump()
    cfg2 = AppConfig.model_validate(dumped)
    assert cfg2.cameras[0].record.audio is True


def test_segment_cmd_audio_false_has_an() -> None:
    """build_ffmpeg_segment_cmd with audio=False includes -an."""
    cmd = build_ffmpeg_segment_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport="tcp",
        chunk_seconds=300,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.mkv",
        container="mkv",
        audio=False,
    )
    assert "-an" in cmd
    assert "-c:a" not in cmd


def test_segment_cmd_audio_true_includes_aac() -> None:
    """build_ffmpeg_segment_cmd with audio=True includes -c:a aac and -map 0:a?."""
    cmd = build_ffmpeg_segment_cmd(
        ffmpeg_path="ffmpeg",
        rtsp_url="rtsp://example.com/stream",
        rtsp_transport="tcp",
        chunk_seconds=300,
        out_pattern="/tmp/out_%Y%m%d_%H%M%S.mkv",
        container="mkv",
        audio=True,
    )
    assert "-c:a" in cmd
    assert "-an" not in cmd
    assert "0:a?" in cmd
