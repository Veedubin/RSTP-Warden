"""Tests for config.py — TS container default and container validation."""

from __future__ import annotations

from pathlib import Path

from rtsp_warden.config import (
    Container,
    RecordConfig,
    StreamRecordConfig,
    load_config,
)


def test_container_literal_includes_ts() -> None:
    """Container.__args__ contains 'ts'."""
    assert "ts" in Container.__args__


def test_stream_record_config_default_container_is_ts() -> None:
    """StreamRecordConfig().container == 'ts'."""
    cfg = StreamRecordConfig()
    assert cfg.container == "ts"


def test_record_config_defaults_to_ts() -> None:
    """RecordConfig().main.container == 'ts' and RecordConfig().sub.container == 'ts'."""
    cfg = RecordConfig()
    assert cfg.main.container == "ts"
    assert cfg.sub.container == "ts"


def test_load_examples_config_uses_ts() -> None:
    """load_config('examples/config.yaml') produces cameras with container='ts'."""
    examples_dir = Path(__file__).parent.parent / "examples"
    config_path = examples_dir / "config.yaml"
    assert config_path.exists(), f"Expected config at {config_path}"

    app_cfg = load_config(str(config_path))
    assert len(app_cfg.cameras) > 0

    for cam in app_cfg.cameras:
        assert cam.record.main.container == "ts"
        assert cam.record.sub.container == "ts"


def test_mp4_and_mkv_still_accepted() -> None:
    """Explicit container: mp4 and container: mkv still validate."""
    cfg_mp4 = StreamRecordConfig(container="mp4")
    assert cfg_mp4.container == "mp4"

    cfg_mkv = StreamRecordConfig(container="mkv")
    assert cfg_mkv.container == "mkv"
