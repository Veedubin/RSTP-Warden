"""Tests for PTZ preset management (Feature 4: Multi-camera PTZ presets)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from rtsp_warden.config import AppConfig, CameraConfig, PTZPresetConfig
from rtsp_warden.onvif.presets import PTZPreset, PTZPresetError, PTZPresetStore
from rtsp_warden.onvif.ptz import OnvifPTZ

# ---------------------------------------------------------------------------
# PTZPreset dataclass tests
# ---------------------------------------------------------------------------


class TestPTZPreset:
    """Tests for the PTZPreset frozen dataclass."""

    def test_to_absolute_move_args(self) -> None:
        """to_absolute_move_args returns a dict with pan/tilt/zoom keys."""
        preset = PTZPreset(name="front_gate", pan=0.25, tilt=0.30, zoom=0.0)
        result = preset.to_absolute_move_args()
        assert result == {"pan": 0.25, "tilt": 0.30, "zoom": 0.0}

    def test_to_absolute_move_args_negative_values(self) -> None:
        """Negative pan/tilt values are passed through correctly."""
        preset = PTZPreset(name="left_corner", pan=-0.75, tilt=-0.1, zoom=0.5)
        result = preset.to_absolute_move_args()
        assert result == {"pan": -0.75, "tilt": -0.1, "zoom": 0.5}

    def test_frozen_immutability(self) -> None:
        """PTZPreset is frozen; assignment raises FrozenInstanceError."""
        preset = PTZPreset(name="gate", pan=0.0, tilt=0.0, zoom=0.0)
        with pytest.raises(AttributeError):
            preset.name = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two presets with same values are equal."""
        a = PTZPreset(name="gate", pan=0.1, tilt=0.2, zoom=0.3)
        b = PTZPreset(name="gate", pan=0.1, tilt=0.2, zoom=0.3)
        assert a == b

    def test_inequality(self) -> None:
        """Presets with different values are not equal."""
        a = PTZPreset(name="gate", pan=0.1, tilt=0.2, zoom=0.3)
        b = PTZPreset(name="driveway", pan=0.1, tilt=0.2, zoom=0.3)
        assert a != b


# ---------------------------------------------------------------------------
# PTZPresetStore tests
# ---------------------------------------------------------------------------


def _make_config(camera_names: list[str] | None = None) -> AppConfig:
    """Build a minimal AppConfig with named cameras (no presets)."""
    names = camera_names or ["front_door"]
    cameras = [
        CameraConfig(
            name=n,
            main_url=f"rtsp://admin:pass@192.168.1.{i}:554/main",
            sub_url=f"rtsp://admin:pass@192.168.1.{i}:554/sub",
        )
        for i, n in enumerate(names, start=100)
    ]
    return AppConfig(cameras=cameras)


def _make_config_with_presets(
    camera_name: str = "front_door",
    presets: list[tuple[str, float, float, float]] | None = None,
) -> AppConfig:
    """Build an AppConfig with a camera that has presets."""
    preset_cfgs = [PTZPresetConfig(name=n, pan=p, tilt=t, zoom=z) for n, p, t, z in (presets or [])]
    cam = CameraConfig(
        name=camera_name,
        main_url="rtsp://admin:pass@192.168.1.100:554/main",
        sub_url="rtsp://admin:pass@192.168.1.100:554/sub",
        presets=preset_cfgs,
    )
    return AppConfig(cameras=[cam])


class TestPTZPresetStoreListPresets:
    """Tests for PTZPresetStore.list_presets()."""

    def test_list_presets_returns_presets_for_known_camera(self) -> None:
        """list_presets returns PTZPreset objects for a camera with presets."""
        cfg = _make_config_with_presets(
            presets=[("gate", 0.25, 0.30, 0.0), ("driveway", 0.75, 0.20, 0.5)]
        )
        store = PTZPresetStore(cfg)
        result = store.list_presets("front_door")
        assert len(result) == 2
        assert result[0] == PTZPreset(name="gate", pan=0.25, tilt=0.30, zoom=0.0)
        assert result[1] == PTZPreset(name="driveway", pan=0.75, tilt=0.20, zoom=0.5)

    def test_list_presets_returns_empty_for_unknown_camera(self) -> None:
        """list_presets returns [] for a camera not in config (no exception)."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        result = store.list_presets("nonexistent")
        assert result == []

    def test_list_presets_returns_empty_for_camera_with_no_presets(self) -> None:
        """list_presets returns [] for a camera that has no presets."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        result = store.list_presets("front_door")
        assert result == []


class TestPTZPresetStoreGetPreset:
    """Tests for PTZPresetStore.get_preset()."""

    def test_get_preset_returns_correct_preset(self) -> None:
        """get_preset returns the matching PTZPreset by name."""
        cfg = _make_config_with_presets(
            presets=[("gate", 0.25, 0.30, 0.0), ("driveway", 0.75, 0.20, 0.5)]
        )
        store = PTZPresetStore(cfg)
        result = store.get_preset("front_door", "driveway")
        assert result is not None
        assert result == PTZPreset(name="driveway", pan=0.75, tilt=0.20, zoom=0.5)

    def test_get_preset_returns_none_for_unknown_preset(self) -> None:
        """get_preset returns None when the preset name doesn't exist."""
        cfg = _make_config_with_presets(presets=[("gate", 0.25, 0.30, 0.0)])
        store = PTZPresetStore(cfg)
        result = store.get_preset("front_door", "nonexistent")
        assert result is None

    def test_get_preset_returns_none_for_unknown_camera(self) -> None:
        """get_preset returns None when the camera doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        result = store.get_preset("nonexistent", "gate")
        assert result is None


class TestPTZPresetStoreGotoPreset:
    """Tests for PTZPresetStore.goto_preset()."""

    @pytest.mark.asyncio
    async def test_goto_preset_calls_absolute_move(self) -> None:
        """goto_preset calls OnvifPTZ.absolute_move with the preset's values."""
        cfg = _make_config_with_presets(presets=[("gate", 0.25, 0.30, 0.0)])
        store = PTZPresetStore(cfg)

        mock_ptz = AsyncMock(spec=OnvifPTZ)
        await store.goto_preset("front_door", "gate", mock_ptz)

        mock_ptz.absolute_move.assert_awaited_once_with(pan=0.25, tilt=0.30, zoom=0.0)

    @pytest.mark.asyncio
    async def test_goto_preset_raises_for_unknown_preset(self) -> None:
        """goto_preset raises PTZPresetError if the preset doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        mock_ptz = AsyncMock(spec=OnvifPTZ)

        with pytest.raises(PTZPresetError, match="not found"):
            await store.goto_preset("front_door", "nonexistent", mock_ptz)

    @pytest.mark.asyncio
    async def test_goto_preset_raises_for_unknown_camera(self) -> None:
        """goto_preset raises PTZPresetError if the camera doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        mock_ptz = AsyncMock(spec=OnvifPTZ)

        with pytest.raises(PTZPresetError, match="not found"):
            await store.goto_preset("nonexistent", "gate", mock_ptz)


class TestPTZPresetStoreSavePreset:
    """Tests for PTZPresetStore.save_preset()."""

    @pytest.mark.asyncio
    async def test_save_preset_adds_to_in_memory_store(self) -> None:
        """save_preset adds a new preset to the in-memory config."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)
        result = await store.save_preset("front_door", "gate", 0.25, 0.30, 0.0)

        assert result == PTZPreset(name="gate", pan=0.25, tilt=0.30, zoom=0.0)
        # Verify it's actually in the config
        presets = store.list_presets("front_door")
        assert len(presets) == 1
        assert presets[0].name == "gate"

    @pytest.mark.asyncio
    async def test_save_preset_overwrites_existing_name(self) -> None:
        """save_preset with same name overwrites the existing preset (idempotent)."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        await store.save_preset("front_door", "gate", 0.25, 0.30, 0.0)
        await store.save_preset("front_door", "gate", 0.50, 0.60, 0.7)

        presets = store.list_presets("front_door")
        assert len(presets) == 1
        assert presets[0] == PTZPreset(name="gate", pan=0.50, tilt=0.60, zoom=0.7)

    @pytest.mark.asyncio
    async def test_save_preset_raises_for_unknown_camera(self) -> None:
        """save_preset raises PTZPresetError if the camera doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        with pytest.raises(PTZPresetError, match="not found"):
            await store.save_preset("nonexistent", "gate", 0.25, 0.30, 0.0)

    @pytest.mark.asyncio
    async def test_save_preset_raises_for_empty_name(self) -> None:
        """save_preset raises PTZPresetError if the name is empty."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        with pytest.raises(PTZPresetError, match="non-empty"):
            await store.save_preset("front_door", "  ", 0.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_save_preset_raises_for_long_name(self) -> None:
        """save_preset raises PTZPresetError if the name exceeds 64 chars."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        with pytest.raises(PTZPresetError, match="64 characters"):
            await store.save_preset("front_door", "x" * 65, 0.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_save_preset_strips_whitespace_from_name(self) -> None:
        """save_preset strips leading/trailing whitespace from the name."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        result = await store.save_preset("front_door", "  gate  ", 0.25, 0.30, 0.0)
        assert result.name == "gate"


class TestPTZPresetStoreDeletePreset:
    """Tests for PTZPresetStore.delete_preset()."""

    @pytest.mark.asyncio
    async def test_delete_preset_removes_from_store(self) -> None:
        """delete_preset removes the preset and returns True."""
        cfg = _make_config_with_presets(
            presets=[("gate", 0.25, 0.30, 0.0), ("driveway", 0.75, 0.20, 0.5)]
        )
        store = PTZPresetStore(cfg)

        result = await store.delete_preset("front_door", "gate")
        assert result is True
        presets = store.list_presets("front_door")
        assert len(presets) == 1
        assert presets[0].name == "driveway"

    @pytest.mark.asyncio
    async def test_delete_preset_returns_false_for_missing(self) -> None:
        """delete_preset returns False if the preset doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        result = await store.delete_preset("front_door", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_preset_returns_false_for_unknown_camera(self) -> None:
        """delete_preset returns False if the camera doesn't exist."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)

        result = await store.delete_preset("nonexistent", "gate")
        assert result is False


class TestPTZPresetConfigValidation:
    """Tests for PTZPresetConfig pydantic model validation."""

    def test_valid_preset_config(self) -> None:
        """PTZPresetConfig accepts valid values."""
        cfg = PTZPresetConfig(name="gate", pan=0.25, tilt=0.30, zoom=0.0)
        assert cfg.name == "gate"
        assert cfg.pan == 0.25

    def test_empty_name_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for empty name."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            PTZPresetConfig(name="", pan=0.0, tilt=0.0, zoom=0.0)

    def test_whitespace_name_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for whitespace-only name."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            PTZPresetConfig(name="   ", pan=0.0, tilt=0.0, zoom=0.0)

    def test_long_name_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for names longer than 64 chars."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="64 characters"):
            PTZPresetConfig(name="x" * 65, pan=0.0, tilt=0.0, zoom=0.0)

    def test_pan_out_of_range_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for pan > 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="pan/tilt"):
            PTZPresetConfig(name="gate", pan=1.5, tilt=0.0, zoom=0.0)

    def test_tilt_out_of_range_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for tilt < -1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="pan/tilt"):
            PTZPresetConfig(name="gate", pan=0.0, tilt=-1.5, zoom=0.0)

    def test_zoom_out_of_range_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for zoom > 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="zoom"):
            PTZPresetConfig(name="gate", pan=0.0, tilt=0.0, zoom=1.5)

    def test_zoom_negative_raises(self) -> None:
        """PTZPresetConfig raises ValidationError for zoom < 0.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="zoom"):
            PTZPresetConfig(name="gate", pan=0.0, tilt=0.0, zoom=-0.5)

    def test_boundary_values_accepted(self) -> None:
        """PTZPresetConfig accepts boundary values (pan=-1/1, zoom=0/1)."""
        cfg = PTZPresetConfig(name="bounds", pan=-1.0, tilt=1.0, zoom=0.0)
        assert cfg.pan == -1.0
        cfg2 = PTZPresetConfig(name="bounds2", pan=1.0, tilt=-1.0, zoom=1.0)
        assert cfg2.zoom == 1.0


class TestConfigPersistence:
    """Tests for config persistence roundtrip (yaml -> AppConfig -> modify -> yaml)."""

    def test_presets_survive_roundtrip(self, tmp_path: Path) -> None:
        """Presets survive a YAML roundtrip: write config, reload, verify presets."""
        presets = [
            PTZPresetConfig(name="gate", pan=0.25, tilt=0.30, zoom=0.0),
            PTZPresetConfig(name="driveway", pan=0.75, tilt=0.20, zoom=0.5),
        ]
        cam = CameraConfig(
            name="front_door",
            main_url="rtsp://admin:pass@192.168.1.100:554/main",
            sub_url="rtsp://admin:pass@192.168.1.100:554/sub",
            presets=presets,
        )
        cfg = AppConfig(cameras=[cam])

        # Serialize to YAML
        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_str, encoding="utf-8")

        # Reload from YAML
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        cfg2 = AppConfig.model_validate(raw)

        # Verify presets survived
        assert len(cfg2.cameras[0].presets) == 2
        assert cfg2.cameras[0].presets[0].name == "gate"
        assert cfg2.cameras[0].presets[0].pan == 0.25
        assert cfg2.cameras[0].presets[1].name == "driveway"
        assert cfg2.cameras[0].presets[1].zoom == 0.5

    @pytest.mark.asyncio
    async def test_save_persist_persists_to_file(self, tmp_path: Path) -> None:
        """save_preset with config_path writes back to the YAML file."""
        cfg = _make_config()
        config_file = tmp_path / "config.yaml"

        # Write initial config
        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        config_file.write_text(yaml_str, encoding="utf-8")

        store = PTZPresetStore(cfg, config_path=config_file)
        await store.save_preset("front_door", "gate", 0.25, 0.30, 0.0)

        # Verify the file was written
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        cfg2 = AppConfig.model_validate(raw)
        assert len(cfg2.cameras[0].presets) == 1
        assert cfg2.cameras[0].presets[0].name == "gate"

    @pytest.mark.asyncio
    async def test_delete_persist_persists_to_file(self, tmp_path: Path) -> None:
        """delete_preset with config_path writes back to the YAML file."""
        cfg = _make_config_with_presets(
            presets=[("gate", 0.25, 0.30, 0.0), ("driveway", 0.75, 0.20, 0.5)]
        )
        config_file = tmp_path / "config.yaml"

        # Write initial config
        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        config_file.write_text(yaml_str, encoding="utf-8")

        store = PTZPresetStore(cfg, config_path=config_file)
        result = await store.delete_preset("front_door", "gate")
        assert result is True

        # Verify the file was updated
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        cfg2 = AppConfig.model_validate(raw)
        assert len(cfg2.cameras[0].presets) == 1
        assert cfg2.cameras[0].presets[0].name == "driveway"

    @pytest.mark.asyncio
    async def test_no_persist_without_config_path(self) -> None:
        """save_preset without config_path only modifies in-memory config."""
        cfg = _make_config()
        store = PTZPresetStore(cfg)  # No config_path

        result = await store.save_preset("front_door", "gate", 0.25, 0.30, 0.0)
        assert result.name == "gate"
        assert len(store.list_presets("front_door")) == 1
        # No file I/O attempted - no exception means success

    @pytest.mark.asyncio
    async def test_save_persist_without_config_path_no_error(self) -> None:
        """delete_preset without config_path only modifies in-memory config."""
        cfg = _make_config_with_presets(presets=[("gate", 0.25, 0.30, 0.0)])
        store = PTZPresetStore(cfg)  # No config_path

        result = await store.delete_preset("front_door", "gate")
        assert result is True
        assert len(store.list_presets("front_door")) == 0


class TestCameraConfigPresets:
    """Tests for CameraConfig.presets field."""

    def test_camera_config_has_empty_presets_by_default(self) -> None:
        """CameraConfig.presets defaults to an empty list."""
        cam = CameraConfig(
            name="test_cam",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
        )
        assert cam.presets == []

    def test_camera_config_presets_populated(self) -> None:
        """CameraConfig.presets can be populated with PTZPresetConfig items."""
        cam = CameraConfig(
            name="test_cam",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
            presets=[
                PTZPresetConfig(name="gate", pan=0.25, tilt=0.30, zoom=0.0),
            ],
        )
        assert len(cam.presets) == 1
        assert cam.presets[0].name == "gate"

    def test_yaml_roundtrip_with_presets(self) -> None:
        """AppConfig with presets survives YAML serialization roundtrip."""
        cfg = AppConfig(
            cameras=[
                CameraConfig(
                    name="cam1",
                    main_url="rtsp://admin:pass@10.0.0.1:554/main",
                    sub_url="rtsp://admin:pass@10.0.0.1:554/sub",
                    presets=[
                        PTZPresetConfig(name="home", pan=0.0, tilt=0.0, zoom=0.0),
                        PTZPresetConfig(name="gate", pan=0.5, tilt=0.2, zoom=0.8),
                    ],
                )
            ]
        )
        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        assert len(cfg2.cameras[0].presets) == 2
        assert cfg2.cameras[0].presets[0].name == "home"
        assert cfg2.cameras[0].presets[1].pan == 0.5
