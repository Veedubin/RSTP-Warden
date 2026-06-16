"""Tests for Sprint 6 Config Foundation: GridZoneConfig, CameraConfig extensions,
AppConfig.retention, DetectorSpec.enabled, and _locked_write_yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rtsp_warden.config import (
    AppConfig,
    CameraConfig,
    DetectorSpec,
    GridZoneConfig,
    RetentionConfig,
)
from rtsp_warden.web.config_lock import _locked_write_yaml

# ---------------------------------------------------------------------------
# GridZoneConfig tests
# ---------------------------------------------------------------------------


class TestGridZoneConfig:
    """Tests for GridZoneConfig pydantic model validation."""

    def test_valid_config(self) -> None:
        """GridZoneConfig accepts valid configuration."""
        zone = GridZoneConfig(
            name="exclude road",
            grid_cols=16,
            grid_rows=16,
            blocked_cells={(0, 0), (1, 0)},
            frame_width=1920,
            frame_height=1080,
        )
        assert zone.name == "exclude road"
        assert zone.grid_cols == 16
        assert zone.grid_rows == 16
        assert zone.blocked_cells == {(0, 0), (1, 0)}
        assert zone.frame_width == 1920
        assert zone.frame_height == 1080
        assert zone.enabled is True

    def test_defaults(self) -> None:
        """GridZoneConfig has correct defaults for grid_cols, grid_rows, enabled."""
        zone = GridZoneConfig(
            name="test_zone",
            frame_width=1280,
            frame_height=720,
        )
        assert zone.grid_cols == 16
        assert zone.grid_rows == 16
        assert zone.blocked_cells == set()
        assert zone.enabled is True

    def test_blocked_cells_out_of_bounds_raises(self) -> None:
        """GridZoneConfig raises ValidationError when blocked_cells are out of bounds."""
        with pytest.raises(ValidationError, match="out of bounds"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(8, 0)},  # col 8 is out of bounds for 8 cols
                frame_width=1920,
                frame_height=1080,
            )

    def test_blocked_cells_row_out_of_bounds_raises(self) -> None:
        """GridZoneConfig raises ValidationError when row is out of bounds."""
        with pytest.raises(ValidationError, match="out of bounds"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(0, 8)},  # row 8 is out of bounds for 8 rows
                frame_width=1920,
                frame_height=1080,
            )

    def test_blocked_cells_negative_col_raises(self) -> None:
        """GridZoneConfig raises ValidationError for negative column."""
        with pytest.raises(ValidationError, match="out of bounds"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=8,
                grid_rows=8,
                blocked_cells={(-1, 0)},
                frame_width=1920,
                frame_height=1080,
            )

    def test_grid_cols_too_small_raises(self) -> None:
        """GridZoneConfig raises ValidationError when grid_cols < 2."""
        with pytest.raises(ValidationError, match=">= 2"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=1,
                grid_rows=8,
                frame_width=1920,
                frame_height=1080,
            )

    def test_grid_rows_too_small_raises(self) -> None:
        """GridZoneConfig raises ValidationError when grid_rows < 2."""
        with pytest.raises(ValidationError, match=">= 2"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=8,
                grid_rows=1,
                frame_width=1920,
                frame_height=1080,
            )

    def test_grid_cols_too_large_raises(self) -> None:
        """GridZoneConfig raises ValidationError when grid_cols > 64."""
        with pytest.raises(ValidationError, match="<= 64"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=65,
                grid_rows=8,
                frame_width=1920,
                frame_height=1080,
            )

    def test_grid_rows_too_large_raises(self) -> None:
        """GridZoneConfig raises ValidationError when grid_rows > 64."""
        with pytest.raises(ValidationError, match="<= 64"):
            GridZoneConfig(
                name="bad_zone",
                grid_cols=8,
                grid_rows=65,
                frame_width=1920,
                frame_height=1080,
            )

    def test_blocked_cells_at_boundary_accepted(self) -> None:
        """GridZoneConfig accepts cells at the maximum boundary (grid_cols-1, grid_rows-1)."""
        zone = GridZoneConfig(
            name="edge_zone",
            grid_cols=16,
            grid_rows=16,
            blocked_cells={(15, 15)},  # max valid cell
            frame_width=1920,
            frame_height=1080,
        )
        assert (15, 15) in zone.blocked_cells

    def test_disabled_zone(self) -> None:
        """GridZoneConfig with enabled=False is accepted."""
        zone = GridZoneConfig(
            name="disabled_zone",
            grid_cols=8,
            grid_rows=8,
            frame_width=1920,
            frame_height=1080,
            enabled=False,
        )
        assert zone.enabled is False


# ---------------------------------------------------------------------------
# CameraConfig extension tests
# ---------------------------------------------------------------------------


class TestCameraConfigSprint6:
    """Tests for CameraConfig Sprint 6 fields: retention, zones, sensitivity, detect_classes."""

    def test_sensitivity_default_is_50(self) -> None:
        """CameraConfig.sensitivity defaults to 50.0."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
        )
        assert cam.sensitivity == 50.0

    def test_sensitivity_valid_range(self) -> None:
        """CameraConfig accepts sensitivity values from 0.0 to 100.0."""
        cam_min = CameraConfig(
            name="cam_min",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
            sensitivity=0.0,
        )
        assert cam_min.sensitivity == 0.0

        cam_max = CameraConfig(
            name="cam_max",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
            sensitivity=100.0,
        )
        assert cam_max.sensitivity == 100.0

    def test_sensitivity_below_range_raises(self) -> None:
        """CameraConfig raises ValidationError for sensitivity < 0."""
        with pytest.raises(ValidationError, match="sensitivity"):
            CameraConfig(
                name="bad",
                main_url="rtsp://a@1.2.3.4:554/main",
                sub_url="rtsp://a@1.2.3.4:554/sub",
                sensitivity=-1.0,
            )

    def test_sensitivity_above_range_raises(self) -> None:
        """CameraConfig raises ValidationError for sensitivity > 100."""
        with pytest.raises(ValidationError, match="sensitivity"):
            CameraConfig(
                name="bad",
                main_url="rtsp://a@1.2.3.4:554/main",
                sub_url="rtsp://a@1.2.3.4:554/sub",
                sensitivity=101.0,
            )

    def test_zones_default_is_empty_list(self) -> None:
        """CameraConfig.zones defaults to an empty list."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
        )
        assert cam.zones == []

    def test_zones_with_grid_zone(self) -> None:
        """CameraConfig accepts zones with GridZoneConfig items."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
            zones=[
                GridZoneConfig(
                    name="exclude_road",
                    grid_cols=8,
                    grid_rows=8,
                    blocked_cells={(0, 0), (1, 0)},
                    frame_width=1920,
                    frame_height=1080,
                ),
            ],
        )
        assert len(cam.zones) == 1
        assert cam.zones[0].name == "exclude_road"
        assert cam.zones[0].blocked_cells == {(0, 0), (1, 0)}

    def test_retention_default_is_none(self) -> None:
        """CameraConfig.retention defaults to None."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
        )
        assert cam.retention is None

    def test_retention_with_values(self) -> None:
        """CameraConfig.retention can be set with RetentionConfig values."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
            retention=RetentionConfig(max_days=90, max_gb=100.0),
        )
        assert cam.retention is not None
        assert cam.retention.max_days == 90
        assert cam.retention.max_gb == 100.0

    def test_detect_classes_default_is_none(self) -> None:
        """CameraConfig.detect_classes defaults to None."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
        )
        assert cam.detect_classes is None

    def test_detect_classes_with_values(self) -> None:
        """CameraConfig.detect_classes accepts a list of strings."""
        cam = CameraConfig(
            name="cam1",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
            detect_classes=["person", "dog", "cat"],
        )
        assert cam.detect_classes == ["person", "dog", "cat"]


# ---------------------------------------------------------------------------
# AppConfig.retention tests
# ---------------------------------------------------------------------------


class TestAppConfigRetention:
    """Tests for AppConfig.retention global fallback field."""

    def test_app_config_has_retention_default(self) -> None:
        """AppConfig.retention defaults to RetentionConfig()."""
        cfg = AppConfig(
            cameras=[
                CameraConfig(
                    name="cam1",
                    main_url="rtsp://a@1.2.3.4:554/main",
                    sub_url="rtsp://a@1.2.3.4:554/sub",
                )
            ]
        )
        assert cfg.retention is not None
        assert cfg.retention.max_days is None
        assert cfg.retention.max_gb is None

    def test_app_config_retention_custom(self) -> None:
        """AppConfig.retention can be set with custom values."""
        cfg = AppConfig(
            cameras=[
                CameraConfig(
                    name="cam1",
                    main_url="rtsp://a@1.2.3.4:554/main",
                    sub_url="rtsp://a@1.2.3.4:554/sub",
                )
            ],
            retention=RetentionConfig(max_days=30, max_gb=50.0),
        )
        assert cfg.retention.max_days == 30
        assert cfg.retention.max_gb == 50.0


# ---------------------------------------------------------------------------
# DetectorSpec.enabled tests
# ---------------------------------------------------------------------------


class TestDetectorSpecEnabled:
    """Tests for DetectorSpec.enabled field (defaults to True)."""

    def test_enabled_default_is_true(self) -> None:
        """DetectorSpec.enabled defaults to True."""
        spec = DetectorSpec(type="motion")
        assert spec.enabled is True

    def test_enabled_can_be_false(self) -> None:
        """DetectorSpec.enabled can be set to False."""
        spec = DetectorSpec(type="motion", enabled=False)
        assert spec.enabled is False


# ---------------------------------------------------------------------------
# _locked_write_yaml tests
# ---------------------------------------------------------------------------


class TestLockedWriteYaml:
    """Tests for atomic config.yaml writing with file locking."""

    def test_writes_yaml_atomically(self, tmp_path: Path) -> None:
        """_locked_write_yaml writes data to a YAML file atomically."""
        config_path = tmp_path / "config.yaml"
        data = {"cameras": [{"name": "test", "main_url": "rtsp://a", "sub_url": "rtsp://b"}]}

        _locked_write_yaml(config_path, data)

        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """_locked_write_yaml creates parent directories if they don't exist."""
        config_path = tmp_path / "nested" / "dir" / "config.yaml"
        data = {"key": "value"}

        _locked_write_yaml(config_path, data)

        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """_locked_write_yaml atomically replaces an existing file."""
        config_path = tmp_path / "config.yaml"
        old_data = {"cameras": []}
        _locked_write_yaml(config_path, old_data)
        assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == old_data

        new_data = {"cameras": [{"name": "updated"}]}
        _locked_write_yaml(config_path, new_data)
        assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == new_data

    def test_preserves_complex_types(self, tmp_path: Path) -> None:
        """_locked_write_yaml handles nested dicts, lists, and numbers."""
        config_path = tmp_path / "config.yaml"
        data = {
            "nested": {"key": "value", "number": 42},
            "items": [1, 2, 3],
            "flag": True,
        }
        _locked_write_yaml(config_path, data)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert loaded == data


# ---------------------------------------------------------------------------
# YAML roundtrip with AppConfig (Sprint 6 fields)
# ---------------------------------------------------------------------------


class TestConfigRoundtripSprint6:
    """Tests for YAML roundtrip with Sprint 6 config fields."""

    def test_yaml_roundtrip_with_zones(self) -> None:
        """CameraConfig with zones survives YAML serialization roundtrip."""
        # Note: set[tuple[int,int]] serializes as list[list[int]] in JSON/YAML mode
        cam = CameraConfig(
            name="driveway",
            main_url="rtsp://admin:pass@192.168.1.51:554/main",
            sub_url="rtsp://admin:pass@192.168.1.51:554/sub",
            zones=[
                GridZoneConfig(
                    name="driveway_only",
                    grid_cols=8,
                    grid_rows=8,
                    blocked_cells={(0, 0), (7, 7)},
                    frame_width=1920,
                    frame_height=1080,
                ),
            ],
        )
        cfg = AppConfig(cameras=[cam])

        # Serialize -> YAML -> deserialize
        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        assert len(cfg2.cameras[0].zones) == 1
        zone = cfg2.cameras[0].zones[0]
        assert zone.name == "driveway_only"
        assert zone.grid_cols == 8
        assert zone.grid_rows == 8
        assert zone.blocked_cells == {(0, 0), (7, 7)}
        assert zone.frame_width == 1920
        assert zone.frame_height == 1080

    def test_yaml_roundtrip_with_sensitivity(self) -> None:
        """CameraConfig with sensitivity survives YAML roundtrip."""
        cam = CameraConfig(
            name="backyard",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
            sensitivity=75.0,
        )
        cfg = AppConfig(cameras=[cam])

        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        assert cfg2.cameras[0].sensitivity == 75.0

    def test_yaml_roundtrip_with_detect_classes(self) -> None:
        """CameraConfig with detect_classes survives YAML roundtrip."""
        cam = CameraConfig(
            name="front_door",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
            detect_classes=["person", "dog", "cat"],
        )
        cfg = AppConfig(cameras=[cam])

        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        assert cfg2.cameras[0].detect_classes == ["person", "dog", "cat"]

    def test_yaml_roundtrip_with_retention(self) -> None:
        """CameraConfig with retention and AppConfig.retention survive YAML roundtrip."""
        cam = CameraConfig(
            name="front_door",
            main_url="rtsp://a:b@1.2.3.4:554/main",
            sub_url="rtsp://a:b@1.2.3.4:554/sub",
            retention=RetentionConfig(max_days=90, max_gb=100.0),
        )
        cfg = AppConfig(
            cameras=[cam],
            retention=RetentionConfig(max_days=30, max_gb=50.0),
        )

        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        # Per-camera retention
        assert cfg2.cameras[0].retention is not None
        assert cfg2.cameras[0].retention.max_days == 90
        assert cfg2.cameras[0].retention.max_gb == 100.0
        # Global retention
        assert cfg2.retention.max_days == 30
        assert cfg2.retention.max_gb == 50.0

    def test_yaml_roundtrip_defaults_preserved(self) -> None:
        """Default values for Sprint 6 fields survive YAML roundtrip."""
        cam = CameraConfig(
            name="minimal",
            main_url="rtsp://a@1.2.3.4:554/main",
            sub_url="rtsp://a@1.2.3.4:554/sub",
        )
        cfg = AppConfig(cameras=[cam])

        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        assert cfg2.cameras[0].sensitivity == 50.0
        assert cfg2.cameras[0].detect_classes is None
        assert cfg2.cameras[0].retention is None
        assert cfg2.cameras[0].zones == []

    def test_yaml_roundtrip_with_all_sprint6_fields(self) -> None:
        """All Sprint 6 fields together survive YAML roundtrip."""
        cam = CameraConfig(
            name="full_cam",
            main_url="rtsp://admin:pass@192.168.1.50:554/main",
            sub_url="rtsp://admin:pass@192.168.1.50:554/sub",
            sensitivity=75.0,
            detect_classes=["person", "car"],
            retention=RetentionConfig(max_days=7),
            zones=[
                GridZoneConfig(
                    name="exclude_street",
                    grid_cols=16,
                    grid_rows=16,
                    blocked_cells={(0, 15), (1, 15)},
                    frame_width=1920,
                    frame_height=1080,
                    enabled=True,
                ),
            ],
        )
        cfg = AppConfig(
            cameras=[cam],
            retention=RetentionConfig(max_days=30),
        )

        data = cfg.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        raw = yaml.safe_load(yaml_str)
        cfg2 = AppConfig.model_validate(raw)

        c = cfg2.cameras[0]
        assert c.sensitivity == 75.0
        assert c.detect_classes == ["person", "car"]
        assert c.retention is not None
        assert c.retention.max_days == 7
        assert len(c.zones) == 1
        assert c.zones[0].name == "exclude_street"
        assert c.zones[0].blocked_cells == {(0, 15), (1, 15)}
        assert cfg2.retention.max_days == 30

    def test_locked_write_yaml_roundtrip_with_appconfig(self, tmp_path: Path) -> None:
        """_locked_write_yaml roundtrip with AppConfig including Sprint 6 fields."""
        cam = CameraConfig(
            name="test_cam",
            main_url="rtsp://admin:pass@10.0.0.1:554/main",
            sub_url="rtsp://admin:pass@10.0.0.1:554/sub",
            sensitivity=65.0,
            detect_classes=["person"],
            zones=[
                GridZoneConfig(
                    name="zone1",
                    grid_cols=8,
                    grid_rows=8,
                    blocked_cells={(0, 0)},
                    frame_width=1280,
                    frame_height=720,
                ),
            ],
        )
        cfg = AppConfig(cameras=[cam])

        config_path = tmp_path / "config.yaml"
        data = cfg.model_dump(mode="json")
        _locked_write_yaml(config_path, data)

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg2 = AppConfig.model_validate(raw)

        assert cfg2.cameras[0].sensitivity == 65.0
        assert cfg2.cameras[0].detect_classes == ["person"]
        assert len(cfg2.cameras[0].zones) == 1
        assert cfg2.cameras[0].zones[0].blocked_cells == {(0, 0)}
