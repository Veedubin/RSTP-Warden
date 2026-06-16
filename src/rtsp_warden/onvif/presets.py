"""PTZ preset management for ONVIF cameras.

Provides PTZPreset (an immutable data class for a named PTZ position),
PTZPresetStore (in-memory manager backed by AppConfig with optional YAML
persistence), and PTZPresetError for domain-specific failures.

Presets are stored as a list of PTZPresetConfig on each CameraConfig in
config.yaml.  The store reads/writes the live AppConfig object; when a
config_path is provided, mutations are persisted back to the YAML file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..config import AppConfig, CameraConfig, PTZPresetConfig
from .ptz import OnvifPTZ

if TYPE_CHECKING:
    pass  # avoid circular imports at runtime

log = logging.getLogger(__name__)


class PTZPresetError(Exception):
    """Raised when a PTZ preset operation fails."""


@dataclass(frozen=True)
class PTZPreset:
    """An immutable PTZ preset position for a camera.

    Attributes:
        name: Human-readable preset label (must be non-empty).
        pan: Pan position, typically -1.0 to 1.0.
        tilt: Tilt position, typically -1.0 to 1.0.
        zoom: Zoom level, typically 0.0 to 1.0.
    """

    name: str
    pan: float
    tilt: float
    zoom: float

    def to_absolute_move_args(self) -> dict[str, float]:
        """Return a dict suitable for passing to OnvifPTZ.absolute_move()."""
        return {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom}


class PTZPresetStore:
    """Manages PTZ presets in-memory; persists to config.yaml on save/delete.

    Operates on the live AppConfig object. When a config_path is provided,
    mutations (save, delete) are written back to the YAML file immediately.

    Args:
        cfg: The live AppConfig instance.
        config_path: Optional path to config.yaml for persistence.
            If None, mutations are in-memory only.
    """

    def __init__(self, cfg: AppConfig, config_path: Path | None = None) -> None:
        self._cfg = cfg
        self._config_path = config_path

    def list_presets(self, camera_name: str) -> list[PTZPreset]:
        """Return presets for a camera, or empty list if camera not found."""
        cam = self._find_camera(camera_name)
        if not cam:
            return []
        return [PTZPreset(name=p.name, pan=p.pan, tilt=p.tilt, zoom=p.zoom) for p in cam.presets]

    def get_preset(self, camera_name: str, preset_name: str) -> PTZPreset | None:
        """Get a single preset by name, or None if not found."""
        cam = self._find_camera(camera_name)
        if not cam:
            return None
        for p in cam.presets:
            if p.name == preset_name:
                return PTZPreset(name=p.name, pan=p.pan, tilt=p.tilt, zoom=p.zoom)
        return None

    async def goto_preset(
        self,
        camera_name: str,
        preset_name: str,
        onvif_ptz: OnvifPTZ,
    ) -> None:
        """Recall a preset by calling OnvifPTZ.absolute_move().

        Args:
            camera_name: Camera to look up the preset for.
            preset_name: Name of the preset to recall.
            onvif_ptz: An OnvifPTZ instance for the target camera.

        Raises:
            PTZPresetError: If the preset or camera is not found.
        """
        preset = self.get_preset(camera_name, preset_name)
        if not preset:
            raise PTZPresetError(f"Preset {preset_name!r} not found for camera {camera_name!r}")
        await onvif_ptz.absolute_move(pan=preset.pan, tilt=preset.tilt, zoom=preset.zoom)

    async def save_preset(
        self,
        camera_name: str,
        name: str,
        pan: float,
        tilt: float,
        zoom: float,
    ) -> PTZPreset:
        """Add or overwrite a preset. Persist to config.yaml if path provided.

        Args:
            camera_name: Camera to save the preset on.
            name: Preset name (must be non-empty after strip).
            pan: Pan position.
            tilt: Tilt position.
            zoom: Zoom level.

        Returns:
            The newly created PTZPreset.

        Raises:
            PTZPresetError: If the camera is not found or name is invalid.
        """
        stripped = name.strip()
        if not stripped:
            raise PTZPresetError("Preset name must be non-empty")
        if len(stripped) > 64:
            raise PTZPresetError("Preset name must be 64 characters or fewer")

        cam = self._find_camera(camera_name)
        if not cam:
            raise PTZPresetError(f"Camera {camera_name!r} not found")

        # Remove existing preset with same name (idempotent overwrite)
        cam.presets = [p for p in cam.presets if p.name != stripped]

        new_preset_cfg = PTZPresetConfig(name=stripped, pan=pan, tilt=tilt, zoom=zoom)
        cam.presets.append(new_preset_cfg)

        if self._config_path:
            await self._persist()

        return PTZPreset(name=stripped, pan=pan, tilt=tilt, zoom=zoom)

    async def delete_preset(self, camera_name: str, preset_name: str) -> bool:
        """Delete a preset. Returns True if it existed, False otherwise.

        Persists to config.yaml if path provided.
        """
        cam = self._find_camera(camera_name)
        if not cam:
            return False

        original_len = len(cam.presets)
        cam.presets = [p for p in cam.presets if p.name != preset_name]

        if len(cam.presets) < original_len:
            if self._config_path:
                await self._persist()
            return True
        return False

    def _find_camera(self, name: str) -> CameraConfig | None:
        """Look up a CameraConfig by name in the live AppConfig."""
        for cam in self._cfg.cameras:
            if cam.name == name:
                return cam
        return None

    async def _persist(self) -> None:
        """Write AppConfig back to the YAML file.

        Uses yaml.safe_dump for serialization. Runs in a thread executor
        to avoid blocking the event loop on disk I/O.
        """
        if not self._config_path:
            return

        import asyncio

        data = self._cfg.model_dump(mode="json")
        content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)

        def _write() -> None:
            self._config_path.write_text(content, encoding="utf-8")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)
        log.info("Persisted config to %s", self._config_path)
