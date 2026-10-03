"""PTZ preset management for ONVIF cameras.

Provides PTZPreset (an immutable data class for a named PTZ position),
PTZPresetStore (in-memory manager backed by AppConfig with optional YAML
persistence), and PTZPresetError for domain-specific failures.

Presets are stored as a list of PTZPresetConfig on each CameraConfig in
config.yaml.  The store reads/writes the live AppConfig object. Writes go
through a persist callable that receives one camera's whole preset list; the
default one (used when a config_path is given) patches that camera's raw
``presets`` key in config.yaml under the config lock, so ``${VAR}`` references
in the file are never replaced by their expanded values.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import AppConfig, CameraConfig, PTZPresetConfig
from .ptz import OnvifPTZ

log = logging.getLogger(__name__)

PresetPersist = Callable[[str, list[dict[str, Any]]], None]
"""Writes one camera's full preset list (JSON-ready dicts) to durable config."""


def _config_file_persist(config_path: Path) -> PresetPersist:
    """Default persist callable: patch the camera's raw ``presets`` key in config.yaml.

    Uses ``web.services.camera_config.patch_camera`` (raw YAML, config lock, atomic
    replace). An empty list removes the key. Raises ``KeyError`` when the camera is
    not in the file and ``OSError`` when the file cannot be written.
    """

    def _persist(camera_name: str, presets: list[dict[str, Any]]) -> None:
        # Imported at call time so importing rtsp_warden.onvif never loads the web layer.
        from ..web.services.camera_config import patch_camera

        if presets:
            patch_camera(config_path, camera_name, {"presets": presets})
        else:
            patch_camera(config_path, camera_name, {}, remove_keys=("presets",))

    return _persist


def _validation_message(exc: ValidationError) -> str:
    """First pydantic error message, without the "Value error, " prefix."""
    errors = exc.errors()
    if not errors:
        return "invalid preset"
    return str(errors[0].get("msg", "invalid preset")).removeprefix("Value error, ")


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
            If None and no ``persist`` is given, mutations are in-memory only.
        persist: Optional callable ``(camera_name, presets)`` that stores one
            camera's whole preset list. Defaults to a raw-YAML patch of
            ``config_path`` when a path is given.
    """

    def __init__(
        self,
        cfg: AppConfig,
        config_path: Path | None = None,
        *,
        persist: PresetPersist | None = None,
    ) -> None:
        self._cfg = cfg
        self._config_path = config_path
        if persist is None and config_path is not None:
            persist = _config_file_persist(config_path)
        self._persist_fn = persist

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
            PTZPresetError: If the camera is not found, the name is invalid, or a
                value is out of range. Nothing changes in memory or on disk.
            OSError, KeyError: From the persist callable; memory is left unchanged.
        """
        stripped = name.strip()
        if not stripped:
            raise PTZPresetError("Preset name must be non-empty")
        if len(stripped) > 64:
            raise PTZPresetError("Preset name must be 64 characters or fewer")

        cam = self._find_camera(camera_name)
        if not cam:
            raise PTZPresetError(f"Camera {camera_name!r} not found")

        try:
            new_preset_cfg = PTZPresetConfig(name=stripped, pan=pan, tilt=tilt, zoom=zoom)
        except ValidationError as exc:
            raise PTZPresetError(_validation_message(exc)) from exc

        # Replace an existing preset with the same name (idempotent overwrite).
        updated = [p for p in cam.presets if p.name != stripped]
        updated.append(new_preset_cfg)

        await self._persist(camera_name, updated)
        cam.presets = updated

        return PTZPreset(name=stripped, pan=pan, tilt=tilt, zoom=zoom)

    async def delete_preset(self, camera_name: str, preset_name: str) -> bool:
        """Delete a preset. Returns True if it existed, False otherwise.

        Persists through the persist callable first; if that raises, memory is
        left unchanged and the exception propagates.
        """
        cam = self._find_camera(camera_name)
        if not cam:
            return False

        remaining = [p for p in cam.presets if p.name != preset_name]
        if len(remaining) == len(cam.presets):
            return False

        await self._persist(camera_name, remaining)
        cam.presets = remaining
        return True

    def _find_camera(self, name: str) -> CameraConfig | None:
        """Look up a CameraConfig by name in the live AppConfig."""
        for cam in self._cfg.cameras:
            if cam.name == name:
                return cam
        return None

    async def _persist(self, camera_name: str, presets: list[PTZPresetConfig]) -> None:
        """Hand one camera's new preset list to the persist callable, off the event loop.

        No-op without a persist callable. Exceptions from the callable propagate;
        callers assign ``cam.presets`` only after this returns.
        """
        if self._persist_fn is None:
            return
        payload = [p.model_dump(mode="json") for p in presets]
        await asyncio.to_thread(self._persist_fn, camera_name, payload)
        log.info("Persisted %d PTZ preset(s) for camera %s", len(payload), camera_name)
