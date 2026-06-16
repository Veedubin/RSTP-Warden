"""Camera listing service for the web UI.

Reads camera configuration from AppConfig (the YAML source of truth)
and augments each entry with live status from the runtime when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...status_model import redact_rtsp_url

if TYPE_CHECKING:
    from ...config import AppConfig


def list_cameras(cfg: AppConfig) -> list[dict[str, Any]]:
    """Return camera dicts for display, derived from the YAML config.

    Each dict contains:
      name, enabled, record_enabled, proxy_mode, proxy_port,
      has_proxy, main_url_redacted, sub_url_redacted, status

    Status defaults to "unknown" when no live runtime is attached.
    """
    cameras: list[dict[str, Any]] = []
    for cam in cfg.cameras:
        cameras.append(
            {
                "name": cam.name,
                "enabled": True,  # cameras in config are considered enabled
                "record_enabled": cam.record.enabled,
                "proxy_mode": cam.proxy.mode,
                "proxy_port": cam.proxy.port,
                "has_proxy": cam.proxy.enabled,
                "main_url_redacted": redact_rtsp_url(cam.main_url),
                "sub_url_redacted": redact_rtsp_url(cam.sub_url),
                "status": "unknown",
                "stream": cam.proxy.stream,
                "bind_host": cam.proxy.bind_host,
            }
        )
    return cameras


def get_camera_by_name(cfg: AppConfig, name: str) -> dict[str, Any] | None:
    """Find a camera dict by name. Returns None if not found."""
    for cam_dict in list_cameras(cfg):
        if cam_dict["name"] == name:
            return cam_dict
    return None


# Maps detector type to the key config fields shown in summary_str.
_DETECTOR_TYPE_FIELDS: dict[str, list[str]] = {
    "motion": ["min_area", "sensitivity"],
    "person": ["min_confidence", "scale_factor", "min_neighbors"],
    "vehicle": ["min_confidence"],
    "custom": ["import_path"],
}


def _build_detector_summary(spec: Any) -> str:
    """Build a human-readable summary string for a DetectorSpec."""
    fields = _DETECTOR_TYPE_FIELDS.get(spec.type, [])
    parts: list[str] = []
    for f in fields:
        val = getattr(spec, f, None)
        if val is not None:
            parts.append(f"{f}={val}")
    return ", ".join(parts) if parts else "(defaults)"


def get_camera_detectors(cfg: AppConfig, camera_name: str) -> list[dict[str, Any]]:
    """Return detector specs for a camera, formatted for display.

    Returns:
        list of {type, enabled, interval_seconds, summary_str, has_roi, has_masks}
        Empty list if camera has no detectors configured or camera not found.
    """
    for cam in cfg.cameras:
        if cam.name == camera_name:
            result: list[dict[str, Any]] = []
            for det in cam.detectors:
                result.append(
                    {
                        "type": det.type,
                        "enabled": det.enabled,
                        "interval_seconds": det.interval_seconds,
                        "summary_str": _build_detector_summary(det),
                        "has_roi": det.roi is not None,
                        "has_masks": det.masks is not None and len(det.masks) > 0,
                    }
                )
            return result
    return []
