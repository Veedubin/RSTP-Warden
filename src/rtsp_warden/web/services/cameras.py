"""Camera listing service for the web UI.

Reads camera configuration from AppConfig (the YAML source of truth)
and augments each entry with live status from the runtime when available.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ...status_model import redact_rtsp_url

if TYPE_CHECKING:
    from ...config import AppConfig


def live_status(cam_rt: Any, now: float | None = None) -> dict[str, Any]:
    """Summarize a CameraRuntime for display.

    status: running | restarting | failed | idle | waiting (event mode, no detection yet)
    """
    now = time.time() if now is None else now
    recorder = getattr(cam_rt, "recorder", None)
    procs = list(recorder.processes()) if recorder is not None else []

    last_frame_age: float | None = None
    hub = getattr(cam_rt, "hub", None)
    if hub is not None:
        try:
            _jpeg, _fid, ts = hub.snapshot()
            if ts:
                last_frame_age = max(0.0, now - float(ts))
        except Exception:
            last_frame_age = None

    # Event-mode recorders keep their ingestors stopped until a detection arrives.
    record_mode = getattr(getattr(getattr(recorder, "camera", None), "record", None), "mode", None)
    event_waiting = record_mode == "event" and not getattr(recorder, "_event_recording", False)

    if not procs:
        status = "idle"
    elif event_waiting:
        status = "waiting"
    else:
        any_dead = any(sp.proc is None or sp.proc.poll() is not None for sp in procs)
        if not any_dead:
            status = "running"
        elif getattr(cam_rt, "next_restart_at", 0.0) > now:
            status = "restarting"
        else:
            status = "failed"

    restart_in: int | None = None
    if status == "restarting":
        restart_in = int(cam_rt.next_restart_at - now)

    return {
        "status": status,
        "restart_in": restart_in,
        "last_error": getattr(cam_rt, "last_error", "") or "",
        "last_frame_age": last_frame_age,
    }


def _find_runtime(rt: Any, name: str) -> Any | None:
    for cam_rt in getattr(rt, "cameras", None) or []:
        if cam_rt.camera.name == name:
            return cam_rt
    return None


def list_cameras(cfg: AppConfig, rt: Any = None) -> list[dict[str, Any]]:
    """Return camera dicts for display, derived from the YAML config.

    Each dict contains:
      name, enabled, record_enabled, proxy_mode, proxy_port, has_proxy,
      main_url_redacted, sub_url_redacted, status, restart_in, last_error,
      last_frame_age, stream, bind_host

    When *rt* (the live AppRuntime) is given, status fields come from
    ``live_status``; otherwise status is "unknown".
    """
    cameras: list[dict[str, Any]] = []
    for cam in cfg.cameras:
        row: dict[str, Any] = {
            "name": cam.name,
            "enabled": True,  # cameras in config are considered enabled
            "record_enabled": cam.record.enabled,
            "proxy_mode": cam.proxy.mode,
            "proxy_port": cam.proxy.port,
            "has_proxy": cam.proxy.enabled,
            "main_url_redacted": redact_rtsp_url(cam.main_url),
            "sub_url_redacted": redact_rtsp_url(cam.sub_url) if cam.sub_url else None,
            "status": "unknown",
            "restart_in": None,
            "last_error": "",
            "last_frame_age": None,
            "stream": cam.proxy.stream,
            "bind_host": cam.proxy.bind_host,
        }
        cam_rt = _find_runtime(rt, cam.name) if rt is not None else None
        if cam_rt is not None:
            row.update(live_status(cam_rt))
        cameras.append(row)
    return cameras


def get_camera_by_name(cfg: AppConfig, name: str, rt: Any = None) -> dict[str, Any] | None:
    """Find a camera dict by name. Returns None if not found."""
    for cam_dict in list_cameras(cfg, rt):
        if cam_dict["name"] == name:
            return cam_dict
    return None
