"""Detection zone route handlers for the rtsp-warden web UI.

Provides admin-only views for managing grid-based detection zones
per camera: list zones, edit/save zone configuration, delete zones,
and hot-reload detectors after zone changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig, CameraConfig, GridZoneConfig
from ..auth_depends import CurrentUser, require_admin
from ..config_lock import _locked_write_yaml
from ..paths import TEMPLATES_DIR

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_cfg(request: Request) -> AppConfig:
    """Get the AppConfig from app state, raising 503 if unavailable."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


def _get_config_path(request: Request) -> Path | None:
    """Try to determine the config file path from app state.

    Returns None if the path is not available (in-memory config only).
    """
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is not None:
        return Path(config_path)
    return None


def _find_camera_config(cfg: AppConfig, name: str) -> CameraConfig | None:
    """Find a CameraConfig object by name."""
    for cam in cfg.cameras:
        if cam.name == name:
            return cam
    return None


@router.get("/{name}/zones", response_class=HTMLResponse)
async def zones_list(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the detection zones page for a camera (admin-only).

    Lists existing zones and provides links to add/edit/delete zones.
    """
    cfg = _get_cfg(request)
    cam = _find_camera_config(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    return _templates.TemplateResponse(
        request,
        "cameras/zones.html",
        {
            "request": request,
            "camera_name": name,
            "zones": cam.zones,
        },
    )


@router.get("/{name}/zones/editor", response_class=HTMLResponse)
async def zones_editor(
    request: Request,
    name: str,
    zone_name: str = "",
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the grid zone editor as an htmx partial (admin-only).

    Query params:
        zone_name: If editing an existing zone, pass its name.
                   If empty, the editor creates a new zone.

    Returns:
        HTML partial with SVG grid overlay and snapshot image.
    """
    cfg = _get_cfg(request)
    cam = _find_camera_config(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Find existing zone if editing
    existing_zone: GridZoneConfig | None = None
    for z in cam.zones:
        if z.name == zone_name:
            existing_zone = z
            break

    # Default values for new zone
    grid_cols = existing_zone.grid_cols if existing_zone else 16
    grid_rows = existing_zone.grid_rows if existing_zone else 16
    frame_width = existing_zone.frame_width if existing_zone else 1920
    frame_height = existing_zone.frame_height if existing_zone else 1080
    blocked_cells = existing_zone.blocked_cells if existing_zone else set()

    # Build snapshot URL from camera proxy config
    snapshot_url = ""
    if cam.proxy.enabled and cam.proxy.mode == "mjpeg":
        host = cam.proxy.bind_host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        snapshot_url = f"http://{host}:{cam.proxy.port}/snapshot.jpg"

    # Serialize blocked cells as list of "col,row" strings for Alpine.js
    blocked_cells_json = [{"col": c, "row": r} for c, r in sorted(blocked_cells)]

    return _templates.TemplateResponse(
        request,
        "cameras/zones_editor.html",
        {
            "request": request,
            "camera_name": name,
            "zone_name": zone_name,
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "blocked_cells_json": blocked_cells_json,
            "snapshot_url": snapshot_url,
        },
    )


@router.post("/{name}/zones")
async def save_zone(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Save a zone configuration (admin-only).

    Form fields:
        zone_name: Human-readable zone label (required).
        grid_cols: Grid columns, 2-64 (required).
        grid_rows: Grid rows, 2-64 (required).
        frame_width: Camera frame width in pixels (required).
        frame_height: Camera frame height in pixels (required).
        blocked_cell: Zero or more "col,row" strings marking blocked cells.

    On success, redirects to /cameras/{name}/zones.
    """
    cfg = _get_cfg(request)
    cam = _find_camera_config(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    form = await request.form()

    # Parse required fields
    zone_name_raw = form.get("zone_name")
    if not zone_name_raw or not str(zone_name_raw).strip():
        raise HTTPException(status_code=422, detail="zone_name is required")
    zone_name = str(zone_name_raw).strip()

    try:
        grid_cols = int(form.get("grid_cols", 16))
        grid_rows = int(form.get("grid_rows", 16))
        frame_width = int(form.get("frame_width", 1920))
        frame_height = int(form.get("frame_height", 1080))
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="grid_cols, grid_rows, frame_width, frame_height must be integers",
        ) from exc

    # Validate grid dimensions
    if not (2 <= grid_cols <= 64):
        raise HTTPException(status_code=422, detail="grid_cols must be between 2 and 64")
    if not (2 <= grid_rows <= 64):
        raise HTTPException(status_code=422, detail="grid_rows must be between 2 and 64")
    if frame_width <= 0 or frame_height <= 0:
        raise HTTPException(status_code=422, detail="frame dimensions must be positive")

    # Parse blocked cells from "col,row" form values
    blocked_cells: set[tuple[int, int]] = set()
    raw_cells = form.getlist("blocked_cell")
    for cell_str in raw_cells:
        cell_str = str(cell_str).strip()
        if not cell_str:
            continue
        try:
            parts = cell_str.split(",")
            col, row = int(parts[0]), int(parts[1])
        except (ValueError, IndexError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid blocked_cell value: {cell_str!r}"
            ) from exc

        # Validate cell bounds
        if not (0 <= col < grid_cols) or not (0 <= row < grid_rows):
            raise HTTPException(
                status_code=422,
                detail=f"Cell ({col},{row}) out of bounds for {grid_cols}x{grid_rows} grid",
            )
        blocked_cells.add((col, row))

    # Build the new zone config
    new_zone = GridZoneConfig(
        name=zone_name,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        blocked_cells=blocked_cells,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # Update camera's zones list: replace existing zone by name, or append
    replaced = False
    updated_zones = []
    for z in cam.zones:
        if z.name == zone_name:
            updated_zones.append(new_zone)
            replaced = True
        else:
            updated_zones.append(z)
    if not replaced:
        updated_zones.append(new_zone)

    cam.zones = updated_zones

    # Persist to config.yaml
    config_path = _get_config_path(request)
    if config_path is not None:
        _persist_zones(config_path, cfg)

    return RedirectResponse(url=f"/cameras/{name}/zones", status_code=303)


@router.post("/{name}/zones/{zone_name}/delete")
async def delete_zone(
    request: Request,
    name: str,
    zone_name: str,
    user: CurrentUser = Depends(require_admin),
) -> RedirectResponse:
    """Delete a zone from a camera (admin-only).

    On success, redirects to /cameras/{name}/zones.
    """
    cfg = _get_cfg(request)
    cam = _find_camera_config(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    cam.zones = [z for z in cam.zones if z.name != zone_name]

    # Persist to config.yaml
    config_path = _get_config_path(request)
    if config_path is not None:
        _persist_zones(config_path, cfg)

    return RedirectResponse(url=f"/cameras/{name}/zones", status_code=303)


@router.post("/{name}/zones/reload")
async def reload_zones(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> JSONResponse:
    """Hot-reload detectors after zone changes (admin-only).

    Calls rebuild_camera_detectors(name) to apply the updated
    zone configuration immediately without a server restart.

    Returns:
        JSON response with status and camera name.
    """
    from ...app import AppRuntime

    app_rt: AppRuntime | None = getattr(request.app.state, "runtime", None)
    if app_rt is None:
        raise HTTPException(status_code=503, detail="Server runtime not initialized")

    cfg = _get_cfg(request)
    cam = _find_camera_config(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    try:
        app_rt.rebuild_camera_detectors(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("failed to reload detectors for camera %s: %s", name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload detectors: {exc}") from exc

    return JSONResponse({"ok": True, "camera": name})


def _persist_zones(config_path: Path, cfg: AppConfig) -> None:
    """Serialize current AppConfig zones back to config.yaml on disk.

    Uses _locked_write_yaml for crash-safe, locked writes.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for i, cam_dict in enumerate(cameras_data):
        cam_name = cam_dict.get("name")
        if cam_name is None:
            continue
        # Find the matching CameraConfig object
        for cam_cfg in cfg.cameras:
            if cam_cfg.name == cam_name:
                if cam_cfg.zones:
                    cameras_data[i]["zones"] = [_zone_to_dict(z) for z in cam_cfg.zones]
                elif "zones" in cameras_data[i]:
                    del cameras_data[i]["zones"]
                break
    data["cameras"] = cameras_data
    _locked_write_yaml(config_path, data)


def _zone_to_dict(zone: GridZoneConfig) -> dict:
    """Convert a GridZoneConfig to a YAML-serializable dict.

    Handles the set-of-tuples blocked_cells by converting to a
    list of [col, row] lists (YAML-friendly).
    """
    return {
        "name": zone.name,
        "grid_cols": zone.grid_cols,
        "grid_rows": zone.grid_rows,
        "blocked_cells": [[c, r] for c, r in sorted(zone.blocked_cells)],
        "frame_width": zone.frame_width,
        "frame_height": zone.frame_height,
        "enabled": zone.enabled,
    }
