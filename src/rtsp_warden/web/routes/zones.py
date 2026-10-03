"""Detection zone route handlers for the rtsp-warden web UI.

Provides admin-only views for managing grid-based detection zones
per camera: list zones, edit/save zone configuration, delete zones,
and hot-reload detectors after zone changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from ...config import AppConfig, GridZoneConfig, validate_camera_zones
from ..auth_depends import CurrentUser, require_admin
from ..services.detection import update_config_yaml
from ._common import find_camera, get_cfg, get_config_path, is_htmx, set_flash, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


@router.get("/{name}/zones", response_class=HTMLResponse)
async def zones_list(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Render the detection zones page for a camera (admin-only).

    Lists existing zones and provides links to add/edit/delete zones.
    """
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    return templates.TemplateResponse(
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
    """Render the grid zone editor (admin-only).

    Query params:
        zone_name: If editing an existing zone, pass its name.
                   If empty, the editor creates a new zone.

    Returns:
        The editor inside the base layout when a link opens it, or the bare
        partial (SVG grid overlay and snapshot image) for an htmx request.
    """
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
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

    # Same-origin snapshot served by this app from the in-process FrameHub.
    snapshot_url = ""
    if cam.proxy.enabled and cam.proxy.mode == "mjpeg":
        snapshot_url = f"/cameras/{name}/snapshot.jpg"

    # Serialize blocked cells as list of "col,row" strings for Alpine.js
    blocked_cells_json = [{"col": c, "row": r} for c, r in sorted(blocked_cells)]

    # The zones page opens the editor with a plain link, so a normal request gets the
    # full page (layout, CSS, Alpine, viewport meta tag); an htmx request gets the partial.
    template_name = (
        "cameras/zones_editor.html" if is_htmx(request) else "cameras/zones_editor_page.html"
    )
    return templates.TemplateResponse(
        request,
        template_name,
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
) -> Response:
    """Save a zone configuration (admin-only).

    Form fields:
        zone_name: Human-readable zone label (required; at most 64 characters, no "/").
        kind: "ignore" or "area". When absent, an existing zone keeps its kind and a
            new zone is an ignore zone (the editor gains the kind control with the
            RW-2 UI pass; until then its saves send no kind).
        grid_cols: Grid columns, 2-64 (required).
        grid_rows: Grid rows, 2-64 (required).
        frame_width: Camera frame width in pixels (required).
        frame_height: Camera frame height in pixels (required).
        blocked_cell: Zero or more "col,row" strings marking blocked cells.

    A change that a rule's ``zones`` list would no longer accept (turning a
    rule's area into an ignore zone) is refused with 422 and nothing is written.
    A config.yaml that cannot be written is reported with its path (R9).

    On success, redirects to /cameras/{name}/zones.
    """
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    form = await request.form()

    # Parse required fields
    zone_name_raw = form.get("zone_name")
    if not zone_name_raw or not str(zone_name_raw).strip():
        raise HTTPException(status_code=422, detail="zone_name is required")
    zone_name = str(zone_name_raw).strip()

    kind_field = form.get("kind")
    if kind_field is None or not str(kind_field).strip():
        existing = next((z for z in cam.zones if z.name == zone_name), None)
        kind_raw = existing.kind if existing is not None else "ignore"
    else:
        kind_raw = str(kind_field).strip()
    if kind_raw not in ("ignore", "area"):
        raise HTTPException(status_code=422, detail="kind must be 'ignore' or 'area'")
    kind: Literal["ignore", "area"] = "area" if kind_raw == "area" else "ignore"

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

    # Build the new zone config (the model enforces the zone-name rules)
    try:
        new_zone = GridZoneConfig(
            name=zone_name,
            kind=kind,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            blocked_cells=blocked_cells,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    except ValidationError as exc:
        detail = "; ".join(str(err["msg"]) for err in exc.errors())
        raise HTTPException(status_code=422, detail=detail) from exc

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

    # Rules name area zones; refuse a change that would make config.yaml fail to load.
    try:
        validate_camera_zones(updated_zones, cam.rules)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    previous_zones = cam.zones
    cam.zones = updated_zones

    # Persist to config.yaml
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            _persist_zones(config_path, cfg)
        except OSError as exc:
            cam.zones = previous_zones
            return _write_failed(request, name, config_path, exc)

    return RedirectResponse(url=f"/cameras/{name}/zones", status_code=303)


@router.post("/{name}/zones/{zone_name}/delete")
async def delete_zone(
    request: Request,
    name: str,
    zone_name: str,
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Delete a zone from a camera (admin-only).

    A zone that a rule names is not deleted (422): config.yaml would fail to load.
    A config.yaml that cannot be written is reported with its path (R9).

    On success, redirects to /cameras/{name}/zones.
    """
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    remaining = [z for z in cam.zones if z.name != zone_name]
    try:
        validate_camera_zones(remaining, cam.rules)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    previous_zones = cam.zones
    cam.zones = remaining

    # Persist to config.yaml
    config_path = get_config_path(request)
    if config_path is not None:
        try:
            _persist_zones(config_path, cfg)
        except OSError as exc:
            cam.zones = previous_zones
            return _write_failed(request, name, config_path, exc)

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

    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    try:
        # Off the event loop: a rebuild sets up the new detectors (an ONNX session load)
        # and tears the old runner down (joins its worker).
        await run_in_threadpool(app_rt.rebuild_camera_detectors, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("failed to reload detectors for camera %s: %s", name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload detectors: {exc}") from exc

    return JSONResponse({"ok": True, "camera": name})


def _write_failed(request: Request, name: str, config_path: Path, exc: OSError) -> Response:
    """Answer a zone save or delete whose config.yaml write failed (R9: never a 500).

    A plain form post gets a 303 back to the zones page with an error flash. The zone
    editor posts with htmx, and an XHR follows a 303 without a full page load, so it
    gets ``HX-Redirect`` instead and the reloaded page shows the flash.
    """
    log.warning("could not write the zones of camera %s to %s: %s", name, config_path, exc)
    url = f"/cameras/{name}/zones"
    response: Response
    if is_htmx(request):
        response = Response(status_code=200, headers={"HX-Redirect": url})
    else:
        response = RedirectResponse(url=url, status_code=303)
    set_flash(response, f"Could not write {config_path}: {exc.strerror or exc}.", "error")
    return response


def _persist_zones(config_path: Path, cfg: AppConfig) -> None:
    """Serialize current AppConfig zones back to config.yaml on disk.

    Reads, changes and writes the raw file as one step (update_config_yaml), so a zone
    save never drops a detector toggle or a field save made at the same moment.
    """

    def mutate(data: dict) -> bool:
        cameras_data = data.get("cameras") or []
        for cam_dict in cameras_data:
            if not isinstance(cam_dict, dict) or cam_dict.get("name") is None:
                continue
            # Find the matching CameraConfig object
            for cam_cfg in cfg.cameras:
                if cam_cfg.name == cam_dict["name"]:
                    if cam_cfg.zones:
                        cam_dict["zones"] = [_zone_to_dict(z) for z in cam_cfg.zones]
                    elif "zones" in cam_dict:
                        del cam_dict["zones"]
                    break
        data["cameras"] = cameras_data
        return True

    update_config_yaml(config_path, mutate)


def _zone_to_dict(zone: GridZoneConfig) -> dict:
    """Convert a GridZoneConfig to a YAML-serializable dict.

    Handles the set-of-tuples blocked_cells by converting to a
    list of [col, row] lists (YAML-friendly).
    """
    return {
        "name": zone.name,
        "kind": zone.kind,
        "grid_cols": zone.grid_cols,
        "grid_rows": zone.grid_rows,
        "blocked_cells": [[c, r] for c, r in sorted(zone.blocked_cells)],
        "frame_width": zone.frame_width,
        "frame_height": zone.frame_height,
        "enabled": zone.enabled,
    }
