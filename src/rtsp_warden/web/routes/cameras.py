"""Camera route handlers for the rtsp-warden web UI.

Provides the camera list, camera detail, the camera status partial for
htmx auto-refresh, the read-only settings page and the same-origin live
preview (MJPEG stream and snapshot). Detection settings (detectors,
sensitivity, classes, rules, retention override, detector reload) live
in ``web/routes/detection.py``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from ...config import CameraConfig
from ..auth_depends import CurrentUser, require_admin, require_user
from ..services.cameras import get_camera_by_name, list_cameras
from ..services.preview import MJPEG_CONTENT_TYPE, find_hub, mjpeg_frames
from ..services.recordings import list_recordings
from ._common import find_camera, get_cfg, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


@router.get("", response_class=HTMLResponse)
async def cameras_list(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Render the camera grid page."""
    cfg = get_cfg(request)
    cameras = list_cameras(cfg, request.app.state.runtime_provider())
    return templates.TemplateResponse(
        request,
        "cameras/list.html",
        {
            "request": request,
            "cameras": cameras,
        },
    )


@router.get("/{name}/snapshot.jpg")
async def camera_snapshot(request: Request, name: str, user=Depends(require_user)) -> Response:
    """Latest JPEG frame from the in-process hub."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    jpeg, _fid, _ts = hub.snapshot()
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{name}/live.mjpeg")
async def camera_live(request: Request, name: str, user=Depends(require_user)) -> StreamingResponse:
    """Same-origin MJPEG stream; replaces links to the side-server on 127.0.0.1."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    return StreamingResponse(
        mjpeg_frames(hub),
        media_type=MJPEG_CONTENT_TYPE,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{name}", response_class=HTMLResponse)
async def camera_detail(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Render a single camera detail page."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Same-origin preview URLs served by this app from the in-process FrameHub.
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        mjpeg_url = f"/cameras/{name}/live.mjpeg"
        snapshot_url = f"/cameras/{name}/snapshot.jpg"

    # Recent recordings for this camera (last 10)
    try:
        recent_recordings, _ = list_recordings(camera_name=name, limit=10)
    except Exception:
        recent_recordings = []

    # Retention info for the detail page
    cam_config = find_camera(cfg, name)
    from ...retention_resolver import resolve_retention

    effective_retention = (
        resolve_retention(cam_config, cfg.retention) if cam_config else cfg.retention
    )
    has_per_camera_retention = cam_config.retention is not None if cam_config else False

    # Sensitivity and detect_classes for the detail page
    sensitivity = cam_config.sensitivity if cam_config else 50.0
    detect_classes = cam_config.detect_classes if cam_config else None

    return templates.TemplateResponse(
        request,
        "cameras/detail.html",
        {
            "request": request,
            "camera": cam,
            "mjpeg_url": mjpeg_url,
            "snapshot_url": snapshot_url,
            "recent_recordings": recent_recordings,
            "effective_retention": effective_retention,
            "has_per_camera_retention": has_per_camera_retention,
            "global_retention": cfg.retention,
            "sensitivity": sensitivity,
            "detect_classes": detect_classes,
        },
    )


@router.get("/{name}/status", response_class=HTMLResponse)
async def camera_status(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Return a partial camera card for htmx auto-refresh."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    return templates.TemplateResponse(
        request,
        "partials/camera_card.html",
        {
            "request": request,
            "camera": cam,
        },
    )


@router.get("/{name}/settings", response_class=HTMLResponse)
async def camera_settings(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> HTMLResponse:
    """Render the read-only camera settings page (admin-only).

    Displays the camera's full configuration and a banner explaining
    that changes require editing config.yaml and restarting the server.
    """
    cfg = get_cfg(request)
    # Find the raw CameraConfig object (not the dict from get_camera_by_name)
    cam_config: CameraConfig | None = None
    for cam in cfg.cameras:
        if cam.name == name:
            cam_config = cam
            break

    if cam_config is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Build a display-friendly dict (redact URLs)
    from ...status_model import redact_rtsp_url

    settings_data = {
        "name": cam_config.name,
        "main_url_redacted": redact_rtsp_url(cam_config.main_url),
        "sub_url_redacted": redact_rtsp_url(cam_config.sub_url) if cam_config.sub_url else None,
        "record_enabled": cam_config.record.enabled,
        "record_output_dir": str(cam_config.record.output_dir),
        "record_container": cam_config.record.main.container,
        "record_chunk_seconds": cam_config.record.main.chunk_seconds,
        "record_transport": cam_config.record.main.rtsp_transport,
        "proxy_enabled": cam_config.proxy.enabled,
        "proxy_mode": cam_config.proxy.mode,
        "proxy_stream": cam_config.proxy.stream,
        "proxy_bind_host": cam_config.proxy.bind_host,
        "proxy_port": cam_config.proxy.port,
        "proxy_fps": cam_config.proxy.fps,
    }

    # Retention settings
    retention = cam_config.record.retention
    settings_data["retention_max_days"] = retention.max_days
    settings_data["retention_max_gb"] = retention.max_gb
    settings_data["retention_keep_last_n"] = retention.keep_last_n

    return templates.TemplateResponse(
        request,
        "cameras/settings.html",
        {
            "request": request,
            "camera_name": name,
            "settings": settings_data,
        },
    )
