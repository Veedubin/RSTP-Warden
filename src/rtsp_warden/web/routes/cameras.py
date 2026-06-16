"""Camera route handlers for the rtsp-warden web UI.

Provides camera list, camera detail, and camera status partial
for htmx auto-refresh.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig, CameraConfig
from ..auth_depends import CurrentUser, require_admin, require_user
from ..paths import TEMPLATES_DIR
from ..services.cameras import get_camera_by_name, get_camera_detectors, list_cameras
from ..services.recordings import list_recordings

router = APIRouter(prefix="/cameras")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_cfg(request: Request) -> AppConfig:
    """Get the AppConfig from app state, raising 503 if unavailable."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


@router.get("", response_class=HTMLResponse)
async def cameras_list(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Render the camera grid page."""
    cfg = _get_cfg(request)
    cameras = list_cameras(cfg)
    return _templates.TemplateResponse(
        request,
        "cameras/list.html",
        {
            "request": request,
            "cameras": cameras,
        },
    )


@router.get("/{name}", response_class=HTMLResponse)
async def camera_detail(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Render a single camera detail page."""
    cfg = _get_cfg(request)
    cam = get_camera_by_name(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Build MJPEG and snapshot URLs for proxy-enabled cameras
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        host = cam.get("bind_host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = cam["proxy_port"]
        mjpeg_url = f"http://{host}:{port}/mjpeg"
        snapshot_url = f"http://{host}:{port}/snapshot.jpg"

    # Recent recordings for this camera (last 10)
    try:
        recent_recordings, _ = list_recordings(camera_name=name, limit=10)
    except Exception:
        recent_recordings = []

    return _templates.TemplateResponse(
        request,
        "cameras/detail.html",
        {
            "request": request,
            "camera": cam,
            "mjpeg_url": mjpeg_url,
            "snapshot_url": snapshot_url,
            "recent_recordings": recent_recordings,
        },
    )


@router.get("/{name}/status", response_class=HTMLResponse)
async def camera_status(request: Request, name: str, user=Depends(require_user)) -> HTMLResponse:
    """Return a partial camera card for htmx auto-refresh."""
    cfg = _get_cfg(request)
    cam = get_camera_by_name(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    return _templates.TemplateResponse(
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
    cfg = _get_cfg(request)
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
        "sub_url_redacted": redact_rtsp_url(cam_config.sub_url),
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

    return _templates.TemplateResponse(
        request,
        "cameras/settings.html",
        {
            "request": request,
            "camera_name": name,
            "settings": settings_data,
        },
    )


@router.get("/{name}/detectors", response_class=HTMLResponse)
async def cameras_detectors_partial(
    request: Request,
    name: str,
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """htmx partial -- refresh detector list for one camera.

    Returns the partial template HTML directly.
    """
    cfg = _get_cfg(request)
    detectors = get_camera_detectors(cfg, name)
    has_roi = any(d["has_roi"] for d in detectors)
    has_masks = any(d["has_masks"] for d in detectors)
    num_masks = sum(1 for d in detectors if d["has_masks"])

    return _templates.TemplateResponse(
        request,
        "partials/detector_list.html",
        {
            "request": request,
            "detectors": detectors,
            "camera_name": name,
            "has_roi": has_roi,
            "has_masks": has_masks,
            "num_masks": num_masks,
        },
    )
