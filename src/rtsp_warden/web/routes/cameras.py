"""Camera route handlers for the rtsp-warden web UI.

Provides the camera list, camera detail, the camera status partial for
htmx auto-refresh, the read-only settings page and the same-origin live
preview (MJPEG stream and snapshot). Detection settings (detectors,
sensitivity, classes, rules, retention override, detector reload) live
in ``web/routes/detection.py``.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse

from ..auth_depends import CurrentUser, require_admin, require_user
from ..services.cameras import get_camera_by_name, list_cameras
from ..services.preview import MJPEG_CONTENT_TYPE, find_hub, mjpeg_frames
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


# htmx stops an "every Ns" poll when a response has this status, then swaps in the
# (empty) body, so the polling element disappears.
HTMX_STOP_POLLING = 286


def _missing_camera(request: Request, name: str) -> Response:
    """Answer a status poll or page request for a camera that is not configured.

    An htmx poll gets 286 with an empty body: htmx removes the polling card or row and
    stops polling, so a deleted camera does not poll a 404 forever. Anything else is 404.
    """
    if request.headers.get("hx-request") == "true":
        return Response(status_code=HTMX_STOP_POLLING)
    raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")


@router.get("/{name}", response_class=HTMLResponse)
async def camera_detail(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> HTMLResponse:
    """Render a single camera detail page.

    Admin-only controls (edit, zones, sensitivity, classes, the retention form) are
    rendered only for admins; viewers would get 403 on every one of them.
    """
    from ...retention_resolver import resolve_retention

    cfg = get_cfg(request)
    cam_config = find_camera(cfg, name)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam_config is None or cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")

    # Same-origin preview URLs served by this app from the in-process FrameHub.
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        mjpeg_url = f"/cameras/{name}/live.mjpeg"
        snapshot_url = f"/cameras/{name}/snapshot.jpg"

    return templates.TemplateResponse(
        request,
        "cameras/detail.html",
        {
            "request": request,
            "camera": cam,
            "is_admin": user.role == "admin",
            "mjpeg_url": mjpeg_url,
            "snapshot_url": snapshot_url,
            "effective_retention": resolve_retention(cam_config, cfg.retention),
            "has_per_camera_retention": cam_config.retention is not None,
            "global_retention": cfg.retention,
            "sensitivity": cam_config.sensitivity,
            "detect_classes": cam_config.detect_classes,
            "zone_count": len(cam_config.zones),
        },
    )


@router.get("/{name}/status", response_class=HTMLResponse)
async def camera_status(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> Response:
    """Return the camera card partial; every card polls this every 5 s."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        return _missing_camera(request, name)
    return templates.TemplateResponse(
        request,
        "partials/camera_card.html",
        {"request": request, "camera": cam},
    )


@router.get("/{name}/status-row", response_class=HTMLResponse)
async def camera_status_row(
    request: Request, name: str, user: CurrentUser = Depends(require_user)
) -> Response:
    """Return the detail page's status ``<tr>``; the row polls this every 5 s."""
    cfg = get_cfg(request)
    cam = get_camera_by_name(cfg, name, request.app.state.runtime_provider())
    if cam is None:
        return _missing_camera(request, name)
    return templates.TemplateResponse(
        request,
        "partials/status_row.html",
        {"request": request, "camera": cam},
    )


@router.get("/{name}/settings")
async def camera_settings_redirect(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> RedirectResponse:
    """Old read-only settings page; camera settings are now edited at /cameras/{name}/edit."""
    if find_camera(get_cfg(request), name) is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    return RedirectResponse(url=f"/cameras/{quote(name, safe='')}/edit", status_code=303)
