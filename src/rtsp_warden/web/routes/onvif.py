"""ONVIF web routes (admin only).

Provides discovery and PTZ control endpoints for ONVIF cameras.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig
from ...onvif.discovery import OnvifDiscovery, OnvifError
from ...onvif.ptz import OnvifPTZ
from ..auth_depends import require_admin
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/onvif", tags=["onvif"])

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

log = logging.getLogger(__name__)

# PTZ action to velocity mapping
PTZ_ACTIONS: dict[str, dict[str, float]] = {
    "left": {"pan": -1.0, "tilt": 0.0, "zoom": 0.0},
    "right": {"pan": 1.0, "tilt": 0.0, "zoom": 0.0},
    "up": {"pan": 0.0, "tilt": 1.0, "zoom": 0.0},
    "down": {"pan": 0.0, "tilt": -1.0, "zoom": 0.0},
    "zoom_in": {"pan": 0.0, "tilt": 0.0, "zoom": 1.0},
    "zoom_out": {"pan": 0.0, "tilt": 0.0, "zoom": -1.0},
}


def _get_cfg(request: Request) -> AppConfig:
    """Get the AppConfig from app state, raising 503 if unavailable."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


@router.get("", response_class=HTMLResponse)
async def onvif_index(request: Request, user=Depends(require_admin)) -> HTMLResponse:
    """Show ONVIF discovery status and PTZ control page."""
    cfg = _get_cfg(request)
    return _templates.TemplateResponse(
        request,
        "onvif/index.html",
        {
            "request": request,
            "cfg": cfg,
        },
    )


@router.post("/discover")
async def onvif_discover(request: Request, user=Depends(require_admin)) -> JSONResponse:
    """Run WS-Discovery and return discovered cameras as JSON."""
    cfg = _get_cfg(request)

    if not cfg.onvif.discovery_enabled:
        return JSONResponse(
            status_code=200,
            content={"cameras": [], "error": "ONVIF discovery is disabled in config"},
        )

    try:
        discovery = OnvifDiscovery(timeout_seconds=cfg.onvif.discovery_timeout_seconds)
        cameras = discovery.discover()
        return JSONResponse(
            status_code=200,
            content={
                "cameras": [
                    {
                        "xaddr": c.xaddr,
                        "address": c.address,
                        "name": c.name,
                        "manufacturer": c.manufacturer,
                        "model": c.model,
                        "types": c.types,
                    }
                    for c in cameras
                ]
            },
        )
    except OnvifError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except Exception as exc:
        log.exception("Unexpected error during ONVIF discovery")
        return JSONResponse(status_code=500, content={"error": f"Discovery failed: {exc}"})


@router.post("/cameras/{name}/ptz")
async def onvif_ptz(request: Request, name: str, user=Depends(require_admin)) -> JSONResponse:
    """Execute a PTZ command on a camera.

    Body: {"action": "left"|"right"|"up"|"down"|"zoom_in"|"zoom_out"|"stop",
           "duration_ms": 500}
    """
    cfg = _get_cfg(request)

    if not cfg.onvif.ptz_enabled:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "PTZ is disabled in config"},
        )

    body = await request.json()
    action = body.get("action", "")

    if action == "stop":
        try:
            # Find camera config to get device URL
            xaddr = _get_camera_xaddr(cfg, name)
            ptz = OnvifPTZ(
                device_xaddr=xaddr,
                username=cfg.onvif.username,
                password=cfg.onvif.password,
                timeout_seconds=cfg.onvif.ptz_timeout_seconds,
            )
            await ptz.stop()
            return JSONResponse(content={"success": True})
        except OnvifError as exc:
            return JSONResponse(content={"success": False, "error": str(exc)})
        except Exception as exc:
            log.exception("Unexpected PTZ stop error for camera %s", name)
            return JSONResponse(content={"success": False, "error": f"PTZ failed: {exc}"})

    if action not in PTZ_ACTIONS:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid PTZ action: {action}"},
        )

    try:
        xaddr = _get_camera_xaddr(cfg, name)
        ptz = OnvifPTZ(
            device_xaddr=xaddr,
            username=cfg.onvif.username,
            password=cfg.onvif.password,
            timeout_seconds=cfg.onvif.ptz_timeout_seconds,
        )

        velocity = PTZ_ACTIONS[action]
        await ptz.continuous_move(**velocity)

        # If duration_ms is specified, wait then stop
        duration_ms = body.get("duration_ms")
        if duration_ms and isinstance(duration_ms, (int, float)):
            await asyncio.sleep(duration_ms / 1000.0)
            await ptz.stop()

        return JSONResponse(content={"success": True})
    except OnvifError as exc:
        return JSONResponse(content={"success": False, "error": str(exc)})
    except Exception as exc:
        log.exception("Unexpected PTZ error for camera %s action %s", name, action)
        return JSONResponse(content={"success": False, "error": f"PTZ failed: {exc}"})


def _get_camera_xaddr(cfg: AppConfig, name: str) -> str:
    """Look up a camera's ONVIF device URL by name.

    For v1, constructs a default URL from the camera's main RTSP URL.
    Future: store ONVIF xaddr from discovery in camera config.
    """
    # Try to find the camera config
    for cam in cfg.cameras:
        if cam.name == name:
            # Derive ONVIF device service URL from RTSP URL
            # This is a best-effort heuristic; cameras may use different ports
            return _derive_onvif_xaddr(cam.main_url)

    # If no camera found by name, raise
    raise OnvifError(f"Camera {name!r} not found in config")


def _derive_onvif_xaddr(rtsp_url: str) -> str:
    """Derive a likely ONVIF device service URL from an RTSP URL.

    Most ONVIF cameras expose the device service on port 80 at /onvif/device_service.
    """
    try:
        # rtsp://user:pass@192.168.1.100:554/stream -> http://192.168.1.100/onvif/device_service
        after_scheme = rtsp_url.split("://", 1)[1] if "://" in rtsp_url else rtsp_url
        # Strip credentials if present
        if "@" in after_scheme:
            after_scheme = after_scheme.split("@", 1)[1]
        host_port = after_scheme.split("/", 1)[0]
        # Remove RTSP port (554) and use default HTTP port
        host = host_port.split(":")[0] if ":" in host_port else host_port
        return f"http://{host}/onvif/device_service"
    except (IndexError, ValueError):
        return rtsp_url
