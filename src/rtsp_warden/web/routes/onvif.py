"""ONVIF web routes (admin only).

Provides discovery, PTZ control, preset management, and event subscription
endpoints for ONVIF cameras.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from ...config import AppConfig
from ...onvif.discovery import OnvifDiscovery, OnvifError
from ...onvif.events import (
    OnvifEvent,
    OnvifEventSubscriber,
    get_subscription_states,
    register_subscriber,
    unregister_subscriber,
)
from ...onvif.presets import PTZPresetError, PTZPresetStore
from ...onvif.ptz import OnvifClient, OnvifPTZ
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


# ---------------------------------------------------------------------------
# PTZ Preset routes
# ---------------------------------------------------------------------------


def _get_config_path(request: Request) -> Path | None:
    """Try to determine the config file path from app state.

    Returns None if the path is not available (in-memory config only).
    """
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is not None:
        return Path(config_path)
    return None


@router.get("/cameras/{name}/ptz", response_class=HTMLResponse)
async def onvif_ptz_page(request: Request, name: str, user=Depends(require_admin)) -> HTMLResponse:
    """Render the PTZ control and presets page for a specific camera."""
    cfg = _get_cfg(request)
    store = PTZPresetStore(cfg, config_path=_get_config_path(request))
    presets = store.list_presets(name)
    return _templates.TemplateResponse(
        request,
        "onvif/index.html",
        {
            "request": request,
            "cfg": cfg,
            "selected_camera": name,
            "presets": [p.__dict__ for p in presets],
        },
    )


@router.get("/cameras/{name}/presets")
async def onvif_list_presets(
    request: Request, name: str, user=Depends(require_admin)
) -> JSONResponse:
    """List PTZ presets for a camera as JSON."""
    cfg = _get_cfg(request)
    store = PTZPresetStore(cfg)
    presets = store.list_presets(name)
    return JSONResponse(
        content={
            "camera": name,
            "presets": [
                {"name": p.name, "pan": p.pan, "tilt": p.tilt, "zoom": p.zoom} for p in presets
            ],
        }
    )


@router.post("/cameras/{name}/ptz/goto")
async def onvif_goto_preset(
    request: Request, name: str, user=Depends(require_admin)
) -> JSONResponse:
    """Move camera to a saved preset position.

    Body: {"preset_name": "front_gate"}
    """
    cfg = _get_cfg(request)

    if not cfg.onvif.ptz_enabled:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "PTZ is disabled in config"},
        )

    body = await request.json()
    preset_name = body.get("preset_name", "")

    store = PTZPresetStore(cfg)
    preset = store.get_preset(name, preset_name)
    if not preset:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": f"Preset {preset_name!r} not found for camera {name!r}",
            },
        )

    try:
        xaddr = _get_camera_xaddr(cfg, name)
        ptz = OnvifPTZ(
            device_xaddr=xaddr,
            username=cfg.onvif.username,
            password=cfg.onvif.password,
            timeout_seconds=cfg.onvif.ptz_timeout_seconds,
        )
        await ptz.absolute_move(pan=preset.pan, tilt=preset.tilt, zoom=preset.zoom)
        return JSONResponse(content={"success": True, "preset": preset.__dict__})
    except OnvifError as exc:
        return JSONResponse(content={"success": False, "error": str(exc)})
    except Exception as exc:
        log.exception("PTZ goto preset error for camera %s preset %s", name, preset_name)
        return JSONResponse(content={"success": False, "error": f"PTZ goto failed: {exc}"})


@router.post("/cameras/{name}/ptz/save")
async def onvif_save_preset(
    request: Request, name: str, user=Depends(require_admin)
) -> RedirectResponse:
    """Save a new PTZ preset and redirect back to the PTZ page.

    Body: {"name": "front_gate", "pan": 0.5, "tilt": 0.3, "zoom": 0.0}
    """
    cfg = _get_cfg(request)
    store = PTZPresetStore(cfg, config_path=_get_config_path(request))

    body = await request.json()
    preset_name = body.get("name", "")
    pan = float(body.get("pan", 0.0))
    tilt = float(body.get("tilt", 0.0))
    zoom = float(body.get("zoom", 0.0))

    try:
        await store.save_preset(name, preset_name, pan, tilt, zoom)
    except PTZPresetError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(exc)},
        )

    redirect_url = f"/onvif/cameras/{name}/ptz"
    return RedirectResponse(url=redirect_url, status_code=303)


@router.post("/cameras/{name}/ptz/{preset_name}/delete")
async def onvif_delete_preset(
    request: Request, name: str, preset_name: str, user=Depends(require_admin)
) -> RedirectResponse:
    """Delete a PTZ preset and redirect back to the PTZ page."""
    cfg = _get_cfg(request)
    store = PTZPresetStore(cfg, config_path=_get_config_path(request))

    await store.delete_preset(name, preset_name)

    redirect_url = f"/onvif/cameras/{name}/ptz"
    return RedirectResponse(url=redirect_url, status_code=303)


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


# ---------------------------------------------------------------------------
# Event subscription routes
# ---------------------------------------------------------------------------


@router.get("/events")
async def onvif_events_status(request: Request, user=Depends(require_admin)) -> JSONResponse:
    """Return JSON list of all active event subscription states."""
    states = get_subscription_states()
    return JSONResponse(content={"subscriptions": states})


@router.post("/cameras/{name}/events/subscribe")
async def onvif_events_subscribe(
    request: Request, name: str, user=Depends(require_admin)
) -> JSONResponse:
    """Start an ONVIF event subscription for a camera.

    Creates an OnvifEventSubscriber, starts it, and registers it
    for status tracking. The subscriber polls PullMessages at the
    configured interval and fires events into the alert system.
    """
    cfg = _get_cfg(request)

    if not cfg.onvif.events_enabled:
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": "ONVIF events are disabled in config"},
        )

    # Check camera exists in config
    camera_cfg = None
    for cam in cfg.cameras:
        if cam.name == name:
            camera_cfg = cam
            break

    if camera_cfg is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"Camera {name!r} not found"},
        )

    # Check if already subscribed
    from ...onvif.events import get_active_subscribers

    if name in get_active_subscribers():
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": f"Already subscribed for camera {name!r}"},
        )

    try:
        xaddr = _get_camera_xaddr(cfg, name)
        client = OnvifClient(
            device_xaddr=xaddr,
            username=cfg.onvif.username,
            password=cfg.onvif.password,
            timeout_seconds=cfg.onvif.ptz_timeout_seconds,
        )

        # Determine topics from camera config
        event_types = [e.type for e in camera_cfg.events] if camera_cfg.events else ["all"]
        topic_map = {
            "motion": ["tns1:VideoSource/MotionAlarm"],
            "tamper": ["tns1:VideoSource/ImagingAlarm"],
            "all": [
                "tns1:VideoSource/MotionAlarm",
                "tns1:VideoSource/ImagingAlarm",
                "tns1:RuleEngine",
            ],
        }
        topics: list[str] = []
        for et in event_types:
            topics.extend(topic_map.get(et, []))

        async def _on_event(event: OnvifEvent) -> None:
            """Default event callback -- logs the event."""
            log.info(
                "ONVIF event from %s: %s (%s)",
                event.camera_name,
                event.event_type.value,
                event.raw_topic,
            )

        subscriber = OnvifEventSubscriber(
            client=client,
            camera_name=name,
            topics=topics,
            callback=_on_event,
            poll_interval_seconds=float(cfg.onvif.events_poll_interval_seconds),
        )

        await subscriber.start()
        register_subscriber(name, subscriber)

        return JSONResponse(
            content={
                "success": True,
                "subscription_ref": subscriber.subscription_ref,
                "camera_name": name,
            }
        )
    except OnvifError as exc:
        return JSONResponse(content={"success": False, "error": str(exc)})
    except Exception as exc:
        log.exception("Failed to subscribe to events for camera %s", name)
        return JSONResponse(content={"success": False, "error": f"Subscription failed: {exc}"})


@router.post("/cameras/{name}/events/unsubscribe")
async def onvif_events_unsubscribe(
    request: Request, name: str, user=Depends(require_admin)
) -> JSONResponse:
    """Stop an ONVIF event subscription for a camera."""
    from ...onvif.events import get_active_subscribers

    subscribers = get_active_subscribers()
    if name not in subscribers:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"No active subscription for camera {name!r}"},
        )

    subscriber = subscribers[name]
    try:
        await subscriber.stop()
        unregister_subscriber(name)
        return JSONResponse(content={"success": True, "camera_name": name})
    except OnvifError as exc:
        return JSONResponse(content={"success": False, "error": str(exc)})
    except Exception as exc:
        log.exception("Failed to unsubscribe events for camera %s", name)
        return JSONResponse(content={"success": False, "error": f"Unsubscribe failed: {exc}"})
