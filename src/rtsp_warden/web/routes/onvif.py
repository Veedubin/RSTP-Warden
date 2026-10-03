"""ONVIF web routes (admin only).

One server-rendered page (``GET /onvif?camera=<name>``) plus htmx fragments for
discovery, PTZ moves, PTZ presets and event subscriptions. Every POST reads form
fields. A request sent by htmx (``HX-Request: true``) gets back the HTML fragment of
the panel it changed; a plain form post gets a 303 back to the page with a flash
message. Input errors answer 422 and unknown cameras or presets answer 404, both
with an HTML fragment (``static/js/warden.js`` swaps 4xx HTML fragments).
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from markupsafe import escape

from ...config import AppConfig, CameraConfig
from ...onvif.discovery import DiscoveredCamera, OnvifDiscovery, OnvifError
from ...onvif.events import (
    OnvifEvent,
    OnvifEventSubscriber,
    get_active_subscribers,
    get_subscription_states,
    register_subscriber,
    unregister_subscriber,
)
from ...onvif.presets import PTZPresetError, PTZPresetStore
from ...onvif.ptz import OnvifClient, OnvifPTZ
from ..auth_depends import require_admin
from ..flash import FlashLevel
from ._common import (
    find_camera,
    get_cfg,
    get_config_path,
    is_htmx,
    set_flash,
    templates,
)

router = APIRouter(prefix="/onvif", tags=["onvif"])


log = logging.getLogger(__name__)

# PTZ direction to velocity mapping ("stop" is handled separately)
PTZ_ACTIONS: dict[str, dict[str, float]] = {
    "left": {"pan": -1.0, "tilt": 0.0, "zoom": 0.0},
    "right": {"pan": 1.0, "tilt": 0.0, "zoom": 0.0},
    "up": {"pan": 0.0, "tilt": 1.0, "zoom": 0.0},
    "down": {"pan": 0.0, "tilt": -1.0, "zoom": 0.0},
    "zoom_in": {"pan": 0.0, "tilt": 0.0, "zoom": 1.0},
    "zoom_out": {"pan": 0.0, "tilt": 0.0, "zoom": -1.0},
}

# Choices offered by the PTZ pad; 0 means "keep moving until Stop".
PTZ_DURATION_CHOICES: tuple[int, ...] = (250, 500, 1000, 2000, 0)
MAX_PTZ_DURATION_MS = 10_000

# Camera event type (OnvifEventConfig.type) to ONVIF topic expressions
EVENT_TOPICS: dict[str, list[str]] = {
    "motion": ["tns1:VideoSource/MotionAlarm"],
    "tamper": ["tns1:VideoSource/ImagingAlarm"],
    "all": [
        "tns1:VideoSource/MotionAlarm",
        "tns1:VideoSource/ImagingAlarm",
        "tns1:RuleEngine",
    ],
}

PTZ_OFF = "PTZ is off. Set onvif.ptz_enabled: true in config.yaml to use it."
EVENTS_OFF = "ONVIF events are off. Set onvif.events_enabled: true in config.yaml to use them."
DISCOVERY_OFF = "Discovery is off. Set onvif.discovery_enabled: true in config.yaml to use it."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page_url(camera_name: str | None = None) -> str:
    """URL of the ONVIF page, with a camera selected when one is given."""
    if not camera_name:
        return "/onvif"
    return "/onvif?" + urlencode({"camera": camera_name})


def _respond(
    request: Request,
    template_name: str,
    context: dict[str, Any],
    *,
    camera_name: str | None,
    flash: str,
    level: FlashLevel,
    status_code: int = 200,
) -> Response:
    """Fragment for htmx; 303 back to the ONVIF page with a flash message otherwise."""
    if is_htmx(request):
        return templates.TemplateResponse(request, template_name, context, status_code=status_code)
    response = RedirectResponse(url=_page_url(camera_name), status_code=303)
    if flash:
        set_flash(response, flash, level)
    return response


def _camera_missing(request: Request, name: str) -> Response:
    """404 HTML fragment for htmx, or a 303 to the ONVIF page with an error flash."""
    message = f"Camera {name!r} is not in config.yaml."
    if is_htmx(request):
        return HTMLResponse(
            f'<p class="flash flash-error" role="alert">{escape(message)}</p>',
            status_code=404,
        )
    response = RedirectResponse(url=_page_url(), status_code=303)
    set_flash(response, message, "error")
    return response


def _derive_onvif_xaddr(camera: CameraConfig) -> str:
    """ONVIF device service URL: the main_url host on ``camera.onvif_port`` (default 80).

    Raises:
        OnvifError: The camera's main_url has no host.
    """
    try:
        host = urlsplit(camera.main_url).hostname
    except ValueError:
        host = None
    if not host:
        raise OnvifError(f"Camera {camera.name!r} has no host in its main_url")
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    port = camera.onvif_port or 80
    netloc = host if port == 80 else f"{host}:{port}"
    return f"http://{netloc}/onvif/device_service"


def _ptz_client(cfg: AppConfig, camera: CameraConfig) -> OnvifPTZ:
    """Build the PTZ client for one camera. Tests replace this to inject a transport."""
    return OnvifPTZ(
        device_xaddr=_derive_onvif_xaddr(camera),
        username=cfg.onvif.username,
        password=cfg.onvif.password,
        timeout_seconds=cfg.onvif.ptz_timeout_seconds,
    )


async def _log_onvif_event(event: OnvifEvent) -> None:
    """Subscription callback: received ONVIF events are logged, nothing else."""
    log.info(
        "ONVIF event from %s: %s (%s)",
        event.camera_name,
        event.event_type.value,
        event.raw_topic,
    )


def _event_client(cfg: AppConfig, camera: CameraConfig) -> OnvifEventSubscriber:
    """Build the PullPoint subscriber for one camera. Tests replace this."""
    client = OnvifClient(
        device_xaddr=_derive_onvif_xaddr(camera),
        username=cfg.onvif.username,
        password=cfg.onvif.password,
        timeout_seconds=cfg.onvif.ptz_timeout_seconds,
    )
    event_types = [e.type for e in camera.events] if camera.events else ["all"]
    topics: list[str] = []
    for event_type in event_types:
        for topic in EVENT_TOPICS.get(event_type, []):
            if topic not in topics:
                topics.append(topic)
    return OnvifEventSubscriber(
        client=client,
        camera_name=camera.name,
        topics=topics,
        callback=_log_onvif_event,
        poll_interval_seconds=float(cfg.onvif.events_poll_interval_seconds),
    )


def _parse_position(label: str, raw: str, low: float, high: float) -> float:
    """Parse one preset coordinate from a form field.

    Raises:
        ValueError: With a user-facing message when the value is not a number in range.
    """
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}.")
    return value


def _write_error(config_path: Path | None, exc: Exception) -> str:
    """User-facing reason a config.yaml write failed: the OS error text or the message."""
    reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
    return f"Could not write {config_path}: {reason}."


def _ptz_context(
    cfg: AppConfig,
    camera: CameraConfig,
    *,
    duration_ms: int = 500,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Template context for ``partials/onvif_ptz.html``."""
    try:
        xaddr: str | None = _derive_onvif_xaddr(camera)
    except OnvifError as exc:
        xaddr = None
        error = error or str(exc)
    return {
        "camera": camera,
        "xaddr": xaddr,
        "ptz_enabled": cfg.onvif.ptz_enabled,
        "duration_ms": duration_ms,
        "duration_choices": PTZ_DURATION_CHOICES,
        "ptz_message": message,
        "ptz_error": error,
    }


def _presets_context(
    cfg: AppConfig,
    camera: CameraConfig,
    *,
    message: str | None = None,
    error: str | None = None,
    preset_form: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Template context for ``partials/onvif_presets.html``."""
    return {
        "camera": camera,
        "presets": PTZPresetStore(cfg).list_presets(camera.name),
        "ptz_enabled": cfg.onvif.ptz_enabled,
        "preset_message": message,
        "preset_error": error,
        "preset_form": preset_form,
    }


def _subscription_rows(cfg: AppConfig) -> list[dict[str, Any]]:
    """One row per configured camera, plus registry entries for removed cameras."""
    states = {s["camera_name"]: s for s in get_subscription_states()}
    rows: list[dict[str, Any]] = []
    for cam in cfg.cameras:
        state = states.pop(cam.name, None)
        rows.append(
            {
                "camera_name": cam.name,
                "subscribed": state is not None,
                "is_running": bool(state and state["is_running"]),
                "last_event_time": state["last_event_time"] if state else None,
            }
        )
    for name, state in states.items():
        rows.append(
            {
                "camera_name": name,
                "subscribed": True,
                "is_running": bool(state["is_running"]),
                "last_event_time": state["last_event_time"],
            }
        )
    return rows


def _subscriptions_context(
    cfg: AppConfig, *, message: str | None = None, error: str | None = None
) -> dict[str, Any]:
    """Template context for ``partials/onvif_subscriptions.html``."""
    return {
        "rows": _subscription_rows(cfg),
        "events_enabled": cfg.onvif.events_enabled,
        "sub_message": message,
        "sub_error": error,
    }


# ---------------------------------------------------------------------------
# Page and discovery
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def onvif_index(
    request: Request, camera: str = "", user=Depends(require_admin)
) -> HTMLResponse:
    """ONVIF page: discovery, the selected camera's PTZ pad and presets, subscriptions."""
    cfg = get_cfg(request)
    selected = find_camera(cfg, camera) if camera else None
    context: dict[str, Any] = {"cfg": cfg, "camera": selected, "camera_error": None}
    if camera and selected is None:
        context["camera_error"] = f"Camera {camera!r} is not in config.yaml."
    if selected is not None:
        context.update(_ptz_context(cfg, selected))
        context.update(_presets_context(cfg, selected))
    context.update(_subscriptions_context(cfg))
    return templates.TemplateResponse(request, "onvif/index.html", context)


@router.post("/discover", response_class=HTMLResponse)
async def onvif_discover(request: Request, user=Depends(require_admin)) -> Response:
    """Run WS-Discovery and return the results as an HTML fragment."""
    cfg = get_cfg(request)
    found: list[DiscoveredCamera] = []
    error: str | None = None
    if not cfg.onvif.discovery_enabled:
        error = DISCOVERY_OFF
    else:
        discovery = OnvifDiscovery(timeout_seconds=cfg.onvif.discovery_timeout_seconds)
        try:
            # discover() blocks in a select loop; keep it off the event loop that
            # also runs the ONVIF subscription poll tasks.
            found = await asyncio.to_thread(discovery.discover)
        except OnvifError as exc:
            error = f"Discovery failed: {exc}"
        except Exception:
            log.exception("Unexpected error during ONVIF discovery")
            error = "Discovery failed; see the server log."

    flash = "No ONVIF cameras answered."
    level: FlashLevel = "info"
    if error:
        flash = error
        level = "error"
    elif found:
        flash = "Found: " + ", ".join(c.address for c in found)
        level = "success"
    return _respond(
        request,
        "partials/onvif_discover.html",
        {"found": found, "discover_error": error},
        camera_name=None,
        flash=flash,
        level=level,
    )


# ---------------------------------------------------------------------------
# PTZ moves
# ---------------------------------------------------------------------------


@router.post("/cameras/{name}/ptz", response_class=HTMLResponse)
async def onvif_ptz(
    request: Request,
    name: str,
    direction: str = Form(""),
    duration_ms: str = Form("500"),
    user=Depends(require_admin),
) -> Response:
    """Move, zoom or stop a camera from the PTZ pad.

    Form: ``direction`` (left, right, up, down, zoom_in, zoom_out, stop) and
    ``duration_ms`` (0 to 10000; 0 keeps moving until Stop).
    """
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    if camera is None:
        return _camera_missing(request, name)

    def reply(
        status_code: int = 200,
        *,
        duration: int = 500,
        message: str | None = None,
        error: str | None = None,
    ) -> Response:
        context = _ptz_context(cfg, camera, duration_ms=duration, message=message, error=error)
        return _respond(
            request,
            "partials/onvif_ptz.html",
            context,
            camera_name=name,
            flash=error or message or "",
            level="error" if error else "success",
            status_code=status_code,
        )

    if direction != "stop" and direction not in PTZ_ACTIONS:
        return reply(422, error=f"Unknown PTZ direction {direction!r}.")
    try:
        duration = int(duration_ms.strip())
    except ValueError:
        duration = -1
    if not 0 <= duration <= MAX_PTZ_DURATION_MS:
        return reply(
            422,
            error=f"Duration must be a whole number of milliseconds from 0 to "
            f"{MAX_PTZ_DURATION_MS}.",
        )
    if not cfg.onvif.ptz_enabled:
        return reply(duration=duration, error=PTZ_OFF)

    label = direction.replace("_", " ")
    try:
        ptz = _ptz_client(cfg, camera)
        if direction == "stop":
            await ptz.stop()
            message = "Stopped."
        else:
            await ptz.continuous_move(**PTZ_ACTIONS[direction])
            if duration > 0:
                await asyncio.sleep(duration / 1000.0)
                await ptz.stop()
                message = f"Moved {label} for {duration} ms."
            else:
                message = f"Moving {label}. Press Stop to halt."
    except OnvifError as exc:
        return reply(duration=duration, error=f"PTZ failed: {exc}")
    except Exception:
        log.exception("Unexpected PTZ error for camera %s direction %s", name, direction)
        return reply(duration=duration, error="PTZ failed; see the server log.")
    return reply(duration=duration, message=message)


# ---------------------------------------------------------------------------
# PTZ presets
# ---------------------------------------------------------------------------


@router.get("/cameras/{name}/ptz")
async def onvif_ptz_page(request: Request, name: str, user=Depends(require_admin)) -> Response:
    """Old per-camera PTZ page; the ONVIF page now shows the selected camera."""
    return RedirectResponse(url=_page_url(name), status_code=303)


@router.get("/cameras/{name}/presets", response_class=HTMLResponse)
async def onvif_list_presets(request: Request, name: str, user=Depends(require_admin)) -> Response:
    """Presets panel for one camera as an HTML fragment."""
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    if camera is None:
        return _camera_missing(request, name)
    return templates.TemplateResponse(
        request, "partials/onvif_presets.html", _presets_context(cfg, camera)
    )


@router.post("/cameras/{name}/ptz/goto", response_class=HTMLResponse)
async def onvif_goto_preset(
    request: Request,
    name: str,
    preset_name: str = Form(""),
    user=Depends(require_admin),
) -> Response:
    """Move a camera to a saved preset. Form: ``preset_name``."""
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    if camera is None:
        return _camera_missing(request, name)

    store = PTZPresetStore(cfg)
    status_code = 200
    message: str | None = None
    error: str | None = None
    if store.get_preset(name, preset_name) is None:
        error = f"Preset {preset_name!r} not found for camera {name!r}."
        status_code = 404
    elif not cfg.onvif.ptz_enabled:
        error = PTZ_OFF
    else:
        try:
            await store.goto_preset(name, preset_name, _ptz_client(cfg, camera))
            message = f"Moved to preset {preset_name!r}."
        except (OnvifError, PTZPresetError) as exc:
            error = f"PTZ failed: {exc}"
        except Exception:
            log.exception("PTZ goto preset error for camera %s preset %s", name, preset_name)
            error = "PTZ failed; see the server log."

    return _respond(
        request,
        "partials/onvif_presets.html",
        _presets_context(cfg, camera, message=message, error=error),
        camera_name=name,
        flash=error or message or "",
        level="error" if error else "success",
        status_code=status_code,
    )


@router.post("/cameras/{name}/ptz/save", response_class=HTMLResponse)
async def onvif_save_preset(
    request: Request,
    name: str,
    preset_name: str = Form(""),
    pan: str = Form("0"),
    tilt: str = Form("0"),
    zoom: str = Form("0"),
    user=Depends(require_admin),
) -> Response:
    """Add or overwrite a preset with typed pan/tilt/zoom values.

    Form: ``preset_name``, ``pan`` and ``tilt`` (-1 to 1), ``zoom`` (0 to 1).
    Written to config.yaml through the locked raw-YAML patch in ``PTZPresetStore``.
    """
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    if camera is None:
        return _camera_missing(request, name)

    config_path = get_config_path(request)
    preset_form = {"preset_name": preset_name, "pan": pan, "tilt": tilt, "zoom": zoom}

    def reply(
        status_code: int = 200, *, message: str | None = None, error: str | None = None
    ) -> Response:
        context = _presets_context(
            cfg,
            camera,
            message=message,
            error=error,
            preset_form=preset_form if error else None,
        )
        return _respond(
            request,
            "partials/onvif_presets.html",
            context,
            camera_name=name,
            flash=error or message or "",
            level="error" if error else "success",
            status_code=status_code,
        )

    try:
        pan_value = _parse_position("Pan", pan, -1.0, 1.0)
        tilt_value = _parse_position("Tilt", tilt, -1.0, 1.0)
        zoom_value = _parse_position("Zoom", zoom, 0.0, 1.0)
    except ValueError as exc:
        return reply(422, error=str(exc))

    store = PTZPresetStore(cfg, config_path=config_path)
    try:
        saved = await store.save_preset(name, preset_name, pan_value, tilt_value, zoom_value)
    except PTZPresetError as exc:
        return reply(422, error=str(exc))
    except KeyError:
        return reply(error=f"Camera {name!r} is not in {config_path}; the preset was not saved.")
    except (OSError, ValueError) as exc:
        return reply(error=_write_error(config_path, exc) + " The preset was not saved.")

    message = f"Saved preset {saved.name!r}."
    if config_path is None:
        message += " It is kept in memory only until the next restart."
    return reply(message=message)


@router.post("/cameras/{name}/ptz/delete", response_class=HTMLResponse)
async def onvif_delete_preset(
    request: Request,
    name: str,
    preset_name: str = Form(""),
    user=Depends(require_admin),
) -> Response:
    """Delete a preset. Form: ``preset_name`` (in the body, so any name works)."""
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    if camera is None:
        return _camera_missing(request, name)

    config_path = get_config_path(request)
    status_code = 200
    message: str | None = None
    error: str | None = None
    try:
        store = PTZPresetStore(cfg, config_path=config_path)
        deleted = await store.delete_preset(name, preset_name)
    except KeyError:
        error = f"Camera {name!r} is not in {config_path}; the preset was not deleted."
    except (OSError, ValueError) as exc:
        error = _write_error(config_path, exc) + " The preset was kept."
    else:
        if deleted:
            message = f"Deleted preset {preset_name!r}."
        else:
            error = f"Preset {preset_name!r} not found for camera {name!r}."
            status_code = 404

    return _respond(
        request,
        "partials/onvif_presets.html",
        _presets_context(cfg, camera, message=message, error=error),
        camera_name=name,
        flash=error or message or "",
        level="error" if error else "success",
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Event subscriptions
# ---------------------------------------------------------------------------


@router.get("/events", response_class=HTMLResponse)
async def onvif_events_status(request: Request, user=Depends(require_admin)) -> HTMLResponse:
    """Subscription table as an HTML fragment."""
    cfg = get_cfg(request)
    return templates.TemplateResponse(
        request, "partials/onvif_subscriptions.html", _subscriptions_context(cfg)
    )


@router.post("/cameras/{name}/events/subscribe", response_class=HTMLResponse)
async def onvif_events_subscribe(
    request: Request, name: str, user=Depends(require_admin)
) -> Response:
    """Start a PullPoint subscription for a camera.

    The subscriber polls PullMessages at ``onvif.events_poll_interval_seconds``.
    Received events are logged only; they do not reach the events table or actions.
    """
    cfg = get_cfg(request)
    camera = find_camera(cfg, name)
    status_code = 200
    message: str | None = None
    error: str | None = None
    if camera is None:
        error = f"Camera {name!r} is not in config.yaml."
        status_code = 404
    elif not cfg.onvif.events_enabled:
        error = EVENTS_OFF
    elif name in get_active_subscribers():
        error = f"Already subscribed to events from {name!r}."
    else:
        try:
            subscriber = _event_client(cfg, camera)
            await subscriber.start()
            register_subscriber(name, subscriber)
            message = f"Subscribed to events from {name!r}."
        except OnvifError as exc:
            error = f"Subscription failed: {exc}"
        except Exception:
            log.exception("Failed to subscribe to events for camera %s", name)
            error = "Subscription failed; see the server log."

    return _respond(
        request,
        "partials/onvif_subscriptions.html",
        _subscriptions_context(cfg, message=message, error=error),
        camera_name=name if camera is not None else None,
        flash=error or message or "",
        level="error" if error else "success",
        status_code=status_code,
    )


@router.post("/cameras/{name}/events/unsubscribe", response_class=HTMLResponse)
async def onvif_events_unsubscribe(
    request: Request, name: str, user=Depends(require_admin)
) -> Response:
    """Stop a camera's event subscription and drop it from the registry."""
    cfg = get_cfg(request)
    status_code = 200
    message: str | None = None
    error: str | None = None
    subscriber = get_active_subscribers().get(name)
    if subscriber is None:
        error = f"No active subscription for {name!r}."
        status_code = 404
    else:
        try:
            await subscriber.stop()
            message = f"Unsubscribed from {name!r} events."
        except OnvifError as exc:
            error = f"Unsubscribe failed: {exc}"
        except Exception:
            log.exception("Failed to unsubscribe events for camera %s", name)
            error = "Unsubscribe failed; see the server log."
        finally:
            unregister_subscriber(name)

    return _respond(
        request,
        "partials/onvif_subscriptions.html",
        _subscriptions_context(cfg, message=message, error=error),
        camera_name=name if find_camera(cfg, name) is not None else None,
        flash=error or message or "",
        level="error" if error else "success",
        status_code=status_code,
    )
