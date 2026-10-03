"""Camera settings page (RW-4): the camera's own HTTP API, Foscam CGI for now. Admin only.

``GET /cameras/{name}/vendor`` renders device info, the main and sub stream profiles,
image tuning, mirror / flip / infrared / OSD and a snapshot, each read live from the camera.
Every section is a form that posts back; an htmx request gets the section's fragment, a
plain post gets a 303 to the page with a flash message. The camera's answers and failures
are shown as text, never as a 500. Nothing here ever prints the URL with its credentials.

The page is enabled per camera with a ``vendor:`` block in config.yaml (type and port);
the enable / disable forms patch only that key of the raw YAML entry, so ``${VAR}``
references survive. Credentials come from ``main_url`` (the same account as RTSP).
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from markupsafe import escape
from starlette.concurrency import run_in_threadpool

from ...config import AppConfig, CameraConfig, VendorConfig
from ...vendors.foscam import (
    IMAGE_COMMANDS,
    STREAMS,
    FoscamClient,
    FoscamError,
    StreamProfile,
    resolution_label,
)
from ..auth_depends import CurrentUser, require_admin
from ..flash import FlashLevel
from ..services import camera_config
from ..services.camera_edit import split_userinfo
from ..services.detection import write_failed_message
from ._common import find_camera, get_cfg, get_config_path, is_htmx, set_flash, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras", tags=["vendor"])

VENDOR_TYPES: tuple[str, ...] = ("foscam",)
# Bit rates offered by the profile editor (bits per second, as the camera counts them).
BIT_RATE_CHOICES: tuple[int, ...] = (131072, 262144, 524288, 1048576, 2097152, 3145728, 4194304)
FRAME_RATE_MAX = 30
GOP_MAX = 200
RESOLUTION_CODE_MAX = 9
IMAGE_FIELDS: tuple[str, ...] = tuple(IMAGE_COMMANDS)


# ---------------------------------------------------------------------------
# Client factory and helpers
# ---------------------------------------------------------------------------


def _client(cfg: AppConfig, camera: CameraConfig) -> FoscamClient:
    """The camera's CGI client. Tests replace this to inject a fake.

    Raises FoscamError when the camera has no ``vendor`` block, no host or no credentials
    in its ``main_url``.
    """
    vendor = camera.vendor
    if vendor is None:
        raise FoscamError(
            f"Camera {camera.name!r} has no vendor block in config.yaml; enable it below."
        )
    userinfo, rest = split_userinfo(camera.main_url)
    try:
        host = urlsplit(rest).hostname
    except ValueError:
        host = None
    if not host:
        raise FoscamError(f"Camera {camera.name!r} has no host in its main_url.")
    username, _sep, password = userinfo.partition(":")
    if not username:
        raise FoscamError(
            f"Camera {camera.name!r} has no credentials in its main_url; the camera API "
            "needs the camera's user name and password."
        )
    return FoscamClient(
        host, port=vendor.port, username=unquote(username), password=unquote(password)
    )


def _page_url(name: str) -> str:
    return f"/cameras/{name}/vendor"


def _camera_or_404(cfg: AppConfig, name: str) -> CameraConfig:
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} is not in config.yaml.")
    return cam


def _kbit(bit_rate: int) -> str:
    return f"{int(bit_rate) // 1024} kbit/s"


def _checked(form: Any, key: str) -> bool:
    return str(form.get(key, "")).strip().lower() in ("on", "1", "true", "yes")


def _form_int(form: Any, key: str, low: int, high: int, *, required: bool = True) -> int | None:
    raw = str(form.get(key, "")).strip()
    if not raw:
        if required:
            raise HTTPException(status_code=422, detail=f"{key} is required")
        return None
    try:
        value = int(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{key} must be a whole number") from None
    if not low <= value <= high:
        raise HTTPException(status_code=422, detail=f"{key} must be between {low} and {high}")
    return value


def _respond(
    request: Request,
    template_name: str,
    context: dict[str, Any],
    *,
    camera_name: str,
    flash: str,
    level: FlashLevel,
) -> Response:
    """Fragment for htmx; 303 back to the settings page with a flash otherwise."""
    if is_htmx(request):
        return templates.TemplateResponse(request, template_name, context)
    response = RedirectResponse(url=_page_url(camera_name), status_code=303)
    if flash:
        set_flash(response, flash, level)
    return response


# ---------------------------------------------------------------------------
# Section contexts (each one reads from the camera; errors become text)
# ---------------------------------------------------------------------------


def _stream_context(
    client: FoscamClient | None, *, message: str | None = None, error: str | None = None
) -> dict[str, Any]:
    streams: list[dict[str, Any]] = []
    if client is not None:
        for name in STREAMS:
            try:
                active = client.stream_type(name)
                profiles = client.stream_profiles(name)
            except FoscamError as exc:
                error = error or str(exc)
                continue
            current = next((p for p in profiles if p.index == active), None)
            choices = sorted(set(BIT_RATE_CHOICES) | ({current.bit_rate} if current else set()))
            streams.append(
                {
                    "name": name,
                    "active": active,
                    "current": current,
                    "profiles": [
                        {
                            "index": p.index,
                            "resolution": p.resolution,
                            "resolution_label": resolution_label(p.resolution),
                            "bit_rate": p.bit_rate,
                            "kbit": _kbit(p.bit_rate),
                            "frame_rate": p.frame_rate,
                            "gop": p.gop,
                            "vbr": p.vbr,
                        }
                        for p in profiles
                    ],
                    "bit_rates": [(b, _kbit(b)) for b in choices],
                }
            )
    return {
        "streams": streams,
        "message": message,
        "error": error,
        "frame_rate_max": FRAME_RATE_MAX,
    }


def _image_context(
    client: FoscamClient | None, *, message: str | None = None, error: str | None = None
) -> dict[str, Any]:
    image = None
    if client is not None:
        try:
            image = client.image_settings()
        except FoscamError as exc:
            error = error or str(exc)
    return {"image": image, "fields": IMAGE_FIELDS, "message": message, "error": error}


def _video_context(
    client: FoscamClient | None, *, message: str | None = None, error: str | None = None
) -> dict[str, Any]:
    video = None
    if client is not None:
        try:
            video = client.video_settings()
        except FoscamError as exc:
            error = error or str(exc)
    return {"video": video, "message": message, "error": error}


def _page_context(request: Request, cfg: AppConfig, cam: CameraConfig) -> dict[str, Any]:
    context: dict[str, Any] = {
        "camera": cam,
        "vendor": cam.vendor,
        "vendor_types": VENDOR_TYPES,
        "page_error": None,
        "device": None,
        "device_error": None,
        "stream": _stream_context(None),
        "image": _image_context(None),
        "video": _video_context(None),
    }
    if cam.vendor is None:
        return context
    try:
        client = _client(cfg, cam)
    except FoscamError as exc:
        context["page_error"] = str(exc)
        return context
    try:
        context["device"] = client.device_info()
    except FoscamError as exc:
        context["device_error"] = str(exc)
    context["stream"] = _stream_context(client)
    context["image"] = _image_context(client)
    context["video"] = _video_context(client)
    return context


def _section_context(cfg: AppConfig, cam: CameraConfig, section: str, work: Any) -> dict[str, Any]:
    """Run ``work(client)`` (returns the success message), then re-read the section.

    A FoscamError from the factory or from ``work`` becomes the section's ``error``.
    """
    builders = {"stream": _stream_context, "image": _image_context, "video": _video_context}
    build = builders[section]
    client: FoscamClient | None = None
    message: str | None = None
    error: str | None = None
    try:
        client = _client(cfg, cam)
        message = work(client)
    except FoscamError as exc:
        error = str(exc)
    return {"camera": cam, section: build(client, message=message, error=error)}


# ---------------------------------------------------------------------------
# Page, enable, disable
# ---------------------------------------------------------------------------


@router.get("/{name}/vendor", response_class=HTMLResponse)
async def vendor_page(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    context = await run_in_threadpool(_page_context, request, cfg, cam)
    return templates.TemplateResponse(request, "cameras/vendor.html", context)


@router.post("/{name}/vendor/enable", response_model=None)
async def vendor_enable(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    form = await request.form()
    vendor_type = str(form.get("vendor_type", "")).strip().lower()
    if vendor_type not in VENDOR_TYPES:
        raise HTTPException(status_code=422, detail=f"vendor_type must be one of {VENDOR_TYPES}")
    port = _form_int(form, "port", 1, 65535)
    assert port is not None
    vendor = VendorConfig(type=vendor_type, port=port)  # type: ignore[arg-type]
    config_path = get_config_path(request)
    response = RedirectResponse(url=_page_url(name), status_code=303)
    if config_path is not None:
        try:
            await run_in_threadpool(
                camera_config.patch_camera,
                config_path,
                name,
                {"vendor": {"type": vendor.type, "port": vendor.port}},
            )
        except KeyError:
            raise HTTPException(
                status_code=409, detail=f"Camera {name!r} is no longer in {config_path}."
            ) from None
        except OSError as exc:
            set_flash(response, write_failed_message(config_path, exc), "error")
            cam.vendor = vendor
            return response
    cam.vendor = vendor
    set_flash(response, f"Camera API enabled ({vendor.type}, port {vendor.port}).", "success")
    return response


@router.post("/{name}/vendor/disable", response_model=None)
async def vendor_disable(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    config_path = get_config_path(request)
    response = RedirectResponse(url=f"/cameras/{name}", status_code=303)
    if config_path is not None:
        try:
            await run_in_threadpool(
                camera_config.patch_camera, config_path, name, {}, remove_keys=("vendor",)
            )
        except KeyError:
            raise HTTPException(
                status_code=409, detail=f"Camera {name!r} is no longer in {config_path}."
            ) from None
        except OSError as exc:
            set_flash(response, write_failed_message(config_path, exc), "error")
            cam.vendor = None
            return response
    cam.vendor = None
    set_flash(response, "Camera API disabled for this camera.", "success")
    return response


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


@router.post("/{name}/vendor/stream", response_model=None)
async def vendor_stream(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    form = await request.form()
    stream = str(form.get("stream", "")).strip().lower()
    if stream not in STREAMS:
        raise HTTPException(status_code=422, detail=f"stream must be one of {STREAMS}")
    action = str(form.get("action", "")).strip().lower()
    if action not in ("use", "save"):
        raise HTTPException(status_code=422, detail="action must be 'use' or 'save'")
    index = _form_int(form, "profile", 0, 3)
    assert index is not None
    profile: StreamProfile | None = None
    if action == "save":
        profile = StreamProfile(
            index=index,
            resolution=_form_int(form, "resolution", 0, RESOLUTION_CODE_MAX) or 0,
            bit_rate=_form_int(form, "bit_rate", 1, 100_000_000) or 0,
            frame_rate=_form_int(form, "frame_rate", 1, FRAME_RATE_MAX) or 0,
            gop=_form_int(form, "gop", 1, GOP_MAX) or 0,
            vbr=_checked(form, "vbr"),
        )

    def work(client: FoscamClient) -> str:
        if action == "use":
            client.set_stream_type(stream, index)
            return f"The {stream} stream now plays profile {index}."
        assert profile is not None
        client.set_stream_profile(stream, profile)
        return f"Profile {index} saved for the {stream} stream."

    context = await run_in_threadpool(_section_context, cfg, cam, "stream", work)
    section = context["stream"]
    return _respond(
        request,
        "partials/vendor_stream.html",
        context,
        camera_name=name,
        flash=section["error"] or section["message"] or "",
        level="error" if section["error"] else "success",
    )


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


@router.post("/{name}/vendor/image", response_model=None)
async def vendor_image(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    form = await request.form()
    wanted = {
        field: value
        for field in IMAGE_FIELDS
        if (value := _form_int(form, field, 0, 100, required=False)) is not None
    }

    def work(client: FoscamClient) -> str:
        current = client.image_settings()
        changed = [(f, v) for f, v in wanted.items() if getattr(current, f) != v]
        for field, value in changed:
            client.set_image_setting(field, value)
        return "Image settings saved." if changed else "No image changes to save."

    context = await run_in_threadpool(_section_context, cfg, cam, "image", work)
    section = context["image"]
    return _respond(
        request,
        "partials/vendor_image.html",
        context,
        camera_name=name,
        flash=section["error"] or section["message"] or "",
        level="error" if section["error"] else "success",
    )


# ---------------------------------------------------------------------------
# Video: mirror, flip, infrared, OSD
# ---------------------------------------------------------------------------


@router.post("/{name}/vendor/video", response_model=None)
async def vendor_video(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    form = await request.form()
    mirror = _checked(form, "mirror")
    flip = _checked(form, "flip")
    infrared_mode = _form_int(form, "infrared_mode", 0, 1)
    infrared_on = _checked(form, "infrared_on")
    osd_timestamp = _checked(form, "osd_timestamp")
    osd_name = _checked(form, "osd_name")
    osd_position = _form_int(form, "osd_position", 0, 3)
    assert infrared_mode is not None and osd_position is not None

    def work(client: FoscamClient) -> str:
        current = client.video_settings()
        if mirror != current.mirror:
            client.set_mirror(mirror)
        if flip != current.flip:
            client.set_flip(flip)
        if infrared_mode != current.infrared_mode:
            client.set_infrared_mode(infrared_mode)
        if infrared_mode == 1:  # manual: the LED state cannot be read back, so it is sent
            client.set_infrared(infrared_on)
        if (osd_timestamp, osd_name, osd_position) != (
            current.osd_timestamp,
            current.osd_name,
            current.osd_position,
        ):
            client.set_osd(
                timestamp=osd_timestamp,
                name=osd_name,
                position=osd_position,
                temp_humid=current.osd_temp_humid,
                mask=current.osd_mask,
            )
        return "Video settings saved."

    context = await run_in_threadpool(_section_context, cfg, cam, "video", work)
    section = context["video"]
    return _respond(
        request,
        "partials/vendor_video.html",
        context,
        camera_name=name,
        flash=section["error"] or section["message"] or "",
        level="error" if section["error"] else "success",
    )


# ---------------------------------------------------------------------------
# Snapshot and reboot
# ---------------------------------------------------------------------------


@router.get("/{name}/vendor/snapshot.jpg")
async def vendor_snapshot(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)

    def fetch() -> bytes:
        return _client(cfg, cam).snapshot()

    try:
        jpeg = await run_in_threadpool(fetch)
    except FoscamError as exc:
        return Response(str(exc), status_code=502, media_type="text/plain")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/{name}/vendor/reboot", response_model=None)
async def vendor_reboot(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)

    def send() -> None:
        _client(cfg, cam).reboot()

    error: str | None = None
    try:
        await run_in_threadpool(send)
    except FoscamError as exc:
        error = str(exc)
    message = (
        error or "Reboot sent. The camera is back in about a minute; its stream restarts by itself."
    )
    if is_htmx(request):
        css = "flash-error" if error else "flash-success"
        return HTMLResponse(f'<p class="flash {css}" role="status">{escape(message)}</p>')
    response = RedirectResponse(url=_page_url(name), status_code=303)
    set_flash(response, message, "error" if error else "success")
    return response
