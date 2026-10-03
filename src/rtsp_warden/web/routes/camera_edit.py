"""Add-camera routes: the form, "Test connection", ONVIF URL fill, and save + hot-add.

Every handler is a plain ``def``: FastAPI runs it in the threadpool, so the ffprobe run,
the ONVIF lookup and the wait for the runtime never block the event loop.

Credentials never reach ``config.yaml``. The saved URLs read
``rtsp://${CAM_<SLUG>_USER}:${CAM_<SLUG>_PASS}@host:port/path`` and the percent-encoded
values go to the ``.env`` file next to ``config.yaml``.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import yaml
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ... import probe
from ...ffmpeg import redact_text
from ...onvif import media as onvif_media
from ...onvif.discovery import OnvifError
from ..auth_depends import CurrentUser, require_admin
from ..env_file import read_env_file, upsert_env_vars
from ..services import camera_config
from ._common import get_cfg, get_config_path, set_flash, templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")

# Seconds the save handler waits for the supervisor to start a hot-added camera.
ADD_TIMEOUT_S = 30.0
# Ports the ONVIF fill tries, in order, and the per-port timeout (4 x 3 s worst case).
ONVIF_PORTS: tuple[int, ...] = (80, 8080, 888, 2020)
ONVIF_TIMEOUT_S = 3.0

# Serialises read-modify-write of config.yaml, .env and cfg.cameras between requests.
_config_edit_lock = threading.Lock()

_HOST_RE = re.compile(r"^(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])$")

_NEW_DEFAULTS: dict[str, object] = {
    "name": "",
    "host": "",
    "username": "",
    "main_url": "",
    "sub_url": "",
    "onvif_port": "",
    "record_enabled": True,
    "password_entered": False,
}


def _render_form(
    request: Request,
    *,
    values: dict[str, object],
    errors: dict[str, str],
    status_code: int = 200,
) -> HTMLResponse:
    """Render cameras/form.html in "new" mode."""
    return templates.TemplateResponse(
        request,
        "cameras/form.html",
        {
            "request": request,
            "mode": "new",
            "title": "Add camera",
            "subtitle": (
                "Saved to config.yaml. The user name and password go to the .env file "
                "next to it, never into config.yaml."
            ),
            "form_action": "/cameras",
            "values": values,
            "errors": errors,
        },
        status_code=status_code,
    )


@router.get("/new", response_class=HTMLResponse)
def new_camera_form(request: Request, user: CurrentUser = Depends(require_admin)) -> HTMLResponse:
    """Render the empty add-camera form."""
    get_cfg(request)
    return _render_form(request, values=dict(_NEW_DEFAULTS), errors={})


def _clean_host(host: str) -> str:
    """Return the stripped host, or raise ValueError with a message for the form."""
    host = host.strip()
    if not host:
        raise ValueError("Enter the camera's IP address or host name.")
    if not _HOST_RE.match(host):
        raise ValueError(
            "Enter only the IP address or host name, without rtsp://, a port or a path."
        )
    return host


def _default_main(host: str) -> tuple[str, str, str]:
    """Stream parts used when the main stream field is empty: rtsp://host:554/."""
    return "rtsp", f"{host}:554", "/"


def _stream_parts(value: str, host: str) -> tuple[str, str, str] | None:
    """Split a stream field into (scheme, netloc without userinfo, path with query).

    The field holds either a path such as ``/videoMain`` (the host comes from the form
    and the port is 554) or an ``rtsp://`` / ``rtsps://`` URL without credentials, which is
    what the ONVIF fill writes. Returns None for an empty field.
    """
    value = value.strip()
    if not value:
        return None
    if "://" not in value:
        if value.startswith("-") or any(ch.isspace() or ch == "#" for ch in value):
            raise ValueError("A stream path cannot start with '-' or contain spaces or '#'.")
        return "rtsp", f"{host}:554", "/" + value.lstrip("/")
    url = probe.validate_rtsp_url(value)
    parts = urlsplit(url)
    if "@" in parts.netloc:
        raise ValueError("Leave the user name and password out of the URL; use the fields above.")
    if not parts.hostname:
        raise ValueError("The stream URL has no host.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("The port in the stream URL must be a number from 1 to 65535.") from None
    if port == 0:
        raise ValueError("The port in the stream URL must be a number from 1 to 65535.")
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return parts.scheme, parts.netloc, path


def _plain_url(parts: tuple[str, str, str]) -> str:
    """The stream URL without credentials, as raw_camera_entry expects it."""
    scheme, netloc, path = parts
    return f"{scheme}://{netloc}{path}"


def _probe_url(parts: tuple[str, str, str], username: str, password: str) -> str:
    """URL handed to ffprobe, shaped like the saved one: percent-encoded credentials.

    Like ``raw_camera_entry``, an empty user name means a camera without credentials.
    """
    if not username:
        return _plain_url(parts)
    scheme, netloc, path = parts
    userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}"
    return f"{scheme}://{userinfo}@{netloc}{path}"


def _failed_probe(message: str) -> probe.ProbeResult:
    return probe.ProbeResult(
        ok=False,
        codec=None,
        width=None,
        height=None,
        fps=None,
        snapshot_jpeg=None,
        error=message,
    )


@router.post("/new/test", response_class=HTMLResponse)
def new_camera_test(
    request: Request,
    host: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    main_url: str = Form(""),
    sub_url: str = Form(""),
    stream: str = Form("main"),
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Probe the main or sub stream with ffprobe and return partials/probe_result.html."""
    cfg = get_cfg(request)
    stream = "sub" if stream == "sub" else "main"
    tested_url = ""
    try:
        host = _clean_host(host)
        parts = _stream_parts(sub_url if stream == "sub" else main_url, host)
        if parts is None:
            if stream == "sub":
                raise ValueError("Enter a sub stream path or URL first.")
            parts = _default_main(host)
        url = probe.validate_rtsp_url(_probe_url(parts, username, password))
        tested_url = redact_text(url)
        result = probe.probe_stream(url, runtime=cfg.runtime)
    except ValueError as exc:
        result = _failed_probe(redact_text(str(exc)))
    except FileNotFoundError:
        result = _failed_probe(
            "ffprobe was not found. Install ffmpeg (it includes ffprobe) "
            "or set runtime.ffprobe_path in config.yaml."
        )
    except OSError as exc:
        result = _failed_probe(f"Could not run ffprobe: {exc.strerror or exc}")
    snapshot_data_uri = ""
    if result.ok and result.snapshot_jpeg:
        encoded = base64.b64encode(result.snapshot_jpeg).decode("ascii")
        snapshot_data_uri = f"data:image/jpeg;base64,{encoded}"
    return templates.TemplateResponse(
        request,
        "partials/probe_result.html",
        {
            "request": request,
            "result": result,
            "stream": stream,
            "tested_url": tested_url,
            "snapshot_data_uri": snapshot_data_uri,
        },
    )


def _parse_port(value: str) -> int | None:
    """Parse an optional port field: blank -> None, else an int in 1..65535."""
    value = value.strip()
    if not value:
        return None
    if not value.isdigit() or not 1 <= int(value) <= 65535:
        raise ValueError("The ONVIF port must be a number from 1 to 65535.")
    return int(value)


@router.post("/new/onvif", response_class=HTMLResponse)
def new_camera_onvif(
    request: Request,
    host: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    onvif_port: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> HTMLResponse:
    """Ask the camera for its stream URIs over ONVIF; fill the form with out-of-band swaps."""
    uris: onvif_media.StreamUris | None = None
    error = ""
    ports_tried = ""
    try:
        host = _clean_host(host)
        port = _parse_port(onvif_port)
        ports = ONVIF_PORTS if port is None else (port,)
        ports_tried = ", ".join(str(p) for p in ports)
        # A fresh event loop per request; discover_stream_uris creates its AsyncClient inside.
        uris = asyncio.run(
            onvif_media.discover_stream_uris(
                host, username, password, ports=ports, timeout_s=ONVIF_TIMEOUT_S
            )
        )
    except ValueError as exc:
        error = redact_text(str(exc))
    except (OnvifError, httpx.HTTPError, OSError) as exc:
        error = redact_text(str(exc)) or type(exc).__name__
    return templates.TemplateResponse(
        request,
        "partials/onvif_fill.html",
        {"request": request, "uris": uris, "error": error, "ports_tried": ports_tried},
    )


class _FieldError(ValueError):
    """A save error shown next to one form field (``form`` for the whole form)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


def _error_text(exc: BaseException) -> str:
    """One line for a flash message: type and message, credentials masked, 300 chars."""
    return redact_text(f"{type(exc).__name__}: {exc}"[:1000])[:300]


def _unreadable_config(config_path: Path) -> str:
    return f"{config_path} could not be read: it is not valid YAML. Fix the file and try again."


def _check_nothing_is_overwritten(
    config_path: Path, name: str, env_values: Mapping[str, str]
) -> None:
    """Raise _FieldError when saving camera ``name`` would overwrite something set by hand.

    ``validate_name`` only knows the cameras loaded at startup: config.yaml may hold a
    camera added by hand since then, and the shared .env (or the service environment)
    may already define this camera's credential variables for a hand-written camera.
    Values equal to the ones about to be written are no conflict (a retried save).
    """
    try:
        on_disk = camera_config.raw_camera_names(config_path)
    except yaml.YAMLError:
        raise _FieldError("form", _unreadable_config(config_path)) from None
    except ValueError as exc:
        raise _FieldError("form", str(exc)) from None
    try:
        camera_config.validate_name(name, on_disk)
    except ValueError as exc:
        raise _FieldError("name", str(exc)) from None
    env_path = config_path.parent / ".env"
    in_file = read_env_file(env_path)
    for key, value in env_values.items():
        if key in in_file and in_file[key] != value:
            raise _FieldError(
                "name",
                f"The .env file next to {config_path.name} already defines {key} with another "
                f"value; choose another camera name or remove the {key} line from {env_path}.",
            )
        current = os.environ.get(key)
        if current is not None and current != value:
            raise _FieldError(
                "name",
                f"rtsp-warden's environment already defines {key} with another value (from "
                "its service settings or a .env read at startup); choose another camera name.",
            )


@router.post("", response_class=HTMLResponse)
def create_camera(
    request: Request,
    name: str = Form(""),
    host: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    main_url: str = Form(""),
    sub_url: str = Form(""),
    onvif_port: str = Form(""),
    record_enabled: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Save a new camera to .env and config.yaml, add it to cfg, and hot-add it."""
    cfg = get_cfg(request)
    values: dict[str, object] = {
        "name": name.strip(),
        "host": host.strip(),
        "username": username,
        "main_url": main_url.strip(),
        "sub_url": sub_url.strip(),
        "onvif_port": onvif_port.strip(),
        "record_enabled": record_enabled == "on",
        "password_entered": bool(password),
    }
    config_path = get_config_path(request)
    if config_path is None:
        return _render_form(
            request,
            values=values,
            errors={
                "form": "This server was started without a config file, so cameras "
                "cannot be saved from the web UI."
            },
            status_code=503,
        )

    with _config_edit_lock:
        errors: dict[str, str] = {}
        try:
            name = camera_config.validate_name(name, [c.name for c in cfg.cameras])
        except ValueError as exc:
            errors["name"] = str(exc)
        main_parts: tuple[str, str, str] | None = None
        sub_parts: tuple[str, str, str] | None = None
        try:
            host = _clean_host(host)
        except ValueError as exc:
            errors["host"] = str(exc)
        else:
            try:
                main_parts = _stream_parts(main_url, host) or _default_main(host)
            except ValueError as exc:
                errors["main_url"] = redact_text(str(exc))
            try:
                sub_parts = _stream_parts(sub_url, host)
            except ValueError as exc:
                errors["sub_url"] = redact_text(str(exc))
        try:
            onvif = _parse_port(onvif_port)
        except ValueError as exc:
            errors["onvif_port"] = str(exc)
        if errors or main_parts is None:
            return _render_form(request, values=values, errors=errors, status_code=422)

        proxy_port = camera_config.allocate_proxy_port(cfg)
        inp = camera_config.NewCameraInput(
            name=name,
            host=host,
            username=username,
            password=password,
            main_url=_plain_url(main_parts),
            sub_url=_plain_url(sub_parts) if sub_parts else None,
            onvif_port=onvif,
            record_enabled=record_enabled == "on",
        )
        try:
            # With a user name, raw_camera_entry puts ${CAM_<SLUG>_USER}:${CAM_<SLUG>_PASS}@
            # into every URL and returns the percent-encoded values; without one, neither.
            entry, env_values = camera_config.raw_camera_entry(inp, proxy_port)
            cam = camera_config.validate_entry(entry, {**os.environ, **env_values})
        except ValueError as exc:
            return _render_form(
                request,
                values=values,
                errors={"form": redact_text(str(exc))},
                status_code=422,
            )

        try:
            _check_nothing_is_overwritten(config_path, name, env_values)
            # .env first: a config.yaml entry whose variables are missing would stop the
            # next start with "the environment variable ... is not set".
            upsert_env_vars(config_path.parent / ".env", env_values)
            camera_config.append_camera(config_path, entry)
        except ValueError as exc:
            field = exc.field if isinstance(exc, _FieldError) else "name"
            return _render_form(request, values=values, errors={field: str(exc)}, status_code=422)
        except yaml.YAMLError:
            return _render_form(
                request,
                values=values,
                errors={"form": _unreadable_config(config_path)},
                status_code=422,
            )
        except OSError as exc:
            log.warning("could not save camera %s: %s", name, exc)
            failed = RedirectResponse("/cameras/new", status_code=303)
            set_flash(
                failed,
                f"Could not save camera {name}: {exc.strerror or exc} "
                f"({exc.filename or config_path}). Make {config_path.parent} writable "
                "by rtsp-warden and try again.",
                "error",
            )
            return failed

        os.environ.update(env_values)
        cfg.cameras = [*cfg.cameras, cam]
        log.info("camera %s added from the web UI (proxy port %d)", name, proxy_port)

    response = RedirectResponse(f"/cameras/{name}", status_code=303)
    runtime = getattr(request.app.state, "runtime", None)
    request_add = getattr(runtime, "request_add_camera", None)
    if request_add is None:
        set_flash(
            response,
            f"Camera {name} saved to {config_path.name}; "
            "it takes effect when rtsp-warden restarts.",
            "info",
        )
        return response
    future: concurrent.futures.Future[None] = request_add(cam)
    try:
        future.result(timeout=ADD_TIMEOUT_S)
    except BaseException as exc:  # SystemExit too: the runtime fails a bad proxy mode with it
        if not future.done():
            # Timed out: the request stays queued or running; never cancel it.
            set_flash(
                response,
                f"Camera {name} saved; the runtime is still starting it. "
                "Its status below updates on its own.",
                "info",
            )
        else:
            set_flash(
                response,
                f"Camera {name} was saved but did not start: {_error_text(exc)}",
                "error",
            )
    else:
        set_flash(response, f"Camera {name} added and started.", "success")
    return response
