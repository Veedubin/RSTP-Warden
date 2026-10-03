"""Camera add, edit and delete routes.

Add: the form, "Test connection", ONVIF URL fill, and save + hot-add. Edit and delete:
the camera detail page's buttons (only changed keys are written; delete keeps files).

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
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit

import httpx
import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from ... import probe
from ...app import CameraNotFoundError, _error_text
from ...config import AppConfig, CameraConfig
from ...ffmpeg import redact_text
from ...onvif import media as onvif_media
from ...onvif.discovery import OnvifError
from ...onvif.events import get_active_subscribers, unregister_subscriber
from ..auth_depends import CurrentUser, require_admin
from ..env_file import read_env_file, remove_env_vars, upsert_env_vars
from ..services import camera_config
from ..services.camera_edit import (
    CameraEditError,
    EditCameraInput,
    apply_plan,
    apply_to_running_config,
    edit_form_values,
    plan_camera_edit,
    read_camera_entry,
    split_userinfo,
)
from ._common import find_camera, get_cfg, get_config_path, set_flash, templates

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


# --- Edit and delete from the camera detail page (RW-2 Task 8) ---------------------------

# Seconds edit and delete wait for the supervisor to restart or remove the camera.
# Stopping one ingest alone can take ~10 s; after this the request goes on in the background.
RUNTIME_REQUEST_TIMEOUT_S = 30.0

_NO_CONFIG_FILE = (
    "This server was started without a config file, so cameras cannot be edited "
    "or deleted from the web UI."
)


def _redirect_flash(
    url: str, message: str, level: Literal["info", "success", "error"] = "info"
) -> RedirectResponse:
    """303 to ``url`` with a one-shot flash message for the next page."""
    response = RedirectResponse(url, status_code=303)
    set_flash(response, message, level)
    return response


def _render_edit_form(
    request: Request,
    cam: CameraConfig,
    *,
    values: dict[str, object],
    errors: dict[str, str],
    status_code: int = 200,
) -> HTMLResponse:
    """Render cameras/form.html in "edit" mode."""
    return templates.TemplateResponse(
        request,
        "cameras/form.html",
        {
            "request": request,
            "mode": "edit",
            "title": f"Edit camera {cam.name}",
            "subtitle": (
                "Saved to config.yaml. A new user name or password goes to the .env file "
                "next to it, never into config.yaml."
            ),
            "form_action": f"/cameras/{cam.name}/edit",
            "values": values,
            "errors": errors,
        },
        status_code=status_code,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeOutcome:
    """How a queued runtime request ended, as far as the route waited for it.

    ``absent``: no runtime is attached, or it does not run this camera (the change
    applies on the next start); ``done``: applied; ``pending``: still queued or running
    when the wait ended (it goes on in the background, never cancelled); ``failed``:
    the request failed and ``error`` says why (credentials masked).
    """

    state: Literal["absent", "done", "pending", "failed"]
    error: str = ""


def _request_runtime(request: Request, method: str, name: str) -> Future[None] | None:
    """Queue ``app.state.runtime.<method>(name)``; None when no runtime is attached.

    The ``request_*`` methods only enqueue: they never block and never raise.
    """
    runtime = getattr(request.app.state, "runtime", None)
    call = getattr(runtime, method, None)
    return None if call is None else call(name)


def _outcome(future: Future[None] | None, exc: BaseException | None) -> _RuntimeOutcome:
    """Classify a wait on ``future`` that ended with ``exc`` (None: it succeeded)."""
    if future is None:
        return _RuntimeOutcome("absent")
    if exc is None:
        return _RuntimeOutcome("done")
    if not future.done():
        return _RuntimeOutcome("pending")
    if isinstance(exc, CameraNotFoundError):
        return _RuntimeOutcome("absent")
    return _RuntimeOutcome("failed", _error_text(exc))


def _wait_for_runtime(request: Request, method: str, name: str) -> _RuntimeOutcome:
    """Queue ``app.state.runtime.<method>(name)`` and wait (blocking) for the supervisor.

    For ``def`` routes, which run in the threadpool.
    """
    future = _request_runtime(request, method, name)
    if future is None:
        return _outcome(None, None)
    try:
        future.result(timeout=RUNTIME_REQUEST_TIMEOUT_S)
    except BaseException as exc:  # SystemExit too: an unknown proxy mode fails with it
        outcome = _outcome(future, exc)
    else:
        outcome = _outcome(future, None)
    _log_outcome(method, name, outcome)
    return outcome


async def _await_runtime(request: Request, method: str, name: str) -> _RuntimeOutcome:
    """Like ``_wait_for_runtime`` for ``async def`` routes, without blocking the event loop.

    ``asyncio.shield`` keeps a timeout from cancelling the wrapped future: an
    ``asyncio.wait_for`` timeout would otherwise cancel a request that is still queued,
    and the supervisor would then skip it.
    """
    future = _request_runtime(request, method, name)
    if future is None:
        return _outcome(None, None)
    wrapped = asyncio.wrap_future(future)
    # Fetch a late failure so asyncio does not log "exception was never retrieved".
    wrapped.add_done_callback(lambda f: f.cancelled() or f.exception())
    try:
        await asyncio.wait_for(asyncio.shield(wrapped), timeout=RUNTIME_REQUEST_TIMEOUT_S)
    except BaseException as exc:  # SystemExit too: an unknown proxy mode fails with it
        if isinstance(exc, asyncio.CancelledError) and not future.done():
            raise  # the request itself was cancelled (client gone, server stopping)
        outcome = _outcome(future, exc)
    else:
        outcome = _outcome(future, None)
    _log_outcome(method, name, outcome)
    return outcome


def _log_outcome(method: str, name: str, outcome: _RuntimeOutcome) -> None:
    if outcome.state == "pending":
        log.warning(
            "%s(%s) did not finish within %gs; it goes on in the background",
            method,
            name,
            RUNTIME_REQUEST_TIMEOUT_S,
        )
    elif outcome.state == "failed":
        log.warning("%s(%s) failed: %s", method, name, outcome.error)


@router.get("/{name}/edit", response_class=HTMLResponse)
def camera_edit_form(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    """Edit form for one camera, filled from its raw config.yaml entry (admin only)."""
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    config_path = get_config_path(request)
    if config_path is None:
        return _redirect_flash(f"/cameras/{name}", _NO_CONFIG_FILE, "error")
    try:
        raw = read_camera_entry(config_path, name)
    except (OSError, KeyError, yaml.YAMLError):
        return _redirect_flash(
            f"/cameras/{name}", f"Camera {name} was not found in {config_path}.", "error"
        )
    return _render_edit_form(request, cam, values=edit_form_values(raw, cam), errors={})


@router.post("/{name}/edit", response_class=HTMLResponse)
def camera_edit_submit(
    request: Request,
    name: str,
    main_url: str = Form(""),
    sub_url: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    onvif_port: str = Form(""),
    record_enabled: str = Form(""),
    user: CurrentUser = Depends(require_admin),
) -> Response:
    """Save the edit form: patch only the changed keys, then restart the camera if needed."""
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    config_path = get_config_path(request)
    if config_path is None:
        return _redirect_flash(f"/cameras/{name}", _NO_CONFIG_FILE, "error")
    inp = EditCameraInput(
        main_url=main_url,
        sub_url=sub_url,
        username=username,
        password=password,
        onvif_port=onvif_port,
        record_enabled=record_enabled == "on",
    )
    values: dict[str, object] = {
        "name": cam.name,
        "username": username,
        "main_url": split_userinfo(main_url.strip())[1],
        "sub_url": split_userinfo(sub_url.strip())[1],
        "onvif_port": onvif_port.strip(),
        "record_enabled": inp.record_enabled,
        "password_entered": bool(password),
        "proxy_port": cam.proxy.port,
    }

    with _config_edit_lock:
        try:
            raw = read_camera_entry(config_path, name)
        except (OSError, KeyError, yaml.YAMLError):
            return _redirect_flash(
                f"/cameras/{name}", f"Camera {name} was not found in {config_path}.", "error"
            )
        try:
            plan = plan_camera_edit(raw, cam, inp)
            updated = camera_config.validate_entry(
                apply_plan(raw, plan), {**os.environ, **plan.env_values}
            )
        except CameraEditError as exc:
            errors = {exc.field_name: str(exc)}
            return _render_edit_form(request, cam, values=values, errors=errors, status_code=422)
        except ValueError as exc:
            errors = {"form": redact_text(str(exc))}
            return _render_edit_form(request, cam, values=values, errors=errors, status_code=422)
        if plan.is_empty:
            return _redirect_flash(f"/cameras/{name}", "Nothing changed.", "info")

        try:
            # .env first: config.yaml must never reference a variable that is not there.
            if plan.env_values:
                upsert_env_vars(config_path.parent / ".env", plan.env_values)
            camera_config.patch_camera(config_path, name, plan.patch, remove_keys=plan.remove_keys)
        except KeyError:
            return _redirect_flash(
                f"/cameras/{name}", f"Camera {name} was not found in {config_path}.", "error"
            )
        except ValueError as exc:
            return _redirect_flash(
                f"/cameras/{name}", f"Could not save camera {name}: {exc}", "error"
            )
        except OSError as exc:
            log.warning("could not save camera %s: %s", name, exc)
            return _redirect_flash(
                f"/cameras/{name}",
                f"Could not save camera {name}: {exc.strerror or exc} "
                f"({exc.filename or config_path}). Make {config_path.parent} writable "
                "by rtsp-warden and try again.",
                "error",
            )
        if plan.env_values:
            os.environ.update(plan.env_values)
        apply_to_running_config(cam, updated)
        changed = [*plan.patch, *plan.remove_keys, *(["login"] if plan.env_values else [])]
        log.info("camera %s edited from the web UI: %s", name, ", ".join(changed))

    message = f"Camera {name} saved."
    level: Literal["info", "success", "error"] = "success"
    if plan.restart:
        outcome = _wait_for_runtime(request, "request_restart_camera", name)
        if outcome.state == "absent":
            message += " The change takes effect when rtsp-warden restarts."
        elif outcome.state == "pending":
            message = (
                f"Camera {name} saved; the camera is still restarting. "
                "Its status below updates on its own."
            )
            level = "info"
        elif outcome.state == "failed":
            message = (
                f"Camera {name} saved, but the running camera could not be restarted: "
                f"{outcome.error}"
            )
            level = "error"
        else:
            message += " Its stream was restarted."
    return _redirect_flash(f"/cameras/{name}", message, level)


async def _stop_onvif_subscription(name: str) -> None:
    """Stop and unregister the camera's ONVIF event subscription, if one is running.

    Subscribers are asyncio tasks on this event loop (``onvif/events.py`` registry),
    which is why ``camera_delete`` is an ``async def`` route.
    """
    subscriber = get_active_subscribers().get(name)
    if subscriber is None:
        return
    try:
        await asyncio.wait_for(subscriber.stop(), timeout=10.0)
    except Exception:
        log.warning("stopping the ONVIF event subscription of %s failed", name, exc_info=True)
    finally:
        unregister_subscriber(name)


def _remove_from_config(cfg: AppConfig, config_path: Path, name: str) -> str | None:
    """Drop camera ``name`` from config.yaml and ``cfg.cameras`` under the edit lock.

    Returns None on success (a camera already missing from the file counts as removed),
    otherwise an error text for the flash message; then nothing was changed.
    """
    with _config_edit_lock:
        try:
            camera_config.remove_camera(config_path, name)
        except KeyError:
            log.info("camera %s was already missing from %s", name, config_path)
        except ValueError as exc:
            return f"Could not delete camera {name}: {exc}"
        except OSError as exc:
            log.warning("could not delete camera %s: %s", name, exc)
            return (
                f"Could not delete camera {name}: {exc.strerror or exc} "
                f"({exc.filename or config_path}). Make {config_path.parent} writable "
                "by rtsp-warden and try again."
            )
        cfg.cameras = [c for c in cfg.cameras if c.name != name]
        _forget_credentials(config_path, name)
    log.info("camera %s deleted from the web UI", name)
    return None


def _forget_credentials(config_path: Path, name: str) -> None:
    """Drop a deleted camera's ``CAM_<SLUG>_*`` from .env and os.environ if unused.

    A hand-written camera may reference the same variables; then they stay. Without
    this, adding a camera of the same name with another login would be refused.
    """
    keys = camera_config.env_var_names(camera_config.env_slug(name))
    try:
        still_used = camera_config.referenced_env_vars(config_path)
        unused = [key for key in keys if key not in still_used]
        remove_env_vars(config_path.parent / ".env", unused)
    except (OSError, ValueError) as exc:
        log.warning("could not remove the login of camera %s from the .env file: %s", name, exc)
        return
    for key in unused:
        os.environ.pop(key, None)


@router.post("/{name}/delete")
async def camera_delete(
    request: Request, name: str, user: CurrentUser = Depends(require_admin)
) -> Response:
    """Remove a camera from config.yaml, cfg and the runtime; its files stay on disk."""
    cfg = get_cfg(request)
    cam = find_camera(cfg, name)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {name!r} not found")
    config_path = get_config_path(request)
    if config_path is None:
        return _redirect_flash(f"/cameras/{name}", _NO_CONFIG_FILE, "error")

    error = await run_in_threadpool(_remove_from_config, cfg, config_path, name)
    if error:
        return _redirect_flash(f"/cameras/{name}", error, "error")
    await _stop_onvif_subscription(name)
    outcome = await _await_runtime(request, "request_remove_camera", name)

    kept = cam.record.output_dir / name
    if outcome.state == "pending":
        return _redirect_flash(
            "/cameras",
            f"Camera {name} removed from {config_path.name}; the runtime is still stopping it. "
            f"Its files were kept in {kept}.",
            "info",
        )
    if outcome.state == "failed":
        return _redirect_flash(
            "/cameras",
            f"Camera {name} was removed from {config_path.name}, but the running camera could "
            f"not be stopped: {outcome.error}. Its files were kept in {kept}.",
            "error",
        )
    return _redirect_flash(
        "/cameras",
        f"Camera {name} removed. Its recordings, thumbnails and clips were kept in {kept}.",
        "success",
    )
