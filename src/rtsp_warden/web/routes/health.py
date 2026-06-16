"""Health route handlers for the rtsp-warden web UI.

Provides liveness probes, full status, and Prometheus-style metrics.
The /healthz, /status.json, and /metrics endpoints replace the
absorbed rtsp_warden.health_server module.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import is_dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.templating import Jinja2Templates

from ... import __version__
from ...status_model import make_empty_status, normalize_status
from ..paths import TEMPLATES_DIR
from ..services.runtime import get_runtime_status

router = APIRouter()

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for dataclasses and to_dict-able objects."""
    if is_dataclass(obj):
        try:
            from dataclasses import asdict

            return asdict(obj)
        except Exception:
            return str(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return str(obj)
    return str(obj)


def _render_metrics(status: Mapping[str, Any]) -> str:
    """Render a small Prometheus-style metrics body from status data."""
    import json as _json

    lines: list[str] = []
    now = float(status.get("now") or time.time())

    app_ok = 1 if bool(status.get("ok", True)) else 0
    lines.append("# HELP rtsp_warden_up 1 if the status provider reports ok")
    lines.append("# TYPE rtsp_warden_up gauge")
    lines.append(f"rtsp_warden_up {app_ok}")

    cameras = status.get("cameras") or []
    if not isinstance(cameras, list):
        cameras = []

    lines.append("# HELP rtsp_warden_cameras Number of configured cameras")
    lines.append("# TYPE rtsp_warden_cameras gauge")
    lines.append(f"rtsp_warden_cameras {len(cameras)}")

    lines.append("# HELP rtsp_warden_camera_up 1 if the camera is healthy")
    lines.append("# TYPE rtsp_warden_camera_up gauge")
    lines.append("# HELP rtsp_warden_stream_ingest_up 1 if the ingest process is running")
    lines.append("# TYPE rtsp_warden_stream_ingest_up gauge")
    lines.append("# HELP rtsp_warden_stream_last_frame_age_seconds Age of last seen frame")
    lines.append("# TYPE rtsp_warden_stream_last_frame_age_seconds gauge")
    lines.append("# HELP rtsp_warden_stream_last_segment_age_seconds Age of last written segment")
    lines.append("# TYPE rtsp_warden_stream_last_segment_age_seconds gauge")

    for cam in cameras:
        if not isinstance(cam, Mapping):
            continue
        cam_name = str(cam.get("name", ""))
        cam_ok = 1 if bool(cam.get("ok", True)) else 0
        lines.append(f"rtsp_warden_camera_up{{camera={_json.dumps(cam_name)}}} {cam_ok}")

        streams = cam.get("streams") or {}
        if not isinstance(streams, Mapping):
            continue

        for stream_name, s in streams.items():
            if not isinstance(s, Mapping):
                continue

            ingest = s.get("ingest") or {}
            ingest_running = (
                1 if bool(getattr(ingest, "get", lambda *_: None)("running", False)) else 0
            )
            lines.append(
                f"rtsp_warden_stream_ingest_up{{camera={_json.dumps(cam_name)},"
                f"stream={_json.dumps(str(stream_name))}}} {ingest_running}"
            )

            last_frame = _as_float(s.get("last_frame_at"))
            if last_frame is not None:
                age = max(0.0, now - last_frame)
                lines.append(
                    f"rtsp_warden_stream_last_frame_age_seconds{{camera={_json.dumps(cam_name)},"
                    f"stream={_json.dumps(str(stream_name))}}} {age:.3f}"
                )

            last_seg = _as_float(s.get("last_segment_at"))
            if last_seg is not None:
                age = max(0.0, now - last_seg)
                lines.append(
                    f"rtsp_warden_stream_last_segment_age_seconds{{camera={_json.dumps(cam_name)},"
                    f"stream={_json.dumps(str(stream_name))}}} {age:.3f}"
                )

    lines.append("")
    return "\n".join(lines)


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe -- always returns ok (no auth required)."""
    return {"status": "ok"}


@router.get("/status.json")
async def status_json(request: Request) -> JSONResponse:
    """Full status JSON (replaces health_server.py /status.json)."""
    cfg = getattr(request.app.state, "cfg", None)
    rt_provider = getattr(request.app.state, "runtime_provider", lambda: None)
    rt = rt_provider() if rt_provider else None
    try:
        status = get_runtime_status(rt, cfg) if cfg else make_empty_status()
        out = dict(normalize_status(status))
        out.setdefault("now", time.time())
        return JSONResponse(content=out)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "now": time.time(), "error": str(e)},
        )


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus-style metrics (replaces health_server.py /metrics)."""
    cfg = getattr(request.app.state, "cfg", None)
    rt_provider = getattr(request.app.state, "runtime_provider", lambda: None)
    rt = rt_provider() if rt_provider else None
    try:
        status = get_runtime_status(rt, cfg) if cfg else make_empty_status()
        body = _render_metrics(normalize_status(status))
        return Response(content=body, media_type="text/plain; version=0.0.4")
    except Exception as e:
        return Response(
            content=f"rtsp_warden_metrics_error 1\n# {e}\n",
            media_type="text/plain; charset=utf-8",
            status_code=500,
        )


@router.get("/health", response_model=None)
async def health(request: Request) -> Response:
    """Full health status endpoint.

    Returns JSON when the client requests application/json,
    otherwise returns an HTML page.
    """
    cfg = getattr(request.app.state, "cfg", None)
    rt_provider = getattr(request.app.state, "runtime_provider", lambda: None)
    rt = rt_provider() if rt_provider else None

    status = (
        get_runtime_status(rt, cfg)
        if cfg
        else {"ok": True, "version": __version__, "cameras": [], "errors": []}
    )

    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return JSONResponse(content=status)

    return _templates.TemplateResponse(
        request,
        "health.html",
        {
            "request": request,
            "status": status,
            "version": status.get("version", __version__),
        },
    )


@router.get("/health/partial", response_class=HTMLResponse)
async def health_partial(request: Request) -> HTMLResponse:
    """Return just the health status table for htmx auto-refresh."""
    cfg = getattr(request.app.state, "cfg", None)
    rt_provider = getattr(request.app.state, "runtime_provider", lambda: None)
    rt = rt_provider() if rt_provider else None

    status = (
        get_runtime_status(rt, cfg)
        if cfg
        else {"ok": True, "version": __version__, "cameras": [], "errors": []}
    )

    return _templates.TemplateResponse(
        request,
        "partials/health_status.html",
        {
            "request": request,
            "status": status,
        },
    )
