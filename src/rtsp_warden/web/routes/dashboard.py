"""Dashboard route handlers for the rtsp-warden web UI.

Renders the main dashboard page with camera overview, recent
recordings, and recent events.
"""

from __future__ import annotations

from datetime import datetime, time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ..auth_depends import require_user
from ..paths import TEMPLATES_DIR
from ..services.cameras import list_cameras
from ..services.events import count_events_by_type, list_events
from ..services.recordings import list_recordings
from ..services.runtime import get_runtime_status

router = APIRouter()

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Render the main dashboard page."""
    cfg = getattr(request.app.state, "cfg", None)
    rt_provider = getattr(request.app.state, "runtime_provider", lambda: None)
    rt = rt_provider() if rt_provider else None

    # Camera data
    cameras = list_cameras(cfg) if cfg else []

    # Runtime status
    status = (
        get_runtime_status(rt, cfg)
        if cfg
        else {"ok": True, "cameras": [], "errors": [], "version": "unknown"}
    )

    # Recent recordings (last 5)
    try:
        recent_recordings, _ = list_recordings(limit=5)
    except Exception:
        recent_recordings = []

    # Recent events (last 10)
    try:
        recent_events, _ = list_events(limit=10)
    except Exception:
        recent_events = []

    # Detections today
    try:
        midnight = datetime.combine(datetime.now().date(), time.min)
        detections_by_type = count_events_by_type(since=midnight)
        detections_today = sum(detections_by_type.values())
    except Exception:
        detections_by_type = {}
        detections_today = 0

    return _templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "cameras": cameras,
            "status": status,
            "recent_recordings": recent_recordings,
            "recent_events": recent_events,
            "detections_today": detections_today,
            "detections_by_type": detections_by_type,
            "version": status.get("version", "unknown"),
        },
    )
