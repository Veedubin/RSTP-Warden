"""Event route handlers for the rtsp-warden web UI.

Provides paginated event list, event detail, and event partial
for htmx auto-refresh.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ..auth_depends import require_user
from ..paths import TEMPLATES_DIR
from ..services.events import get_event_by_id, list_events

router = APIRouter(prefix="/events")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("", response_class=HTMLResponse)
async def events_list(
    request: Request,
    camera: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user=Depends(require_user),
) -> HTMLResponse:
    """Render the events list page with optional filters."""
    start_dt = None
    end_dt = None
    if start:
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            pass
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
        except ValueError:
            pass

    try:
        events, total = list_events(
            camera_name=camera,
            event_type=event_type,
            severity=severity,
            start=start_dt,
            end=end_dt,
            limit=limit,
            offset=offset,
        )
    except Exception:
        events, total = [], 0

    return _templates.TemplateResponse(
        request,
        "events/list.html",
        {
            "request": request,
            "events": events,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filter_camera": camera or "",
            "filter_type": event_type or "",
            "filter_severity": severity or "",
            "filter_start": start or "",
            "filter_end": end or "",
        },
    )


@router.get("/partial", response_class=HTMLResponse)
async def events_partial(request: Request, user=Depends(require_user)) -> HTMLResponse:
    """Return just the events table for htmx auto-refresh (no pagination)."""
    try:
        events, _ = list_events(limit=20)
    except Exception:
        events = []

    return _templates.TemplateResponse(
        request,
        "partials/event_row.html",
        {
            "request": request,
            "events": events,
        },
    )


@router.get("/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: int, user=Depends(require_user)) -> HTMLResponse:
    """Render a single event detail page."""
    evt = get_event_by_id(event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    return _templates.TemplateResponse(
        request,
        "events/detail.html",
        {
            "request": request,
            "event": evt,
        },
    )
