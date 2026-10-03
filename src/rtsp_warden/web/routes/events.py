"""Event pages for the rtsp-warden web UI.

``/events`` is a card grid with camera / label / date filters and an htmx
refresh of page 1 (``/events/partial`` returns exactly the grid fragment);
``/events/{id}`` shows the full thumbnail, the clip player and the action runs.
Thumbnail and clip files are served from the paths stored on the event row,
resolved inside the configured ``record.output_dir`` (never from request input).
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from ... import __version__
from ...config import AppConfig, ClipsConfig
from ..auth_depends import require_user
from ..services.events import (
    PAGE_SIZE,
    clip_playlist,
    estimate_clip_seconds,
    event_filter_options,
    event_media_path,
    get_event_by_id,
    list_event_action_runs,
    list_events,
    parse_date_range,
)
from ._common import templates

router = APIRouter(prefix="/events")

_CLIP_MEDIA_TYPES = {".mp4": "video/mp4", ".ts": "video/mp2t"}


def _cfg(request: Request) -> AppConfig | None:
    return getattr(request.app.state, "cfg", None)


def _clean(value: str | None) -> str | None:
    """An empty or blank query value means "no filter" (the form always sends every field)."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _events_url(path: str, filters: dict[str, str | None], page: int = 1) -> str:
    """Build a URL with the active filters (urlencoded) and the page when it is not 1."""
    params = [(key, value) for key, value in filters.items() if value]
    if page > 1:
        params.append(("page", str(page)))
    return f"{path}?{urlencode(params)}" if params else path


def _grid_context(
    request: Request,
    camera: str | None,
    label: str | None,
    date_from: str | None,
    date_to: str | None,
    page: int,
) -> dict[str, Any]:
    filters = {
        "camera": _clean(camera),
        "label": _clean(label),
        "from": _clean(date_from),
        "to": _clean(date_to),
    }
    since, until = parse_date_range(filters["from"], filters["to"])
    events, total = list_events(
        camera_name=filters["camera"],
        label=filters["label"],
        since=since,
        until=until,
        limit=PAGE_SIZE,
        offset=(page - 1) * PAGE_SIZE,
        cfg=_cfg(request),
    )
    pages = max(1, math.ceil(total / PAGE_SIZE))
    return {
        "events": events,
        "total": total,
        "page": page,
        "pages": pages,
        "filters": filters,
        "filtered": any(filters.values()),
        "partial_url": _events_url("/events/partial", filters, page),
        "prev_url": _events_url("/events", filters, page - 1) if page > 1 else None,
        "next_url": _events_url("/events", filters, page + 1) if page < pages else None,
        "version": __version__,
    }


@router.get("", response_class=HTMLResponse)
def events_list(
    request: Request,
    camera: str | None = Query(default=None),
    label: str | None = Query(default=None),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    user=Depends(require_user),
) -> HTMLResponse:
    """Render the events page: filter form, card grid, pagination."""
    context = _grid_context(request, camera, label, date_from, date_to, page)
    options = event_filter_options(_cfg(request))
    context["camera_options"] = options["cameras"]
    context["label_options"] = options["labels"]
    return templates.TemplateResponse(request, "events/list.html", context)


@router.get("/partial", response_class=HTMLResponse)
def events_partial(
    request: Request,
    camera: str | None = Query(default=None),
    label: str | None = Query(default=None),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    user=Depends(require_user),
) -> HTMLResponse:
    """Return only the card grid (the inner HTML of ``#event-grid``) for the htmx refresh."""
    context = _grid_context(request, camera, label, date_from, date_to, page)
    return templates.TemplateResponse(request, "events/_grid.html", context)


@router.get("/{event_id}", response_class=HTMLResponse)
def event_detail(request: Request, event_id: int, user=Depends(require_user)) -> HTMLResponse:
    """Render one event: thumbnail, details, clip player, action runs."""
    event = get_event_by_id(event_id, cfg=_cfg(request))
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return templates.TemplateResponse(
        request,
        "events/detail.html",
        {
            "event": event,
            "action_runs": list_event_action_runs(event_id),
            "version": __version__,
        },
    )


@router.get("/{event_id}/thumbnail.jpg")
def event_thumbnail(request: Request, event_id: int, user=Depends(require_user)) -> FileResponse:
    """Serve the event's thumbnail; 404 when it never existed or retention removed it."""
    path = event_media_path(_cfg(request), event_id, "thumbnail")
    if path is None:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/{event_id}/clip")
def event_clip(request: Request, event_id: int, user=Depends(require_user)) -> FileResponse:
    """Serve the event's clip file (.mp4, or the .ts fallback the HLS playlist points at)."""
    path = event_media_path(_cfg(request), event_id, "clip")
    media_type = _CLIP_MEDIA_TYPES.get(path.suffix.lower()) if path is not None else None
    if path is None or media_type is None:
        raise HTTPException(status_code=404, detail="Clip not available")
    return FileResponse(path, media_type=media_type)


@router.get("/{event_id}/clip.m3u8")
def event_clip_playlist(request: Request, event_id: int, user=Depends(require_user)) -> Response:
    """One-entry HLS playlist so hls.js can play a .ts clip (browsers cannot play raw TS)."""
    cfg = _cfg(request)
    event = get_event_by_id(event_id, cfg=cfg)
    if event is None or event["clip_kind"] != "ts":
        raise HTTPException(status_code=404, detail="No .ts clip for this event")
    clips_cfg = cfg.clips if cfg is not None else ClipsConfig()
    seconds = estimate_clip_seconds(
        event["created_at"],
        event["ended_at"],
        pre_seconds=clips_cfg.pre_seconds,
        post_seconds=clips_cfg.post_seconds,
        max_duration=clips_cfg.max_duration,
    )
    return Response(
        clip_playlist(event_id, seconds),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache"},
    )
