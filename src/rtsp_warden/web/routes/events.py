"""Event route handlers for the rtsp-warden web UI.

Provides paginated event list, event detail, event partial
for htmx auto-refresh, and clip generation trigger.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from ...clips import ClipError, ClipGenerator
from ...config import AppConfig
from ...db.schema import (
    create_clip,
    list_clips_for_event,
    update_clip_status,
)
from ..auth_depends import require_user
from ..paths import TEMPLATES_DIR
from ..services.events import get_event_by_id, list_events

log = logging.getLogger(__name__)

router = APIRouter(prefix="/events")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_clip_generator(request: Request) -> ClipGenerator | None:
    """Build a ClipGenerator from app config, or return None if clips disabled."""
    cfg: AppConfig | None = getattr(request.app.state, "cfg", None)
    if cfg is None or not cfg.clips.enabled:
        return None

    # Resolve recordings root from config
    recordings_dir = cfg.cameras[0].record.output_dir if cfg.cameras else Path("./recordings")
    output_dir_str = cfg.clips.output_dir.format(recordings_root=str(recordings_dir))

    # We need a fresh ClipsConfig with resolved output_dir
    resolved_cfg = cfg.clips.model_copy(update={"output_dir": output_dir_str})

    return ClipGenerator(
        cfg=resolved_cfg,
        recordings_dir=recordings_dir,
        ffmpeg_path=cfg.runtime.ffmpeg_path,
    )


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

    # Check for existing clips for this event
    existing_clips = list_clips_for_event(event_id)

    # Check if clips feature is enabled
    cfg: AppConfig | None = getattr(request.app.state, "cfg", None)
    clips_enabled = cfg is not None and cfg.clips.enabled

    return _templates.TemplateResponse(
        request,
        "events/detail.html",
        {
            "request": request,
            "event": evt,
            "clips": existing_clips,
            "clips_enabled": clips_enabled,
        },
    )


@router.post("/{event_id}/clip", status_code=303)
async def generate_clip(
    request: Request,
    event_id: int,
    user=Depends(require_user),
) -> RedirectResponse:
    """Trigger clip generation for an event. Returns redirect to clip detail."""
    evt = get_event_by_id(event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    # Idempotency: if a ready clip already exists, redirect to it
    existing = list_clips_for_event(event_id)
    ready_clips = [c for c in existing if c.status == "ready"]
    if ready_clips:
        return RedirectResponse(url=f"/clips/{ready_clips[0].id}", status_code=303)

    # Check if a pending clip already exists
    pending_clips = [c for c in existing if c.status == "pending"]
    if pending_clips:
        return RedirectResponse(url=f"/clips/{pending_clips[0].id}", status_code=303)

    # Get clip generator
    gen = _get_clip_generator(request)
    if gen is None:
        raise HTTPException(status_code=400, detail="Clip generation is not enabled")

    # Determine camera name and stream from the event
    camera_name = (
        evt.get("camera_name", "") if isinstance(evt, dict) else getattr(evt, "camera_name", "")
    )
    event_time = (
        evt.get("created_at", datetime.now(timezone.utc))
        if isinstance(evt, dict)
        else getattr(evt, "created_at", datetime.now(timezone.utc))
    )
    camera_id = evt.get("camera_id") if isinstance(evt, dict) else getattr(evt, "camera_id", None)

    # Use "sub" stream for clips (lower quality, smaller files)
    stream = "sub"

    # Create a pending clip record
    clip = create_clip(
        event_id=event_id,
        camera_id=camera_id,
        recording_id=f"{camera_name}_{stream}",
        path="",  # Will be updated after generation
        status="pending",
    )

    try:
        output_path = gen.generate(
            camera_name=camera_name,
            stream=stream,
            event_start=event_time,
            event_id=event_id,
        )

        # Update clip with final path and size
        file_size = output_path.stat().st_size if output_path.exists() else 0
        update_clip_status(
            clip_id=clip.id,
            status="ready",
            error_message=None,
        )
        # Also update path and size via direct DB update
        from ...db.engine import get_session
        from ...db.models import Clip as ClipModel

        with get_session() as session:
            db_clip = session.query(ClipModel).filter(ClipModel.id == clip.id).first()
            if db_clip is not None:
                db_clip.path = str(output_path)
                db_clip.size_bytes = file_size
                db_clip.duration_seconds = gen._cfg.pre_seconds + gen._cfg.post_seconds
                session.commit()

        return RedirectResponse(url=f"/clips/{clip.id}", status_code=303)

    except ClipError as exc:
        log.error("[clips] failed to generate clip for event %d: %s", event_id, exc)
        update_clip_status(
            clip_id=clip.id,
            status="failed",
            error_message=str(exc)[:1024],
        )
        raise HTTPException(status_code=500, detail=f"Clip generation failed: {exc}") from exc
