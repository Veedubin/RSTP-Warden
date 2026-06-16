"""Recording route handlers for the rtsp-warden web UI.

Provides paginated recording list and recording detail pages with
HLS player integration for time-windowed playback.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ..auth_depends import require_user
from ..paths import TEMPLATES_DIR
from ..services.recordings import get_recording_by_id, list_recordings

router = APIRouter(prefix="/recordings")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("", response_class=HTMLResponse)
async def recordings_list(
    request: Request,
    camera: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user=Depends(require_user),
) -> HTMLResponse:
    """Render the recordings list page with optional filters."""
    # Parse date strings to datetime objects
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
        recordings, total = list_recordings(
            camera_name=camera,
            start=start_dt,
            end=end_dt,
            limit=limit,
            offset=offset,
        )
    except Exception:
        recordings, total = [], 0

    return _templates.TemplateResponse(
        request,
        "recordings/list.html",
        {
            "request": request,
            "recordings": recordings,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filter_camera": camera or "",
            "filter_start": start or "",
            "filter_end": end or "",
        },
    )


@router.get("/{recording_id}", response_class=HTMLResponse)
async def recording_detail(
    request: Request, recording_id: int, user=Depends(require_user)
) -> HTMLResponse:
    """Render a single recording detail page with HLS player."""
    rec = get_recording_by_id(recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

    # Build HLS timeline URL if the recording has timestamps and a path.
    htl_src = None
    if rec.get("start_time") and rec.get("path"):
        start_ts = rec["start_time"].timestamp()
        # Use end_time if available, otherwise add 3600s (1 hour) as a buffer
        if rec.get("end_time"):
            end_ts = rec["end_time"].timestamp()
        else:
            end_ts = start_ts + 3600

        camera_name = rec.get("camera_name", "")
        stream_name = rec.get("stream", "")
        if camera_name and stream_name:
            htl_src = f"/htl/{camera_name}/{stream_name}/{start_ts}/{end_ts}.m3u8"

    return _templates.TemplateResponse(
        request,
        "recordings/detail.html",
        {
            "request": request,
            "recording": rec,
            "htl_src": htl_src,
        },
    )
