"""Clip route handlers for the rtsp-warden web UI.

Provides endpoints for clip detail viewing and file download.
Clip generation is triggered via the events route (POST /events/{id}/clip).
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.templating import Jinja2Templates

from ...db.schema import get_clip
from ..auth_depends import require_user
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/clips")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/{clip_id}", response_class=HTMLResponse)
async def clip_detail(
    request: Request,
    clip_id: int,
    user=Depends(require_user),
) -> HTMLResponse:
    """Render a single clip detail page."""
    clip = get_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")

    return _templates.TemplateResponse(
        request,
        "clips/detail.html",
        {
            "request": request,
            "clip": clip,
        },
    )


@router.get("/{clip_id}/download")
async def download_clip(
    clip_id: int,
    user=Depends(require_user),
) -> FileResponse:
    """Stream the MP4 clip file as a download."""
    clip = get_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")

    if clip.status != "ready":
        raise HTTPException(status_code=400, detail=f"Clip status is '{clip.status}', not 'ready'")

    if not os.path.exists(clip.path):
        raise HTTPException(status_code=404, detail="Clip file not found on disk")

    return FileResponse(
        clip.path,
        media_type="video/mp4",
        filename=os.path.basename(clip.path),
    )
