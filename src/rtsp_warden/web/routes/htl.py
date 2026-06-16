"""HLS Timeline (HTL) route handlers for the rtsp-warden web UI.

Provides endpoints for time-windowed HLS playlist generation and
segment file serving, enabling historical recording playback.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from ..auth_depends import require_user
from ..services.htl import build_htl_playlist, get_recordings_dir, scan_segments

router = APIRouter()


@router.get("/htl/{camera_name}/{stream_name}/{start_ts}/{end_ts}.m3u8")
async def htl_playlist(
    request: Request,
    camera_name: str,
    stream_name: str,
    start_ts: float,
    end_ts: float,
    user=Depends(require_user),
) -> PlainTextResponse:
    """Generate a dynamic m3u8 playlist for a time window.

    Scans the on-disk segments for the given camera/stream and builds
    a virtual HLS playlist that references segments overlapping the
    requested time range.
    """
    cfg = request.app.state.cfg
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not available")

    recordings_dir = get_recordings_dir(cfg, camera_name, stream_name)
    if recordings_dir is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")

    # Validate timestamps
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="start_ts must be less than end_ts")

    # Scan for segments in the recordings directory
    segments = scan_segments(recordings_dir)

    # Build the m3u8 playlist
    playlist = build_htl_playlist(
        camera_name=camera_name,
        stream_name=stream_name,
        start_ts=start_ts,
        end_ts=end_ts,
        segments_dir=recordings_dir,
        segments_index=segments,
    )

    return PlainTextResponse(
        content=playlist,
        media_type="application/vnd.apple.mpegurl",
    )


@router.get("/segments/{camera_name}/{stream_name}/{path:path}")
async def serve_segment(
    request: Request,
    camera_name: str,
    stream_name: str,
    path: str,
    user=Depends(require_user),
) -> FileResponse:
    """Serve a .ts segment file from the recordings directory.

    TODO: Future -- per-segment access checks (user has access to this camera).
    For Sprint 2: any authenticated user can access any segment.
    """
    cfg = request.app.state.cfg
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not available")

    recordings_dir = get_recordings_dir(cfg, camera_name, stream_name)
    if recordings_dir is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")

    # Path traversal protection: resolve and verify the path stays within
    # the recordings directory.
    if not path:
        raise HTTPException(status_code=400, detail="Empty segment path")

    # Reject paths with directory traversal components
    if ".." in path.split("/") or path.startswith("/"):
        raise HTTPException(status_code=403, detail="Invalid path")

    resolved = (recordings_dir / path).resolve()
    recordings_root = recordings_dir.resolve()

    if not str(resolved).startswith(str(recordings_root)):
        raise HTTPException(status_code=403, detail="Invalid path")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Segment '{path}' not found")

    # Determine content type based on extension
    content_type = "video/mp2t"
    if path.endswith(".m3u8"):
        content_type = "application/vnd.apple.mpegurl"

    return FileResponse(
        path=str(resolved),
        media_type=content_type,
    )
