"""JSON API routes (auth-gated, used by the timeline JS).

Provides endpoints that return structured JSON data for the recording
timeline UI and other front-end components.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth_depends import require_user
from ..services.htl import get_recordings_dir
from ..services.recordings import get_recording_by_id
from ..services.timeline import build_timeline

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/recordings/{recording_id}/timeline")
def get_recording_timeline(
    request: Request,
    recording_id: int,
    user=Depends(require_user),
) -> dict:
    """Return timeline data (segments + events) for a recording.

    Returns a JSON object with recording metadata, an array of segment
    descriptors (start, end, path, size), and an array of event
    descriptors (id, type, severity, ts).
    """
    rec = get_recording_by_id(recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not available")

    recordings_dir = get_recordings_dir(cfg, rec["camera_name"], rec["stream"])
    if recordings_dir is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{rec['camera_name']}' not found in configuration",
        )

    timeline = build_timeline(
        recording_id=rec["id"],
        camera_name=rec["camera_name"],
        stream=rec["stream"],
        start_time=rec["start_time"],
        end_time=rec["end_time"],
        recordings_dir=recordings_dir,
    )

    return {
        "recording_id": timeline.recording_id,
        "camera_name": timeline.camera_name,
        "stream": timeline.stream,
        "start_ts": timeline.start_ts,
        "end_ts": timeline.end_ts,
        "segments": [
            {
                "start": s.start_ts,
                "end": s.end_ts,
                "path": s.path,
                "size": s.size_bytes,
            }
            for s in timeline.segments
        ],
        "events": [
            {
                "id": e.id,
                "type": e.event_type,
                "severity": e.severity,
                "ts": e.ts_unix,
                "object_type": e.object_type,
                "color": e.color,
            }
            for e in timeline.events
        ],
    }
