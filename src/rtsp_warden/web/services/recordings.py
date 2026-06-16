"""Recording listing service for the web UI.

Queries the recordings table with optional filters and pagination.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ...db.engine import get_session
from ...db.models import Camera, Recording


def list_recordings(
    camera_name: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return (recordings, total_count) filtered by camera/time, paginated.

    Each recording dict includes a camera_name field resolved from the
    cameras table. If the FK camera no longer exists, camera_name is
    set to "Unknown".
    """
    with get_session() as session:
        query = session.query(Recording)

        # Join to resolve camera name
        query = query.join(Camera, Recording.camera_id == Camera.id, isouter=True)

        if camera_name is not None:
            query = query.filter(Camera.name == camera_name)
        if start is not None:
            query = query.filter(Recording.start_time >= start)
        if end is not None:
            query = query.filter(Recording.start_time <= end)

        total = query.count()
        rows = query.order_by(Recording.start_time.desc()).offset(offset).limit(limit).all()

        results: list[dict[str, Any]] = []
        for row in rows:
            cam_name = row.camera.name if row.camera else "Unknown"
            results.append(
                {
                    "id": row.id,
                    "camera_name": cam_name,
                    "stream": row.stream,
                    "path": row.path,
                    "size_bytes": row.size_bytes,
                    "start_time": row.start_time,
                    "end_time": row.end_time,
                    "container": row.container,
                    "created_at": row.created_at,
                }
            )
        return results, total


def get_recording_by_id(recording_id: int) -> dict[str, Any] | None:
    """Return a single recording dict by ID, or None."""
    with get_session() as session:
        row = session.query(Recording).filter(Recording.id == recording_id).first()
        if row is None:
            return None
        cam_name = row.camera.name if row.camera else "Unknown"
        return {
            "id": row.id,
            "camera_name": cam_name,
            "stream": row.stream,
            "path": row.path,
            "size_bytes": row.size_bytes,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "container": row.container,
            "created_at": row.created_at,
        }
