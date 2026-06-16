"""Event listing service for the web UI.

Queries the events table with optional filters and pagination.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ...db.engine import get_session
from ...db.models import Camera, Event


def list_events(
    camera_name: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return (events, total_count) filtered by camera/type/severity/time, paginated."""
    with get_session() as session:
        query = session.query(Event)

        # Join to resolve camera name
        query = query.join(Camera, Event.camera_id == Camera.id, isouter=True)

        if camera_name is not None:
            query = query.filter(Camera.name == camera_name)
        if event_type is not None:
            query = query.filter(Event.event_type == event_type)
        if severity is not None:
            query = query.filter(Event.severity == severity)
        if start is not None:
            query = query.filter(Event.created_at >= start)
        if end is not None:
            query = query.filter(Event.created_at <= end)

        total = query.count()
        rows = query.order_by(Event.created_at.desc()).offset(offset).limit(limit).all()

        results: list[dict[str, Any]] = []
        for row in rows:
            cam_name = row.camera.name if row.camera else "Unknown"
            results.append(
                {
                    "id": row.id,
                    "camera_name": cam_name,
                    "event_type": row.event_type,
                    "severity": row.severity,
                    "message": row.message,
                    "metadata_json": row.metadata_json,
                    "created_at": row.created_at,
                }
            )
        return results, total


def count_events_by_type(since: datetime | None = None) -> dict[str, int]:
    """Return {event_type: count} for the events table.

    Args:
        since: Only count events with created_at >= since. If None, all-time.
    """
    from sqlalchemy import func as sa_func

    with get_session() as session:
        query = session.query(Event.event_type, sa_func.count(Event.id)).group_by(Event.event_type)
        if since is not None:
            query = query.filter(Event.created_at >= since)
        rows = query.all()
        return {row[0]: row[1] for row in rows}


def get_recent_events(limit: int = 10, event_type: str | None = None) -> list[dict[str, Any]]:
    """Return the most recent events, optionally filtered by type.

    Same return shape as list_events (the dict, not the Event object).
    """
    with get_session() as session:
        query = session.query(Event).join(Camera, Event.camera_id == Camera.id, isouter=True)

        if event_type is not None:
            query = query.filter(Event.event_type == event_type)

        rows = query.order_by(Event.created_at.desc()).limit(limit).all()

        results: list[dict[str, Any]] = []
        for row in rows:
            cam_name = row.camera.name if row.camera else "Unknown"
            results.append(
                {
                    "id": row.id,
                    "camera_name": cam_name,
                    "event_type": row.event_type,
                    "severity": row.severity,
                    "message": row.message,
                    "metadata_json": row.metadata_json,
                    "created_at": row.created_at,
                }
            )
        return results


def get_event_by_id(event_id: int) -> dict[str, Any] | None:
    """Return a single event dict by ID, or None."""
    with get_session() as session:
        row = session.query(Event).filter(Event.id == event_id).first()
        if row is None:
            return None
        cam_name = row.camera.name if row.camera else "Unknown"
        return {
            "id": row.id,
            "camera_name": cam_name,
            "event_type": row.event_type,
            "severity": row.severity,
            "message": row.message,
            "metadata_json": row.metadata_json,
            "created_at": row.created_at,
        }
